"""
plots/aggregate_stats.py
-------------------------
Quantitative, aggregated counterpart to the per-image breaking-point plots
in `plotter.py`.

It produces two CSV tables:

1. ``severity_by_group.csv``
   One row per (perturbation_type, category, mask_area_bucket).
   Descriptive statistics on *when* the breaking point occurs:
   break rate, mean/std of the breaking-point severity (normalised level
   fraction), and a 95% confidence interval on that mean.

2. ``paired_significance.csv``
   One row per (perturbation_type, metric).
   A paired test (baseline vs. breaking-point / last-available level) on
   obj_iou, io_ratio and wdp, across every token that has both a baseline
   and a post-perturbation value. Reports both a paired t-test and a
   Wilcoxon signed-rank test (more robust with small / non-normal samples).

Design choices made explicit
-----------------------------
- Breaking-point detection reuses ``detect_breaking_point`` from
  ``plotter.py`` verbatim (same thresholds, same WDP-arbiter logic), so the
  numbers in this table are guaranteed consistent with the per-image plots.
  Any future change to the detection criterion only needs to happen in one
  place.
- Mask area is computed from the **baseline** masks only (``masks_dir``),
  never from the perturbed masks. For perturbations that resize or move the
  object (Scale, Jigsaw, Rotate, Shear) the area at each level is the
  manipulated variable itself — using it to bucket "intrinsic" object size
  would be circular. The baseline mask is the only area measurement that is
  independent of the perturbation being studied.
- For tokens where no breaking point is detected, the "value at break" used
  in the paired test falls back to the value at the **last available
  perturbation level** (i.e. the most severe one tested). This keeps every
  token in the paired sample instead of silently dropping the ones that
  never break, which would bias the test toward only the fragile cases.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

_PLOTS_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(_PLOTS_DIR))

from data_loader import METRICS, load_results
from plotter import (
    WDP_PENALTY_NO_HEATMAP,
    WDP_PENALTY_NO_TOKEN,
    _BASELINE_X,
    _METRIC_DIRECTION,
    _relative_degradation,
    _with_baseline,
    detect_breaking_point,
)
from stats_plotter import (
    plot_degradation_curves,
    plot_paired_significance,
    plot_severity_per_perturbation,
)

logger = logging.getLogger(__name__)

_DEFAULT_AREA_EDGES = (0.02, 0.10)  # ratio thresholds: small < 0.02 <= medium < 0.10 <= large

# Perturbations whose levels must be analysed in DESCENDING numeric order,
# because for them a HIGHER numeric parameter means a WEAKER perturbation
# (and a lower one a stronger perturbation), the opposite of the usual
# convention. Examples: CannyEdge (higher threshold = fewer edges),
# JPEGCompression (higher quality = less compression), Saturation (higher
# factor = closer to the original), Poisson/Shot noise (higher scale = less
# noise). Listing them here makes the levels run from the weakest to the
# strongest perturbation, with the baseline (0.0) still first.
_REVERSED_LEVEL_TYPES = {
    "CannyEdge",
    "JPEGCompression",
    "Saturation",
    "PoissonNoise",
    "ShotNoise",
}


def _ordered_levels(values, ptype: str) -> List[float]:
    """
    Order the perturbation levels for analysis. The baseline (0.0) always
    comes first; the remaining levels are sorted ascending, except for the
    perturbations in ``_REVERSED_LEVEL_TYPES`` (e.g. CannyEdge), whose levels
    are sorted descending so a higher threshold (less edge) precedes a lower
    one (more edge).
    """
    vals = sorted(set(float(v) for v in values))
    baseline = [v for v in vals if v == _BASELINE_X]
    rest = [v for v in vals if v != _BASELINE_X]
    if ptype in _REVERSED_LEVEL_TYPES:
        rest.sort(reverse=True)
    return baseline + rest


# ---------------------------------------------------------------------------
# Mask area
# ---------------------------------------------------------------------------

def compute_mask_areas(masks_dir: str) -> Dict[Tuple[str, str], float]:
    """
    Compute the area ratio (mask pixels / total pixels) for every
    (img_id, category) pair found in *masks_dir*, using the **baseline**
    masks only.

    Implemented standalone (PIL only) rather than reusing
    ``experiment/mask_loader.py``: that module imports ``new_eval.py``,
    which pulls in ``torch``/``transformers`` purely to read a binary PNG.
    A statistics/aggregation script has no reason to depend on the model
    stack, so mask loading is duplicated here in a minimal form instead.

    Expected filename convention (same as the rest of the pipeline):
    ``<img_id>_<category>.png``.

    Returns
    -------
    dict
        ``{(img_id, category): area_ratio}``
    """
    masks_path = Path(masks_dir)
    areas: Dict[Tuple[str, str], float] = {}

    if not masks_path.exists():
        logger.warning("Masks directory not found: %s — areas will be unknown.", masks_path)
        return areas

    for png in sorted(masks_path.glob("*.png")):
        name_no_ext = png.stem
        parts = name_no_ext.rsplit("_", 1)
        if len(parts) != 2:
            logger.debug("Filename does not match <img_id>_<category>.png, skipping: %s", png.name)
            continue
        img_id, category = parts

        arr = np.array(Image.open(png).convert("L"))
        total = arr.size
        if total == 0:
            continue
        ratio = float((arr > 0).sum()) / total
        areas[(img_id, category)] = ratio

    logger.info("Computed mask area for %d (img_id, category) pairs from '%s'.",
                len(areas), masks_dir)
    return areas


def bucket_area(ratio: Optional[float], edges: Tuple[float, float] = _DEFAULT_AREA_EDGES) -> str:
    """Bucket a mask area ratio into 'small' / 'medium' / 'large' / 'unknown'."""
    if ratio is None or np.isnan(ratio):
        return "unknown"
    lo, hi = edges
    if ratio < lo:
        return "small"
    if ratio < hi:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Per-token series construction (mirrors the logic in plotter.py exactly,
# so severity numbers match what the per-image figures show)
# ---------------------------------------------------------------------------

def _series_with_sentinels(cat_df: pd.DataFrame, x_values: List[float], metric: str) -> Tuple[np.ndarray, List[bool]]:
    """
    Build the metric series for one (image, category) token, applying the
    same sentinel-value treatment for missing data used in the visual
    breaking-point plots (WDP_PENALTY_* instead of silent NaN).
    """
    y, missing = [], []
    for x in x_values:
        vals = cat_df.loc[cat_df["param_value"] == x, metric].dropna()
        if len(vals) > 0:
            val = vals.mean()
            missing.append(False)
            if metric == "wdp" and val == 0.0:
                y.append(WDP_PENALTY_NO_HEATMAP)
            else:
                y.append(val)
        else:
            missing.append(True)
            y.append(0.0 if metric in ("obj_iou", "io_ratio") else WDP_PENALTY_NO_TOKEN)
    return np.array(y, dtype=float), missing


def _build_token_record(
    ptype: str,
    img_id: str,
    category: str,
    cat_df: pd.DataFrame,
    x_values: List[float],
    x_labels: List[str],
    degradation_threshold: float,
    wdp_arbiter_min: float,
    min_consecutive: int,
) -> dict:
    """Build one row of the long-format per-token table."""
    series: Dict[str, np.ndarray] = {}
    baseline_vals: Dict[str, float] = {}

    for metric in METRICS:
        y, _ = _series_with_sentinels(cat_df, x_values, metric)
        series[metric] = y
        baseline_vals[metric] = y[0]

    bp_idx = detect_breaking_point(
        obj_iou_means=series["obj_iou"],
        io_ratio_means=series["io_ratio"],
        wdp_means=series["wdp"],
        baseline_vals=baseline_vals,
        threshold=degradation_threshold,
        wdp_arbiter_min=wdp_arbiter_min,
        min_consecutive=min_consecutive,
    )

    n_levels = len(x_values)
    broke = bp_idx is not None
    # Fallback comparison point for tokens that never break: the most
    # severe level actually tested, so every token contributes to the
    # paired test instead of only the fragile ones.
    compare_idx = bp_idx if broke else (n_levels - 1)

    # Normalised severity fraction: 0 = broke at the earliest possible
    # opportunity, 1 = broke only at the most extreme level tested.
    #
    # bp_idx == 0 is a distinct sentinel value returned by
    # detect_breaking_point() meaning "already at the worst possible
    # metric value before any real margin could be observed" (e.g. the
    # baseline obj_iou/io_ratio is already 0). That is the most extreme
    # case of fragility, so it must map to fraction 0.0 directly — running
    # it through the general formula below would produce a negative value,
    # since bp_idx - 1 < 0 in that case.
    if not broke:
        severity_fraction = np.nan
    elif bp_idx == 0:
        severity_fraction = 0.0
    else:
        severity_fraction = (bp_idx - 1) / max(n_levels - 2, 1)

    # A token "broken at baseline" (bp_idx == 0) already failed without any
    # perturbation: it is not a perturbation effect, so it is flagged and
    # excluded from the severity / break-rate analysis downstream. Only
    # ``broke_by_perturbation`` (bp_idx > 0) counts as a genuine break.
    baseline_broken = broke and bp_idx == 0
    broke_by_perturbation = broke and bp_idx is not None and bp_idx > 0

    record = {
        "perturbation_type": ptype,
        "img_id": img_id,
        "category": category,
        "n_levels": n_levels,
        "broke": broke,
        "baseline_broken": baseline_broken,
        "broke_by_perturbation": broke_by_perturbation,
        "breaking_point_level_index": bp_idx if broke else np.nan,
        "breaking_point_param_value": x_values[bp_idx] if broke else np.nan,
        "breaking_point_param_label": x_labels[bp_idx] if broke else None,
        "breaking_point_level_fraction": severity_fraction,
    }

    for metric in METRICS:
        record[f"baseline_{metric}"] = baseline_vals[metric]
        record[f"compare_{metric}"] = series[metric][compare_idx]

    return record


# ---------------------------------------------------------------------------
# Long-format per-token table
# ---------------------------------------------------------------------------

def build_token_table(
    df: pd.DataFrame,
    degradation_threshold: float = 0.40,
    wdp_arbiter_min: float = 0.0,
    min_consecutive: int = 2,
    perturbation_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build the long-format table with one row per (perturbation_type,
    img_id, category) token, including breaking-point severity and the
    baseline/compare metric values needed for the paired tests.
    """
    types = perturbation_types or sorted(
        t for t in df["perturbation_type"].unique() if t != "Normale"
    )
    records = []

    for ptype in types:
        type_df = _with_baseline(df, ptype)
        if type_df.empty:
            continue

        x_values = _ordered_levels(type_df["param_value"].unique(), ptype)
        # Re-derive labels locally (avoids importing the plotting-only helper)
        label_map = (
            type_df.drop_duplicates("param_value")
            .set_index("param_value")["perturbation_param"]
            .to_dict()
        )
        x_labels = ["Normale" if x == _BASELINE_X else label_map.get(x, str(x)) for x in x_values]

        pairs = type_df[["img_id", "category"]].drop_duplicates().itertuples(index=False)
        for img_id, category in pairs:
            cat_df = type_df[(type_df["img_id"] == img_id) & (type_df["category"] == category)]
            records.append(
                _build_token_record(
                    ptype, img_id, category, cat_df, x_values, x_labels,
                    degradation_threshold, wdp_arbiter_min, min_consecutive,
                )
            )

    table = pd.DataFrame.from_records(records)
    logger.info("Built per-token table: %d rows across %d perturbation type(s).",
                len(table), len(types))
    return table


# ---------------------------------------------------------------------------
# Table C — per-level stats (all tokens pooled, by perturbation × level)
# ---------------------------------------------------------------------------

def build_level_stats_table(
    df: pd.DataFrame,
    perturbation_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate each metric at every perturbation LEVEL, pooling all tokens
    (every image/category) of a perturbation type together.

    Rationale
    ---------
    The per-(category, area) groups in ``severity_by_group.csv`` are tiny,
    so a confidence interval there is meaningless. Pooling all tokens at a
    given level instead yields a large sample (tens of tokens), making the
    mean / std / 95% CI statistically informative. The resulting table is a
    degradation profile: how each metric behaves as the perturbation grows,
    i.e. *how far* the model holds before it breaks.

    The same sentinel treatment as the breaking-point plots is applied
    (``_series_with_sentinels``), so a lost token contributes 0 for
    obj_iou/io_ratio and the WDP penalty for wdp — consistent with the
    breaking-point definition. Because every token gets a value at every
    level, ``n_tokens`` is constant across levels within a perturbation.

    Returns one row per (perturbation_type, level, metric).
    """
    types = perturbation_types or sorted(
        t for t in df["perturbation_type"].unique() if t != "Normale"
    )
    rows = []

    for ptype in types:
        type_df = _with_baseline(df, ptype)
        if type_df.empty:
            continue

        x_values = _ordered_levels(type_df["param_value"].unique(), ptype)
        label_map = (
            type_df.drop_duplicates("param_value")
            .set_index("param_value")["perturbation_param"]
            .to_dict()
        )
        x_labels = ["Normale" if x == _BASELINE_X else label_map.get(x, str(x)) for x in x_values]

        # Stack the sentinel series of every token into [n_tokens × n_levels].
        series_by_metric: Dict[str, List[np.ndarray]] = {m: [] for m in METRICS}
        pairs = type_df[["img_id", "category"]].drop_duplicates().itertuples(index=False)
        for img_id, category in pairs:
            cat_df = type_df[(type_df["img_id"] == img_id) & (type_df["category"] == category)]
            for metric in METRICS:
                y, _ = _series_with_sentinels(cat_df, x_values, metric)
                series_by_metric[metric].append(y)

        for metric in METRICS:
            stack = series_by_metric[metric]
            mat = np.vstack(stack) if stack else np.empty((0, len(x_values)))
            for j, x in enumerate(x_values):
                col = mat[:, j]
                col = col[~np.isnan(col)]
                n = len(col)
                mean = float(col.mean()) if n else np.nan
                std = float(col.std(ddof=1)) if n > 1 else np.nan
                if n >= 2 and not np.isnan(std):
                    sem = std / np.sqrt(n)
                    t_crit = stats.t.ppf(0.975, df=n - 1)
                    ci_lo = mean - t_crit * sem
                    ci_hi = mean + t_crit * sem
                else:
                    ci_lo = ci_hi = np.nan
                rows.append({
                    "perturbation_type": ptype,
                    "level_index": j,
                    "param_value": x,
                    "param_label": x_labels[j],
                    "metric": metric,
                    "n_tokens": n,
                    "mean": mean,
                    "std": std,
                    "ci95_low": ci_lo,
                    "ci95_high": ci_hi,
                })

    out = pd.DataFrame.from_records(rows)
    logger.info("Built per-level table: %d rows across %d perturbation type(s).",
                len(out), len(types))
    return out


# ---------------------------------------------------------------------------
# Table A — severity by group
# ---------------------------------------------------------------------------

def _mean_ci95(values: pd.Series, min_n_for_ci: int = 5) -> Tuple[float, float, float, float]:
    """
    Return (mean, std, ci_low, ci_high) using a t-distribution CI.

    Below *min_n_for_ci* samples the CI is suppressed (returned as NaN)
    rather than reported: with n=2-4 the t-distribution has very fat tails
    (e.g. t_crit(df=1) ≈ 12.7), producing intervals that are technically
    correct but practically meaningless — often extending far outside the
    [0, 1] range the severity fraction is defined on. Mean and std are
    still reported regardless of n, since they remain informative even
    for small groups; only the CI is gated.

    The returned bounds are clipped to [0, 1] since severity_fraction is
    bounded by construction. This is a standard (if approximate) practice
    for a bounded statistic — a more rigorous alternative would be a
    bootstrap or beta-distribution CI, which is heavier to justify here
    given the typically small per-group sample sizes.
    """
    values = values.dropna()
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(values.mean())
    if n == 1:
        return mean, np.nan, np.nan, np.nan
    std = float(values.std(ddof=1))
    if n < min_n_for_ci:
        return mean, std, np.nan, np.nan
    sem = std / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_lo = max(0.0, mean - t_crit * sem)
    ci_hi = min(1.0, mean + t_crit * sem)
    return mean, std, ci_lo, ci_hi


def build_severity_table(
    token_table: pd.DataFrame,
    mask_areas: Dict[Tuple[str, str], float],
    area_edges: Tuple[float, float] = _DEFAULT_AREA_EDGES,
) -> pd.DataFrame:
    """
    Aggregate the per-token table into descriptive statistics grouped by
    (perturbation_type, category, mask_area_bucket).
    """
    t = token_table.copy()
    t["mask_area_ratio"]  = t.apply(lambda r: mask_areas.get((r["img_id"], r["category"]), np.nan), axis=1)
    t["mask_area_bucket"] = t["mask_area_ratio"].apply(lambda r: bucket_area(r, area_edges))

    rows = []
    group_cols = ["perturbation_type", "category", "mask_area_bucket"]
    for keys, g in t.groupby(group_cols, dropna=False):
        ptype, category, bucket = keys
        n_baseline_broken = int(g["baseline_broken"].sum())
        # Baseline-valid tokens only: a token already failed at baseline is
        # not a perturbation effect, so it is excluded from severity and from
        # the break-rate denominator.
        gv = g[~g["baseline_broken"]]
        n_valid = len(gv)
        broke = gv.loc[gv["broke_by_perturbation"], "breaking_point_level_fraction"]
        n_broken = int(gv["broke_by_perturbation"].sum())

        mean_frac, std_frac, ci_lo, ci_hi = _mean_ci95(broke)
        mean_area = g["mask_area_ratio"].mean()

        rows.append({
            "perturbation_type": ptype,
            "category": category,
            "mask_area_bucket": bucket,
            "n_tokens": n_valid,
            "n_baseline_broken": n_baseline_broken,
            "n_broken": n_broken,
            "break_rate": n_broken / n_valid if n_valid else np.nan,
            "mean_severity_fraction": mean_frac,
            "std_severity_fraction": std_frac,
            "ci95_low": ci_lo,
            "ci95_high": ci_hi,
            "mean_mask_area_ratio": mean_area,
        })

    out = pd.DataFrame(rows).sort_values(
        ["perturbation_type", "category", "mask_area_bucket"]
    ).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Table B — paired significance (baseline vs. break / last level)
# ---------------------------------------------------------------------------

def _paired_stats(baseline: np.ndarray, compare: np.ndarray, direction: str) -> dict:
    """
    Run the paired baseline-vs-compare analysis for one metric on one group
    of tokens. Returns the effect-size and test fields shared by the paired
    table and the master table.

    The relative degradation is signed so that a POSITIVE value always means
    "worse", regardless of whether the metric is higher- or lower-better.
    Both a paired t-test and a Wilcoxon signed-rank test are reported; with
    fewer than 3 valid pairs the tests are skipped (NaN p-values + a note),
    since they are not meaningful below that sample size.
    """
    valid = ~(np.isnan(baseline) | np.isnan(compare))
    b, c = baseline[valid], compare[valid]
    n = len(b)

    diff = np.array([_relative_degradation(cv, bv, direction) if bv != 0 else np.nan
                     for bv, cv in zip(b, c)])
    diff = diff[~np.isnan(diff)]

    out = {
        "n_pairs": n,
        "mean_baseline": float(np.mean(b)) if n else np.nan,
        "mean_compare": float(np.mean(c)) if n else np.nan,
        "mean_relative_degradation": float(np.mean(diff)) if len(diff) else np.nan,
        "std_relative_degradation": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
    }

    if n < 3:
        out.update({
            "ttest_stat": np.nan, "ttest_pvalue": np.nan,
            "wilcoxon_stat": np.nan, "wilcoxon_pvalue": np.nan,
            "note": f"n={n} < 3: test not meaningful",
        })
    else:
        t_stat, t_p = stats.ttest_rel(b, c)
        try:
            w_stat, w_p = stats.wilcoxon(b, c)
        except ValueError:
            # All differences are zero, or another degenerate case
            w_stat, w_p = np.nan, np.nan
        out.update({
            "ttest_stat": float(t_stat), "ttest_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat), "wilcoxon_pvalue": float(w_p),
            "note": "",
        })
    return out


def build_paired_significance_table(token_table: pd.DataFrame) -> pd.DataFrame:
    """
    For each (perturbation_type, metric), run a paired test comparing the
    baseline value against the value at the breaking point (or, for tokens
    that never break, at the most severe level tested).
    """
    rows = []
    for ptype, g in token_table.groupby("perturbation_type"):
        for metric in METRICS:
            st = _paired_stats(
                g[f"baseline_{metric}"].to_numpy(),
                g[f"compare_{metric}"].to_numpy(),
                _METRIC_DIRECTION[metric],
            )
            rows.append({"perturbation_type": ptype, "metric": metric, **st})

    out = pd.DataFrame(rows).sort_values(["perturbation_type", "metric"]).reset_index(drop=True)
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build aggregated severity and paired-significance tables "
                    "from the pipeline's result CSVs.",
    )
    parser.add_argument("--output-dir", default="output",
                        help="Root directory containing the pipeline's results.csv files.")
    parser.add_argument("--masks-dir", default="masks",
                        help="Directory with baseline ground-truth masks, used for area buckets.")
    parser.add_argument("--out-dir", default="plots/output/stats",
                        help="Directory where the two output CSVs are written.")
    parser.add_argument("--perturbations", nargs="+", default=None, metavar="NAME",
                        help="Limit to these perturbation type names (default: all).")
    parser.add_argument("--degradation-threshold", type=float, default=0.40)
    parser.add_argument("--wdp-arbiter-min", type=float, default=0.0)
    parser.add_argument("--min-consecutive", type=int, default=2)
    parser.add_argument("--area-small-max", type=float, default=_DEFAULT_AREA_EDGES[0],
                        help="Upper bound (exclusive) of the 'small' mask-area bucket.")
    parser.add_argument("--area-medium-max", type=float, default=_DEFAULT_AREA_EDGES[1],
                        help="Upper bound (exclusive) of the 'medium' mask-area bucket.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating PNG figures; write the CSV tables only.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    df = load_results(args.output_dir)
    if df.empty:
        logger.warning("No data found in '%s' — nothing to aggregate.", args.output_dir)
        sys.exit(0)

    mask_areas = compute_mask_areas(args.masks_dir)

    token_table = build_token_table(
        df,
        degradation_threshold=args.degradation_threshold,
        wdp_arbiter_min=args.wdp_arbiter_min,
        min_consecutive=args.min_consecutive,
        perturbation_types=args.perturbations,
    )

    severity_table = build_severity_table(
        token_table, mask_areas,
        area_edges=(args.area_small_max, args.area_medium_max),
    )
    paired_table = build_paired_significance_table(token_table)
    level_table  = build_level_stats_table(df, perturbation_types=args.perturbations)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    severity_path = out_dir / "severity_by_group.csv"
    paired_path   = out_dir / "paired_significance.csv"
    token_path    = out_dir / "per_token_raw.csv"
    level_path    = out_dir / "level_stats.csv"

    severity_table.to_csv(severity_path, index=False)
    paired_table.to_csv(paired_path, index=False)
    token_table.to_csv(token_path, index=False)
    level_table.to_csv(level_path, index=False)

    logger.info("Severity table  -> %s  (%d rows)", severity_path, len(severity_table))
    logger.info("Paired tests     -> %s  (%d rows)", paired_path, len(paired_table))
    logger.info("Per-token raw    -> %s  (%d rows)", token_path, len(token_table))
    logger.info("Per-level stats  -> %s  (%d rows)", level_path, len(level_table))

    if not args.no_plots:
        fig_dir = out_dir / "figures"
        plot_degradation_curves(level_table, fig_dir)
        plot_severity_per_perturbation(token_table, fig_dir)
        plot_paired_significance(paired_table, fig_dir)
        logger.info("Figures          -> %s", fig_dir)


if __name__ == "__main__":
    main()