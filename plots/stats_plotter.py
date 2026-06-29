"""
plots/stats_plotter.py
----------------------
Visual counterpart to ``aggregate_stats.py``. Turns the aggregated tables
into figures so the numbers in the CSVs get an intuitive visual reading.

Figures produced:

1. ``degradation_curve_<ptype>.png`` (one per perturbation type)
   Mean metric value across all tokens at each perturbation level, wrapped
   in nested bands: a light +/- std band (data spread) and a darker 95% CI
   band (uncertainty of the mean). The breaking threshold is drawn so it is
   visible *how far* the perturbation can go before the model breaks.

2. ``severity_<ptype>.png`` (one per perturbation type)
   Strip of every broken token on a 0->1 severity axis (0 = breaks early /
   fragile, 1 = breaks late / robust), with a diamond at the mean and its
   95% CI. All tokens of the perturbation are pooled (n = 20-65), so the
   mean and CI are statistically meaningful.

3. ``paired_significance_<ptype>.png`` (one per perturbation type)
   Compact paired comparison per perturbation: a baseline bar vs an at-break
   bar for each metric, with a significance bracket (stars + p-value from
   Wilcoxon, falling back to the t-test) and the number of pairs.

Design note
-----------
Reuses the visual constants from ``plotter.py`` (DPI, palette, metric
labels, filename sanitiser) so these figures match the style of the
per-image breaking-point plots.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from data_loader import METRICS
from plotter import (
    _FIGURE_DPI,
    _METRIC_LABELS,
    _METRIC_YLIM,
    _PALETTE,
    _safe_filename,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _p_to_stars(p: float) -> str:
    """Map a p-value to the conventional significance annotation."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1 — Severity per perturbation type (all tokens pooled)
# ---------------------------------------------------------------------------

def plot_severity_per_perturbation(
    token_table: pd.DataFrame, out_dir, min_n_for_ci: int = 5,
) -> None:
    """
    One compact figure per perturbation type showing *how early* its tokens
    break. Every broken token is a dot on a 0->1 severity axis (0 = breaks at
    the first level / fragile, 1 = breaks only at the most extreme level /
    robust); a diamond marks the mean with its 95% CI. Sample size, broken
    count and break rate are reported in the title.

    Pools all tokens of the perturbation together (n = 20-65), so the mean
    and CI are statistically meaningful — unlike the old per-(category, area)
    breakdown where almost every group had n<5.

    Tokens already failed at baseline (``baseline_broken``) are excluded: they
    reflect the model's intrinsic inability on that object, not the effect of
    the perturbation. Their count is reported in the title as "excluded".
    """
    save_dir = Path(out_dir)
    _ensure_dir(save_dir)
    if token_table.empty:
        logger.warning("Token table is empty — skipping severity plots.")
        return

    rng = np.random.default_rng(0)

    for ptype, g_all in token_table.groupby("perturbation_type"):
        n_baseline_broken = int(g_all["baseline_broken"].sum())
        n_no_baseline = int(g_all["no_baseline_data"].sum())
        # Baseline-valid tokens only: tokens already failed at baseline, or
        # never tested at baseline, are excluded from severity and break rate.
        g = g_all[~g_all["baseline_broken"] & ~g_all["no_baseline_data"]]
        n_total = len(g)
        sev = g.loc[g["broke_by_perturbation"], "breaking_point_level_fraction"].dropna().to_numpy()
        n_broken = len(sev)
        break_rate = n_broken / n_total if n_total else np.nan

        fig, ax = plt.subplots(figsize=(7.5, 3.0), constrained_layout=True)
        color = _PALETTE[0]

        if n_broken > 0:
            mean = float(np.mean(sev))
            std = float(np.std(sev, ddof=1)) if n_broken > 1 else np.nan

            # raw broken tokens as a jittered strip
            yj = rng.uniform(-0.16, 0.16, size=n_broken)
            ax.scatter(sev, yj, s=32, alpha=0.5, color=color,
                       edgecolors="none", zorder=2, label="single broken token")

            # mean marker + 95% CI (only when the sample supports it)
            if n_broken >= min_n_for_ci and not np.isnan(std):
                sem = std / np.sqrt(n_broken)
                half = float(stats.t.ppf(0.975, df=n_broken - 1) * sem)
                ci_lo, ci_hi = max(0.0, mean - half), min(1.0, mean + half)
                ax.errorbar(
                    [mean], [0], xerr=[[mean - ci_lo], [ci_hi - mean]],
                    fmt="D", color="black", ecolor="black", capsize=6,
                    markersize=10, linewidth=1.8, zorder=3,
                    label=f"mean = {mean:.2f}   95% CI [{ci_lo:.2f}, {ci_hi:.2f}]",
                )
            else:
                ax.scatter([mean], [0], marker="D", color="black", s=90, zorder=3,
                           label=f"mean = {mean:.2f}   (CI n/d, n<{min_n_for_ci})")

            ax.legend(loc="upper center", fontsize=8, ncol=2,
                      frameon=True, framealpha=0.9)
        else:
            ax.text(0.5, 0, "no broken tokens", ha="center", va="center",
                    color="grey", fontsize=11, transform=ax.transData)

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Severity fraction")
        ax.text(0.0, -0.44, "← fragile (breaks early)", fontsize=8, color="dimgrey", ha="left")
        ax.text(1.0, -0.44, "robust (breaks late) →", fontsize=8, color="dimgrey", ha="right")
        ax.grid(axis="x", alpha=0.35, linewidth=0.4)

        ax.set_title(
            f"Breaking-point severity — {ptype}\n"
            f"baseline-valid tokens = {n_total}   broken = {n_broken}   "
            f"break rate = {break_rate*100:.0f}%\n"
            f"(excluded: failed at baseline = {n_baseline_broken}, "
            f"no baseline data = {n_no_baseline})",
            fontsize=10, fontweight="bold",
        )
        path = save_dir / f"severity_{_safe_filename(str(ptype))}.png"
        fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved severity plot: %s", path)


# ---------------------------------------------------------------------------
# 1b — Degradation curves: metric vs perturbation level (all images pooled)
# ---------------------------------------------------------------------------

def plot_degradation_curves(level_table: pd.DataFrame, out_dir) -> None:
    """
    One figure per perturbation type, with a subplot per metric showing the
    mean metric value across all tokens at each perturbation level, wrapped
    in nested bands: a light +/- std band (spread of the data) and a darker
    95% CI band (uncertainty of the mean). The breaking threshold (-40% from
    baseline) is drawn for the higher-is-better metrics so it is visible
    *how far* the perturbation can go before the model breaks.

    The single global ``Normale`` baseline (one block in the level table) is
    prepended as the first point of every curve, so the per-perturbation
    levels are all read against the same reference.
    """
    save_dir = Path(out_dir)
    _ensure_dir(save_dir)
    if level_table.empty:
        logger.warning("Level table is empty — skipping degradation curves.")
        return

    # Single global baseline per metric (the one "Normale" block).
    base = level_table[level_table["perturbation_type"] == "Normale"]
    baseline = {}
    for metric in METRICS:
        r = base[base["metric"] == metric]
        if r.empty:
            baseline[metric] = (np.nan, np.nan, np.nan, np.nan)
        else:
            r0 = r.iloc[0]
            baseline[metric] = (float(r0["mean"]), float(r0["std"]),
                                float(r0["ci95_low"]), float(r0["ci95_high"]))

    for ptype, g in level_table.groupby("perturbation_type"):
        if ptype == "Normale":
            continue

        fig, axes = plt.subplots(
            1, len(METRICS),
            figsize=(6.0 * len(METRICS), 5.0), constrained_layout=True,
        )
        if len(METRICS) == 1:
            axes = [axes]

        n_tokens = int(g["n_tokens"].max()) if not g["n_tokens"].isna().all() else 0

        for ax, metric in zip(axes, METRICS):
            sub = g[g["metric"] == metric].sort_values("level_index")
            bmean, bstd, bci_lo, bci_hi = baseline[metric]

            # Prepend the global baseline as the first ("Normale") point.
            mean  = np.concatenate(([bmean],  sub["mean"].to_numpy(dtype=float)))
            std   = np.concatenate(([bstd],   sub["std"].to_numpy(dtype=float)))
            ci_lo = np.concatenate(([bci_lo], sub["ci95_low"].to_numpy(dtype=float)))
            ci_hi = np.concatenate(([bci_hi], sub["ci95_high"].to_numpy(dtype=float)))
            labels = ["Normale"] + list(sub["param_label"])
            x = np.arange(len(mean))
            color = _PALETTE[METRICS.index(metric) % len(_PALETTE)]

            std_filled = np.nan_to_num(std, nan=0.0)
            ax.fill_between(x, mean - std_filled, mean + std_filled,
                            color=color, alpha=0.18, label="± std", zorder=1)
            ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.40,
                            label="95% CI", zorder=2)
            ax.plot(x, mean, marker="o", markersize=5, linewidth=2.0,
                    color=color, zorder=3)

            # annotate each perturbation level with its % change vs baseline
            pct = sub["pct_vs_baseline"].to_numpy(dtype=float)
            for xi, yi, pv in zip(x[1:], mean[1:], pct):
                if np.isnan(yi) or np.isnan(pv):
                    continue
                ax.annotate(f"{pv:+.0f}%", xy=(xi, yi), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=6.5, color=color, fontweight="bold", zorder=4)

            if not np.isnan(bmean):
                ax.axhline(bmean, color="grey", linestyle=":", linewidth=1.1,
                           alpha=0.8, label=f"baseline ({bmean:.2f})", zorder=1)
                if metric in ("obj_iou", "io_ratio"):
                    ax.axhline(bmean * 0.6, color="red", linestyle="--", linewidth=1.1,
                               alpha=0.7, label="break thr (-40%)", zorder=1)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_xlabel("Perturbation level")
            ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
            ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=11, fontweight="bold")

            lo, hi = _METRIC_YLIM.get(metric, (0.0, None))
            if lo is not None and hi is not None:
                ax.set_ylim(lo, hi)
            elif lo is not None:
                ax.set_ylim(bottom=lo)

            ax.grid(axis="y", alpha=0.4, linewidth=0.4)
            ax.legend(fontsize=7, loc="best")

        fig.suptitle(
            f"Degradation curve (mean over all images) — {ptype}   (n={n_tokens} tokens)",
            fontsize=13, fontweight="bold",
        )
        path = save_dir / f"degradation_curve_{_safe_filename(str(ptype))}.png"
        fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved degradation curve: %s", path)


# ---------------------------------------------------------------------------
# 2 — Paired significance (one small figure per perturbation type)
# ---------------------------------------------------------------------------

def plot_paired_significance(paired_table: pd.DataFrame, out_dir) -> None:
    """
    One compact figure per perturbation type (``paired_significance_<ptype>``),
    with a subplot per metric showing the paired comparison directly: a
    "baseline" bar vs an "at break" bar, with a significance bracket on top
    (stars + p-value) and the number of pairs in the subplot title.

    This replaces the previous single, overcrowded figure that squeezed all
    perturbations onto one axis: here each perturbation gets its own small,
    readable panel.
    """
    save_dir = Path(out_dir)
    _ensure_dir(save_dir)
    if paired_table.empty:
        logger.warning("Paired table is empty — skipping significance plots.")
        return

    for ptype, g in paired_table.groupby("perturbation_type"):
        g = g.set_index("metric")

        fig, axes = plt.subplots(
            1, len(METRICS), figsize=(3.6 * len(METRICS), 3.6),
            constrained_layout=True,
        )
        if len(METRICS) == 1:
            axes = [axes]

        for ax, metric in zip(axes, METRICS):
            color = _PALETTE[METRICS.index(metric) % len(_PALETTE)]
            if metric not in g.index:
                ax.set_axis_off()
                continue

            row = g.loc[metric]
            base = float(row["mean_baseline"])
            comp = float(row["mean_compare"])
            n = int(row["n_pairs"]) if not pd.isna(row["n_pairs"]) else 0
            p = row["wilcoxon_pvalue"]
            if pd.isna(p):
                p = row["ttest_pvalue"]
            stars = _p_to_stars(p)

            bars = ax.bar([0, 1], [base, comp], width=0.6,
                          color=["lightgrey", color], edgecolor="black", linewidth=0.6)
            for b, val in zip(bars, (base, comp)):
                if not np.isnan(val):
                    ax.annotate(f"{val:.2f}", xy=(b.get_x() + b.get_width() / 2, val),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=8)

            # significance bracket above the taller bar
            top = np.nanmax([base, comp])
            if not np.isnan(top):
                y = top * 1.12 + 1e-6
                ax.plot([0, 0, 1, 1], [top * 1.04, y, y, top * 1.04],
                        color="black", linewidth=1.0)
                p_txt = "" if pd.isna(p) else f"\np={p:.2g}"
                ax.text(0.5, y, f"{stars}{p_txt}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
                ax.set_ylim(0, y * 1.25)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["baseline", "at break"], fontsize=8)
            ax.set_title(f"{_METRIC_LABELS.get(metric, metric)}  (n={n})",
                         fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.4, linewidth=0.4)

        fig.suptitle(
            f"Paired test: baseline vs. break — {ptype}\n"
            "* p<0.05   ** p<0.01   *** p<0.001   ns = not significant",
            fontsize=11, fontweight="bold",
        )
        path = save_dir / f"paired_significance_{_safe_filename(str(ptype))}.png"
        fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved paired-significance plot: %s", path)


