import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from data_loader import METRICS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "obj_iou": "Object IoU",
    "io_ratio": "I/O Ratio",
    "wdp":      "WDP",
}

_METRIC_YLIM = {
    "obj_iou": (0.0, 1.0),
    "io_ratio": (0.0, 1.0),
    "wdp":     (0.0, 1.0),
}

_FIGURE_DPI  = 150
_PALETTE     = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_BASELINE_X    = 0.0          # param_value assigned to "Normale"
_BASELINE_TYPE = "Normale"    # perturbation_type label for baseline rows


def _with_baseline(df: pd.DataFrame, ptype: str) -> pd.DataFrame:
    if ptype == _BASELINE_TYPE:
        return df[df["perturbation_type"] == ptype].copy()

    perturbed = df[df["perturbation_type"] == ptype].copy()
    baseline  = df[df["perturbation_type"] == _BASELINE_TYPE].copy()

    if baseline.empty:
        logger.warning(
            "No baseline ('%s') rows found — plots for '%s' will not include "
            "a starting reference point.", _BASELINE_TYPE, ptype,
        )
        return perturbed

    # Re-tag baseline rows so they blend into this perturbation's DataFrame
    baseline["perturbation_type"]  = ptype
    baseline["perturbation_param"] = _BASELINE_TYPE
    baseline["param_value"]        = _BASELINE_X

    return pd.concat([baseline, perturbed], ignore_index=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


def _ordered_x(x_values) -> List[float]:
    return sorted(set(float(x) for x in x_values))


def _x_labels(sub_df: pd.DataFrame, x_values: List[float]) -> List[str]:
    mapping: Dict[float, str] = (
        sub_df.drop_duplicates("param_value")
        .set_index("param_value")["perturbation_param"]
        .to_dict()
    )
    labels = []
    for x in x_values:
        if x == _BASELINE_X:
            labels.append("Normale")
        else:
            labels.append(mapping.get(x, str(x)))
    return labels


def _metric_axes(fig_width: float = 15.0, fig_height: float = 4.5):
    fig, axes = plt.subplots(
        1, len(METRICS),
        figsize=(fig_width, fig_height),
        constrained_layout=True,
    )
    for ax, metric in zip(axes, METRICS):
        ax.set_xlabel("Perturbation strength")
        ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
        ax.set_title(_METRIC_LABELS.get(metric, metric))
        lo, hi = _METRIC_YLIM.get(metric, (None, None))
        if lo is not None:
            ax.set_ylim(lo, hi)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        ax.grid(axis="y", which="minor", linewidth=0.2, alpha=0.3)
    return fig, axes


def _apply_x_axis(ax, x_values: List[float], labels: List[str]) -> None:
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    # Highlight the baseline column
    if _BASELINE_X in x_values:
        baseline_pos = x_values.index(_BASELINE_X)
        ax.axvline(
            x=baseline_pos,
            color="grey", linewidth=0.9, linestyle=":",
            alpha=0.7, label="baseline",
        )


def _mark_missing(ax, x_pos: int, metric: str) -> None:
    lo, _ = _METRIC_YLIM.get(metric, (0.0, 1.0))
    ax.plot(x_pos, lo + 0.02, marker="x", color="lightgrey",
            markersize=8, markeredgewidth=1.5, zorder=5)
    ax.annotate(
        "N/A",
        xy=(x_pos, lo + 0.02),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center", fontsize=5.5,
        color="grey",
    )


def _add_legend(axes, ncol: int = 4) -> None:
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    axes[-1].legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        fontsize=7,
        framealpha=0.8,
    )


def _series_for_category(
    cat_df: pd.DataFrame,
    x_values: List[float],
    metric: str,
) -> np.ndarray:
    y = []
    for x in x_values:
        vals = cat_df.loc[cat_df["param_value"] == x, metric].dropna()
        y.append(vals.mean() if len(vals) else np.nan)
    return np.array(y, dtype=float)


# ---------------------------------------------------------------------------
# 1 – Overview: one figure per perturbation type
# ---------------------------------------------------------------------------

def plot_perturbation_overview(
    df: pd.DataFrame,
    out_dir: str,
    perturbation_types: Optional[List[str]] = None,
) -> None:
    save_dir = Path(out_dir) / "overview"
    _ensure_dir(save_dir)

    types = perturbation_types or sorted(df["perturbation_type"].unique())

    for ptype in types:
        sub = _with_baseline(df, ptype)
        if sub.empty:
            continue

        x_values   = _ordered_x(sub["param_value"].unique())
        x_labels   = _x_labels(sub, x_values)
        categories = sorted(sub["category"].unique())

        fig, axes = _metric_axes()
        fig.suptitle(f"Perturbation: {ptype}", fontsize=13, fontweight="bold")

        for cat_idx, category in enumerate(categories):
            color  = _PALETTE[cat_idx % len(_PALETTE)]
            cat_df = sub[sub["category"] == category]

            for ax, metric in zip(axes, METRICS):
                means, stds = [], []
                for x in x_values:
                    vals = cat_df.loc[cat_df["param_value"] == x, metric].dropna()
                    means.append(vals.mean() if len(vals) else np.nan)
                    stds.append(vals.std()  if len(vals) > 1 else 0.0)

                means = np.array(means, dtype=float)
                stds  = np.array(stds,  dtype=float)

                # Skip entirely-missing categories
                if np.all(np.isnan(means)):
                    logger.debug("overview [%s/%s] all values missing — skipped.", ptype, category)
                    continue

                x_pos = list(range(len(x_values)))
                ax.plot(x_pos, means, marker="o", markersize=4,
                        label=category, color=color, linewidth=1.4)
                ax.fill_between(x_pos,
                                np.where(np.isnan(means - stds), np.nan, means - stds),
                                np.where(np.isnan(means + stds), np.nan, means + stds),
                                alpha=0.15, color=color)

                # Mark missing individual points
                for i, (x, y) in enumerate(zip(x_values, means)):
                    if np.isnan(y):
                        _mark_missing(ax, i, metric)

                _apply_x_axis(ax, x_values, x_labels)

        _add_legend(axes)
        save_path = save_dir / f"{_safe_filename(ptype)}_overview.png"
        fig.savefig(save_path, dpi=_FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved overview plot: %s", save_path)


# ---------------------------------------------------------------------------
# 3 – Heatmap: images × param_values
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    df: pd.DataFrame,
    out_dir: str,
    perturbation_types: Optional[List[str]] = None,
) -> None:
    save_dir = Path(out_dir) / "heatmaps"
    _ensure_dir(save_dir)

    types = perturbation_types or sorted(df["perturbation_type"].unique())

    for ptype in types:
        sub = _with_baseline(df, ptype)
        if sub.empty:
            continue

        x_values = _ordered_x(sub["param_value"].unique())
        x_labels = _x_labels(sub, x_values)
        img_ids  = sorted(sub["img_id"].unique())

        for metric in METRICS:
            matrix = np.full((len(img_ids), len(x_values)), np.nan)
            for i, img_id in enumerate(img_ids):
                for j, x in enumerate(x_values):
                    vals = sub.loc[
                        (sub["img_id"] == img_id) & (sub["param_value"] == x),
                        metric,
                    ].dropna()
                    if len(vals):
                        matrix[i, j] = vals.mean()

            fig_h = max(4.0, 0.35 * len(img_ids))
            fig_w = max(6.0, 0.7  * len(x_values))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

            # Use a masked array so NaN cells render as white
            masked = np.ma.masked_invalid(matrix)
            cmap   = plt.cm.RdYlGn.copy()
            cmap.set_bad(color="white")

            im = ax.imshow(masked, aspect="auto", cmap=cmap,
                           vmin=0.0, vmax=1.0, interpolation="nearest")
            fig.colorbar(im, ax=ax, label=_METRIC_LABELS.get(metric, metric),
                         fraction=0.03, pad=0.02)

            ax.set_xticks(range(len(x_values)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(img_ids)))
            ax.set_yticklabels(img_ids, fontsize=7)
            ax.set_xlabel("Perturbation strength")
            ax.set_ylabel("Image ID")
            ax.set_title(
                f"{ptype}  —  {_METRIC_LABELS.get(metric, metric)}",
                fontsize=11, fontweight="bold",
            )

            # Annotate each cell
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    text = f"{val:.2f}" if not np.isnan(val) else "–"
                    color = "black" if not np.isnan(val) else "lightgrey"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=6, color=color)

            fname = f"{_safe_filename(ptype)}_{metric}_heatmap.png"
            save_path = save_dir / fname
            fig.savefig(save_path, dpi=_FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved heatmap: %s", save_path)


# ---------------------------------------------------------------------------
# 4 – Image profile: one image × one perturbation, all params
# ---------------------------------------------------------------------------

def plot_image_profile(
    df: pd.DataFrame,
    out_dir: str,
    perturbation_types: Optional[List[str]] = None,
    image_ids: Optional[List[str]] = None,
) -> None:
    save_dir = Path(out_dir) / "image_profile"
    _ensure_dir(save_dir)

    types = perturbation_types or sorted(df["perturbation_type"].unique())
    ids   = image_ids or sorted(df["img_id"].unique())

    for ptype in types:
        type_df = _with_baseline(df, ptype)
        if type_df.empty:
            continue

        ptype_dir = save_dir / _safe_filename(ptype)
        _ensure_dir(ptype_dir)

        x_values = _ordered_x(type_df["param_value"].unique())
        x_labels = _x_labels(type_df, x_values)
        x_pos    = list(range(len(x_values)))

        for img_id in ids:
            img_df = type_df[type_df["img_id"] == img_id]
            if img_df.empty:
                continue

            categories = sorted(img_df["category"].unique())

            fig, axes = _metric_axes(fig_width=15.0, fig_height=5.0)
            fig.suptitle(
                f"Image {img_id}  —  {ptype}",
                fontsize=13, fontweight="bold",
            )

            for ax, metric in zip(axes, METRICS):
                cat_series: Dict[str, np.ndarray] = {}

                # ---- per-category thin lines ----
                for cat_idx, category in enumerate(categories):
                    color  = _PALETTE[cat_idx % len(_PALETTE)]
                    cat_df = img_df[img_df["category"] == category]
                    y      = _series_for_category(cat_df, x_values, metric)

                    if np.all(np.isnan(y)):
                        logger.debug(
                            "image_profile [%s/%s/%s] all values missing — skipped.",
                            ptype, img_id, category,
                        )
                        continue

                    cat_series[category] = y

                    ax.plot(x_pos, y, marker="o", markersize=5, linewidth=1.3,
                            color=color, alpha=0.75, label=category, zorder=2)

                    # Annotate present values; mark missing ones
                    for i, v in enumerate(y):
                        if not np.isnan(v):
                            ax.annotate(
                                f"{v:.2f}",
                                xy=(i, v),
                                xytext=(0, 6),
                                textcoords="offset points",
                                ha="center", fontsize=6,
                                color=color, alpha=0.85,
                            )
                        else:
                            _mark_missing(ax, i, metric)

                # ---- bold mean line across categories with data ----
                # np.nanmean warns "Mean of empty slice" when an entire
                # x-position has NaN across ALL categories (token not
                # generated by any category at that param level).
                # The result is still NaN — correct — so we silence it.
                if cat_series:
                    with np.errstate(all="ignore"):
                        all_means = np.nanmean(
                            np.vstack(list(cat_series.values())), axis=0
                        )
                    ax.plot(
                        x_pos, all_means,
                        marker="D", markersize=6, linewidth=2.2,
                        color="black", linestyle="--",
                        label="mean (all cat.)", zorder=3,
                    )
                    for i, v in enumerate(all_means):
                        if not np.isnan(v):
                            ax.annotate(
                                f"{v:.2f}",
                                xy=(i, v),
                                xytext=(0, -12),
                                textcoords="offset points",
                                ha="center", fontsize=6.5,
                                fontweight="bold", color="black",
                            )

                _apply_x_axis(ax, x_values, x_labels)

            _add_legend(axes)

            save_path = ptype_dir / f"{_safe_filename(str(img_id))}.png"
            fig.savefig(save_path, dpi=_FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved image profile: %s", save_path)



# ---------------------------------------------------------------------------
# Breaking-point detection — helpers
# ---------------------------------------------------------------------------

# obj_iou and io_ratio: higher = better  → degradation = drop from baseline
# wdp: lower = better (it is a penalty)  → degradation = rise from baseline
_METRIC_DIRECTION = {
    "obj_iou":  "higher_better",
    "io_ratio": "higher_better",
    "wdp":      "lower_better",
}


def _relative_degradation(
    value: float, baseline: float, direction: str, eps: float = 1e-9
) -> float:
    if direction == "higher_better":
        return (baseline - value) / (abs(baseline) + eps)
    return (value - baseline) / (abs(baseline) + eps)


def detect_breaking_point(
    obj_iou_means: np.ndarray,
    io_ratio_means: np.ndarray,
    wdp_means: np.ndarray,
    baseline_vals: Dict[str, float],
    threshold: float = 0.40,
    wdp_arbiter_min: float = 0.0,
    min_consecutive: int = 1,
) -> Optional[int]:
    n = len(obj_iou_means)
    run_start: Optional[int] = None
    run_len = 0

    bv_obj = baseline_vals.get("obj_iou", np.nan)
    bv_io  = baseline_vals.get("io_ratio", np.nan)
    bv_wdp = baseline_vals.get("wdp", np.nan)

    for i in range(1, n):
        # --- primary: composite of obj_iou + io_ratio ---
        degs = []
        if not np.isnan(obj_iou_means[i]) and not np.isnan(bv_obj):
            degs.append(_relative_degradation(obj_iou_means[i], bv_obj, "higher_better"))
        if not np.isnan(io_ratio_means[i]) and not np.isnan(bv_io):
            degs.append(_relative_degradation(io_ratio_means[i], bv_io, "higher_better"))

        if not degs:
            run_start, run_len = None, 0
            continue

        primary_deg = float(np.mean(degs))

        # --- arbiter: wdp must confirm ---
        if not np.isnan(wdp_means[i]) and not np.isnan(bv_wdp):
            wdp_deg = _relative_degradation(wdp_means[i], bv_wdp, "lower_better")
        else:
            wdp_deg = np.nan

        wdp_confirms = np.isnan(wdp_deg) or (wdp_deg >= wdp_arbiter_min)

        if primary_deg >= threshold and wdp_confirms:
            if run_start is None:
                run_start = i
            run_len += 1
            if run_len >= min_consecutive:
                return run_start
        else:
            run_start, run_len = None, 0

    return None


# ---------------------------------------------------------------------------
# 4 – Breaking-point profile
# ---------------------------------------------------------------------------

def plot_breaking_point_profile(
    df: pd.DataFrame,
    out_dir: str,
    perturbation_types: Optional[List[str]] = None,
    image_ids: Optional[List[str]] = None,
    degradation_threshold: float = 0.20,
    wdp_arbiter_min: float = 0.0,
    min_consecutive: int = 1,
) -> None:
    save_dir = Path(out_dir) / "breaking_point"
    _ensure_dir(save_dir)

    types = perturbation_types or sorted(df["perturbation_type"].unique())
    ids   = image_ids or sorted(df["img_id"].unique())

    for ptype in types:
        type_df = _with_baseline(df, ptype)
        if type_df.empty:
            continue

        ptype_dir = save_dir / _safe_filename(ptype)
        _ensure_dir(ptype_dir)

        x_values = _ordered_x(type_df["param_value"].unique())
        x_labels = _x_labels(type_df, x_values)
        x_pos    = list(range(len(x_values)))

        for img_id in ids:
            img_df = type_df[type_df["img_id"] == img_id]
            if img_df.empty:
                continue

            categories = sorted(img_df["category"].unique())

            fig, axes = _metric_axes(fig_width=15.0, fig_height=5.5)
            fig.suptitle(
                f"Breaking Point  |  image {img_id}  —  {ptype}"
                f"  (primary threshold: {degradation_threshold*100:.0f} %,"
                f"  wdp arbiter ≥ {wdp_arbiter_min*100:.0f} %)",
                fontsize=11, fontweight="bold",
            )

            # ----------------------------------------------------------------
            # Build mean series for every metric
            # ----------------------------------------------------------------
            mean_series: Dict[str, np.ndarray] = {}
            baseline_vals: Dict[str, float]    = {}

            for metric in METRICS:
                cat_series: Dict[str, np.ndarray] = {}
                for category in categories:
                    cat_df = img_df[img_df["category"] == category]
                    y = _series_for_category(cat_df, x_values, metric)
                    if not np.all(np.isnan(y)):
                        cat_series[category] = y

                if cat_series:
                    with np.errstate(all="ignore"):
                        means = np.nanmean(np.vstack(list(cat_series.values())), axis=0)
                    mean_series[metric]   = means
                    baseline_vals[metric] = means[0] if not np.isnan(means[0]) else np.nan

            # ----------------------------------------------------------------
            # Detect the single breaking point
            # ----------------------------------------------------------------
            _empty = np.full(len(x_values), np.nan)
            bp_idx = detect_breaking_point(
                obj_iou_means  = mean_series.get("obj_iou",  _empty),
                io_ratio_means = mean_series.get("io_ratio", _empty),
                wdp_means      = mean_series.get("wdp",      _empty),
                baseline_vals  = baseline_vals,
                threshold      = degradation_threshold,
                wdp_arbiter_min= wdp_arbiter_min,
                min_consecutive= min_consecutive,
            )

            if bp_idx is not None:
                logger.debug(
                    "[%s/%s] breaking point at index %d ('%s')",
                    ptype, img_id, bp_idx, x_labels[bp_idx],
                )

            # ----------------------------------------------------------------
            # Draw metric subplots
            # ----------------------------------------------------------------
            for ax, metric in zip(axes, METRICS):
                direction = _METRIC_DIRECTION[metric]

                # Per-category thin lines
                for cat_idx, category in enumerate(categories):
                    color  = _PALETTE[cat_idx % len(_PALETTE)]
                    cat_df = img_df[img_df["category"] == category]
                    y = _series_for_category(cat_df, x_values, metric)
                    if np.all(np.isnan(y)):
                        continue
                    ax.plot(x_pos, y, marker="o", markersize=4, linewidth=1.2,
                            color=color, alpha=0.55, label=category, zorder=2)
                    for i, v in enumerate(y):
                        if np.isnan(v):
                            _mark_missing(ax, i, metric)

                # Bold mean line
                if metric in mean_series:
                    means = mean_series[metric]
                    ax.plot(x_pos, means, marker="D", markersize=5,
                            linewidth=2.0, color="black", linestyle="--",
                            label="mean", zorder=3)

                    # Horizontal baseline reference
                    bv = baseline_vals.get(metric, np.nan)
                    if not np.isnan(bv):
                        ax.axhline(bv, color="steelblue", linewidth=0.9,
                                   linestyle=":", alpha=0.8,
                                   label=f"baseline ({bv:.2f})", zorder=1)

                lo, hi = _METRIC_YLIM.get(metric, (0.0, 1.0))

                # ---- Single breaking point marked on every subplot ----
                if bp_idx is not None:
                    ax.axvline(bp_idx, color="red", linewidth=1.6,
                               linestyle="--", alpha=0.85, zorder=4)
                    ax.axvspan(bp_idx, len(x_pos) - 1,
                               color="red", alpha=0.07, zorder=0)

                    if metric in mean_series:
                        bp_val = mean_series[metric][bp_idx]
                        bv     = baseline_vals.get(metric, np.nan)

                        if not np.isnan(bp_val) and not np.isnan(bv):
                            deg_pct = _relative_degradation(bp_val, bv, direction) * 100

                            # On wdp subplot: clarify its arbiter role
                            if metric == "wdp":
                                label = (f"Δ wdp = {deg_pct:+.1f} %"
                                         f"\n(arbiter {'✓ confirmed' if deg_pct >= wdp_arbiter_min*100 else '– not triggered'})")
                            else:
                                label = f"Δ = {deg_pct:+.1f} %"

                            ax.annotate(
                                label,
                                xy=(bp_idx, bp_val),
                                xytext=(min(bp_idx + 0.4, len(x_pos) - 1.5),
                                        lo + (hi - lo) * 0.88),
                                fontsize=6.5, color="darkred", fontweight="bold",
                                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.9),
                                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow",
                                          ec="darkred", alpha=0.85),
                                zorder=5,
                            )
                else:
                    ax.text(
                        0.97, 0.05, "No break detected",
                        transform=ax.transAxes,
                        ha="right", va="bottom",
                        fontsize=7, color="grey", fontstyle="italic",
                    )

                ax.set_title(
                    _METRIC_LABELS.get(metric, metric)
                    + (" ← primary" if metric in ("obj_iou", "io_ratio") else " ← arbiter"),
                    fontsize=9,
                )
                ax.set_xlabel("Perturbation strength")
                ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
                if lo is not None:
                    ax.set_ylim(lo, hi)
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.grid(axis="y", linewidth=0.4, alpha=0.6)
                _apply_x_axis(ax, x_values, x_labels)

            _add_legend(axes)

            save_path = ptype_dir / f"{_safe_filename(str(img_id))}.png"
            fig.savefig(save_path, dpi=_FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved breaking point profile: %s", save_path)