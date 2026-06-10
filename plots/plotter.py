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
    "wdp":     (0.0, None),
}

_FIGURE_DPI  = 150
_PALETTE     = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_BASELINE_X    = 0.0          # param_value assigned to "Normale"
WDP_PENALTY_NO_HEATMAP = 5.0
WDP_PENALTY_NO_TOKEN = 5.0
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
        if lo is not None and hi is not None:
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
        
        if len(vals) > 0:
            val = vals.mean()
            # First case: token generated but heatmap empty (TAM < 20%) 
            if metric == "wdp" and val == 0.0:
                y.append(WDP_PENALTY_NO_HEATMAP)
            else:
                y.append(val)
        else:
            # Second case: token lost (NaN in CSV)
            y.append(0.0 if metric in ["obj_iou", "io_ratio"] else WDP_PENALTY_NO_TOKEN)
            
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
                missing_mask = []  # points where token is completely missing across ALL perturbation levels for this category
                
                for x in x_values:
                    vals = cat_df.loc[cat_df["param_value"] == x, metric].dropna()
                    
                    if len(vals) > 0:
                        val = vals.mean()
                        missing_mask.append(False)
                        # First case: token generated but heatmap empty (TAM < 20%)
                        if metric == "wdp" and val == 0.0:
                            means.append(WDP_PENALTY_NO_HEATMAP)
                            stds.append(0.0) 
                        else:
                            means.append(val)
                            stds.append(vals.std() if len(vals) > 1 else 0.0)
                    else:
                        missing_mask.append(True)
                        # Second case: token lost (NaN in CSV)
                        means.append(0.0 if metric in ["obj_iou", "io_ratio"] else WDP_PENALTY_NO_TOKEN)
                        stds.append(0.0)

                means = np.array(means, dtype=float)
                stds  = np.array(stds,  dtype=float)

                # skip entire line if token is completely missing across all perturbation levels (e.g., token not generated for this category at all)
                if np.all(missing_mask):
                     continue
                     
                x_pos = list(range(len(x_values)))
                
                # Principal line with mean values
                ax.plot(x_pos, means, marker="o", markersize=4,
                        label=category, color=color, linewidth=1.4)
                
                # draw "X" markers for points where the token is completely missing (NaN across all perturbation levels for this category)
                for i, is_missing in enumerate(missing_mask):
                    if is_missing:
                        ax.plot(x_pos[i], means[i], marker="X", color="red", markersize=6, zorder=4)
                
                # std shading, clipped to valid metric range
                ax.fill_between(x_pos,
                                np.where(np.isnan(means), np.nan, np.clip(means - stds, 0.0, None if metric == "wdp" else 1.0)),
                                np.where(np.isnan(means), np.nan, np.clip(means + stds, 0.0, None if metric == "wdp" else 1.0)),
                                alpha=0.15, color=color)
                
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
    min_consecutive: int = 2,
) -> Optional[int]:
    n = len(obj_iou_means)
    run_start: Optional[int] = None
    run_len = 0

    bv_obj = baseline_vals.get("obj_iou", np.nan)
    bv_io  = baseline_vals.get("io_ratio", np.nan)
    bv_wdp = baseline_vals.get("wdp", np.nan)

    # If the baseline is missing or already at the worst possible value, we cannot detect a break (since we have no reference or room to degrade), so we return 0 to indicate an immediate break at the first perturbation level.
    if (not np.isnan(bv_obj) and bv_obj == 0.0) or (not np.isnan(bv_io) and bv_io == 0.0):
        return 0

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
    degradation_threshold: float = 0.40,
    wdp_arbiter_min: float = 0.0,
    min_consecutive: int = 2,
) -> None:
    save_dir = Path(out_dir) / "breaking_point_per_token"
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
            n_cats = len(categories)

            # Create a grid of subplots: one row per category, one column per metric
            fig, axes_2d = plt.subplots(
                n_cats, len(METRICS), 
                figsize=(15.0, 4.5 * n_cats), 
                constrained_layout=True
            )
            
            # If there's only one category, axes_2d will be 1D; we need to make it 2D for consistent indexing
            if n_cats == 1:
                axes_2d = np.array([axes_2d])
                
            fig.suptitle(
                f"Per-Token Breaking Point | image {img_id} — {ptype}\n"
                f"(threshold: {degradation_threshold*100:.0f}%, min_consecutive: {min_consecutive})",
                fontsize=14, fontweight="bold",
            )

            # For each category, plot the metric lines and detect the breaking point based on the defined criteria
            for row_idx, category in enumerate(categories):
                axes = axes_2d[row_idx]
                color = _PALETTE[row_idx % len(_PALETTE)]
                cat_df = img_df[img_df["category"] == category]
                
                cat_series: Dict[str, np.ndarray] = {}
                baseline_vals: Dict[str, float] = {}
                token_missing_mask = {} # to track if token is completely missing (NaN) for each metric across all perturbation levels for this category

                for metric in METRICS:
                    y = []
                    missing_mask = []
                    
                    for x in x_values:
                        vals = cat_df.loc[cat_df["param_value"] == x, metric].dropna()
                        if len(vals) > 0:
                            val = vals.mean()
                            missing_mask.append(False)
                            if metric == "wdp" and val == 0.0:
                                y.append(WDP_PENALTY_NO_HEATMAP)
                            else:
                                y.append(val)
                        else:
                            # Token not generated
                            missing_mask.append(True)
                            y.append(0.0 if metric in ["obj_iou", "io_ratio"] else WDP_PENALTY_NO_TOKEN)
                            
                    y = np.array(y, dtype=float)
                    cat_series[metric] = y
                    baseline_vals[metric] = y[0]
                    token_missing_mask[metric] = missing_mask

                # Detect breaking point for this category using the defined criteria
                bp_idx = detect_breaking_point(
                    obj_iou_means  = cat_series["obj_iou"],
                    io_ratio_means = cat_series["io_ratio"],
                    wdp_means      = cat_series["wdp"],
                    baseline_vals  = baseline_vals,
                    threshold      = degradation_threshold,
                    wdp_arbiter_min= wdp_arbiter_min,
                    min_consecutive= min_consecutive,
                )

                # Plot the metric lines, baseline, and annotate the breaking point if detected
                for col_idx, metric in enumerate(METRICS):
                    ax = axes[col_idx]
                    direction = _METRIC_DIRECTION[metric]
                    y = cat_series[metric]
                    missing = token_missing_mask[metric]

                    # line plot for this category and metric
                    ax.plot(x_pos, y, marker="o", markersize=6, linewidth=2.0,
                            color=color, label=f"{category}", zorder=3)
                            
                    # mark points where the token is completely missing across all perturbation levels for this category (NaN in CSV) with a red "X" and annotation for WDP
                    for i, is_missing in enumerate(missing):
                        if is_missing:
                            ax.plot(x_pos[i], y[i], marker="X", color="red", 
                                    markersize=10, zorder=4)
                            if metric == "wdp":
                                ax.annotate("TOKEN PERSO", xy=(i, y[i]), xytext=(0, 10),
                                            textcoords="offset points", ha="center", 
                                            fontsize=7, color="red", fontweight="bold")

                    # Baseline
                    bv = baseline_vals.get(metric, np.nan)
                    if not np.isnan(bv):
                        ax.axhline(bv, color="grey", linewidth=1.2, linestyle=":", alpha=0.8,
                                   label=f"baseline ({bv:.2f})", zorder=1)

                    # breaking point annotation
                    if bp_idx is not None:
                        ax.axvline(bp_idx, color="red", linewidth=1.6, linestyle="--", alpha=0.85, zorder=2)
                        
                        if bp_idx == 0:
                            ax.text(0.05, 0.90, "Rotto in\nBaseline", transform=ax.transAxes,
                                    fontsize=8, color="darkred", fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.25", fc="lightpink", ec="darkred", alpha=0.85), zorder=5)
                        else:
                            ax.axvspan(bp_idx, len(x_pos) - 1, color="red", alpha=0.07, zorder=0)
                            bp_val = y[bp_idx]
                            
                            if not missing[bp_idx]: # Se il token c'è, calcoliamo la percentuale
                                deg_pct = _relative_degradation(bp_val, bv, direction) * 100
                                label = f"Δ = {deg_pct:+.1f} %" if metric != "wdp" else f"Δ wdp = {deg_pct:+.1f} %"
                            else:
                                label = "CROLLO TOTALE"
                                
                            current_lo, current_hi = ax.get_ylim()
                            ax.annotate(
                                label, xy=(bp_idx, bp_val),
                                xytext=(min(bp_idx + 0.4, len(x_pos) - 1.5), current_lo + (current_hi - current_lo) * 0.88),
                                fontsize=7.5, color="darkred", fontweight="bold",
                                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
                                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="darkred", alpha=0.9), zorder=5,
                            )

                    # titles, labels, limits, grid
                    ax.set_title(f"{category} | {_METRIC_LABELS.get(metric, metric)}", fontsize=10, fontweight="bold")
                    ax.set_xlabel("Perturbation strength")
                    ax.set_ylabel(_METRIC_LABELS.get(metric, metric))
                    
                    lo, hi = _METRIC_YLIM.get(metric, (0.0, 1.0))
                    if lo is not None and hi is not None:
                        ax.set_ylim(lo, hi)
                    elif lo is not None:
                        ax.set_ylim(bottom=lo)
                        
                    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                    ax.grid(axis="y", linewidth=0.4, alpha=0.6)
                    _apply_x_axis(ax, x_values, x_labels)
                    ax.legend(loc="best", fontsize=8)

            save_path = ptype_dir / f"{_safe_filename(str(img_id))}.png"
            fig.savefig(save_path, dpi=_FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved per-token breaking point profile: %s", save_path)