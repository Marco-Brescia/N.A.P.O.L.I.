import logging
import re
from pathlib import Path
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import math
    
sys.path.append(str(Path(__file__).resolve().parent.parent))
from perturbations.config import perturbation_ranges

from data_loader import METRICS

logger = logging.getLogger(__name__)

def generate_expected_params(config: dict) -> dict:
    expected_params = {}
    for name, cfg in config.items():
        start = cfg["min"]
        stop = cfg["max"]
        step = round(cfg["step"], 2)

        print(f"Generating expected params for {name}: start={start}, stop={stop}, step={step}")
        
        if cfg["type"] == int:
            stop_inclusive = int(stop) + (1 if step > 0 else -1)
            values = list(range(int(start), stop_inclusive, int(step)))
        else:
            raw_values = np.arange(start, stop + step * 0.01, step)

            # round(...) can yield -0.0 from a tiny negative float residue
            # (e.g. 0.9 - 6*0.15 with a negative step); adding 0.0 normalises
            # the negative zero back to a plain 0.0.
            values = [round(v, 2) + 0.0 for v in raw_values]
            
        expected_params[name] = values
    return expected_params

_EXPECTED_PARAMS = generate_expected_params(perturbation_ranges)

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
_BASELINE_X    = -1.0         # sentinel param_value for "Normale"; must not
                              # collide with any real level (e.g. Saturation
                              # factor_0.0, which is a genuine perturbation).
_BASELINE_TYPE = "Normale"    # perturbation_type label for baseline rows

WDP_PENALTY_NO_HEATMAP = 5.0
WDP_PENALTY_NO_TOKEN = 5.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-.]", "_", name)


# Perturbations whose levels must be ordered DESCENDING (baseline still first),
# because for them a higher numeric parameter means a WEAKER perturbation:
# CannyEdge (higher threshold = fewer edges), JPEGCompression (higher quality
# = less compression), Saturation (higher factor = closer to the original),
# Poisson/Shot noise (higher scale = less noise). Must stay in sync with the
# same set in aggregate_stats.py.
_REVERSED_LEVEL_TYPES = {
    "CannyEdge",
    "JPEGCompression",
    "Saturation",
    "PoissonNoise",
    "ShotNoise",
}


def _ordered_x(x_values, ptype: Optional[str] = None) -> List[float]:
    vals = sorted(set(float(x) for x in x_values))
    if ptype in _REVERSED_LEVEL_TYPES:
        baseline = [v for v in vals if v == _BASELINE_X]
        rest = sorted((v for v in vals if v != _BASELINE_X), reverse=True)
        return baseline + rest
    return vals


def _x_labels(sub_df: pd.DataFrame, x_values: List[float], ptype: Optional[str] = None) -> List[str]:
    mapping: Dict[float, str] = (
        sub_df.drop_duplicates("param_value")
        .set_index("param_value")["perturbation_param"]
        .to_dict()
    )
    
    # try to find a common prefix for the labels (e.g., "patch_ratio_" in "patch_ratio_0.07")
    prefix = ""
    for val, label in mapping.items():
        if val != _BASELINE_X and str(val) in label:
            prefix = label.replace(str(val), "")
            break
            
    labels = []
    for x in x_values:
        if x == _BASELINE_X:
            labels.append("Normale")
        elif x in mapping:
            labels.append(mapping[x])
        else:
            # If the value was forced and is missing in the CSV, reconstruct the name
            labels.append(f"{prefix}{x}")
    return labels


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

    baseline["perturbation_type"]  = ptype
    baseline["perturbation_param"] = _BASELINE_TYPE
    baseline["param_value"]        = _BASELINE_X

    return pd.concat([baseline, perturbed], ignore_index=True)


def _apply_x_axis(ax, x_values: List[float], labels: List[str]) -> None:
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    if _BASELINE_X in x_values:
        baseline_pos = x_values.index(_BASELINE_X)
        ax.axvline(
            x=baseline_pos,
            color="grey", linewidth=0.9, linestyle=":",
            alpha=0.7, label="baseline",
        )


def _plot_trigger_heatmap(fig, inner_gs, img_id: str, ptype: str, param_label: str, category: str, base_results_dir: str, images_dir: str, has_breakpoint: bool) -> None:
    """
    Plots heatmaps only if a breaking point exists. Otherwise, leaves the space completely empty.
    Maintains original image aspect ratio.
    """
    if not has_breakpoint:
        return

    base_path = Path(base_results_dir)
    dir_to_check = base_path / ptype / param_label / str(img_id) / "heatmaps"
    
    if not dir_to_check.exists() and param_label == "Normale":
        dir_to_check = base_path / "Normale" / str(img_id) / "heatmaps"
        
    # Heatmaps are now generated as .jpg; fall back to legacy .png if present.
    heatmap_full = dir_to_check / f"{img_id}_{category}_heatmap.jpg"
    if not heatmap_full.exists():
        heatmap_full = dir_to_check / f"{img_id}_{category}_heatmap.png"
    heatmap_thresh = dir_to_check / f"{img_id}_{category}_thresholded.jpg"
    
    if heatmap_full.exists() or heatmap_thresh.exists():
        ax_full = fig.add_subplot(inner_gs[0, 0])
        ax_thresh = fig.add_subplot(inner_gs[1, 0])
        ax_full.axis("off")
        ax_thresh.axis("off")

        if heatmap_full.exists():
            ax_full.imshow(plt.imread(str(heatmap_full)))
        if heatmap_thresh.exists():
            ax_thresh.imshow(plt.imread(str(heatmap_thresh)))
            
        ax_full.set_title(f"State: {param_label}", fontsize=12, fontweight="bold", color="darkred")
            
    else:
        ax_single = fig.add_subplot(inner_gs[:, 0])
        ax_single.axis("off")

        base_img_path = Path(images_dir)
        if param_label == "Normale":
            raw_img_path_base = base_img_path / f"{img_id}"
        else:
            raw_img_path_base = base_img_path / f"{img_id}_{ptype}_{param_label}"
            
        found_img = None
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".JPEG"]:
            candidate = Path(str(raw_img_path_base) + ext)
            if candidate.exists():
                found_img = candidate
                break
                
        if found_img:
            ax_single.imshow(plt.imread(str(found_img)))
            ax_single.text(0.5, 0.5, f"Token lost\n({param_label})", 
                    transform=ax_single.transAxes,
                    ha='center', va='center', fontsize=11, color='red', fontweight='bold',
                    bbox=dict(facecolor='#ffe6e6', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.5'))
                    
        ax_single.set_title(f"State: {param_label}", fontsize=12, fontweight="bold", color="darkred")


# ---------------------------------------------------------------------------
# Heatmap: images × param_values
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

        x_values = _ordered_x(sub["param_value"].unique(), ptype)

        # Ensure that all expected perturbation parameter values are included in the x_values list, even if they are missing from the data. 
        # This is important for consistent plotting across different perturbation types.
        if ptype in _EXPECTED_PARAMS:
            for expected_val in _EXPECTED_PARAMS[ptype]:
                if expected_val not in x_values:
                    x_values.append(expected_val)
            x_values = _ordered_x(set(x_values), ptype)

        x_labels = _x_labels(sub, x_values, ptype)
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
# Breaking-point detection
# ---------------------------------------------------------------------------

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

    if (not np.isnan(bv_obj) and bv_obj == 0.0) or (not np.isnan(bv_io) and bv_io == 0.0):
        return 0

    for i in range(1, n):
        degs = []
        if not np.isnan(obj_iou_means[i]) and not np.isnan(bv_obj):
            degs.append(_relative_degradation(obj_iou_means[i], bv_obj, "higher_better"))
        if not np.isnan(io_ratio_means[i]) and not np.isnan(bv_io):
            degs.append(_relative_degradation(io_ratio_means[i], bv_io, "higher_better"))

        if not degs:
            run_start, run_len = None, 0
            continue

        primary_deg = float(np.mean(degs))

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
            
            if i == n - 1:
                return run_start    # if we reach the end and are still in a run, consider it a breaking point
        else:
            run_start, run_len = None, 0

    return None


def plot_breaking_point_profile(
    df: pd.DataFrame,
    out_dir: str,
    base_results_dir: str, 
    images_dir: str = "images_perturbed",
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

        x_values = _ordered_x(type_df["param_value"].unique(), ptype)

        # Ensure that all expected perturbation parameter values are included in the x_values list, even if they are missing from the data.
        # This is important for consistent plotting across different perturbation types.
        if ptype in _EXPECTED_PARAMS:
            for expected_val in _EXPECTED_PARAMS[ptype]:
                if expected_val not in x_values:
                    x_values.append(expected_val)
            x_values = _ordered_x(set(x_values), ptype)

        x_labels = _x_labels(type_df, x_values, ptype)
        x_pos    = list(range(len(x_values)))

        for img_id in ids:
            img_df = type_df[type_df["img_id"] == img_id]
            if img_df.empty:
                continue

            categories = sorted(img_df["category"].unique())
            n_cats = len(categories)

            # The grid now has len(METRICS) + 1 columns (1 for the stacked heatmaps + 3 for metric plots)
            n_cols = len(METRICS) + 1
            fig = plt.figure(figsize=(20.0, 6.0 * n_cats), constrained_layout=True)
            
            fig.suptitle(
                f"Per-Token Breaking Point | image {img_id} — {ptype}\n"
                f"(threshold: {degradation_threshold*100:.0f}%, min_consecutive: {min_consecutive})",
                fontsize=14, fontweight="bold",
            )
            
            outer_gs = fig.add_gridspec(n_cats, n_cols, width_ratios=[1.2, 1.0, 1.0, 1.0])

            for row_idx, category in enumerate(categories):
                inner_gs = outer_gs[row_idx, 0].subgridspec(2, 1, hspace=0.02)
                ax_metrics = [fig.add_subplot(outer_gs[row_idx, col + 1]) for col in range(len(METRICS))]
                color = _PALETTE[row_idx % len(_PALETTE)]

                cat_df = img_df[img_df["category"] == category]
                
                cat_series: Dict[str, np.ndarray] = {}
                baseline_vals: Dict[str, float] = {}
                token_missing_mask = {}

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
                            missing_mask.append(True)
                            y.append(0.0 if metric in ["obj_iou", "io_ratio"] else WDP_PENALTY_NO_TOKEN)
                            
                    y = np.array(y, dtype=float)
                    cat_series[metric] = y
                    baseline_vals[metric] = y[0]
                    token_missing_mask[metric] = missing_mask

                # Detect breaking point
                bp_idx = detect_breaking_point(
                    obj_iou_means  = cat_series["obj_iou"],
                    io_ratio_means = cat_series["io_ratio"],
                    wdp_means      = cat_series["wdp"],
                    baseline_vals  = baseline_vals,
                    threshold      = degradation_threshold,
                    wdp_arbiter_min= wdp_arbiter_min,
                    min_consecutive= min_consecutive,
                )

                # plot heatmap only if a breaking point is detected; otherwise, leave the space empty
                has_bp = bp_idx is not None
                bp_label = x_labels[bp_idx] if has_bp else "Normale"
                
                _plot_trigger_heatmap(fig, inner_gs, img_id, ptype, bp_label, category, base_results_dir, images_dir, has_breakpoint=has_bp)

                # plot metric lines and annotations
                for col_idx, metric in enumerate(METRICS):
                    ax = ax_metrics[col_idx]
                    direction = _METRIC_DIRECTION[metric]
                    y = cat_series[metric]
                    missing = token_missing_mask[metric]

                    ax.plot(x_pos, y, marker="o", markersize=6, linewidth=2.0,
                            color=color, label=f"{category}", zorder=3)

                    # values annotation (only for non-missing points)
                    for i, (val, is_missing) in enumerate(zip(y, missing)):
                        if not is_missing and not np.isnan(val):
                            ax.annotate(
                                f"{val:.2f}",
                                xy=(i, val),
                                xytext=(0, 6),
                                textcoords="offset points",
                                ha="center", fontsize=6.5,
                                color=color, fontweight="bold", alpha=0.9,
                            )
                            
                    # lost token annotation (for missing points)
                    for i, is_missing in enumerate(missing):
                        if is_missing:
                            ax.plot(x_pos[i], y[i], marker="X", color="red", markersize=10, zorder=4)
                            if metric == "wdp":
                                ax.annotate("TOKEN LOST", xy=(i, y[i]), xytext=(0, 10),
                                            textcoords="offset points", ha="center", 
                                            fontsize=7, color="red", fontweight="bold")

                    # baseline annotation
                    bv = baseline_vals.get(metric, np.nan)
                    if not np.isnan(bv):
                        ax.axhline(bv, color="grey", linewidth=1.2, linestyle=":", alpha=0.8,
                                   label=f"baseline ({bv:.2f})", zorder=1)

                    # Breaking point annotation
                    if bp_idx is not None:
                        ax.axvline(bp_idx, color="red", linewidth=1.6, linestyle="--", alpha=0.85, zorder=2)
                        
                        if bp_idx == 0:
                            ax.text(0.05, 0.90, f"[{category}]\nBroken in\nBaseline", transform=ax.transAxes,
                                    fontsize=8, color="darkred", fontweight="bold",
                                    bbox=dict(boxstyle="round,pad=0.25", fc="lightpink", ec="darkred", alpha=0.85), zorder=5)
                        else:
                            ax.axvspan(bp_idx, len(x_pos) - 1, color="red", alpha=0.07, zorder=0)
                            bp_val = y[bp_idx]
                            
                            if not missing[bp_idx]:
                                deg_pct = _relative_degradation(bp_val, bv, direction) * 100
                                # Added the token category name into the annotation label
                                label = f"{category}\nΔ = {deg_pct:+.1f} %" if metric != "wdp" else f"{category}\nΔ wdp = {deg_pct:+.1f} %"
                            else:
                                label = f"{category}\nTOKEN LOST"
                                
                            current_lo, current_hi = ax.get_ylim()
                            ax.annotate(
                                label, xy=(bp_idx, bp_val),
                                xytext=(min(bp_idx + 0.4, len(x_pos) - 1.5), current_lo + (current_hi - current_lo) * 0.85),
                                fontsize=8, color="darkred", fontweight="bold",
                                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
                                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="darkred", alpha=0.9), zorder=5,
                            )

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