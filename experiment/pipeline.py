import csv
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from config import Config
from heatmap import build_global_heatmap, find_target_tokens
from models import ImageTask, MaskEntry
from new_eval import compute_all_metrics, run_inference
from viz import make_layer_grid, save_thresholded_viz

logger = logging.getLogger(__name__)

_CSV_FIELDS = ["img_id", "prompt", "output", "target_token_id",
               "readable_token", "obj_iou", "io_ratio", "wdp"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_tokens(raw_labels: List[str]) -> List[str]:
    return [
        t.replace("Ġ", "").replace("\u00a0", "").strip().lower()
        for t in raw_labels
    ]


def _save_tokens(token_labels: List[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(token_labels))


def _write_csv(records: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    logger.info("CSV saved: %s", path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_image(
    task: ImageTask,
    masks: List[MaskEntry],
    model,
    processor,
    model_type: str,
    cfg: Config,
    inference_lock: threading.Lock
) -> None:
    """
    Run the full analysis pipeline for one image.
    """
    image_out_dir = os.path.join(cfg.output_dir, task.output_folder, task.image_id)
    heatmaps_dir = os.path.join(image_out_dir, cfg.heatmaps_subdir)
    os.makedirs(heatmaps_dir, exist_ok=True)

    with inference_lock:
    # Inference
        ctx = run_inference(model, processor, task.image_path, cfg.prompt, model_type)

    # Attach the original BGR array to context so heatmap.py can build overlays
    ctx["orig_bgr"] = task.image_array
    h, w = task.image_array.shape[:2]
    ctx["orig_shape"] = (h, w)

    clean_tokens = _clean_tokens(ctx["token_labels"])

    # Persist token labels (checkpoint for whether we can skip this image on future runs)
    tokens_path = os.path.join(image_out_dir, f"{task.image_id}_tokens.txt")
    _save_tokens(ctx["token_labels"], tokens_path)
    logger.debug("Token labels saved: %s", tokens_path)

    csv_records: List[dict] = []

    for entry in masks:
        logger.info("[%s] Evaluating category: %r", task.image_id, entry.category)

        token_ids = find_target_tokens(clean_tokens, entry.category)

        if not token_ids:
            logger.warning(
                "[%s] Token %r not found in generated output — skipping mask.",
                task.image_id, entry.category,
            )
            continue

        logger.debug("[%s] Token %r matched at position(s) %s.", task.image_id, entry.category, token_ids)

        # Heatmap generation
        with inference_lock:
            global_heatmap, layer_images = build_global_heatmap(
                inference_context=ctx,
                token_ids=token_ids,
                processor=processor,
                model=model,
                binary_mask=entry.mask,
                layer_relevance_threshold=cfg.layer_relevance_threshold,
            )

        # Layer grid 
        grid_path = os.path.join(heatmaps_dir, f"{task.image_id}_{entry.category}_heatmap.jpg")
        make_layer_grid(
            layer_images=layer_images,
            token_label=entry.category,
            out_path=grid_path,
        )
        del layer_images  # Free memory eagerly

        # Thresholded visualisation
        threshold_path = os.path.join(
            heatmaps_dir, f"{task.image_id}_{entry.category}_thresholded.jpg"
        )
        save_thresholded_viz(
            heatmap_array=global_heatmap,
            original_img_path=task.image_path,
            save_path=threshold_path,
            mask_shape=entry.mask.shape,
            threshold=cfg.viz_threshold,
        )

        # Metrics
        metrics = compute_all_metrics(global_heatmap, entry.mask)
        csv_token_id = token_ids[0] if len(token_ids) == 1 else str(token_ids)

        csv_records.append({
            "img_id": task.image_id,
            "prompt": cfg.prompt.replace("\n", " ").strip(),
            "output": ctx["gen_text"].replace("\n", " ").strip(),
            "target_token_id": csv_token_id,
            "readable_token": entry.category,
            "obj_iou": round(metrics["obj_iou"], 4),
            "io_ratio": round(metrics["io_ratio"], 4),
            "wdp": round(metrics["wdp"], 4),
        })

    if csv_records:
        _write_csv(csv_records, os.path.join(image_out_dir, "results.csv"))
    else:
        logger.warning("[%s] No results produced (no token matches).", task.image_id)