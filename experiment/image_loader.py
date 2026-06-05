import ast
import logging
import os
from typing import Dict, List

import cv2

from config import STRONG_PERTURBATIONS, Config
from models import ImageTask, MaskEntry

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".png"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_filename(base_name: str) -> tuple[str, str, str]:
    """
    Split a filename (no extension) into (img_id, perturbation_type, param).

    Examples
    --------
    "img01"              -> ("img01", "",       "Normale")
    "img01_Jigsaw_4"     -> ("img01", "Jigsaw", "4")
    "img01_Brightness_50"-> ("img01", "Brightness", "50")
    """
    if "_" not in base_name:
        return base_name, "", "Normale"

    img_id, suffix = base_name.split("_", 1)

    if "_" in suffix:
        perturbation_type, param = suffix.split("_", 1)
    else:
        perturbation_type, param = suffix, "Normale"

    return img_id, perturbation_type, param


def _build_mask_key(img_id: str, base_name: str, perturbation_type: str) -> str:
    if perturbation_type in STRONG_PERTURBATIONS:
        return base_name
    return img_id


def _build_output_folder(perturbation_type: str, param: str) -> str:
    if perturbation_type and perturbation_type != param:
        return f"{perturbation_type}/{param}"
    return param


def _is_completed(
    cfg: Config,
    task_folder: str,
    img_id: str,
    mask_entries: List[MaskEntry],
) -> bool:
    """
    Return True when ALL expected heatmap files and the tokens file are
    already present on disk for a given image.
    """
    checkpoint_dir = os.path.join(cfg.output_dir, task_folder, img_id)
    tokens_file = os.path.join(checkpoint_dir, f"{img_id}_tokens.txt")
    heatmaps_dir = os.path.join(checkpoint_dir, cfg.heatmaps_subdir)

    if not (os.path.isdir(checkpoint_dir) and os.path.exists(tokens_file)):
        return False

    try:
        with open(tokens_file, "r", encoding="utf-8") as fh:
            raw = ast.literal_eval(fh.read())
        generated_tokens = {
            t.replace("Ġ", "").replace("\u00a0", "").strip().lower() for t in raw
        }
    except Exception:
        logger.debug("Corrupt tokens file for %s/%s — will reprocess.", task_folder, img_id)
        return False

    for entry in mask_entries:
        target = entry.category.lower()
        if target not in generated_tokens:
            # Model did not generate this token last time; must reprocess.
            return False
        if not os.path.isdir(heatmaps_dir):
            return False
        existing = os.listdir(heatmaps_dir)
        if not any(f"_{target}_heatmap" in fname.lower() for fname in existing):
            return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_task_queue(
    cfg: Config,
    grouped_masks: Dict[str, List[MaskEntry]],
) -> List[ImageTask]:
    """
    Scan *cfg.images_dir*, apply checkpoint filtering, and return the list of
    :class:`ImageTask` objects that still need to be processed.
    """
    tasks: List[ImageTask] = []
    skipped = 0

    for filename in sorted(os.listdir(cfg.images_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            continue

        base_name = os.path.splitext(filename)[0]
        img_id, perturbation_type, param = _parse_filename(base_name)
        mask_key = _build_mask_key(img_id, base_name, perturbation_type)
        output_folder = _build_output_folder(perturbation_type, param)

        mask_entries = grouped_masks.get(mask_key, [])
        if not mask_entries:
            logger.warning(
                "No masks found for key %r (file: %s) — image will be skipped.",
                mask_key, filename,
            )
            continue

        if _is_completed(cfg, output_folder, img_id, mask_entries):
            logger.info("Skip (already completed): %s / %s", output_folder, img_id)
            skipped += 1
            continue

        image_path = os.path.join(cfg.images_dir, filename)
        image_array = cv2.imread(image_path)
        if image_array is None:
            logger.error("Could not read image: %s — skipping.", image_path)
            continue

        tasks.append(
            ImageTask(
                image_id=img_id,
                mask_key=mask_key,
                image_path=image_path,
                output_folder=output_folder,
                perturbation_type=perturbation_type,
                perturbation_param=param,
                image_array=image_array,
            )
        )

    logger.info(
        "Task queue built: %d to process, %d skipped (checkpoint).",
        len(tasks), skipped,
    )
    return tasks