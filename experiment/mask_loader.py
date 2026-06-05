import logging
import os
from collections import defaultdict
from typing import Dict, List

from models import MaskEntry
from new_eval import load_binary_mask

logger = logging.getLogger(__name__)


def load_all_masks(
    *mask_dirs: str,
) -> Dict[str, List[MaskEntry]]:
    """
    Load every ``*.png`` file found in *mask_dirs* and group them by mask key.
    """
    grouped: Dict[str, List[MaskEntry]] = defaultdict(list)

    for directory in mask_dirs:
        if not os.path.exists(directory):
            logger.warning("Mask directory not found, skipping: %s", directory)
            continue

        for filename in sorted(os.listdir(directory)):
            if not filename.endswith(".png"):
                continue

            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.rsplit("_", 1)

            if len(parts) != 2:
                logger.warning("Filename does not match expected pattern, skipping: %s", filename)
                continue

            key, category = parts
            mask_path = os.path.join(directory, filename)
            binary_mask = load_binary_mask(mask_path)

            grouped[key].append(MaskEntry(category=category, mask=binary_mask))
            logger.debug("[%s] Loaded mask: %s  key=%r  category=%r",
                         os.path.basename(directory), filename, key, category)

    total = sum(len(v) for v in grouped.values())
    logger.info("Loaded %d masks across %d unique keys from %d director(y/ies).",
                total, len(grouped), len(mask_dirs))
    return grouped