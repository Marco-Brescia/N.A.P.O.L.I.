from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MaskEntry:
    category: str
    mask: np.ndarray           # H x W, dtype uint8, values in {0, 255}


@dataclass
class ImageTask:
    image_id: str              # e.g. "img01"
    mask_key: str              # key into the grouped-masks dict
    image_path: str            # absolute or relative path to the image file
    output_folder: str         # sub-folder inside OUT_DIR  (e.g. "Jigsaw/4")
    perturbation_type: str     # e.g. "Jigsaw", "" for baseline
    perturbation_param: str    # e.g. "4", "Normale"
    image_array: np.ndarray    # BGR image already loaded with cv2 (avoids re-reads)