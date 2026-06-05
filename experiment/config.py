from dataclasses import dataclass, field
from typing import List


# Perturbations that spatially displace objects and therefore require dedicated (per-image)
STRONG_PERTURBATIONS: List[str] = ["Jigsaw", "Rotate", "Scale", "Shear", "Cutout"]


@dataclass(frozen=True)
class Config:
    # Directories
    images_dir: str = "images_perturbed"
    masks_dir: str = "masks"
    perturbed_masks_dir: str = "masks_perturbed"
    output_dir: str = "output"
    heatmaps_subdir: str = "heatmaps"

    # Inference
    prompt: str = "Describe this image in detail, use simple terms"
    max_workers: int = 4

    # Thresholds
    layer_relevance_threshold: float = 0.2
    viz_threshold: float = 0.66

    # Retry
    max_retries: int = 10
    retry_sleep_seconds: float = 1.0