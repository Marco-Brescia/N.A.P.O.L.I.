import logging
from typing import List

import numpy as np

from new_eval import _build_logitlens_logits, _pnorm, compute_all_metrics
from tam import TAM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token matching
# ---------------------------------------------------------------------------

def find_target_tokens(generated_tokens_lower: List[str], target_category: str) -> List[int]:
    """
    Identify the token ID(s) corresponding to the target category.
    1. **Exact match** — a single token whose cleaned form equals the target.
    2. **Subword match** — a contiguous span of tokens that, when concatenated, equals the target (handles tokenisers that split words into pieces).
    """
    _clean = lambda s: s.replace(" ", "").replace("-", "").replace("_", "").lower()

    target = _clean(target_category)
    tokens = [_clean(t) for t in generated_tokens_lower]

    # exact single-token match
    if target in tokens:
        return [tokens.index(target)]

    # contiguous subword match 
    for start in range(len(tokens)):
        accumulated = ""
        for end in range(start, len(tokens)):
            accumulated += tokens[end]

            if accumulated == target:
                return list(range(start, end + 1))

            # Early exit: accumulated string already longer or diverged
            if len(accumulated) > len(target) or not target.startswith(accumulated):
                break

    return []


# ---------------------------------------------------------------------------
# Per-layer heatmap generation
# ---------------------------------------------------------------------------

def build_global_heatmap(
    inference_context: dict,
    token_ids: List[int],
    processor,
    model,
    binary_mask: np.ndarray,
    layer_relevance_threshold: float,
) -> tuple[np.ndarray, List[tuple]]:
    """
    Generate one heatmap per transformer layer, accumulate the relevant ones
    into a single global heatmap, and return both the global map and the
    per-layer overlay images (for the grid visualisation).
    """
    vision_shape = inference_context["vision_shape"]
    total_layers = inference_context["n_layers"]
    orig_h, orig_w = inference_context.get("orig_shape", (binary_mask.shape[0], binary_mask.shape[1]))

    global_accumulator = np.zeros(vision_shape, dtype=np.float32)
    layer_images = []

    layers_to_process = list(range(total_layers))
    if (total_layers - 1) not in layers_to_process:
        layers_to_process.append(total_layers - 1)

    for layer_idx in layers_to_process:
        layer_logits = _build_logitlens_logits(
            inference_context["outputs"], model, layer_idx, total_layers
        )
        layer_accumulator = np.zeros(vision_shape, dtype=np.float32)

        # Aggregate heatmaps across all subword tokens for this layer
        for token_id in token_ids:
            token_heatmap = TAM(
                inference_context["generated_ids"][0].tolist(),
                vision_shape,
                layer_logits,
                inference_context["special_ids"],
                inference_context["vis_inputs"],
                processor,
                "",        # Empty path: disables TAM's own disk I/O
                token_id,
                [],
                False,
            )
            layer_accumulator += token_heatmap

        # Build visual overlay for the grid
        layer_norm = _pnorm(layer_accumulator)
        layer_8u = (layer_norm * 255).astype(np.uint8)
        upscaled = cv2.resize(layer_8u, (orig_w, orig_h)) 
        colormap = cv2.applyColorMap(upscaled, cv2.COLORMAP_JET) 

        # We need the original BGR image for the overlay
        orig_bgr = inference_context.get("orig_bgr")
        if orig_bgr is not None:
            overlay = cv2.addWeighted(colormap, 0.6, orig_bgr, 0.4, 0) 
        else:
            overlay = colormap

        layer_images.append((layer_idx, overlay))

        # Conditionally accumulate into global map
        layer_metrics = compute_all_metrics(layer_accumulator, binary_mask)
        if round(layer_metrics["io_ratio"], 4) > layer_relevance_threshold:
            global_accumulator += layer_accumulator

    # Produce final normalised 8-bit global heatmap
    global_heatmap_8u = (_pnorm(global_accumulator) * 255.0).astype(np.uint8)
    return global_heatmap_8u, layer_images

import cv2