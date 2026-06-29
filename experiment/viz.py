import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from new_eval import _pnorm

logger = logging.getLogger(__name__)


def save_thresholded_viz(
    heatmap_array: np.ndarray,
    original_img_path: str,
    save_path: str,
    mask_shape: tuple,
    threshold: float = 0.66,
) -> None:
    """
    Write a three-panel debug image to *save_path*: [ Original | Heatmap overlay | Thresholded mask (red) ]
    """
    h, w = mask_shape[:2]

    orig_bgr = cv2.imread(original_img_path)
    if orig_bgr is None:
        logger.warning("Could not read source image for visualisation: %s", original_img_path)
        return
    orig_bgr = cv2.resize(orig_bgr, (w, h))

    # Normalise and convert to 8-bit
    heatmap_resized = cv2.resize(heatmap_array, (w, h))
    heatmap_norm = _pnorm(heatmap_resized)
    heatmap_8u = (heatmap_norm * 255).astype(np.uint8)

    # Heatmap overlay panel
    colormap = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
    heatmap_panel = cv2.addWeighted(colormap, 0.6, orig_bgr, 0.4, 0)

    # Thresholded mask panel (active pixels highlighted in red)
    threshold_value = int(255 * threshold)
    _, binary_pred = cv2.threshold(heatmap_8u, threshold_value, 255, cv2.THRESH_BINARY)

    threshold_panel = orig_bgr.copy()
    threshold_panel[binary_pred == 255] = [0, 0, 255]
    threshold_panel = cv2.addWeighted(threshold_panel, 0.6, orig_bgr, 0.4, 0)
    cv2.putText(
        threshold_panel,
        f"Threshold: {threshold_value} ({threshold * 100:.0f}%)",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    final_viz = np.hstack((orig_bgr, heatmap_panel, threshold_panel))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final_viz)
    logger.debug("Thresholded viz saved: %s", save_path)


def make_layer_grid(
    layer_images: List[Tuple[int, Optional[np.ndarray]]],
    token_label: str,
    out_path: str,
    cols: int = 8,
    pad: int = 8,
    label_h: int = 22,
    bg: Tuple[int, int, int] = (0, 0, 0),
    fg: Tuple[int, int, int] = (255, 255, 255),
    quality: int = 85,
    subsampling: int = 2,
) -> None:
    """
    Build a PIL grid from a list of ``(layer_index, bgr_image)`` pairs and
    save it to *out_path*.  All compositing is done in RAM — no temporary
    files are written.

    The output format follows *out_path*'s extension. For ``.jpg``/``.jpeg``
    the grid is written directly as JPEG (quality 85, 4:2:0 by default) — much
    smaller than PNG and visually near-identical for these photo-like
    heatmaps — so no separate PNG-to-JPG conversion step is needed.
    """
    # Convert BGR -> RGB and wrap in PIL
    tiles: List[Tuple[int, Optional[Image.Image]]] = []
    for layer_idx, bgr in layer_images:
        pil_img = None
        if bgr is not None:
            pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        tiles.append((layer_idx, pil_img))

    valid_images = [img for _, img in tiles if img is not None]
    if not valid_images:
        logger.warning("No valid layer images to build grid for token %r.", token_label)
        return

    tw = max(img.size[0] for img in valid_images)
    th = max(img.size[1] for img in valid_images)

    resized: List[Tuple[int, Optional[Image.Image]]] = [
        (li, img.resize((tw, th), Image.BILINEAR) if img and img.size != (tw, th) else img)
        for li, img in tiles
    ]

    n = len(resized)
    rows = math.ceil(n / cols)
    canvas_w = cols * tw + (cols + 1) * pad
    canvas_h = rows * (th + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (li, img) in enumerate(resized):
        row, col = divmod(i, cols)
        x0 = pad + col * (tw + pad)
        y0 = pad + row * (th + label_h + pad)

        # Layer label background + text
        draw.rectangle([x0, y0, x0 + tw, y0 + label_h], fill=bg)
        draw.text((x0 + 4, y0 + 4), f"L{li}", fill=fg, font=font)

        if img:
            canvas.paste(img, (x0, y0 + label_h))
        else:
            draw.rectangle([x0, y0 + label_h, x0 + tw, y0 + label_h + th], outline=fg)

    # Header strip
    header_h = 30
    out_img = Image.new("RGB", (canvas_w, canvas_h + header_h), bg)
    out_img.paste(canvas, (0, header_h))
    draw2 = ImageDraw.Draw(out_img)
    draw2.text((pad, 7), f"token: {token_label[:70]}", fill=fg, font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        # out_img is already RGB, so JPEG (no alpha) is safe to write directly.
        out_img.save(str(out_path), "JPEG", quality=quality, optimize=True,
                     subsampling=subsampling)
    else:
        out_img.save(str(out_path))
    logger.debug("Layer grid saved: %s", out_path)