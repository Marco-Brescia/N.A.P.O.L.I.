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
    use_otsu: bool = False,
) -> None:
    """ Generate a set of 3 images: original, heatmap overlay, and thresholded overlay. Save them side-by-side. """
    h, w = mask_shape[:2]

    orig_bgr = cv2.imread(original_img_path)
    if orig_bgr is None:
        logger.warning("Impossibile leggere l'immagine sorgente: %s", original_img_path)
        return
    orig_bgr = cv2.resize(orig_bgr, (w, h))

    # resize heatmap to match original image dimensions
    hm_resized = cv2.resize(heatmap_array, (w, h))

    # Thresholding
    if use_otsu:
        hm_final = hm_resized
        t_val, binary_pred = cv2.threshold(hm_final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        label_text = f"Otsu Soglia: {int(t_val)}"
    else:
        hm_norm = _pnorm(hm_resized)
        hm_final = (hm_norm * 255).astype(np.uint8)
        t_val = int(255 * threshold)
        _, binary_pred = cv2.threshold(hm_final, t_val, 255, cv2.THRESH_BINARY)
        label_text = f"Soglia Fissa: {t_val} ({threshold * 100:.0f}%)"

    # Safety check 
    if binary_pred.sum() == 0:
        return

    # Heatmap overlay
    heatmap_color = cv2.applyColorMap(hm_final, cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(heatmap_color, 0.6, orig_bgr, 0.4, 0)

    # Threshold overlay
    overlay = orig_bgr.copy()
    overlay[binary_pred == 255] = [0, 0, 255]
    overlay = cv2.addWeighted(overlay, 0.6, orig_bgr, 0.4, 0)
    
    cv2.putText(
        overlay, label_text, (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
    )

    debug_vis = np.hstack((orig_bgr, heatmap_overlay, overlay))
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), debug_vis)
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