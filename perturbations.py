import os
import cv2
from new_eval import load_binary_mask
import numpy as np

IMAGES_DIR = "test_phase0"
MASKS_DIR  = "masks_experiment"
OUT_DIR    = "images_perturbed"

os.makedirs(OUT_DIR, exist_ok=True)

# Configure base step and range for each perturbation type
perturbation_ranges = {
    "MedianFilter":        {"param": "ksize",    "min": 3,    "max": 15,   "step": 2,    "type": int},
    "GaussianNoise":       {"param": "std",      "min": 5,    "max": 85,   "step": 20.0,   "type": float},
    "MultiplicativeNoise": {"param": "variance", "min": 0.05, "max": 0.35, "step": 0.100, "type": float},
    "GaussianLowPass":     {"param": "sigma",    "min": 0.5,  "max": 5.0,  "step": 1.50,  "type": float},
    "ZoomBlur":            {"param": "zoom_max", "min": 1.05, "max": 1.55, "step": 0.10, "type": float},
    "ISONoise":            {"param": "intensity","min": 0.5,  "max": 4.5,  "step": 1.0,  "type": float},
    "Brightness":          {"param": "factor",   "min": 0.5,  "max": 2.0,  "step": 0.50,  "type": float},
    "Contrast":            {"param": "factor",   "min": 0.7,  "max": 2.2,  "step": 0.50,  "type": float},
    "Fog":                 {"param": "intensity", "min": 0.2, "max": 1.4,  "step": 0.30,  "type": float},
    "JPEGCompression":     {"param": "quality",  "min": 90,   "max": 30,   "step": -30.0,  "type": float},
    "Rotate":              {"param": "angle",    "min": 36,   "max": 324,   "step": 36,   "type": int},
    "Cutout":              {"param": "ratio",    "min": 0.1,  "max": 1,  "step": 0.1,  "type": float},
    "CannyEdge":           {"param": "th",       "min": 100,  "max": 250,  "step": 75,   "type": int},
    "ImpulseNoise":        {"param": "ratio",      "min": 0.01, "max": 0.15, "step":  0.02,  "type": float},
    "ShotNoise":           {"param": "scale",      "min": 20,   "max": 100,  "step": 20.0,   "type": float},
    "PoissonNoise":        {"param": "scale",      "min": 10,   "max": 80,   "step": 17.5,   "type": float},
    "DefocusBlur":         {"param": "radius",     "min": 1,    "max": 9,    "step":  2,      "type": int},
    "MotionBlur":          {"param": "ksize",      "min": 5,    "max": 25,   "step":  5,      "type": int},
    "Saturation":          {"param": "factor",     "min": 0.0,  "max": 2.0,  "step":  0.50,  "type": float},
    "GammaContrast":       {"param": "gamma",      "min": 0.4,  "max": 2.0,  "step":  0.40,  "type": float},
    "Snow":                {"param": "intensity",  "min": 0.1,  "max": 1.0,  "step":  0.10,  "type": float},
    "Frost":               {"param": "intensity",  "min": 0.1,  "max": 1.0,  "step":  0.10,  "type": float},
    "Splatter":            {"param": "num_drops",  "min": 20,   "max": 200,  "step": 45.0,   "type": int},
    "Pixelate":            {"param": "block_size", "min": 2,    "max": 20,   "step":  3,      "type": int},
    "Scale":               {"param": "factor",     "min": 0.5,  "max": 1.5,  "step":  0.25,  "type": float},
    "Shear":               {"param": "angle",      "min": -30,  "max": 30,   "step": 15.0,   "type": float},
    "Jigsaw":              {"param": "grid_size",  "min": 2,    "max": 8,    "step":  2,      "type": int},
    "CoarseDropout":       {"param": "hole_size",  "min": 8,    "max": 40,   "step":  8,      "type": int},
    "AdversarialNoise":    {"param": "eps",        "min": 2,    "max": 32,   "step":  7.5,   "type": float},
    "AdversarialPatch":    {"param": "patch_ratio","min": 0.05, "max": 0.50, "step":  0.075, "type": float},
}

# Perturbation functions

def apply_median_filter(img, ksize=9):
    k = int(ksize)
    if k % 2 == 0: k += 1
    return cv2.medianBlur(img, max(1, k))

def apply_gaussian_noise(img, std=50):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_multiplicative_noise(img, variance=0.20):
    std = np.sqrt(variance)
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) * (1 + noise), 0, 255).astype(np.uint8)

def apply_gaussian_low_pass(img, sigma=2):
    k = int(4 * sigma + 1)
    if k % 2 == 0: k += 1
    return cv2.GaussianBlur(img, (k, k), sigma)

def apply_zoom_blur(img, zoom_max=1.10, steps=5):
    h, w = img.shape[:2]
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(steps):
        zoom = 1.0 + (zoom_max - 1.0) * (i / max(1, steps - 1))
        nh, nw = int(h * zoom), int(w * zoom)
        resized = cv2.resize(img, (nw, nh))
        y1, x1 = (nh - h) // 2, (nw - w) // 2
        result += cv2.resize(resized[y1:y1+h, x1:x1+w], (w, h)).astype(np.float32)
    return (result / steps).astype(np.uint8)

def apply_iso_noise(img, intensity=1.5):
    img_f = img.astype(np.float32) / 255.0
    p_scale = 100.0 / max(intensity, 0.01)
    noisy = np.random.poisson(img_f * p_scale) / p_scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)

def apply_brightness(img, factor):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def apply_contrast(img, factor):
    mean = np.mean(img)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

def apply_fog(img, intensity):
    fog_layer = np.full(img.shape, 255, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - intensity, fog_layer, intensity, 0)

def apply_jpeg(img, quality):
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(enc, 1)

def apply_rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def apply_cutout(img, ratio, mask_list=None):
    h, w = img.shape[:2]
    res = img.copy()
    
    if mask_list and len(mask_list) > 0:
        for cat, m_data in mask_list:
            y_coords, x_coords = np.where(m_data > 0)
            mask_area = len(y_coords)
            
            if mask_area > 0:
                s = int(np.sqrt(mask_area * ratio))
                s = max(1, s)
                
                cy_exact = int(np.mean(y_coords))
                cx_exact = int(np.mean(x_coords))
                
                obj_h = np.max(y_coords) - np.min(y_coords)
                obj_w = np.max(x_coords) - np.min(x_coords)
                
                # limit to 15% of object size to avoid excessive shift
                limit_y = max(1, int(obj_h * 0.15))
                limit_x = max(1, int(obj_w * 0.15))
                
                # apply random offset to center of cutout within limits
                offset_y = np.random.randint(-limit_y, limit_y + 1)
                offset_x = np.random.randint(-limit_x, limit_x + 1)
                
                cy = cy_exact + offset_y
                cx = cx_exact + offset_x

                y = max(0, min(cy - s // 2, h - s))
                x = max(0, min(cx - s // 2, w - s))
                
                res[y:y+s, x:x+s] = 0
    else:
        s = max(1, int(np.sqrt((h * w) * ratio)))
        limit_y = max(1, int(h * 0.10))
        limit_x = max(1, int(w * 0.10))
        
        cy = (h // 2) + np.random.randint(-limit_y, limit_y + 1)
        cx = (w // 2) + np.random.randint(-limit_x, limit_x + 1)
        
        y = max(0, min(cy - (s // 2), h - s))
        x = max(0, min(cx - (s // 2), w - s))
        res[y:y+s, x:x+s] = 0
        
    return res

def apply_canny(img, th):
    """
    Canny edge detection: extracts edges and blends them with the original image.
    th: upper threshold for Canny (lower threshold is set to th/2).
    """
    edges = cv2.Canny(img, th/2, th)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(img, 1.0, edges_bgr, 0.5, 0)

def apply_impulse_noise(img, ratio=0.05):
    """
    Impulse noise (salt-and-pepper).
    ratio: fraction of pixels to corrupt (e.g., 0.05 = 5%).
    """
    out = img.copy()
    h, w = out.shape[:2]
    n_pixels = int(h * w * ratio)
 
    # Withe pixel (salt), coordinates random
    ys = np.random.randint(0, h, n_pixels)
    xs = np.random.randint(0, w, n_pixels)
    out[ys, xs] = 255
 
    # Black pixel (pepper), coordinates random
    ys = np.random.randint(0, h, n_pixels)
    xs = np.random.randint(0, w, n_pixels)
    out[ys, xs] = 0
 
    return out
 
 
def apply_shot_noise(img, scale=50.0):
    """
    Shot noise (Poisson photography model): simulates photn shot noise.
    scale: scaling factor for the Poisson distribution.
    """
    img_f = img.astype(np.float32) / 255.0
    img_f = np.clip(img_f, 1e-6, 1.0)
    noisy = np.random.poisson(img_f * scale).astype(np.float32) / scale
    return np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
 
 
def apply_poisson_noise(img, scale=40.0):
    img_f = img.astype(np.float32)
    noisy = np.random.poisson(img_f / 255.0 * scale) / scale * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)
 
def apply_defocus_blur(img, radius=3):
    """
    Defocus blur: simulates out-of-focus blur by convolving with a disk-shaped kernel.
    radius: disk radius in pixels (es. 3 → diameter 7x7).
    """
    r = max(1, int(radius))
    k_size = 2 * r + 1
    kernel = np.zeros((k_size, k_size), dtype=np.float32)
    cv2.circle(kernel, (r, r), r, 1.0, -1)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)
 
 
def apply_motion_blur(img, ksize=15, angle=0.0):
    """
    Motion blur: simulates camera or object motion by convolving with a linear kernel.
    ksize: kernel size (length of the motion blur in pixels).
    angle: direction of motion in degrees (0 = horizontal, 90 = vertical, etc.).
    """
    k = max(3, int(ksize))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k 
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return cv2.filter2D(img, -1, kernel)

def apply_saturation(img, factor=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
 
 
def apply_gamma_contrast(img, gamma=1.0):
    """
    gamma contrast: compression or expansion of the tonal curve.
    gamma=1.0 → inaltered , gamma=0.4 → more clear, gamma=2.0 → darker.
    """
    inv_gamma = 1.0 / max(gamma, 1e-6)
    # Look-up table (efficiency)
    lut = np.array([
        np.clip((i / 255.0) ** inv_gamma * 255.0, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(img, lut)

def apply_snow(img, intensity=0.3):
    """
    intensity: [0,1] density and opacity of the snow effect
    """
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
    n_flakes = int(h * w * intensity * 0.02)
 
    for _ in range(n_flakes):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(1, max(2, int(4 * intensity)))
        alpha = np.random.uniform(0.5, 1.0) * intensity * 2
        alpha = min(alpha, 1.0)
 
        # Draw a simple circular "flake" and blend it with the image
        flake = np.zeros_like(out)
        cv2.circle(flake, (x, y), radius, (255, 255, 255), -1)
        out = cv2.addWeighted(out, 1.0, flake, alpha, 0)
 
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
def apply_frost(img, intensity=0.3):
    """
    intensity: [0,1] opacity of the frost layer blended with the original image.
    """
    h, w = img.shape[:2]
 
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    for sigma in [2, 4, 8, 16, 32]:
        layer = np.random.normal(0, 1, (h, w)).astype(np.float32)
        layer = cv2.GaussianBlur(layer, (0, 0), sigma)
        noise += amplitude * layer
        amplitude *= 0.5
 
    # Normalize in [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
 
    frost_layer = np.stack([
        noise * 220,   # B
        noise * 235,   # G
        noise * 255,   # R
    ], axis=-1).astype(np.float32)
 
    out = cv2.addWeighted(img.astype(np.float32), 1.0 - intensity,
                          frost_layer, intensity, 0)
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
def apply_splatter(img, num_drops=80):
    """
    num_drops: number of random "splatter drops" to add.
    """
    out = img.copy().astype(np.float32)
    h, w = out.shape[:2]
 
    overlay = np.zeros_like(out)
 
    for _ in range(int(num_drops)):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        rx = np.random.randint(1, 3)
        ry = np.random.randint(4, max(5, int(h * 0.02)))
        angle = np.random.uniform(-15, 15)
        pixel_color = img[
            min(y, h - 1), min(x, w - 1)
        ].astype(np.float32)
        drop_color = np.clip(pixel_color * 1.3 + 30, 0, 255).tolist()
        cv2.ellipse(overlay, (x, y), (rx, ry), angle, 0, 360,
                    drop_color, -1)
 
    alpha = 0.55
    out = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)
    return np.clip(out, 0, 255).astype(np.uint8)
 
def apply_pixelate(img, block_size=8):
    """
    Pixelate: reduce the image resolution by resizing down and then up with nearest neighbor.
    block_size: size of the pixelation blocks in pixels (e.g. 8 → 8x8 blocks).
    """
    h, w = img.shape[:2]
    b = max(2, int(block_size))
    small = cv2.resize(img, (max(1, w // b), max(1, h // b)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_scale(img, factor=1.0):
    """
    scale of the image by a factor, then crop or pad to original size.
    factor < 1 → zoom out ( black borders),
    factor > 1 → zoom in (central crop).
    """
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
 
    if factor < 1.0:
        # Center pad with black borders
        out = np.zeros_like(img)
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        out[y0:y0 + new_h, x0:x0 + new_w] = resized
    else:
        # Crop center region of the zoomed image
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        out = resized[y0:y0 + h, x0:x0 + w]
 
    return out
 
 
def apply_shear(img, angle=10.0):
    """
    Apply a shear transformation to the image.
    angle: sheer angle in degrees (positive = rightward, negative = leftward).
    """
    h, w = img.shape[:2]
    shear = np.tan(np.deg2rad(angle))
    M = np.float32([
        [1, shear, -shear * h / 2],
        [0, 1,     0             ]
    ])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
 
 
def apply_jigsaw(img, grid_size=4):
    """
    Jigsaw puzzle: split the image into a grid and shuffle the patches.
    grid_size: number of cells per dimension (e.g. 4 → 4x4 grid = 16 patches).
    // jigsaw da tenere  per rigenerare la maschera del target.
    """
    h, w = img.shape[:2]
    n = max(2, int(grid_size))
    out = img.copy()
 
    # Single cell size (last row/col may be larger if h,w are not divisible by n)
    cell_h = h // n
    cell_w = w // n
 
    if cell_h < 1 or cell_w < 1:
        return out  # grid too fine for image size, return original
 
    # build list of patches and their coordinates (used for shuffling and reassembling)
    patches = []
    coords = []
    for row in range(n):
        for col in range(n):
            y1 = row * cell_h
            x1 = col * cell_w
            y2 = y1 + cell_h if row < n - 1 else h
            x2 = x1 + cell_w if col < n - 1 else w
            patches.append(img[y1:y2, x1:x2].copy())
            coords.append((y1, x1, y2, x2))
 
    # index shuffle
    indices = list(range(len(patches)))
    np.random.shuffle(indices)
 
    # Replace each cell with the shuffled patch, resizing if necessary to fit the destination cell.
    # Border cells may have different sizes, so we resize the patch to fit the destination cell.
    for dst_idx, src_idx in enumerate(indices):
        y1, x1, y2, x2 = coords[dst_idx]
        dst_h = y2 - y1
        dst_w = x2 - x1
        src_patch = patches[src_idx]
        if src_patch.shape[:2] != (dst_h, dst_w):
            src_patch = cv2.resize(src_patch, (dst_w, dst_h),
                                   interpolation=cv2.INTER_LINEAR)
        out[y1:y2, x1:x2] = src_patch
 
    return out

def apply_coarse_dropout(img, n_holes=20, hole_size=16):
    """
    Coarse Dropout: remove random black rectangles (cutout) from the image.
    n_holes: number of rectangles to drop.
    hole_size: size of each square hole in pixels (side length).
    """
    out = img.copy()
    h, w = out.shape[:2]
    s = max(1, int(hole_size))
 
    for _ in range(int(n_holes)):
        x = np.random.randint(0, max(1, w - s))
        y = np.random.randint(0, max(1, h - s))
        out[y:y + s, x:x + s] = 0
 
    return out
 
 
def apply_adversarial_noise(img, eps=8.0):
    """
    Adversarial noise sintetico (FGSM-inspired, senza rete neurale).
    Simula il gradiente del segnale usando il gradiente dell'immagine stessa
    (derivate di Sobel) come proxy del gradiente della loss, aggiungendo
    perturbazione nella direzione del segno del gradiente.
 
    eps: ampiezza massima della perturbazione in [0,255].
 
    Nota: questo NON è un attacco adversariale vero (richiederebbe un modello).
    È un proxy sintetico riproducibile che produce pattern strutturati
    ad alta frequenza analoghi a quelli FGSM, utile per valutare la
    robustezza percettiva senza dipendenze da PyTorch/TensorFlow.
    """
    eps = float(eps)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
 
    # Gradiente Sobel as a proxy of the loss gradient
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = gx + gy  # sum (not magnitude) to preserve sign and create structured patterns
 
    # Perturbation: eps * sign(grad), broadcast on BRG channels
    sign_grad = np.sign(grad).astype(np.float32)
    perturbation = eps * sign_grad
    perturbation_bgr = np.stack([perturbation] * 3, axis=-1)
 
    out = img.astype(np.float32) + perturbation_bgr
    return np.clip(out, 0, 255).astype(np.uint8)
 
 
def apply_adversarial_patch(img, patch_ratio=0.15):
    """
    Adversarial patch sintetica: sovrappone una texture strutturata ad alta
    frequenza (scacchiera con rumore) in una zona prominente dell'immagine.
 
    patch_ratio: frazione dell'area immagine occupata dalla patch [0,1].
 
    La patch è una scacchiera ε-perturbata che massimizza la confusione
    percettiva senza richiedere un modello: simula l'aspetto visivo delle
    patch adversariali reali (pattern ad alta frequenza, elevato contrasto locale).
    """
    h, w = img.shape[:2]
    out = img.copy()
 
    # Square patch size
    patch_area = int(h * w * patch_ratio)
    ps = max(8, int(np.sqrt(patch_area)))
    ps = min(ps, h, w)  # limit patch size to image dimensions
 
    # Positioning: around the center with jitter up to 10% of dimensions to avoid perfect centering
    cy = h // 2 + np.random.randint(-h // 10, h // 10 + 1)
    cx = w // 2 + np.random.randint(-w // 10, w // 10 + 1)
    y0 = max(0, min(cy - ps // 2, h - ps))
    x0 = max(0, min(cx - ps // 2, w - ps))
 
    # Patch gen: gaussian noise + checkerboard pattern to create high-frequency texture
    sq = 8  # size of checkerboard squares in pixels
    checkerboard = np.zeros((ps, ps), dtype=np.float32)
    for r in range(ps):
        for c in range(ps):
            checkerboard[r, c] = 255.0 if ((r // sq) + (c // sq)) % 2 == 0 else 0.0
 
    noise = np.random.normal(0, 25, (ps, ps)).astype(np.float32)
    patch_gray = np.clip(checkerboard + noise, 0, 255)
    patch_bgr = np.stack([patch_gray] * 3, axis=-1).astype(np.uint8)
 
    # Blend con alpha 0.85 per renderla visibile ma non completamente piatta
    # Blend (alpha 0.85) to make it visible but not completely flat
    roi = out[y0:y0 + ps, x0:x0 + ps].astype(np.float32)
    blended = cv2.addWeighted(roi, 0.15, patch_bgr.astype(np.float32), 0.85, 0)
    out[y0:y0 + ps, x0:x0 + ps] = np.clip(blended, 0, 255).astype(np.uint8)
 
    return out

perturbations = {
    "MedianFilter": apply_median_filter, 
    "GaussianNoise": apply_gaussian_noise,
    "MultiplicativeNoise": apply_multiplicative_noise, 
    "GaussianLowPass": apply_gaussian_low_pass,
    "ZoomBlur": apply_zoom_blur, 
    "ISONoise": apply_iso_noise,
    "Brightness": apply_brightness, 
    "Contrast": apply_contrast, 
    "Fog": apply_fog,
    "JPEGCompression": apply_jpeg, 
    "Rotate": apply_rotate, 
    "Cutout": apply_cutout, 
    "CannyEdge": apply_canny,
    "ImpulseNoise": apply_impulse_noise,
    "ShotNoise": apply_shot_noise,
    "PoissonNoise": apply_poisson_noise,
    "DefocusBlur": apply_defocus_blur,
    "MotionBlur": apply_motion_blur,
    "Saturation": apply_saturation,
    "GammaContrast": apply_gamma_contrast,
    "Snow": apply_snow,
    "Frost": apply_frost,
    "Splatter": apply_splatter,
    "Pixelate": apply_pixelate,
    "Scale": apply_scale,
    "Shear": apply_shear,
    "Jigsaw": apply_jigsaw,
    "CoarseDropout": apply_coarse_dropout,
    "AdversarialNoise": apply_adversarial_noise,
    "AdversarialPatch": apply_adversarial_patch,
}

imgs = {}
for f in os.listdir(IMAGES_DIR):
    if f.endswith(".jpg") or f.endswith(".png"):
        base_name = os.path.splitext(f)[0]
        if '_' not in base_name:
            imgs[base_name] = {"id": base_name, "img": cv2.imread(os.path.join(IMAGES_DIR, f))}

grouped_masks = {}
for f in os.listdir(MASKS_DIR):
    if f.endswith('.png'):
        img_id = f.split('_')[0] 
        cat_category = f.split('_')[1].split('.')[0]
        mask_data = load_binary_mask(os.path.join(MASKS_DIR, f))
        if img_id not in grouped_masks: grouped_masks[img_id] = []
        grouped_masks[img_id].append((cat_category, mask_data))

# Perturbation generation and saving

for img_id, img_data in imgs.items():
    original_img = img_data["img"]
    if original_img is None: continue
    
    for p_name, p_func in perturbations.items():
        cfg = perturbation_ranges[p_name]
        val = cfg["min"]
        
        # 'Step' cycle
        while (cfg["step"] > 0 and val <= cfg["max"]) or (cfg["step"] < 0 and val >= cfg["max"]):
            p_val = cfg["type"](val)
            if p_name == "Cutout":
                perturbed_img = p_func(original_img, p_val, grouped_masks.get(img_id, []))
            else:
                perturbed_img = p_func(original_img, p_val)

            val = round(val, 2)
            
            out_filename = f"{img_id}_{p_name}_{cfg['param']}_{val}.jpg"
            
            out_path = os.path.join(OUT_DIR, out_filename)
            cv2.imwrite(out_path, perturbed_img)
            
            val += cfg["step"]

    print(f"Completata immagine: {img_id}")

print("Elaborazione completata.")