import cv2
import numpy as np

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

def apply_rotate(img, angle, masks_list=None):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    
    # Applica la stessa rotazione alle maschere, se presenti
    if masks_list:
        rotated_masks = []
        for category_name, mask in masks_list:
            rotated_mask = cv2.warpAffine(mask, M, (w, h))
            rotated_masks.append((category_name, rotated_mask))
    
    return rotated_img, rotated_masks

def apply_cutout(img, ratio, mask_list=None):
    h, w = img.shape[:2]
    res = img.copy()
    
    if mask_list and len(mask_list) > 0:
        for _, m_data in mask_list:
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
 
    # White pixel (salt), random coordinates
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
    
    # Generazione in blocco (vettoriale) di tutte le coordinate e parametri
    xs = np.random.randint(0, w, n_flakes)
    ys = np.random.randint(0, h, n_flakes)
    radii = np.random.randint(1, max(2, int(4 * intensity)) + 1, n_flakes)
    alphas = np.clip(np.random.uniform(0.5, 1.0, n_flakes) * intensity * 2, 0, 1.0)
    
    # Usiamo un solo layer vuoto
    flake_layer = np.zeros_like(out)
    
    for i in range(n_flakes):
        cv2.circle(flake_layer, (xs[i], ys[i]), radii[i], (255, 255, 255), -1)

    
    avg_alpha = np.mean(alphas)
    out = cv2.addWeighted(out, 1.0, flake_layer, avg_alpha, 0)
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

def _get_scale_params(h, w, factor):
    """
    (Privata) Calcola le nuove dimensioni e le coordinate (y0, x0) 
    per il ritaglio o il padding, senza toccare l'immagine.
    """
    new_h, new_w = int(h * factor), int(w * factor)
    
    if factor < 1.0:
        # Calcolo bordi per lo zoom out
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
    else:
        # Calcolo ritaglio per lo zoom in
        y0 = (new_h - h) // 2
        x0 = (new_w - w) // 2
        
    return new_h, new_w, y0, x0

def _apply_scale_from_params(array, h, w, new_h, new_w, y0, x0, factor, is_mask=False):
    """
    (Privata) Applica il ridimensionamento e il crop/pad a un singolo array 
    usando le coordinate pre-calcolate.
    """
    # Interpolazione: Nearest per le maschere, Lineare per l'immagine
    interp_method = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    
    resized = cv2.resize(array, (new_w, new_h), interpolation=interp_method)

    if factor < 1.0:
        # Crea la tela nera e incolla l'immagine rimpicciolita al centro
        out = np.zeros_like(array)
        out[y0:y0 + new_h, x0:x0 + new_w] = resized
    else:
        # Ritaglia il centro dell'immagine ingrandita
        out = resized[y0:y0 + h, x0:x0 + w]
        
    return out

def apply_scale(img, factor=1.0, masks_list=None):
    """
    Applica l'effetto Scale a un'immagine e alle sue maschere.
    factor < 1 → zoom out (bordi neri aggiunti).
    factor > 1 → zoom in (ritaglio centrale).
    """
    h, w = img.shape[:2]
    
    # 1. Calcoliamo la matematica una volta sola
    new_h, new_w, y0, x0 = _get_scale_params(h, w, factor)
    
    # 2. Applichiamo la trasformazione all'immagine principale (is_mask=False)
    perturbed_img = _apply_scale_from_params(img, h, w, new_h, new_w, y0, x0, factor, is_mask=False)
    
    # 3. Applichiamo la stessa identica trasformazione a tutte le maschere (is_mask=True)
    perturbed_masks = []
    if masks_list:
        for category_name, mask_array in masks_list:
            pert_mask = _apply_scale_from_params(mask_array, h, w, new_h, new_w, y0, x0, factor, is_mask=True)
            perturbed_masks.append((category_name, pert_mask))
            
    return perturbed_img, perturbed_masks
 
 
def apply_shear(img, angle=10.0, masks_list=None):
    """
    Applica una trasformazione di shear (inclinazione) all'immagine e alle sue maschere.
    angle: angolo di shear in gradi (positivo = verso destra, negativo = verso sinistra).
    """
    h, w = img.shape[:2]
    
    # Calcolo della matrice affine M 
    shear = np.tan(np.deg2rad(angle))
    M = np.float32([
        [1, shear, -shear * h / 2],
        [0, 1,     0             ]
    ])
    
    # Elaborazione Immagine 
    perturbed_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Elaborazione Maschere (Interpolazione Nearest per preservare i binari 0/1)
    perturbed_masks = []
    
    if masks_list:
        for category_name, mask_array in masks_list:
            pert_mask = cv2.warpAffine(mask_array, M, (w, h), flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            perturbed_masks.append((category_name, pert_mask))
            
    return perturbed_img, perturbed_masks

def _get_jigsaw_indices(grid_size):
    """ Genera solo l'ordine casuale dei tasselli."""
    n = max(2, int(grid_size))
    total_patches = n * n
    indices = list(range(total_patches))
    np.random.shuffle(indices)
    return indices, n

def _apply_jigsaw_from_indices(img, indices, n, is_mask=False):
    """ Applica gli indici a una singola immagine o maschera."""
    h, w = img.shape[:2]
    out = img.copy()

    cell_h = h // n
    cell_w = w // n

    if cell_h < 1 or cell_w < 1:
        return out 

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

    interp_method = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR

    for dst_idx, src_idx in enumerate(indices):
        y1, x1, y2, x2 = coords[dst_idx]
        dst_h = y2 - y1
        dst_w = x2 - x1
        
        src_patch = patches[src_idx]
        if src_patch.shape[:2] != (dst_h, dst_w):
            src_patch = cv2.resize(src_patch, (dst_w, dst_h), interpolation=interp_method)
            
        out[y1:y2, x1:x2] = src_patch

    return out


def apply_jigsaw(img, grid_size=4, masks_list=None):
    """
    Applica l'effetto Jigsaw a un'immagine e alle sue maschere, garantendo 
    che tutte subiscano esattamente la stessa permutazione spaziale.
    
    Parametri:
    - img: L'immagine originale (numpy array)
    - masks_list: Lista di tuple nel formato [(nome_categoria, array_maschera), ...]
    - grid_size: Numero di celle per lato (es. 4 per una griglia 4x4)
    
    Ritorna:
    - perturbed_img: L'immagine a puzzle
    - perturbed_masks: Lista di tuple aggiornata [(nome_categoria, maschera_puzzle), ...]
    """
    
    # Generiamo il rimescolamento una sola volta
    indices, n = _get_jigsaw_indices(grid_size)
    
    # Perturbiamo l'immagine principale
    perturbed_img = _apply_jigsaw_from_indices(img, indices, n, is_mask=False)
    
    # Applichiamo lo stesso rimescolamento a tutte le maschere
    perturbed_masks = []
    if masks_list is not None:
        for category_name, mask_array in masks_list:
            pert_mask = _apply_jigsaw_from_indices(mask_array, indices, n, is_mask=True)
            perturbed_masks.append((category_name, pert_mask))
        
    return perturbed_img, perturbed_masks

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
 
 
def apply_adversarial_patch(img, patch_ratio=0.15, mask_list=None):
    if not mask_list:
        raise ValueError("Errore critico: mask_list vuota o None.")

    h, w = img.shape[:2]
    
    out = img.copy() 
    
    patches_to_draw = []

    # Scorriamo TUTTE le maschere per trovare i 3, 4 o 5 punti in cui disegnare
    for _, m_data in mask_list:
        y_coords, x_coords = np.where(m_data > 0)
        mask_area = len(y_coords)
        
        if mask_area > 0:
            ps = int(np.sqrt(mask_area * patch_ratio))
            ps = max(8, ps)
            ps = min(ps, h, w)
            
            cy_exact = int(np.mean(y_coords))
            cx_exact = int(np.mean(x_coords))
            
            obj_h = np.max(y_coords) - np.min(y_coords)
            obj_w = np.max(x_coords) - np.min(x_coords)
            
            limit_y = max(1, int(obj_h * 0.15))
            limit_x = max(1, int(obj_w * 0.15))
            
            offset_y = np.random.randint(-limit_y, limit_y + 1)
            offset_x = np.random.randint(-limit_x, limit_x + 1)
            
            cy = cy_exact + offset_y
            cx = cx_exact + offset_x

            y0 = max(0, min(cy - ps // 2, h - ps))
            x0 = max(0, min(cx - ps // 2, w - ps))
            
            # Aggiungiamo le coordinate alla lista
            patches_to_draw.append((y0, x0, ps))

    # Applichiamo TUTTE le patch calcolate sulla STESSA immagine "out"
    for y0, x0, ps in patches_to_draw:
        sq = max(4, ps // 8) 
        
        r_idx, c_idx = np.indices((ps, ps))
        checkerboard = (((r_idx // sq) + (c_idx // sq)) % 2 == 0).astype(np.float32) * 255.0
        
        noise = np.random.normal(0, 25, (ps, ps)).astype(np.float32)
        patch_gray = np.clip(checkerboard + noise, 0, 255)
        patch_bgr = np.stack([patch_gray] * 3, axis=-1).astype(np.uint8)
        
        # Ritagliamo il pezzo della tela "out", lo misceliamo e lo RI-INCOLLIAMO su "out"
        roi = out[y0:y0 + ps, x0:x0 + ps].astype(np.float32)
        blended = cv2.addWeighted(roi, 0.15, patch_bgr.astype(np.float32), 0.85, 0)
        out[y0:y0 + ps, x0:x0 + ps] = np.clip(blended, 0, 255).astype(np.uint8)

    # Restituiamo la singola tela con tutte le patch applicate
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
    "AdversarialPatch": apply_adversarial_patch,
}