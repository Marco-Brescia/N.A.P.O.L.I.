import os
import cv2
from new_eval import load_binary_mask
import numpy as np

IMAGES_DIR = "test_phase0"
MASKS_DIR  = "masks_experiment"
OUT_DIR    = "images_perturbed"

os.makedirs(OUT_DIR, exist_ok=True)

# Configurazione con il "passo" (step) per ogni perturbazione
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
    "CannyEdge":           {"param": "th",       "min": 100,  "max": 250,  "step": 75,   "type": int}
}

# Funzioni di perturbazione

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
                
                # limite scostamento
                limit_y = max(1, int(obj_h * 0.15))
                limit_x = max(1, int(obj_w * 0.15))
                
                # Applica lo scostamento casuale
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
    edges = cv2.Canny(img, th/2, th)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(img, 1.0, edges_bgr, 0.5, 0)

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
    "CannyEdge": apply_canny
}

# --- LOGICA DI CARICAMENTO (TUA) ---

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

# Generazione perturbazioni e salvataggio

for img_id, img_data in imgs.items():
    original_img = img_data["img"]
    if original_img is None: continue
    
    for p_name, p_func in perturbations.items():
        cfg = perturbation_ranges[p_name]
        val = cfg["min"]
        
        # Ciclo basato sul passo (step)
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