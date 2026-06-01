import os
import cv2
from new_eval import load_binary_mask

# Importiamo i moduli creati sopra
import config
from effects import perturbations

os.makedirs(config.OUT_DIR, exist_ok=True)

# Image and mask loading
imgs = {}
for f in os.listdir(config.IMAGES_DIR):
    if f.endswith(".jpg") or f.endswith(".png"):
        base_name = os.path.splitext(f)[0]
        if '_' not in base_name:
            imgs[base_name] = {"id": base_name, "img": cv2.imread(os.path.join(config.IMAGES_DIR, f))}

grouped_masks = {}
for f in os.listdir(config.MASKS_DIR):
    if f.endswith('.png'):
        img_id = f.split('_')[0] 
        cat_category = f.split('_')[1].split('.')[0]
        mask_data = load_binary_mask(os.path.join(config.MASKS_DIR, f))
        if img_id not in grouped_masks: 
            grouped_masks[img_id] = []
        grouped_masks[img_id].append((cat_category, mask_data))

# Perturbation generation and saving
for img_id, img_data in imgs.items():
    original_img = img_data["img"]
    if original_img is None: 
        continue
    
    for p_name, p_func in perturbations.items():
        cfg = config.perturbation_ranges[p_name]
        val = cfg["min"]
        
        while (cfg["step"] > 0 and val <= cfg["max"]) or (cfg["step"] < 0 and val >= cfg["max"]):
            p_val = cfg["type"](val)
            if p_name == "Cutout":
                perturbed_img = p_func(original_img, p_val, grouped_masks.get(img_id, []))
            else:
                perturbed_img = p_func(original_img, p_val)

            val = round(val, 2)
            
            out_filename = f"{img_id}_{p_name}_{cfg['param']}_{val}.jpg"
            out_path = os.path.join(config.OUT_DIR, out_filename)
            cv2.imwrite(out_path, perturbed_img)
            
            val += cfg["step"]

    print(f"Completed image: {img_id}")

print("Elaboration successfully completed. Perturbed images saved in:", config.OUT_DIR)