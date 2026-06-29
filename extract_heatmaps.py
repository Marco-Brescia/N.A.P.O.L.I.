import os
import shutil

SOURCE_DIR = "output"
DEST_DIR = "extracted_heatmaps"

def collect_heatmaps_for_image(img_id):
    out_folder = os.path.join(DEST_DIR, img_id)
    os.makedirs(out_folder, exist_ok=True)
    
    found_count = 0
    
    for root, _, files in os.walk(SOURCE_DIR):
        if os.path.basename(root) == img_id:
            parts = root.replace("\\", "/").split("/")
            if len(parts) >= 4:
                param = parts[-2]
                perturbation = parts[-3]
            else:
                param = "Normale"
                perturbation = "Originale"
            
            if perturbation == SOURCE_DIR or perturbation == "":
                perturbation = "Originale"

            heatmaps_dir = os.path.join(root, "heatmaps")
            
            # --- LOGICA CORRETTA ---
            if os.path.exists(heatmaps_dir) and os.listdir(heatmaps_dir) != []:
                # Se la cartella heatmaps ESISTE, copia le heatmap (jpg o png),
                # escludendo le _thresholded.*
                for file in os.listdir(heatmaps_dir):
                    if "_heatmap." in file:
                        ext = os.path.splitext(file)[1]
                        token_name = file.replace(f"{img_id}_", "").replace(f"_heatmap{ext}", "")
                        new_hm_name = f"{perturbation}_{param}_{token_name}{ext}"

                        shutil.copy2(os.path.join(heatmaps_dir, file), os.path.join(out_folder, new_hm_name))
                        found_count += 1
            else:
                # Se la cartella heatmaps NON ESISTE, salva il TXT
                txt_filename = f"{img_id}_tokens.txt"
                if txt_filename in files:
                    new_txt_name = f"{img_id}_{perturbation}_{param}.txt"
                    
                    shutil.copy2(os.path.join(root, txt_filename), os.path.join(out_folder, new_txt_name))
                    found_count += 1

    # Pulizia
    if found_count > 0 or len(os.listdir(out_folder)) > 0:
        print(f"[{img_id}] -> OK: Salvato in {out_folder}")
    else:
        os.rmdir(out_folder) 

def recover_ids_from_folders(source_dir=SOURCE_DIR):
    ids = set()
    for _, dirs, _ in os.walk(source_dir):
        for d in dirs:
            if d.isdigit():
                ids.add(d)
    return sorted(ids)

if __name__ == "__main__":
    for img_id in recover_ids_from_folders(SOURCE_DIR):
        collect_heatmaps_for_image(img_id)