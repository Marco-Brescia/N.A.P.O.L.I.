import os
import cv2
import csv
import concurrent.futures
import threading
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from pathlib import Path
import time
import ast

from tam import TAM
from new_eval import (
    MODEL_NAME, 
    _build_logitlens_logits,
    _check_norm,
    _pnorm, 
    compute_all_metrics, 
    load_binary_mask, 
    load_model, 
    run_inference,
)
from demo import _make_layer_grid_for_token

def save_thresholded_viz(heatmap: np.ndarray, orig_img_path: str, save_path: str, mask_shape: tuple, threshold: float = 0.66):
    """Genera l'immagine a 3 pannelli (Originale | Heatmap | Maschera > 66% in Rosso)"""
    h, w = mask_shape[:2]
    
    orig_bgr = cv2.imread(orig_img_path)
    if orig_bgr is None: return
    orig_bgr = cv2.resize(orig_bgr, (w, h))

    hm_resized = cv2.resize(heatmap, (w, h))
    hm_norm = _pnorm(hm_resized)
    hm_8u = (hm_norm * 255).astype(np.uint8)

    t_val = int(255 * threshold)  # 168
    _, pred = cv2.threshold(hm_8u, t_val, 255, cv2.THRESH_BINARY)
    
    heatmap_color = cv2.applyColorMap(hm_8u, cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(heatmap_color, 0.6, orig_bgr, 0.4, 0)

    thresh_overlay = orig_bgr.copy()
    thresh_overlay[pred == 255] = [0, 0, 255]
    thresh_overlay = cv2.addWeighted(thresh_overlay, 0.6, orig_bgr, 0.4, 0)
    
    cv2.putText(thresh_overlay, f"Soglia: {t_val} ({threshold*100:.0f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    debug_vis = np.hstack((orig_bgr, heatmap_overlay, thresh_overlay))
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), debug_vis)

start_process_time = time.time()

IMAGES_DIR = "imgs"
MASKS_DIR  = "masks"
OUT_DIR    = "out_example"
HEAT_DIR   = "heatmaps"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

PROMPT = "Describe this image in detail, use simple terms"

print_lock = threading.Lock()
inference_lock = threading.Lock()

imgs = []

def run_inference_thread(img): # Funzione per eseguire l'inferenza in un thread separato, con gestione di errori
    masks = grouped_masks.get(img['id'], [])
    
    if len(masks) == 0:
        with print_lock:
            print(f"Immagine ID: {img['id']} - Cartella: {img['folder']} - Zero maschere trovate. Skippata.")
            print("-" * 60)
        return
        
    # Creazione ON THE FLY della cartella per questa immagine
    img_out_dir = os.path.join(OUT_DIR, img['folder'], img['id'])
    os.makedirs(os.path.join(img_out_dir, HEAT_DIR), exist_ok=True)
        
    tentativo = 1
    
    while True:
        local_logs = []
        local_logs.append(f"Immagine ID: {img['id']} - Cartella: {img['folder']} - Numero di maschere: {len(masks)} [Tentativo {tentativo}]")
        
        ctx = None

        try:
            results_per_img = []

            # Inferenza sincronizzata
            with inference_lock:
                ctx = run_inference(model, processor, img['path'], PROMPT, model_type) 
            
            token_labels_lower = [label.replace('Ġ', '').replace(' ', '').replace('\u2581', '').strip().lower() for label in ctx['token_labels']]

            # Salvataggio token labels
            token_txt_path = os.path.join(img_out_dir, f"{img['id']}_tokens.txt")
            with open(token_txt_path, "w", encoding="utf-8") as f_tok:
                f_tok.write(str(ctx['token_labels']))
                local_logs.append(f"Token labels salvati in: {token_txt_path}")

            n_layers = ctx["n_layers"]

            # Cicla sulle maschere per lavorare solo sul token target
            for mask in masks:
                category = mask[0]
                target_mask = mask[1]
                local_logs.append(f"Categoria maschera: {category} per immagine {img['id']}")
                
                token_id = None
                
                # Se trova il token continua...
                if category in token_labels_lower:
                    local_logs.append(f"Token '{category}' trovato! Generazione della GRIGLIA in corso...")
                    token_id = token_labels_lower.index(category)
                    
                    heatmap_path = os.path.join(img_out_dir, HEAT_DIR, f"{img['id']}_{category}_heatmap.png")
                    
                    temp_paths = []
                    final_heatmap_array = None
                    
                    layers_to_plot = list(range(n_layers))
                    if (n_layers - 1) not in layers_to_plot:
                        layers_to_plot.append(n_layers - 1)

                    # Genera heatmap singole per layer
                    for l_idx in layers_to_plot:
                        logits_l = _build_logitlens_logits(ctx["outputs"], model, l_idx, n_layers)
                        tmp_path = heatmap_path.replace(".png", f"_tmp_L{l_idx}.png")
                        
                        hm_array = TAM(
                            ctx["generated_ids"][0].tolist(),
                            ctx["vision_shape"],
                            logits_l, 
                            ctx["special_ids"],
                            ctx["vis_inputs"],
                            processor,
                            tmp_path,
                            token_id,
                            [],
                            False
                        )
                        temp_paths.append((l_idx, Path(tmp_path)))
                        
                        if l_idx == n_layers - 1:
                            final_heatmap_array = hm_array

                    # Creazione grid finale con tutte le heatmap
                    _make_layer_grid_for_token(
                        layer_heatmap_paths=temp_paths,
                        token_label=category,
                        out_path=Path(heatmap_path),
                        cols=8
                    )

                    # Elimina file temporanei
                    for _, p in temp_paths:
                        if p.exists():
                            os.remove(p)

                    # metriche su ultimo layer
                    if img['perturbation'] == "Rotate":
                        try:
                            angle = float(img['param_str'].replace('angle_', ''))
                            h, w = target_mask.shape[:2]
                            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                            target_mask = cv2.warpAffine(target_mask, M, (w, h), flags=cv2.INTER_NEAREST)
                            target_mask = (target_mask > 0).astype(np.uint8)
                        except Exception as e:
                            local_logs.append(f"Errore rotazione: {e}")

                    metrics = compute_all_metrics(final_heatmap_array, target_mask)

                    debug_path = os.path.join(img_out_dir, HEAT_DIR, f"{img['id']}_{category}_debug_thresholded.jpg")
                    save_thresholded_viz(final_heatmap_array, img['path'], debug_path, target_mask.shape)

                    results_per_img.append({
                        "img_id": img['id'],
                        "prompt": PROMPT.replace('\n', ' ').strip(),
                        "output": ctx["gen_text"].replace('\n', ' ').strip(),
                        "target_token_id": token_id,
                        "readable_token": category,
                        "obj_iou": round(metrics["obj_iou"], 4),
                        "func_iou": round(metrics["func_iou"], 4),
                        "f1_iou": round(metrics["f1_iou"], 4),
                        "iou_hard": round(metrics["iou_hard"], 4),
                        "io_ratio": round(metrics["io_ratio"], 4),
                        "wdp": round(metrics["wdp"], 4),
                        "precision": round(metrics["precision"], 4)
                    })
                else:
                    local_logs.append(f"Errore: Il modello non ha generato il token '{category}' per l'immagine {img['id']}.")

            # Salvataggio metriche in CSV
            if results_per_img:
                csv_path = os.path.join(img_out_dir, "results.csv")
                fieldnames = ["img_id", "prompt", "output", "target_token_id", "readable_token", "obj_iou", "func_iou", "f1_iou", "iou_hard", "io_ratio", "wdp", "precision"]
                
                with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results_per_img)
                local_logs.append(f"CSV salvato in: {csv_path}")

            # Pulisci cache
            del ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with print_lock:
                print("\n".join(local_logs))
                print("-" * 60)
            break 

        except Exception as e:
            if ctx is not None: del ctx
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with print_lock:
                print(f"\n[! ERRORE TENTATIVO {tentativo} !] L'immagine {img['id']} ha causato un crash:")
                print(f"Errore: {e}")
                print(">>> Collisione tra thread rilevata. Attendo 1 secondo e riprovo...")
                print("-" * 60)
            
            tentativo += 1
            time.sleep(1) 

# --- Caricamento maschere e raggruppamento per ID immagine ---
grouped_masks = {}
for f in os.listdir(MASKS_DIR):
    if f.endswith('.png'):
        img_id = f.split('_')[0] 
        cat_category = f.split('_')[1].split('.')[0]
        
        mask_path = os.path.join(MASKS_DIR, f)
        mask_data = load_binary_mask(mask_path)
        mask_tuple = (cat_category, mask_data)
        
        print(f"Caricata maschera: {f} per immagine: {img_id} con categoria: {mask_tuple[0]}")
        
        if img_id not in grouped_masks:
            grouped_masks[img_id] = []
        grouped_masks[img_id].append(mask_tuple)


# --- Caricamento immagini con logica checkpoint per evitare rielaborazione di immagini già processate correttamente ---
for f in os.listdir(IMAGES_DIR):
    radix_folder = ""
    if f.endswith('.jpg') or f.endswith('.png'):
        base_name = os.path.splitext(f)[0]
        if '_' in base_name:
            img_id, suffix = base_name.split('_', 1)
            radix_folder, folder_suffix = suffix.split('_', 1) if '_' in suffix else (suffix, 'Normale')
        else:
            img_id = base_name
            folder_suffix = 'Normale'
            
        if radix_folder and radix_folder != folder_suffix:
            full_folder = f"{radix_folder}/{folder_suffix}"
        else:
            full_folder = folder_suffix
            
        full_folder = full_folder.strip('/')
        
        #Logica di checkpoint: controlliamo se esiste già una cartella per questa immagine e se contiene i file token e heatmap necessari
        img_check_dir = os.path.join(OUT_DIR, full_folder, img_id)
        token_check_file = os.path.join(img_check_dir, f"{img_id}_tokens.txt")
        check_heatmap_dir = os.path.join(img_check_dir, HEAT_DIR)
        
        tokens_target = grouped_masks.get(img_id, [])
        tokens_target = [t[0].lower() for t in tokens_target]
        
        is_complete = False
        
        if os.path.isdir(img_check_dir) and os.path.exists(token_check_file):
            try:
                # cosa dice il modello
                with open(token_check_file, "r", encoding="utf-8") as f_tok:
                    raw_tokens = ast.literal_eval(f_tok.read())
                gen_tokens = [l.replace('Ġ', '').replace(' ', '').replace('\u2581', '').strip().lower() for l in raw_tokens]
                
                is_complete = True
                for t_target in tokens_target:
                    # se il token target è tra quelli generati, allora deve esserci anche la heatmap corrispondente
                    if t_target in gen_tokens:
                        if not os.path.exists(check_heatmap_dir):
                            is_complete = False
                            break
                        
                        heatmap_files = os.listdir(check_heatmap_dir)
                        if not any(f"_{t_target}_heatmap" in f_name.lower() for f_name in heatmap_files):
                            is_complete = False
                            break
            except Exception:
                is_complete = False # File corrotto
                
        if is_complete:
            print(f"Skip: Immagine ID {img_id} ({full_folder}) già completata correttamente.")
            continue 
            
        imgs.append({
            "id": img_id, 
            "path": os.path.join(IMAGES_DIR, f), 
            "folder": full_folder, 
            "perturbation": radix_folder,
            "param_str": folder_suffix,
            "img": cv2.imread(os.path.join(IMAGES_DIR, f))
        })

print(f"\nTotale nuove immagini in coda: {len(imgs)}\n" + "-"*60)

# --- Caricamento modello ---
model, processor, model_type = load_model(MODEL_NAME)
model.eval()
_check_norm(model)
print(f"Model type: {model_type}")

# Esecuzione in parallelo
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(run_inference_thread, imgs)

end_process_time = time.time()
elapsed_time = end_process_time - start_process_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\n[{'='*50}]")
print(f"Processo completo! Tempo totale di esecuzione: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")