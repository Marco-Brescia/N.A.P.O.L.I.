import os
import cv2
import csv
import ast
import time
import math
import torch
import threading
import numpy as np
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

PERTURBAZIONI_FORTI = ["Jigsaw", "Rotate", "Scale", "Shear", "Cutout"]  # Queste perturbazioni richiedono un trattamento speciale per le maschere

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

SOGLIA_METRICA = 0.2  # Soglia minima per considerare un layer rilevante nell'accumulatore globale

# ---------------------------------------------------------------------------
# Funzioni di Supporto
# ---------------------------------------------------------------------------

def find_target_tokens(generated_tokens_lower: list, target_category: str) -> list:
    """
    Cerca il target nei token:
    1. Match esatto di un singolo token (priorità).
    2. Unione di più token contigui (es. 'gir' + 'affe' == 'giraffe').
    """
    target_clean = target_category.replace(' ', '').replace('-', '').replace('_', '').lower()
    tokens_clean = [t.replace(' ', '').replace('-', '').replace('_', '') for t in generated_tokens_lower]
    
    # Match Esatto
    if target_clean in tokens_clean:
        return [tokens_clean.index(target_clean)]
        
    # Match Contiguo (Sub-words)
    for i in range(len(tokens_clean)):
        current_aggregated_str = ""
        for j in range(i, len(tokens_clean)):
            current_aggregated_str += tokens_clean[j]
            
            if current_aggregated_str == target_clean:
                return list(range(i, j + 1))
            
            if len(current_aggregated_str) > len(target_clean):
                break
                
            if not target_clean.startswith(current_aggregated_str):
                break
                
    return [] 

def save_thresholded_viz(heatmap_array: np.ndarray, original_img_path: str, save_path: str, mask_shape: tuple, threshold: float = 0.66):
    """Genera l'immagine a 3 pannelli (Originale | Heatmap | Maschera sogliata in Rosso)"""
    h, w = mask_shape[:2]
    
    orig_bgr_img = cv2.imread(original_img_path)
    if orig_bgr_img is None: 
        return
    orig_bgr_img = cv2.resize(orig_bgr_img, (w, h))

    heatmap_resized = cv2.resize(heatmap_array, (w, h))
    heatmap_normalized = _pnorm(heatmap_resized)
    heatmap_8bit = (heatmap_normalized * 255).astype(np.uint8)

    threshold_value = int(255 * threshold)
    _, binary_prediction = cv2.threshold(heatmap_8bit, threshold_value, 255, cv2.THRESH_BINARY)
    
    heatmap_color_map = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(heatmap_color_map, 0.6, orig_bgr_img, 0.4, 0)

    threshold_overlay = orig_bgr_img.copy()
    threshold_overlay[binary_prediction == 255] = [0, 0, 255]
    threshold_overlay = cv2.addWeighted(threshold_overlay, 0.6, orig_bgr_img, 0.4, 0)
    
    cv2.putText(threshold_overlay, f"Soglia: {threshold_value} ({threshold*100:.0f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

    final_debug_visualization = np.hstack((orig_bgr_img, heatmap_overlay, threshold_overlay))
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), final_debug_visualization)


def _make_layer_grid_in_memory(layer_images, token_label, out_path, cols=8, pad=8, label_h=22, bg=(0,0,0), fg=(255,255,255)):
    """Crea una griglia PIL elaborando direttamente array OpenCV dalla RAM, senza salvare file intermedi."""
    tiles = []
    for layer_idx, img_cv2 in layer_images:
        im = None
        if img_cv2 is not None:
            # Converte da BGR (OpenCV) a RGB (PIL) in memoria
            img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img_rgb)
        tiles.append((layer_idx, im))

    valid = [im for _, im in tiles if im is not None]
    if not valid:
        return
        
    tw = max(im.size[0] for im in valid)
    th = max(im.size[1] for im in valid)
    resized = [(li, im.resize((tw,th), Image.BILINEAR) if im and im.size!=(tw,th) else im) for li, im in tiles]

    rows = math.ceil(len(resized) / cols)
    W = cols*tw + (cols+1)*pad
    H = rows*(th+label_h) + (rows+1)*pad
    canvas = Image.new("RGB", (W, H), bg)
    draw   = ImageDraw.Draw(canvas)
    
    try: font = ImageFont.load_default()
    except Exception: font = None

    for i, (li, im) in enumerate(resized):
        r, c = i // cols, i % cols
        x0 = pad + c*(tw+pad); y0 = pad + r*(th+label_h+pad)
        draw.rectangle([x0, y0, x0+tw, y0+label_h], fill=bg)
        draw.text((x0+4, y0+4), f"L{li}", fill=fg, font=font)
        if im:
            canvas.paste(im, (x0, y0+label_h))
        else:
            draw.rectangle([x0, y0+label_h, x0+tw, y0+label_h+th], outline=fg)

    header_h = 30
    out_img = Image.new("RGB", (W, H+header_h), bg)
    out_img.paste(canvas, (0, header_h))
    draw2 = ImageDraw.Draw(out_img)
    draw2.text((pad, 7), f"token: {token_label[:70]}", fill=fg, font=font)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_img.save(str(out_path), quality=95)


# ---------------------------------------------------------------------------
# Core Inference Logic (Thread)
# ---------------------------------------------------------------------------

def run_inference_thread(image_task: dict): 
    image_id = image_task['id']
    target_masks = grouped_masks.get(image_task['mask_key'], [])
    
    if not target_masks:
        with print_lock:
            print(f"Immagine ID: {image_id} - Cartella: {image_task['folder']} - Zero maschere trovate. Skippata.\n{'-'*60}, Chiave di maschera: '{image_task['mask_key']}'")
        return
        
    image_output_dir = os.path.join(OUT_DIR, image_task['folder'], image_id)
    heatmaps_output_dir = os.path.join(image_output_dir, HEAT_DIR)
    os.makedirs(heatmaps_output_dir, exist_ok=True)

    for category_name, original_binary_mask in target_masks:
        binary_mask = original_binary_mask.copy()
        
    attempt = 1
    
    while True:
        thread_logs = [f"Immagine ID: {image_id} - Cartella: {image_task['folder']} - Numero maschere: {len(target_masks)} [Tentativo {attempt}]"]
        inference_context = None

        try:
            results_for_csv = []

            # Inferenza sincronizzata per evitare crash CUDA
            with inference_lock:
                inference_context = run_inference(model, processor, image_task['path'], PROMPT, model_type) 
            
            clean_token_labels = [label.replace('Ġ', '').replace(' ', '').replace(' ', '').strip().lower() for label in inference_context['token_labels']]

            # Salvataggio token labels testuali
            tokens_text_path = os.path.join(image_output_dir, f"{image_id}_tokens.txt")
            with open(tokens_text_path, "w", encoding="utf-8") as f_tok:
                f_tok.write(str(inference_context['token_labels']))
                thread_logs.append(f"Token labels salvati in: {tokens_text_path}")

            total_layers = inference_context["n_layers"]

            # Processamento per singola maschera (oggetto)
            for category_name, binary_mask in target_masks:
                global_heatmap_accumulator = np.zeros(inference_context["vision_shape"], dtype=np.float32)
                thread_logs.append(f"Valutazione maschera: {category_name} per immagine {image_id}")
                
                matched_token_ids = find_target_tokens(clean_token_labels, category_name)
                
                if matched_token_ids:
                    thread_logs.append(f"Token '{category_name}' trovato! Generazione GRIGLIA in corso...")
                    
                    csv_token_identifier = matched_token_ids[0] if len(matched_token_ids) == 1 else str(matched_token_ids)
                    base_heatmap_path = os.path.join(heatmaps_output_dir, f"{image_id}_{category_name}_heatmap.png")
                    
                    # Somma in memoria per layer e creazione griglia in RAM
                    layer_images_in_memory = []
                    
                    layers_to_plot = list(range(total_layers))
                    if (total_layers - 1) not in layers_to_plot:
                        layers_to_plot.append(total_layers - 1)

                    # Generazione heatmap per singolo layer
                    for layer_idx in layers_to_plot:
                        layer_logits = _build_logitlens_logits(inference_context["outputs"], model, layer_idx, total_layers)
                        
                        layer_heatmap_accumulator = np.zeros(inference_context["vision_shape"], dtype=np.float32)
                        
                        # Aggregazione heatmap sub-tokens per il layer corrente
                        for token_id in matched_token_ids:
                            token_heatmap_array = TAM(
                                inference_context["generated_ids"][0].tolist(),
                                inference_context["vision_shape"],
                                layer_logits, 
                                inference_context["special_ids"],
                                inference_context["vis_inputs"],
                                processor,
                                "", # NESSUN PATH: Disabilita il salvataggio su disco in TAM
                                token_id,
                                [],
                                False
                            )
                            layer_heatmap_accumulator += token_heatmap_array
                            
                        # Generazione overlay visivo per il layer
                        h_orig, w_orig = image_task['img'].shape[:2]
                        layer_hm_norm = _pnorm(layer_heatmap_accumulator)
                        layer_hm_8u = (layer_hm_norm * 255).astype(np.uint8)
                        
                        upscaled_heatmap = cv2.resize(layer_hm_8u, (w_orig, h_orig))
                        heatmap_color_map = cv2.applyColorMap(upscaled_heatmap, cv2.COLORMAP_JET)
                        
                        layer_overlay = cv2.addWeighted(heatmap_color_map, 0.6, image_task['img'], 0.4, 0)
                        
                        # Aggiungiamo l'immagine alla lista in memoria per la griglia
                        layer_images_in_memory.append((layer_idx, layer_overlay))
                        
                        # Accumulo globale solo se il layer è rilevante
                        layer_metrics = compute_all_metrics(layer_heatmap_accumulator, binary_mask)
                        if round(layer_metrics["io_ratio"], 4) > SOGLIA_METRICA: 
                            global_heatmap_accumulator += layer_heatmap_accumulator

                    # Creazione griglia combinata interamente in RAM
                    _make_layer_grid_in_memory(
                        layer_images=layer_images_in_memory,
                        token_label=category_name,
                        out_path=base_heatmap_path,
                        cols=8
                    )

                    # Pulizia array per ottimizzare l'uso della memoria
                    del layer_images_in_memory
                    
                    # Metriche finali sull'accumulatore globale
                    final_global_heatmap = (_pnorm(global_heatmap_accumulator) * 255.0).astype(np.uint8)
                    final_metrics = compute_all_metrics(final_global_heatmap, binary_mask)

                    final_thresholded_path = os.path.join(heatmaps_output_dir, f"{image_id}_{category_name}_thresholded.jpg")
                    save_thresholded_viz(final_global_heatmap, image_task['path'], final_thresholded_path, binary_mask.shape)

                    results_for_csv.append({
                        "img_id": image_id,
                        "prompt": PROMPT.replace('\n', ' ').strip(),
                        "output": inference_context["gen_text"].replace('\n', ' ').strip(),
                        "target_token_id": csv_token_identifier,
                        "readable_token": category_name,
                        "obj_iou": round(final_metrics["obj_iou"], 4),
                        "io_ratio": round(final_metrics["io_ratio"], 4),
                        "wdp": round(final_metrics["wdp"], 4),
                    })
                else:
                    thread_logs.append(f"Errore: Il modello non ha generato il token '{category_name}' per l'immagine {image_id}.")
                    
            if results_for_csv:
                csv_output_path = os.path.join(image_output_dir, "results.csv")
                csv_headers = ["img_id", "prompt", "output", "target_token_id", "readable_token", "obj_iou", "io_ratio", "wdp"]
                
                with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    writer.writeheader()
                    writer.writerows(results_for_csv)
                thread_logs.append(f"CSV salvato in: {csv_output_path}")

            # Pulizia memoria
            del inference_context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with print_lock:
                print("\n".join(thread_logs))
                print("-" * 60)
            break 

        except Exception as e:
            if inference_context is not None: 
                del inference_context
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with print_lock:
                print(f"\n[! ERRORE TENTATIVO {attempt} !] L'immagine {image_id} ha causato un crash:\nErrore: {e}\n>>> Collisione/OOM rilevato. Attendo 1 secondo e riprovo...\n{'-'*60}")
            
            attempt += 1
            time.sleep(1) 


# ---------------------------------------------------------------------------
# Setup & Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start_process_time = time.time()

    IMAGES_DIR = "images_perturbed"
    MASKS_DIR  = "masks"
    OUT_DIR    = "output"
    HEAT_DIR   = "heatmaps"
    PERTURBED_MASKS_DIR = "masks_perturbed"
    PROMPT     = "Describe this image in detail, use simple terms"

    os.makedirs(OUT_DIR, exist_ok=True)

    print_lock = threading.Lock()
    inference_lock = threading.Lock()

    # --- Caricamento maschere ---
    grouped_masks = defaultdict(list)
    cartelle_da_leggere = [MASKS_DIR, PERTURBED_MASKS_DIR]
    print(f"Caricamento maschere da '{MASKS_DIR}' e '{PERTURBED_MASKS_DIR}'...")

    for cartella in cartelle_da_leggere:
        # Se una cartella non esiste
        if not os.path.exists(cartella): 
            continue
            
        for filename in os.listdir(cartella):
            if not filename.endswith('.png'):
                continue
                
            name_no_ext = os.path.splitext(filename)[0]
            
            parts = name_no_ext.rsplit('_', 1) 
            
            chiave = parts[0]     # Diventa "img01" OPPURE "img01_Jigsaw_4"
            categoria = parts[1]  # Diventa sempre e solo la categoria (es. "dog")
                
            mask_path = os.path.join(cartella, filename)
            binary_mask = load_binary_mask(mask_path)

            print(f"[{os.path.basename(cartella)}] Caricata: {filename} -> Chiave: '{chiave}' | Categoria: '{categoria}'")
                
            grouped_masks[chiave].append((categoria, binary_mask))

    

    # --- Caricamento immagini e Checkpoint Logic ---
    image_tasks_queue = []
    for filename in os.listdir(IMAGES_DIR):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
            
        # SPLIT ESATTO COME LO VOLEVI TU
        base_name = os.path.splitext(filename)[0]
        perturbation_type = ""
        
        if '_' in base_name:
            img_id, suffix = base_name.split('_', 1)
            perturbation_type, perturbation_param = suffix.split('_', 1) if '_' in suffix else (suffix, 'Normale')
        else:
            img_id = base_name
            perturbation_param = 'Normale'
            
        # LOGICA DELLA CHIAVE PER LE MASCHERE
        # Se sposta gli oggetti -> usiamo il nome completo (es. "img01_Jigsaw_4")
        # Altrimenti -> usiamo solo l'ID (es. "img01")
        if perturbation_type in PERTURBAZIONI_FORTI:
            chiave_maschera = base_name
        else:
            chiave_maschera = img_id

        full_folder_path = f"{perturbation_type}/{perturbation_param}" if (perturbation_type and perturbation_type != perturbation_param) else perturbation_param
        full_folder_path = full_folder_path.strip('/')
        
        image_checkpoint_dir = os.path.join(OUT_DIR, full_folder_path, img_id)
        tokens_checkpoint_file = os.path.join(image_checkpoint_dir, f"{img_id}_tokens.txt")
        results_checkpoint_dir = os.path.join(image_checkpoint_dir, HEAT_DIR)
        
        
        is_already_completed = False
        
        if os.path.isdir(image_checkpoint_dir) and os.path.exists(tokens_checkpoint_file):
            try:
                required_targets = [target[0].lower() for target in grouped_masks.get(chiave_maschera, [])]
                with open(tokens_checkpoint_file, "r", encoding="utf-8") as f_tok:
                    raw_tokens_list = ast.literal_eval(f_tok.read())
                
                generated_tokens = [t.replace('Ġ', '').replace(' ', '').replace(' ', '').strip().lower() for t in raw_tokens_list]
                
                is_already_completed = True
                for target_token in required_targets:
                    if target_token in generated_tokens:
                        if not os.path.exists(results_checkpoint_dir):
                            is_already_completed = False
                            break
                        
                        existing_heatmaps = os.listdir(results_checkpoint_dir)
                        if not any(f"_{target_token}_heatmap" in fname.lower() for fname in existing_heatmaps):
                            is_already_completed = False
                            break
            except Exception:
                is_already_completed = False # File corrotto, da rielaborare
                
        if is_already_completed:
            print(f"Skip: Immagine ID {img_id} ({full_folder_path}) già completata correttamente.")
            continue 
            
        image_tasks_queue.append({
            "id": img_id, 
            "mask_key": chiave_maschera,
            "path": os.path.join(IMAGES_DIR, filename), 
            "folder": full_folder_path, 
            "perturbation": perturbation_type,
            "param_str": perturbation_param,
            "img": cv2.imread(os.path.join(IMAGES_DIR, filename))
        })

    print(f"\nTotale nuove immagini in coda: {len(image_tasks_queue)}\n{'-'*60}")

    # --- Setup Modello ---
    model, processor, model_type = load_model(MODEL_NAME)
    model.eval()
    _check_norm(model)
    print(f"Model type: {model_type}")

    # --- Esecuzione Multithread ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(run_inference_thread, image_tasks_queue)

    # --- Statistiche Finali ---
    total_elapsed_time = time.time() - start_process_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n[{'='*50}]\nProcesso completo! Tempo totale di esecuzione: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")