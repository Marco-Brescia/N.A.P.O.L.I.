# N.A.P.O.L.I.

**N**oise-**A**ugmented **P**erturbation **O**bservatory for **L**anguage-**i**mage **I**nference

N.A.P.O.L.I. è una pipeline per studiare la robustezza di un Vision-Language
Model (VLM) di fronte a perturbazioni controllate dell'immagine. Per ogni
oggetto presente in un'immagine, la pipeline misura quanto bene il modello
riesce a *localizzare* quell'oggetto — tramite l'analisi delle sue mappe di
attenzione interne (TAM / logit-lens) confrontate con una maschera di
ground-truth — e come questa capacità di localizzazione degrada
all'aumentare dell'intensità di 29 diverse perturbazioni (rumore, blur,
occlusioni, trasformazioni geometriche, ecc.).

L'obiettivo finale è individuare automaticamente il **breaking point**: il
livello di perturbazione a partire dal quale il modello perde
sistematicamente la capacità di attenzionare correttamente un oggetto.

---

## Indice

- [Struttura del progetto](#struttura-del-progetto)
- [Flusso di lavoro](#flusso-di-lavoro)
- [1. Generazione delle perturbazioni](#1-generazione-delle-perturbazioni)
- [2. Inferenza VLM ed estrazione heatmap](#2-inferenza-vlm-ed-estrazione-heatmap)
- [3. Analisi e breaking point](#3-analisi-e-breaking-point)
- [Convenzioni sui nomi dei file](#convenzioni-sui-nomi-dei-file)
- [Perturbazioni supportate](#perturbazioni-supportate)
- [Metriche](#metriche)
- [Metodologia del breaking point](#metodologia-del-breaking-point)

---

## Struttura del progetto

```
.
├── imgs/                       # Immagini originali (baseline)
├── masks/                      # Maschere GT baseline:  <img_id>_<categoria>.png
│
├── perturbations/               # STAGE 1 — generazione delle perturbazioni
│   ├── config.py                 # Range e step di ogni perturbazione
│   ├── effects.py                # Implementazione delle 29 funzioni di perturbazione
│   └── run.py                    # Entry point: genera images_perturbed/ e masks_perturbed/
│
├── images_perturbed/            # Output di stage 1 (generato, non versionato)
├── masks_perturbed/             # Output di stage 1 per perturbazioni che spostano gli oggetti
│
├── experiment/                  # STAGE 2 — inferenza VLM + estrazione heatmap
│   ├── config.py                  # Configurazione centralizzata (path, soglie, retry)
│   ├── models.py                  # Dataclass: ImageTask, MaskEntry
│   ├── mask_loader.py             # Caricamento e raggruppamento delle maschere
│   ├── image_loader.py            # Scansione immagini + logica di checkpoint
│   ├── viz.py                     # Visualizzazione (heatmap, griglie per layer)
│   ├── heatmap.py                 # Token matching + generazione heatmap TAM
│   ├── pipeline.py                # Elaborazione di una singola immagine
│   ├── runner.py                  # Esecuzione parallela + retry + lock CUDA
│   └── main.py                    # Entry point dello stage 2
│
├── output/                      # Output di stage 2: results.csv + heatmap per immagine
│
├── plots/                        # STAGE 3 — analisi e individuazione breaking point
│   ├── data_loader.py             # Caricamento dei risultati
│   ├── plotter.py                 # Generazione dei grafici (heatmap, breaking point)
│   ├── main.py                    # Entry point dello stage 3 (CLI)
│   └── output/                    # Grafici generati (non versionato)
```

---

## Flusso di lavoro

La pipeline è organizzata in tre stage sequenziali, ciascuno con il proprio
entry point. Tutti i comandi vanno eseguiti **dalla root del progetto**.

```
 imgs/ + masks/
      │
      ▼
┌─────────────────────┐
│  perturbations/     │   genera 29 perturbazioni × N livelli di intensità
│  run.py             │   per ogni immagine in imgs/
└─────────────────────┘
      │
      ▼
 images_perturbed/ + masks_perturbed/
      │
      ▼
┌─────────────────────┐
│  experiment/        │   inferenza VLM + estrazione heatmap
│  main.py            │   calcolo metriche per ogni oggetto target
└─────────────────────┘
      │
      ▼
 output/<Perturbazione>/<param>/<img_id>/results.csv + heatmaps/
      │
      ▼
┌─────────────────────┐
│  plots/             │   aggregazione risultati + rilevamento breaking point
│  main.py            │   generazione grafici per ogni immagine/token
└─────────────────────┘
      │
      ▼
 plots/output/breaking_point_per_token/<Perturbazione>/<img_id>.png
```

---

## 1. Generazione delle perturbazioni

```bash
python perturbations/run.py
```

Legge le immagini da `imgs/` e le relative maschere da `masks/`, applica
tutte le 29 perturbazioni configurate in `perturbations/config.py` (ognuna
su un range di intensità a step fissi), e scrive:

- `images_perturbed/<img_id>_<Perturbazione>_<param>_<valore>.jpg`
- `masks_perturbed/<img_id>_<Perturbazione>_<param>_<valore>_<categoria>.png`
  — **solo** per le perturbazioni che spostano spazialmente gli oggetti
  (`Jigsaw`, `Rotate`, `Scale`, `Shear`), per cui la maschera baseline non è
  più valida e va ricalcolata per ogni livello di intensità.

Per `Cutout` e `AdversarialPatch` la maschera originale viene invece
riutilizzata internamente solo per decidere *dove* applicare il ritaglio o
la patch, senza generare una nuova maschera.

### Configurazione

Tutti i parametri (cartelle di I/O, range min/max, step) sono centralizzati
in `perturbations/config.py`. Per aggiungere una nuova perturbazione:

1. Implementare la funzione `apply_<nome>(img, ...)` in `effects.py`.
2. Registrarla nel dizionario `perturbations` in fondo al file.
3. Aggiungere una entry corrispondente in `perturbation_ranges` in `config.py`.

---

## 2. Inferenza ed estrazione heatmap

```bash
python experiment/main.py
```

### Cosa fa

1. Carica tutte le maschere da `masks/` e `masks_perturbed/`, raggruppandole
   per chiave immagine.
2. Scansiona `images_perturbed/` applicando una **logica di checkpoint**: le
   immagini già elaborate correttamente (token generati + heatmap già
   presenti) vengono saltate automaticamente, rendendo l'esecuzione
   ripetibile in sicurezza.
3. Carica il modello VLM una sola volta (`MODEL_NAME` in `new_eval.py`).
4. Elabora le immagini in parallelo (4 worker thread di default),
   serializzando le chiamate di inferenza CUDA.
5. Per ogni immagine e ogni categoria target:
   - Verifica se il modello ha effettivamente generato il token
     corrispondente nella descrizione prodotta.
   - Genera una heatmap per ogni layer del modello (TAM / logit-lens).
   - Accumula i layer "rilevanti" (sopra una soglia di `io_ratio`) in una
     heatmap globale.
   - Calcola `obj_iou`, `io_ratio`, `wdp` confrontando la heatmap globale
     con la maschera di ground-truth.
6. Salva `results.csv` e le heatmap per ogni immagine sotto `output/`.

### Configurazione

```python
# experiment/config.py
@dataclass(frozen=True)
class Config:
    images_dir: str = "images_perturbed"
    masks_dir: str = "masks"
    perturbed_masks_dir: str = "masks_perturbed"
    output_dir: str = "output"
    prompt: str = "Describe this image in detail, use simple terms"
    max_workers: int = 4
    layer_relevance_threshold: float = 0.2
    viz_threshold: float = 0.66
    max_retries: int = 10
```

---

## 3. Analisi e breaking point

```bash
python plots/main.py
```

### Opzioni CLI

| Opzione | Default | Descrizione |
|---|---|---|
| `--output-dir` | `output` | Cartella contenente i `results.csv` generati dallo stage 2 |
| `--plots-dir` | `plots/output` | Cartella di destinazione dei grafici |
| `--plots` | `heatmap breaking_point` | Tipi di grafico da generare |
| `--perturbations` | tutte | Limita l'analisi a specifiche perturbazioni |
| `--images` | tutte | Limita l'analisi a specifiche immagini |
| `--degradation-threshold` | `0.40` | Soglia di degrado relativo (`obj_iou`+`io_ratio`) per dichiarare un breaking point |
| `--wdp-arbiter-min` | `0.0` | Degrado minimo richiesto al `wdp` per confermare il breaking point |
| `--min-consecutive` | `2` | Step consecutivi di degrado richiesti prima di dichiarare il break |
| `--verbose` / `-v` | — | Logging a livello DEBUG |

**Esempio:**
```bash
python plots/main.py \
    --plots breaking_point \
    --perturbations Jigsaw Cutout \
    --images 000000000650 \
    --degradation-threshold 0.35
```

La generazione dei grafici è parallelizzata con `ProcessPoolExecutor`: per i
grafici per-immagine (`breaking_point`) viene lanciato un processo per ogni
immagine, per accelerare l'elaborazione su dataset di grandi dimensioni.

### Output generati

```
plots/output/
├── heatmaps/
│   └── <Perturbazione>_<metrica>_heatmap.png    # griglia immagini × livelli di perturbazione
└── breaking_point_per_token/
    └── <Perturbazione>/<img_id>.png              # analisi per singolo token/categoria
```

Ogni immagine `breaking_point_per_token` mostra, per ogni token/categoria
presente nell'immagine, una riga con:
- L'andamento delle 3 metriche lungo i livelli di perturbazione, a partire
  dalla baseline (`Normale`).
- La heatmap reale del modello al momento del breaking point (se rilevato).
- Annotazioni per i token persi (non generati dal modello) e per il punto
  di rottura individuato.

---

## Convenzioni sui nomi dei file

**Immagini perturbate** (`images_perturbed/`):
```
<img_id>_<Perturbazione>_<param>_<valore>.jpg
```
Esempio: `000000000650_AdversarialPatch_patch_ratio_0.07.jpg`

**Maschere** (`masks/`, `masks_perturbed/`):
```
<chiave>_<categoria>.png
```
Esempio: `000000015746_fire hydrant.png` → chiave `000000015746`, categoria `fire hydrant`

**Output dello stage 2** (`output/`):
```
output/<Perturbazione>/<param>/<img_id>/
├── <img_id>_tokens.txt
├── results.csv
└── heatmaps/
    ├── <img_id>_<categoria>_heatmap.png        # griglia per-layer
    └── <img_id>_<categoria>_thresholded.jpg    # overlay sogliato
```

**`results.csv`**:
```csv
img_id,prompt,output,target_token_id,readable_token,obj_iou,io_ratio,wdp
000000106757,"Describe this image...","The image shows a woman...",9,elephant,0.5353,0.7021,0.0479
```

---

## Perturbazioni supportate

29 perturbazioni raggruppate in 8 categorie, ognuna applicata su un range
di intensità configurabile in `perturbations/config.py`:

| Categoria | Perturbazioni |
|---|---|
| **Noise** | GaussianNoise, ImpulseNoise, ShotNoise, ISONoise, PoissonNoise, MultiplicativeNoise |
| **Blur/Smoothing** | GaussianLowPass, DefocusBlur, MotionBlur, ZoomBlur, MedianFilter |
| **Illumination/Color** | Brightness, Contrast, Saturation, GammaContrast |
| **Environmental** | Snow, Frost, Fog, Splatter |
| **Digital** | JPEGCompression, Pixelate |
| **Geometric** | Scale, Shear, Jigsaw, Rotate |
| **Occlusion** | Cutout, CoarseDropout |
| **Edge-Based** | CannyEdge, AdversarialPatch |

> Nota: per `Jigsaw`, `Rotate`, `Scale` e `Shear` la maschera di ground-truth
> viene rigenerata per ogni livello di intensità, perché la perturbazione
> sposta spazialmente l'oggetto target. Per tutte le altre perturbazioni si
> riutilizza la maschera baseline.

---

## Metriche

Le metriche sono calcolate confrontando, per ogni categoria target, la
heatmap di attenzione del modello (sogliata) con la maschera di
ground-truth, in `new_eval.py`.

| Metrica | Definizione | Direzione |
|---|---|---|
| `obj_iou` | Intersection over Union tra heatmap sogliata (Otsu) e maschera GT | più alto = meglio |
| `io_ratio` | Frazione della massa della heatmap che cade dentro la maschera GT | più alto = meglio |
| `wdp` (Weighted Distance Penalty) | Penalità per attivazione fuori dalla maschera GT, pesata per distanza euclidea | più basso = meglio |

`wdp` è una metrica di penalità: misura quanto l'attenzione del modello si
disperde fuori dall'oggetto target, pesando l'errore in base a quanto è
lontano dalla maschera. Per questo motivo gioca il ruolo di **arbitro** nel
rilevamento del breaking point (vedi sotto).

---

## Metodologia del breaking point

Il breaking point è determinato a **livello di singolo token** (non di
intera immagine), combinando due segnali con ruoli distinti:

1. **Segnale primario** — degrado medio relativo di `obj_iou` e `io_ratio`
   rispetto al valore baseline. Una caduta ≥ `degradation_threshold` indica
   degrado nelle prestazioni.

3. **Arbitro** — `wdp` deve confermare il degrado (non deve migliorare).
   Questo evita falsi positivi nei casi in cui `obj_iou` e `io_ratio`
   oscillano senza un reale deterioramento della localizzazione:
   se l'attenzione del modello resta concentrata
   sull'oggetto (`wdp` stabile o in calo), il breaking point **non** viene
   dichiarato.

Il break è confermato solo se la condizione si mantiene per
`min_consecutive` step consecutivi, per escludere picchi isolati non
rappresentativi di un degrado sistematico.

### Casi particolari gestiti

- **Token non generato** dal modello → trattato come degrado massimo
  (penalità sentinella `WDP_PENALTY_NO_TOKEN`), non come dato mancante
  silenzioso.
- **Heatmap vuota** (token generato ma TAM sotto soglia) → penalità
  sentinella `WDP_PENALTY_NO_HEATMAP`.
- **Baseline già a valore nullo** → impossibile stabilire un riferimento di
  degrado; il breaking point viene dichiarato immediatamente (step 0).
