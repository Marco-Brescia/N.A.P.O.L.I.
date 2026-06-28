import logging
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from config import Config
from image_loader import build_task_queue
from mask_loader import load_all_masks

from new_eval import MODEL_NAME, _check_norm, load_model
from runner import PipelineRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = Config()
    start = time.time()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load masks
    logger.info("Loading masks from '%s' and '%s'...", cfg.masks_dir, cfg.perturbed_masks_dir)
    grouped_masks = load_all_masks(cfg.masks_dir, cfg.perturbed_masks_dir)

    # Build task queue (applies checkpoint filtering) ---
    logger.info("Scanning images in '%s'...", cfg.images_dir)
    tasks = build_task_queue(cfg, grouped_masks)

    if not tasks:
        logger.info("Nothing to do — all images already processed.")
        return

    # Load model 
    logger.info("Loading model: %s", MODEL_NAME)
    model, processor, model_type = load_model(MODEL_NAME)
    model.eval()
    _check_norm(model)
    logger.info("Model type: %s", model_type)

    # Run pipeline
    runner = PipelineRunner(
        model=model,
        processor=processor,
        model_type=model_type,
        grouped_masks=grouped_masks,
        cfg=cfg,
    )
    runner.run(tasks)

    # Summary
    elapsed = time.time() - start
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    logger.info("Pipeline complete. Total time: %02dh %02dm %05.2fs", int(h), int(m), s)


if __name__ == "__main__":
    main()