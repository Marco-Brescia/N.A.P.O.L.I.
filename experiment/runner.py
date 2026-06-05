import concurrent.futures
import logging
import time
import threading
from typing import Dict, List

import torch

from config import Config
from models import ImageTask, MaskEntry
from pipeline import process_image

logger = logging.getLogger(__name__)


class PipelineRunner:

    def __init__(self, model, processor, model_type: str,
                 grouped_masks: Dict[str, List[MaskEntry]], cfg: Config):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.grouped_masks = grouped_masks
        self.cfg = cfg

        # Serialise CUDA inference across threads to prevent context collisions.
        self._inference_lock = threading.Lock()

    def _run_with_retry(self, task: ImageTask) -> None:
        masks = self.grouped_masks.get(task.mask_key, [])

        if not masks:
            logger.warning(
                "[%s] No masks for key %r — skipping.", task.image_id, task.mask_key
            )
            return

        for attempt in range(1, self.cfg.max_retries + 1):
            logger.info(
                "[%s] Processing %s/%s (attempt %d/%d).",
                task.image_id, task.output_folder, task.image_id,
                attempt, self.cfg.max_retries,
            )
            try:
                with self._inference_lock:
                    process_image(
                        task=task,
                        masks=masks,
                        model=self.model,
                        processor=self.processor,
                        model_type=self.model_type,
                        cfg=self.cfg,
                    )

                logger.info("[%s] Completed successfully.", task.image_id)
                return  # Success — exit retry loop

            except Exception as exc:
                _flush_cuda()
                if attempt < self.cfg.max_retries:
                    logger.warning(
                        "[%s] Attempt %d failed: %s — retrying in %.1fs.",
                        task.image_id, attempt, exc, self.cfg.retry_sleep_seconds,
                    )
                    time.sleep(self.cfg.retry_sleep_seconds)
                else:
                    logger.error(
                        "[%s] All %d attempts failed. Last error: %s",
                        task.image_id, self.cfg.max_retries, exc,
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, tasks: List[ImageTask]) -> None:
        logger.info(
            "Starting runner: %d tasks, %d workers.",
            len(tasks), self.cfg.max_workers,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.cfg.max_workers
        ) as executor:
            futures = {executor.submit(self._run_with_retry, t): t for t in tasks}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                exc = future.exception()
                if exc:
                    logger.error("[%s] Task raised an unhandled exception: %s",
                                 task.image_id, exc)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _flush_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()