"""
benchmark_runner.py — Async benchmark engine.

Runs each model sequentially (load → run all images → unload → next model),
pushes progress events to an asyncio.Queue for WebSocket consumers.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config_loader import Config
from models.base import BaseCaptchaModel
from result_store import save_results

logger = logging.getLogger(__name__)

# ─── State shared with FastAPI ──────────────────────────────────────────────

class BenchmarkState:
    def __init__(self) -> None:
        self.running: bool = False
        self.cancelled: bool = False
        self.run_id: str | None = None
        self.total_models: int = 0
        self.current_model_index: int = 0
        self.current_model: str = ""
        self.total_images: int = 0
        self.current_image_index: int = 0
        self.accuracy: dict[str, float] = {}
        self.last_results: dict[str, Any] | None = None
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def reset(self) -> None:
        self.running = False
        self.cancelled = False
        self.run_id = None
        self.total_models = 0
        self.current_model_index = 0
        self.current_model = ""
        self.total_images = 0
        self.current_image_index = 0
        self.accuracy = {}
        self.error = None
        self.started_at = None
        self.finished_at = None


# Singleton state and event queue
state = BenchmarkState()
event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()


# ─── Text normalisation ─────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """
    Normalise prediction for comparison.
    - Strips all non-alphanumeric characters (handles VLM quotes, dots, spaces)
    - Lowercases
    Example: ' "A3F2." ' → 'a3f2'
    """
    return re.sub(r"[^a-zA-Z0-9]", "", text).lower()



# ─── Image catalog ──────────────────────────────────────────────────────────

def build_image_list(config: Config) -> list[tuple[str, str, str]]:
    """
    Return list of (captcha_id, image_path, ground_truth) tuples for images
    that exist on disk AND have a label entry.
    """
    items = []
    for captcha_id, ground_truth in config.labels.items():
        fname = f"captcha_{captcha_id}.png"
        img_path = config.dataset_dir / fname
        if img_path.exists():
            items.append((captcha_id, str(img_path), ground_truth))
        else:
            logger.warning("Image not found, skipping: %s", img_path)
    return items


# ─── Runner ─────────────────────────────────────────────────────────────────

async def run_benchmark(models: list[BaseCaptchaModel], config: Config) -> None:
    """
    Main benchmark coroutine. Designed to run in a background asyncio task.
    Updates the global `state` and pushes events to `event_queue`.
    """
    global state, event_queue

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    state.run_id = run_id
    state.running = True
    state.cancelled = False
    state.error = None
    state.started_at = time.time()
    state.accuracy = {}

    items = build_image_list(config)
    if not items:
        state.running = False
        state.error = "No images found in dataset directory matching labels."
        await event_queue.put({"event": "error", "detail": state.error})
        return

    state.total_images = len(items)
    state.total_models = len(models)

    results: dict[str, Any] = {
        "run_id": run_id,
        "dataset": {
            "total": len(items),
            "labels_file": str(config.labels_file),
        },
        "models": {},
    }

    total_steps = len(models) * len(items)
    steps_done = 0

    for model_idx, model in enumerate(models):
        if state.cancelled:
            break

        state.current_model_index = model_idx
        state.current_model = model.name

        # Load model (blocking I/O — run in thread pool so event loop stays responsive)
        await event_queue.put({"event": "model_loading", "model": model.name})
        try:
            await asyncio.get_running_loop().run_in_executor(None, model.load)
        except Exception as exc:
            logger.exception("Failed to load model %s", model.name)
            await event_queue.put({"event": "model_load_error", "model": model.name, "detail": str(exc)})
            steps_done += len(items)
            continue

        await event_queue.put({"event": "model_loaded", "model": model.name})

        correct = 0
        predictions = []
        model_start = time.time()

        for img_idx, (captcha_id, img_path, ground_truth) in enumerate(items):
            if state.cancelled:
                break

            state.current_image_index = img_idx

            try:
                raw_pred = await asyncio.get_running_loop().run_in_executor(
                    None, model.predict, img_path
                )
            except Exception as exc:
                logger.exception("Inference failed for %s / %s", model.name, img_path)
                raw_pred = ""

            is_correct = normalise(raw_pred) == normalise(ground_truth)
            if is_correct:
                correct += 1

            predictions.append(
                {
                    "id": captcha_id,
                    "file": f"captcha_{captcha_id}.png",
                    "expected": ground_truth,
                    "predicted": raw_pred,
                    "correct": is_correct,
                }
            )

            steps_done += 1
            model_accuracy = correct / (img_idx + 1)
            state.accuracy[model.name] = round(model_accuracy, 4)

            overall_progress = steps_done / total_steps if total_steps > 0 else 0
            model_progress = (img_idx + 1) / len(items)

            # ETA estimation
            elapsed = time.time() - (state.started_at or time.time())
            eta_s = (elapsed / steps_done) * (total_steps - steps_done) if steps_done > 0 else None

            await event_queue.put(
                {
                    "event": "progress",
                    "model": model.name,
                    "model_index": model_idx,
                    "total_models": len(models),
                    "image": f"captcha_{captcha_id}.png",
                    "id": captcha_id,
                    "predicted": raw_pred,
                    "expected": ground_truth,
                    "correct": is_correct,
                    "model_progress": round(model_progress, 4),
                    "overall_progress": round(overall_progress, 4),
                    "accuracy": dict(state.accuracy),
                    "eta_s": round(eta_s, 1) if eta_s is not None else None,
                }
            )

        model_duration = time.time() - model_start
        final_accuracy = correct / len(items) if items else 0.0
        state.accuracy[model.name] = round(final_accuracy, 4)

        results["models"][model.name] = {
            "accuracy": round(final_accuracy, 4),
            "correct": correct,
            "total": len(items),
            "duration_s": round(model_duration, 2),
            "predictions": predictions,
        }

        # Unload to free memory before next model
        try:
            await asyncio.get_running_loop().run_in_executor(None, model.unload)
        except Exception:
            pass

        await event_queue.put(
            {
                "event": "model_done",
                "model": model.name,
                "accuracy": round(final_accuracy, 4),
                "correct": correct,
                "total": len(items),
            }
        )

    state.finished_at = time.time()
    state.running = False

    if state.cancelled:
        await event_queue.put({"event": "cancelled"})
        return

    # Persist results
    try:
        save_results(config.output_dir, run_id, results)
    except Exception as exc:
        logger.exception("Failed to save results")
        state.error = str(exc)

    state.last_results = results
    await event_queue.put(
        {
            "event": "done",
            "run_id": run_id,
            "accuracy": dict(state.accuracy),
            "duration_s": round(state.finished_at - state.started_at, 2),
        }
    )
