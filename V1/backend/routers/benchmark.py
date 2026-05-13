"""
routers/benchmark.py — REST endpoints for benchmark control.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

import benchmark_runner as br
from config_loader import Config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Injected at startup by main.py
_config: Config | None = None
_models: list | None = None
_bg_task: asyncio.Task | None = None


def init(config: Config, models: list) -> None:
    global _config, _models
    _config = config
    _models = models


@router.post("/start")
async def start_benchmark():
    global _bg_task

    if br.state.running:
        raise HTTPException(status_code=409, detail="Benchmark already running.")

    if not _models:
        raise HTTPException(status_code=400, detail="No enabled models configured.")

    br.state.reset()
    # Drain stale events from queue
    while not br.event_queue.empty():
        br.event_queue.get_nowait()

    _bg_task = asyncio.create_task(br.run_benchmark(_models, _config))
    return {"status": "started", "run_id": "pending"}


@router.post("/stop")
async def stop_benchmark():
    if not br.state.running:
        raise HTTPException(status_code=409, detail="No benchmark is running.")
    br.state.cancelled = True
    return {"status": "cancellation_requested"}


@router.get("/status")
async def get_status():
    s = br.state
    elapsed = None
    if s.started_at is not None:
        import time
        end = s.finished_at if s.finished_at else time.time()
        elapsed = round(end - s.started_at, 1)

    return {
        "running": s.running,
        "cancelled": s.cancelled,
        "run_id": s.run_id,
        "current_model": s.current_model,
        "model_index": s.current_model_index,
        "total_models": s.total_models,
        "total_images": s.total_images,
        "current_image_index": s.current_image_index,
        "accuracy": s.accuracy,
        "error": s.error,
        "elapsed_s": elapsed,
    }


@router.get("/results")
async def get_results():
    if br.state.last_results is None:
        raise HTTPException(status_code=404, detail="No results available yet.")
    return br.state.last_results


@router.get("/models")
async def list_models():
    if _config is None:
        raise HTTPException(status_code=503, detail="Config not loaded.")
    return [
        {
            "name": m.name,
            "type": m.type,
            "enabled": m.enabled,
            "model_id": m.model_id,
        }
        for m in _config.models
    ]
