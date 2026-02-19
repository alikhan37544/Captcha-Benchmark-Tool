"""
main.py — FastAPI application entry point.
Uses lifespan context manager (replaces deprecated @app.on_event).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config_loader import load_config
from model_loader import load_models
from routers import benchmark as benchmark_router
from routers import ws as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get(
    "CONFIG_PATH",
    str(Path(__file__).parent.parent / "config.yaml"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Loading config from: %s", CONFIG_PATH)
    config = load_config(CONFIG_PATH)
    models = load_models(config)
    benchmark_router.init(config, models)

    enabled = [m.name for m in config.models if m.enabled]
    logger.info(
        "Ready. %d image(s) labeled | %d model(s) enabled: %s",
        len(config.labels),
        len(models),
        ", ".join(enabled) or "(none)",
    )

    # Serve captcha images at /captchas/<filename> for the frontend viewer
    dataset_dir = Path(config.dataset_dir)
    if dataset_dir.exists():
        app.mount(
            "/captchas",
            StaticFiles(directory=str(dataset_dir)),
            name="captchas",
        )
        logger.info("Serving captcha images from %s at /captchas", dataset_dir)

    # Start the WebSocket fan-out dispatcher
    import asyncio
    from routers.ws import _dispatcher
    dispatcher_task = asyncio.create_task(_dispatcher(), name="ws_dispatcher")

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    dispatcher_task.cancel()
    try:
        await dispatcher_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Captcha Benchmark Tool — Backend",
    description="Benchmarks multiple OCR / VLM models against a labeled captcha dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(benchmark_router.router)
app.include_router(ws_router.router)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
