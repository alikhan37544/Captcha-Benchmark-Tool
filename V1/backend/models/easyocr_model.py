"""
models/easyocr_model.py — EasyOCR-based captcha solver.
"""
from __future__ import annotations

from .base import BaseCaptchaModel


class EasyOCRModel(BaseCaptchaModel):
    def __init__(self, name: str, languages: list[str] | None = None):
        super().__init__(name)
        self._languages = languages or ["en"]
        self._reader = None

    def load(self) -> None:
        import easyocr

        self._reader = easyocr.Reader(self._languages, gpu=_gpu_available())

    def predict(self, image_path: str) -> str:
        if self._reader is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        results = self._reader.readtext(image_path, detail=0)
        return "".join(results).strip()

    def unload(self) -> None:
        self._reader = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
