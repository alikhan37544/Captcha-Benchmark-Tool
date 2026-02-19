"""
models/tesseract_model.py — pytesseract-based captcha solver.

Improvements:
  - Image is preprocessed before OCR: greyscale → upscale 2× → threshold
    This alone can boost accuracy dramatically on low-res captchas.
  - Configurable PSM (default 8 = single word)
  - Alphanumeric character whitelist
"""
from __future__ import annotations

import logging

from .base import BaseCaptchaModel

logger = logging.getLogger(__name__)


class TesseractModel(BaseCaptchaModel):
    def __init__(self, name: str, psm: int = 8):
        super().__init__(name)
        # OEM 3 = LSTM engine; PSM configurable
        self._config = (
            f"--oem 3 --psm {psm} "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        )

    def load(self) -> None:
        import pytesseract  # noqa: F401 — verify importable

    def predict(self, image_path: str) -> str:
        import pytesseract
        from PIL import Image, ImageFilter

        img = Image.open(image_path).convert("L")  # greyscale

        # Upscale 2× — Tesseract struggles with tiny images
        w, h = img.size
        img = img.resize((w * 2, h * 2), Image.LANCZOS)

        # Adaptive threshold to binarise (helps with noisy / gradient backgrounds)
        img = img.point(lambda p: 255 if p > 128 else 0)

        # Mild sharpening to clarify character edges
        img = img.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(img, config=self._config)
        return text.strip()

    def unload(self) -> None:
        pass
