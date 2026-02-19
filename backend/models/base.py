"""
models/base.py — Abstract base class for all captcha solver backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCaptchaModel(ABC):
    """Every model driver must implement this interface."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load(self) -> None:
        """Load model weights / initialize libraries. Called once before inference."""

    @abstractmethod
    def predict(self, image_path: str) -> str:
        """
        Run inference on a single captcha image.

        :param image_path: Absolute path to the captcha PNG/JPEG.
        :returns: Predicted text string (raw, un-normalised).
        """

    def unload(self) -> None:
        """Optional: release GPU memory / handles after a run."""
