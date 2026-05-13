"""
models/custom_model.py — Custom PyTorch checkpoint with CTC/greedy decoding.

Expects a model that accepts a (1, C, H, W) image tensor and outputs
logits of shape (T, 1, vocab_size) suitable for CTC decoding.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseCaptchaModel

logger = logging.getLogger(__name__)


class CustomModel(BaseCaptchaModel):
    def __init__(self, name: str, model_path: str, vocab: str):
        super().__init__(name)
        self._model_path = Path(model_path).resolve()
        self._vocab = vocab  # e.g. "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._model = None
        self._device = None

    def load(self) -> None:
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # weights_only=True prevents arbitrary code execution from untrusted checkpoints
        # (CVE-2025-32434). If your checkpoint contains a full nn.Module (not just a
        # state_dict), set weights_only=False and ONLY load models you trust.
        try:
            self._model = torch.load(
                self._model_path, map_location=self._device, weights_only=True
            )
        except Exception:
            logger.warning(
                "weights_only load failed for %s — falling back to full load. "
                "Only do this with TRUSTED checkpoints!",
                self._model_path,
            )
            self._model = torch.load(self._model_path, map_location=self._device)
        self._model.eval()
        logger.info("Loaded custom model from %s", self._model_path)

    def predict(self, image_path: str) -> str:
        import torch
        import torchvision.transforms as T
        from PIL import Image

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        transform = T.Compose(
            [
                T.Grayscale(num_output_channels=1),
                T.Resize((64, 200)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,)),
            ]
        )

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self._device)

        with torch.inference_mode():
            logits = self._model(tensor)  # (T, 1, vocab_size+1)

        return _ctc_greedy_decode(logits, self._vocab)

    def unload(self) -> None:
        self._model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _ctc_greedy_decode(logits, vocab: str) -> str:
    """Greedy CTC decoder — collapses repeated tokens and blank (index 0)."""
    import torch

    # logits: (T, B, C) — take argmax across class dimension
    indices = logits.squeeze(1).argmax(dim=-1).tolist()  # (T,)
    chars = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != 0:  # 0 is blank
                chars.append(vocab[idx - 1] if idx - 1 < len(vocab) else "")
        prev = idx
    return "".join(chars)
