"""
models/lmstudio_model.py — LMStudio (and local vision servers) via OpenAI-compat API.

Sends the captcha image as a base64 data URL inside a vision message.
No local model weights are needed; everything runs inside LMStudio.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from .base import BaseCaptchaModel

logger = logging.getLogger(__name__)


class LMStudioModel(BaseCaptchaModel):
    def __init__(
        self,
        name: str,
        model_id: str,
        base_url: str,
        prompt: str,
        api_key: str = "lmstudio",
    ):
        super().__init__(name)
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._prompt = prompt
        self._api_key = api_key
        self._client = None

    def load(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )
        logger.info("LMStudio client ready → %s (%s)", self._base_url, self._model_id)

    def predict(self, image_path: str) -> str:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        b64 = _image_to_b64(image_path)
        mime = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ],
            max_tokens=32,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def unload(self) -> None:
        self._client = None


def _image_to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
