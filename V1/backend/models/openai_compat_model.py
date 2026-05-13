"""
models/openai_compat_model.py — Generic OpenAI-compatible API driver.

Works with Ollama, OpenRouter, vLLM, Groq, or any server that exposes
the OpenAI chat completions API with vision support.
"""
from __future__ import annotations

import logging

from .lmstudio_model import LMStudioModel, _image_to_b64

logger = logging.getLogger(__name__)


class OpenAICompatModel(LMStudioModel):
    """
    Extends LMStudioModel — identical implementation, just a distinct type
    so the factory can label it separately and operators can supply a real api_key.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        base_url: str,
        prompt: str,
        api_key: str = "sk-no-key",
    ):
        super().__init__(
            name=name,
            model_id=model_id,
            base_url=base_url,
            prompt=prompt,
            api_key=api_key,
        )
