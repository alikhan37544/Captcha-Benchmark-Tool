"""
model_loader.py — Factory that instantiates model drivers from config.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config_loader import Config, ModelConfig

from models.base import BaseCaptchaModel

logger = logging.getLogger(__name__)


def load_models(config: "Config") -> list[BaseCaptchaModel]:
    """Return a list of instantiated (but not yet loaded) model objects for enabled entries."""
    instances: list[BaseCaptchaModel] = []

    for mc in config.models:
        if not mc.enabled:
            logger.debug("Skipping disabled model: %s", mc.name)
            continue

        model = _build_model(mc, config.prompt)
        instances.append(model)
        logger.info("Registered model: %s (%s)", mc.name, mc.type)

    return instances


def _build_model(mc: "ModelConfig", prompt: str) -> BaseCaptchaModel:
    t = mc.type.lower()

    if t == "easyocr":
        from models.easyocr_model import EasyOCRModel
        return EasyOCRModel(name=mc.name)

    elif t == "tesseract":
        from models.tesseract_model import TesseractModel
        return TesseractModel(name=mc.name, psm=mc.psm)

    elif t == "huggingface":
        if not mc.model_id:
            raise ValueError(f"Model '{mc.name}' requires a model_id for type '{t}'")
        from models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(name=mc.name, model_id=mc.model_id, prompt=prompt)

    elif t == "lmstudio":
        if not mc.model_id or not mc.base_url:
            raise ValueError(f"Model '{mc.name}' requires model_id and base_url for type '{t}'")
        from models.lmstudio_model import LMStudioModel
        return LMStudioModel(
            name=mc.name,
            model_id=mc.model_id,
            base_url=mc.base_url,
            prompt=prompt,
            api_key=mc.api_key or "lmstudio",
        )

    elif t == "openai_compat":
        if not mc.model_id or not mc.base_url:
            raise ValueError(f"Model '{mc.name}' requires model_id and base_url for type '{t}'")
        from models.openai_compat_model import OpenAICompatModel
        return OpenAICompatModel(
            name=mc.name,
            model_id=mc.model_id,
            base_url=mc.base_url,
            prompt=prompt,
            api_key=mc.api_key or "sk-no-key",
        )

    elif t == "custom":
        if not mc.model_path or not mc.vocab:
            raise ValueError(f"Model '{mc.name}' requires model_path and vocab for type '{t}'")
        from models.custom_model import CustomModel
        return CustomModel(name=mc.name, model_path=mc.model_path, vocab=mc.vocab)

    else:
        raise ValueError(f"Unknown model type '{t}' for model '{mc.name}'")
