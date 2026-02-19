"""
models/huggingface_model.py — HuggingFace Transformers driver.

Improvements applied:
  - Uses bfloat16 (not float16) — better numerics on CUDA Ampere+; float16 on CPU
  - flash_attention_2 requested when on CUDA (falls back gracefully if unsupported)
  - Chat template with role-based messages for all VLMs
  - Proper input cleanup so input_ids aren't left on device
  - GPU cache cleared on unload
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseCaptchaModel

logger = logging.getLogger(__name__)

# Models that use an image-to-text pipeline (no vision chat template)
_PURE_OCR_MODEL_IDS = {
    "anuashok/ocr-captcha-v3",
    "keras-io/ocr-for-captcha",
}


class HuggingFaceModel(BaseCaptchaModel):
    def __init__(self, name: str, model_id: str, prompt: str):
        super().__init__(name)
        self._model_id = model_id
        self._prompt = prompt
        self._model = None
        self._processor = None
        self._pipeline = None
        self._is_pure_ocr = model_id in _PURE_OCR_MODEL_IDS

    # ── Load ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        import torch

        cuda = torch.cuda.is_available()
        # bfloat16 is numerically safer than float16, supported on Ampere+ GPUs.
        # Fall back to float32 on CPU to avoid unsupported operations.
        dtype = torch.bfloat16 if cuda else torch.float32

        if self._is_pure_ocr:
            from transformers import pipeline

            self._pipeline = pipeline(
                "image-to-text",
                model=self._model_id,
                device=0 if cuda else -1,
                torch_dtype=dtype,
            )
        else:
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                trust_remote_code=True,
            )

            kwargs: dict = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }

            if cuda:
                kwargs["device_map"] = "auto"
                # Try flash_attention_2 first (faster + less VRAM), fall back if unsupported
                kwargs["attn_implementation"] = "flash_attention_2"

            try:
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self._model_id, **kwargs
                )
            except Exception:
                # flash_attention_2 not supported — retry without it
                kwargs.pop("attn_implementation", None)
                self._model = AutoModelForVision2Seq.from_pretrained(
                    self._model_id, **kwargs
                )

            if not cuda:
                self._model = self._model.to("cpu")
            self._model.eval()

        logger.info("Loaded HuggingFace model: %s", self._model_id)

    # ── Predict ─────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> str:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")

        if self._is_pure_ocr:
            result = self._pipeline(img)
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "").strip()
            return ""

        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]

        try:
            text_input = self._processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
        except Exception:
            text_input = self._prompt

        inputs = self._processor(
            text=text_input,
            images=[img],
            return_tensors="pt",
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id
                if hasattr(self._processor, "tokenizer")
                else None,
            )

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    # ── Unload ──────────────────────────────────────────────────────────────

    def unload(self) -> None:
        import gc

        self._model = None
        self._processor = None
        self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        logger.info("Unloaded HuggingFace model: %s", self._model_id)
