"""
config_loader.py — Reads config.yaml and labels CSV, exposes a Config dataclass.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str
    type: str  # easyocr | tesseract | huggingface | lmstudio | openai_compat | custom
    enabled: bool = True
    # HuggingFace / LMStudio / OpenAI-compat
    model_id: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    # Tesseract
    psm: int = 8
    # Custom PyTorch
    model_path: str | None = None
    vocab: str | None = None
    # Extra / forward-compat
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    dataset_dir: Path
    labels_file: Path
    output_dir: Path
    prompt: str
    models: list[ModelConfig]
    # id -> ground_truth (loaded from CSV)
    labels: dict[str, str] = field(default_factory=dict)


def load_config(config_path: str = "config.yaml") -> Config:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    base_dir = config_path.parent

    dataset_dir = (base_dir / raw["dataset_dir"]).resolve()
    labels_file = (base_dir / raw["labels_file"]).resolve()
    output_dir = (base_dir / raw["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = raw.get(
        "prompt",
        "What text is shown in this captcha image? Reply with ONLY the text, no explanation.",
    )

    models: list[ModelConfig] = []
    for m in raw.get("models", []):
        known_keys = {
            "name", "type", "enabled", "model_id", "base_url",
            "api_key", "psm", "model_path", "vocab",
        }
        extra = {k: v for k, v in m.items() if k not in known_keys}
        models.append(
            ModelConfig(
                name=m["name"],
                type=m["type"],
                enabled=m.get("enabled", True),
                model_id=m.get("model_id"),
                base_url=m.get("base_url"),
                api_key=m.get("api_key"),
                psm=m.get("psm", 8),
                model_path=m.get("model_path"),
                vocab=m.get("vocab"),
                extra=extra,
            )
        )

    # --- Load labels ---
    labels: dict[str, str] = {}
    if labels_file.exists():
        with open(labels_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                captcha_id = str(row["id"]).strip()
                value = str(row["value"]).strip()
                labels[captcha_id] = value

    return Config(
        dataset_dir=dataset_dir,
        labels_file=labels_file,
        output_dir=output_dir,
        prompt=prompt,
        models=models,
        labels=labels,
    )
