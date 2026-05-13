"""
result_store.py — Saves benchmark results to JSON and CSV.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_results(output_dir: Path, run_id: str, results: dict[str, Any]) -> dict[str, Path]:
    """
    Write results JSON and CSV.

    results shape expected:
    {
        "run_id": str,
        "dataset": {"total": int, "labels_file": str},
        "models": {
            model_name: {
                "accuracy": float,
                "correct": int,
                "total": int,
                "duration_s": float,
                "predictions": [
                    {"id": str, "file": str, "expected": str, "predicted": str, "correct": bool}
                ]
            }
        }
    }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_id = run_id.replace(":", "-").replace(" ", "_")
    json_path = output_dir / f"results_{safe_id}.json"
    csv_path = output_dir / f"results_{safe_id}.csv"

    # --- JSON ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results JSON saved → %s", json_path)

    # --- CSV (flat, one row per prediction per model) ---
    rows = []
    for model_name, model_data in results.get("models", {}).items():
        for pred in model_data.get("predictions", []):
            rows.append(
                {
                    "run_id": run_id,
                    "model": model_name,
                    "id": pred["id"],
                    "file": pred["file"],
                    "expected": pred["expected"],
                    "predicted": pred["predicted"],
                    "correct": pred["correct"],
                    "model_accuracy": model_data["accuracy"],
                }
            )

    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Results CSV saved → %s", csv_path)

    return {"json": json_path, "csv": csv_path}
