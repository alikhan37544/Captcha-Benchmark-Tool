#!/usr/bin/env python3
"""
Benchmark captcha-solving models against the cleaned ground truth dataset.

Sends captcha images to vision APIs (OpenAI-compatible, Gemini native) with
configurable parallelism, rate-limiting, retry logic, and deep metrics.

Usage:
    python benchmark.py

Dependencies (install with pip):
    pip install requests pandas Pillow
"""

import io
import json
import time
import base64
import csv
import re
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from PIL import Image

# ============================================================================
# CONFIGURATION — edit these before running
# ============================================================================

# ── API mode ─────────────────────────────────────────────────────────────────
#   "openai_compat" → OpenAI / Ollama / LM Studio / DeepSeek / OpenRouter / vLLM
#   "gemini"         → Gemini native API (or Gemini proxy)
API_MODE = "openai_compat"

# ── For openai_compat mode ───────────────────────────────────────────────────
#   • Ollama:     http://localhost:11434/v1/chat/completions
#   • LM Studio:  http://localhost:1234/v1/chat/completions
#   • OpenAI:     https://api.openai.com/v1/chat/completions
#   • DeepSeek:   https://api.deepseek.com/v1/chat/completions
#   • OpenRouter: https://openrouter.ai/api/v1/chat/completions
#   • vLLM:       http://localhost:8000/v1/chat/completions
OPENAI_URL = "http://localhost:1234/v1/chat/completions"
OPENAI_API_KEY = ""           # leave empty if not required (Ollama, LM Studio, vLLM)

# ── For gemini mode ──────────────────────────────────────────────────────────
#   Native:   https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
#   Proxy:    https://your-proxy.com/api/google/v1/generate
GEMINI_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = ""           # for native API, appended as ?key=... ; for proxy, sent as Bearer token

# ── Model ────────────────────────────────────────────────────────────────────
#   openai_compat: model name as known by the provider (e.g. "llava:7b", "gpt-4o")
#   gemini:        ignored (URL already specifies the model)
MODEL_NAME = "zai-org/glm-4.6v-flash"

# ── Rate limiting, retries & concurrency ──────────────────────────────────────
CONCURRENCY               = 4     # 1 = sequential; >1 = parallel requests
DELAY_BETWEEN_REQUESTS_MS = 0     # 2000 = 2 s between requests (enforced across threads)
MAX_RETRIES               = 3     # 3 = retry up to 3 times on failure / rate-limit

# ── Dataset ──────────────────────────────────────────────────────────────────
MAX_SAMPLES = -1                  # number of images to test; -1 = all

PROMPT = (
    "You are a captcha solver. Look at this captcha image and read the text "
    "shown in it. Reply with ONLY the characters you see, in UPPERCASE. "
    "Do not add any spaces, punctuation, explanations, or additional text. "
    "Output exactly the characters and nothing else."
)

DATASET_CSV = "dataset/cleaned_ground_truth.csv"
IMAGES_DIR  = "raw/images"
OUTPUT_DIR  = "benchmark_runs"
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Refusal patterns — for error-category labelling only
_REFUSAL_RE = re.compile(
    r"\b(sorry|i\s+(can'?t|cannot|am\s+unable)|not\s+(able|allowed|possible))",
    re.IGNORECASE,
)

# Thread‑safe rate‑limit gate
_rate_lock = threading.Lock()
_last_call = 0.0


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def _encode_image_raw(image_path: Path) -> tuple[str, str]:
    """Return (mime_type, raw_base64_string)."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        fmt = (img.format or "PNG").upper()
        mime = f"image/{fmt.lower()}"
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return mime, raw_b64


def _wait_rate_limit() -> None:
    """Enforce DELAY_BETWEEN_REQUESTS_MS across all threads."""
    if DELAY_BETWEEN_REQUESTS_MS <= 0:
        return
    global _last_call
    with _rate_lock:
        now = time.perf_counter()
        wait = (DELAY_BETWEEN_REQUESTS_MS / 1000) - (now - _last_call)
        if wait > 0:
            time.sleep(wait)
            now = time.perf_counter()
        _last_call = now


# ---------------------------------------------------------------------------
# API call dispatchers (single request — does NOT handle retries)
# ---------------------------------------------------------------------------

def _call_openai_compat(b64_url: str) -> str:
    """OpenAI-compatible chat-completions endpoint."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": b64_url}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": 50,
        "temperature": 0,
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        if "content" in data:
            return data["content"] or ""
        if "response" in data:
            return str(data["response"])
        return json.dumps(data, ensure_ascii=False)


def _call_gemini(mime_type: str, raw_b64: str) -> str:
    """Gemini native generateContent endpoint (or proxy with same format)."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": PROMPT},
                {"inlineData": {"mimeType": mime_type, "data": raw_b64}},
            ],
        }],
    }
    url = GEMINI_URL

    if GEMINI_API_KEY and "generativelanguage.googleapis.com" in url:
        url += ("&" if "?" in url else "?") + f"key={GEMINI_API_KEY}"
    elif GEMINI_API_KEY:
        headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    inner = data.get("data", data) if isinstance(data, dict) else data
    try:
        return inner["candidates"][0]["content"]["parts"][0]["text"] or ""
    except (KeyError, IndexError, TypeError):
        pass
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Strip → remove all whitespace → uppercase."""
    return re.sub(r"\s+", "", str(text).strip()).upper()


def categorise_error(raw: str) -> str:
    if raw.startswith("API_ERROR:"):
        return "api_error"
    if _REFUSAL_RE.search(raw):
        return "refusal"
    if len(normalize(raw)) != 6:
        return "wrong_length"
    return "wrong_value"


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "…"
    if seconds < 120:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Per-image worker (called by thread pool)
# ---------------------------------------------------------------------------

def _process_one(images_dir: Path, img_name: str, expected: str) -> dict:
    """Encode the image, call the API (with retries), return a result row."""
    img_path = images_dir / img_name

    if not img_path.exists():
        return {
            "image": img_name, "expected": expected,
            "predicted_raw": "FILE_NOT_FOUND", "correct": False,
            "response_time_ms": 0, "error_category": "file_missing",
        }

    raw = ""
    last_exc: Optional[Exception] = None
    t0 = time.perf_counter()

    for attempt in range(1, MAX_RETRIES + 1):
        _wait_rate_limit()                       # honour per-request delay

        t_attempt = time.perf_counter()
        try:
            if API_MODE == "openai_compat":
                mime, raw64 = _encode_image_raw(img_path)
                raw = _call_openai_compat(f"data:{mime};base64,{raw64}")
            else:  # gemini
                mime, raw64 = _encode_image_raw(img_path)
                raw = _call_gemini(mime, raw64)
            last_exc = None
            break
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status == 429:
                wait_s = 2 ** attempt
                print(f"\n  [{img_name}] Rate-limited (429) — waiting {wait_s}s — retry {attempt}/{MAX_RETRIES}")
                time.sleep(wait_s)
                last_exc = exc
                continue
            raw = f"API_ERROR: HTTP {status} — {exc}"
            last_exc = exc
            break
        except Exception as exc:
            raw = f"API_ERROR: {exc}"
            last_exc = exc
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            break

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if last_exc and not raw:
        raw = f"API_ERROR: {last_exc}"

    is_correct = (normalize(raw) == expected)
    category = "" if is_correct else categorise_error(raw)

    return {
        "image": img_name,
        "expected": expected,
        "predicted_raw": raw,
        "correct": is_correct,
        "response_time_ms": int(elapsed_ms),
        "error_category": category,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dataset_path = BASE_DIR / DATASET_CSV
    images_dir   = BASE_DIR / IMAGES_DIR
    output_dir   = BASE_DIR / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load & subset -------------------------------------------------------
    df = pd.read_csv(dataset_path)
    full_count = len(df)

    if MAX_SAMPLES > 0 and MAX_SAMPLES < full_count:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

    total = len(df)
    mode_label = f"{API_MODE}  |  {OPENAI_URL if API_MODE == 'openai_compat' else GEMINI_URL}"
    print(f"Loaded {full_count} samples  →  testing {total}")
    print(f"Mode: {mode_label}  |  Model: {MODEL_NAME}")
    print(f"Concurrency: {CONCURRENCY}  |  Delay: {DELAY_BETWEEN_REQUESTS_MS} ms  |  Max retries: {MAX_RETRIES}\n")

    # --- submit all work -----------------------------------------------------
    results: list[dict] = [{}] * total   # preserve submission order
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures: dict[Future, int] = {}

        for idx, (_, row) in enumerate(df.iterrows()):
            img_name = str(row["image"])
            expected = str(row["captcha_value"]).strip().upper()
            fut = executor.submit(_process_one, images_dir, img_name, expected)
            futures[fut] = idx

        # --- gather, update progress with ETA ---------------------------------
        completed = 0

        # Print initial empty bar
        print(f"\r|{'░' * 30}| 0/{total} (0.0%)  ETA: …", end="", flush=True)

        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            completed += 1

            # ETA estimate from measured throughput
            elapsed = time.perf_counter() - wall_start
            throughput = completed / elapsed if elapsed > 0 else 0
            remaining = total - completed
            eta_s = remaining / throughput if throughput > 0 else 0
            eta_str = _format_eta(eta_s)

            # Progress bar
            pct = completed / total
            width = 30
            filled = int(width * pct)
            bar = "█" * filled + "░" * (width - filled)
            print(f"\r|{bar}| {completed}/{total} ({pct * 100:.1f}%)  ETA: {eta_str}",
                  end="", flush=True)

    print()   # newline after progress bar is done

    # --- write CSV -----------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", MODEL_NAME)
    results_csv  = output_dir / f"benchmark_{safe_name}_{timestamp}.csv"
    metrics_json = output_dir / f"benchmark_{safe_name}_{timestamp}_metrics.json"

    fieldnames = ["image", "expected", "predicted_raw", "correct", "response_time_ms", "error_category"]
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # --- metrics -------------------------------------------------------------
    n         = len(results)
    correct_n = sum(1 for r in results if r["correct"])
    accuracy  = (correct_n / n * 100) if n > 0 else 0

    char_correct = 0
    char_total   = 0
    pos_correct  = {i: 0 for i in range(6)}
    pos_total    = {i: 0 for i in range(6)}

    for r in results:
        npred = normalize(r["predicted_raw"])
        nexp  = normalize(r["expected"])
        max_l = max(len(npred), len(nexp))
        for i in range(max_l):
            char_total += 1
            if i < 6:
                pos_total[i] += 1
            pc = npred[i] if i < len(npred) else ""
            ec = nexp[i] if i < len(nexp) else ""
            if pc == ec:
                char_correct += 1
                if i < 6:
                    pos_correct[i] += 1

    char_accuracy = (char_correct / char_total * 100) if char_total > 0 else 0

    pos_accuracy = {
        i: round(pos_correct[i] / pos_total[i] * 100, 2) if pos_total[i] > 0 else 0
        for i in range(6)
    }

    error_counts: dict[str, int] = {}
    for r in results:
        cat = r["error_category"]
        if cat:
            error_counts[cat] = error_counts.get(cat, 0) + 1

    len_dist: dict[int, int] = {}
    for r in results:
        ln = len(normalize(r["predicted_raw"]))
        len_dist[ln] = len_dist.get(ln, 0) + 1

    times = [r["response_time_ms"] for r in results if r["response_time_ms"] > 0]
    time_stats = {}
    if times:
        sorted_t = sorted(times)
        p95_idx  = int(len(sorted_t) * 0.95)
        time_stats = {
            "min_ms":    round(min(times), 1),
            "max_ms":    round(max(times), 1),
            "mean_ms":   round(statistics.mean(times), 1),
            "median_ms": round(statistics.median(times), 1),
            "p95_ms":    round(sorted_t[min(p95_idx, len(sorted_t) - 1)], 1),
        }

    metrics = {
        "model":                     MODEL_NAME,
        "api_mode":                  API_MODE,
        "endpoint":                  OPENAI_URL if API_MODE == "openai_compat" else GEMINI_URL,
        "concurrency":               CONCURRENCY,
        "timestamp":                 timestamp,
        "samples":                   n,
        "accuracy_exact_match":      round(accuracy, 2),
        "accuracy_character_level":  round(char_accuracy, 2),
        "accuracy_per_position":     pos_accuracy,
        "error_breakdown":           error_counts,
        "response_length_distribution": len_dist,
        "response_time":             time_stats,
    }

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- print summary -------------------------------------------------------
    wall_total = time.perf_counter() - wall_start
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK RESULTS  —  {MODEL_NAME}")
    print(f"{'=' * 60}")
    print(f" Concurrency:             {CONCURRENCY}")
    print(f" Wall-clock:              {wall_total:.0f}s")
    print(f" Samples tested:          {n}")
    print(f" Exact-match accuracy:    {correct_n}/{n} = {accuracy:.2f}%")
    print(f" Character accuracy:      {char_correct}/{char_total} = {char_accuracy:.2f}%")
    print(f"\n Per-position accuracy:")
    for pos in range(6):
        print(f"    Position {pos}:  {pos_correct[pos]}/{pos_total[pos]} = {pos_accuracy[pos]:.2f}%")
    print(f"\n Error breakdown:")
    for cat, cnt in sorted(error_counts.items()):
        print(f"    {cat}:  {cnt}")
    print(f"\n Response length distribution:")
    for ln in sorted(len_dist):
        print(f"    length {ln}:  {len_dist[ln]}")
    if time_stats:
        print(f"\n Response time (ms):")
        print(f"    min={time_stats['min_ms']},  max={time_stats['max_ms']},  "
              f"mean={time_stats['mean_ms']},  median={time_stats['median_ms']},  "
              f"p95={time_stats['p95_ms']}")
    print(f"\n Results CSV:    {results_csv}")
    print(f" Metrics JSON:   {metrics_json}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
