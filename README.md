<div align="center">

# ⚡ Captcha Benchmark Tool

**A real-time benchmark harness for OCR and Vision-Language models against labeled captcha datasets.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript)](https://typescriptlang.org)

</div>

---

## Overview

Run **any number of captcha-solving models** against your labeled dataset simultaneously. The frontend shows you **live inference** — the current image, what each model predicted, whether it was right — along with real-time accuracy charts updating after every single prediction.

### Supported Model Types

| Type | Description | Example Models |
|------|-------------|---------------|
| `easyocr` | EasyOCR multi-language OCR | — |
| `tesseract` | Tesseract OCR (with image preprocessing) | — |
| `huggingface` | 🤗 Transformers — OCR models and VLMs | dots.ocr-1.5, GLM-OCR, SmolVLM2, Qwen2.5-VL-7B, MiniCPM-V-2.6, Moondream2, olmOCR-2, nanoVLM, LFM2-VL-450M |
| `lmstudio` | LMStudio local vision server (no GPU code needed) | Any model loaded in LMStudio |
| `openai_compat` | Any OpenAI-compat REST API with vision | Ollama, OpenRouter, vLLM, Groq |
| `custom` | Custom PyTorch checkpoint + CTC decoder | Your own trained model |

---

## Prerequisites

### System

- Python **3.10+**
- Node.js **18+**
- *(For Tesseract)* `tesseract-ocr` system binary:
  ```bash
  # Ubuntu / Debian
  sudo apt install tesseract-ocr

  # Arch
  sudo pacman -S tesseract tesseract-data-eng

  # macOS
  brew install tesseract
  ```
- *(For GPU acceleration)* CUDA 11.8+ with matching PyTorch build

---

## Quick Start

### 1 — Prepare your dataset

```
Captcha-Benchmark-Tool/
└── data/
    ├── captchas/
    │   ├── captcha_1.png
    │   ├── captcha_2.png
    │   └── ...
    └── labels.csv
```

**`labels.csv` format** — only `id` and `value` are used, extra columns are ignored:

```csv
id,value,blobname,createdat,updatedat
1,A3F2,blob_001.png,2024-01-01,2024-01-01
2,XK9P,blob_002.png,2024-01-01,2024-01-01
```

The tool maps `id → captcha_{id}.png` automatically.

### 2 — Enable models in `config.yaml`

Open `config.yaml` in the project root. Set `enabled: true` on the models you want to run:

```yaml
# Tip: start with just EasyOCR and Tesseract for a quick sanity check
models:
  - name: "EasyOCR"
    type: easyocr
    enabled: true

  - name: "Tesseract"
    type: tesseract
    psm: 8          # 8 = single word, 7 = single line, 6 = block
    enabled: true

  - name: "SmolVLM2"
    type: huggingface
    model_id: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    enabled: true   # Will download ~4GB on first run
```

### 3 — Set up and start the backend

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
# or: uvicorn main:app --reload --port 8000
```

Backend is now running at **http://localhost:8000**
Interactive API docs: **http://localhost:8000/docs**

### 4 — Set up and start the frontend

```bash
cd frontend

npm install        # first time only
npm run dev
```

Frontend is now running at **http://localhost:5173**

### 5 — Run a benchmark

Open **http://localhost:5173** in your browser and click **▶ Start Benchmark**.

---

## Configuration Reference

Full `config.yaml` options:

```yaml
# Path to the directory containing captcha_{id}.png files
dataset_dir: ./data/captchas

# CSV file with id,value columns
labels_file: ./data/labels.csv

# Where to write result JSON and CSV files after each run
output_dir: ./results

# Prompt sent to VLM / LLM-based models alongside the image
prompt: "What text is shown in this captcha image? Reply with ONLY the text characters, no punctuation, no explanation."

models:
  # ─── EasyOCR ───────────────────────────────────────────────
  - name: "EasyOCR"
    type: easyocr
    enabled: true

  # ─── Tesseract ─────────────────────────────────────────────
  - name: "Tesseract"
    type: tesseract
    psm: 8          # Page Segmentation Mode (6, 7, or 8 work well for captchas)
    enabled: true

  # ─── HuggingFace (OCR models and VLMs) ─────────────────────
  - name: "SmolVLM2"
    type: huggingface
    model_id: HuggingFaceTB/SmolVLM2-2.2B-Instruct
    enabled: false

  - name: "Qwen2.5-VL-7B"
    type: huggingface
    model_id: Qwen/Qwen2.5-VL-7B-Instruct
    enabled: false

  # ─── LMStudio ──────────────────────────────────────────────
  - name: "MyLMStudioModel"
    type: lmstudio
    base_url: http://localhost:1234/v1
    model_id: "model-name-as-shown-in-lmstudio"
    enabled: false

  # ─── Generic OpenAI-compat (Ollama, OpenRouter, vLLM…) ─────
  - name: "OllamaLlava"
    type: openai_compat
    base_url: http://localhost:11434/v1
    model_id: "llava:7b"
    api_key: "ollama"
    enabled: false

  # ─── Custom PyTorch ────────────────────────────────────────
  - name: "MyCNN"
    type: custom
    model_path: ./models/cnn_model.pth
    vocab: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    enabled: false
```

---

## All Supported HuggingFace Models

| Model Name | `model_id` | Params | Notes |
|------------|-----------|--------|-------|
| dots.ocr-1.5 | `rednote-hilab/dots.ocr-1.5` | 3.0B | 2026 SOTA document/scene OCR |
| GLM-OCR | `zai-org/GLM-OCR` | 0.9B | Hyper-efficient structured OCR |
| ocr-captcha-v3 | `anuashok/ocr-captcha-v3` | 0.3B | Fine-tuned on wavy/distorted captchas |
| SmolVLM2 | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | 2.2B | Balanced speed/reasoning |
| LFM2-VL-450M | `LiquidAI/LFM2-VL-450M` | 0.45B | Extremely fast, variable resolution |
| MiniCPM-V-2.6 | `openbmb/MiniCPM-V-2_6` | 8.0B | SOTA OCRBench, high-res support |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` | 7.0B | Strong multilingual VLM |
| olmOCR-2 | `allenai/olmOCR-2-7B-1025` | 7.0B | Document → text/markdown |
| Moondream2 | `vikhyatk/moondream2` | 2.0B | Fast, small footprint |
| ocr-for-captcha | `keras-io/ocr-for-captcha` | 0.3B | CNN-RNN alphanumeric captchas |
| nanoVLM | `huggingface/nanoVLM` | 0.22B | Minimalist, extreme speed |

---

## Output Files

After each benchmark run two files are saved to `./results/`:

### `results_YYYY-MM-DDTHH-MM-SS.json`
```json
{
  "run_id": "2026-02-19T07-05-40",
  "dataset": { "total": 500, "labels_file": "data/labels.csv" },
  "models": {
    "EasyOCR": {
      "accuracy": 0.82,
      "correct": 410,
      "total": 500,
      "duration_s": 34.5,
      "predictions": [
        { "id": "1", "file": "captcha_1.png", "expected": "A3F2", "predicted": "A3F2", "correct": true }
      ]
    }
  }
}
```

### `results_YYYY-MM-DDTHH-MM-SS.csv`
Flat format, one row per prediction per model:

```
run_id,model,id,file,expected,predicted,correct,model_accuracy
2026-02-19T07-05-40,EasyOCR,1,captcha_1.png,A3F2,A3F2,True,0.82
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/benchmark/start` | Start a benchmark run |
| `POST` | `/benchmark/stop` | Cancel the current run |
| `GET` | `/benchmark/status` | Current run state, progress, accuracy |
| `GET` | `/benchmark/results` | Full results of last completed run |
| `GET` | `/benchmark/models` | All configured models + enabled state |
| `WS` | `/ws/progress` | Live event stream (WebSocket) |
| `GET` | `/captchas/{filename}` | Serve a captcha image |
| `GET` | `/docs` | Interactive Swagger API docs |

### WebSocket Events

Connect to `ws://localhost:8000/ws/progress`. Events are JSON objects:

```js
// Fired for every image processed
{ "event": "progress", "model": "EasyOCR", "id": "1", "image": "captcha_1.png",
  "predicted": "A3F2", "expected": "A3F2", "correct": true,
  "model_progress": 0.42, "overall_progress": 0.21,
  "accuracy": { "EasyOCR": 0.82 }, "eta_s": 45.2 }

{ "event": "model_loading", "model": "SmolVLM2" }   // weight download/load started
{ "event": "model_loaded",  "model": "SmolVLM2" }   // inference ready
{ "event": "model_done",    "model": "SmolVLM2", "accuracy": 0.74, "correct": 370, "total": 500 }
{ "event": "done",   "run_id": "...", "accuracy": { ... }, "duration_s": 120.5 }
{ "event": "cancelled" }
{ "event": "error",  "detail": "..." }
{ "event": "heartbeat" }                             // keepalive every 2s
```

---

## Tips & Troubleshooting

### HuggingFace models downloading slowly
Models are cached in `~/.cache/huggingface/hub/` after the first download. Run `huggingface-cli download <model_id>` in advance to pre-cache.

### GPU out-of-memory on large models
- Enable only one large model at a time in `config.yaml`
- The runner unloads each model before loading the next, so peak VRAM = single model
- For 7B+ models you need at least 16GB VRAM (with bfloat16)

### Tesseract not found
```bash
# Verify the tesseract binary is on PATH:
tesseract --version
# If not, set the path in code by adding to tesseract_model.py:
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
```

### LMStudio connection refused
1. Open LMStudio → Local Server tab → Load a model → Start server
2. Confirm `base_url` in `config.yaml` matches (default: `http://localhost:1234/v1`)
3. Verify the model supports vision (look for `vision` in the model card)

### Frontend can't reach backend
The frontend expects the backend at `http://localhost:8000`. Make sure:
- Both servers are running
- No firewall is blocking port 8000
- CORS is unrestricted (it is, by default)

---

## Project Structure

```
Captcha-Benchmark-Tool/
├── config.yaml                    ← Model list, paths, prompt
├── README.md
├── docker-compose.yml
├── .gitignore
│
├── data/                          ← Your dataset goes here (gitignored)
│   ├── captchas/
│   │   └── captcha_{id}.png
│   └── labels.csv
│
├── results/                       ← Output JSON + CSV (gitignored)
│
└── backend/
│   ├── main.py                    ← FastAPI app (lifespan, CORS, static files)
│   ├── config_loader.py           ← YAML + CSV parser
│   ├── model_loader.py            ← Model factory
│   ├── benchmark_runner.py        ← Async benchmark engine
│   ├── result_store.py            ← JSON/CSV writer
│   ├── requirements.txt
│   ├── models/
│   │   ├── base.py                ← Abstract base class
│   │   ├── easyocr_model.py
│   │   ├── tesseract_model.py     ← Includes image preprocessing
│   │   ├── huggingface_model.py   ← bfloat16, flash_attention_2, VLM+OCR
│   │   ├── lmstudio_model.py      ← Vision via OpenAI SDK
│   │   ├── openai_compat_model.py ← Ollama / OpenRouter / vLLM
│   │   └── custom_model.py        ← PyTorch + CTC decoder
│   └── routers/
│       ├── benchmark.py           ← REST endpoints
│       └── ws.py                  ← WebSocket + ConnectionManager
│
└── frontend/
    └── src/
        ├── App.tsx                ← Main layout
        ├── index.css              ← Dark design system
        ├── hooks/
        │   └── useBenchmarkSocket.ts  ← WS + API calls
        └── components/
            ├── BenchmarkStatus.tsx    ← Progress bars, ETA, controls
            ├── LiveInferenceViewer.tsx← Image + prediction flash
            ├── AccuracyChart.tsx      ← recharts bar chart
            ├── ModelList.tsx          ← Model sidebar
            └── ResultsTable.tsx       ← Sortable, filterable results
```
