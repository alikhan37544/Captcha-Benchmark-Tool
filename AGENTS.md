# AGENTS.md

## Repo layout

Two independent projects — no shared code between them:

- **V1/** — Benchmarking app: FastAPI backend + React/Vite frontend
- **V2/** — Dataset creation pipeline (Python 3.11)

All commands below are relative to the workspace root unless a `workdir` is specified.

---

## Command Execution Restrictions

**NEVER run:**
- Install commands (`pip install`, `npm install`, `brew install`, `apt-get`, etc.)
- Server / long-running processes (`npm run dev`, `uvicorn`, `docker-compose up`, `jupyter notebook`, etc.)
- Destructive or system-modifying commands (`rm`, `mv`, `chmod`, `git push`, etc.)
- Commands that write outside the project workspace

**Allowed:**
- Build commands (`npm run build`, `vite build`)
- Lint / typecheck commands (`npm run lint`, `eslint`, `tsc`)
- Test commands (`pytest`, `npm test`)
- Read-only / inspection commands (`ls`, glob, grep, `python --version`, `pip list`)
- Version check commands

If a command in a category above is needed, tell the user the exact command to run themselves.

---

## V1 — Benchmarking App

### Structure
```
V1/
├── config.yaml              ← model config (loaded by backend at startup)
├── docker-compose.yml       ← Dockerfiles referenced but not yet created
├── backend/                 ← FastAPI (Python 3.10+)
│   ├── main.py              ← entry point, lifespan, CORS, static mount
│   ├── models/              ← base.py + 6 model implementations
│   ├── routers/             ← benchmark.py (REST), ws.py (WebSocket)
│   ├── config_loader.py     ← YAML + CSV parsing
│   ├── model_loader.py      ← model factory
│   ├── benchmark_runner.py  ← async benchmark engine
│   └── result_store.py      ← JSON/CSV output
└── frontend/                ← React 19 + Vite + TypeScript 5.9
    └── src/
        ├── App.tsx
        ├── hooks/useBenchmarkSocket.ts
        └── components/      ← BenchmarkStatus, LiveInferenceViewer, etc.
```

### Backend

The backend expects `config.yaml` in `V1/` (one level above `backend/`). Set `CONFIG_PATH` env var to override.

The backend serves captcha images via `StaticFiles` mounted at `/captchas`, pointing to the dataset dir from config.

```bash
# From workspace root:
workdir=V1/backend

# Run backend (user must do this):
python main.py
# or: uvicorn main:app --reload --port 8000

# Install deps (user must do this):
pip install -r requirements.txt
```

There are **no tests** in the backend. There is **no lint/typecheck** set up for Python.

### Frontend

```bash
workdir=V1/frontend

# Build (typecheck + vite):
npm run build       # runs: tsc -b && vite build

# Lint:
npm run lint        # runs: eslint .

# Dev server (user must run):
npm run dev         # Vite dev server at :5173

# Install deps (user must run):
npm install
```

The frontend is **not** configured with a Vite proxy — it connects to `http://localhost:8000` directly for API calls. CORS is wide open on the backend.

### Config

`V1/config.yaml` is the single source of truth for models, dataset paths, and prompt. The backend reads it at startup via `config_loader.py`. Dataset expects `captcha_{id}.png` filenames matched against `id` column from `labels.csv`.

---

## V2 — Dataset Pipeline

Requires **Python 3.11** specifically (not 3.10, not 3.12).

### Structure
```
V2/
├── create_dataset.py        ← generates dataset/ground_truth.csv
├── benchmark.py             ← sends images to vision API, records responses
├── results_eda.py           ← analyses benchmark CSVs, single-run or 2-model comparison
├── requirements.txt
├── raw/                     ← user drops files here
│   ├── logs/                ←   CSV exports (needs captchaBlobId + captchaValue cols)
│   └── images/              ←   PNG/JPG captcha files
├── dataset/                 ← auto-created output dir
├── benchmark_runs/           ← benchmark output CSVs + metrics JSON + analysis charts
└── notebooks/eda.ipynb      ← exploratory analysis
```

### Commands

```bash
workdir=V2

# Generate ground truth dataset:
python create_dataset.py

# Custom paths:
python create_dataset.py --logs-dir path/to/logs --images-dir path/to/images --output path/to/output.csv

# Benchmark (edit API_URL / API_KEY / MODEL_NAME at top of script first):
python benchmark.py

# Analyse latest benchmark run:
python results_eda.py

# Compare two specific runs:
python results_eda.py benchmark_runs/run_a.csv benchmark_runs/run_b.csv

# Launch EDA notebook (user must run):
jupyter notebook notebooks/eda.ipynb

# Install deps (user must run):
pip install -r requirements.txt
```

There are **no tests** and **no lint/typecheck** set up for V2.

---

## Environment & gotchas

- **No CI/CD** — no `.github/workflows`, no pre-commit hooks.
- **No tests** in either V1 or V2 — no `pytest`, no `vitest`, no test runner.
- **Dockerfiles** referenced in `docker-compose.yml` do not exist yet.
- **gitignore** covers `data/`, `results/`, `models/*.pth`, `models/*.pt`, `models/*.bin`, and standard Python/Node artifacts.
- V1 backend imports require running from `V1/backend/` directory (relative imports like `from models.base import ...`).
- V1 frontend uses `verbatimModuleSyntax: true` — use `import type` for type-only imports.
