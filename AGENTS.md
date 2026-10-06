# ⚙️ Captcha-Benchmark-Tool - Agent Guide & Repository Manual

This repository contains tools (V1 and V2) designed for the **benchmarking, dataset generation, and analysis of CAPTCHA recognition models**. It is structured into two independent, yet related, components.

## 🎯 Overview and Goal
The primary objective is to build a robust platform to:
1.  **Generate Ground Truth Data (V2):** Process raw inputs and generate labeled datasets (`.csv`) and analysis figures from raw images.
2.  **Benchmark Models:** Run model inferences against these controlled datasets and record performance metrics in structured results.
3.  **Visualize/Report (V1):** Provide a web interface to visualize the benchmark process, view live results, and manage the overall workflow.

## 📁 Repository Structure & Components

The repository is split into two main directories: `V1` (the service application) and `V2` (the data pipeline).

### 💻 V1: Benchmarking Application (`V1/`)
This is the live FastAPI backend with a React frontend that provides the user interface for running and viewing benchmarks.

#### 📂 Structure Overview
```
V1/
├── config.yaml              # <<< PRIMARY CONFIG SOURCE: Must be updated for dataset paths, model names, and general settings.
├── docker-compose.yml       # References Docker setup (requires manual configuration)
├── backend/                 # FastAPI Backend (Python 3.10+)
│   ├── main.py              # Entry point: Initializes FastAPI, loads config, defines API routes.
│   ├── models/              # Model implementation base classes and concrete model wrappers.
│   ├── routers/             # REST and WebSocket endpoints for the client interaction.
│   ├── config_loader.py     # Handles loading configuration from YAML/CSV files.
│   └── benchmark_runner.py  # Core async engine that orchestrates API calls to models.
└── frontend/                # React/Vite Client (TypeScript 5.9)
    ├── src/
    │   └── components/      # UI elements: Status views, result displays, etc.
```

#### ⚙️ V1 Workflow & Setup Instructions

**🛑 CRITICAL RESTRICTIONS:** Never run build/install commands or long-running processes directly in the terminal if you need to pass them to a user. Always state the command for the user.

*   **Prerequisites:** Node.js (for frontend) and Python 3.10+ (for backend).
*   **Setup Dependencies:**
    ```bash
    # From workspace root:
    cd V1/backend
    pip install -r requirements.txt
    # Next, set up the frontend dependencies in a separate terminal session:
    cd ../frontend
    npm install
    ```
*   **Run Services (Manual Execution Required):**
    ```bash
    # 1. Start Backend (In Terminal 1)
    (cd V1/backend && uvicorn main:app --reload --port 8000)

    # 2. Start Frontend (In Terminal 2)
    (cd V1/frontend && npm run dev) # Usually runs at http://localhost:5173
    ```
*   **Key Points:**
    *   The backend serves captcha images via a static mount (`/captchas`).
    *   API calls from the frontend are configured to point directly to `http://localhost:8000`.

---

### 🐍 V2: Dataset Generation Pipeline (`V2/`)
This is the core data processing engine, requiring specific dependencies and Python versions. It should be run *before* using V1 for fresh benchmarks.

#### 📂 Structure Overview
```
V2/
├── create_dataset.py        # Script to generate the ground truth labels (.csv) from raw image pools.
├── benchmark.py             # Main script: Iterates over images, calls external APIs (e.g., vision model), and records responses into a result CSV.
├── results_eda.py           # Analysis tool: Reads multiple run CSVs to generate comparative reports or metrics.
├── requirements.txt        # Python dependencies for V2 scripts (Requires Python 3.11).
├── raw/                     # Data drop zone: Contains source images and API logs.
│   └── images/              # PNG/JPG CAPTCHA files used as input.
└── dataset/                 # Output folder: Stores derived labeled data (e.g., ground_truth.csv, distribution plots).
```

#### ⚙️ V2 Workflow & Execution Steps

1.  **Check Environment:** Ensure Python 3.11 is active (`python --version`).
2.  **Setup Dependencies:**
    ```bash
    # From workspace root:
    pip install -r requirements.txt
    ```
3.  **Generate Ground Truth (Pre-requisite for Benchmarking):** This step uses the raw images to create a labeled dataset file.
    ```bash
    python V2/create_dataset.py --logs-dir path/to/raw_api_logs --images-dir path/to/raw_images --output data/ground_truth.csv
    ```
4.  **Run Benchmark:** Use the created dataset to test models against the raw images. *Requires API keys and model names to be edited at the top of `benchmark.py`.*
    ```bash
    python V2/benchmark.py
    # This saves results to V2/benchmark_runs/latest_run.csv
    ```
5.  **Analyze Results:** Use this script for comparative analysis or generating final reports.
    ```bash
    # Example: comparing 'run_A' and 'run_B'
    python V2/results_eda.py benchmark_runs/run_A.csv benchmark_runs/run_B.csv
    ```

## ⚠️ Critical Development Guidelines (Pitfalls)

1.  **Environment Isolation:** Do not mix commands between V1 and V2 without changing the working directory (`cd`). They use different dependencies and execution flows.
2.  **Data Flow:** **V2 $\rightarrow$ V1**. Always generate data in V2 first, then point V1's `config.yaml` to the resulting dataset/results.
3.  **Testing:** This repository currently lacks unit or integration tests. All new functionality must be accompanied by manual validation steps.

***

*This document was generated based on a deep analysis of the existing codebase structure and intended purpose.*