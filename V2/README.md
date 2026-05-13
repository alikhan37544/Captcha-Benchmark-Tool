# Captcha Benchmark Tool - V2

Ground truth dataset creation and analysis pipeline for captcha benchmarking.

## Project Structure

```
Captcha-Benchmark-Tool/
├── V1/                        # Legacy code (backend, frontend, etc.)
└── V2/                        # V2 - Dataset & Analysis Pipeline
    ├── README.md               # This file
    ├── create_dataset.py       # Step 1: Generate ground_truth.csv
    ├── requirements.txt        # Python dependencies
    ├── raw/                    # Raw data (drop files here)
    │   ├── logs/               #   CSV log exports from database
    │   └── images/             #   Downloaded captcha images
    ├── dataset/                # Generated datasets (auto-created)
    │   └── ground_truth.csv    #   Filtered dataset (image, captcha_value)
    └── notebooks/              # Jupyter notebooks
        └── eda.ipynb           #   Exploratory data analysis
```

## Setup

Requires **Python 3.11**:

```bash
# 1. Create virtual environment (inside V2/)
python3.11 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Or install directly from the EDA notebook — it includes an auto-install cell with pinned versions:

## Workflow

### Step 1: Prepare Raw Data

1. **Drop database logs** into `raw/logs/` — CSV exports from the captcha logging database. Must contain columns `captchaBlobId` and `captchaValue`. Any number of CSV files can be placed here; they will all be processed.

2. **Drop captcha images** into `raw/images/` — PNG/JPG image files where the filename matches the `captchaBlobId` column in the logs (e.g., `captcha_4921.png`).

### Step 2: Create Ground Truth Dataset

```bash
# Default: reads from raw/logs/ and raw/images/, outputs to dataset/ground_truth.csv
python create_dataset.py

# Custom paths
python create_dataset.py --logs-dir path/to/logs --images-dir path/to/images --output path/to/output.csv
```

This filters the log data to only keep rows where the corresponding image file exists on disk, producing a clean `ground_truth.csv` with two columns:
- `image` — the image filename (e.g., `captcha_4921.png`)
- `captcha_value` — the ground truth text (e.g., `Y4MGWA`)

### Step 3: Exploratory Data Analysis

```bash
# Run from V2/ directory with venv activated
jupyter notebook notebooks/eda.ipynb
```

The EDA notebook covers:
- Data quality checks (image-dataset sync verification)
- Captcha length distribution
- Character frequency analysis (digits vs letters, positional analysis)
- Duplicate captcha value analysis
- Image property analysis (dimensions, file sizes)
- Sample image visualization
- Duplicate value visual comparison

### Step 4 (Upcoming): Benchmarking

The ground truth dataset will be used to benchmark captcha-solving models. This step is TBD.

## Rebuilding the Dataset

If you add more images or logs, simply re-run:

```bash
# With venv activated:
python create_dataset.py
```

This will re-scan `raw/logs/` and `raw/images/` and regenerate `dataset/ground_truth.csv`.

## Data Format Reference

### Raw Logs CSV (input)

| id | captchaBlobId | captchaValue | status | portal | createdAt | updatedAt |
|----|---------------|-------------|--------|--------|-----------|-----------|
| 1  | captcha_1.png | 52N6HX      | SUCCESS| PMJAY  | 2026-02-19| 2026-02-19|

### Ground Truth CSV (output)

| image           | captcha_value |
|-----------------|---------------|
| captcha_1.png   | 52N6HX        |
| captcha_2.png   | KR5WPN        |

## Current Dataset Stats

- **1380** samples
- All images are **240x70 RGB**
- Captcha values are **6 characters** (dominantly), composed of 7 digits (2-8) and 20 uppercase letters (A-Z excluding I, O, U, and with K, S, V included)
- **929** unique captcha values, with duplicates sharing the same text but different images