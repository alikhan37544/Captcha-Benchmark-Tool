#!/usr/bin/env python3
"""
Exploratory Data Analysis for benchmark results.

Usage:
    python results_eda.py                              # latest run
    python results_eda.py benchmark_runs/run.csv        # specific run
    python results_eda.py run1.csv run2.csv             # compare two models

Dependencies (install with pip):
    pip install pandas matplotlib numpy pillow
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARK_DIR = "benchmark_runs"
OUTPUT_DIR    = "benchmark_runs/analysis"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Attempt to import seaborn for nicer styling; fall back gracefully
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _extract_model_name(csv_path: Path) -> str:
    """Derive a readable model name from the benchmark CSV filename.
    Filenames are: benchmark_{model_name}_{timestamp}.csv"""
    stem = csv_path.stem                     # benchmark_llava_7b_20260101_120000
    parts = stem.split("_")
    # parts = ['benchmark', 'llava', '7b', '20260101', '120000']
    if len(parts) >= 4:
        # Last two parts are date & time (8-digit + 6-digit)
        if len(parts[-1]) == 6 and parts[-1].isdigit() and len(parts[-2]) == 8 and parts[-2].isdigit():
            return "_".join(parts[1:-2])
    # Fallback: drop 'benchmark_' prefix and last two underscore groups
    return stem.replace("benchmark_", "", 1)


def _find_latest_csvs() -> list[Path]:
    """Return the 1 or 2 most-recent benchmark CSV files."""
    d = BASE_DIR / BENCHMARK_DIR
    csvs = sorted(d.glob("benchmark_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No benchmark_*.csv files found in {d}")
    # Group by base model name (strip timestamp suffix)
    groups: dict[str, list[Path]] = {}
    for p in csvs:
        m = re.match(r"(.+)_\d{8}_\d{6}\.csv$", p.name)
        key = m.group(1) if m else p.stem
        groups.setdefault(key, []).append(p)
    # For each group pick the latest and return at most 2
    latest = [sorted(v, key=lambda x: x.stat().st_mtime)[-1] for v in groups.values()]
    latest = sorted(latest, key=lambda x: x.stat().st_mtime, reverse=True)[:2]
    return latest


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip()).upper()


# ---------------------------------------------------------------------------
# Single-run analysis
# ---------------------------------------------------------------------------

def _analyse_single(df: pd.DataFrame, model_name: str, metrics: Optional[dict], out_dir: Path):
    """Produce charts for a single benchmark run."""
    n = len(df)
    correct_df = df[df["correct"] == True]
    incorrect_df = df[df["correct"] != True]
    correct_n = len(correct_df)
    accuracy = (correct_n / n * 100) if n > 0 else 0

    safe = _sanitise(model_name)

    # --- 1. Accuracy overview ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    char_acc = metrics.get("accuracy_character_level", 0) if metrics else 0
    bars = ax.bar(["Exact Match", "Character Level"],
                  [accuracy, char_acc],
                  color=["#27ae60", "#3498db"])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy Overview — {model_name}")
    fig.savefig(out_dir / f"{safe}_accuracy_overview.png")
    plt.close(fig)

    # --- 2. Per-position accuracy --------------------------------------------
    if metrics and metrics.get("accuracy_per_position"):
        pos = metrics["accuracy_per_position"]
        pos_keys = sorted(pos.keys())
        pos_vals = [pos[k] for k in pos_keys]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([f"Pos {k}" for k in pos_keys], pos_vals, color="#8e44ad")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}%", ha="center", fontweight="bold", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Per-Position Accuracy — {model_name}")
        fig.savefig(out_dir / f"{safe}_per_position.png")
        plt.close(fig)

    # --- 3. Error breakdown --------------------------------------------------
    err = df[df["error_category"] != ""]
    err_counts = err["error_category"].value_counts()
    if len(err_counts) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = {"wrong_value": "#e74c3c", "wrong_length": "#f39c12",
                  "refusal": "#e67e22", "api_error": "#c0392b"}
        bar_colors = [colors.get(c, "#95a5a6") for c in err_counts.index]
        bars = ax.bar(err_counts.index, err_counts.values, color=bar_colors)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(int(bar.get_height())), ha="center", fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_title(f"Error Breakdown — {model_name}")
        fig.savefig(out_dir / f"{safe}_errors.png")
        plt.close(fig)

    # --- 4. Response time distribution ---------------------------------------
    times = df["response_time_ms"].dropna()
    times = times[times > 0]
    if len(times) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(times, bins=30, color="#2c3e50", edgecolor="white", alpha=0.85)
        axes[0].axvline(times.mean(), color="#e74c3c", linestyle="--",
                        label=f"Mean: {times.mean():.0f} ms")
        axes[0].axvline(times.median(), color="#27ae60", linestyle="--",
                        label=f"Median: {times.median():.0f} ms")
        axes[0].set_xlabel("Response Time (ms)")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title(f"Response Time Distribution — {model_name}")
        axes[0].legend()

        # Box plot split by correct/wrong
        correct_times = correct_df["response_time_ms"].dropna()
        incorrect_times = incorrect_df["response_time_ms"].dropna()
        box_data = []
        box_labels = []
        if len(correct_times) > 0:
            box_data.append(correct_times)
            box_labels.append("Correct")
        if len(incorrect_times) > 0:
            box_data.append(incorrect_times)
            box_labels.append("Incorrect")
        if box_data:
            axes[1].boxplot(box_data, labels=box_labels, patch_artist=True,
                            boxprops=dict(facecolor="#3498db", alpha=0.6))
            axes[1].set_ylabel("Response Time (ms)")
            axes[1].set_title(f"Response Time by Correctness — {model_name}")

        fig.savefig(out_dir / f"{safe}_response_times.png")
        plt.close(fig)

    # --- 5. Response length distribution -------------------------------------
    df["pred_len"] = df["predicted_raw"].apply(lambda x: len(normalize(x)))
    len_counts = df["pred_len"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    max_len = max(len_counts.index) if len(len_counts) > 0 else 10
    all_lens = list(range(0, max_len + 1))
    vals = [len_counts.get(ln, 0) for ln in all_lens]
    colors = ["#27ae60" if ln == 6 else "#e74c3c" for ln in all_lens]
    ax.bar(all_lens, vals, color=colors)
    ax.set_xlabel("Normalised Response Length")
    ax.set_ylabel("Count")
    ax.set_title(f"Response Length Distribution — {model_name}  (green = expected length 6)")
    ax.set_xticks(all_lens)
    fig.savefig(out_dir / f"{safe}_length_dist.png")
    plt.close(fig)

    # --- 6. Results table (top-10 correct, top-10 wrong) ---------------------
    sample_rows = []
    top_correct = correct_df.head(5)
    top_wrong = incorrect_df.head(5)
    for _, r in pd.concat([top_correct, top_wrong]).iterrows():
        sample_rows.append(r.to_dict())
    sample_df = pd.DataFrame(sample_rows)
    if len(sample_df) > 0:
        sample_df.to_csv(out_dir / f"{safe}_samples.csv", index=False)

    print(f"\n── {model_name} ──")
    char_acc = metrics.get("accuracy_character_level", 0) if metrics else 0
    print(f"  Samples: {n}  |  Exact: {accuracy:.2f}%  |  Char: {char_acc:.2f}%")
    if metrics and metrics.get("response_time"):
        rt = metrics["response_time"]
        print(f"  Time: mean={rt['mean_ms']:.0f} ms  median={rt['median_ms']:.0f} ms  p95={rt['p95_ms']:.0f} ms")


# ---------------------------------------------------------------------------
# Comparison analysis
# ---------------------------------------------------------------------------

def _analyse_comparison(df_a: pd.DataFrame, name_a: str, ma: Optional[dict],
                        df_b: pd.DataFrame, name_b: str, mb: Optional[dict],
                        out_dir: Path):
    """Side-by-side charts for two models."""
    safe_a = _sanitise(name_a)
    safe_b = _sanitise(name_b)
    base = f"{safe_a}_vs_{safe_b}"

    n_a = len(df_a)
    n_b = len(df_b)
    acc_a = (df_a["correct"].sum() / n_a * 100) if n_a > 0 else 0
    acc_b = (df_b["correct"].sum() / n_b * 100) if n_b > 0 else 0
    char_a = ma["accuracy_character_level"] if ma else 0
    char_b = mb["accuracy_character_level"] if mb else 0

    # --- 1. Accuracy comparison ----------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    bars1 = ax.bar(x - width / 2, [acc_a, acc_b], width, label="Exact Match",
                   color=["#27ae60", "#27ae60"])
    bars2 = ax.bar(x + width / 2, [char_a, char_b], width, label="Character Level",
                   color=["#3498db", "#3498db"])
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([name_a, name_b])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Comparison")
    ax.legend()
    fig.savefig(out_dir / f"{base}_accuracy.png")
    plt.close(fig)

    # --- 2. Per-position accuracy comparison ---------------------------------
    if ma and mb and ma.get("accuracy_per_position") and mb.get("accuracy_per_position"):
        pa = ma["accuracy_per_position"]
        pb = mb["accuracy_per_position"]
        pos_keys = sorted(pa.keys())
        x = np.arange(len(pos_keys))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, [pa[k] for k in pos_keys], width, label=name_a, color="#8e44ad")
        ax.bar(x + width / 2, [pb[k] for k in pos_keys], width, label=name_b, color="#f39c12")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Pos {k}" for k in pos_keys])
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Per-Position Accuracy Comparison")
        ax.legend()
        fig.savefig(out_dir / f"{base}_per_position.png")
        plt.close(fig)

    # --- 3. Error breakdown comparison ---------------------------------------
    err_a = df_a[df_a["error_category"] != ""]["error_category"].value_counts()
    err_b = df_b[df_b["error_category"] != ""]["error_category"].value_counts()
    all_cats = sorted(set(list(err_a.index) + list(err_b.index)))
    if all_cats:
        x = np.arange(len(all_cats))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, [err_a.get(c, 0) for c in all_cats], width, label=name_a, color="#e74c3c")
        ax.bar(x + width / 2, [err_b.get(c, 0) for c in all_cats], width, label=name_b, color="#f39c12")
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats)
        ax.set_ylabel("Count")
        ax.set_title("Error Breakdown Comparison")
        ax.legend()
        fig.savefig(out_dir / f"{base}_errors.png")
        plt.close(fig)

    # --- 4. Response time comparison (box plots) -----------------------------
    times_a = df_a["response_time_ms"].dropna()
    times_a = times_a[times_a > 0]
    times_b = df_b["response_time_ms"].dropna()
    times_b = times_b[times_b > 0]
    if len(times_a) > 0 and len(times_b) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([times_a, times_b], labels=[name_a, name_b], patch_artist=True,
                   boxprops=dict(facecolor="#3498db", alpha=0.6))
        ax.set_ylabel("Response Time (ms)")
        ax.set_title("Response Time Comparison")
        fig.savefig(out_dir / f"{base}_response_times.png")
        plt.close(fig)

    # --- 5. Response length distribution comparison --------------------------
    len_a = df_a["predicted_raw"].apply(lambda x: len(normalize(x)))
    len_b = df_b["predicted_raw"].apply(lambda x: len(normalize(x)))
    max_l = max(len_a.max(), len_b.max()) if len(len_a) > 0 and len(len_b) > 0 else 20
    all_lens = list(range(0, int(max_l) + 1))
    a_counts = [int((len_a == ln).sum()) for ln in all_lens]
    b_counts = [int((len_b == ln).sum()) for ln in all_lens]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors_a = ["#27ae60" if ln == 6 else "#e74c3c" for ln in all_lens]
    colors_b = ["#27ae60" if ln == 6 else "#e74c3c" for ln in all_lens]
    axes[0].bar(all_lens, a_counts, color=colors_a)
    axes[0].set_title(f"Response Length — {name_a}")
    axes[0].set_xlabel("Normalised Length")
    axes[0].set_xticks(all_lens)
    axes[1].bar(all_lens, b_counts, color=colors_b)
    axes[1].set_title(f"Response Length — {name_b}")
    axes[1].set_xlabel("Normalised Length")
    axes[1].set_xticks(all_lens)
    fig.suptitle("Response Length Distribution Comparison (green = expected length 6)")
    fig.savefig(out_dir / f"{base}_length_dist.png")
    plt.close(fig)

    # --- Print comparison summary --------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"COMPARISON:  {name_a}  vs  {name_b}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {name_a:<20} {name_b:<20}")
    print(f"{'-' * 70}")
    print(f"{'Exact-match accuracy':<30} {acc_a:<20.2f}% {acc_b:<20.2f}%")
    print(f"{'Character accuracy':<30} {char_a:<20.2f}% {char_b:<20.2f}%")
    if ma and mb and ma.get("response_time") and mb.get("response_time"):
        rta = ma["response_time"]
        rtb = mb["response_time"]
        print(f"{'Mean response time (ms)':<30} {rta['mean_ms']:<20.0f} {rtb['mean_ms']:<20.0f}")
        print(f"{'Median response time (ms)':<30} {rta['median_ms']:<20.0f} {rtb['median_ms']:<20.0f}")
        print(f"{'P95 response time (ms)':<30} {rta['p95_ms']:<20.0f} {rtb['p95_ms']:<20.0f}")
    print(f"{'API errors':<30} {int((df_a['error_category'] == 'api_error').sum()):<20} "
          f"{int((df_b['error_category'] == 'api_error').sum()):<20}")
    print(f"{'Refusals':<30} {int((df_a['error_category'] == 'refusal').sum()):<20} "
          f"{int((df_b['error_category'] == 'refusal').sum()):<20}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = BASE_DIR / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine which CSV(s) to load --------------------------------------
    csv_paths: list[Path] = []

    if len(sys.argv) >= 3:
        # Manual: two CSV paths given for comparison
        csv_paths = [Path(p) for p in sys.argv[1:3]]
    elif len(sys.argv) == 2:
        # Manual: single CSV path
        csv_paths = [Path(sys.argv[1])]
    else:
        # Auto-detect latest runs
        csv_paths = _find_latest_csvs()
        if len(csv_paths) == 2:
            print(f"Auto-detected 2 runs — comparison mode:")
            for p in csv_paths:
                print(f"  {p}")
        else:
            print(f"Auto-detected latest run: {csv_paths[0] if csv_paths else 'none'}")

    if not csv_paths:
        print("No benchmark CSVs found.")
        return

    # --- Load data -----------------------------------------------------------
    dfs: list[pd.DataFrame] = []
    names: list[str] = []
    metricss: list[Optional[dict]] = []

    for p in csv_paths:
        if not p.exists():
            print(f"ERROR: file not found — {p}")
            return
        df = pd.read_csv(p)
        name = _extract_model_name(p)

        # Try loading companion metrics JSON
        metrics = None
        json_path = p.with_name(p.stem + "_metrics.json")
        if json_path.exists():
            with open(json_path) as jf:
                metrics = json.load(jf)

        dfs.append(df)
        names.append(name)
        metricss.append(metrics)

    # --- Analyse -------------------------------------------------------------
    if len(dfs) == 1:
        _analyse_single(dfs[0], names[0], metricss[0], out_dir)
    else:
        _analyse_single(dfs[0], names[0], metricss[0], out_dir)
        _analyse_single(dfs[1], names[1], metricss[1], out_dir)
        _analyse_comparison(dfs[0], names[0], metricss[0],
                            dfs[1], names[1], metricss[1], out_dir)

    print(f"\nCharts saved to: {out_dir}")


if __name__ == "__main__":
    main()
