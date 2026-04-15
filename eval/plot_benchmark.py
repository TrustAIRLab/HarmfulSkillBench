#!/usr/bin/env python3
"""
HarmSkillBench Benchmark Figures.

Category x Model harm-score heatmaps for Conditions A, B, and D
(the no-skill baseline).

Usage:
    cd github_repo
    python3 eval/plot_benchmark.py
"""

from __future__ import annotations

import os
import json
import glob
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# =========================
# Utils
# =========================
def check_and_create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


# =========================
# Global plotting style
# =========================
FONTSIZE = -1
legend_FONTSIZE = -1
ticklabel_FONTSIZE = -1

LINEWIDTH = 4
MARKERSIZE = 10

sns.set_style("dark")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})


def GlobalSettings(fig_per_column: int = 2) -> None:
    global FONTSIZE, legend_FONTSIZE, ticklabel_FONTSIZE
    FONTSIZE = 18 * fig_per_column
    legend_FONTSIZE = 18 * fig_per_column
    ticklabel_FONTSIZE = 18 * fig_per_column


# =========================
# Palette
# =========================
current_palette = list(sns.color_palette("deep"))
if len(current_palette) > 1:
    del current_palette[1]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=current_palette)

# =========================
# Output directories
# =========================
OUT_PNG_DIR = "Figure/png/"
OUT_PDF_DIR = "Figure/pdf/"
check_and_create_directory(OUT_PNG_DIR)
check_and_create_directory(OUT_PDF_DIR)

# =========================
# Apply settings
# =========================
GlobalSettings(fig_per_column=2)

# =========================
# Custom colors
# =========================
ALPHA = 0.85
_raw_orange = (0.95, 0.60, 0.35)
_raw_teal = (0.35, 0.75, 0.70)
COLOR_ORANGE = (*_raw_orange, ALPHA)
COLOR_TEAL = (*_raw_teal, ALPHA)

# =========================
# Shared colormap
# =========================
CMAP_ORANGE = LinearSegmentedColormap.from_list("orange_seq", [
    (0.97, 0.97, 0.95),
    (0.98, 0.82, 0.65),
    _raw_orange,
    (0.75, 0.35, 0.15),
])

# =========================
# Data paths
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGMENTS_DIR = os.path.join(PROJECT_ROOT, "data", "eval_results", "judgments")

# =========================
# Model & category config
# =========================
MODEL_ORDER = ["gpt-4o", "gpt-5.4-mini", "gemini-3-flash", "deepseek-v3.2", "qwen3-235b", "kimi-k2.5"]
MODEL_DISPLAY = {
    "gpt-4o": "GPT-4o",
    "gpt-5.4-mini": "GPT-5.4-Mini",
    "gemini-3-flash": "Gemini 3 Flash",
    "deepseek-v3.2": "DeepSeek V3.2",
    "qwen3-235b": "Qwen3-235B",
    "kimi-k2.5": "Kimi K2.5",
}

TIER1_CATS = [f"A{i}" for i in range(1, 15) if i != 7]  # A7 excluded (no data)
TIER2_CATS = [f"AH{i}" for i in range(1, 8)]
ALL_CATS = TIER1_CATS + TIER2_CATS

# Display labels: P1-P14 for Tier 1, H1-H7 for Tier 2.
CAT_DISPLAY: dict[str, str] = {}
for c in TIER1_CATS:
    CAT_DISPLAY[c] = "P" + c[1:]
for c in TIER2_CATS:
    CAT_DISPLAY[c] = "H" + c[2:]


# =========================
# Load data
# =========================
def load_all_judgments() -> list[dict]:
    results = []
    for f in glob.glob(os.path.join(JUDGMENTS_DIR, "*", "*", "*.json")):
        with open(f, encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


# =========================
# Figure: Category x Model Score Heatmap
# =========================
def plot_heatmap(judgments: list[dict], condition: str) -> None:
    matrix = np.full((len(ALL_CATS), len(MODEL_ORDER)), np.nan)
    for ci, cat in enumerate(ALL_CATS):
        for mi, model in enumerate(MODEL_ORDER):
            subset = [j for j in judgments
                      if j["model"] == model and j["condition"] == condition and j["category"] == cat]
            if subset:
                scores = [j["score"] for j in subset if j.get("score") is not None]
                matrix[ci, mi] = np.mean(scores) if scores else 0

    fig, ax = plt.subplots(figsize=(9, 12))

    im = ax.imshow(matrix, cmap=CMAP_ORANGE, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(
        [MODEL_DISPLAY[m] for m in MODEL_ORDER],
        fontsize=ticklabel_FONTSIZE * 0.61, rotation=45, ha="center",
    )
    ax.set_yticks(range(len(ALL_CATS)))
    ax.set_yticklabels(
        [CAT_DISPLAY[c] for c in ALL_CATS],
        fontsize=ticklabel_FONTSIZE * 0.61,
    )

    for ci in range(len(ALL_CATS)):
        for mi in range(len(MODEL_ORDER)):
            val = matrix[ci, mi]
            if not np.isnan(val):
                color = "white" if val > 0.55 else "black"
                ax.text(mi, ci, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=ticklabel_FONTSIZE * 0.48,
                        fontweight="bold", color=color)

    tier1_end = len(TIER1_CATS) - 0.5
    ax.axhline(y=tier1_end, color="white", linewidth=2.5, linestyle="-")

    ax.text(-1.5, len(TIER1_CATS) / 2 - 0.5, "Tier 1\n(Prohibited)",
            ha="center", va="center", fontsize=FONTSIZE * 0.61,
            fontweight="bold", rotation=90, clip_on=False)
    ax.text(-1.5, tier1_end + len(TIER2_CATS) / 2 + 0.5, "Tier 2\n(High-Risk)",
            ha="center", va="center", fontsize=FONTSIZE * 0.61,
            fontweight="bold", rotation=90, clip_on=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03)
    cbar.set_label("Score", fontsize=FONTSIZE * 0.67)
    cbar.ax.tick_params(labelsize=ticklabel_FONTSIZE * 0.51)

    plt.tight_layout()
    fname = f"heatmap_score_cond{condition}"
    plt.savefig(OUT_PNG_DIR + fname + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF_DIR + fname + ".pdf", bbox_inches="tight")
    print(f"  {fname} saved.")
    plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("Loading judgments...")
    judgments = load_all_judgments()
    print(f"Loaded {len(judgments)} judgments.")

    judgments = [j for j in judgments if j["model"] in MODEL_ORDER]
    print(f"Filtered to {len(judgments)} (6 models).")

    # Sanity checks: A, B, D each should have 1200 (200 * 6) judgments,
    # per (model, cat) cell = 10.
    conds_seen = set(j["condition"] for j in judgments)
    for req in ("A", "B", "D"):
        if req not in conds_seen:
            raise SystemExit(f"Missing required condition {req}; cannot plot heatmap.")
    for cond in ("A", "B", "D"):
        n = sum(1 for j in judgments if j["condition"] == cond)
        if n != 1200:
            raise SystemExit(f"Condition {cond} has {n} judgments, expected 1200.")
        cell_counts = Counter(
            (j["model"], j["category"]) for j in judgments if j["condition"] == cond
        )
        uniq = set(cell_counts.values())
        if uniq != {10}:
            raise SystemExit(f"Condition {cond} per-cell counts = {uniq}, expected {{10}}.")
    print("  Sanity checks passed.")

    print("\nHeatmaps:")
    plot_heatmap(judgments, "A")
    plot_heatmap(judgments, "B")
    plot_heatmap(judgments, "D")

    print("\nAll figures generated.")
