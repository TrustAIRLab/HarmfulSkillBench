"""
HarmSkillBench Analysis

Computes metrics from judge results across multiple dimensions.

Usage:
    python3 eval/analyze.py                           # All models
    python3 eval/analyze.py --models gpt-4o           # Single model
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGMENTS_DIR = PROJECT_ROOT / "data" / "eval_results" / "judgments"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_judgments(models=None, conditions=None):
    results = []
    for f in glob.glob(str(JUDGMENTS_DIR / "*" / "*" / "*.json")):
        with open(f, encoding="utf-8") as fh:
            j = json.load(fh)
        if models and j["model"] not in models:
            continue
        if conditions and j["condition"] not in conditions:
            continue
        results.append(j)
    return results


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def compute_metrics(items):
    n = len(items)
    if n == 0:
        return {}

    labels = Counter(j["label"] for j in items)
    scores = [j["score"] for j in items if j.get("score") is not None]
    harm = [j["harmfulness"] for j in items if j.get("harmfulness") is not None]

    return {
        "n": n,
        "refuse": labels.get("refuse", 0),
        "comply": labels.get("comply", 0),
        "parse_error": labels.get("parse_error", 0) + labels.get("api_error", 0),
        "refuse_rate": labels.get("refuse", 0) / n * 100,
        "comply_rate": labels.get("comply", 0) / n * 100,
        "mean_score": sum(scores) / len(scores) if scores else 0,
        "mean_harm": sum(harm) / len(harm) if harm else 0,
    }


def fmt_pct(val):
    return f"{val:5.1f}%"


def fmt_score(val):
    return f"{val:.3f}"


def print_bar(pct, width=30):
    filled = int(pct / 100 * width)
    return "#" * filled + "-" * (width - filled)


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------
def section_overall(judgments):
    print()
    print("=" * 80)
    print("1. OVERALL BY MODEL x CONDITION")
    print("=" * 80)

    models = sorted(set(j["model"] for j in judgments))
    conditions = sorted(set(j["condition"] for j in judgments))

    header = f"{'Model':<15} {'Cond':>4} {'n':>4}  {'Refuse%':>8} {'Comply%':>8} {'Score':>7}"
    print(header)
    print("-" * len(header))

    for model in models:
        for cond in conditions:
            subset = [j for j in judgments if j["model"] == model and j["condition"] == cond]
            m = compute_metrics(subset)
            if not m:
                continue
            print(f"{model:<15} {cond:>4} {m['n']:>4}  "
                  f"{fmt_pct(m['refuse_rate']):>8} "
                  f"{fmt_pct(m['comply_rate']):>8} {fmt_score(m['mean_score']):>7}")


def section_tier(judgments):
    print()
    print("=" * 80)
    print("2. BY TIER x CONDITION")
    print("=" * 80)

    models = sorted(set(j["model"] for j in judgments))
    conditions = sorted(set(j["condition"] for j in judgments))

    for model in models:
        print(f"\n  Model: {model}")
        header = f"  {'Tier':<25} {'Cond':>4} {'n':>4}  {'Refuse%':>8} {'Comply%':>8} {'Score':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for tier, tier_label in [("tier1", "Tier 1 (Prohibited)"), ("tier2", "Tier 2 (High-Risk)")]:
            for cond in conditions:
                subset = [j for j in judgments
                          if j["model"] == model and j["tier"] == tier and j["condition"] == cond]
                m = compute_metrics(subset)
                if not m:
                    continue
                print(f"  {tier_label:<25} {cond:>4} {m['n']:>4}  "
                      f"{fmt_pct(m['refuse_rate']):>8} "
                      f"{fmt_pct(m['comply_rate']):>8} {fmt_score(m['mean_score']):>7}")


def section_category(judgments, condition="B"):
    print()
    print("=" * 80)
    print(f"3. BY CATEGORY (Condition {condition})")
    print("=" * 80)

    models = sorted(set(j["model"] for j in judgments))

    for model in models:
        print(f"\n  Model: {model}")
        subset = [j for j in judgments if j["model"] == model and j["condition"] == condition]

        cat_data = defaultdict(list)
        for j in subset:
            cat_data[j["category"]].append(j)

        header = f"  {'Cat':<6} {'n':>3}  {'Refuse%':>8} {'Comply%':>8} {'Score':>7} {'Harm':>5}  {'Bar'}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        tier1_cats = sorted(
            [c for c in cat_data if c.startswith("A") and not c.startswith("AH")],
            key=lambda x: int(x[1:]),
        )
        tier2_cats = sorted(
            [c for c in cat_data if c.startswith("AH")], key=lambda x: int(x[2:])
        )

        for cat_group, group_label in [
            (tier1_cats, "--- Tier 1 Prohibited ---"),
            (tier2_cats, "--- Tier 2 High-Risk ---"),
        ]:
            print(f"  {group_label}")
            for cat in cat_group:
                m = compute_metrics(cat_data[cat])
                bar = print_bar(m["comply_rate"], 20)
                print(f"  {cat:<6} {m['n']:>3}  "
                      f"{fmt_pct(m['refuse_rate']):>8} "
                      f"{fmt_pct(m['comply_rate']):>8} {fmt_score(m['mean_score']):>7} "
                      f"{m['mean_harm']:>5.1f}  {bar}")


def section_score_distribution(judgments, condition="B"):
    print()
    print("=" * 80)
    print(f"4. SCORE DISTRIBUTION (Condition {condition}, non-refused only)")
    print("=" * 80)

    models = sorted(set(j["model"] for j in judgments))

    for model in models:
        subset = [j for j in judgments
                  if j["model"] == model and j["condition"] == condition
                  and j["label"] != "refuse" and j.get("score") is not None]
        if not subset:
            continue

        scores = [j["score"] for j in subset]
        print(f"\n  Model: {model} (n={len(subset)} non-refused)")
        print(f"    Mean score:   {sum(scores)/len(scores):.3f}")
        print(f"    Min score:    {min(scores):.3f}")
        print(f"    Max score:    {max(scores):.3f}")

        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        for lo, hi in bins:
            count = sum(1 for s in scores if lo <= s < hi)
            pct = count / len(scores) * 100
            bar = "#" * int(pct / 2)
            label = f"[{lo:.1f}-{hi:.1f})" if hi < 1.01 else f"[{lo:.1f}-1.0]"
            print(f"    {label:>10}  {count:3d} ({pct:5.1f}%) {bar}")


def section_worst_categories(judgments, condition="B"):
    print()
    print("=" * 80)
    print(f"5. MOST VULNERABLE CATEGORIES (Condition {condition}, by compliance rate)")
    print("=" * 80)

    models = sorted(set(j["model"] for j in judgments))

    for model in models:
        subset = [j for j in judgments if j["model"] == model and j["condition"] == condition]
        cat_data = defaultdict(list)
        for j in subset:
            cat_data[j["category"]].append(j)

        print(f"\n  Model: {model}")
        ranked = []
        for cat, items in cat_data.items():
            m = compute_metrics(items)
            ranked.append((cat, m["comply_rate"], m["mean_score"], m["n"]))

        ranked.sort(key=lambda x: -x[1])
        for i, (cat, comp_rate, mean_score, n) in enumerate(ranked[:10]):
            print(f"    {i+1:2d}. {cat:<6}  Comply={comp_rate:5.1f}%  "
                  f"Score={mean_score:.3f}  (n={n})")


def section_condition_comparison(judgments):
    """Compare Condition A vs B per skill."""
    print()
    print("=" * 80)
    print("6. CONDITION A vs B COMPARISON (per skill)")
    print("=" * 80)

    models = sorted(set(j["model"] for j in judgments))

    for model in models:
        a_items = {j["anon_id"]: j for j in judgments if j["model"] == model and j["condition"] == "A"}
        b_items = {j["anon_id"]: j for j in judgments if j["model"] == model and j["condition"] == "B"}

        common = set(a_items.keys()) & set(b_items.keys())
        if not common:
            continue

        transitions = Counter()
        for anon_id in common:
            a_label = a_items[anon_id]["label"]
            b_label = b_items[anon_id]["label"]
            transitions[(a_label, b_label)] += 1

        print(f"\n  Model: {model} ({len(common)} skills with both A and B)")
        print(f"  Transition matrix (rows=Cond A, cols=Cond B):")
        labels = ["refuse", "comply"]
        ab_label = "A \\ B"
        header = f"  {ab_label:<22}" + "".join(f"{l:>22}" for l in labels)
        print(header)
        for a_label in labels:
            row = f"  {a_label:<22}"
            for b_label in labels:
                count = transitions.get((a_label, b_label), 0)
                row += f"{count:>22}"
            print(row)

        a_metrics = compute_metrics(list(a_items.values()))
        b_metrics = compute_metrics(list(b_items.values()))
        print(f"\n  Refuse rate:     A={a_metrics['refuse_rate']:5.1f}%  ->  "
              f"B={b_metrics['refuse_rate']:5.1f}%  "
              f"(delta={b_metrics['refuse_rate']-a_metrics['refuse_rate']:+.1f}%)")
        print(f"  Comply rate:     A={a_metrics['comply_rate']:5.1f}%  ->  "
              f"B={b_metrics['comply_rate']:5.1f}%  "
              f"(delta={b_metrics['comply_rate']-a_metrics['comply_rate']:+.1f}%)")
        print(f"  Mean score:      A={a_metrics['mean_score']:.3f}  ->  "
              f"B={b_metrics['mean_score']:.3f}  "
              f"(delta={b_metrics['mean_score']-a_metrics['mean_score']:+.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze HarmSkillBench results")
    parser.add_argument("--models", nargs="+", default=None, help="Filter by models")
    parser.add_argument("--conditions", nargs="+", default=None, help="Filter by conditions")
    args = parser.parse_args()

    judgments = load_judgments(models=args.models, conditions=args.conditions)
    print(f"Loaded {len(judgments)} judgments")

    if not judgments:
        print("No judgments found.")
        return

    section_overall(judgments)
    section_tier(judgments)

    conditions = set(j["condition"] for j in judgments)
    for cond in sorted(conditions):
        section_category(judgments, condition=cond)

    if "B" in conditions:
        section_score_distribution(judgments, condition="B")
        section_worst_categories(judgments, condition="B")

    if "A" in conditions and "B" in conditions:
        section_condition_comparison(judgments)


if __name__ == "__main__":
    main()
