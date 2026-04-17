"""
Merge script for masked prediction shift analysis.

Combines masked_results_*.json from SLURM array jobs, computes per-word
monthly taboo hit rates, fits trend lines, and generates:

  1. Per-word taboo rate plots (rate over time)
  2. Group comparison plot (candidates vs established vs controls)
  3. Summary table with slopes for each word

The core argument: if a word's taboo hit rate INCREASES over time,
its contexts are becoming more drug-related — it's an emerging euphemism.
Static analysis (Zhu et al.) can't distinguish "always drug-related"
from "becoming drug-related." We can.
"""

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import re

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
MIN_INSTANCES_PER_MONTH = 5   # Need enough instances for a reliable rate
MIN_MONTHS = 3                # Need enough months to fit a trend
OUTPUT_DIR = "masked_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 1. AGGREGATE PARTIAL FILES (final results + unfinished checkpoints)
# ──────────────────────────────────────────────────────────────

# Find all final results
final_files = sorted(glob.glob("masked_results_*.json"))
# Find task IDs that have final results
final_task_ids = set()
for f in final_files:
    m = re.search(r"masked_results_(\d+)\.json", f)
    if m:
        final_task_ids.add(m.group(1))

# Find checkpoints that DON'T have a corresponding final file
checkpoint_files = sorted(glob.glob("masked_checkpoint_*.json"))
orphan_checkpoints = []
for f in checkpoint_files:
    m = re.search(r"masked_checkpoint_(\d+)\.json", f)
    if m and m.group(1) not in final_task_ids:
        orphan_checkpoints.append(f)

print(f"Found {len(final_files)} completed results + {len(orphan_checkpoints)} unfinished checkpoints.")
all_files = final_files + orphan_checkpoints

if not all_files:
    print("No results or checkpoints found. Exiting.")
    exit()

# word -> month -> {hits, total}
all_data = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))
word_groups = {}
total_stats = {"rows_read": 0, "processed": 0, "total_hits": 0}
config = {}

for f_path in all_files:
    is_checkpoint = "checkpoint" in f_path
    label = "CHECKPOINT" if is_checkpoint else "final"
    print(f"  Reading {f_path} ({label})...", end=" ", flush=True)
    try:
        with open(f_path, "r") as f:
            content = json.load(f)

        # Checkpoints store data under "accumulator", final files under "data"
        data_key = "accumulator" if is_checkpoint else "data"
        if data_key not in content:
            print(f"SKIP (no '{data_key}' key)")
            continue

        for word, months in content[data_key].items():
            for month, counts in months.items():
                all_data[word][month]["hits"] += counts["hits"]
                all_data[word][month]["total"] += counts["total"]

        if "word_groups" in content:
            word_groups.update(content["word_groups"])

        if "stats" in content:
            for key in total_stats:
                total_stats[key] += content["stats"].get(key, 0)

        if "config" in content and not config:
            config = content["config"]

        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

print(f"\nMerge complete.")
print(f"  Total unique words:  {len(all_data)}")
print(f"  Total processed:     {total_stats['processed']}")
print(f"  Total hits:          {total_stats['total_hits']}")
print(f"  Overall hit rate:    {total_stats['total_hits']/max(1,total_stats['processed']):.4f}")
if orphan_checkpoints:
    print(f"  NOTE: {len(orphan_checkpoints)} tasks were incomplete — results are partial for those.")


# ──────────────────────────────────────────────────────────────
# 2. COMPUTE PER-WORD TABOO RATE TIME SERIES
# ──────────────────────────────────────────────────────────────
print(f"\nAnalyzing trends (min {MIN_INSTANCES_PER_MONTH} instances/month, min {MIN_MONTHS} months)...")

plot_dir = os.path.join(OUTPUT_DIR, "plots_per_word")
os.makedirs(plot_dir, exist_ok=True)

results = []

for word, months_data in sorted(all_data.items()):
    group = word_groups.get(word.lower(), "unknown")

    # Skip anchors — we don't need to check if "cocaine" predicts drug words
    if group == "anchor":
        continue

    # Filter months with enough instances
    valid_months = {
        m: d for m, d in months_data.items()
        if d["total"] >= MIN_INSTANCES_PER_MONTH and m != "unknown"
    }

    time_keys = sorted(valid_months.keys())
    if len(time_keys) < MIN_MONTHS:
        continue

    months = []
    rates = []
    counts = []

    for month in time_keys:
        d = valid_months[month]
        rate = d["hits"] / d["total"]
        months.append(month)
        rates.append(rate)
        counts.append(d["total"])

    rates_arr = np.array(rates)

    # Trend statistics
    x = np.arange(len(rates_arr), dtype=float)
    slope = float(np.polyfit(x, rates_arr, 1)[0]) if len(x) > 1 else 0.0
    drift = float(rates_arr[-1] - rates_arr[0])
    mean_rate = float(np.mean(rates_arr))
    std_rate = float(np.std(rates_arr))

    results.append({
        "word": word,
        "group": group,
        "n_months": len(months),
        "total_instances": sum(counts),
        "months": months,
        "rates": rates,
        "counts": counts,
        "mean_rate": round(mean_rate, 4),
        "std_rate": round(std_rate, 4),
        "slope": round(slope, 6),
        "drift": round(drift, 4),
    })

    # ── Per-word plot ──
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(months, rates, marker="o", color="crimson", linewidth=2, markersize=5)
    ax1.set_ylabel("Taboo Hit Rate (fraction)", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Month")
    ax1.set_title(
        f"Masked Prediction Shift: \"{word}\" [{group}]  "
        f"(slope={slope:.4f}, mean={mean_rate:.3f}, n={sum(counts)})"
    )

    ax2 = ax1.twinx()
    ax2.bar(months, counts, alpha=0.15, color="gray")
    ax2.set_ylabel("Monthly Instances", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    plt.xticks(rotation=45, ha="right")
    ax1.grid(True, alpha=0.2)
    fig.tight_layout()

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", word)
    plt.savefig(os.path.join(plot_dir, f"{safe_name}.png"), dpi=150)
    plt.close()

print(f"Per-word plots saved to {plot_dir}/")


# ──────────────────────────────────────────────────────────────
# 3. GROUP COMPARISON PLOT
# The money figure: three groups on one chart.
# ──────────────────────────────────────────────────────────────
def make_group_plot():
    plot_path = os.path.join(OUTPUT_DIR, "group_comparison_masked.png")

    all_months_set = sorted(set(m for r in results for m in r["months"]))
    month_to_idx = {m: i for i, m in enumerate(all_months_set)}

    group_colors = {
        "euphemism_candidate": ("crimson", "Euphemism Candidates"),
        "established_euphemism": ("forestgreen", "Established Euphemisms"),
        "comparison": ("gray", "Comparison Words"),
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for group_key, (color, group_label) in group_colors.items():
        group_results = [r for r in results if r["group"] == group_key]
        if not group_results:
            continue

        # Build matrix: (n_words, n_months) with NaN for missing
        rate_grid = np.full((len(group_results), len(all_months_set)), np.nan)
        for w_idx, r in enumerate(group_results):
            for m_idx, month in enumerate(r["months"]):
                global_idx = month_to_idx[month]
                rate_grid[w_idx, global_idx] = r["rates"][m_idx]

        mean_line = np.nanmean(rate_grid, axis=0)
        std_line = np.nanstd(rate_grid, axis=0)

        valid = ~np.isnan(mean_line)
        valid_months = [all_months_set[i] for i in range(len(all_months_set)) if valid[i]]
        valid_mean = mean_line[valid]
        valid_std = std_line[valid]

        ax.plot(valid_months, valid_mean, marker="o", color=color,
                linewidth=2, markersize=4, label=f"{group_label} (n={len(group_results)})")
        ax.fill_between(valid_months, valid_mean - valid_std, valid_mean + valid_std,
                        color=color, alpha=0.1)

    ax.set_ylabel("Taboo Hit Rate (fraction of contexts with drug prediction)")
    ax.set_xlabel("Month")
    ax.set_title("Masked Prediction Shift: Group Comparison")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Group comparison plot saved: {plot_path}")


make_group_plot()


# ──────────────────────────────────────────────────────────────
# 4. SAVE SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
results_sorted = sorted(results, key=lambda r: r["slope"], reverse=True)

summary = []
for r in results_sorted:
    summary.append({
        "word": r["word"],
        "group": r["group"],
        "n_months": r["n_months"],
        "total_instances": r["total_instances"],
        "mean_rate": r["mean_rate"],
        "std_rate": r["std_rate"],
        "slope": r["slope"],
        "drift": r["drift"],
    })

summary_path = os.path.join(OUTPUT_DIR, "masked_drift_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved: {summary_path} ({len(summary)} words)")

# Print by group
for group_name in ["euphemism_candidate", "established_euphemism", "comparison"]:
    group_rows = [r for r in results_sorted if r["group"] == group_name]
    if not group_rows:
        continue
    print(f"\n── {group_name.upper()} ──")
    print(f"{'Word':<20} {'Mean Rate':>10} {'Slope':>10} {'Drift':>10} {'Instances':>10}")
    print("─" * 65)
    for r in group_rows:
        print(
            f"{r['word']:<20} "
            f"{r['mean_rate']:>10.4f} "
            f"{r['slope']:>10.4f} "
            f"{r['drift']:>10.4f} "
            f"{r['total_instances']:>10}"
        )


# ──────────────────────────────────────────────────────────────
# 5. COMPARISON WITH COSINE SIMILARITY (if available)
# ──────────────────────────────────────────────────────────────
cosine_summary_path = "results/drift_summary_template.json"
if os.path.exists(cosine_summary_path):
    print(f"\n{'='*60}")
    print("Cross-method comparison (masked prediction vs cosine similarity)")
    print(f"{'='*60}")

    with open(cosine_summary_path, "r") as f:
        cosine_results = json.load(f)

    cosine_map = {r["word"]: r for r in cosine_results}
    masked_map = {r["word"]: r for r in results_sorted}

    shared = set(cosine_map.keys()) & set(masked_map.keys())
    if shared:
        masked_slopes = np.array([masked_map[w]["slope"] for w in shared])
        cosine_slopes = np.array([cosine_map[w]["centroid_slope"] for w in shared])
        corr = np.corrcoef(masked_slopes, cosine_slopes)[0, 1]
        print(f"  Words in both: {len(shared)}")
        print(f"  Slope correlation (Pearson): {corr:.4f}")

        # Agreement on direction (both positive = both say drifting toward drug)
        same_direction = sum(
            1 for w in shared
            if (masked_map[w]["slope"] > 0) == (cosine_map[w]["centroid_slope"] > 0)
        )
        print(f"  Direction agreement: {same_direction}/{len(shared)} "
              f"({100*same_direction/len(shared):.1f}%)")
else:
    print(f"\n  (Cosine summary not found at {cosine_summary_path} — run cosine pipeline first for cross-method comparison)")

print("\nDone!")