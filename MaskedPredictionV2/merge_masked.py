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

TIME GRANULARITY:
  --half-year flag aggregates months into 6-month bins (YYYY-H1/H2)
  for smoother trends and less noise. Default is monthly.
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
import argparse

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
MIN_INSTANCES_PER_PERIOD = 5   # Need enough instances for a reliable rate
MIN_PERIODS = 3                # Need enough periods to fit a trend
OUTPUT_DIR = "masked_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--half-year", action="store_true",
    help="Aggregate into 6-month bins (YYYY-H1/H2) instead of monthly"
)
args = parser.parse_args()
USE_HALF_YEAR = args.half_year

if USE_HALF_YEAR:
    print("Mode: HALF-YEAR aggregation")
else:
    print("Mode: MONTHLY (default)")


# ──────────────────────────────────────────────────────────────
# HALF-YEAR HELPERS
# ──────────────────────────────────────────────────────────────
def month_to_half_year(month_str):
    """Convert 'YYYY-MM' to 'YYYY-H1' or 'YYYY-H2'."""
    try:
        year, month = month_str.split("-")
        half = "H1" if int(month) <= 6 else "H2"
        return f"{year}-{half}"
    except Exception:
        return "unknown"


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

# Apply half-year aggregation if requested
if USE_HALF_YEAR:
    print("\nAggregating to half-year periods...")
    aggregated_data = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))
    for word, months_data in all_data.items():
        for month, data in months_data.items():
            period = month_to_half_year(month)
            aggregated_data[word][period]["hits"] += data["hits"]
            aggregated_data[word][period]["total"] += data["total"]
    all_data = aggregated_data

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
time_label = "half_year" if USE_HALF_YEAR else "monthly"
print(f"\nAnalyzing trends (min {MIN_INSTANCES_PER_PERIOD} instances/period, min {MIN_PERIODS} periods)...")

plot_dir = os.path.join(OUTPUT_DIR, f"plots_per_word_{time_label}")
os.makedirs(plot_dir, exist_ok=True)

results = []

for word, periods_data in sorted(all_data.items()):
    group = word_groups.get(word.lower(), "unknown")

    # Skip anchors
    if group == "anchor":
        continue

    # Filter periods with enough instances
    valid_periods = {
        m: d for m, d in periods_data.items()
        if d["total"] >= MIN_INSTANCES_PER_PERIOD and m != "unknown"
    }

    time_keys = sorted(valid_periods.keys())
    if len(time_keys) < MIN_PERIODS:
        continue

    periods = []
    rates = []
    counts = []

    for period in time_keys:
        d = valid_periods[period]
        rate = d["hits"] / d["total"]
        periods.append(period)
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
        "n_periods": len(periods),
        "total_instances": sum(counts),
        "periods": periods,
        "rates": rates,
        "counts": counts,
        "mean_rate": round(mean_rate, 4),
        "std_rate": round(std_rate, 4),
        "slope": round(slope, 6),
        "drift": round(drift, 4),
    })

    # ── Per-word plot ──
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(range(len(periods)), rates, marker="o", color="crimson", linewidth=2, markersize=5)
    ax1.set_ylabel("Taboo Hit Rate (fraction)", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Time Period")
    ax1.set_title(
        f"Masked Prediction Shift: \"{word}\" [{group}]  "
        f"(slope={slope:.4f}, mean={mean_rate:.3f}, n={sum(counts)})"
    )
    ax1.set_xticks(range(len(periods)))
    ax1.set_xticklabels(periods, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(range(len(periods)), counts, alpha=0.15, color="gray")
    ax2.set_ylabel("Instances", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.grid(True, alpha=0.2)
    fig.tight_layout()

    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", word)
    plt.savefig(os.path.join(plot_dir, f"{safe_name}.png"), dpi=150)
    plt.close()

print(f"Per-word plots saved to {plot_dir}/")


# ──────────────────────────────────────────────────────────────
# 2b. PRINT AND SAVE PER-WORD BIN DATA
# ──────────────────────────────────────────────────────────────
data_dir = os.path.join(OUTPUT_DIR, f"data_per_word_{time_label}")
os.makedirs(data_dir, exist_ok=True)

all_rows_csv = []

for r in results:
    word = r["word"]
    group = r["group"]
    periods = r["periods"]
    rates = r["rates"]
    counts = r["counts"]

    # Print table
    print(f"\n── {word} [{group}] (slope={r['slope']:.4f}, mean={r['mean_rate']:.4f}) ──")
    print(f"  {'Period':<12} {'Hits':>8} {'Total':>8} {'Rate':>8}")
    print(f"  {'─'*40}")

    word_data = all_data.get(word, {})
    for m_idx, period in enumerate(periods):
        d = word_data.get(period, {"hits": 0, "total": 0})
        hits = d["hits"]
        total = d["total"]
        rate = rates[m_idx]
        print(f"  {period:<12} {hits:>8} {total:>8} {rate:>8.4f}")

        all_rows_csv.append({
            "word": word,
            "group": group,
            "period": period,
            "hits": hits,
            "total": total,
            "rate": round(rate, 4),
        })

    # Save per-word CSV
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", word)
    word_csv_path = os.path.join(data_dir, f"{safe_name}.csv")
    with open(word_csv_path, "w") as f:
        f.write("period,hits,total,rate\n")
        for m_idx, period in enumerate(periods):
            d = word_data.get(period, {"hits": 0, "total": 0})
            f.write(f"{period},{d['hits']},{d['total']},{rates[m_idx]:.4f}\n")

# Save combined CSV
combined_csv_path = os.path.join(OUTPUT_DIR, f"all_words_per_period_{time_label}.csv")
with open(combined_csv_path, "w") as f:
    f.write("word,group,period,hits,total,rate\n")
    for row in all_rows_csv:
        f.write(f"{row['word']},{row['group']},{row['period']},{row['hits']},{row['total']},{row['rate']}\n")

print(f"\nPer-word CSVs saved to {data_dir}/")
print(f"Combined CSV saved to {combined_csv_path}")


# ──────────────────────────────────────────────────────────────
# 3. GROUP COMPARISON PLOT
# ──────────────────────────────────────────────────────────────
def make_group_plot():
    plot_path = os.path.join(OUTPUT_DIR, f"group_comparison_masked_{time_label}.png")

    all_periods_set = sorted(set(m for r in results for m in r["periods"]))
    period_to_idx = {m: i for i, m in enumerate(all_periods_set)}

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

        rate_grid = np.full((len(group_results), len(all_periods_set)), np.nan)
        for w_idx, r in enumerate(group_results):
            for m_idx, period in enumerate(r["periods"]):
                global_idx = period_to_idx[period]
                rate_grid[w_idx, global_idx] = r["rates"][m_idx]

        mean_line = np.nanmean(rate_grid, axis=0)
        std_line = np.nanstd(rate_grid, axis=0)

        valid = ~np.isnan(mean_line)
        valid_indices = [i for i in range(len(all_periods_set)) if valid[i]]
        valid_mean = mean_line[valid]
        valid_std = std_line[valid]

        ax.plot(valid_indices, valid_mean, marker="o", color=color,
                linewidth=2, markersize=4, label=f"{group_label} (n={len(group_results)})")
        ax.fill_between(valid_indices, valid_mean - valid_std, valid_mean + valid_std,
                        color=color, alpha=0.1)

    ax.set_ylabel("Taboo Hit Rate (fraction of contexts with drug prediction)")
    ax.set_xlabel("Time Period")
    ax.set_title("Masked Prediction Shift: Group Comparison")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(len(all_periods_set)))
    ax.set_xticklabels(all_periods_set, rotation=45, ha="right")
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
        "n_periods": r["n_periods"],
        "total_instances": r["total_instances"],
        "mean_rate": r["mean_rate"],
        "std_rate": r["std_rate"],
        "slope": r["slope"],
        "drift": r["drift"],
    })

summary_path = os.path.join(OUTPUT_DIR, f"masked_drift_summary_{time_label}.json")
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
cosine_summary_path = f"results/drift_summary_template_{time_label}.json"
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