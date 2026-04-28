"""
Merge script: combines partial_results_*.json from SLURM array jobs.

PRIMARY ANALYSIS (centroid):
  For each word, computes monthly cosine similarity against a single
  drug centroid (average of all anchor embeddings). This answers:
  "Is this word becoming more drug-related over time?"

SECONDARY ANALYSIS (per-anchor):
  Also computes similarity against each anchor individually, discovering
  which specific drug a word is most associated with. Supplementary detail.

Group comparison plot shows the core argument:
  - Euphemism candidates: similarity should INCREASE over time
  - Established euphemisms: similarity should be STABLE HIGH
  - Comparison words: similarity should be FLAT LOW

Uses TWO centroid types for robustness:
  Option A (template): fixed reference from canonical sentences
  Option B (corpus): derived from real Reddit contexts of anchor words

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
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats as scipy_stats
import os
import re
import argparse

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
HIDDEN_DIM = 768
MIN_INSTANCES_PER_PERIOD = 5
MIN_PERIODS = 3
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Spelling variants — merge these before analysis
WORD_ALIASES = {
    "grey death": "gray death",
    "oui'd": "ouid",
    "oui\u2019d": "ouid",
}

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


def aggregate_to_half_year(word_data):
    """
    Re-aggregate monthly sum/count data into half-year bins.
    word_data: month -> {sum: np.array, count: int}
    Returns: half_year -> {sum: np.array, count: int}
    """
    aggregated = defaultdict(lambda: {"sum": np.zeros(HIDDEN_DIM), "count": 0})
    for month, data in word_data.items():
        period = month_to_half_year(month)
        aggregated[period]["sum"] += data["sum"]
        aggregated[period]["count"] += data["count"]
    return dict(aggregated)


# ──────────────────────────────────────────────────────────────
# 1. AGGREGATE PARTIAL FILES
# ──────────────────────────────────────────────────────────────
partial_files = sorted(glob.glob("partial_results_*.json"))
print(f"Found {len(partial_files)} partial files.")
if not partial_files:
    print("No partial files found. Exiting.")
    exit()

# Accumulate: word -> month -> {sum, count}
all_data = defaultdict(lambda: defaultdict(lambda: {
    "sum": np.zeros(HIDDEN_DIM), "count": 0,
}))

# Collect anchor embeddings
all_template_embeddings = defaultdict(list)  # anchor -> list of vectors
corpus_anchor_sums = defaultdict(lambda: {"sum": np.zeros(HIDDEN_DIM), "count": 0})
word_groups = {}
total_stats = {"rows_read": 0, "embedded": 0, "skipped_no_span": 0, "skipped_no_tokens": 0}

for f_path in partial_files:
    print(f"  Reading {f_path}...", end=" ", flush=True)
    try:
        with open(f_path, "r") as f:
            content = json.load(f)

        # Accumulate word embeddings
        for cand, slices in content["data"].items():
            cand_normalized = WORD_ALIASES.get(cand.lower(), cand.lower())
            for month, stats in slices.items():
                entry = all_data[cand_normalized][month]
                entry["sum"] += np.array(stats["sum_embedding"])
                entry["count"] += stats["count"]

        # Collect template anchor embeddings (Option A)
        if "anchor_template_embeddings" in content:
            for anchor, vec in content["anchor_template_embeddings"].items():
                all_template_embeddings[anchor].append(np.array(vec))

        # Accumulate corpus anchor sums (Option B)
        if "corpus_anchor_sums" in content:
            for anchor, data in content["corpus_anchor_sums"].items():
                corpus_anchor_sums[anchor]["sum"] += np.array(data["sum"])
                corpus_anchor_sums[anchor]["count"] += data["count"]

        # Merge word groups
        if "word_groups" in content:
            word_groups.update(content["word_groups"])

        # Stats
        if "stats" in content:
            for key in total_stats:
                total_stats[key] += content["stats"].get(key, 0)

        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")

# Apply half-year aggregation if requested
if USE_HALF_YEAR:
    print("\nAggregating to half-year periods...")
    aggregated_data = {}
    for word, month_data in all_data.items():
        aggregated_data[word] = defaultdict(lambda: {"sum": np.zeros(HIDDEN_DIM), "count": 0})
        for month, data in month_data.items():
            period = month_to_half_year(month)
            aggregated_data[word][period]["sum"] += data["sum"]
            aggregated_data[word][period]["count"] += data["count"]
    all_data = aggregated_data

print(f"\nMerge complete.")
print(f"  Total unique words: {len(all_data)}")
print(f"  Total embedded:     {total_stats['embedded']}")


# ──────────────────────────────────────────────────────────────
# 2. BUILD ANCHOR REFERENCE EMBEDDINGS
# ──────────────────────────────────────────────────────────────

# Option A: template-based (average across partial files; should be near-identical)
template_anchors = {}
for anchor, vecs in all_template_embeddings.items():
    template_anchors[anchor] = np.mean(vecs, axis=0)
print(f"\nOption A: {len(template_anchors)} template anchor embeddings.")

# Option B: corpus-based (average real usage across all files)
corpus_anchors = {}
for anchor, data in corpus_anchor_sums.items():
    if data["count"] > 0:
        corpus_anchors[anchor] = data["sum"] / data["count"]
print(f"Option B: {len(corpus_anchors)} corpus anchor embeddings "
      f"(from {sum(d['count'] for d in corpus_anchor_sums.values())} instances).")


# ──────────────────────────────────────────────────────────────
# 3. ANALYSIS FUNCTION
# ──────────────────────────────────────────────────────────────
def run_analysis(anchor_embeddings, label):
    """
    For each word, compute:
      1. Monthly similarity to the CENTROID (primary — is it drug-related?)
      2. Monthly similarity to each anchor (secondary — which drug?)
    Returns a list of result dicts.
    """
    print(f"\n{'='*60}")
    print(f"Analysis: {label}")
    print(f"{'='*60}")

    time_label = "half_year" if USE_HALF_YEAR else "monthly"
    plot_dir_main = os.path.join(OUTPUT_DIR, f"plots_{label}_centroid_{time_label}")
    plot_dir_anchors = os.path.join(OUTPUT_DIR, f"plots_{label}_per_anchor_{time_label}")
    os.makedirs(plot_dir_main, exist_ok=True)
    os.makedirs(plot_dir_anchors, exist_ok=True)

    anchor_names = sorted(anchor_embeddings.keys())
    anchor_matrix = np.array([anchor_embeddings[a] for a in anchor_names])  # (n_anchors, 768)

    # THE CENTROID: single vector = mean of all anchor embeddings
    centroid = anchor_matrix.mean(axis=0).reshape(1, -1)  # (1, 768)
    print(f"Centroid built from {len(anchor_names)} anchors.")

    results = []

    for word, slices in sorted(all_data.items()):
        group = word_groups.get(word.lower(), "unknown")

        # Skip anchor words — measuring cocaine vs the drug centroid isn't useful
        if group == "anchor":
            continue

        # Filter periods
        valid_periods = {
            m: d for m, d in slices.items()
            if d["count"] >= MIN_INSTANCES_PER_PERIOD and m != "unknown"
        }
        time_keys = sorted(valid_periods.keys())
        if len(time_keys) < MIN_PERIODS:
            continue

        # ── Compute similarities ──
        periods = []
        centroid_sims = []       # PRIMARY: similarity to drug centroid
        all_anchor_sims = []     # SECONDARY: similarity to each anchor
        counts = []

        for period in time_keys:
            data = valid_periods[period]
            mean_emb = (data["sum"] / data["count"]).reshape(1, -1)

            # Primary: similarity to centroid
            c_sim = cosine_similarity(mean_emb, centroid)[0][0]

            # Secondary: similarity to all anchors
            a_sims = cosine_similarity(mean_emb, anchor_matrix)[0]  # (n_anchors,)

            periods.append(period)
            centroid_sims.append(c_sim)
            all_anchor_sims.append(a_sims)
            counts.append(data["count"])

        centroid_sims = np.array(centroid_sims)
        sim_matrix = np.array(all_anchor_sims)  # (n_periods, n_anchors)

        # ── Primary stats: centroid drift ──
        x = np.arange(len(centroid_sims), dtype=float)
        centroid_slope = float(np.polyfit(x, centroid_sims, 1)[0]) if len(x) > 1 else 0.0
        centroid_drift = float(centroid_sims[-1] - centroid_sims[0])
        centroid_mean = float(np.mean(centroid_sims))
        centroid_std = float(np.std(centroid_sims))

        # Statistical significance: is the trend real or noise?
        # Pearson r: correlation between time index and similarity
        # p-value: probability of observing this correlation by chance
        if len(x) > 2:
            r_val, p_val = scipy_stats.pearsonr(x, centroid_sims)
            centroid_r = float(r_val)
            centroid_p = float(p_val)
        else:
            centroid_r = 0.0
            centroid_p = 1.0

        # ── Secondary stats: per-anchor ──
        mean_sims_per_anchor = sim_matrix.mean(axis=0)
        best_anchor_idx = np.argmax(mean_sims_per_anchor)
        best_anchor = anchor_names[best_anchor_idx]
        best_sims = sim_matrix[:, best_anchor_idx]
        best_slope = float(np.polyfit(x, best_sims, 1)[0]) if len(x) > 1 else 0.0

        # Steepest increasing anchor
        slopes_per_anchor = []
        for a_idx in range(len(anchor_names)):
            a_sims = sim_matrix[:, a_idx]
            a_slope = float(np.polyfit(x, a_sims, 1)[0]) if len(x) > 1 else 0.0
            slopes_per_anchor.append(a_slope)
        steepest_idx = np.argmax(slopes_per_anchor)
        steepest_anchor = anchor_names[steepest_idx]
        steepest_slope = slopes_per_anchor[steepest_idx]

        results.append({
            "word": word,
            "group": group,
            "n_periods": len(periods),
            "total_instances": sum(counts),
            "periods": periods,
            "counts": counts,
            "centroid_sims": centroid_sims.tolist(),
            "centroid_slope": round(centroid_slope, 6),
            "centroid_drift": round(centroid_drift, 4),
            "centroid_mean_sim": round(centroid_mean, 4),
            "centroid_std_sim": round(centroid_std, 4),
            "centroid_r": round(centroid_r, 4),
            "centroid_p": round(centroid_p, 4),
            "centroid_significant": centroid_p < 0.05,
            "best_anchor": best_anchor,
            "best_anchor_mean_sim": round(float(mean_sims_per_anchor[best_anchor_idx]), 4),
            "best_anchor_slope": round(best_slope, 6),
            "steepest_anchor": steepest_anchor,
            "steepest_anchor_slope": round(steepest_slope, 6),
            "per_anchor_mean_sim": {
                anchor_names[i]: round(float(mean_sims_per_anchor[i]), 4)
                for i in range(len(anchor_names))
            },
        })

        # ── PRIMARY PLOT: word vs drug centroid over time ──
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(range(len(periods)), centroid_sims, marker="o", color="royalblue",
                 linewidth=2, markersize=5)
        ax1.set_ylabel("Cosine Similarity to Drug Centroid", color="royalblue")
        ax1.tick_params(axis="y", labelcolor="royalblue")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Time Period")
        ax1.set_title(
            f"\"{word}\" [{group}] — Drug Centroid Drift [{label}]  "
            f"(slope={centroid_slope:.4f}, r={centroid_r:.3f}, p={centroid_p:.3f}, n={sum(counts)})"
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
        plt.savefig(os.path.join(plot_dir_main, f"{safe_name}.png"), dpi=150)
        plt.close()

        # ── SECONDARY PLOT: top 3 individual anchors ──
        top3_indices = np.argsort(mean_sims_per_anchor)[-3:][::-1]
        fig, ax1 = plt.subplots(figsize=(12, 5))

        colors = ["darkorange", "forestgreen", "crimson"]
        for rank, a_idx in enumerate(top3_indices):
            a_name = anchor_names[a_idx]
            a_sims = sim_matrix[:, a_idx]
            a_slope = slopes_per_anchor[a_idx]
            ax1.plot(
                range(len(periods)), a_sims, marker="o", color=colors[rank],
                linewidth=2, markersize=4,
                label=f"{a_name} (slope={a_slope:.4f})",
            )

        ax1.set_ylabel("Cosine Similarity to Individual Anchor")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Time Period")
        ax1.set_title(f"\"{word}\" [{group}] — Top 3 Anchors [{label}]")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.set_xticks(range(len(periods)))
        ax1.set_xticklabels(periods, rotation=45, ha="right")

        ax2 = ax1.twinx()
        ax2.bar(range(len(periods)), counts, alpha=0.15, color="gray")
        ax2.set_ylabel("Instances", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        plt.savefig(os.path.join(plot_dir_anchors, f"{safe_name}.png"), dpi=150)
        plt.close()

    print(f"Centroid plots saved to {plot_dir_main}/")
    print(f"Per-anchor plots saved to {plot_dir_anchors}/")
    return results, anchor_names


# ──────────────────────────────────────────────────────────────
# 4. RUN BOTH ANALYSES
# ──────────────────────────────────────────────────────────────
results_a, anchors_a = run_analysis(template_anchors, "template")

if corpus_anchors:
    results_b, anchors_b = run_analysis(corpus_anchors, "corpus")
else:
    print("\n[WARN] No corpus anchor data — skipping Option B analysis.")
    results_b = None


# ──────────────────────────────────────────────────────────────
# 5. GROUP COMPARISON PLOT
# ──────────────────────────────────────────────────────────────
def make_group_plot(results, label):
    """
    PRIMARY FIGURE for the paper.
    Average centroid similarity time series within each group.
    """
    time_label = "half_year" if USE_HALF_YEAR else "monthly"
    plot_path = os.path.join(OUTPUT_DIR, f"group_comparison_{label}_{time_label}.png")

    all_periods = sorted(set(
        m for r in results for m in r["periods"]
    ))
    period_to_idx = {m: i for i, m in enumerate(all_periods)}

    group_colors = {
        "euphemism_candidate": ("royalblue", "Euphemism Candidates"),
        "established_euphemism": ("forestgreen", "Established Euphemisms"),
        "comparison": ("gray", "Comparison Words"),
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for group_key, (color, group_label) in group_colors.items():
        group_results = [r for r in results if r["group"] == group_key]
        if not group_results:
            continue

        sim_grid = np.full((len(group_results), len(all_periods)), np.nan)
        for w_idx, r in enumerate(group_results):
            for m_idx, period in enumerate(r["periods"]):
                global_idx = period_to_idx[period]
                sim_grid[w_idx, global_idx] = r["centroid_sims"][m_idx]

        # Average across words for each period (ignoring NaN)
        mean_line = np.nanmean(sim_grid, axis=0)
        std_line = np.nanstd(sim_grid, axis=0)

        # Only plot period where we have at least 1 word
        valid = ~np.isnan(mean_line)
        valid_indices = [i for i in range(len(all_periods)) if valid[i]]
        valid_mean = mean_line[valid]
        valid_std = std_line[valid]

        ax.plot(valid_indices, valid_mean, marker="o", color=color,
                linewidth=2, markersize=4, label=f"{group_label} (n={len(group_results)})")
        ax.fill_between(valid_indices, valid_mean - valid_std, valid_mean + valid_std,
                        color=color, alpha=0.1)

    ax.set_ylabel("Cosine Similarity to Drug Centroid")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time Period")
    ax.set_title(f"Group Comparison [{label}]: Euphemism Emergence Signal")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(len(all_periods)))
    ax.set_xticklabels(all_periods, rotation=45, ha="right")
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Group comparison plot saved: {plot_path}")


make_group_plot(results_a, "template")
if results_b:
    make_group_plot(results_b, "corpus")


# ──────────────────────────────────────────────────────────────
# 6. SAVE SUMMARY TABLES
# ──────────────────────────────────────────────────────────────
def save_summary(results, label):
    """Save summary JSON and print results by group."""
    time_label = "half_year" if USE_HALF_YEAR else "monthly"
    results_sorted = sorted(results, key=lambda r: r["centroid_slope"], reverse=True)

    # Clean up for JSON
    summary = []
    for r in results_sorted:
        summary.append({
            "word": r["word"],
            "group": r["group"],
            "n_periods": r["n_periods"],
            "total_instances": r["total_instances"],
            # PRIMARY: centroid
            "centroid_mean_sim": r["centroid_mean_sim"],
            "centroid_slope": r["centroid_slope"],
            "centroid_drift": r["centroid_drift"],
            "centroid_r": r["centroid_r"],
            "centroid_p": r["centroid_p"],
            "centroid_significant": r["centroid_significant"],
            # SECONDARY: per-anchor
            "best_anchor": r["best_anchor"],
            "best_anchor_mean_sim": r["best_anchor_mean_sim"],
            "best_anchor_slope": r["best_anchor_slope"],
            "steepest_anchor": r["steepest_anchor"],
            "steepest_anchor_slope": r["steepest_anchor_slope"],
            "per_anchor_mean_sim": r["per_anchor_mean_sim"],
        })

    out_path = os.path.join(OUTPUT_DIR, f"drift_summary_{label}_{time_label}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path} ({len(summary)} words)")

    # Print grouped results with significance
    for group_name in ["euphemism_candidate", "established_euphemism", "comparison"]:
        group_rows = [r for r in results_sorted if r["group"] == group_name]
        if not group_rows:
            continue
        sig_count = sum(1 for r in group_rows if r["centroid_significant"])
        print(f"\n── {group_name.upper()} [{label}] ({sig_count}/{len(group_rows)} significant) ──")
        print(f"{'Word':<18} {'Mean Sim':>9} {'Slope':>9} {'r':>7} {'p':>7} {'Sig':>4} │ "
              f"{'Best Anchor':<15} {'Instances':>10}")
        print("─" * 100)
        for r in group_rows:
            sig_marker = " *" if r["centroid_significant"] else "  "
            print(
                f"{r['word']:<18} "
                f"{r['centroid_mean_sim']:>9.4f} {r['centroid_slope']:>9.4f} "
                f"{r['centroid_r']:>7.3f} {r['centroid_p']:>7.3f} {sig_marker:>4} │ "
                f"{r['best_anchor']:<15} {r['total_instances']:>10}"
            )

    # ── GROUP-LEVEL STATISTICAL TEST ──
    # Are euphemism candidate slopes significantly different from comparison word slopes?
    candidate_slopes = [r["centroid_slope"] for r in results_sorted if r["group"] == "euphemism_candidate"]
    comparison_slopes = [r["centroid_slope"] for r in results_sorted if r["group"] == "comparison"]

    if len(candidate_slopes) >= 2 and len(comparison_slopes) >= 2:
        print(f"\n── GROUP-LEVEL TEST [{label}] ──")
        print(f"  Candidate slopes:   mean={np.mean(candidate_slopes):.6f}, n={len(candidate_slopes)}")
        print(f"  Comparison slopes:  mean={np.mean(comparison_slopes):.6f}, n={len(comparison_slopes)}")

        # Mann-Whitney U: non-parametric, doesn't assume normality
        # Tests whether candidate slopes tend to be larger than comparison slopes
        u_stat, u_p = scipy_stats.mannwhitneyu(
            candidate_slopes, comparison_slopes, alternative="greater"
        )
        print(f"  Mann-Whitney U (candidates > comparisons): U={u_stat:.1f}, p={u_p:.4f}")
        if u_p < 0.05:
            print(f"  → SIGNIFICANT: euphemism candidates drift more than comparison words (p={u_p:.4f})")
        else:
            print(f"  → Not significant at α=0.05 (p={u_p:.4f})")

        # Also report established euphemisms vs comparison
        established_slopes = [r["centroid_slope"] for r in results_sorted if r["group"] == "established_euphemism"]
        if len(established_slopes) >= 2:
            print(f"  Established slopes: mean={np.mean(established_slopes):.6f}, n={len(established_slopes)}")


save_summary(results_a, "template")
if results_b:
    save_summary(results_b, "corpus")


# ──────────────────────────────────────────────────────────────
# 7. CROSS-CENTROID AGREEMENT
# ──────────────────────────────────────────────────────────────
if results_b:
    print(f"\n{'='*60}")
    print("Cross-centroid agreement check")
    print(f"{'='*60}")

    map_a = {r["word"]: r for r in results_a}
    map_b = {r["word"]: r for r in results_b}
    shared = set(map_a.keys()) & set(map_b.keys())

    if shared:
        # PRIMARY: Do centroid slopes correlate?
        slopes_a = np.array([map_a[w]["centroid_slope"] for w in shared])
        slopes_b = np.array([map_b[w]["centroid_slope"] for w in shared])
        corr_centroid = np.corrcoef(slopes_a, slopes_b)[0, 1]
        print(f"  Words in both: {len(shared)}")
        print(f"  Centroid slope correlation (Pearson): {corr_centroid:.4f}")

        # SECONDARY: Do they agree on which specific anchor?
        anchor_agreement = sum(
            1 for w in shared
            if map_a[w]["best_anchor"] == map_b[w]["best_anchor"]
        )
        print(f"  Best-anchor agreement: {anchor_agreement}/{len(shared)} "
              f"({100*anchor_agreement/len(shared):.1f}%)")

print("\nDone!")