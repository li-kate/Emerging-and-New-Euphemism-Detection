import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

OUTPUT_DIR = Path("/home/hice1/kmehta301/scratch/Final_Euphemism_Detection/masked_prediction/masked_prediction_outputs")

# --------------------------------------------------
# Find result files
# --------------------------------------------------
result_files = sorted(OUTPUT_DIR.glob("masked_results_*.json"))
print(f"Found {len(result_files)} masked result JSON files")

if not result_files:
    raise FileNotFoundError(f"No masked_results_*.json files found in {OUTPUT_DIR}")

# --------------------------------------------------
# Combined structures
# --------------------------------------------------
# word -> period -> {hits, total}
combined_data = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))

combined_stats = {
    "rows_read": 0,
    "processed": 0,
    "skipped_no_span": 0,
    "total_hits": 0,
}

combined_word_groups = {}
config_example = None

# --------------------------------------------------
# Read and merge all result files
# --------------------------------------------------
for f in result_files:
    try:
        with open(f, "r") as fp:
            obj = json.load(fp)

        # Save one config example
        if config_example is None and "config" in obj:
            config_example = obj["config"]

        # Merge word groups
        for word, group in obj.get("word_groups", {}).items():
            combined_word_groups[word] = group

        # Merge stats
        stats = obj.get("stats", {})
        for key in combined_stats:
            combined_stats[key] += stats.get(key, 0)

        # Merge nested data
        for word, periods in obj.get("data", {}).items():
            for period, counts in periods.items():
                combined_data[word][period]["hits"] += counts.get("hits", 0)
                combined_data[word][period]["total"] += counts.get("total", 0)

        print(f"Merged {f.name}")

    except Exception as e:
        print(f"Skipping {f}: {e}")

# --------------------------------------------------
# Build combined JSON output
# --------------------------------------------------
combined_json = {
    "data": {},
    "word_groups": combined_word_groups,
    "stats": combined_stats,
    "config": config_example,
    "n_source_files": len(result_files),
}

for word, periods in combined_data.items():
    combined_json["data"][word] = {}
    for period, counts in periods.items():
        combined_json["data"][word][period] = {
            "hits": counts["hits"],
            "total": counts["total"],
        }

combined_json_path = OUTPUT_DIR / "masked_results_combined.json"
with open(combined_json_path, "w") as f:
    json.dump(combined_json, f, indent=2)

print(f"Saved combined JSON: {combined_json_path}")

# --------------------------------------------------
# Flatten to CSV
# --------------------------------------------------
rows = []
for word, periods in combined_data.items():
    for period, counts in periods.items():
        total = counts["total"]
        hits = counts["hits"]
        taboo_rate = hits / total if total > 0 else 0.0
        rows.append({
            "canonical_phrase": word,
            "period": period,
            "taboo_hits": hits,
            "n": total,
            "taboo_rate": taboo_rate,
            "word_group": combined_word_groups.get(word, ""),
        })

df = pd.DataFrame(rows)

if not df.empty:
    df = df.sort_values(["canonical_phrase", "period"]).reset_index(drop=True)

combined_csv_path = OUTPUT_DIR / "masked_pred_agg_combined.csv"
df.to_csv(combined_csv_path, index=False)

print(f"Saved combined CSV: {combined_csv_path}")
print(f"Combined rows: {len(df)}")

# --------------------------------------------------
# Optional: candidate-level summary across all periods
# --------------------------------------------------
if not df.empty:
    summary_df = (
        df.groupby(["canonical_phrase", "word_group"], as_index=False)
        .agg(
            taboo_hits=("taboo_hits", "sum"),
            n=("n", "sum"),
        )
    )
    summary_df["taboo_rate"] = summary_df["taboo_hits"] / summary_df["n"]

    summary_csv_path = OUTPUT_DIR / "masked_pred_candidate_summary_combined.csv"
    summary_df = summary_df.sort_values("taboo_rate", ascending=False).reset_index(drop=True)
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Saved candidate summary CSV: {summary_csv_path}")

print("\nDONE")
