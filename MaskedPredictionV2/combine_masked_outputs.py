import pandas as pd
import json
from pathlib import Path

OUTPUT_DIR = Path("/home/hice1/kmehta301/scratch/Final_Euphemism_Detection/masked_prediction/masked_prediction_outputs")

# ---- 1. Combine instance-level files ----
instance_files = sorted(OUTPUT_DIR.glob("masked_pred_instances_task_*.csv"))
dfs = []

print(f"Found {len(instance_files)} instance files")

for f in instance_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"Skipping {f}: {e}")

instances = pd.concat(dfs, ignore_index=True)
instances.to_csv(OUTPUT_DIR / "masked_pred_instances_combined.csv", index=False)

print(f"Combined instances: {len(instances)} rows")

# ---- 2. Recompute aggregation (better than merging agg files) ----
agg = (
    instances.groupby(["canonical_phrase", "period"], as_index=False)
    .agg(taboo_rate=("taboo_in_topk", "mean"), n=("taboo_in_topk", "size"))
)

agg.to_csv(OUTPUT_DIR / "masked_pred_agg_combined.csv", index=False)

print(f"Combined agg rows: {len(agg)}")

# ---- 3. Combine run summaries (optional) ----
summary_files = sorted(OUTPUT_DIR.glob("masked_pred_run_summary_task_*.json"))
summaries = []

for f in summary_files:
    try:
        with open(f) as fp:
            summaries.append(json.load(fp))
    except Exception as e:
        print(f"Skipping {f}: {e}")

with open(OUTPUT_DIR / "masked_pred_run_summary_combined.json", "w") as f:
    json.dump(summaries, f, indent=2)

print(f"Combined {len(summaries)} summaries")

print("\nDONE")