#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs_cosine_similarity

# echo "Submitting cosine similarity array: 0-$((NUM_PERIODS - 1))"
echo "Submitting cosine similarity array: 0-1"

# Submit one task per half-year.
COSINE_JOB=$(sbatch --parsable \
    --array="0-1" \
    submit_sm.sbatch)

echo "Submitted cosine similarity array: $COSINE_JOB"

# Merge only after every array task completes successfully.
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${COSINE_JOB}" \
    merge_ms.sbatch)

echo "Submitted merge job: $MERGE_JOB"
echo "The merge will run after all cosine tasks succeed."
echo
echo "Monitor jobs with: squeue -u \$USER"
echo "Inspect logs in: logs_cosine_similarity/"

# to run:
    # chmod +x submit_cs_pipeline.sh
    # ./submit_cs_pipeline.sh
