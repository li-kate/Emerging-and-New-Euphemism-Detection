#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs_cosine_similarity

# 24 monthly tasks: task IDs 0-11 = 2015, 12-23 = 2016.
# Limit concurrent jobs to 4 to avoid flooding the GPU partition.
echo "Submitting monthly cosine-similarity array: 0-23 (max 4 concurrent tasks)"

COSINE_JOB=$(sbatch --parsable \
    --array="1-11" \
    submit_sm.sbatch)

echo "Submitted monthly cosine-similarity array: $COSINE_JOB"

# Merge only after every array task succeeds.
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${COSINE_JOB}" \
    merge_sm_monthly.sbatch)

echo "Submitted monthly merge job: $MERGE_JOB"
echo "The merge will run only after all 24 monthly tasks succeed."
echo
echo "Monitor jobs with: squeue -u \$USER"
echo "Inspect logs in: logs_cosine_similarity/"
echo
echo "Monthly intermediate files: cosine_similarity_results/shards/monthly/"
echo "Final half-year shards: cosine_similarity_results/shards/cosine_2015H1.json etc."