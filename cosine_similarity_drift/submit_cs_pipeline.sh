#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Must match FREQ_DIR in cosine_similarity_preprocess.py.
HALF_YEAR_CSV="../frequency_matrix/freq_half_years.csv"

mkdir -p logs_cosine_similarity

# Verify the input exists.
if [[ ! -f "$HALF_YEAR_CSV" ]]; then
    echo "ERROR: Cannot find $HALF_YEAR_CSV"
    echo "Update HALF_YEAR_CSV to match your frequency-matrix directory."
    exit 1
fi

# Count data rows in the CSV, handling either a header or no header.
NUM_PERIODS=$(python3 -c '
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))

if rows and any(cell.strip().lower() == "half_year" for cell in rows[0]):
    rows = rows[1:]

print(sum(1 for row in rows if row and row[0].strip()))
' "$HALF_YEAR_CSV")

if [[ "$NUM_PERIODS" -eq 0 ]]; then
    echo "ERROR: No half-years found in $HALF_YEAR_CSV"
    exit 1
fi

echo "Found $NUM_PERIODS half-year periods."
# echo "Submitting cosine similarity array: 0-$((NUM_PERIODS - 1))"
echo "Submitting cosine similarity array: 0-1"

# Submit one task per half-year.
COSINE_JOB=$(sbatch --parsable \
    --array="0-1" \
    submit_cs.sbatch)

echo "Submitted cosine similarity array: $COSINE_JOB"

# Merge only after every array task completes successfully.
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${COSINE_JOB}" \
    merge_cs.sbatch)

echo "Submitted merge job: $MERGE_JOB"
echo "The merge will run after all cosine tasks succeed."
echo
echo "Monitor jobs with: squeue -u \$USER"
echo "Inspect logs in: logs_cosine_similarity/"

# to run:
    # chmod +x submit_cosine_pipeline.sh
    # ./submit_cs_pipeline.sh
