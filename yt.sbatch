#!/bin/bash
#SBATCH --job-name=euphemism_csv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/yt/first_pass_%j.out
#SBATCH --error=./logs/yt/first_pass_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH="youtube_comments_dataset.csv"
OUTPUT_DIR="./results/yt"

# Optional: explicit column names (leave empty for auto-detection)
TEXT_COLUMN=""
TIMESTAMP_COLUMN=""

# ============================================================================
# SETUP
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/yt "$OUTPUT_DIR"

module load anaconda3
conda activate base

# ============================================================================
# RUN
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "CSV: $CSV_PATH"
echo "============================================"

CMD="python -u first_pass.py \
    --source csv \
    --csv-path $CSV_PATH \
    --output ${OUTPUT_DIR}/first_pass_yt.jsonl \
    --task-id 0 \
    --num-tasks 1 \
    --experiments"

if [ -n "$TEXT_COLUMN" ]; then
    CMD="$CMD --text-column $TEXT_COLUMN"
fi
if [ -n "$TIMESTAMP_COLUMN" ]; then
    CMD="$CMD --timestamp-column $TIMESTAMP_COLUMN"
fi

eval $CMD

echo "Done."