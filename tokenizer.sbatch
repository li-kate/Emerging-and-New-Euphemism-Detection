#!/bin/bash
#SBATCH --job-name=euphemism_first_pass
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=./logs/first_pass_%A_%a.out
#SBATCH --error=./logs/first_pass_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# CONFIGURATION
# ============================================================================

CRAWL_ID="CC-MAIN-2024-10"
MAX_WET_FILES=100                # Total across ALL tasks
OUTPUT_DIR="./results"
TABOO_PATH="./taboo_words_decisions.py"

# Pipeline settings
TEMPLATE="taboo"
THRESHOLD="0.65"

# Set to "single" for one config, "experiments" for full experiment suite
RUN_MODE="experiments"

# ============================================================================
# SETUP
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/ml $OUTPUT_DIR cc_cache

module load anaconda3
conda activate base

# ============================================================================
# RUN
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Crawl: $CRAWL_ID"
echo "Mode: $RUN_MODE"
echo "WET files: $MAX_WET_FILES total, ~$(( MAX_WET_FILES / SLURM_ARRAY_TASK_COUNT )) per task"
echo "============================================"

if [ "$RUN_MODE" = "experiments" ]; then
    # Experiment mode: runs ~12 configs, each writes exp_*_taskN.jsonl
    # The --template/--threshold flags are IGNORED in experiment mode
    # (each experiment defines its own config)
    python -u first_pass.py \
        --source common_crawl \
        --crawl-id $CRAWL_ID \
        --max-wet-files $MAX_WET_FILES \
        --output ${OUTPUT_DIR}/placeholder.jsonl \
        --task-id $SLURM_ARRAY_TASK_ID \
        --num-tasks $SLURM_ARRAY_TASK_COUNT \
        --experiments
else
    # Single run mode: uses the template/threshold you set above
    python -u first_pass.py \
        --source common_crawl \
        --crawl-id $CRAWL_ID \
        --max-wet-files $MAX_WET_FILES \
        --template $TEMPLATE \
        --threshold $THRESHOLD \
        --per-category-thresholds \
        --output ${OUTPUT_DIR}/first_pass_${CRAWL_ID}.jsonl \
        --task-id $SLURM_ARRAY_TASK_ID \
        --num-tasks $SLURM_ARRAY_TASK_COUNT
fi

echo "Task $SLURM_ARRAY_TASK_ID complete."

# ============================================================================
# AFTER ALL TASKS FINISH — run merge manually or with a dependency job:
#
# Single run:
#   python merge_results.py \
#       --input-dir ./results \
#       --output ./results/first_pass_merged.jsonl \
#       --dedup
#
# Experiments:
#   python merge_results.py \
#       --input-dir ./results \
#       --output ./results/merged \
#       --experiments \
#       --dedup
#
# Or submit a merge job that waits for the array to finish:
#   sbatch --dependency=afterok:$SLURM_JOB_ID run_merge.sbatch
# ============================================================================