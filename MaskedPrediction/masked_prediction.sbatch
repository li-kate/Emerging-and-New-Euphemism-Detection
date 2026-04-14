#!/bin/bash
#SBATCH --job-name=masked_pred_matches
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-9
#SBATCH --output=masked_pred_matches_%A_%a.out
#SBATCH --error=masked_pred_matches_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1

# Array job: 10 parallel tasks (tasks 0-9)
# Each task processes a shard of the matches JSONL
# Outputs: masked_prediction_outputs/masked_pred_instances_TASK*.csv
# Merge after all complete:
#   cat masked_prediction_outputs/masked_pred_instances_task_*.csv > merged_results.csv
#   cat masked_prediction_outputs/masked_pred_agg_task_*.csv > merged_agg.csv

NUM_TASKS=10
TASK_ID=${SLURM_ARRAY_TASK_ID}

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cd SLURM_SUBMIT_DIR failed"; exit 1; }

# Use the directory you submitted from (keeps scratch vs repo layouts working).
MATCHES_JSONL="${MATCHES_JSONL:-$SLURM_SUBMIT_DIR/ccnews_head200_matches.jsonl}"
TABOO_WORDS="${TABOO_WORDS:-$SLURM_SUBMIT_DIR/taboo_words_refined.py}"

echo "=== Masked Prediction Job (Matches Format) ==="
echo "Job ID: $SLURM_JOB_ID  Task: ${TASK_ID}/${NUM_TASKS}  Host: $HOSTNAME"
echo "Matches file: $MATCHES_JSONL"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'None — CPU mode')"

module load anaconda3 2>/dev/null || true
set +u
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate base 2>/dev/null || true

echo "Running masked_prediction.py with matches file..."
python -u masked_prediction.py \
  --second-pass "$MATCHES_JSONL" \
  --taboo "$TABOO_WORDS" \
  --output-dir ./masked_prediction_outputs \
  --model bert-base-uncased \
  --top-k 15 \
  --max-instances 0 \
  --max-per-group 50 \
  --time-bucket halfyear \
  --log-every 50 \
  --task-id "$TASK_ID" \
  --num-tasks "$NUM_TASKS" \
  --enable-multiprocessing \
  --num-workers 8 \
  2>&1

EXIT_CODE=$?
echo "Task ${TASK_ID} completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
