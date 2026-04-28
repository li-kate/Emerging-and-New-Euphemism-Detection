#!/bin/bash
#SBATCH --job-name=masked_pred_final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=masked_pred_%j.out
#SBATCH --error=masked_pred_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

TASK_ID=0
NUM_TASKS=1

SCRIPT_DIR="/storage/ice1/9/7/kmehta301/Final_Euphemism_Detection/masked_prediction"
MATCHES_JSONL="/home/hice1/kmehta301/scratch/Final_Euphemism_Detection/new_pipeline/matches/ccnews_matches_normalized_for_masked_prediction.jsonl"
TABOO_WORDS="/home/hice1/kmehta301/scratch/Final_Euphemism_Detection/masked_prediction/taboo_words_refined.py"
OUTPUT_DIR="/home/hice1/kmehta301/scratch/Final_Euphemism_Detection/masked_prediction/masked_prediction_outputs"
MASKED_SCRIPT="$SCRIPT_DIR/masked_prediction.py"

[[ -f "$MASKED_SCRIPT" ]] || { echo "ERROR: missing $MASKED_SCRIPT"; exit 1; }
[[ -f "$MATCHES_JSONL" ]] || { echo "ERROR: missing $MATCHES_JSONL"; exit 1; }
[[ -f "$TABOO_WORDS" ]] || { echo "ERROR: missing $TABOO_WORDS"; exit 1; }

mkdir -p "$OUTPUT_DIR"

echo "=== Masked Prediction Job ==="
echo "Job ID: $SLURM_JOB_ID  Host: $HOSTNAME"
echo "Script: $MASKED_SCRIPT"
echo "Matches: $MATCHES_JSONL"
echo "Taboo: $TABOO_WORDS"
echo "Output: $OUTPUT_DIR"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'None (CPU mode)')"

export PYTHONNOUSERSITE=1
source /home/hice1/kmehta301/scratch/venvs/euph_scratch/bin/activate

python -u "$MASKED_SCRIPT" \
  --second-pass "$MATCHES_JSONL" \
  --taboo "$TABOO_WORDS" \
  --output-dir "$OUTPUT_DIR" \
  --model bert-base-uncased \
  --top-k 15 \
  --max-instances 0 \
  --max-per-group 100 \
  --time-bucket halfyear \
  --log-every 50 \
  --task-id "$TASK_ID" \
  --num-tasks "$NUM_TASKS" \
  --enable-multiprocessing \
  --num-workers 4

EXIT_CODE=$?
echo "Task ${TASK_ID} completed with exit code: $EXIT_CODE"
exit $EXIT_CODE
