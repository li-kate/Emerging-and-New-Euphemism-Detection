#!/bin/bash
#SBATCH --job-name=hit_matrix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=./logs_hit_matrix/hit_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Single-month hit-matrix MAP task (standalone).
#
# This scores ONE month of the corpus with BERT and writes a shard. It is a
# SLURM ARRAY job: the array index selects the month (see --shard-index /
# $SLURM_ARRAY_TASK_ID). Submit the full pipeline (map array + merge) with
# submit_hit_matrix_pipeline.sh instead of this file for a real run; this
# standalone script is handy for debugging a single month.
#
# Unlike the frequency build, scoring needs a GPU (a BERT forward pass per
# occurrence), so a GPU is requested here.
#
# The frequency matrix (freq_dir) and the pilot config (pilot_words.json) must
# already exist: the frequency matrix defines the target words + denominator,
# and pilot_creation.py selects the control words scored per-context.
#
# Example (score the first month only):
#   sbatch --array=0 submit_hit_matrix.sbatch
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_hit_matrix

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

# Dependencies. torch/transformers for the MLM; the rest match the other
# submit scripts. Install a CUDA build of torch appropriate to the cluster if
# the default wheel does not match the driver.
pip install numpy pandas nltk zstandard orjson torch transformers

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  Array task: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $HOSTNAME  Started: $(date)"
echo "============================================"

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"
FREQ_DIR="freq_drug_subreddits"
DEA_FILE="../../data_collection/word_lists/drug_synonyms.txt"
PILOT_FILE="pilot_words.json"
OUT_DIR="hits_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"

python -u hit_matrix.py \
    --mode map \
    --reddit-monthly-dir "$DUMP_DIR" \
    --subreddit-file "$SUBREDDIT_FILE" \
    --freq-dir "$FREQ_DIR" \
    --dea-file "$DEA_FILE" \
    --pilot-file "$PILOT_FILE" \
    --shard-dir "$SHARD_DIR" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --model-name bert-base-uncased \
    --batch-size 64 \
    --fp16

echo "Finished: $(date)"
