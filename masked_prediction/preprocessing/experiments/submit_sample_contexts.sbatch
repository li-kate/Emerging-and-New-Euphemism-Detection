#!/bin/bash
#SBATCH --job-name=sample_contexts
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=./logs_sample_contexts/sample_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Single-month CONTEXT-SAMPLE experiment MAP task (standalone).
#
# Samples ONE word's full contexts for ONE month and writes a *.sample.json
# shard. SLURM ARRAY job: the array index selects the month (--shard-index /
# $SLURM_ARRAY_TASK_ID). For a real run use submit_sample_contexts_pipeline.sh
# (map array + merge); this standalone script is handy for debugging a single
# month.
#
# This is a SEPARATE, diagnostic experiment from the hit matrix
# (hit_matrix.py): it keeps the full context text for manual spot-checking
# (hit_matrix_plan.md 8) and does not affect the matrix. Set WORD to the target
# word to sample. The frequency matrix (freq_dir) must already exist.
#
# Example (sample "molly" from the first month only):
#   WORD=molly sbatch --array=0 submit_sample_contexts.sbatch
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_sample_contexts

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

pip install numpy pandas nltk zstandard orjson torch transformers

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  Array task: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $HOSTNAME  Started: $(date)"
echo "============================================"

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"
FREQ_DIR="freq_drug_subreddits_unfiltered"
DEA_FILE="../../data_collection/word_lists/drug_synonyms.txt"
OUT_DIR="sample_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"

WORD="${WORD:?Set WORD to the target word to sample, e.g. WORD=molly sbatch ...}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

python -u sample_contexts.py \
    --word "$WORD" \
    --reddit-monthly-dir "$DUMP_DIR" \
    --subreddit-file "$SUBREDDIT_FILE" \
    --freq-dir "$FREQ_DIR" \
    --dea-file "$DEA_FILE" \
    --shard-dir "$SHARD_DIR" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --model-name ../BERT_model_reddit \
    --start-year 2019 \
    --batch-size 64 \
    --sample-size "$SAMPLE_SIZE" \
    --sample-seed "$SAMPLE_SEED" \
    --fp16

echo "Finished: $(date)"
