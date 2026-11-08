#!/bin/bash
#SBATCH --job-name=pilot_hits
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=./logs_pilot_hits/pilot_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Single-month PILOT experiment MAP task (standalone).
#
# Scores ONE month of the corpus for the pilot/control words only and writes a
# *.pilot.json shard of per-context rows. SLURM ARRAY job: the array index
# selects the month (--shard-index / $SLURM_ARRAY_TASK_ID). For a real run use
# submit_pilot_hits_pipeline.sh (map array + merge); this standalone script is
# handy for debugging a single month.
#
# This is a SEPARATE experiment from the hit matrix (hit_matrix.py). It reuses
# the same scoring core, so its numbers match the matrix. The frequency matrix
# (freq_dir) and the pilot config (pilot_words.json) must already exist.
#
# Example (score the first month only):
#   sbatch --array=0 submit_pilot_hits.sbatch
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_pilot_hits

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
PILOT_FILE="pilot_words.json"
OUT_DIR="pilot_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"

python -u pilot_hits.py \
    --reddit-monthly-dir "$DUMP_DIR" \
    --subreddit-file "$SUBREDDIT_FILE" \
    --freq-dir "$FREQ_DIR" \
    --dea-file "$DEA_FILE" \
    --pilot-file "$PILOT_FILE" \
    --shard-dir "$SHARD_DIR" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --model-name ../BERT_model_reddit \
    --start-year 2019 \
    --batch-size 64 \
    --fp16

echo "Finished: $(date)"
