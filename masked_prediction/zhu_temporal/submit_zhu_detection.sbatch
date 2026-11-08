#!/bin/bash
#SBATCH --job-name=zhu_detect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=./logs_zhu_detect/zhu_%A_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Single-month Zhu euphemism-detection MAP task (standalone).
#
# This scores ONE month of the corpus with the Reddit-tuned BERT masked LM
# (zhu_detection_month.py) and writes a shard holding per-half-year candidate
# score sums. It is a SLURM ARRAY job: the array index selects the month (see
# --shard-index / $SLURM_ARRAY_TASK_ID). For a real run submit the full
# pipeline (map array + merge) with submit_zhu_detection_pipeline.sh; this
# standalone file is handy for debugging a single month.
#
# Like the hit matrix, scoring needs a GPU (a BERT forward pass per masked
# sentence), so a GPU is requested here.
#
# This reuses Zhu's detection scoring WITHOUT modifying zhu_model: the model
# directory + vocab are passed in, and the MLM / informativeness filter is
# reproduced inside zhu_detection_month.py.
#
# Example (score the first month only):
#   sbatch --array=0 submit_zhu_detection.sbatch
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_zhu_detect

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

# Dependencies. pytorch_pretrained_bert is what zhu_model/detection.py uses for
# BertForMaskedLM; the rest match the other submit scripts. Install a CUDA build
# of torch appropriate to the cluster if the default wheel does not match the
# driver.
pip install numpy nltk zstandard orjson torch pytorch_pretrained_bert

echo "============================================"
echo "Job ID: $SLURM_JOB_ID  Array task: ${SLURM_ARRAY_TASK_ID:-NA}"
echo "Node: $HOSTNAME  Started: $(date)"
echo "============================================"

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="/home/hice1/kli605/scratch/Emerging-and-New-Euphemism-Detection/masked_prediction/preprocessing/drug_subreddit_list.txt"
# Zhu's euphemism answer key; input keywords are derived from it exactly as
# zhu_model/read_file.py does (the formal names before each ':').
ANSWER_FILE="../zhu_model/data/euphemism_answer_drug.txt"
MODEL_DIR="../BERT_model_reddit"
VOCAB_FILE="../BERT_model_reddit/vocab.txt"
OUT_DIR="zhu_detect_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"

python -u zhu_detection_month.py \
    --reddit-monthly-dir "$DUMP_DIR" \
    --subreddit-file "$SUBREDDIT_FILE" \
    --answer-file "$ANSWER_FILE" \
    --model-dir "$MODEL_DIR" \
    --vocab-file "$VOCAB_FILE" \
    --shard-dir "$SHARD_DIR" \
    --shard-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --start-year 2019 \
    --mlm-k 50 \
    --thres 5

echo "Finished: $(date)"
