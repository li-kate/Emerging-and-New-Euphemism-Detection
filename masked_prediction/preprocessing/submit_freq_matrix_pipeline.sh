#!/bin/bash
set -euo pipefail

# ============================================================================
# Parallel frequency-matrix pipeline: map (SLURM array) + merge (reduce).
#
#   1. Map array: each task counts its round-robin slice of the Reddit dump
#      files (freq_matrix.py --mode map) and writes one shard.
#   2. Merge: after all map tasks succeed (afterok), merge_freq_matrix.py sums
#      the shards, applies the postfilter, and writes the final matrix.
#
# Modeled on submit_drug_centroid_pipeline.sh. Counting is CPU-only, so no GPU
# is requested.
#
# to run:
#   chmod +x submit_freq_matrix_pipeline.sh
#   ./submit_freq_matrix_pipeline.sh
# ============================================================================

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs_freq_matrix

# --- Configuration -----------------------------------------------------------
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"   # relative to this folder
OUT_DIR="freq_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="freq"
DTYPE="int32"

# Number of map array tasks. Each task handles multiple dump files via a
# round-robin stride, so this does NOT need to equal the file count. Override
# from the environment, e.g.  NUM_SHARDS=20 ./submit_freq_matrix_pipeline.sh
NUM_SHARDS="${NUM_SHARDS:-40}"

# --- Count dump files --------------------------------------------------------
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob
NUM_FILES=${#dumps[@]}

if [[ "$NUM_FILES" -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi

# Never launch more tasks than files (extra tasks would get empty slices).
if [[ "$NUM_SHARDS" -gt "$NUM_FILES" ]]; then
    echo "NUM_SHARDS ($NUM_SHARDS) > files ($NUM_FILES); clamping to $NUM_FILES."
    NUM_SHARDS="$NUM_FILES"
fi

echo "Found $NUM_FILES Reddit dump files; using $NUM_SHARDS map tasks."

# Start the shard directory clean so a rerun never merges stale shards from a
# previous attempt.
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.part.json "$SHARD_DIR"/*.part.json.tmp 2>/dev/null || true

# --- Map array ---------------------------------------------------------------
# --num-shards is passed explicitly (do not rely on SLURM_ARRAY_TASK_COUNT),
# and --shard-index comes from SLURM_ARRAY_TASK_ID inside the task.
MAP_JOB=$(sbatch --parsable \
    --job-name=freq_map \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=6:00:00 \
    --array="0-$((NUM_SHARDS - 1))" \
    --output="logs_freq_matrix/map_%A_%a.out" \
    --mail-type=FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install numpy pandas nltk zstandard orjson
        python -u -c 'import nltk; nltk.download(\"stopwords\")'
        python -u freq_matrix.py \
            --mode map \
            --reddit-monthly-dir '$DUMP_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --shard-dir '$SHARD_DIR' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\" \
            --num-shards $NUM_SHARDS
    ")

echo "Submitted map array: $MAP_JOB (0-$((NUM_SHARDS - 1)))"

# --- Merge (reduce), after all map tasks succeed -----------------------------
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${MAP_JOB}" \
    --job-name=freq_merge \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=32G \
    --time=1:00:00 \
    --output="logs_freq_matrix/merge_%j.out" \
    --mail-type=END,FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install numpy pandas nltk zstandard orjson
        python -u merge_freq_matrix.py \
            --shard-dir '$SHARD_DIR' \
            --out-dir '$OUT_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --stem '$STEM' \
            --dtype '$DTYPE'
    ")

echo "Submitted merge job: $MERGE_JOB"
echo "The merge will start after all map tasks succeed (afterok)."
