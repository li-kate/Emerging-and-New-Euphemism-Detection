#!/bin/bash
set -euo pipefail

# ============================================================================
# TEST pipeline: run the frequency-matrix map+merge on only the FIRST 2 MONTHS.
#
# Same shape as submit_freq_matrix_pipeline.sh but:
#   * array is 0-1 (two map tasks, one per month)
#   * freq_matrix.py gets --max-months 2 so the month index space is capped
#   * writes to a separate test output dir so a real run is never clobbered
#
# Use this to sanity-check the whole flow (and your SLURM limits) cheaply
# before launching all months.
#
# to run:
#   chmod +x test_freq_matrix_pipeline.sh
#   ./test_freq_matrix_pipeline.sh
# ============================================================================

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs_freq_matrix_test

# --- Configuration -----------------------------------------------------------
# Scripts live one directory up (the preprocessing folder).
SCRIPT_DIR=".."
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
# Reuse the subreddit list from the preprocessing folder.
SUBREDDIT_FILE="../drug_subreddit_list.txt"
OUT_DIR="freq_drug_subreddits_test2"
SHARD_DIR="${OUT_DIR}/shards"
STEM="freq"
DTYPE="int32"

# How many months to test, and the array size that must match it.
MAX_MONTHS="${MAX_MONTHS:-2}"

# --- Sanity: dumps exist -----------------------------------------------------
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob
if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi
echo "Found ${#dumps[@]} dump files; testing the first $MAX_MONTHS month(s)."

# --- Clean stale shards ------------------------------------------------------
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.part.json "$SHARD_DIR"/*.part.json.tmp 2>/dev/null || true

# --- Map array: one task per test month --------------------------------------
MAP_JOB=$(sbatch --parsable \
    --job-name=freq_map_test \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=6:00:00 \
    --array="0-$((MAX_MONTHS - 1))" \
    --output="logs_freq_matrix_test/map_%A_%a.out" \
    --mail-type=FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install numpy pandas nltk zstandard orjson
        python -u -c 'import nltk; nltk.download(\"stopwords\")'
        python -u '$SCRIPT_DIR/freq_matrix.py' \
            --mode map \
            --split month \
            --max-months $MAX_MONTHS \
            --reddit-monthly-dir '$DUMP_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --shard-dir '$SHARD_DIR' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\"
    ")

echo "Submitted test map array: $MAP_JOB (0-$((MAX_MONTHS - 1)))"

# --- Merge (reduce), after the test map tasks succeed ------------------------
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${MAP_JOB}" \
    --job-name=freq_merge_test \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=32G \
    --time=1:00:00 \
    --output="logs_freq_matrix_test/merge_%j.out" \
    --mail-type=END,FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install numpy pandas nltk zstandard orjson
        python -u '$SCRIPT_DIR/merge_freq_matrix.py' \
            --shard-dir '$SHARD_DIR' \
            --out-dir '$OUT_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --stem '$STEM' \
            --dtype '$DTYPE'
    ")

echo "Submitted test merge job: $MERGE_JOB"
echo "Merge starts after both test map tasks succeed (afterok)."
echo "Test outputs -> $OUT_DIR/"
