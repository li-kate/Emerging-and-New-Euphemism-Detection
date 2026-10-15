#!/bin/bash
set -euo pipefail

# ============================================================================
# Parallel frequency-matrix pipeline: map (SLURM array) + merge (reduce).
#
#   1. Map array: ONE TASK PER MONTH. Each task counts a single month of the
#      Reddit dumps (RC_/RS_ YYYY-MM, multi-part pieces grouped) via
#      freq_matrix.py --mode map --split month, and writes one shard.
#   2. Merge: after all map tasks succeed (afterok), merge_freq_matrix.py sums
#      the shards, applies the postfilter, and writes the final matrix.
#
# The array is sized to the number of months, so submitting once runs all
# months in parallel. Modeled on submit_drug_centroid_pipeline.sh. Counting is
# CPU-only, so no GPU is requested.
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

# Optional cap on how many array tasks run at once (not how many months). This
# is the SLURM array throttle "%N". Empty means no throttle. Override with e.g.
#   MAX_CONCURRENT=20 ./submit_freq_matrix_pipeline.sh
MAX_CONCURRENT="${MAX_CONCURRENT:-}"

# --- Count distinct months ---------------------------------------------------
# Group dump files by RC_/RS_ YYYY-MM prefix (parts share a month), matching
# group_by_month() in freq_matrix.py. NUM_MONTHS == number of array tasks.
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob

if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi

# Extract the "RC_YYYY-MM" / "RS_YYYY-MM" prefix from each basename, dedupe.
months=()
while IFS= read -r mkey; do
    months+=("$mkey")
done < <(
    for f in "${dumps[@]}"; do
        base=$(basename "$f")
        # RC_2018-07_part000.zst -> RC_2018-07
        echo "${base:0:10}"
    done | sort -u
)

NUM_MONTHS=${#months[@]}
echo "Found ${#dumps[@]} dump files across $NUM_MONTHS month(s)."
echo "First/last month: ${months[0]} ... ${months[$((NUM_MONTHS - 1))]}"

# --- Clean stale shards ------------------------------------------------------
# Start fresh so a rerun never merges stale shards from a previous attempt.
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.part.json "$SHARD_DIR"/*.part.json.tmp 2>/dev/null || true

# --- Map array: one task per month -------------------------------------------
ARRAY_SPEC="0-$((NUM_MONTHS - 1))"
if [[ -n "$MAX_CONCURRENT" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

MAP_JOB=$(sbatch --parsable \
    --job-name=freq_map \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=6:00:00 \
    --array="$ARRAY_SPEC" \
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
            --split month \
            --reddit-monthly-dir '$DUMP_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --shard-dir '$SHARD_DIR' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\"
    ")

echo "Submitted map array: $MAP_JOB ($ARRAY_SPEC) — one task per month"

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
