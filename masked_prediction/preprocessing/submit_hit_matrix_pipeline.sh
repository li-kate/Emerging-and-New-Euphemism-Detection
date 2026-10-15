#!/bin/bash
set -euo pipefail

# ============================================================================
# Hit-matrix pipeline: pilot selection -> map (SLURM array) -> merge (reduce).
#
#   0. Pilot: pilot_creation.py selects the control words (pilot_words.json)
#      from the frequency matrix + DEA list. Runs once, up front, because every
#      map task reads it. CPU-only and quick, so it runs on the submit node
#      here; move it into its own job if your login node forbids compute.
#   1. Map array: ONE TASK PER MONTH. Each task scores a single month with BERT
#      (hit_matrix.py --mode map) and writes one shard. GPU per task.
#   2. Merge: after all map tasks succeed (afterok), merge_hit_matrix.py sums
#      the shards into the summary table + matrices aligned to the frequency
#      matrix.
#
# Mirrors submit_freq_matrix_pipeline.sh (same month grouping and array sizing);
# the differences are the pilot pre-step and the GPU request on the map array.
#
# PREREQUISITE: the frequency matrix must already be built (freq_dir below),
# since it defines the target words + denominator and its scope hash is checked.
#
# to run:
#   chmod +x submit_hit_matrix_pipeline.sh
#   ./submit_hit_matrix_pipeline.sh
# ============================================================================

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs_hit_matrix

# --- Configuration -----------------------------------------------------------
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"      # must match the frequency build
FREQ_DIR="freq_drug_subreddits"               # existing frequency matrix
FREQ_STEM="freq"
DEA_FILE="../../data_collection/word_lists/drug_synonyms.txt"
PILOT_FILE="pilot_words.json"
OUT_DIR="hits_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="hits"

MODEL_NAME="bert-base-uncased"
BATCH_SIZE=64
MIN_CONTEXTS=30

# Pilot selection knobs (see pilot_creation.py / hit_matrix_plan.md 5b).
PILOT_SEED=0
PILOT_MIN_FREQ=50
PILOT_MIN_PERIODS=8
PILOT_N=30

# SLURM array throttle; see submit_freq_matrix_pipeline.sh for the rationale.
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"

PY_DEPS="numpy pandas nltk zstandard orjson torch transformers"

# --- Sanity: the frequency matrix must exist ---------------------------------
if [[ ! -f "${FREQ_DIR}/${FREQ_STEM}_words.csv" || ! -f "${FREQ_DIR}/${FREQ_STEM}_meta.json" ]]; then
    echo "ERROR: frequency matrix not found in ${FREQ_DIR}"
    echo "       (need ${FREQ_STEM}_words.csv and ${FREQ_STEM}_meta.json)."
    echo "       Build it first with submit_freq_matrix_pipeline.sh."
    exit 1
fi

# --- Step 0: pilot selection (runs once, up front) ---------------------------
echo "Selecting pilot words -> ${PILOT_FILE}"
python -u pilot_creation.py \
    --freq-dir "$FREQ_DIR" \
    --stem "$FREQ_STEM" \
    --dea-file "$DEA_FILE" \
    --out "$PILOT_FILE" \
    --seed "$PILOT_SEED" \
    --min-freq "$PILOT_MIN_FREQ" \
    --min-periods "$PILOT_MIN_PERIODS" \
    --n-pilot "$PILOT_N"

# --- Count distinct months (array size) --------------------------------------
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob
if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi

months=()
while IFS= read -r mkey; do
    months+=("$mkey")
done < <(
    for f in "${dumps[@]}"; do
        base=$(basename "$f")
        echo "${base:0:10}"   # RC_YYYY-MM / RS_YYYY-MM
    done | sort -u
)
NUM_MONTHS=${#months[@]}
echo "Found ${#dumps[@]} dump files across $NUM_MONTHS month(s)."
echo "First/last month: ${months[0]} ... ${months[$((NUM_MONTHS - 1))]}"

# --- Clean stale shards ------------------------------------------------------
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.part.json "$SHARD_DIR"/*.part.json.tmp 2>/dev/null || true

# --- Map array: one GPU task per month ---------------------------------------
ARRAY_SPEC="0-$((NUM_MONTHS - 1))"
if [[ -n "$MAX_CONCURRENT" && "$MAX_CONCURRENT" -gt 0 ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
    echo "Submitting $NUM_MONTHS month tasks, running <= $MAX_CONCURRENT at once."
else
    echo "Submitting $NUM_MONTHS month tasks with no concurrency throttle."
fi

MAP_JOB=$(sbatch --parsable \
    --job-name=hit_map \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --mem=32G \
    --time=8:00:00 \
    --array="$ARRAY_SPEC" \
    --output="logs_hit_matrix/map_%A_%a.out" \
    --mail-type=FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install $PY_DEPS
        python -u hit_matrix.py \
            --mode map \
            --reddit-monthly-dir '$DUMP_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --freq-dir '$FREQ_DIR' \
            --freq-stem '$FREQ_STEM' \
            --dea-file '$DEA_FILE' \
            --pilot-file '$PILOT_FILE' \
            --shard-dir '$SHARD_DIR' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\" \
            --model-name '$MODEL_NAME' \
            --batch-size $BATCH_SIZE \
            --fp16
    ")
echo "Submitted map array: $MAP_JOB ($ARRAY_SPEC) — one GPU task per month"

# --- Merge (reduce), after all map tasks succeed -----------------------------
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${MAP_JOB}" \
    --job-name=hit_merge \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=32G \
    --time=1:00:00 \
    --output="logs_hit_matrix/merge_%j.out" \
    --mail-type=END,FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install $PY_DEPS
        python -u merge_hit_matrix.py \
            --shard-dir '$SHARD_DIR' \
            --freq-dir '$FREQ_DIR' \
            --freq-stem '$FREQ_STEM' \
            --out-dir '$OUT_DIR' \
            --stem '$STEM' \
            --min-contexts $MIN_CONTEXTS
    ")
echo "Submitted merge job: $MERGE_JOB"
echo "The merge will start after all map tasks succeed (afterok)."
