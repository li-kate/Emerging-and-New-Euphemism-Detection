#!/bin/bash
set -euo pipefail

# ============================================================================
# Context-sample experiment pipeline: map (SLURM array, batched) -> merge.
#
# Samples ONE word's full contexts per half-year for manual spot-checking
# (sample_contexts.py), one month per task, then unions the per-month
# reservoirs into <stem>_sample.json (merge_sample_contexts.py). This is a
# SEPARATE, diagnostic experiment; the matrix pipeline does not touch it.
#
# Submit-cap aware and month-filtered exactly like the matrix pipeline.
#
# PREREQUISITES: the frequency matrix (freq_dir) must already exist. Set WORD
# to the target word to sample.
#
# to run:
#   chmod +x submit_sample_contexts_pipeline.sh _sample_batch_controller.sh
#   WORD=molly ./submit_sample_contexts_pipeline.sh
# ============================================================================

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_sample_contexts

# --- Configuration -----------------------------------------------------------
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"
FREQ_DIR="freq_drug_subreddits_unfiltered"
FREQ_STEM="freq"
DEA_FILE="../../data_collection/word_lists/drug_synonyms.txt"
OUT_DIR="sample_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="hits"

MODEL_NAME="../BERT_model_reddit"
START_YEAR=2019
BATCH_SIZE=64

# The word to sample, and the reservoir size/seed.
WORD="${WORD:?Set WORD to the target word to sample, e.g. WORD=molly ./submit_sample_contexts_pipeline.sh}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

MAX_SUBMIT="${MAX_SUBMIT:-15}"

PY_DEPS="numpy pandas nltk zstandard orjson torch transformers"

# --- Sanity: the frequency matrix must exist ---------------------------------
if [[ ! -f "${FREQ_DIR}/${FREQ_STEM}_words.csv" || ! -f "${FREQ_DIR}/${FREQ_STEM}_meta.json" ]]; then
    echo "ERROR: frequency matrix not found in ${FREQ_DIR}"
    echo "       (need ${FREQ_STEM}_words.csv and ${FREQ_STEM}_meta.json)."
    exit 1
fi

# --- Months to score = dump months that are IN the frequency matrix ----------
HY_FILE="${FREQ_DIR}/${FREQ_STEM}_half_years.csv"
if [[ ! -f "$HY_FILE" ]]; then
    echo "ERROR: $HY_FILE not found; cannot restrict months to the matrix."
    exit 1
fi

mapfile -t FREQ_HYS < <(tail -n +2 "$HY_FILE" | tr -d '\r' | sed '/^$/d' | sort -u)
if [[ ${#FREQ_HYS[@]} -eq 0 ]]; then
    echo "ERROR: no half-years listed in $HY_FILE"
    exit 1
fi
FREQ_HY_SET="|$(IFS='|'; echo "${FREQ_HYS[*]}")|"
echo "Frequency matrix covers ${#FREQ_HYS[@]} half-year(s): ${FREQ_HYS[*]}"

shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob
if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi

all_months=()
while IFS= read -r mkey; do
    all_months+=("$mkey")
done < <(
    for f in "${dumps[@]}"; do
        base=$(basename "$f")
        echo "${base:0:10}"
    done | sort -u
)

months=()
for mkey in "${all_months[@]}"; do
    yr="${mkey:3:4}"
    mon="${mkey:8:2}"
    half=$(( 10#$mon <= 6 ? 1 : 2 ))
    hy="${yr}H${half}"
    if [[ "$FREQ_HY_SET" == *"|$hy|"* ]]; then
        months+=("$mkey")
    fi
done

NUM_MONTHS=${#months[@]}
SKIPPED=$(( ${#all_months[@]} - NUM_MONTHS ))
echo "Found ${#dumps[@]} dump files across ${#all_months[@]} dump month(s)."
echo "Keeping $NUM_MONTHS month(s) in the frequency matrix; skipping $SKIPPED outside it."
if [[ $NUM_MONTHS -eq 0 ]]; then
    echo "ERROR: no dump months fall within the frequency matrix's half-years."
    exit 1
fi
echo "First/last kept month: ${months[0]} ... ${months[$((NUM_MONTHS - 1))]}"
echo "Sampling up to $SAMPLE_SIZE contexts per half-year for word '$WORD' (seed $SAMPLE_SEED)."
echo "  -> unioned sample will be written to ${OUT_DIR}/${STEM}_sample.json"

# --- Clean stale shards ------------------------------------------------------
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.sample.json "$SHARD_DIR"/*.sample.json.tmp 2>/dev/null || true

MONTHS_ARG="${months[*]}"

# --- Map: self-chaining batches of <= (MAX_SUBMIT - 1) month tasks -----------
if [[ -z "$MAX_SUBMIT" || "$MAX_SUBMIT" -lt 2 ]]; then
    echo "ERROR: MAX_SUBMIT must be >= 2 (got '$MAX_SUBMIT')."
    exit 1
fi

BATCH_MAX=$(( MAX_SUBMIT - 1 ))
NUM_BATCHES=$(( (NUM_MONTHS + BATCH_MAX - 1) / BATCH_MAX ))
echo "Splitting $NUM_MONTHS kept month(s) into $NUM_BATCHES batch(es) of <= $BATCH_MAX" \
     "(leaving one submit slot for the controller)."

CONTROLLER="$SCRIPT_DIR/_sample_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller script not found or not executable: $CONTROLLER"
    exit 1
fi

export DUMP_DIR SUBREDDIT_FILE FREQ_DIR FREQ_STEM DEA_FILE
export OUT_DIR SHARD_DIR STEM MODEL_NAME START_YEAR BATCH_SIZE PY_DEPS
export BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER MONTHS_ARG
export WORD SAMPLE_SIZE SAMPLE_SEED

BATCH_INDEX=0 "$CONTROLLER"

echo ""
echo "Sample pipeline launched: controller walks $NUM_BATCHES batch(es) -> merge."
echo "At most one batch (<= $BATCH_MAX tasks) + one controller/merge job is queued at a time."
