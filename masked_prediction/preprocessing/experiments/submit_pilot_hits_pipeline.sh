#!/bin/bash
set -euo pipefail

# ============================================================================
# Pilot experiment pipeline: map (SLURM array, batched) -> merge.
#
# Scores the pilot/control words per-context (pilot_hits.py), one month per
# task, then concatenates the shards into <stem>_pilot.csv (merge_pilot_hits.py).
# This is a SEPARATE experiment from the hit matrix; the matrix pipeline
# (submit_hit_matrix_pipeline.sh) does not touch it. They share the scoring
# core, so pilot numbers match the matrix.
#
# Submit-cap aware and month-filtered exactly like the matrix pipeline: the
# months kept are those whose half-year is in the frequency matrix, split into
# batches of at most MAX_SUBMIT-1 tasks, walked by _pilot_batch_controller.sh.
#
# PREREQUISITES: the frequency matrix (freq_dir) and pilot_words.json must
# already exist.
#
# to run:
#   chmod +x submit_pilot_hits_pipeline.sh _pilot_batch_controller.sh
#   ./submit_pilot_hits_pipeline.sh
# ============================================================================

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_pilot_hits

# --- Configuration -----------------------------------------------------------
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"
FREQ_DIR="freq_drug_subreddits_unfiltered"
FREQ_STEM="freq"
DEA_FILE="../../data_collection/word_lists/drug_synonyms.txt"
PILOT_FILE="pilot_words.json"
OUT_DIR="pilot_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="hits"

MODEL_NAME="../BERT_model_reddit"
START_YEAR=2019
BATCH_SIZE=64

MAX_SUBMIT="${MAX_SUBMIT:-15}"

PY_DEPS="numpy pandas nltk zstandard orjson torch transformers"

# --- Sanity: the frequency matrix and pilot config must exist ----------------
if [[ ! -f "${FREQ_DIR}/${FREQ_STEM}_words.csv" || ! -f "${FREQ_DIR}/${FREQ_STEM}_meta.json" ]]; then
    echo "ERROR: frequency matrix not found in ${FREQ_DIR}"
    echo "       (need ${FREQ_STEM}_words.csv and ${FREQ_STEM}_meta.json)."
    exit 1
fi
if [[ ! -f "$PILOT_FILE" ]]; then
    echo "ERROR: pilot config not found: $PILOT_FILE"
    echo "       Build it first (pilot_creation.py) before running this pipeline."
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

# --- Clean stale shards ------------------------------------------------------
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.pilot.json "$SHARD_DIR"/*.pilot.json.tmp 2>/dev/null || true

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

CONTROLLER="$SCRIPT_DIR/_pilot_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller script not found or not executable: $CONTROLLER"
    exit 1
fi

export DUMP_DIR SUBREDDIT_FILE FREQ_DIR FREQ_STEM DEA_FILE PILOT_FILE
export OUT_DIR SHARD_DIR STEM MODEL_NAME START_YEAR BATCH_SIZE PY_DEPS
export BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER MONTHS_ARG

BATCH_INDEX=0 "$CONTROLLER"

echo ""
echo "Pilot pipeline launched: controller walks $NUM_BATCHES batch(es) -> merge."
echo "At most one batch (<= $BATCH_MAX tasks) + one controller/merge job is queued at a time."
