#!/bin/bash
set -euo pipefail

# ============================================================================
# Hit-matrix pipeline: map (SLURM array, batched) -> merge.
#
#   1. Map: ONE TASK PER MONTH. Each task scores a single month with BERT
#      (hit_matrix.py --mode map) and writes one shard. GPU per task.
#   2. Merge: after the final map batch leaves the queue, merge_hit_matrix.py
#      sums the shards into the summary table + matrices aligned to the
#      frequency matrix.
#
# This is the MATRIX-ONLY pipeline: it builds the hit matrix and nothing else.
# The pilot per-context table and the manual spot-check context sample are
# separate experiments with their own pipelines (submit_pilot_hits_pipeline.sh,
# submit_sample_contexts_pipeline.sh).
#
# Submit-cap aware (like submit_freq_matrix_pipeline.sh). The cluster QOS caps
# how many jobs a user may have in the queue at once, and a dependent job is
# still *submitted* (queued) immediately — so the kept months cannot be thrown
# at the scheduler as one big array if that array exceeds the cap. Instead the
# months are split into consecutive BATCHES of at most MAX_SUBMIT-1 tasks: this
# script submits the first batch, and a tiny self-chaining controller
# (_hit_batch_controller.sh) submits each later batch only once the previous one
# has left the queue, then the merge after the last batch. At any time at most
# one batch + one controller/merge job is in the queue, so the submit cap is
# never exceeded.
#
# Nothing in this pipeline runs Python on the submit node; the only submit-node
# work is globbing dump filenames and filtering months (no Python needed).
#
# Mirrors submit_freq_matrix_pipeline.sh + _freq_batch_controller.sh (same
# month grouping, same batching). The hit-matrix differences: a GPU request on
# each map task, and restricting the months to those whose half-year is in the
# frequency matrix.
#
# PREREQUISITES: the frequency matrix must already be built (freq_dir below),
# since it defines the target words + denominator and its scope hash is checked.
#
# to run:
#   chmod +x submit_hit_matrix_pipeline.sh _hit_batch_controller.sh
#   ./submit_hit_matrix_pipeline.sh
# ============================================================================

# Resolve the directory holding this script and its companions. When run
# directly this is the script's own folder; when submitted with sbatch, SLURM
# copies the script to a throwaway spool dir, so $0 is useless — use
# $SLURM_SUBMIT_DIR (the directory sbatch was invoked from) instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_hit_matrix

# --- Configuration -----------------------------------------------------------
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"      # must match the frequency build
FREQ_DIR="freq_drug_subreddits_unfiltered"               # existing frequency matrix
FREQ_STEM="freq"
DEA_FILE="../../data_collection/word_lists/drug_synonyms.txt"
OUT_DIR="hits_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="hits"

MODEL_NAME="../BERT_model_reddit"
START_YEAR=2019
BATCH_SIZE=64

# QOS submit cap: the MOST jobs this user may have in the queue at once
# (running + pending). One array task counts as one queued job, so an array
# bigger than this is rejected at submit time with QOSMaxSubmitJobPerUserLimit.
# To stay under it, the kept months are split into consecutive batches of at
# most MAX_SUBMIT-1 tasks (one slot is left for the chained controller/merge
# job), and _hit_batch_controller.sh submits them one at a time. Within a
# batch the "%" throttle starts the next GPU task as soon as one finishes.
#
# Default 10 stays well under the ICE QOS limit; raise toward the real cap for
# fewer batches (more GPUs used at once), or lower it to be a good neighbor. It
# must stay <= the real cap.
MAX_SUBMIT="${MAX_SUBMIT:-15}"

PY_DEPS="numpy pandas nltk zstandard orjson torch transformers"

# --- Sanity: the frequency matrix and pilot config must exist ----------------
if [[ ! -f "${FREQ_DIR}/${FREQ_STEM}_words.csv" || ! -f "${FREQ_DIR}/${FREQ_STEM}_meta.json" ]]; then
    echo "ERROR: frequency matrix not found in ${FREQ_DIR}"
    echo "       (need ${FREQ_STEM}_words.csv and ${FREQ_STEM}_meta.json)."
    echo "       Build it first with submit_freq_matrix_pipeline.sh."
    exit 1
fi
# --- Months to score = dump months that are IN the frequency matrix ----------
# The frequency matrix was built on a specific scope and year range; its
# freq_half_years.csv lists exactly the half-years it covers. Scoring a month
# whose half-year is not in that file is wasted work (its hit cells would have
# no frequency denominator and be masked anyway) and needlessly inflates the
# array toward the QOS submit cap. So: glob the dump months, map each to its
# half-year (months 1-6 -> H1, 7-12 -> H2, matching corpus_stream.half_year_label),
# and keep only those whose half-year appears in the frequency matrix.
HY_FILE="${FREQ_DIR}/${FREQ_STEM}_half_years.csv"
if [[ ! -f "$HY_FILE" ]]; then
    echo "ERROR: $HY_FILE not found; cannot restrict months to the matrix."
    exit 1
fi

# Half-years present in the frequency matrix (skip the CSV header).
mapfile -t FREQ_HYS < <(tail -n +2 "$HY_FILE" | tr -d '\r' | sed '/^$/d' | sort -u)
if [[ ${#FREQ_HYS[@]} -eq 0 ]]; then
    echo "ERROR: no half-years listed in $HY_FILE"
    exit 1
fi
# Build a lookup set "|2012H1|2012H2|..." for cheap membership tests.
FREQ_HY_SET="|$(IFS='|'; echo "${FREQ_HYS[*]}")|"
echo "Frequency matrix covers ${#FREQ_HYS[@]} half-year(s): ${FREQ_HYS[*]}"

shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob
if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi

# Distinct dump months (RC_/RS_ YYYY-MM), then filter to the matrix's half-years.
all_months=()
while IFS= read -r mkey; do
    all_months+=("$mkey")
done < <(
    for f in "${dumps[@]}"; do
        base=$(basename "$f")
        echo "${base:0:10}"   # RC_YYYY-MM / RS_YYYY-MM
    done | sort -u
)

months=()
for mkey in "${all_months[@]}"; do
    yr="${mkey:3:4}"      # YYYY
    mon="${mkey:8:2}"     # MM
    # Gate on START_YEAR as well as the frequency matrix's half-years. The freq
    # matrix can cover earlier half-years (e.g. 2015) than we want to score;
    # hit_matrix.py drops every pre-START_YEAR record anyway (stream_corpus
    # "year" filter), so scoring those months just wastes GPU tasks on empty
    # shards. Skip them at selection time.
    if (( 10#$yr < START_YEAR )); then
        continue
    fi
    # strip a leading zero so "06" -> 6 for the arithmetic compare
    half=$(( 10#$mon <= 6 ? 1 : 2 ))
    hy="${yr}H${half}"
    if [[ "$FREQ_HY_SET" == *"|$hy|"* ]]; then
        months+=("$mkey")
    fi
done

NUM_MONTHS=${#months[@]}
SKIPPED=$(( ${#all_months[@]} - NUM_MONTHS ))
echo "Found ${#dumps[@]} dump files across ${#all_months[@]} dump month(s)."
echo "Keeping $NUM_MONTHS month(s) in the frequency matrix and >= $START_YEAR;" \
     "skipping $SKIPPED (outside the matrix's half-years or before $START_YEAR)."
if [[ $NUM_MONTHS -eq 0 ]]; then
    echo "ERROR: no dump months fall within the frequency matrix's half-years."
    echo "       Check that DUMP_DIR and the frequency build cover the same period."
    exit 1
fi
echo "First/last kept month: ${months[0]} ... ${months[$((NUM_MONTHS - 1))]}"

# --- Clean stale shards ------------------------------------------------------
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.part.json "$SHARD_DIR"/*.part.json.tmp 2>/dev/null || true

# The map array must index into EXACTLY the kept months (not re-glob all dump
# months), so pass the kept list to hit_matrix.py --months. select_month_key()
# then restricts to this list and the array index selects the Nth kept month,
# keeping the array index <-> month mapping correct. The batch controller
# turns a batch's array range into the matching slice of this list.
MONTHS_ARG="${months[*]}"

# --- Map: self-chaining batches of <= (MAX_SUBMIT - 1) month tasks -----------
# A dependency only delays when a job *runs* — a dependent job is still
# *submitted* immediately and still counts against the submit cap. So we cannot
# submit all batches up front, even chained. We submit only batch 0 here, plus
# the controller chains each later batch once its predecessor leaves the queue,
# and finally the merge. At any moment the queue holds at most one map batch
# (<= BATCH_MAX) plus one controller/merge job, respecting the submit cap.
if [[ -z "$MAX_SUBMIT" || "$MAX_SUBMIT" -lt 2 ]]; then
    # Need room for the batch array AND one controller/merge job under the cap.
    echo "ERROR: MAX_SUBMIT must be >= 2 (got '$MAX_SUBMIT')."
    exit 1
fi

# A batch's array may use at most (MAX_SUBMIT - 1) tasks, leaving one submit
# slot for the pending controller (next batch) or merge job.
BATCH_MAX=$(( MAX_SUBMIT - 1 ))
NUM_BATCHES=$(( (NUM_MONTHS + BATCH_MAX - 1) / BATCH_MAX ))
echo "Splitting $NUM_MONTHS kept month(s) into $NUM_BATCHES batch(es) of <= $BATCH_MAX" \
     "(leaving one submit slot for the controller)."

CONTROLLER="$SCRIPT_DIR/_hit_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller script not found or not executable: $CONTROLLER"
    echo "       (expected alongside this pipeline script, in \$SLURM_SUBMIT_DIR"
    echo "        when run via sbatch). chmod +x it if it exists."
    exit 1
fi

# Export everything the controller needs so each chained invocation is
# self-contained (it runs on a compute node with only these in its env).
export DUMP_DIR SUBREDDIT_FILE FREQ_DIR FREQ_STEM DEA_FILE
export OUT_DIR SHARD_DIR STEM MODEL_NAME START_YEAR BATCH_SIZE PY_DEPS
export BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER MONTHS_ARG

# Kick off batch 0. The controller submits the batch array, then chains the
# next controller invocation (or the merge) after it.
BATCH_INDEX=0 "$CONTROLLER"

echo ""
echo "Pipeline launched: controller walks $NUM_BATCHES batch(es) -> merge."
echo "At most one batch (<= $BATCH_MAX tasks) + one controller/merge job is queued at a time."
