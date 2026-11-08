#!/bin/bash
set -euo pipefail

# ============================================================================
# Temporal Zhu euphemism-detection pipeline: map (SLURM array, batched) -> merge.
#
#   1. Map: ONE TASK PER MONTH. Each task scores a single month with the
#      Reddit-tuned BERT masked LM (zhu_detection_month.py) and writes one
#      shard holding per-half-year candidate score sums. GPU per task.
#   2. Merge: after the final map batch leaves the queue, merge_zhu_detection.py
#      sums the shards per half-year and writes the ranked candidate list for
#      each half-year.
#
# This runs Zhu et al.'s DETECTION stage over time: instead of one ranked
# candidate list for the whole corpus, you get one per half-year, so emerging
# euphemisms can be tracked period by period. The identification stage
# (classification_model.py / identification.py) is downstream and not run here.
#
# zhu_model is NOT modified: the model dir + vocab are passed in and the MLM /
# informativeness filter is reproduced inside zhu_detection_month.py.
#
# Submit-cap aware, exactly like submit_hit_matrix_pipeline.sh. The cluster QOS
# caps how many jobs a user may queue at once, and a dependent job is still
# *submitted* immediately — so the months cannot be thrown at the scheduler as
# one big array if it exceeds the cap. The months are split into consecutive
# BATCHES of at most MAX_SUBMIT-1 tasks: this script submits the first batch,
# and _zhu_batch_controller.sh submits each later batch only once the previous
# one has left the queue, then the merge after the last batch. At any time at
# most one batch + one controller/merge job is queued, so the cap holds.
#
# Nothing here runs Python on the submit node; the only submit-node work is
# globbing dump filenames and filtering months by year (no Python needed).
#
# Unlike the hit matrix, there is no frequency matrix, so months are NOT
# restricted to a precomputed half-year axis — every dump month inside
# [START_YEAR, END_YEAR] is scored, and the half-year buckets fall out of the
# data at merge time.
#
# to run:
#   chmod +x submit_zhu_detection_pipeline.sh _zhu_batch_controller.sh
#   ./submit_zhu_detection_pipeline.sh
# ============================================================================

# Resolve the directory holding this script and its companions. When run
# directly this is the script's own folder; when submitted with sbatch, SLURM
# copies the script to a throwaway spool dir, so $0 is useless — use
# $SLURM_SUBMIT_DIR (the directory sbatch was invoked from) instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_zhu_detect

# --- Configuration -----------------------------------------------------------
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="/home/hice1/kli605/scratch/Emerging-and-New-Euphemism-Detection/masked_prediction/preprocessing/drug_subreddit_list.txt"   # empty for all of Reddit
# Zhu's euphemism answer key. Input keywords are derived from it exactly as
# zhu_model/read_file.py does (the formal names before each ':'), so the masked
# sentences match a real Zhu detection run.
ANSWER_FILE="../zhu_model/data/euphemism_answer_drug.txt"
MODEL_DIR="../BERT_model_reddit"                    # Zhu's Reddit-tuned BERT
VOCAB_FILE="../BERT_model_reddit/vocab.txt"
OUT_DIR="zhu_detect_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="zhu_detect"

START_YEAR=2019
END_YEAR=2026

# Zhu scoring knobs (defaults match zhu_model/detection.py).
MLM_K=50          # top-k predictions per mask (Zhu MLM_k=50)
THRES=5           # informativeness threshold (Zhu euphemism_detection thres=5)
TOP_N=500         # keep top-N candidates per half-year in the merge outputs

# QOS submit cap: the MOST jobs this user may have in the queue at once
# (running + pending). One array task counts as one queued job, so an array
# bigger than this is rejected at submit time. The months are split into
# consecutive batches of at most MAX_SUBMIT-1 tasks (one slot left for the
# chained controller/merge job), submitted one batch at a time. Within a batch
# the "%" throttle starts the next GPU task as soon as one finishes.
MAX_SUBMIT="${MAX_SUBMIT:-15}"

PY_DEPS="numpy nltk zstandard orjson torch pytorch_pretrained_bert"

# --- Sanity: the model + keywords must exist ---------------------------------
if [[ ! -f "$VOCAB_FILE" ]]; then
    echo "ERROR: BERT vocab not found at $VOCAB_FILE"
    echo "       Point VOCAB_FILE at zhu's BERT_model_reddit/vocab.txt."
    exit 1
fi
if [[ ! -f "$ANSWER_FILE" ]]; then
    echo "ERROR: answer file not found at $ANSWER_FILE"
    echo "       Point ANSWER_FILE at zhu's data/euphemism_answer_<cat>.txt."
    exit 1
fi
# SUBREDDIT_FILE may be empty (= all of Reddit); only check when set.
if [[ -n "$SUBREDDIT_FILE" && ! -f "$SUBREDDIT_FILE" ]]; then
    echo "ERROR: subreddit list not found at $SUBREDDIT_FILE"
    echo "       Point SUBREDDIT_FILE at the scope list (e.g."
    echo "       ../../drug_lists/drug_subreddit_list.txt), or leave it empty"
    echo "       to score all of Reddit."
    exit 1
fi

# --- Months to score = every dump month inside [START_YEAR, END_YEAR] --------
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob
if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi

# Distinct dump months (RC_/RS_ YYYY-MM), then keep those within the year range.
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
    if (( 10#$yr >= START_YEAR && 10#$yr <= END_YEAR )); then
        months+=("$mkey")
    fi
done

NUM_MONTHS=${#months[@]}
SKIPPED=$(( ${#all_months[@]} - NUM_MONTHS ))
echo "Found ${#dumps[@]} dump files across ${#all_months[@]} dump month(s)."
echo "Keeping $NUM_MONTHS month(s) in [$START_YEAR, $END_YEAR]; skipping $SKIPPED outside it."
if [[ $NUM_MONTHS -eq 0 ]]; then
    echo "ERROR: no dump months fall within [$START_YEAR, $END_YEAR]."
    exit 1
fi
echo "First/last kept month: ${months[0]} ... ${months[$((NUM_MONTHS - 1))]}"

# --- Clean stale shards ------------------------------------------------------
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.part.json "$SHARD_DIR"/*.part.json.tmp 2>/dev/null || true

# The map array must index into EXACTLY the kept months (not re-glob all dump
# months), so pass the kept list to zhu_detection_month.py --months.
# select_month_key() then restricts to this list and the array index selects
# the Nth kept month. The batch controller turns a batch's array range into the
# matching slice of this list.
MONTHS_ARG="${months[*]}"

# --- Map: self-chaining batches of <= (MAX_SUBMIT - 1) month tasks -----------
if [[ -z "$MAX_SUBMIT" || "$MAX_SUBMIT" -lt 2 ]]; then
    echo "ERROR: MAX_SUBMIT must be >= 2 (got '$MAX_SUBMIT')."
    exit 1
fi

# A batch's array may use at most (MAX_SUBMIT - 1) tasks, leaving one submit
# slot for the pending controller (next batch) or merge job.
BATCH_MAX=$(( MAX_SUBMIT - 1 ))
NUM_BATCHES=$(( (NUM_MONTHS + BATCH_MAX - 1) / BATCH_MAX ))
echo "Splitting $NUM_MONTHS kept month(s) into $NUM_BATCHES batch(es) of <= $BATCH_MAX" \
     "(leaving one submit slot for the controller)."

CONTROLLER="$SCRIPT_DIR/_zhu_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller script not found or not executable: $CONTROLLER"
    echo "       (expected alongside this pipeline script, in \$SLURM_SUBMIT_DIR"
    echo "        when run via sbatch). chmod +x it if it exists."
    exit 1
fi

# Export everything the controller needs so each chained invocation is
# self-contained (it runs on a compute node with only these in its env).
export DUMP_DIR SUBREDDIT_FILE ANSWER_FILE MODEL_DIR VOCAB_FILE
export OUT_DIR SHARD_DIR STEM START_YEAR MLM_K THRES TOP_N PY_DEPS
export BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER MONTHS_ARG

# Kick off batch 0. The controller submits the batch array, then chains the
# next controller invocation (or the merge) after it.
BATCH_INDEX=0 "$CONTROLLER"

echo ""
echo "Pipeline launched: controller walks $NUM_BATCHES batch(es) -> merge."
echo "At most one batch (<= $BATCH_MAX tasks) + one controller/merge job is queued at a time."
