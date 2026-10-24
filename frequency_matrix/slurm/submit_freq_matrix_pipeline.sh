#!/bin/bash
set -euo pipefail

# ============================================================================
# Parallel frequency-matrix pipeline: map (SLURM array) + merge (reduce).
#
#   1. Map: ONE TASK PER MONTH. Each task counts a single month of the Reddit
#      dumps (RC_/RS_ YYYY-MM, multi-part pieces grouped) via
#      freq_matrix.py --mode map --split month, and writes one shard.
#   2. Merge: after the last map batch, merge_freq_matrix.py sums the shards,
#      applies the chosen postfilter, and writes the final matrix.
#
# Submit-cap aware: the cluster QOS limits how many jobs a user may have in the
# queue at once (running + pending), and a 278-task array exceeds it. A SLURM
# dependency does not help on its own, because a dependent job is still
# *submitted* (queued) immediately. So this script does NOT submit all months
# up front. It submits the first batch (<= MAX_SUBMIT-1 tasks) plus a small
# self-chaining controller job (_freq_batch_controller.sh) that submits the
# next batch only once the current one has left the queue, and the merge after
# the final batch. At any time at most one batch + one controller/merge job is
# queued, so the submit cap is never exceeded.
#
# Counting is CPU-only, so no GPU is requested.
#
# to run:
#   chmod +x submit_freq_matrix_pipeline.sh _freq_batch_controller.sh
#   ./submit_freq_matrix_pipeline.sh
# ============================================================================

# Resolve the directory holding this script and its companions. When run
# directly this is the script's own folder; when submitted with sbatch, SLURM
# copies the script to a throwaway spool dir, so $0 is useless — use
# $SLURM_SUBMIT_DIR (the directory sbatch was invoked from) instead.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_freq_matrix

# --- Configuration -----------------------------------------------------------
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"   # relative to this folder
OUT_DIR="freq_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="freq"
DTYPE="int32"

# Postfilter (applied once, at the merge step). The map tasks never
# postfilter — it is a global threshold and runs after the shards are summed.
#   POSTFILTER_MODE=per-half-year  null cells < MIN_FREQ in their own
#                                  half-year, drop words nulled in every
#                                  half-year (leakage-safe; default).
#   POSTFILTER_MODE=global         legacy: keep a word if any half-year
#                                  reaches MIN_FREQ, leave its cells as-is.
#   NO_POSTFILTER=1                skip the postfilter entirely (overrides
#                                  POSTFILTER_MODE); keep every word and raw
#                                  count. THIS IS THE DEFAULT — the first
#                                  build is intentionally unfiltered so the
#                                  postfilter can be chosen/applied later.
#                                  Set NO_POSTFILTER=0 to enable POSTFILTER_MODE.
#   MIN_FREQ                       threshold used by both modes (default 30,
#                                  freq_matrix.py's MIN_FREQ).
POSTFILTER_MODE="${POSTFILTER_MODE:-per-half-year}"
NO_POSTFILTER="${NO_POSTFILTER:-1}"
MIN_FREQ="${MIN_FREQ:-}"

# Assemble the postfilter CLI args passed to the merge step.
POSTFILTER_ARGS=""
if [[ "$NO_POSTFILTER" == "1" ]]; then
    POSTFILTER_ARGS="--no-postfilter"
else
    POSTFILTER_ARGS="--postfilter-mode $POSTFILTER_MODE"
fi
if [[ -n "$MIN_FREQ" ]]; then
    POSTFILTER_ARGS="$POSTFILTER_ARGS --min-freq $MIN_FREQ"
fi
echo "Postfilter (merge step): ${POSTFILTER_ARGS:-<default>}"

# Sample this many removed terms per filter bucket (examples only, logged and
# saved to freq_meta.json). 0 = off. Each map task samples its own month;
# the merge shows the same count, pooled from the per-shard samples.
SAMPLE_REMOVED="${SAMPLE_REMOVED:-50}"

# QOS submit cap: the MOST jobs this user may have in the queue at once
# (running + pending). One array task counts as one queued job, so an array
# bigger than this is rejected at submit time with
# QOSMaxSubmitJobPerUserLimit — which is why all 278 months cannot be
# submitted as a single array.
#
# To stay under the cap, the months are split into consecutive BATCHES of at
# most MAX_SUBMIT tasks. Each batch is one array; batch N+1 is submitted with
# a dependency on batch N finishing (afterany, so one bad month does not stall
# the rest), so only one batch (<= MAX_SUBMIT jobs) is ever in the queue. The
# merge then depends on the final batch. Within a batch, the "%" throttle
# still starts the next task as soon as one finishes.
#
# Default 10 stays well under the ICE QOS limit (coc-ice / coe-ice). Raise it
# toward the real cap for fewer batches, or lower it to be a good neighbor; it
# must stay <= the real cap.
MAX_SUBMIT="${MAX_SUBMIT:-10}"

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

# --- Map: self-chaining batches of <= MAX_SUBMIT month tasks -----------------
# IMPORTANT: a dependency only delays when a job *runs* — a dependent job is
# still *submitted* immediately and still counts against the submit cap. So we
# cannot submit all batches up front, even chained. Instead we submit only the
# FIRST batch now, plus a tiny "controller" job that, when that batch finishes,
# submits the next batch (and its own successor), and finally the merge. At any
# moment the queue holds at most one map batch (<= MAX_SUBMIT) plus the single
# pending controller/merge job, so the submit cap is respected.
#
# The array index is the absolute month index (--shard-index selects the Nth
# month in freq_matrix.py), so batches are contiguous ranges: 0-49, 50-99, ...
if [[ -z "$MAX_SUBMIT" || "$MAX_SUBMIT" -lt 2 ]]; then
    # Need room for the batch array AND one controller/merge job under the cap.
    echo "ERROR: MAX_SUBMIT must be >= 2 (got '$MAX_SUBMIT')."
    exit 1
fi

# The map array of a batch may use at most (MAX_SUBMIT - 1) tasks, leaving one
# slot for the pending controller (next batch) or merge job.
BATCH_MAX=$(( MAX_SUBMIT - 1 ))
NUM_BATCHES=$(( (NUM_MONTHS + BATCH_MAX - 1) / BATCH_MAX ))
echo "Splitting $NUM_MONTHS months into $NUM_BATCHES batch(es) of <= $BATCH_MAX" \
     "(leaving one submit slot for the controller)."

CONTROLLER="$SCRIPT_DIR/_freq_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller script not found or not executable: $CONTROLLER"
    echo "       (expected alongside this pipeline script, in \$SLURM_SUBMIT_DIR"
    echo "        when run via sbatch). chmod +x it if it exists."
    exit 1
fi

# Export everything the controller needs so each chained invocation is
# self-contained (it runs on a compute node with only these in its env).
export DUMP_DIR SUBREDDIT_FILE OUT_DIR SHARD_DIR STEM DTYPE
export POSTFILTER_ARGS SAMPLE_REMOVED BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER

# Kick off batch 0. The controller submits the batch array, then chains the
# next controller invocation (or the merge) after it.
BATCH_INDEX=0 "$CONTROLLER"

echo "Pipeline launched: controller will walk $NUM_BATCHES batch(es), then merge."
echo "At most one batch (<= $BATCH_MAX tasks) + one controller/merge job is queued at a time."
