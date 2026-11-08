#!/bin/bash
set -euo pipefail

# ============================================================================
# Map every MISSING month of the temporal Zhu detection run, then re-merge ALL
# shards.
#
# A "missing" month is one that has a Reddit dump in DUMP_DIR (inside the year
# range) but no shard (<SHARD_DIR>/<month_key>.part.json) yet — i.e. a month
# that never ran or whose map task failed before writing its shard. The map
# step failing on a few months is common and usually NOT a data problem: each
# month is a separate GPU array task, and a task can land on a node whose GPU
# is momentarily busy/unavailable ("CUDA-capable device(s) is/are busy or
# unavailable"), so it dies before the model even loads. Because the pipeline
# chains batches with `afterany`, a failed month does not stall the rest — it
# just leaves no shard, and the merge silently omits it. This script fills
# those gaps.
#
# It maps only the missing months and leaves every existing shard untouched,
# then re-sums all shards (old + new) into the final per-half-year ranked
# lists. Mirrors frequency_matrix/slurm/submit_freq_missing.sh.
#
#   WHY it does not clean the shard dir:
#   submit_zhu_detection_pipeline.sh wipes all shards first (fresh full build).
#   This script intentionally keeps them, so the gaps are filled and merged
#   back with the months already scored.
#
#   "Missing" means NO shard file. A month with a wrong/incomplete shard counts
#   as present and is skipped. Delete that shard first to force a rescore.
#
# Submit-cap aware: like the main pipeline, missing months are mapped in
# batches of <= MAX_SUBMIT-1 tasks via the self-chaining controller, so the
# QOS submit cap is never exceeded. The merge runs after the final batch.
#
# Usage:
#   ./submit_zhu_missing.sh            # map all missing months, then merge
#   ./submit_zhu_missing.sh --dry-run  # just list missing months, submit nothing
#
# Keep the scoring config (ANSWER_FILE, MLM_K, THRES, MAX_CHARS, year range,
# scope) IDENTICAL to the original run, or the merge's agreement check will
# reject the mixed shards.
# ============================================================================

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        *) echo "ERROR: unrecognized argument: '$arg'" >&2
           echo "Usage: $0 [--dry-run]" >&2
           exit 1 ;;
    esac
done

# --- Configuration (match submit_zhu_detection_pipeline.sh) ------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_zhu_detect

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="../../drug_lists/drug_subreddit_list.txt"   # empty for all of Reddit
ANSWER_FILE="../zhu_model/data/euphemism_answer_drug.txt"
MODEL_DIR="../BERT_model_reddit"
VOCAB_FILE="../BERT_model_reddit/vocab.txt"
OUT_DIR="zhu_detect_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="zhu_detect"

START_YEAR=2019
END_YEAR=2026

MLM_K=50
THRES=5
TOP_N=500

MAX_SUBMIT="${MAX_SUBMIT:-15}"
PY_DEPS="numpy nltk zstandard orjson torch pytorch_pretrained_bert"

# --- All dump months inside [START_YEAR, END_YEAR] ---------------------------
# Group RC_/RS_ YYYY-MM dumps (parts share a month), matching group_by_month().
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob
if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR" >&2
    exit 1
fi

all_months=()
while IFS= read -r mkey; do
    all_months+=("$mkey")
done < <(
    for f in "${dumps[@]}"; do
        base=$(basename "$f")
        echo "${base:0:10}"   # RC_YYYY-MM / RS_YYYY-MM
    done | sort -u
)

# Keep only months inside the year range (the main pipeline scores exactly
# these, so only these can be "missing").
kept_months=()
for mkey in "${all_months[@]}"; do
    yr="${mkey:3:4}"
    if (( 10#$yr >= START_YEAR && 10#$yr <= END_YEAR )); then
        kept_months+=("$mkey")
    fi
done
echo "Dump dir has ${#dumps[@]} file(s); ${#kept_months[@]} month(s) in" \
     "[$START_YEAR, $END_YEAR]."

# --- Missing months (in range, dump present, no shard) -----------------------
mkdir -p "$SHARD_DIR"
missing=()
for m in "${kept_months[@]}"; do
    [[ -f "${SHARD_DIR}/${m}.part.json" ]] || missing+=("$m")
done

NUM_MONTHS=${#missing[@]}
if [[ $NUM_MONTHS -eq 0 ]]; then
    echo "No missing months: every in-range dump month already has a shard."
    echo "(If a shard is incomplete, delete it and re-run to rescore that month.)"
    exit 0
fi

echo "Missing $NUM_MONTHS month(s) (in range, dump present, no shard):"
printf '  %s\n' "${missing[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "--dry-run: nothing submitted."
    exit 0
fi

# --- Sanity: model + answer key must exist -----------------------------------
if [[ ! -f "$VOCAB_FILE" ]]; then
    echo "ERROR: BERT vocab not found at $VOCAB_FILE" >&2
    exit 1
fi
if [[ ! -f "$ANSWER_FILE" ]]; then
    echo "ERROR: answer file not found at $ANSWER_FILE" >&2
    exit 1
fi
if [[ -n "$SUBREDDIT_FILE" && ! -f "$SUBREDDIT_FILE" ]]; then
    echo "ERROR: subreddit list not found at $SUBREDDIT_FILE" >&2
    exit 1
fi

# --- Batch the missing months under the submit cap via the controller --------
if [[ -z "$MAX_SUBMIT" || "$MAX_SUBMIT" -lt 2 ]]; then
    echo "ERROR: MAX_SUBMIT must be >= 2 (got '$MAX_SUBMIT')." >&2
    exit 1
fi
BATCH_MAX=$(( MAX_SUBMIT - 1 ))
NUM_BATCHES=$(( (NUM_MONTHS + BATCH_MAX - 1) / BATCH_MAX ))
echo "Mapping $NUM_MONTHS missing month(s) in $NUM_BATCHES batch(es) of" \
     "<= $BATCH_MAX (one submit slot left for the controller/merge)."

CONTROLLER="$SCRIPT_DIR/_zhu_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller not found or not executable: $CONTROLLER" >&2
    exit 1
fi

# The controller indexes months by --shard-index into the list it is handed
# via MONTHS_ARG (passed to zhu_detection_month.py --months). Handing it ONLY
# the missing months means array index N maps to the Nth missing month, so the
# gaps are exactly what gets mapped. Existing shards are left untouched; the
# merge re-sums all of them.
MONTHS_ARG="${missing[*]}"

export DUMP_DIR SUBREDDIT_FILE ANSWER_FILE MODEL_DIR VOCAB_FILE
export OUT_DIR SHARD_DIR STEM START_YEAR MLM_K THRES TOP_N PY_DEPS
export BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER MONTHS_ARG

BATCH_INDEX=0 "$CONTROLLER"

echo ""
echo "Launched: controller walks $NUM_BATCHES batch(es) of missing months," \
     "then merges all shards."

shopt -s nullglob
existing_shards=("$SHARD_DIR"/*.part.json)
shopt -u nullglob
echo "Shards in ${SHARD_DIR} right now: ${#existing_shards[@]} (map adds up to" \
     "$NUM_MONTHS; merge re-sums all). Compare to n_shards_merged in the merge" \
     "log when done."
