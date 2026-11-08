#!/bin/bash
set -euo pipefail

# ============================================================================
# Map every MISSING month of the frequency matrix, then re-merge ALL shards.
#
# A "missing" month is one that has a Reddit dump in DUMP_DIR but no shard
# (<SHARD_DIR>/<month_key>.part.json) yet — i.e. a month that never ran or
# whose map task failed before writing its shard. This script maps only those
# months and leaves every existing shard untouched, then re-sums all shards
# (old + new) into the final matrix.
#
#   WHY it does not clean the shard dir:
#   submit_freq_matrix_pipeline.sh wipes all shards first (fresh full build).
#   This script intentionally keeps them, so the gaps are filled and merged
#   back with the months already counted.
#
#   "Missing" means NO shard file. A month with a wrong/incomplete shard counts
#   as present and is skipped. Delete that shard first to force a recount.
#
# Submit-cap aware: like the main pipeline, missing months are mapped in
# batches of <= MAX_SUBMIT-1 tasks via the self-chaining controller, so the
# QOS submit cap is never exceeded even if many months are missing. The merge
# runs after the final batch.
#
# Usage:
#   ./submit_freq_missing.sh            # map all missing months, then merge
#   ./submit_freq_missing.sh --dry-run  # just list missing months, submit nothing
#
# Postfilter at the merge step uses the same env vars as the main pipeline
# (NO_POSTFILTER / POSTFILTER_MODE / MIN_FREQ). Keep them identical to the
# original run so the re-merged matrix is filtered the same way.
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

# --- Configuration (match submit_freq_matrix_pipeline.sh) --------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_freq_matrix

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"   # relative to this folder
OUT_DIR="freq_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="freq"
DTYPE="int32"
SAMPLE_REMOVED="${SAMPLE_REMOVED:-50}"
MAX_SUBMIT="${MAX_SUBMIT:-10}"

# Postfilter (applied once, at the merge step). See the main pipeline for the
# meaning of each; NO_POSTFILTER=1 is the default there and here.
POSTFILTER_MODE="${POSTFILTER_MODE:-per-half-year}"
NO_POSTFILTER="${NO_POSTFILTER:-1}"
MIN_FREQ="${MIN_FREQ:-}"

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

# --- All months present in the dump dir --------------------------------------
# Group RC_/RS_ YYYY-MM dumps (parts share a month), matching group_by_month()
# in freq_matrix.py.
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
        echo "${base:0:10}"   # RC_2015-03_part000.zst -> RC_2015-03
    done | sort -u
)
echo "Dump dir has ${#dumps[@]} file(s) across ${#all_months[@]} month(s)."

# --- Missing months (dump present, no shard) ---------------------------------
mkdir -p "$SHARD_DIR"
missing=()
for m in "${all_months[@]}"; do
    [[ -f "${SHARD_DIR}/${m}.part.json" ]] || missing+=("$m")
done

NUM_MONTHS=${#missing[@]}
if [[ $NUM_MONTHS -eq 0 ]]; then
    echo "No missing months: every dump month already has a shard. Nothing to do."
    echo "(If a shard is incomplete, delete it and re-run to recount that month.)"
    exit 0
fi

echo "Missing $NUM_MONTHS month(s) (dump present, no shard):"
printf '  %s\n' "${missing[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "--dry-run: nothing submitted."
    exit 0
fi

# The map tasks select their month by array index into this explicit --input
# list, so build the list of dump files whose month is missing. --split month
# regroups multi-part months into one shard.
keep=()
for f in "${dumps[@]}"; do
    base=$(basename "$f"); mkey="${base:0:10}"
    for want in "${missing[@]}"; do
        if [[ "$mkey" == "$want" ]]; then keep+=("$f"); break; fi
    done
done
INPUT_FILES="${keep[*]}"

# --- Batch the missing months under the submit cap via the controller --------
if [[ -z "$MAX_SUBMIT" || "$MAX_SUBMIT" -lt 2 ]]; then
    echo "ERROR: MAX_SUBMIT must be >= 2 (got '$MAX_SUBMIT')." >&2
    exit 1
fi
BATCH_MAX=$(( MAX_SUBMIT - 1 ))
NUM_BATCHES=$(( (NUM_MONTHS + BATCH_MAX - 1) / BATCH_MAX ))
echo "Mapping $NUM_MONTHS missing month(s) in $NUM_BATCHES batch(es) of" \
     "<= $BATCH_MAX (one submit slot left for the controller/merge)."

CONTROLLER="$SCRIPT_DIR/_freq_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller not found or not executable: $CONTROLLER" >&2
    exit 1
fi

# The controller maps by --shard-index into the month set it is given. It was
# written for --split month over the WHOLE dump dir; here we instead pass an
# explicit --input list of only the missing months, so point the controller at
# that list via DUMP_INPUT. (The controller reads DUMP_DIR OR DUMP_INPUT.)
export DUMP_DIR DUMP_INPUT="$INPUT_FILES"
export SUBREDDIT_FILE OUT_DIR SHARD_DIR STEM DTYPE
export POSTFILTER_ARGS SAMPLE_REMOVED BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER

BATCH_INDEX=0 "$CONTROLLER"

echo "Launched: controller walks $NUM_BATCHES batch(es) of missing months," \
     "then merges all shards."

shopt -s nullglob
existing_shards=("$SHARD_DIR"/*.part.json)
shopt -u nullglob
echo "Shards in ${SHARD_DIR} right now: ${#existing_shards[@]} (map adds" \
     "$NUM_MONTHS; merge re-sums all). Compare to n_shards_merged in the" \
     "merge log and month_status in freq_meta.json when done."
