#!/bin/bash
set -euo pipefail

# ============================================================================
# Map every MISSING month of the hit-matrix run, then re-merge ALL shards.
#
# A "missing" month is one that has a Reddit dump in DUMP_DIR, falls inside the
# frequency matrix's half-years (the only months the main pipeline scores), but
# has no shard (<SHARD_DIR>/<month_key>.part.json) yet — i.e. a month that
# never ran or whose map task failed before writing its shard. The map step
# failing on a few months is common and usually NOT a data problem: each month
# is a separate GPU array task, and a task can land on a node whose GPU is
# momentarily busy/unavailable ("CUDA-capable device(s) is/are busy or
# unavailable"), so it dies before the model even loads. Because the pipeline
# chains batches with `afterany`, a failed month does not stall the rest — it
# just leaves no shard, and the merge silently omits it. This script fills
# those gaps.
#
# It maps only the missing months and leaves every existing shard untouched,
# then re-sums all shards (old + new) into the final summary table + matrices
# aligned to the frequency matrix. Mirrors zhu_temporal/submit_zhu_missing.sh.
#
#   WHY it does not clean the shard dir:
#   submit_hit_matrix_pipeline.sh wipes all shards first (fresh full build).
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
# PREREQUISITES: the frequency matrix must already be built (freq_dir below),
# since it defines the target words + denominator and its scope hash is checked.
#
# Usage:
#   ./submit_hit_matrix_missing.sh            # map all missing months, then merge
#   ./submit_hit_matrix_missing.sh --dry-run  # just list missing months, submit nothing
#
# Keep the scoring config (FREQ_DIR, DEA_FILE, MODEL_NAME, BATCH_SIZE, scope)
# IDENTICAL to the original run, or the merge's scope-hash check will reject the
# mixed shards.
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

# --- Configuration (match submit_hit_matrix_pipeline.sh) ---------------------
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$SCRIPT_DIR"
mkdir -p logs_hit_matrix

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SUBREDDIT_FILE="drug_subreddit_list.txt"      # must match the frequency build
FREQ_DIR="freq_drug_subreddits_unfiltered"               # existing frequency matrix
FREQ_STEM="freq"
DEA_FILE="../../data_collection/word_lists/drug_synonyms.txt"
OUT_DIR="hits_drug_subreddits"
SHARD_DIR="${OUT_DIR}/shards"
STEM="hits"

MODEL_NAME="bert-base-uncased"
START_YEAR=2019
BATCH_SIZE=64

# Pin the cumulative-causal frequency floor to the one baked into the existing
# shards (target_selection.min_freq). merge_hit_matrix.py aborts if shards
# disagree on min_freq, so the gap-fill months MUST use the same floor.
MIN_FREQ=2

MAX_SUBMIT="${MAX_SUBMIT:-15}"
PY_DEPS="numpy pandas nltk zstandard orjson torch transformers"

# --- Sanity: the frequency matrix must exist ---------------------------------
if [[ ! -f "${FREQ_DIR}/${FREQ_STEM}_words.csv" || ! -f "${FREQ_DIR}/${FREQ_STEM}_meta.json" ]]; then
    echo "ERROR: frequency matrix not found in ${FREQ_DIR}" >&2
    echo "       (need ${FREQ_STEM}_words.csv and ${FREQ_STEM}_meta.json)." >&2
    echo "       Build it first with submit_freq_matrix_pipeline.sh." >&2
    exit 1
fi

# --- Months in scope = dump months whose half-year is IN the frequency matrix
# Same filter as the main pipeline: only these months can be "missing". Months
# 1-6 -> H1, 7-12 -> H2 (matching corpus_stream.half_year_label).
HY_FILE="${FREQ_DIR}/${FREQ_STEM}_half_years.csv"
if [[ ! -f "$HY_FILE" ]]; then
    echo "ERROR: $HY_FILE not found; cannot restrict months to the matrix." >&2
    exit 1
fi

mapfile -t FREQ_HYS < <(tail -n +2 "$HY_FILE" | tr -d '\r' | sed '/^$/d' | sort -u)
if [[ ${#FREQ_HYS[@]} -eq 0 ]]; then
    echo "ERROR: no half-years listed in $HY_FILE" >&2
    exit 1
fi
FREQ_HY_SET="|$(IFS='|'; echo "${FREQ_HYS[*]}")|"
echo "Frequency matrix covers ${#FREQ_HYS[@]} half-year(s): ${FREQ_HYS[*]}"

# --- All dump months, then filter to the matrix's half-years -----------------
# Group RC_/RS_ YYYY-MM dumps (parts share a month).
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

kept_months=()
for mkey in "${all_months[@]}"; do
    yr="${mkey:3:4}"      # YYYY
    mon="${mkey:8:2}"     # MM
    # Gate on START_YEAR as well as the frequency matrix's half-years. The freq
    # matrix can cover earlier half-years (e.g. 2015) than we want to score;
    # hit_matrix.py would drop every pre-START_YEAR record anyway (stream_corpus
    # "year" filter), producing empty shards and wasting GPU tasks. So skip
    # those months at selection time instead of submitting them.
    if (( 10#$yr < START_YEAR )); then
        continue
    fi
    half=$(( 10#$mon <= 6 ? 1 : 2 ))
    hy="${yr}H${half}"
    if [[ "$FREQ_HY_SET" == *"|$hy|"* ]]; then
        kept_months+=("$mkey")
    fi
done
echo "Dump dir has ${#dumps[@]} file(s); ${#kept_months[@]} month(s) in the" \
     "frequency matrix's half-years and >= $START_YEAR (of ${#all_months[@]}" \
     "dump month(s))."

# --- Missing months (in scope, dump present, no shard) -----------------------
mkdir -p "$SHARD_DIR"
missing=()
for m in "${kept_months[@]}"; do
    [[ -f "${SHARD_DIR}/${m}.part.json" ]] || missing+=("$m")
done

NUM_MONTHS=${#missing[@]}
if [[ $NUM_MONTHS -eq 0 ]]; then
    echo "No missing months: every in-scope dump month already has a shard."
    echo "(If a shard is incomplete, delete it and re-run to rescore that month.)"
    exit 0
fi

echo "Missing $NUM_MONTHS month(s) (in scope, dump present, no shard):"
printf '  %s\n' "${missing[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "--dry-run: nothing submitted."
    exit 0
fi

# --- Sanity: word list must exist --------------------------------------------
# NOTE: MODEL_NAME is a Hugging Face hub id (bert-base-uncased), not a local
# dir, so it is not checked here; transformers downloads it at run time.
if [[ ! -f "$DEA_FILE" ]]; then
    echo "ERROR: DEA/synonyms word list not found at $DEA_FILE" >&2
    exit 1
fi
if [[ ! -f "$SUBREDDIT_FILE" ]]; then
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

CONTROLLER="$SCRIPT_DIR/_hit_batch_controller.sh"
if [[ ! -x "$CONTROLLER" ]]; then
    echo "ERROR: controller not found or not executable: $CONTROLLER" >&2
    exit 1
fi

# The controller indexes months by --shard-index into the list it is handed via
# MONTHS_ARG (passed to hit_matrix.py --months). Handing it ONLY the missing
# months means array index N maps to the Nth missing month, so the gaps are
# exactly what gets mapped. Existing shards are left untouched; the merge
# re-sums all of them.
MONTHS_ARG="${missing[*]}"

export DUMP_DIR SUBREDDIT_FILE FREQ_DIR FREQ_STEM DEA_FILE
export OUT_DIR SHARD_DIR STEM MODEL_NAME START_YEAR BATCH_SIZE PY_DEPS
export BATCH_MAX NUM_MONTHS NUM_BATCHES CONTROLLER MONTHS_ARG MIN_FREQ

BATCH_INDEX=0 "$CONTROLLER"

echo ""
echo "Launched: controller walks $NUM_BATCHES batch(es) of missing months," \
     "then merges all shards."

shopt -s nullglob
existing_shards=("$SHARD_DIR"/*.part.json)
shopt -u nullglob
echo "Shards in ${SHARD_DIR} right now: ${#existing_shards[@]} (map adds up to" \
     "$NUM_MONTHS; merge re-sums all). Compare to the shard count in the merge" \
     "log when done."
