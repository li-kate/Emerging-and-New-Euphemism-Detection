#!/bin/bash
set -euo pipefail

# ============================================================================
# Map + merge the frequency matrix for just one YEAR (and optionally only the
# RC or RS dumps of that year), then re-merge ALL shards into the final matrix.
#
# Use this to (re)build a single year's months — e.g. to add months that
# failed in the main run — without touching the other months' shards. The map
# tasks write/overwrite only the targeted months' *.part.json; the merge then
# sums EVERY shard in SHARD_DIR (old + new) into the final matrix.
#
#   WHY this does not clean the shard dir:
#   submit_freq_matrix_pipeline.sh deletes all shards first (fresh full build).
#   This script intentionally does NOT, so the year you rebuild is merged back
#   together with the months already counted. Running the same month again just
#   overwrites that one shard (atomic rename in write_shard).
#
# Usage:
#   ./submit_freq_year.sh YEAR [PREFIX] [--missing-only]
#     YEAR            four-digit year, e.g. 2015
#     PREFIX          RC | RS | both   (default: both)
#     --missing-only  map only months that have a dump but NO shard yet
#                     (i.e. the months that are missing from SHARD_DIR). Use
#                     this to fill gaps left by failed tasks without recounting
#                     months that already have a good shard.
#
# Examples:
#   ./submit_freq_year.sh 2015 RC                 # all comment dumps of 2015
#   ./submit_freq_year.sh 2018 RS                 # all submission dumps of 2018
#   ./submit_freq_year.sh 2015                    # both RC and RS of 2015
#   ./submit_freq_year.sh 2015 both --missing-only # only 2015 months w/o a shard
#
# Postfilter at the merge step is controlled by the same env vars as the main
# pipeline (NO_POSTFILTER / POSTFILTER_MODE / MIN_FREQ). Keep them identical to
# your original run so the re-merged matrix uses the same filtering.
# ============================================================================

# --- Args --------------------------------------------------------------------
YEAR=""
PREFIX="both"
MISSING_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --missing-only) MISSING_ONLY=1 ;;
        RC|RS|both)     PREFIX="$arg" ;;
        [0-9][0-9][0-9][0-9]) YEAR="$arg" ;;
        *) echo "ERROR: unrecognized argument: '$arg'" >&2
           echo "Usage: $0 YEAR [RC|RS|both] [--missing-only]" >&2
           exit 1 ;;
    esac
done

if [[ ! "$YEAR" =~ ^[0-9]{4}$ ]]; then
    echo "ERROR: YEAR must be four digits, e.g. 2015. Got: '${YEAR:-<none>}'" >&2
    echo "Usage: $0 YEAR [RC|RS|both] [--missing-only]" >&2
    exit 1
fi

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

# --- Collect this year's dump files ------------------------------------------
# Build the glob(s) for the requested prefix(es). A month may be multi-part
# (RC_2018-07_part000.zst); the glob RC_YYYY-*.zst catches every part, and
# --split month groups the parts back into one shard per month.
shopt -s nullglob
case "$PREFIX" in
    RC)   dumps=("$DUMP_DIR"/RC_"$YEAR"-*.zst) ;;
    RS)   dumps=("$DUMP_DIR"/RS_"$YEAR"-*.zst) ;;
    both) dumps=("$DUMP_DIR"/RC_"$YEAR"-*.zst "$DUMP_DIR"/RS_"$YEAR"-*.zst) ;;
esac
shopt -u nullglob

if [[ ${#dumps[@]} -eq 0 ]]; then
    echo "ERROR: No ${PREFIX} dumps for $YEAR found in $DUMP_DIR" >&2
    echo "       (looked for ${PREFIX}_${YEAR}-*.zst)" >&2
    exit 1
fi

# Count distinct month keys (RX_YYYY-MM) so the array is sized correctly: one
# task per month, mirroring group_by_month() in freq_matrix.py.
months=()
while IFS= read -r mkey; do
    months+=("$mkey")
done < <(
    for f in "${dumps[@]}"; do
        base=$(basename "$f")
        echo "${base:0:10}"   # RC_2015-03_part000.zst -> RC_2015-03
    done | sort -u
)
NUM_MONTHS=${#months[@]}

echo "Found ${#dumps[@]} dump file(s) across $NUM_MONTHS month(s) for" \
     "${PREFIX} ${YEAR}:"
printf '  %s\n' "${months[@]}"

mkdir -p "$SHARD_DIR"

# --- Show which of these months already have a shard -------------------------
# A shard is <SHARD_DIR>/<month_key>.part.json (write_shard names it by the
# month key). This lets the run report what exists before and after.
have_shard() { [[ -f "${SHARD_DIR}/${1}.part.json" ]]; }

present=(); absent=()
for m in "${months[@]}"; do
    if have_shard "$m"; then present+=("$m"); else absent+=("$m"); fi
done
echo "Existing shards for ${PREFIX} ${YEAR}: ${#present[@]}/${NUM_MONTHS}"
[[ ${#present[@]} -gt 0 ]] && printf '  present: %s\n' "${present[@]}"
[[ ${#absent[@]}  -gt 0 ]] && printf '  MISSING: %s\n' "${absent[@]}"

# --- --missing-only: restrict to months that have no shard yet ---------------
if [[ "$MISSING_ONLY" == "1" ]]; then
    if [[ ${#absent[@]} -eq 0 ]]; then
        echo "--missing-only: every ${PREFIX} ${YEAR} month already has a" \
             "shard. Nothing to map."
        echo "(Re-run without --missing-only to force a recount, or just" \
             "re-merge.)"
        exit 0
    fi
    months=("${absent[@]}")
    NUM_MONTHS=${#months[@]}
    # Keep only the dump files whose month key is in the missing set.
    keep=()
    for f in "${dumps[@]}"; do
        base=$(basename "$f"); mkey="${base:0:10}"
        for want in "${months[@]}"; do
            if [[ "$mkey" == "$want" ]]; then keep+=("$f"); break; fi
        done
    done
    dumps=("${keep[@]}")
    echo "--missing-only: mapping $NUM_MONTHS missing month(s):"
    printf '  %s\n' "${months[@]}"
fi

# The map tasks receive the SAME explicit --input list (so only these files are
# considered) plus --split month, and select their month by array index. We
# pass the files via --input rather than --months so a part-split month is one
# key and the shell does the year/prefix selection.
INPUT_FILES="${dumps[*]}"

# --- Map: one array task per month -------------------------------------------
# Array is 0..NUM_MONTHS-1; freq_matrix.py's --shard-index selects the Nth
# month of the (sorted) --input set. % throttle == NUM_MONTHS runs them all as
# slots free up. No batching/controller here: a single year is small enough to
# fit under the submit cap; if a year ever exceeds it, lower the %throttle.
MAP_JOB=$(sbatch --parsable \
    --job-name="freq_${PREFIX}${YEAR}_map" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=6:00:00 \
    --array="0-$((NUM_MONTHS - 1))%${NUM_MONTHS}" \
    --output="logs_freq_matrix/year_map_%A_%a.out" \
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
            --input ${INPUT_FILES} \
            --subreddit-file '${SUBREDDIT_FILE}' \
            --shard-dir '${SHARD_DIR}' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\" \
            --sample-removed ${SAMPLE_REMOVED}
    ")
echo "Submitted map array: $MAP_JOB (0-$((NUM_MONTHS - 1)))"

# --- Merge: re-sum ALL shards once the map array leaves the queue ------------
# afterany (not afterok): one failed month should not block re-merging the
# rest. A month that failed to decompress is recorded in freq_meta.json's
# month_status by the map/merge code.
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterany:${MAP_JOB}" \
    --job-name="freq_${PREFIX}${YEAR}_merge" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=32G \
    --time=1:00:00 \
    --output="logs_freq_matrix/year_merge_%j.out" \
    --mail-type=END,FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install numpy pandas nltk zstandard orjson
        python -u merge_freq_matrix.py \
            --shard-dir '${SHARD_DIR}' \
            --out-dir '${OUT_DIR}' \
            --subreddit-file '${SUBREDDIT_FILE}' \
            --stem '${STEM}' \
            --dtype '${DTYPE}' \
            ${POSTFILTER_ARGS} \
            --sample-removed ${SAMPLE_REMOVED}
    ")
echo "Submitted merge: $MERGE_JOB (after $MAP_JOB)"

# Report the total shard count now (before the map runs). After the jobs
# finish, the merge log prints n_shards_merged and the month_status summary, so
# comparing this number to that confirms the new months landed.
shopt -s nullglob
existing_shards=("$SHARD_DIR"/*.part.json)
shopt -u nullglob
echo "Shards in ${SHARD_DIR} right now: ${#existing_shards[@]} (the map will" \
     "add/overwrite ${NUM_MONTHS}; merge re-sums all of them)."
echo "Done. Map ${PREFIX} ${YEAR} (${NUM_MONTHS} month(s)), then re-merge all shards."
