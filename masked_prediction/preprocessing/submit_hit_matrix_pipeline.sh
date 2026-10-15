#!/bin/bash
set -euo pipefail

# ============================================================================
# Hit-matrix pipeline: pilot (SLURM job) -> map (SLURM array) -> merge (reduce).
#
#   0. Pilot: pilot_creation.py selects the control words (pilot_words.json)
#      from the frequency matrix + DEA list. Runs as its OWN SLURM job (not on
#      the submit node) so numpy/pandas are installed in the job environment —
#      the submit node may not have them. Every map task reads its output, so
#      the map array waits for it via afterok.
#   1. Map array: ONE TASK PER MONTH. Each task scores a single month with BERT
#      (hit_matrix.py --mode map) and writes one shard. GPU per task. Starts
#      after the pilot job succeeds (afterok).
#   2. Merge: after all map tasks succeed (afterok), merge_hit_matrix.py sums
#      the shards into the summary table + matrices aligned to the frequency
#      matrix.
#
# Nothing in this pipeline runs Python on the submit node; the only submit-node
# work is globbing dump filenames to size the array (no Python needed).
#
# Mirrors submit_freq_matrix_pipeline.sh (same month grouping and array sizing);
# the differences are the pilot pre-job and the GPU request on the map array.
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

# --- Step 0: pilot selection (its own SLURM job) -----------------------------
# Runs on a compute node so numpy/pandas get installed in the job env. The map
# array depends on this via afterok, so pilot_words.json always exists before
# any month task reads it. CPU-only; no GPU requested.
PILOT_JOB=$(sbatch --parsable \
    --job-name=hit_pilot \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=16G \
    --time=0:30:00 \
    --output="logs_hit_matrix/pilot_%j.out" \
    --mail-type=FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install numpy pandas
        python -u pilot_creation.py \
            --freq-dir '$FREQ_DIR' \
            --stem '$FREQ_STEM' \
            --dea-file '$DEA_FILE' \
            --out '$PILOT_FILE' \
            --seed $PILOT_SEED \
            --min-freq $PILOT_MIN_FREQ \
            --min-periods $PILOT_MIN_PERIODS \
            --n-pilot $PILOT_N
    ")
echo "Submitted pilot job: $PILOT_JOB -> $PILOT_FILE"

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
echo "Keeping $NUM_MONTHS month(s) in the frequency matrix; skipping $SKIPPED outside it."
if [[ $NUM_MONTHS -eq 0 ]]; then
    echo "ERROR: no dump months fall within the frequency matrix's half-years."
    echo "       Check that DUMP_DIR and the frequency build cover the same period."
    exit 1
fi
echo "First/last kept month: ${months[0]} ... ${months[$((NUM_MONTHS - 1))]}"

# --- Guard against the QOS submit cap ----------------------------------------
# %MAX_CONCURRENT only limits how many RUN at once; the whole array still
# counts against the QOS "max submitted jobs per user" cap. If the array is
# bigger than MAX_SUBMIT, submission will be rejected (QOSMaxSubmitJobPerUserLimit).
# Set MAX_SUBMIT to your QOS cap (ICE default 500) minus whatever you already
# have queued. If the kept array still exceeds it, submit in slices with
# --months, e.g.:
#   ./submit_hit_matrix_pipeline.sh   # edit MONTHS_SLICE below to a subset
MAX_SUBMIT="${MAX_SUBMIT:-500}"
if [[ $NUM_MONTHS -gt $MAX_SUBMIT ]]; then
    echo ""
    echo "ERROR: $NUM_MONTHS tasks exceeds MAX_SUBMIT=$MAX_SUBMIT (QOS submit cap)."
    echo "       Even restricted to the matrix, the array is too large to submit at once."
    echo "       Submit in slices by setting MONTHS_SLICE to a subset of these months"
    echo "       and rerunning, or raise MAX_SUBMIT if your QOS allows it."
    echo "       Kept months: ${months[*]}"
    exit 1
fi

# --- Clean stale shards ------------------------------------------------------
mkdir -p "$SHARD_DIR"
rm -f "$SHARD_DIR"/*.part.json "$SHARD_DIR"/*.part.json.tmp 2>/dev/null || true

# The map array must index into EXACTLY the kept months (not re-glob all dump
# months), so pass the kept list to hit_matrix.py --months. select_month_key()
# then restricts to this list and the array index selects the Nth kept month,
# keeping the array index <-> month mapping correct.
MONTHS_ARG="${months[*]}"

# --- Map array: one GPU task per month ---------------------------------------
ARRAY_SPEC="0-$((NUM_MONTHS - 1))"
if [[ -n "$MAX_CONCURRENT" && "$MAX_CONCURRENT" -gt 0 ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
    echo "Submitting $NUM_MONTHS month tasks, running <= $MAX_CONCURRENT at once."
else
    echo "Submitting $NUM_MONTHS month tasks with no concurrency throttle."
fi

MAP_JOB=$(sbatch --parsable \
    --dependency="afterok:${PILOT_JOB}" \
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
            --months $MONTHS_ARG \
            --model-name '$MODEL_NAME' \
            --batch-size $BATCH_SIZE \
            --fp16
    ")
echo "Submitted map array: $MAP_JOB ($ARRAY_SPEC) — one GPU task per month"
echo "The map array will start after the pilot job succeeds (afterok)."

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
echo ""
echo "Chain: pilot ($PILOT_JOB) -> map ($MAP_JOB) -> merge ($MERGE_JOB)"
