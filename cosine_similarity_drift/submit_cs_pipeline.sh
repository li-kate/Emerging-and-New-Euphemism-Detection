
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

mkdir -p logs_cosine_similarity

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BATCH_SIZE=45

ACCOUNT="gts-smussmann3"
QOS="embers"

# Months already completed or excluded.
declare -A SKIP_MONTHS

# Already completed.
for month in \
    2015-07 2015-08 2015-09 \
    2015-10 2015-11 2015-12
do
    SKIP_MONTHS["$month"]=1
done

# Corrupted month.
SKIP_MONTHS["2015-03"]=1

# Skip entire 2024H2.
for month in \
    2024-07 2024-08 2024-09 \
    2024-10 2024-11 2024-12
do
    SKIP_MONTHS["$month"]=1
done

# Skip entire 2026H2.
for month in \
    2026-07 2026-08 2026-09 \
    2026-10 2026-11 2026-12
do
    SKIP_MONTHS["$month"]=1
done

# --------------------------------------------------
# Build task mapping from similarity_matrix.py
# --------------------------------------------------

export PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}"

mapfile -t ALL_MONTHS < <(
    python - <<'PY'
import similarity_matrix as sm

for month in sm.get_month_tasks():
    print(month)
PY
)

if (( ${#ALL_MONTHS[@]} == 0 )); then
    echo "ERROR: No monthly tasks found."
    exit 1
fi

TASK_IDS=()

for task_id in "${!ALL_MONTHS[@]}"; do
    month="${ALL_MONTHS[$task_id]}"

    if [[ -n "${SKIP_MONTHS[$month]:-}" ]]; then
        echo "Skipping task $task_id: $month"
        continue
    fi

    TASK_IDS+=("$task_id")
done

NUM_ELIGIBLE=${#TASK_IDS[@]}

if (( NUM_ELIGIBLE == 0 )); then
    echo "ERROR: No eligible tasks found."
    exit 1
fi

NUM_BATCHES=$(( (NUM_ELIGIBLE + BATCH_SIZE - 1) / BATCH_SIZE ))

# --------------------------------------------------
# Submit final matrix assembly
# --------------------------------------------------

submit_matrix_assembly() {
    local monthly_merge_job="$1"

    local matrix_job
    matrix_job=$(sbatch --parsable \
        --dependency="afterok:${monthly_merge_job}" \
        --account="$ACCOUNT" \
        --qos="$QOS" \
        --job-name=cosine_matrix \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=16G \
        --time=01:00:00 \
        --output="logs_cosine_similarity/matrix_%j.out" \
        --wrap="
            set -euo pipefail
            cd '$SCRIPT_DIR'
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            export PYTHONPATH='$SCRIPT_DIR/..':\${PYTHONPATH:-}
            python -u merge_similarity_matrix.py
        ")

    matrix_job="${matrix_job%%;*}"

    echo "Submitted final matrix assembly job: $matrix_job"
}

# --------------------------------------------------
# Submit half-year merger
# --------------------------------------------------

submit_half_year_merge() {
    local last_array_job="$1"

    local merge_job
    merge_job=$(sbatch --parsable \
        --dependency="afterok:${last_array_job}" \
        merge_sm_monthly.sbatch)

    merge_job="${merge_job%%;*}"

    echo "Submitted monthly-to-half-year merger: $merge_job"

    submit_matrix_assembly "$merge_job"
}

# --------------------------------------------------
# Submit one batch
# --------------------------------------------------

submit_batch() {
    local batch_index="$1"

    if (( batch_index < 0 || batch_index >= NUM_BATCHES )); then
        echo "ERROR: Invalid batch index: $batch_index"
        exit 1
    fi

    local start=$((batch_index * BATCH_SIZE))
    local remaining=$((NUM_ELIGIBLE - start))
    local count=$BATCH_SIZE

    if (( remaining < count )); then
        count=$remaining
    fi

    local batch_tasks=("${TASK_IDS[@]:start:count}")
    local array_spec
    array_spec=$(IFS=,; echo "${batch_tasks[*]}")

    echo ""
    echo "------------------------------------------"
    echo "Submitting batch $((batch_index + 1))/$NUM_BATCHES"
    echo "Task IDs: $array_spec"
    echo "Tasks in batch: $count"
    echo "------------------------------------------"

    local array_job
    array_job=$(sbatch --parsable \
        --array="$array_spec" \
        submit_sm.sbatch)

    array_job="${array_job%%;*}"

    echo "Submitted monthly cosine-similarity array: $array_job"

    local next_batch=$((batch_index + 1))

    if (( next_batch < NUM_BATCHES )); then

        local script_quoted
        printf -v script_quoted '%q' "$SCRIPT_PATH"

        local controller_job
        controller_job=$(sbatch --parsable \
            --dependency="afterok:${array_job}" \
            --account="$ACCOUNT" \
            --qos="$QOS" \
            --job-name=cosine_batch_controller \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=1G \
            --time=00:05:00 \
            --output="logs_cosine_similarity/controller_%j.out" \
            --wrap="bash $script_quoted --submit-batch $next_batch")

        controller_job="${controller_job%%;*}"

        echo "Submitted controller: $controller_job"
        echo "Next batch will submit after this batch succeeds."

    else
        echo "All monthly cosine-similarity batches submitted."
        submit_half_year_merge "$array_job"
    fi
}

# --------------------------------------------------
# Main
# --------------------------------------------------

echo ""
echo "=========================================="
echo "COSINE SIMILARITY SUBMISSION"
echo "=========================================="

echo "Total original monthly tasks: ${#ALL_MONTHS[@]}"
echo "Eligible monthly tasks: $NUM_ELIGIBLE"
echo "Skipped monthly tasks: $((${#ALL_MONTHS[@]} - NUM_ELIGIBLE))"
echo "Batch size: $BATCH_SIZE"
echo "Total batches: $NUM_BATCHES"

echo ""
echo "Excluded half-years:"
echo "  2015H2: already completed"
echo "  2024H2: skipped"
echo "  2026H2: skipped"

if [[ "${1:-}" == "--submit-batch" ]]; then
    if [[ -z "${2:-}" || ! "${2:-}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Expected a numeric batch index."
        exit 1
    fi

    submit_batch "$2"
else
    submit_batch 0
fi