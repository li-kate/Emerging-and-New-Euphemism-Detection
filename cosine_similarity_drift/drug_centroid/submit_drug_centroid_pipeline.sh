
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

mkdir -p logs_hy_centroids

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"

# Maximum tasks submitted in each array.
BATCH_SIZE=45

START_TASK=162
ACCOUNT="gts-smussmann3"
QOS="embers"

SKIP_MONTHS=(
    2015-01 2015-02 2015-03 2015-04 2015-05 2015-06
    2024-07 2024-08 2024-09 2024-10 2024-11 2024-12
)

# --------------------------------------------------
# Build original task-to-dump mapping and eligible IDs
# --------------------------------------------------

shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob

NUM_FILES=${#dumps[@]}

if (( NUM_FILES == 0 )); then
    echo "ERROR: No Reddit dump files found in $DUMP_DIR"
    exit 1
fi

TASK_IDS=()

for i in "${!dumps[@]}"; do
    if (( i < START_TASK )); then
        continue
    fi

    filename=$(basename "${dumps[$i]}")

    if [[ "$filename" =~ ([0-9]{4}-[0-9]{2}) ]]; then
        month="${BASH_REMATCH[1]}"
    else
        echo "ERROR: Could not extract month from $filename"
        exit 1
    fi

    skip=false

    for skipped_month in "${SKIP_MONTHS[@]}"; do
        if [[ "$month" == "$skipped_month" ]]; then
            skip=true
            break
        fi
    done

    if [[ "$skip" == false ]]; then
        TASK_IDS+=("$i")
    fi
done

NUM_ELIGIBLE=${#TASK_IDS[@]}

if (( NUM_ELIGIBLE == 0 )); then
    echo "ERROR: No eligible tasks found."
    exit 1
fi

NUM_BATCHES=$(( (NUM_ELIGIBLE + BATCH_SIZE - 1) / BATCH_SIZE ))

# --------------------------------------------------
# Submit final merge job
# --------------------------------------------------

submit_merge() {
    local last_array_job="$1"

    local merge_job
    merge_job=$(sbatch --parsable \
        --dependency="afterok:${last_array_job}" \
        --account="$ACCOUNT" \
        --qos="$QOS" \
        --job-name=drug_merge \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=8G \
        --time=00:30:00 \
        --mail-type=BEGIN,END,FAIL \
        --output="logs_hy_centroids/merge_%j.out" \
        --wrap="
            set -euo pipefail
            cd '$SCRIPT_DIR'
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            python -u drug_centroid_merge_results.py
        ")

    merge_job="${merge_job%%;*}"

    echo "Submitted final merge job: $merge_job"
}

# --------------------------------------------------
# Submit one batch, then schedule the next batch
# --------------------------------------------------

submit_batch() {
    local batch_index="$1"

    if (( batch_index < 0 || batch_index >= NUM_BATCHES )); then
        echo "ERROR: Invalid batch index: $batch_index"
        exit 1
    fi

    local start=$(( batch_index * BATCH_SIZE ))
    local remaining=$(( NUM_ELIGIBLE - start ))
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
    echo "Tasks in this batch: $count"
    echo "------------------------------------------"

    local array_job
    array_job=$(sbatch --parsable \
        --job-name=drug_centroid \
        --account="$ACCOUNT" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --gres=gpu:1 \
        --time=8:00:00 \
        --array="$array_spec" \
        --mail-type=FAIL \
        --output="logs_hy_centroids/array_%A_%a.out" \
        --wrap="
            set -euo pipefail
            cd '$SCRIPT_DIR'
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            python -u drug_centroid_embeddings.py
        ")

    array_job="${array_job%%;*}"

    echo "Submitted embedding array: $array_job"

    local next_batch=$((batch_index + 1))

    if (( next_batch < NUM_BATCHES )); then

        # Submit a lightweight controller that submits the next batch
        # only after this entire array completes successfully.

        local script_quoted
        printf -v script_quoted '%q' "$SCRIPT_PATH"

        local controller_job
        controller_job=$(sbatch --parsable \
            --dependency="afterok:${array_job}" \
            --account="$ACCOUNT" \
            --qos="$QOS" \
            --job-name=drug_batch_controller \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=1G \
            --time=00:05:00 \
            --output="logs_hy_centroids/controller_%j.out" \
            --wrap="bash $script_quoted --submit-batch $next_batch")

        controller_job="${controller_job%%;*}"

        echo "Submitted controller: $controller_job"
        echo "It will submit batch $((next_batch + 1)) after batch $((batch_index + 1)) succeeds."

    else
        echo "All embedding batches submitted."
        submit_merge "$array_job"
    fi
}

# --------------------------------------------------
# Main entry point
# --------------------------------------------------

echo "Found $NUM_FILES total dump files."
echo "Starting from task $START_TASK."
echo "Skipping half-years: 2015H1 and 2024H2."
echo "Eligible tasks: $NUM_ELIGIBLE"
echo "Batch size: $BATCH_SIZE"
echo "Total batches: $NUM_BATCHES"

if [[ "${1:-}" == "--submit-batch" ]]; then
    if [[ -z "${2:-}" || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Expected a numeric batch index."
        exit 1
    fi

    submit_batch "$2"
else
    echo ""
    echo "Starting automatic batch submission."
    submit_batch 0
fi