
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"

mkdir -p logs_drug_centroid

DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
BATCH_SIZE=45
START_TASK=45
ACCOUNT="gts-smussmann3"
QOS="embers"

# Keep this ordering identical to the Python task-to-dump mapping.
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob

NUM_FILES=${#dumps[@]}

if (( NUM_FILES == 0 )); then
    echo "ERROR: No Reddit dump files found in $DUMP_DIR"
    exit 1
fi

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
        --output="logs_drug_centroid/merge_%j.out" \
        --wrap="
            set -euo pipefail
            cd '$SCRIPT_DIR'
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            python -u drug_centroid_merge_results.py
        ")

    merge_job="${merge_job%%;*}"
    echo "Submitted merge job: $merge_job"
}

submit_batch() {
    local start="$1"
    local expected_count="$2"

    # Recount files in each controller to detect changes during the run.
    shopt -s nullglob
    local current_dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
    shopt -u nullglob

    if (( ${#current_dumps[@]} != expected_count )); then
        echo "ERROR: Dump-file count changed. Expected $expected_count, found ${#current_dumps[@]}."
        exit 1
    fi

    local end=$((start + BATCH_SIZE - 1))
    if (( end >= expected_count )); then
        end=$((expected_count - 1))
    fi

    echo "Submitting array tasks $start-$end out of $expected_count dumps."

    local array_job
    array_job=$(sbatch --parsable \
        --job-name=drug_centroid \
        --account="$ACCOUNT" \
        --qos="$QOS" \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --mem=32G \
        --gres=gpu:1 \
        --time=3:00:00 \
        --array="${start}-${end}" \
        --mail-type=FAIL \
        --output="logs_drug_centroid/array_%A_%a.out" \
        --wrap="
            set -euo pipefail
            cd '$SCRIPT_DIR'
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            python -u drug_centroid_embeddings.py
        ")

    array_job="${array_job%%;*}"
    echo "Submitted embedding array: $array_job ($start-$end)"

    local next_start=$((end + 1))

    if (( next_start < expected_count )); then
        # SLURM launches the controller only after the entire array succeeds.
        # The controller submits the next batch without requiring a terminal.
        local controller_job
        controller_job=$(sbatch --parsable \
            --dependency="afterok:${array_job}" \
            --account="$ACCOUNT" \
            --qos="$QOS" \
            --job-name=drug_centroid_ctl \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task=1 \
            --mem=1G \
            --time=00:05:00 \
            --output="logs_drug_centroid/controller_%j.out" \
            --wrap="
                set -euo pipefail
                bash '$SCRIPT_PATH' --continue '$next_start' '$expected_count'
            ")

        controller_job="${controller_job%%;*}"
        echo "Submitted controller: $controller_job"
        echo "It will schedule the next batch after array $array_job succeeds."
    else
        echo "Final embedding batch submitted."
        submit_merge "$array_job"
    fi
}

if [[ "${1:-}" == "--continue" ]]; then
    if (( $# != 3 )); then
        echo "ERROR: Invalid controller arguments."
        exit 1
    fi

    START="$2"
    EXPECTED_COUNT="$3"

    if [[ ! "$START" =~ ^[0-9]+$ || ! "$EXPECTED_COUNT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid task range."
        exit 1
    fi

    echo "Controller running: scheduling tasks from $START."
    submit_batch "$START" "$EXPECTED_COUNT"
else
    echo "Found $NUM_FILES Reddit dump files."
    echo "Batch size: $BATCH_SIZE"
    echo "Initial submission; remaining batches will be scheduled automatically."

    submit_batch "$START_TASK" "$NUM_FILES"
fi