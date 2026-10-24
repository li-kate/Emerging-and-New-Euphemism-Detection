#!/bin/bash
set -euo pipefail

# ============================================================================
# Self-chaining batch controller for submit_freq_matrix_pipeline.sh.
#
# Not meant to be run by hand — submit_freq_matrix_pipeline.sh seeds it with
# BATCH_INDEX=0 and all the config exported into the environment. Each
# invocation:
#
#   1. submits the map array for its batch (a contiguous range of month
#      indices, at most BATCH_MAX tasks), then
#   2. if more batches remain, submits the NEXT controller invocation as a
#      tiny one-task job that depends (afterany) on this batch's array — so
#      the next batch is only *submitted* once this batch has left the queue;
#   3. otherwise submits the merge job, depending (afterany) on this (final)
#      batch.
#
# Because the next batch is not submitted until the current one finishes, the
# queue never holds more than one map batch (<= BATCH_MAX) plus the single
# pending controller/merge job. That keeps the total under the QOS submit cap
# (BATCH_MAX = MAX_SUBMIT - 1, leaving one slot for this chained job).
#
# afterany (not afterok) is used throughout so that one failed month never
# stalls the remaining batches or the merge; merge_freq_matrix.py itself fails
# only if there are no shards at all.
# ============================================================================

# Required config, exported by the pipeline script (or a prior controller).
: "${BATCH_INDEX:?BATCH_INDEX must be set}"
: "${DUMP_DIR:?}"
: "${SUBREDDIT_FILE:?}"
: "${OUT_DIR:?}"
: "${SHARD_DIR:?}"
: "${STEM:?}"
: "${DTYPE:?}"
: "${SAMPLE_REMOVED:?}"
: "${BATCH_MAX:?}"
: "${NUM_MONTHS:?}"
: "${NUM_BATCHES:?}"
: "${CONTROLLER:?}"
# POSTFILTER_ARGS may legitimately be empty; default it so set -u is happy.
POSTFILTER_ARGS="${POSTFILTER_ARGS:-}"

# This controller runs either on the submit host (batch 0, from the pipeline
# script) or on a compute node (later batches, as a chained job). Run sbatch
# from the directory holding the Python scripts either way.
cd "$(dirname "$CONTROLLER")"
mkdir -p logs_freq_matrix

# --- Compute this batch's contiguous month-index range -----------------------
START=$(( BATCH_INDEX * BATCH_MAX ))
END=$(( START + BATCH_MAX - 1 ))
if (( END > NUM_MONTHS - 1 )); then
    END=$(( NUM_MONTHS - 1 ))
fi
BATCH_SIZE=$(( END - START + 1 ))

# Run the whole batch at once (% throttle == batch size): a freed slot starts
# the next task in the batch immediately.
ARRAY_SPEC="${START}-${END}%${BATCH_SIZE}"

MAP_JOB=$(sbatch --parsable \
    --job-name="freq_map_b${BATCH_INDEX}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=6:00:00 \
    --array="$ARRAY_SPEC" \
    --output="logs_freq_matrix/map_%A_%a.out" \
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
            --reddit-monthly-dir '$DUMP_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --shard-dir '$SHARD_DIR' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\" \
            --sample-removed $SAMPLE_REMOVED
    ")

echo "Batch $((BATCH_INDEX + 1))/$NUM_BATCHES: map array $MAP_JOB (months $ARRAY_SPEC)"

NEXT_INDEX=$(( BATCH_INDEX + 1 ))

if (( NEXT_INDEX < NUM_BATCHES )); then
    # More batches remain: submit the next controller invocation as a tiny
    # one-task job that waits for this batch to leave the queue, then submits
    # the next batch. It re-exports the same config plus the new BATCH_INDEX.
    NEXT_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
        --job-name="freq_ctl_b${NEXT_INDEX}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=2G \
        --time=00:10:00 \
        --output="logs_freq_matrix/controller_%j.out" \
        --mail-type=FAIL \
        --export="ALL,BATCH_INDEX=${NEXT_INDEX}" \
        --wrap="set -euo pipefail; '$CONTROLLER'")
    echo "  chained next controller: $NEXT_JOB (BATCH_INDEX=$NEXT_INDEX, after $MAP_JOB)"
else
    # Final batch: submit the merge after it leaves the queue.
    MERGE_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
        --job-name=freq_merge \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=32G \
        --time=1:00:00 \
        --output="logs_freq_matrix/merge_%j.out" \
        --mail-type=END,FAIL \
        --wrap="
            set -euo pipefail
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            pip install numpy pandas nltk zstandard orjson
            python -u merge_freq_matrix.py \
                --shard-dir '$SHARD_DIR' \
                --out-dir '$OUT_DIR' \
                --subreddit-file '$SUBREDDIT_FILE' \
                --stem '$STEM' \
                --dtype '$DTYPE' \
                $POSTFILTER_ARGS \
                --sample-removed $SAMPLE_REMOVED
        ")
    echo "  final batch — submitted merge: $MERGE_JOB (after $MAP_JOB)"
fi
