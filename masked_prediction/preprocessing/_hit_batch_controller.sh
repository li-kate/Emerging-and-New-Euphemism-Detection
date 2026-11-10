#!/bin/bash
set -euo pipefail

# ============================================================================
# Self-chaining batch controller for submit_hit_matrix_pipeline.sh.
#
# Not meant to be run by hand — submit_hit_matrix_pipeline.sh seeds it with
# BATCH_INDEX=0 and all the config exported into the environment. It mirrors
# the frequency build's _freq_batch_controller.sh, with two differences:
#   * each map task requests a GPU (a BERT forward pass per occurrence);
#   * the array indexes into the KEPT months (those whose half-year is in the
#     frequency matrix), passed through as MONTHS_ARG and handed to
#     hit_matrix.py --months, so the array index <-> month mapping stays
#     correct for a filtered month list.
#
# Each invocation:
#   1. submits the map array for its batch (a contiguous range of kept-month
#      indices, at most BATCH_MAX tasks), then
#   2. if more batches remain, submits the NEXT controller invocation as a
#      tiny one-task job that depends (afterany) on this batch's array — so the
#      next batch is only *submitted* once this batch has left the queue;
#   3. otherwise submits the merge job, depending (afterany) on this (final)
#      batch.
#
# Because the next batch is not submitted until the current one finishes, the
# queue never holds more than one map batch (<= BATCH_MAX) plus the single
# pending controller/merge job. That keeps the total under the QOS submit cap
# (BATCH_MAX = MAX_SUBMIT - 1, leaving one slot for the chained controller/merge
# job).
#
# afterany (not afterok) chains batches and the merge so one failed month never
# stalls the rest; merge_hit_matrix.py fails only if there are no shards at all.
# ============================================================================

# Required config, exported by the pipeline script (or a prior controller).
: "${BATCH_INDEX:?BATCH_INDEX must be set}"
: "${DUMP_DIR:?}"
: "${SUBREDDIT_FILE:?}"
: "${FREQ_DIR:?}"
: "${FREQ_STEM:?}"
: "${DEA_FILE:?}"
: "${OUT_DIR:?}"
: "${SHARD_DIR:?}"
: "${STEM:?}"
: "${MODEL_NAME:?}"
: "${START_YEAR:?}"
: "${BATCH_SIZE:?}"
: "${PY_DEPS:?}"
: "${BATCH_MAX:?}"
: "${NUM_MONTHS:?}"
: "${NUM_BATCHES:?}"
: "${CONTROLLER:?}"
: "${MONTHS_ARG:?}"

# Optional: pin the cumulative-causal frequency floor. When unset, hit_matrix.py
# uses its own default (so the full-build pipeline is unchanged). The missing-
# months script sets this to match the floor baked into the existing shards,
# since merge_hit_matrix.py requires every shard to share one min_freq.
MIN_FREQ_ARG=""
if [[ -n "${MIN_FREQ:-}" ]]; then
    MIN_FREQ_ARG="--min-freq $MIN_FREQ"
fi

# This controller runs either on the submit host (batch 0, from the pipeline
# script) or on a compute node (later batches, as a chained job). Run sbatch
# from the directory holding the Python scripts either way.
cd "$(dirname "$CONTROLLER")"
mkdir -p logs_hit_matrix

# --- Compute this batch's contiguous kept-month-index range ------------------
START=$(( BATCH_INDEX * BATCH_MAX ))
END=$(( START + BATCH_MAX - 1 ))
if (( END > NUM_MONTHS - 1 )); then
    END=$(( NUM_MONTHS - 1 ))
fi
BATCH_SIZE_TASKS=$(( END - START + 1 ))

# Run the whole batch at once (% throttle == batch size): a freed GPU slot
# starts the next task in the batch immediately.
ARRAY_SPEC="${START}-${END}%${BATCH_SIZE_TASKS}"

MAP_JOB=$(sbatch --parsable \
    --job-name="hit_map_b${BATCH_INDEX}" \
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
            --shard-dir '$SHARD_DIR' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\" \
            --months $MONTHS_ARG \
            --model-name '$MODEL_NAME' \
            --start-year $START_YEAR \
            --batch-size $BATCH_SIZE \
            $MIN_FREQ_ARG \
            --fp16
    ")

echo "Batch $((BATCH_INDEX + 1))/$NUM_BATCHES: map array $MAP_JOB (kept-month indices $ARRAY_SPEC)"

NEXT_INDEX=$(( BATCH_INDEX + 1 ))

if (( NEXT_INDEX < NUM_BATCHES )); then
    # More batches remain: submit the next controller invocation as a tiny
    # one-task job that waits for this batch to leave the queue, then submits
    # the next batch. It re-exports the same config plus the new BATCH_INDEX.
    NEXT_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
        --job-name="hit_ctl_b${NEXT_INDEX}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=2G \
        --time=00:10:00 \
        --output="logs_hit_matrix/controller_%j.out" \
        --mail-type=FAIL \
        --export="ALL,BATCH_INDEX=${NEXT_INDEX}" \
        --wrap="set -euo pipefail; '$CONTROLLER'")
    echo "  chained next controller: $NEXT_JOB (BATCH_INDEX=$NEXT_INDEX, after $MAP_JOB)"
else
    # Final batch: submit the merge after it leaves the queue.
    MERGE_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
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
                --stem '$STEM'
        ")
    echo "  final batch — submitted merge: $MERGE_JOB (after $MAP_JOB)"
fi
