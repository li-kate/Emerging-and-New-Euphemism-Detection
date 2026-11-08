#!/bin/bash
set -euo pipefail

# ============================================================================
# Self-chaining batch controller for submit_zhu_detection_pipeline.sh.
#
# Not meant to be run by hand — submit_zhu_detection_pipeline.sh seeds it with
# BATCH_INDEX=0 and all the config exported into the environment. It mirrors
# the hit matrix's _hit_batch_controller.sh: each map task requests a GPU (a
# BERT forward pass per masked sentence), and the array indexes into the full
# month list passed through as MONTHS_ARG and handed to
# zhu_detection_month.py --months, so the array index <-> month mapping stays
# correct.
#
# Each invocation:
#   1. submits the map array for its batch (a contiguous range of month
#      indices, at most BATCH_MAX tasks), then
#   2. if more batches remain, submits the NEXT controller invocation as a
#      tiny one-task job that depends (afterany) on this batch's array — so the
#      next batch is only *submitted* once this batch has left the queue;
#   3. otherwise submits the merge job, depending (afterany) on this (final)
#      batch.
#
# Because the next batch is not submitted until the current one finishes, the
# queue never holds more than one map batch (<= BATCH_MAX) plus the single
# pending controller/merge job, keeping the total under the QOS submit cap
# (BATCH_MAX = MAX_SUBMIT - 1, leaving one slot for the chained controller/merge
# job).
#
# afterany (not afterok) chains batches and the merge so one failed month never
# stalls the rest; merge_zhu_detection.py fails only if there are no shards at
# all.
# ============================================================================

# Required config, exported by the pipeline script (or a prior controller).
: "${BATCH_INDEX:?BATCH_INDEX must be set}"
: "${DUMP_DIR:?}"
: "${SUBREDDIT_FILE:?}"
: "${ANSWER_FILE:?}"
: "${MODEL_DIR:?}"
: "${VOCAB_FILE:?}"
: "${OUT_DIR:?}"
: "${SHARD_DIR:?}"
: "${STEM:?}"
: "${START_YEAR:?}"
: "${MLM_K:?}"
: "${THRES:?}"
: "${TOP_N:?}"
: "${PY_DEPS:?}"
: "${BATCH_MAX:?}"
: "${NUM_MONTHS:?}"
: "${NUM_BATCHES:?}"
: "${CONTROLLER:?}"
: "${MONTHS_ARG:?}"

# This controller runs either on the submit host (batch 0, from the pipeline
# script) or on a compute node (later batches, as a chained job). Run sbatch
# from the directory holding the Python scripts either way.
cd "$(dirname "$CONTROLLER")"
mkdir -p logs_zhu_detect

# --- Compute this batch's contiguous month-index range -----------------------
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
    --job-name="zhu_map_b${BATCH_INDEX}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --mem=32G \
    --time=8:00:00 \
    --array="$ARRAY_SPEC" \
    --output="logs_zhu_detect/map_%A_%a.out" \
    --mail-type=FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install $PY_DEPS
        python -u zhu_detection_month.py \
            --reddit-monthly-dir '$DUMP_DIR' \
            --subreddit-file '$SUBREDDIT_FILE' \
            --answer-file '$ANSWER_FILE' \
            --model-dir '$MODEL_DIR' \
            --vocab-file '$VOCAB_FILE' \
            --shard-dir '$SHARD_DIR' \
            --shard-index \"\$SLURM_ARRAY_TASK_ID\" \
            --months $MONTHS_ARG \
            --start-year $START_YEAR \
            --mlm-k $MLM_K \
            --thres $THRES
    ")

echo "Batch $((BATCH_INDEX + 1))/$NUM_BATCHES: map array $MAP_JOB (month indices $ARRAY_SPEC)"

NEXT_INDEX=$(( BATCH_INDEX + 1 ))

if (( NEXT_INDEX < NUM_BATCHES )); then
    # More batches remain: submit the next controller invocation as a tiny
    # one-task job that waits for this batch to leave the queue, then submits
    # the next batch. It re-exports the same config plus the new BATCH_INDEX.
    NEXT_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
        --job-name="zhu_ctl_b${NEXT_INDEX}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=2G \
        --time=00:10:00 \
        --output="logs_zhu_detect/controller_%j.out" \
        --mail-type=FAIL \
        --export="ALL,BATCH_INDEX=${NEXT_INDEX}" \
        --wrap="set -euo pipefail; '$CONTROLLER'")
    echo "  chained next controller: $NEXT_JOB (BATCH_INDEX=$NEXT_INDEX, after $MAP_JOB)"
else
    # Final batch: submit the merge after it leaves the queue.
    MERGE_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
        --job-name=zhu_merge \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=16G \
        --time=1:00:00 \
        --output="logs_zhu_detect/merge_%j.out" \
        --mail-type=END,FAIL \
        --wrap="
            set -euo pipefail
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            pip install $PY_DEPS
            python -u merge_zhu_detection.py \
                --shard-dir '$SHARD_DIR' \
                --out-dir '$OUT_DIR' \
                --stem '$STEM' \
                --top-n '$TOP_N'
        ")
    echo "  final batch — submitted merge: $MERGE_JOB (after $MAP_JOB)"
fi
