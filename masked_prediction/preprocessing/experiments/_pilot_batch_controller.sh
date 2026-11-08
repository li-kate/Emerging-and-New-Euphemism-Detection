#!/bin/bash
set -euo pipefail

# ============================================================================
# Self-chaining batch controller for submit_pilot_hits_pipeline.sh.
#
# Not meant to be run by hand — submit_pilot_hits_pipeline.sh seeds it with
# BATCH_INDEX=0 and the config exported into the environment. Mirrors
# _hit_batch_controller.sh exactly, but runs pilot_hits.py (per-context pilot
# rows) instead of hit_matrix.py and merges with merge_pilot_hits.py.
# ============================================================================

: "${BATCH_INDEX:?BATCH_INDEX must be set}"
: "${DUMP_DIR:?}"
: "${SUBREDDIT_FILE:?}"
: "${FREQ_DIR:?}"
: "${FREQ_STEM:?}"
: "${DEA_FILE:?}"
: "${PILOT_FILE:?}"
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

cd "$(dirname "$CONTROLLER")"
mkdir -p logs_pilot_hits

START=$(( BATCH_INDEX * BATCH_MAX ))
END=$(( START + BATCH_MAX - 1 ))
if (( END > NUM_MONTHS - 1 )); then
    END=$(( NUM_MONTHS - 1 ))
fi
BATCH_SIZE_TASKS=$(( END - START + 1 ))

ARRAY_SPEC="${START}-${END}%${BATCH_SIZE_TASKS}"

MAP_JOB=$(sbatch --parsable \
    --job-name="pilot_map_b${BATCH_INDEX}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --mem=32G \
    --time=8:00:00 \
    --array="$ARRAY_SPEC" \
    --output="logs_pilot_hits/map_%A_%a.out" \
    --mail-type=FAIL \
    --wrap="
        set -euo pipefail
        module load anaconda3
        source \"\$(conda info --base)/etc/profile.d/conda.sh\"
        conda activate base
        pip install $PY_DEPS
        python -u pilot_hits.py \
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
            --start-year $START_YEAR \
            --batch-size $BATCH_SIZE \
            --fp16
    ")

echo "Batch $((BATCH_INDEX + 1))/$NUM_BATCHES: pilot map array $MAP_JOB (kept-month indices $ARRAY_SPEC)"

NEXT_INDEX=$(( BATCH_INDEX + 1 ))

if (( NEXT_INDEX < NUM_BATCHES )); then
    NEXT_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
        --job-name="pilot_ctl_b${NEXT_INDEX}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=2G \
        --time=00:10:00 \
        --output="logs_pilot_hits/controller_%j.out" \
        --mail-type=FAIL \
        --export="ALL,BATCH_INDEX=${NEXT_INDEX}" \
        --wrap="set -euo pipefail; '$CONTROLLER'")
    echo "  chained next controller: $NEXT_JOB (BATCH_INDEX=$NEXT_INDEX, after $MAP_JOB)"
else
    MERGE_JOB=$(sbatch --parsable \
        --dependency="afterany:${MAP_JOB}" \
        --job-name=pilot_merge \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=16G \
        --time=1:00:00 \
        --output="logs_pilot_hits/merge_%j.out" \
        --mail-type=END,FAIL \
        --wrap="
            set -euo pipefail
            module load anaconda3
            source \"\$(conda info --base)/etc/profile.d/conda.sh\"
            conda activate base
            pip install $PY_DEPS
            python -u merge_pilot_hits.py \
                --shard-dir '$SHARD_DIR' \
                --out-dir '$OUT_DIR' \
                --stem '$STEM'
        ")
    echo "  final batch — submitted pilot merge: $MERGE_JOB (after $MAP_JOB)"
fi
