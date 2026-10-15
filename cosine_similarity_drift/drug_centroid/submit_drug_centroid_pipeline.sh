#!/bin/bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs_drug_centroid

# Count monthly Reddit dumps.
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
shopt -s nullglob
dumps=("$DUMP_DIR"/RC_*.zst "$DUMP_DIR"/RS_*.zst)
shopt -u nullglob

NUM_FILES=${#dumps[@]}

if [[ "$NUM_FILES" -eq 0 ]]; then
    echo "ERROR: No RC_*.zst or RS_*.zst files found in $DUMP_DIR"
    exit 1
fi

echo "Found $NUM_FILES total Reddit dump files."

# Submit the embedding array.
EMBED_JOB=$(sbatch --parsable \
    --job-name=drug_centroid \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --gres=gpu:1 \
    --time=6:00:00 \
    --array="0-$((NUM_FILES - 1))" \
    --output="logs_drug_centroid/array_%A_%a.out" \
    --wrap='
        set -euo pipefail
        module load anaconda3
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate base
        python -u drug_centroid_embeddings.py
    ')

echo "Submitted embedding array: $EMBED_JOB"

# Submit the merge job, dependent on successful completion of the array.
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${EMBED_JOB}" \
    --job-name=drug_merge \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=8G \
    --time=00:30:00 \
    --output="logs_drug_centroid/merge_%j.out" \
    --wrap='
        set -euo pipefail
        module load anaconda3
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate base
        python -u drug_centroid_merge_results.py
    ')

echo "Submitted merge job: $MERGE_JOB"
echo "The merge will start after all embedding tasks succeed."

# to run:
    # chmod +x submit_drug_centroid_pipeline.sh
    # ./submit_drug_centroid_pipeline.sh