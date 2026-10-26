#!/bin/bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs_drug_centroid_test

# Count monthly Reddit dumps.
DUMP_DIR="/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
shopt -s nullglob
TEST_MONTHS=("2015-01" "2015-02")

dumps=()
for month in "${TEST_MONTHS[@]}"; do
    for prefix in RC RS; do
        file="$DUMP_DIR/${prefix}_${month}.zst"
        if [[ ! -f "$file" ]]; then
            echo "ERROR: Missing expected test file: $file"
            exit 1
        fi
        dumps+=("$file")
    done
done

NUM_FILES=${#dumps[@]}
echo "Testing $NUM_FILES dump files across ${#TEST_MONTHS[@]} months."
printf '  %s\n' "${dumps[@]}"

# Submit the embedding array.
EMBED_JOB=$(sbatch --parsable \
    --job-name=drug_centroid_test \
    --account=gts-smussmann3 \
    --qos=embers \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=32G \
    --gres=gpu:1 \
    --time=6:00:00 \
    --array="0-$((NUM_FILES - 1))" \
    --mail-type=BEGIN,END,FAIL \
    --output="logs_drug_centroid_test/array_%A_%a.out" \
    --wrap='
        set -euo pipefail
        module load anaconda3
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate base
    
        echo "===== SLURM / GPU DIAGNOSTICS ====="
        echo "Host: $(hostname)"
        echo "Job ID: $SLURM_JOB_ID"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi
    
        python -u check_gpu.py
    
        python -u drug_centroid_embeddings_test.py
    ')

echo "Submitted embedding array: $EMBED_JOB"

# Submit the merge job, dependent on successful completion of the array.
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${EMBED_JOB}" \
    --account=gts-smussmann3 \
    --qos=embers \
    --job-name=drug_merge_test \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=8G \
    --time=00:30:00 \
    --mail-type=BEGIN,END,FAIL \
    --output="logs_drug_centroid_test/merge_%j.out" \
    --wrap='
        set -euo pipefail
        module load anaconda3
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate base
        python -u drug_centroid_merge_results_test.py
    ')

echo "Submitted merge job: $MERGE_JOB"
echo "The merge will start after all embedding tasks succeed."

# to run:
    # chmod +x test_drug_centroid_pipeline.sh
    # ./test_drug_centroid_pipeline.sh