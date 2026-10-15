#!/bin/bash
#SBATCH --job-name=drift_parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-10
#SBATCH --output=logs_drug_centroid/array_%A_%a.out

# ============================================================================
# SETUP
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_drug_centroid

module load anaconda3
conda activate base

pip install numpy torch transformers zstandard pyahocorasick orjson

# ============================================================================
# RUN
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Node: $HOSTNAME"
echo "Process Started at: $(date)"
echo "============================================"

# Unbuffered output for real-time logs
python -u drug_centroid_embeddings.py

echo "Process Finished at: $(date)"