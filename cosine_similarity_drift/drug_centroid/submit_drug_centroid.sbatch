#!/bin/bash
#SBATCH --job-name=drug_centroid_resume
#SBATCH --account=gts-smussmann3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=120,126,128,129
#SBATCH --time=6:00:00
#SBATCH --mail=BEGIN,END,FAIL
#SBATCH --output=logs_hy_centroids/array_%A_%a.out

# ============================================================================
# SETUP
# ============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

module load anaconda3
conda activate base

# pip install numpy torch transformers zstandard pyahocorasick orjson

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