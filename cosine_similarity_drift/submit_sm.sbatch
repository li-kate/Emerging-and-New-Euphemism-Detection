#!/bin/bash
#SBATCH --job-name=cosine-similarity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=logs_cosine_similarity/array_%A_%a.out

set -euo pipefail

# ============================================================================
# SETUP
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs_cosine_similarity

module load anaconda3
conda activate base

# ============================================================================
# RUN
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Node: $HOSTNAME"
echo "Process Started at: $(date)"
echo "============================================"

# Unbuffered output for real-time logs
python -u similarity_matrix.py

echo "Process Finished at: $(date)"
