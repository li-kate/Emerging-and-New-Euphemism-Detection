#!/bin/bash
#SBATCH --job-name=cosine-test
#SBATCH --account=gts-smussmann3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
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
export PYTHONPATH="$SLURM_SUBMIT_DIR/..:$PYTHONPATH"
python -u similarity_matrix.py

echo "Process Finished at: $(date)"
