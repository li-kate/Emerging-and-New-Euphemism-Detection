#!/bin/bash
#SBATCH --job-name=drift_parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-10
#SBATCH --output=logs2/array_%A_%a.ou

# ============================================================================
# SETUP
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs2

module load anaconda3
conda activate base

pip install torch transformers scikit-learn matplotlib tqdm

# If you need specific pytorch/transformers versions:
# pip install torch transformers scikit-learn matplotlib tqdm --user

# ============================================================================
# RUN
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU Node: $HOSTNAME"
echo "Process Started at: $(date)"
echo "============================================"

# Unbuffered output for real-time logs
python -u analyze_emerging_csd_parallel.py

echo "Process Finished at: $(date)"