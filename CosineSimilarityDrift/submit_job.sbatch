#!/bin/bash
#SBATCH --job-name=drift_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                       # Higher memory for large BERT processing
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=12:00:00                 # GPUs are faster, shorter time needed
#SBATCH --output=./logs/drift_%j.out
#SBATCH --error=./logs/drift_%j.err

# ============================================================================
# SETUP
# ============================================================================

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs plots

module load anaconda3
conda activate base

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
python -u analyze_emerging_csd.py

echo "Process Finished at: $(date)"