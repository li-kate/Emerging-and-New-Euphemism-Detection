#!/bin/bash
#SBATCH --job-name=masked_pred_halfyear
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=44-80
#SBATCH --output=logs/masked_%A_%a.out
#SBATCH --error=logs/masked_%A_%a.err

set -euo pipefail

# Ensure the log directory exists
mkdir -p logs

echo "=== Masked Prediction Job ==="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $HOSTNAME"

# The script uses DATA_DIR = "../matches" internally. 
# Make sure you are running from the directory containing masked_prediction.py.
python -u masked_prediction.py

EXIT_CODE=$?
echo "Task ${SLURM_ARRAY_TASK_ID} completed with exit code: $EXIT_CODE"
exit $EXIT_CODE