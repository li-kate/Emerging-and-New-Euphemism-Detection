#!/bin/bash
#SBATCH --job-name=cs_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs_cosine_similarity/merge_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load anaconda3
conda activate base

python -u merge_cosine_similarity.py
