#!/bin/bash
#SBATCH --job-name=drug_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs_drug_centroid/merge_%j.out

cd "$SLURM_SUBMIT_DIR"

module load anaconda3
conda activate based

python -u drug_centroid_merge_results.py