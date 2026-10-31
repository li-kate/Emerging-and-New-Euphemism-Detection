#!/bin/bash
#SBATCH --job-name=merge-cosine-monthly
#SBATCH --account=gts-smussmann3
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs_cosine_similarity/merge_monthly_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

python merge_similarity_matrix_monthly.py