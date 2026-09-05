#!/bin/bash
#SBATCH --account=gts-smussmann3
#SBATCH --job-name=ubscrape
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=defining-logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

cd $SLURM_SUBMIT_DIR

mkdir -p defining-logs

module load anaconda3
conda activate base

python -m ubscrape --define-all
