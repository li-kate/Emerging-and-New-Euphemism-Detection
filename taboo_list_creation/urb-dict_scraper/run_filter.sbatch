#!/bin/bash
#SBATCH --account=gts-smussmann3
#SBATCH --job-name=drug-filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs_filter/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

cd $SLURM_SUBMIT_DIR

mkdir -p logs_filter

module load anaconda3
conda activate ubscrape

python filter_drug_definitions.py