#!/bin/bash
#SBATCH --job-name=plots
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                       # Higher memory for large BERT processing
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=12:00:00                 # GPUs are faster, shorter time needed
#SBATCH --output=logs3/plotting_%j.out

module load anaconda3
conda activate base

python merge_plot.py