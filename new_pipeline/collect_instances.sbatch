#!/bin/bash
#SBATCH --job-name=collect_instances
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/task_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# Move to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Load your modules
module load anaconda3

# Activate your environment
conda activate base

# Run your script
python -u collect_instances.py \
    --words drug_words.txt anchors_and_baselines.txt comparison_words.txt \
    --reddit-monthly-dir ../reddit_dump/ \
    --output ./matches/ \
    --slurm-task-id $SLURM_ARRAY_TASK_ID
