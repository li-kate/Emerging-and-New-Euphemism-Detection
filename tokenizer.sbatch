#!/bin/bash
#SBATCH --job-name=tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/task_%a.out
#SBATCH --mail-type=BEGIN,END,FAIL

# Move to the directory where you submitted the job
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Load your modules
module load anaconda3

# Activate your environment
conda activate medieval_env

# Run your script
python -u tokenizer.py \
    --input_dir /storage/ice-shared/vip-vyf/early_modern_english_data/eebo_all/simplified_content \
    --output_dir /storage/ice-shared/vip-vyf/early_modern_english_data/tokenized_data/ \
    --method moses \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_tasks $SLURM_ARRAY_TASK_COUNT