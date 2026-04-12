#!/bin/bash
#SBATCH --job-name=tv_transform
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=./logs/tv_trans_%j.out
#SBATCH --error=./logs/tv_trans_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# SETUP
# ============================================================================
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs ./results/tv_final

module load anaconda3
conda activate your_env_name  # Ensure you've installed dateparser, tmdbsimple, and datasets

# ============================================================================
# EXECUTION
# ============================================================================
echo "Running Array Task: $SLURM_ARRAY_TASK_ID"

python -u process_tv_v3.py \
    --output_dir "./results/tv_final" \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --num_tasks "$SLURM_ARRAY_TASK_COUNT"

echo "Job Finished."