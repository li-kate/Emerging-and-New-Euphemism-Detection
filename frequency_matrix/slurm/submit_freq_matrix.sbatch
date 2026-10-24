#!/bin/bash
#SBATCH --job-name=freq_matrix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=./logs_freq_matrix/freq_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# NOTE: this is a SINGLE (non-array) job on purpose.
# The frequency matrix is built in one pass over the whole corpus so that a
# single [half_years x words] matrix and its vocabulary are consistent. It
# does not use a SLURM job array like collect_instances.py.
# No GPU is requested — counting is pure CPU work.
# ============================================================================

# Postfilter controls (override at submit time, e.g.
#   sbatch --export=ALL,NO_POSTFILTER=1 submit_freq_matrix.sbatch
# or POSTFILTER_MODE=global, or MIN_FREQ=50).
#   POSTFILTER_MODE=per-half-year  null cells < MIN_FREQ in their own
#                                  half-year, drop words nulled in every
#                                  half-year (leakage-safe; default).
#   POSTFILTER_MODE=global         legacy: keep a word if any half-year
#                                  reaches MIN_FREQ, leave its cells as-is.
#   NO_POSTFILTER=1                skip the postfilter entirely (overrides
#                                  POSTFILTER_MODE). THIS IS THE DEFAULT — the
#                                  first build is intentionally unfiltered so
#                                  the postfilter can be chosen/applied later.
#                                  Set NO_POSTFILTER=0 to enable POSTFILTER_MODE.
#   MIN_FREQ                       threshold for both modes (default 30).
POSTFILTER_MODE="${POSTFILTER_MODE:-per-half-year}"
NO_POSTFILTER="${NO_POSTFILTER:-1}"
MIN_FREQ="${MIN_FREQ:-}"

POSTFILTER_ARGS=""
if [[ "$NO_POSTFILTER" == "1" ]]; then
    POSTFILTER_ARGS="--no-postfilter"
else
    POSTFILTER_ARGS="--postfilter-mode $POSTFILTER_MODE"
fi
if [[ -n "$MIN_FREQ" ]]; then
    POSTFILTER_ARGS="$POSTFILTER_ARGS --min-freq $MIN_FREQ"
fi

# Move to the directory where you submitted the job (this folder).
cd "$SLURM_SUBMIT_DIR"

# Create logs directory if it doesn't exist.
mkdir -p logs_freq_matrix

# Load modules and activate the environment.
module load anaconda3
conda activate base

# Install dependencies (matches the pattern used by the other submit scripts).
pip install numpy pandas nltk zstandard orjson

# Pre-download the NLTK stopwords corpus now, in case the compute node that
# runs the job has no network access (filter.py needs it at import time).
python -u -c "import nltk; nltk.download('stopwords')"

# ============================================================================
# RUN
# ============================================================================

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Process Started at: $(date)"
echo "Postfilter: ${POSTFILTER_ARGS:-<default>}"
echo "============================================"

# Build the frequency matrix over the full Reddit dump directory, scoped to
# the drug subreddit list. Outputs land in ./freq_drug_subreddits/ next to
# this script.
python -u freq_matrix.py \
    --reddit-monthly-dir /storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst \
    --subreddit-file drug_subreddit_list.txt \
    --out-dir freq_drug_subreddits \
    --stem freq \
    --dtype int32 \
    $POSTFILTER_ARGS

echo "Process Finished at: $(date)"
