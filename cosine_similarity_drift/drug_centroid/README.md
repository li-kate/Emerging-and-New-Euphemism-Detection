# Drug Centroid Construction

## Overview

This pipeline constructs contextual BERT embeddings for known drug-related terms using Reddit data. It is designed to produce a global drug centroid and individual term centroids for use in emerging euphemism detection and semantic drift analysis.

The pipeline uses `bert-base-uncased` to embed each matched occurrence in its surrounding context. Only posts and comments from the subreddits listed in `drug_subreddit_list.txt` are processed.

### What the pipeline produces

- **Global drug centroid:** The average contextual embedding across all successfully embedded drug-term occurrences. Each occurrence contributes equally, so frequently occurring terms contribute more than infrequent terms.
- **Per-term centroids:** One average contextual embedding for each drug term, calculated using all successfully embedded occurrences of that term.
- **Occurrence counts:** Counts of embedded occurrences for each term and month, supporting later frequency and temporal analyses.

## How the pipeline works

1. **Load target terms and subreddit filters.** Read the known drug terms from `drug_synonyms.txt` and the allowed subreddit names from `drug_subreddit_list.txt`.
2. **Process Reddit dumps.** Each SLURM array task processes one `RC_*.zst` or `RS_*.zst` dump file.
3. **Find term occurrences.** Match target terms using case-insensitive, whole-word matching.
4. **Extract contextual embeddings.** For each occurrence, tokenize its surrounding text and extract up to 64 WordPieces before and after the target term. The target's WordPiece hidden states are averaged to obtain one contextual embedding.
5. **Aggregate results.** Accumulate embedding sums and occurrence counts by term and month. Each task periodically checkpoints its progress.
6. **Merge partial results.** After every embedding task completes successfully, the merge job combines the partial results and calculates the global and per-term centroids.

The context window uses BERT's standard special tokens and remains within its maximum input length for ordinary target terms.

## Before running

### 1. Configure the paths

In `drug_centroid_embeddings.py`, set:

- `REDDIT_MONTHLY_DIR`: Directory containing the compressed Reddit dump files.
- `SYNONYMS_FILE`: Path to `drug_synonyms.txt`.
- `SUBREDDIT_LIST_FILE`: Path to `drug_subreddit_list.txt`.
- `OUTPUT_DIR`: Directory where task checkpoints and partial results will be stored.

In `drug_centroid_merge_results.py`, set:

- `RESULTS_DIR`: **Must exactly match `OUTPUT_DIR` in `drug_centroid_embeddings.py`.**
- `SYNONYMS_FILE`: Path to the same drug synonym list.
- `OUTPUT_DIR`: Directory where the final merged centroid JSON will be written.

Ensure the configured paths are accessible from the PACE compute nodes. Relative paths are resolved from the job's working directory, so absolute paths are preferable for shared data and output directories.

### 2. Check the Python environment

The active Conda environment must have the required dependencies installed:

- `numpy`
- `torch`
- `transformers`
- `zstandard`
- `pyahocorasick`
- `orjson` (optional; used for faster JSON parsing when available)

The embedding jobs also require access to a compatible GPU and permission to load the pretrained `bert-base-uncased` model. Install dependencies before submitting the jobs rather than installing packages separately in each array task.

### 3. Check the input files

Confirm that:

- The Reddit directory contains `RC_*.zst` and/or `RS_*.zst` files.
- `drug_synonyms.txt` contains one target term per line.
- `drug_subreddit_list.txt` contains the allowed subreddit names, one per line.
- The output and checkpoint directories are writable.

Lines beginning with `#` are ignored in the list files.

## Running the pipeline

From the directory containing `submit_drug_centroid_pipeline.sh`, run:

```bash
chmod +x submit_drug_centroid_pipeline.sh
./submit_drug_centroid_pipeline.sh
```

The submission script automatically counts the available Reddit dump files and submits:

1. **Embedding array:** One task per dump file, with one GPU requested per task.
2. **Merge job:** Starts only after all embedding tasks finish successfully.

You do not need to manually specify the array size or submit the merge job separately.

**Important:** Use the submission script as the main entry point. Do not also submit the standalone embedding and merge job scripts for the same run, since doing so could duplicate processing.

## Monitoring jobs

Check the job queue:

```bash
squeue -u "$USER"
```

Check the output and error logs under `logs_drug_centroid/`:

```text
logs_drug_centroid/
├── array_<jobid>_<taskid>.out
└── merge_<jobid>.out
```

The embedding logs report progress, matched occurrences, successfully embedded occurrences, and final task statistics. The merge log reports the aggregation results and final output location.

## Output files

### Intermediate files

The embedding tasks write files such as:

```text
partial_results_0.json
partial_results_1.json
...
```

These contain per-term, per-month embedding sums and occurrence counts, along with task statistics. Checkpoints allow an interrupted task to resume from its saved progress.

Keep these files until the merge job completes successfully.

### Final merged output

The merge job writes `global_drug_centroid.json` (or the filename configured in the merge script).

The output contains:

- `drug_centroid`: Global occurrence-weighted contextual embedding.
- `term_centroids`: Contextual embedding for each individual target term.
- `term_counts`: Number of embedded occurrences per term.
- `monthly_counts`: Occurrence counts grouped by month.
- `statistics`: Summary statistics, including task coverage and terms without embeddings.

The global centroid is calculated by dividing the sum of all embedded occurrence vectors by the total number of successfully embedded occurrences. It is **not** calculated by averaging the per-term centroids, because that would give each distinct term equal weight regardless of its frequency.