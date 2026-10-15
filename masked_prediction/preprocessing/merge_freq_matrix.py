"""
Reduce step for the parallel frequency-matrix build.

``freq_matrix.py --mode map`` writes one partial-count shard per SLURM array
task into ``--shard-dir``. This script sums all those shards into one global
count, applies the postfilter once (it is a global threshold, so it must run
after merging, not per shard), assembles the dense ``[half_years, words]``
matrix, and writes the same final outputs the single-process build produces:
``freq.npy``, label files, the stop-word list, totals, and ``freq_meta.json``.

Scope note: the scope flags here (``--subreddit-file``, ``--start-year`` /
``--end-year``) are only recorded in the metadata. They do not re-filter
anything — the map tasks already applied the subreddit and year filters. Pass
the *same* values the map tasks used so ``freq_meta.json`` is accurate; the
recorded subreddit hash lets the hit script confirm the scope.

Example:

    python merge_freq_matrix.py \\
        --shard-dir freq_drug_subreddits/shards \\
        --out-dir freq_drug_subreddits \\
        --subreddit-file drug_subreddit_list.txt \\
        --dtype int32
"""

import argparse
import glob
import json
import logging
import os
from collections import defaultdict

import numpy as np

# Reuse everything shared with the single-process build.
from freq_matrix import (
    MIN_FREQ,
    SHARD_SUFFIX,
    assemble_matrix,
    build_output_names,
    log_postfilter_counts,
    log_prefilter_counts,
    postfilter,
    save_outputs,
)
from subreddit_scope import load_subreddits, subreddit_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_shards(shard_dir):
    """Return the sorted list of shard files in ``shard_dir``."""
    pattern = os.path.join(shard_dir, f"*{SHARD_SUFFIX}")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No shards matching *{SHARD_SUFFIX} found in {shard_dir}. "
            f"Did the map array run and finish?"
        )
    return shards


def merge_shards(shard_files):
    """
    Sum all shards into global counts and tallies.

    Returns the same four values ``build_counts`` returns, so the downstream
    assemble/postfilter/save path is identical to the single-process build:
      counts            - dict[word] -> dict[half_year] -> int
      totals_raw        - dict[half_year] -> int
      prefilter_counts  - dict[rule] -> int
      half_years        - set of half-year labels
    Also returns the summed scope drop counts for logging.
    """
    counts = defaultdict(lambda: defaultdict(int))
    totals_raw = defaultdict(int)
    prefilter_counts = {"short_terms": 0, "multi_word_terms": 0, "stop_words": 0}
    drop_counts = defaultdict(int)
    half_years = set()

    for i, path in enumerate(shard_files, 1):
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)

        for word, by_hy in shard["counts"].items():
            dst = counts[word]
            for hy, c in by_hy.items():
                dst[hy] += c
                half_years.add(hy)

        for hy, c in shard.get("totals_raw", {}).items():
            totals_raw[hy] += c
            half_years.add(hy)

        for rule, c in shard.get("prefilter_counts", {}).items():
            prefilter_counts[rule] = prefilter_counts.get(rule, 0) + c

        for reason, c in shard.get("drop_counts", {}).items():
            drop_counts[reason] += c

        for hy in shard.get("half_years", []):
            half_years.add(hy)

        logger.info(
            "Merged shard %d/%d: %s (running: %s terms)",
            i,
            len(shard_files),
            os.path.basename(path),
            f"{len(counts):,}",
        )

    return counts, totals_raw, prefilter_counts, half_years, drop_counts


def main():
    parser = argparse.ArgumentParser(
        description="Merge frequency-matrix shards into the final matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--shard-dir",
        required=True,
        help="Directory holding the *.part.json shards from --mode map.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the final matrix and label files.",
    )
    parser.add_argument(
        "--stem",
        default="freq",
        help="Filename stem for all outputs.",
    )
    parser.add_argument(
        "--dtype",
        choices=["int32", "int64"],
        default="int32",
        help="Integer size for the matrix.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=MIN_FREQ,
        help="Postfilter: keep a word reaching this count in >=1 half-year.",
    )

    # Scope / metadata (recorded only; must match the map run).
    parser.add_argument(
        "--subreddit-file",
        default=None,
        help="Same subreddit file the map tasks used (recorded in metadata).",
    )
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--version", default="unversioned")

    args = parser.parse_args()

    dtype = np.int32 if args.dtype == "int32" else np.int64
    names = build_output_names(args.stem)

    # Reduce.
    shard_files = find_shards(args.shard_dir)
    logger.info("Found %d shard(s) in %s", len(shard_files), args.shard_dir)

    counts, totals_raw, prefilter_counts, half_years, drop_counts = merge_shards(
        shard_files
    )

    if not counts:
        logger.error("No terms after merging shards. Check the map outputs.")
        raise SystemExit(1)

    # Log prefilter (summed across shards) and scope drops.
    n_kept_terms = len(counts)
    n_prefiltered = sum(prefilter_counts.values())
    log_prefilter_counts(prefilter_counts, n_kept_terms + n_prefiltered)
    logger.info(
        "Scope drops — subreddit:%s empty:%s year:%s bad_ts:%s parse:%s",
        f"{drop_counts['subreddit']:,}",
        f"{drop_counts['empty']:,}",
        f"{drop_counts['year']:,}",
        f"{drop_counts['bad_timestamp']:,}",
        f"{drop_counts['parse_error']:,}",
    )

    # Assemble + postfilter (identical to the single-process path).
    freq, half_years, words = assemble_matrix(counts, half_years, dtype)
    logger.info("Pre-postfilter matrix: %s", freq.shape)

    total_before = len(words)
    freq, words, postfilter_counts = postfilter(freq, words, args.min_freq)
    log_postfilter_counts(postfilter_counts, total_before, args.min_freq)

    totals_filtered = {h: int(freq[i].sum()) for i, h in enumerate(half_years)}

    # Scope metadata.
    allowed = load_subreddits(args.subreddit_file)
    scope = {
        "mode": "subreddits" if allowed is not None else "all",
        "file": args.subreddit_file,
        "hash": subreddit_hash(allowed),
        "n_listed": len(allowed) if allowed is not None else None,
        "n_records_dropped_by_subreddit_filter": int(drop_counts["subreddit"]),
        "n_shards_merged": len(shard_files),
    }

    save_outputs(
        args.out_dir,
        freq,
        half_years,
        words,
        totals_raw,
        totals_filtered,
        prefilter_counts,
        postfilter_counts,
        scope,
        args,
        names,
    )

    logger.info(
        "Final matrix: %s half-years x %s words (%s), merged from %d shards, "
        "saved to %s",
        len(half_years),
        f"{len(words):,}",
        args.dtype,
        len(shard_files),
        args.out_dir,
    )


if __name__ == "__main__":
    main()
