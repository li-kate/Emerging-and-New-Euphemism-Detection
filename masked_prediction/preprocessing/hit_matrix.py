"""
Build the hit matrix: how often a masked language model predicts a DEA drug
term in place of each target word, per half-year.

This is the per-month MAP step of the hit matrix pipeline described in
``hit_matrix_plan.md``. One SLURM array task scores one month of the corpus and
writes a shard; ``merge_hit_matrix.py`` sums the shards into the final summary
table and matrices (aligned to the frequency matrix).

This script is **matrix-only**: it scores every eligible target word and keeps
only the aggregated per (half_year, word) cell counts. The two optional
analyses that used to ride along on this pass now live in their own scripts,
each with its own map + merge, so a plain matrix rerun never touches them:

  * per-context rows for the pilot/control words -> ``pilot_hits.py`` /
    ``merge_pilot_hits.py``.
  * a random sample of one word's full contexts for manual spot-checking ->
    ``sample_contexts.py`` / ``merge_sample_contexts.py``.

All three share the same scoring core (``hit_matrix_core.py``), so their
numbers stay comparable.

What one task does, in a single pass over its month (plan section 5): for every
occurrence of every eligible target word, mask the one occurrence, run BERT,
and update running per (half_year, word) counts — ``n_scored``, ``hits_k`` for
every k in ``KS``, ``min_rank``, ``sum(log(best_rank))``, ``sum(soft_score)``.

Per context (plan section 4): mask the one occurrence, run BERT, slice logits
at the mask, exclude the target word + junk tokens, and over the full
vocabulary compute the best (lowest) rank reached by any DEA term and the total
softmax probability on DEA terms. Only those two scalars leave the GPU.

Occurrences come from the SAME tokenization the frequency matrix used (shared
``corpus_stream.tokenize`` / ``iter_token_spans``), so ``n_scored`` can be
checked against the frequency denominator (plan section 8).

Scope safety: at startup the subreddit mode + list hash are compared against
the frequency matrix's ``freq_meta.json``; a mismatch aborts, since dividing by
the wrong denominator would silently corrupt every hit rate.

Example (one month, local test):

    python hit_matrix.py \\
        --mode map --shard-index 0 \\
        --reddit-monthly-dir /data/reddit \\
        --subreddit-file drug_subreddit_list.txt \\
        --freq-dir freq_drug_subreddits \\
        --dea-file ../../data_collection/word_lists/drug_synonyms.txt \\
        --shard-dir hits_drug_subreddits/shards
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

# Local shared modules (same directory). sys.path is adjusted before these
# imports, so they intentionally do not sit at the top of the file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_stream import (
    resolve_inputs,
    select_month_key,
    stream_corpus,
)
from hit_matrix_core import (
    DEFAULT_MAX_SIDE_WORDS,
    DEFAULT_MIN_CONTEXT_WORDS,
    KS,
    SHARD_SUFFIX,
    build_scoring_tensors,
    build_exclusion_ids,
    iter_scored_occurrences,
    load_dea_terms,
    load_freq_meta,
    load_model,
    load_target_words,
    new_cell,
    resolve_dea_token_ids,
    score_batch,
    update_cell,
    verify_scope,
)
from subreddit_scope import load_subreddits, subreddit_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# SHARD IO
# ============================================================================


def write_shard(shard_dir, shard_name, cells, dropped, drop_counts,
                dea_meta, files, scope, target_stats=None):
    """Write one month's partial hit counts as a JSON shard."""
    os.makedirs(shard_dir, exist_ok=True)
    final_path = os.path.join(shard_dir, f"{shard_name}{SHARD_SUFFIX}")
    tmp_path = final_path + ".tmp"

    payload = {
        "shard_name": shard_name,
        "ks": KS,
        "scope": scope,
        "dea": dea_meta,
        # Pre-scoring frequency floor + kept/dropped counts (min_freq gate).
        "target_selection": target_stats or {},
        # cells keyed "half_year\tword" so JSON stays flat and mergeable.
        "cells": {f"{hy}\t{w}": cell for (hy, w), cell in cells.items()},
        "dropped_contexts": dropped,
        "drop_counts": dict(drop_counts),
        "files": list(files),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, final_path)
    logger.info("Wrote shard %s (%d cells)", final_path, len(cells))
    return final_path


# ============================================================================
# MAIN
# ============================================================================


def process_month(stream, target_words, eligible_from, dea_ids, exclude_ids,
                  model, tokenizer, device, args):
    """Single pass over a month's records: build contexts, score, aggregate.

    Scores every eligible occurrence of every target word into running per
    (half_year, word) cells. The eligibility rule and context construction are
    applied by ``iter_scored_occurrences`` (shared core), so this script and
    the experiments select and build contexts identically.
    """
    dea_ids_t, exclude_ids_t = build_scoring_tensors(dea_ids, exclude_ids, device)
    mask_id = tokenizer.mask_token_id

    cells = defaultdict(new_cell)
    dropped = defaultdict(int)
    n_contexts = 0

    # Batch buffers.
    buf_ctx, buf_word, buf_hy, buf_target_ids = [], [], [], []

    def flush():
        nonlocal buf_ctx, buf_word, buf_hy, buf_target_ids
        if not buf_ctx:
            return
        best_rank, soft_score = score_batch(
            buf_ctx, model, tokenizer, device, dea_ids_t, exclude_ids_t,
            buf_target_ids, mask_id, max_length=args.max_length,
        )
        for w, hy, r, s in zip(buf_word, buf_hy, best_rank, soft_score):
            update_cell(cells[(hy, w)], r, s)
        buf_ctx, buf_word, buf_hy, buf_target_ids = [], [], [], []

    occurrences = iter_scored_occurrences(
        stream, target_words, eligible_from, tokenizer,
        args.min_context_words, args.max_side_words, dropped,
    )
    for word, half_year, context, tgt_ids in occurrences:
        buf_ctx.append(context)
        buf_word.append(word)
        buf_hy.append(half_year)
        buf_target_ids.append(tgt_ids)
        n_contexts += 1
        if len(buf_ctx) >= args.batch_size:
            flush()

    flush()
    return cells, dropped, n_contexts


def main():
    parser = argparse.ArgumentParser(
        description="Build the hit matrix, one month per task (hit_matrix_plan.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode", choices=["map"], default="map",
        help="Only map (per-month) mode exists; merge_hit_matrix.py reduces.",
    )

    # Inputs.
    src = parser.add_argument_group("inputs")
    src.add_argument("--input", nargs="+")
    src.add_argument("--input-dir")
    src.add_argument("--reddit-monthly-dir")

    # Month selection (SLURM array).
    sh = parser.add_argument_group("month/shard")
    sh.add_argument("--shard-index", type=int, default=None,
                    help="Array index selecting the Nth month. Defaults to "
                    "$SLURM_ARRAY_TASK_ID.")
    sh.add_argument("--months", nargs="+", default=None,
                    help="Restrict to these month keys (e.g. RC_2015-01).")
    sh.add_argument("--max-months", type=int, default=None,
                    help="Process only the first N months (test runs).")
    sh.add_argument("--shard-dir", required=True,
                    help="Directory to write this month's shard into.")

    # Scope.
    sc = parser.add_argument_group("scope")
    sc.add_argument("--subreddit-file", default=None,
                    help="Must match the frequency matrix's scope.")
    sc.add_argument("--start-year", type=int, default=2012)
    sc.add_argument("--end-year", type=int, default=2026)

    # Frequency matrix / word sources.
    fm = parser.add_argument_group("frequency matrix")
    fm.add_argument("--freq-dir", required=True,
                    help="Directory with freq_words.csv and freq_meta.json.")
    fm.add_argument("--freq-stem", default="freq")
    fm.add_argument(
        "--min-freq", type=int, default=2,
        help="Pre/at-scoring cost/noise floor, applied with a CUMULATIVE "
        "CAUSAL (real-time) rule: a word is scored in a half-year only once it "
        "has reached this count in that half-year or an earlier one, and from "
        "then on. A future spike never makes it eligible in an earlier "
        "half-year (no look-ahead). The frequency matrix is kept unfiltered; "
        "this chooses the scored subset at runtime. 1 = score every column "
        "from its first half-year (no floor). This is NOT the reliability gate "
        "— apply a frequency/min-contexts cutoff downstream when computing "
        "hit rates.",
    )

    # DEA list.
    parser.add_argument("--dea-file", required=True)

    # Model.
    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-name", default="bert-base-uncased")
    mdl.add_argument("--batch-size", type=int, default=64)
    mdl.add_argument("--max-length", type=int, default=512)
    mdl.add_argument("--fp16", action="store_true",
                     help="Load the model in half precision for the forward pass.")

    # Context length rule.
    ctx = parser.add_argument_group("context")
    ctx.add_argument("--min-context-words", type=int,
                     default=DEFAULT_MIN_CONTEXT_WORDS)
    ctx.add_argument("--max-side-words", type=int,
                     default=DEFAULT_MAX_SIDE_WORDS)

    args = parser.parse_args()

    if args.shard_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        args.shard_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.shard_index is None:
        parser.error("--shard-index is required (or set SLURM_ARRAY_TASK_ID).")

    all_paths = resolve_inputs(
        input=args.input, input_dir=args.input_dir,
        reddit_monthly_dir=args.reddit_monthly_dir,
    )
    if not all_paths:
        parser.error("Provide --input, --input-dir, or --reddit-monthly-dir.")

    month_key, files = select_month_key(
        all_paths, args.shard_index, months=args.months,
        max_months=args.max_months,
    )
    if month_key is None:
        logger.info("Task %d has no month assigned; nothing to do.",
                    args.shard_index)
        return
    logger.info("Month task %d: %s (%d file(s))",
                args.shard_index, month_key, len(files))

    # Scope + scope check against the frequency matrix.
    allowed = load_subreddits(args.subreddit_file)
    freq_meta = load_freq_meta(args.freq_dir, args.freq_stem)
    verify_scope(freq_meta, allowed, args.subreddit_file)

    target_words, eligible_from, _half_years, target_stats = load_target_words(
        args.freq_dir, args.freq_stem, min_freq=args.min_freq,
    )
    dea_terms = load_dea_terms(args.dea_file)

    # Model.
    model, tokenizer, device = load_model(args.model_name, fp16=args.fp16)

    dea_ids, _dea_kept, dea_dropped = resolve_dea_token_ids(dea_terms, tokenizer)
    exclude_ids = build_exclusion_ids(tokenizer)

    # Stream the month and score.
    drop_counts = defaultdict(int)
    stream = stream_corpus(
        files, args.start_year, args.end_year, allowed, drop_counts,
    )
    cells, dropped, n_contexts = process_month(
        stream, target_words, eligible_from, dea_ids, exclude_ids,
        model, tokenizer, device, args,
    )

    logger.info(
        "Month %s: %d contexts scored, %d cells, dropped=%s",
        month_key, n_contexts, len(cells), dict(dropped),
    )

    scope = {
        "mode": "subreddits" if allowed is not None else "all",
        "file": args.subreddit_file,
        "hash": subreddit_hash(allowed),
    }
    dea_meta = {
        "file": os.path.abspath(args.dea_file),
        "n_single_token": len(dea_ids),
        "n_dropped": len(dea_dropped),
        "model_name": args.model_name,
    }

    write_shard(
        args.shard_dir, month_key, cells, dict(dropped),
        drop_counts, dea_meta, files, scope, target_stats,
    )


if __name__ == "__main__":
    main()
