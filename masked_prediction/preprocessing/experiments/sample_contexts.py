"""
Context-sample experiment (map step): keep a random sample of ONE word's full
contexts per half-year, for manual spot-checking (``hit_matrix_plan.md`` 8).

This used to ride along on the matrix pass in ``hit_matrix.py`` behind
``--sample-word``. It is now its own pipeline so a plain matrix rerun never
touches it. The matrix run stores only two scalars per context and never the
text, so there is normally nothing to inspect by hand; this script collects the
full context string plus its ``best_rank`` / ``soft_score`` for the chosen word
so a few real examples per half-year can be eyeballed (the plan's "spot-check a
few contexts by hand" / "print the text, the top predictions, and best_rank"
check). Diagnostic only: it does not affect the matrix cells or the pilot rows.

It scores the same way as the matrix (shared ``hit_matrix_core``), restricts
the target set to the single chosen word, and gates it with the matrix's
cumulative-causal ``--min-freq`` rule, so the sampled contexts are exactly the
ones the matrix would have scored for that word.

``merge_sample_contexts.py`` unions the per-month reservoirs into one sample per
half-year.

Example (one month, local test):

    python sample_contexts.py \\
        --shard-index 0 \\
        --word molly \\
        --reddit-monthly-dir /data/reddit \\
        --subreddit-file drug_subreddit_list.txt \\
        --freq-dir freq_drug_subreddits \\
        --dea-file ../../data_collection/word_lists/drug_synonyms.txt \\
        --shard-dir sample_drug_subreddits/shards
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_stream import (
    resolve_inputs,
    select_month_key,
    stream_corpus,
)
from hit_matrix_core import (
    DEFAULT_MAX_SIDE_WORDS,
    DEFAULT_MIN_CONTEXT_WORDS,
    build_scoring_tensors,
    build_exclusion_ids,
    iter_scored_occurrences,
    load_dea_terms,
    load_freq_meta,
    load_model,
    load_target_words,
    resolve_dea_token_ids,
    score_batch,
    verify_scope,
)
from subreddit_scope import load_subreddits, subreddit_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Sample shard suffix — distinct from the matrix (``.part.json``) and pilot
# (``.pilot.json``) suffixes so all three can share a directory.
SAMPLE_SHARD_SUFFIX = ".sample.json"


# ============================================================================
# PER-HALF-YEAR CONTEXT SAMPLING (manual spot-check, plan section 8)
# ============================================================================


class ReservoirSampler:
    """
    Keep a uniform random sample of up to ``k`` contexts per half-year for a
    single target word, in one streaming pass (reservoir sampling, Algorithm R).

    Uniformity is per half-year: every occurrence of the word in a half-year has
    the same chance of ending up in that half-year's reservoir, regardless of
    how many occurrences that half-year ends up having.
    """

    def __init__(self, k, seed=0):
        self.k = k
        # Each half-year gets its own reservoir and its own seen-counter.
        self._pools = defaultdict(list)
        self._seen = defaultdict(int)
        self._rng = random.Random(seed)

    def offer(self, half_year, record):
        """Offer one scored context to the ``half_year`` reservoir."""
        if self.k <= 0:
            return
        self._seen[half_year] += 1
        n = self._seen[half_year]
        pool = self._pools[half_year]
        if len(pool) < self.k:
            pool.append(record)
        else:
            # Replace an existing entry with probability k/n.
            j = self._rng.randint(0, n - 1)
            if j < self.k:
                pool[j] = record

    def as_dict(self):
        """Return ``{half_year: [records...]}`` plus the per-half-year totals."""
        return {
            "k": self.k,
            "samples": {hy: list(pool) for hy, pool in self._pools.items()},
            "n_seen": dict(self._seen),
        }


# ============================================================================
# SHARD IO
# ============================================================================


def write_shard(shard_dir, shard_name, word_sample, dropped, drop_counts,
                dea_meta, files, scope, target_stats=None):
    """Write one month's per-half-year word sample as a JSON shard."""
    os.makedirs(shard_dir, exist_ok=True)
    final_path = os.path.join(shard_dir, f"{shard_name}{SAMPLE_SHARD_SUFFIX}")
    tmp_path = final_path + ".tmp"

    payload = {
        "shard_name": shard_name,
        "scope": scope,
        "dea": dea_meta,
        "target_selection": target_stats or {},
        # {"word", "k", "samples": {hy: [records]}, "n_seen": {hy: count}}
        "word_sample": word_sample,
        "dropped_contexts": dropped,
        "drop_counts": dict(drop_counts),
        "files": list(files),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, final_path)
    n_kept = sum(len(v) for v in word_sample.get("samples", {}).values())
    logger.info("Wrote sample shard %s (%d contexts)", final_path, n_kept)
    return final_path


# ============================================================================
# MAIN
# ============================================================================


def process_month(stream, word, eligible_from, dea_ids, exclude_ids,
                  model, tokenizer, device, args, sampler):
    """Single pass over a month: score the chosen word's contexts, sample them.

    The target set handed to ``iter_scored_occurrences`` is the single chosen
    word, so only its occurrences are built and scored; the full context string
    is kept (unlike the matrix) and offered to the per-half-year reservoir.
    """
    dea_ids_t, exclude_ids_t = build_scoring_tensors(dea_ids, exclude_ids, device)
    mask_id = tokenizer.mask_token_id

    dropped = defaultdict(int)
    context_counter = 0

    buf_ctx, buf_hy, buf_cid, buf_target_ids = [], [], [], []

    def flush():
        nonlocal buf_ctx, buf_hy, buf_cid, buf_target_ids
        if not buf_ctx:
            return
        best_rank, soft_score = score_batch(
            buf_ctx, model, tokenizer, device, dea_ids_t, exclude_ids_t,
            buf_target_ids, mask_id, max_length=args.max_length,
        )
        for ctx, hy, cid, r, s in zip(
            buf_ctx, buf_hy, buf_cid, best_rank, soft_score
        ):
            sampler.offer(hy, {
                "context_id": int(cid),
                "best_rank": int(r),
                "soft_score": float(s),
                "context": ctx,
            })
        buf_ctx, buf_hy, buf_cid, buf_target_ids = [], [], [], []

    occurrences = iter_scored_occurrences(
        stream, {word}, eligible_from, tokenizer,
        args.min_context_words, args.max_side_words, dropped,
    )
    for _word, half_year, context, tgt_ids in occurrences:
        buf_ctx.append(context)
        buf_hy.append(half_year)
        buf_cid.append(context_counter)
        buf_target_ids.append(tgt_ids)
        context_counter += 1
        if len(buf_ctx) >= args.batch_size:
            flush()

    flush()
    return dropped, context_counter


def main():
    parser = argparse.ArgumentParser(
        description="Sample one word's contexts per half-year for manual "
        "spot-checking (hit_matrix_plan.md 8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--word", required=True,
        help="Target word whose contexts are sampled (lowercased to match the "
        "occurrence rule).",
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
                    help="Directory to write this month's sample shard into.")

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
        help="Cumulative-causal cost/noise floor, identical to the matrix "
        "build, so the word is sampled in exactly the half-years the matrix "
        "scored it.",
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

    # Sampling.
    smp = parser.add_argument_group("sampling")
    smp.add_argument(
        "--sample-size", type=int, default=1000,
        help="Max contexts to keep per half-year. If a half-year has fewer "
        "occurrences, all of them are kept.",
    )
    smp.add_argument(
        "--sample-seed", type=int, default=0,
        help="Seed for the reservoir sampler (reproducible).",
    )

    args = parser.parse_args()

    if args.shard_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        args.shard_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.shard_index is None:
        parser.error("--shard-index is required (or set SLURM_ARRAY_TASK_ID).")

    word = args.word.lower()

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

    # Eligibility for the one word, from the SAME cumulative-causal rule as the
    # matrix (so the sampled half-years line up with the scored ones).
    all_targets, eligible_from, _half_years, target_stats = load_target_words(
        args.freq_dir, args.freq_stem, min_freq=args.min_freq,
    )
    if word not in all_targets:
        logger.warning(
            "--word %r is not eligible in the matrix (below --min-freq %s, or "
            "absent from the frequency matrix); no contexts will be sampled.",
            word, args.min_freq,
        )
        eligible_from = {}
    else:
        eligible_from = {word: eligible_from[word]}

    dea_terms = load_dea_terms(args.dea_file)

    # Model.
    model, tokenizer, device = load_model(args.model_name, fp16=args.fp16)

    dea_ids, _dea_kept, dea_dropped = resolve_dea_token_ids(dea_terms, tokenizer)
    exclude_ids = build_exclusion_ids(tokenizer)

    sampler = ReservoirSampler(args.sample_size, seed=args.sample_seed)
    logger.info(
        "Sampling up to %d contexts per half-year for word %r (seed=%d).",
        args.sample_size, word, args.sample_seed,
    )

    # Stream the month and score.
    drop_counts = defaultdict(int)
    stream = stream_corpus(
        files, args.start_year, args.end_year, allowed, drop_counts,
    )
    dropped, n_contexts = process_month(
        stream, word, eligible_from, dea_ids, exclude_ids,
        model, tokenizer, device, args, sampler,
    )

    word_sample = {"word": word, **sampler.as_dict()}
    n_kept = sum(len(v) for v in word_sample["samples"].values())
    logger.info(
        "Month %s: %d contexts scored, %d sampled across %d half-year(s), "
        "dropped=%s",
        month_key, n_contexts, n_kept, len(word_sample["samples"]),
        dict(dropped),
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
        args.shard_dir, month_key, word_sample, dict(dropped),
        drop_counts, dea_meta, files, scope, target_stats,
    )


if __name__ == "__main__":
    main()
