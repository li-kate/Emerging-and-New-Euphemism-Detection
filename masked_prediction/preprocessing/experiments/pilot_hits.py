"""
Pilot experiment (map step): per-context DEA-substitution scores for the
pilot/control words only.

This used to ride along on the matrix pass in ``hit_matrix.py``. It is now its
own pipeline so a plain matrix rerun never touches it (see
``hit_matrix_plan.md`` section 5, "pilot words only"). It scores exactly the
same way as the matrix — same context construction, same ``best_rank`` /
``soft_score`` — via the shared ``hit_matrix_core``, so pilot numbers stay
comparable to the matrix cells.

The pilot set is small (a few dozen control words from ``pilot_words.json``),
so restricting the target set to those words makes a dedicated corpus pass
cheap: the vast majority of occurrences are skipped before any BERT call.

What one task does, in a single pass over its month: for every occurrence of a
pilot word, mask it, run BERT, and append the per-context row
``(word, half_year, context_id, best_rank, soft_score)``. ``merge_pilot_hits.py``
concatenates the rows across months into ``<stem>_pilot.csv``. These per-context
rows support finer analysis on the control set (bootstrap CIs, other k, median)
and choosing k (plan section 6).

The pilot words are still gated by the matrix's cumulative-causal ``--min-freq``
rule, and the scope is still checked against the frequency matrix, so a pilot
word is scored in exactly the half-years the matrix would have scored it.

Example (one month, local test):

    python pilot_hits.py \\
        --shard-index 0 \\
        --reddit-monthly-dir /data/reddit \\
        --subreddit-file drug_subreddit_list.txt \\
        --freq-dir freq_drug_subreddits \\
        --dea-file ../../data_collection/word_lists/drug_synonyms.txt \\
        --pilot-file pilot_words.json \\
        --shard-dir pilot_drug_subreddits/shards
"""

import argparse
import json
import logging
import os
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

# Pilot shard suffix — distinct from the matrix's ``.part.json`` so pilot and
# matrix shards can share a directory without colliding, and each merge only
# picks up its own shards.
PILOT_SHARD_SUFFIX = ".pilot.json"


# ============================================================================
# LOADERS
# ============================================================================


def load_pilot_words(pilot_file):
    """
    Load the flat pilot word set from ``pilot_words.json`` (pilot_creation.py).

    Prefers the ``pilot_words`` convenience field; falls back to the union of
    ``pilot_positives`` + ``pilot_negatives`` (+ ``hard_negatives``) if absent.
    """
    if not pilot_file:
        raise SystemExit(
            "pilot_hits.py needs --pilot-file (pilot_words.json from "
            "pilot_creation.py); there is nothing to score without it."
        )
    with open(pilot_file, encoding="utf-8") as f:
        cfg = json.load(f)
    if "pilot_words" in cfg:
        words = set(cfg["pilot_words"])
    else:
        pos = [p["word"] if isinstance(p, dict) else p
               for p in cfg.get("pilot_positives", [])]
        words = set(pos) | set(cfg.get("pilot_negatives", [])) \
            | set(cfg.get("hard_negatives", []))
    logger.info("Loaded %d pilot words from %s", len(words), pilot_file)
    return words


# ============================================================================
# SHARD IO
# ============================================================================


def write_shard(shard_dir, shard_name, pilot_rows, dropped, drop_counts,
                dea_meta, files, scope, target_stats=None):
    """Write one month's per-context pilot rows as a JSON shard."""
    os.makedirs(shard_dir, exist_ok=True)
    final_path = os.path.join(shard_dir, f"{shard_name}{PILOT_SHARD_SUFFIX}")
    tmp_path = final_path + ".tmp"

    payload = {
        "shard_name": shard_name,
        "scope": scope,
        "dea": dea_meta,
        "target_selection": target_stats or {},
        # list of [word, half_year, context_id, best_rank, soft_score]
        "pilot_rows": pilot_rows,
        "dropped_contexts": dropped,
        "drop_counts": dict(drop_counts),
        "files": list(files),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, final_path)
    logger.info("Wrote pilot shard %s (%d rows)", final_path, len(pilot_rows))
    return final_path


# ============================================================================
# MAIN
# ============================================================================


def process_month(stream, pilot_words, eligible_from, dea_ids, exclude_ids,
                  model, tokenizer, device, args):
    """Single pass over a month: build pilot contexts, score, collect rows.

    The target set handed to ``iter_scored_occurrences`` is the pilot set, so
    only pilot occurrences are built and scored; everything else is skipped
    before any BERT call.
    """
    dea_ids_t, exclude_ids_t = build_scoring_tensors(dea_ids, exclude_ids, device)
    mask_id = tokenizer.mask_token_id

    pilot_rows = []
    dropped = defaultdict(int)
    context_counter = 0

    buf_ctx, buf_word, buf_hy, buf_cid, buf_target_ids = [], [], [], [], []

    def flush():
        nonlocal buf_ctx, buf_word, buf_hy, buf_cid, buf_target_ids
        if not buf_ctx:
            return
        best_rank, soft_score = score_batch(
            buf_ctx, model, tokenizer, device, dea_ids_t, exclude_ids_t,
            buf_target_ids, mask_id, max_length=args.max_length,
        )
        for w, hy, cid, r, s in zip(
            buf_word, buf_hy, buf_cid, best_rank, soft_score
        ):
            pilot_rows.append([w, hy, cid, int(r), float(s)])
        buf_ctx, buf_word, buf_hy, buf_cid, buf_target_ids = [], [], [], [], []

    occurrences = iter_scored_occurrences(
        stream, pilot_words, eligible_from, tokenizer,
        args.min_context_words, args.max_side_words, dropped,
    )
    for word, half_year, context, tgt_ids in occurrences:
        buf_ctx.append(context)
        buf_word.append(word)
        buf_hy.append(half_year)
        buf_cid.append(context_counter)
        buf_target_ids.append(tgt_ids)
        context_counter += 1
        if len(buf_ctx) >= args.batch_size:
            flush()

    flush()
    return pilot_rows, dropped, context_counter


def main():
    parser = argparse.ArgumentParser(
        description="Pilot per-context scores, one month per task "
        "(hit_matrix_plan.md 5/6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
                    help="Directory to write this month's pilot shard into.")

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
        "build, so a pilot word is scored in exactly the half-years the matrix "
        "scored it. 1 = score every pilot word from its first half-year.",
    )

    # DEA + pilot.
    parser.add_argument("--dea-file", required=True)
    parser.add_argument("--pilot-file", required=True,
                        help="pilot_words.json from pilot_creation.py.")

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

    # Target set = the pilot words, gated by the SAME cumulative-causal
    # eligibility as the matrix (so half-years line up), then intersected with
    # the pilot list.
    all_targets, eligible_from, _half_years, target_stats = load_target_words(
        args.freq_dir, args.freq_stem, min_freq=args.min_freq,
    )
    pilot_words = load_pilot_words(args.pilot_file)
    unknown_pilot = pilot_words - all_targets
    if unknown_pilot:
        logger.warning(
            "%d pilot words are not eligible in the matrix and will never be "
            "scored: %s", len(unknown_pilot),
            ", ".join(sorted(unknown_pilot)[:10]),
        )
    scored_pilot = pilot_words & all_targets
    # Restrict eligible_from to the scored pilot words (iter_scored_occurrences
    # looks up eligible_from[tok] for every target it sees).
    eligible_from = {w: eligible_from[w] for w in scored_pilot}
    logger.info("Scoring %d eligible pilot word(s).", len(scored_pilot))

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
    pilot_rows, dropped, n_contexts = process_month(
        stream, scored_pilot, eligible_from, dea_ids, exclude_ids,
        model, tokenizer, device, args,
    )

    logger.info(
        "Month %s: %d pilot contexts scored, dropped=%s",
        month_key, n_contexts, dict(dropped),
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
        args.shard_dir, month_key, pilot_rows, dict(dropped),
        drop_counts, dea_meta, files, scope, target_stats,
    )


if __name__ == "__main__":
    main()
