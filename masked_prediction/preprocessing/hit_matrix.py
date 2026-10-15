"""
Build the hit matrix: how often a masked language model predicts a DEA drug
term in place of each target word, per half-year.

This is the per-month MAP step of the hit matrix pipeline described in
``hit_matrix_plan.md``. One SLURM array task scores one month of the corpus and
writes a shard; ``merge_hit_matrix.py`` sums the shards into the final summary
table and matrices (aligned to the frequency matrix).

What one task does, in a single pass over its month (plan section 5):

  * **All target words** (the columns of the frequency matrix): maintain
    running per (half_year, word) counts — ``n_scored``, ``hits_k`` for every k
    in ``KS``, ``min_rank``, ``sum(log(best_rank))``, ``sum(soft_score)``.
  * **Pilot words only** (from ``pilot_words.json``): additionally append the
    per-context row ``(word, half_year, context_id, best_rank, soft_score)`` so
    the control set supports finer analysis (bootstrap CIs, other k, median).

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
        --pilot-file pilot_words.json \\
        --shard-dir hits_drug_subreddits/shards
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Local shared modules (same directory). sys.path is adjusted before these
# imports, so they intentionally do not sit at the top of the file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_stream import (
    iter_token_spans,
    resolve_inputs,
    select_month_key,
    stream_corpus,
)
from subreddit_scope import load_subreddits, subreddit_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# k values stored for every word (plan section 5). Chosen generously so k can
# be picked after the run without rescoring; 50 is the recall-oriented cap.
KS = [5, 10, 15, 20, 50]

SHARD_SUFFIX = ".part.json"

# Sentence context length bounds (plan section 3b), in whitespace words.
DEFAULT_MIN_CONTEXT_WORDS = 7
DEFAULT_MAX_SIDE_WORDS = 32  # words kept on each side of the target when truncating


# ============================================================================
# LOADERS
# ============================================================================


def load_target_words(freq_dir, stem="freq"):
    """Target words = the frequency matrix columns (``<stem>_words.csv``)."""
    import pandas as pd

    words_path = os.path.join(freq_dir, f"{stem}_words.csv")
    if not os.path.exists(words_path):
        raise FileNotFoundError(
            f"Target word list not found: {words_path}. Run freq_matrix.py / "
            f"merge_freq_matrix.py first."
        )
    words = pd.read_csv(words_path)["word"].astype(str).tolist()
    logger.info("Loaded %d target words from %s", len(words), words_path)
    return set(words)


def load_freq_meta(freq_dir, stem="freq"):
    """Load the frequency matrix metadata (``<stem>_meta.json``)."""
    meta_path = os.path.join(freq_dir, f"{stem}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Frequency metadata not found: {meta_path}. Needed for the scope "
            f"check (plan section 2b/8)."
        )
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def load_dea_terms(dea_file):
    """Load DEA terms (one per line, lowercased); blanks and ``#`` ignored."""
    terms = set()
    with open(dea_file, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            terms.add(s.lower())
    logger.info("Loaded %d raw DEA terms from %s", len(terms), dea_file)
    return terms


def load_pilot_words(pilot_file):
    """
    Load the flat pilot word set from ``pilot_words.json`` (pilot_creation.py).

    Prefers the ``pilot_words`` convenience field; falls back to the union of
    ``pilot_positives`` + ``pilot_negatives`` (+ ``hard_negatives``) if absent.
    Returns an empty set (with a warning) if no pilot file is given, so a run
    without a control set still aggregates all words.
    """
    if not pilot_file:
        logger.warning(
            "No --pilot-file given; no per-context pilot rows will be written."
        )
        return set()
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
# DEA TOKEN IDS (single-token terms only, as the BERT tokenizer sees them)
# ============================================================================


def resolve_dea_token_ids(dea_terms, tokenizer):
    """
    Map DEA terms to single BERT vocab ids, keeping only single-token terms.

    A term is kept iff the tokenizer encodes it (no special tokens) as exactly
    one wordpiece. Multi-word / multi-wordpiece terms (e.g. "bath salts",
    "oxycodone") are dropped and logged, per the plan's "single-token terms
    only" rule (sections 2 and 4).

    Returns ``(dea_ids, kept_terms, dropped_terms)`` where ``dea_ids`` is a
    sorted list of vocab ids and ``kept_terms`` maps id -> term.
    """
    kept = {}
    dropped = []
    for term in sorted(dea_terms):
        ids = tokenizer.encode(term, add_special_tokens=False)
        # Also try a leading space variant for tokenizers that distinguish it;
        # BERT WordPiece does not, but this keeps the rule tokenizer-agnostic.
        if len(ids) != 1:
            ids_sp = tokenizer.encode(" " + term, add_special_tokens=False)
            if len(ids_sp) == 1:
                ids = ids_sp
        if len(ids) == 1:
            kept[ids[0]] = term
        else:
            dropped.append(term)

    dea_ids = sorted(kept)
    logger.info(
        "DEA terms: %d single-token kept, %d dropped (multi-token). "
        "Dropped sample: %s",
        len(dea_ids),
        len(dropped),
        ", ".join(dropped[:10]) + ("..." if len(dropped) > 10 else ""),
    )
    if not dea_ids:
        raise ValueError(
            "No single-token DEA terms survived. Check the DEA list and the "
            "tokenizer (plan section 2)."
        )
    return dea_ids, kept, dropped


def build_exclusion_ids(tokenizer):
    """
    Vocab ids to exclude from ranking for every context (plan section 4.3):
    subword continuation pieces, punctuation/symbol-only tokens, and special
    tokens. The target word is excluded per-context separately.
    """
    exclude = set(tokenizer.all_special_ids)
    vocab = tokenizer.get_vocab()  # token string -> id
    for tok, tid in vocab.items():
        # WordPiece continuation pieces ("##ing") are never a standalone word.
        if tok.startswith("##"):
            exclude.add(tid)
            continue
        # Punctuation / symbol-only tokens.
        core = tok.removeprefix("##")
        if core and not any(ch.isalnum() for ch in core):
            exclude.add(tid)
    return exclude


# ============================================================================
# CONTEXT CONSTRUCTION (plan section 3b)
# ============================================================================


def _split_sentences(text):
    """
    Lightweight sentence split with char offsets: returns ``[(start, end), ...]``.

    No new dependency: split on ``.!?`` runs followed by whitespace. Reddit text
    is messy, so this is deliberately simple; the min-extend / max-truncate
    rules below handle the fallout (run-ons, missing punctuation).
    """
    spans = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        if text[i] in ".!?":
            j = i + 1
            while j < n and text[j] in ".!?":
                j += 1
            if j >= n or text[j].isspace():
                spans.append((start, j))
                while j < n and text[j].isspace():
                    j += 1
                start = j
                i = j
                continue
        i += 1
    if start < n:
        spans.append((start, n))
    return spans


def build_context(text, occ_start, occ_end, mask_token,
                  min_words, max_side_words):
    """
    Build the masked context for one occurrence at ``[occ_start, occ_end)``.

    Returns the context string with that single occurrence replaced by
    ``mask_token`` and surrounding text preserved, or ``None`` (with a reason)
    if the occurrence cannot be used.

    Steps (plan section 3b):
      1. Take the sentence containing the occurrence.
      2. If it is shorter than ``min_words``, extend outward (whole text) until
         the minimum is met.
      3. If longer than ``2 * max_side_words`` around the target, truncate to a
         window of ``max_side_words`` words on each side of the occurrence.
      4. Replace exactly the target occurrence with a single mask token; other
         occurrences in the window stay visible (one mask per occurrence).

    Returns ``(context_str, None)`` or ``(None, reason)``.
    """
    sentences = _split_sentences(text)
    sent_start, sent_end = 0, len(text)
    for s, e in sentences:
        if s <= occ_start < e:
            sent_start, sent_end = s, e
            break

    # Count words in the sentence; extend to the full text if too short.
    sent = text[sent_start:sent_end]
    if len(sent.split()) < min_words:
        sent_start, sent_end = 0, len(text)

    left = text[sent_start:occ_start]
    right = text[occ_end:sent_end]

    # Truncate overly long contexts to a word window around the target.
    left_words = left.split()
    right_words = right.split()
    if len(left_words) > max_side_words:
        left = " ".join(left_words[-max_side_words:])
    if len(right_words) > max_side_words:
        right = " ".join(right_words[:max_side_words])

    context = f"{left} {mask_token} {right}".strip()
    context = " ".join(context.split())  # collapse whitespace
    if mask_token not in context:
        return None, "mask_lost"
    return context, None


# ============================================================================
# SCORING (plan section 4)
# ============================================================================


def score_batch(contexts, model, tokenizer, device, dea_ids_t, exclude_ids_t,
                target_ids_per_ctx, mask_id, max_length=512):
    """
    Score a batch of masked contexts.

    Returns ``(best_rank, soft_score)`` as numpy arrays of shape ``[B]``.

    best_rank: rank (1 = top) of the highest-ranked DEA term over the full
    vocab, after excluding the target word and junk tokens.
    soft_score: total softmax probability on DEA terms at the mask position.
    """
    import torch

    enc = tokenizer(
        contexts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [B, T, V]

    # Locate the (single) mask position in each row.
    mask_pos = (input_ids == mask_id).float().argmax(dim=1)  # [B]
    b_idx = torch.arange(input_ids.size(0), device=device)
    mask_logits = logits[b_idx, mask_pos].float()  # [B, V]

    # Exclusions: junk tokens for all, then the per-context target word id(s).
    mask_logits[:, exclude_ids_t] = float("-inf")
    for i, tids in enumerate(target_ids_per_ctx):
        if tids:
            mask_logits[i, tids] = float("-inf")

    log_probs = torch.log_softmax(mask_logits, dim=-1)
    soft_score = log_probs[:, dea_ids_t].exp().sum(-1)  # [B]

    # Best DEA logit per row, then rank over the full vocab (strict >, so ties
    # share a rank). A target that is itself a DEA term was excluded above, so
    # it cannot produce a trivial hit.
    best_dea_logit = mask_logits[:, dea_ids_t].max(dim=-1).values  # [B]
    best_rank = (mask_logits > best_dea_logit[:, None]).sum(-1) + 1  # [B]

    return (
        best_rank.to("cpu").numpy(),
        soft_score.to("cpu").numpy().astype(np.float64),
    )


# ============================================================================
# AGGREGATION
# ============================================================================


def new_cell():
    cell = {
        "n_scored": 0,
        "min_rank": None,
        "log_rank_sum": 0.0,
        "soft_sum": 0.0,
    }
    for k in KS:
        cell[f"hits_{k}"] = 0
    return cell


def update_cell(cell, best_rank, soft_score):
    cell["n_scored"] += 1
    r = int(best_rank)
    cell["min_rank"] = r if cell["min_rank"] is None else min(cell["min_rank"], r)
    cell["log_rank_sum"] += math.log(r)
    cell["soft_sum"] += float(soft_score)
    for k in KS:
        if r <= k:
            cell[f"hits_{k}"] += 1


# ============================================================================
# SHARD IO
# ============================================================================


def write_shard(shard_dir, shard_name, cells, pilot_rows, dropped, drop_counts,
                dea_meta, files, scope):
    """Write one month's partial hit counts + pilot rows as a JSON shard."""
    os.makedirs(shard_dir, exist_ok=True)
    final_path = os.path.join(shard_dir, f"{shard_name}{SHARD_SUFFIX}")
    tmp_path = final_path + ".tmp"

    payload = {
        "shard_name": shard_name,
        "ks": KS,
        "scope": scope,
        "dea": dea_meta,
        # cells keyed "half_year\tword" so JSON stays flat and mergeable.
        "cells": {f"{hy}\t{w}": cell for (hy, w), cell in cells.items()},
        "pilot_rows": pilot_rows,  # list of [word, half_year, context_id, best_rank, soft_score]
        "dropped_contexts": dropped,
        "drop_counts": dict(drop_counts),
        "files": list(files),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, final_path)
    logger.info(
        "Wrote shard %s (%d cells, %d pilot rows)",
        final_path, len(cells), len(pilot_rows),
    )
    return final_path


# ============================================================================
# SCOPE CHECK
# ============================================================================


def verify_scope(freq_meta, allowed, subreddit_file):
    """
    Abort if this run's subreddit scope differs from the frequency matrix's.

    Compares mode (all vs. subreddits) and the subreddit list hash recorded in
    ``freq_meta.json`` (plan sections 2b and 8). Dividing hits by a denominator
    built on a different scope would silently corrupt every rate.
    """
    freq_scope = freq_meta.get("scope", {})
    this_mode = "subreddits" if allowed is not None else "all"
    this_hash = subreddit_hash(allowed)

    if freq_scope.get("mode") != this_mode or freq_scope.get("hash") != this_hash:
        raise SystemExit(
            "Scope mismatch with the frequency matrix.\n"
            f"  freq_meta: mode={freq_scope.get('mode')} "
            f"hash={freq_scope.get('hash')}\n"
            f"  this run:  mode={this_mode} hash={this_hash} "
            f"(subreddit-file={subreddit_file})\n"
            "The hit matrix divides by the frequency matrix, so both must use "
            "the same subreddit scope. Re-run with the matching --subreddit-file."
        )
    logger.info("Scope check passed (mode=%s).", this_mode)


# ============================================================================
# MAIN
# ============================================================================


def process_month(stream, target_words, pilot_words, dea_ids, exclude_ids,
                  model, tokenizer, device, args):
    """Single pass over a month's records: build contexts, score, aggregate."""
    import torch

    mask_token = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    dea_ids_t = torch.tensor(dea_ids, device=device, dtype=torch.long)
    exclude_ids_t = torch.tensor(sorted(exclude_ids), device=device,
                                 dtype=torch.long)

    cells = defaultdict(new_cell)
    pilot_rows = []
    dropped = defaultdict(int)
    context_counter = 0

    # Batch buffers.
    buf_ctx, buf_word, buf_hy, buf_cid, buf_target_ids = [], [], [], [], []

    def flush():
        nonlocal buf_ctx, buf_word, buf_hy, buf_cid, buf_target_ids
        if not buf_ctx:
            return
        best_rank, soft_score = score_batch(
            buf_ctx, model, tokenizer, device, dea_ids_t, exclude_ids_t,
            buf_target_ids, mask_id, max_length=args.max_length,
        )
        for w, hy, cid, r, s in zip(buf_word, buf_hy, buf_cid, best_rank, soft_score):
            update_cell(cells[(hy, w)], r, s)
            if w in pilot_words:
                pilot_rows.append([w, hy, cid, int(r), float(s)])
        buf_ctx, buf_word, buf_hy, buf_cid, buf_target_ids = [], [], [], [], []

    for half_year, text in stream:
        # One context per occurrence of each target word in the text.
        for tok, start, end in iter_token_spans(text):
            if tok not in target_words:
                continue
            context, reason = build_context(
                text, start, end, mask_token,
                args.min_context_words, args.max_side_words,
            )
            if context is None:
                dropped[reason] += 1
                continue

            # Pre-compute the target word's single-token ids to exclude for this
            # context (surface form + lowercased), so a target == DEA term can't
            # trivially hit.
            tgt_ids = tokenizer.encode(tok, add_special_tokens=False)
            tgt_ids = [i for i in tgt_ids] if len(tgt_ids) == 1 else []

            buf_ctx.append(context)
            buf_word.append(tok)
            buf_hy.append(half_year)
            buf_cid.append(context_counter)
            buf_target_ids.append(tgt_ids)
            context_counter += 1

            if len(buf_ctx) >= args.batch_size:
                flush()

    flush()
    return cells, pilot_rows, dropped, context_counter


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

    # DEA + pilot.
    parser.add_argument("--dea-file", required=True)
    parser.add_argument("--pilot-file", default=None,
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

    target_words = load_target_words(args.freq_dir, args.freq_stem)
    pilot_words = load_pilot_words(args.pilot_file)
    unknown_pilot = pilot_words - target_words
    if unknown_pilot:
        logger.warning(
            "%d pilot words are not in the target set and will never be "
            "scored: %s", len(unknown_pilot),
            ", ".join(sorted(unknown_pilot)[:10]),
        )
    dea_terms = load_dea_terms(args.dea_file)

    # Model.
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s", args.model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dtype = torch.float16 if (args.fp16 and device.type == "cuda") else None
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name, torch_dtype=dtype,
    ).to(device)
    model.eval()

    dea_ids, _dea_kept, dea_dropped = resolve_dea_token_ids(dea_terms, tokenizer)
    exclude_ids = build_exclusion_ids(tokenizer)

    # Stream the month and score.
    drop_counts = defaultdict(int)
    stream = stream_corpus(
        files, args.start_year, args.end_year, allowed, drop_counts,
    )
    cells, pilot_rows, dropped, n_contexts = process_month(
        stream, target_words, pilot_words, dea_ids, exclude_ids,
        model, tokenizer, device, args,
    )

    logger.info(
        "Month %s: %d contexts scored, %d cells, %d pilot rows, dropped=%s",
        month_key, n_contexts, len(cells), len(pilot_rows), dict(dropped),
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
        args.shard_dir, month_key, cells, pilot_rows, dict(dropped),
        drop_counts, dea_meta, files, scope,
    )


if __name__ == "__main__":
    main()
