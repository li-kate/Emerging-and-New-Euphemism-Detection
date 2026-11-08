"""
Shared core for the hit-matrix pipeline: loaders, context construction,
scoring, cell aggregation, and the scope check.

This module is the single home for everything the hit-matrix **map** step and
its experiments need in common, so the matrix build and the experiments cannot
drift apart (they must score contexts identically, or their numbers are not
comparable).

Consumers:

  * ``hit_matrix.py``          — matrix-only map: aggregate every eligible word
                                 into per (half_year, word) cells.
  * ``merge_hit_matrix.py``    — reduces the matrix shards (imports ``KS`` and
                                 ``SHARD_SUFFIX``).
  * ``pilot_hits.py``          — experiment: per-context rows for pilot words.
  * ``sample_contexts.py``     — experiment: a random sample of one word's
                                 contexts for manual spot-checking.

What lives here (nothing experiment-specific):

  * ``KS``                     — the k values stored for every word.
  * ``SHARD_SUFFIX``           — matrix shard filename suffix.
  * loaders                    — ``load_target_words``, ``load_freq_meta``,
                                 ``load_dea_terms``.
  * tokenization -> ids        — ``resolve_dea_token_ids``,
                                 ``build_exclusion_ids``.
  * context construction       — ``_split_sentences``, ``build_context``.
  * scoring                    — ``score_batch`` (best_rank + soft_score).
  * cell aggregation           — ``new_cell``, ``update_cell``.
  * scope check                — ``verify_scope``.
  * occurrence iteration       — ``iter_scored_occurrences``, the one place the
                                 eligibility rule + context build are applied,
                                 so the matrix and the experiments select and
                                 build contexts the same way.

The per-context signal (``best_rank``, ``soft_score``) is the same for every
consumer; what differs is only how they *store* it (cells, pilot rows, a
sample), which is why that lives in the individual scripts and this does not.
"""

import logging
import math
import os
from collections import defaultdict

import numpy as np

from corpus_stream import iter_token_spans
from subreddit_scope import subreddit_hash

logger = logging.getLogger(__name__)

# k values stored for every word (plan section 5). Chosen generously so k can
# be picked after the run without rescoring; 50 is the recall-oriented cap.
KS = [5, 10, 15, 20, 50]

# Matrix (cells-only) shard filename suffix. The experiments use their own
# suffixes so their shards never collide with the matrix shards in a shard dir.
SHARD_SUFFIX = ".part.json"

# Sentence context length bounds (plan section 3b), in whitespace words.
DEFAULT_MIN_CONTEXT_WORDS = 7
DEFAULT_MAX_SIDE_WORDS = 32  # words kept on each side of the target when truncating


# ============================================================================
# LOADERS
# ============================================================================


def load_target_words(freq_dir, stem="freq", min_freq=1):
    """
    Target words to score = the frequency matrix columns, gated by a
    **cumulative, causal (real-time)** per-half-year frequency floor.

    The frequency matrix is the unfiltered archive of every surface form and
    its per-half-year counts. This is the pipeline's **cost/noise gate**: it
    decides which columns are worth a (expensive) BERT forward pass, before any
    scoring. It is applied here, at scoring time, so the floor is a runtime
    choice rather than baked into the matrix — the same unfiltered matrix can
    be scored at different floors without a rebuild.

    Eligibility rule (cumulative, "current half-year or before"): a word
    becomes eligible in the **first** half-year ``H*`` where its count reaches
    ``min_freq``, and stays eligible for every half-year ``>= H*``. This is the
    real-time / emerging-term reading: once a term has surfaced it keeps being
    watched, and a *future* frequency spike never makes it eligible in an
    *earlier* half-year (no look-ahead / leakage). A word that never reaches
    the floor in any half-year is dropped entirely (no cells, no GPU).

    The caller enforces this per occurrence by comparing the occurrence's
    half-year against the word's first-eligible half-year
    (``iter_scored_occurrences``); this function just computes that per-word
    scalar. ``min_freq <= 1`` makes every column eligible from its first
    half-year (equivalent to the unfiltered matrix, no floor).

    Returns ``(target_words, eligible_from, half_years, stats)``:
      * ``target_words``    - set of words with at least one eligible half-year.
      * ``eligible_from``   - dict ``word -> first-eligible half-year label``
                              (e.g. ``"2015H2"``); a word is scored in a
                              half-year ``hy`` iff ``hy >= eligible_from[word]``
                              (string comparison; ``YYYYH1``/``YYYYH2`` sorts
                              chronologically).
      * ``half_years``      - the ordered half-year axis (for reference/logging).
      * ``stats``           - the floor, rule, and kept/dropped counts.
    """
    import pandas as pd

    words_path = os.path.join(freq_dir, f"{stem}_words.csv")
    if not os.path.exists(words_path):
        raise FileNotFoundError(
            f"Target word list not found: {words_path}. Run freq_matrix.py / "
            f"merge_freq_matrix.py first."
        )
    words = pd.read_csv(words_path)["word"].astype(str).tolist()
    n_total = len(words)

    half_years_path = os.path.join(freq_dir, f"{stem}_half_years.csv")
    if not os.path.exists(half_years_path):
        raise FileNotFoundError(
            f"Half-year axis not found: {half_years_path}. Needed to order the "
            f"half-years for the cumulative eligibility rule."
        )
    # Chronological order. YYYYH1/YYYYH2 already sorts lexicographically, but
    # trust the file's row order (it matches the frequency matrix rows).
    half_years = pd.read_csv(half_years_path)["half_year"].astype(str).tolist()

    if min_freq is None or min_freq <= 1:
        # No floor: every word eligible from its first half-year, i.e. the
        # earliest axis label. (A word with no occurrences anywhere still has
        # no contexts, so this is harmless.)
        first_hy = half_years[0] if half_years else ""
        eligible_from = {w: first_hy for w in words}
        logger.info(
            "Loaded %d target words from %s (no frequency floor; min_freq=%s; "
            "all eligible from %s onward).",
            n_total, words_path, min_freq, first_hy,
        )
        return set(words), eligible_from, half_years, {
            "min_freq": 1 if (min_freq is None) else int(min_freq),
            "rule": "cumulative_causal",
            "n_total": n_total,
            "n_kept": n_total,
            "n_dropped": 0,
        }

    matrix_path = os.path.join(freq_dir, f"{stem}.npy")
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(
            f"Frequency matrix not found: {matrix_path}. Needed to apply "
            f"--min-freq {min_freq}. Build it with freq_matrix.py / "
            f"merge_freq_matrix.py, or run with --min-freq 1 to score every "
            f"column."
        )
    # mmap: the matrix can be ~5M columns. We need, per column, the FIRST row
    # (half-year) that reaches the floor.
    freq = np.load(matrix_path, mmap_mode="r")
    if freq.shape[1] != n_total:
        raise ValueError(
            f"Frequency matrix columns ({freq.shape[1]}) do not match "
            f"{stem}_words.csv ({n_total}). Rebuild so they align."
        )
    if freq.shape[0] != len(half_years):
        raise ValueError(
            f"Frequency matrix rows ({freq.shape[0]}) do not match "
            f"{stem}_half_years.csv ({len(half_years)}). Rebuild so they align."
        )

    # Cumulative causal keep: a word is eligible from the first half-year it
    # reaches the floor. ``reaches`` is a boolean [half_years x words] matrix;
    # argmax down each column gives the first True row (or 0 if none), and
    # ``col_any`` tells us whether that column had any True at all.
    reaches = np.asarray(freq) >= min_freq
    col_any = reaches.any(axis=0)
    first_row = reaches.argmax(axis=0)  # 0 when the column is all-False

    hy_arr = np.asarray(half_years, dtype=object)
    eligible_from = {}
    for w, keep, r in zip(words, col_any, first_row):
        if keep:
            eligible_from[w] = hy_arr[r]
    target_words = set(eligible_from)
    n_kept = len(target_words)

    logger.info(
        "Loaded %d target words from %s; kept %d (%.1f%%) at min_freq=%d "
        "(cumulative causal rule: eligible from the first half-year a word "
        "reaches the floor, onward), dropped %d never reaching the floor.",
        n_total, words_path, n_kept, 100.0 * n_kept / max(n_total, 1),
        min_freq, n_total - n_kept,
    )
    return target_words, eligible_from, half_years, {
        "min_freq": int(min_freq),
        "rule": "cumulative_causal",
        "n_total": n_total,
        "n_kept": n_kept,
        "n_dropped": n_total - n_kept,
    }


def load_freq_meta(freq_dir, stem="freq"):
    """Load the frequency matrix metadata (``<stem>_meta.json``)."""
    import json

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
# OCCURRENCE ITERATION (shared selection + context build)
# ============================================================================


def iter_scored_occurrences(stream, target_words, eligible_from, tokenizer,
                            min_context_words, max_side_words, dropped):
    """
    Yield ``(word, half_year, context_str, target_token_ids)`` for every target
    occurrence that should be scored, applying the cumulative-causal
    eligibility rule and the context-construction rule ONCE, so the matrix and
    the experiments select and build contexts identically.

    ``stream`` yields ``(half_year, text)`` pairs (``corpus_stream.stream_corpus``).
    ``eligible_from`` maps each target word to its first-eligible half-year
    label; an occurrence is skipped (counted under ``before_eligible`` in
    ``dropped``) when its half-year precedes that word's first-eligible
    half-year, so a word is never "looked at" in a half-year before it emerged.
    Context-construction failures are counted under their own reason in
    ``dropped``.

    ``target_token_ids`` is the target surface form's single-token id list
    (empty unless it is exactly one wordpiece), to be excluded per context so a
    target that is itself a DEA term cannot trivially hit.

    ``dropped`` is a dict updated in place with per-reason skip counts.
    """
    mask_token = tokenizer.mask_token
    for half_year, text in stream:
        # One context per occurrence of each target word in the text.
        for tok, start, end in iter_token_spans(text):
            if tok not in target_words:
                continue
            # Cumulative causal eligibility: skip occurrences in half-years
            # before the word first reached the frequency floor. YYYYH1/YYYYH2
            # labels sort chronologically, so a string compare is the
            # "at or before" test.
            if half_year < eligible_from[tok]:
                dropped["before_eligible"] += 1
                continue
            context, reason = build_context(
                text, start, end, mask_token,
                min_context_words, max_side_words,
            )
            if context is None:
                dropped[reason] += 1
                continue

            # Pre-compute the target word's single-token ids to exclude for this
            # context, so a target == DEA term can't trivially hit.
            tgt_ids = tokenizer.encode(tok, add_special_tokens=False)
            tgt_ids = list(tgt_ids) if len(tgt_ids) == 1 else []

            yield tok, half_year, context, tgt_ids


def build_scoring_tensors(dea_ids, exclude_ids, device):
    """Build the (dea_ids, exclude_ids) GPU tensors shared by every batch."""
    import torch

    dea_ids_t = torch.tensor(dea_ids, device=device, dtype=torch.long)
    exclude_ids_t = torch.tensor(sorted(exclude_ids), device=device,
                                 dtype=torch.long)
    return dea_ids_t, exclude_ids_t


def load_model(model_name, fp16=False):
    """
    Load the MLM + tokenizer onto the best available device.

    Returns ``(model, tokenizer, device)``. ``fp16`` loads the model in half
    precision for the forward pass (GPU only).
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if (fp16 and device.type == "cuda") else None
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, tokenizer, device
