"""
Run Zhu et al.'s euphemism *detection* stage one month at a time.

This is the per-month MAP step of a temporal version of the Zhu pipeline,
built the same way ``hit_matrix.py`` is: one SLURM array task scores a single
month of the corpus and writes a shard, and ``merge_zhu_detection.py`` sums the
shards into ranked candidate lists. Here, though, shards are summed **per
half-year** (not into one global result), so you get a separate ranked
euphemism-candidate list for each half-year and can watch candidates emerge
over time.

What the detection stage does (faithful to ``zhu_model/detection.py``):

  * For every sentence that contains one of the input keywords (the known
    target terms, e.g. the DEA drug list), mask that keyword with ``[MASK]``.
  * Run the Reddit-tuned BERT masked LM and take its top ``MLM_k`` predictions
    for the mask.
  * Keep only *informative* masked sentences: a sentence is discarded unless at
    least two of the input keywords themselves appear in the mask's top
    ``thres`` predictions (``filter_uninformative``). This is Zhu's noise gate.
  * For each surviving sentence, add every predicted token's probability to a
    running ``MLM_score`` for that token, after dropping punctuation, English
    stopwords, the input keywords, ``drug``/``drugs``, and ``##`` word-pieces.
  * Candidates ranked by total ``MLM_score`` are the euphemism candidates.

This script does **not import** ``zhu_model`` (its modules run heavy
module-level setup and pull in the identification stage). Instead it reuses the
exact same scoring and filtering logic, parameterized, so the numbers match a
single-corpus Zhu run restricted to one month. The model, vocab, ``MLM_k``,
``thres``, and the uninformative filter are all the knobs Zhu exposes.

Length filter (faithful to Zhu): Zhu's ``read_file.read_raw_text`` keeps only
source lines with ``len(line) <= 150`` CHARACTERS before masking, so it never
over-runs BERT's position limit. The same character cap is applied here via
``--max-chars`` (default 150): longer texts are skipped entirely (not cropped).
Set ``--max-chars 0`` to disable it.

Difference from Zhu's one-shot ``MLM``: there is no ``ms_limit`` random sample
and no global shuffle. Every eligible masked sentence in the month is scored,
and the per-token score sums are bucketed by the sentence's half-year so the
merge can total each half-year independently. (A cap is still available via
``--ms-limit-per-half-year`` if a half-year is too large to score in full.)

Scope safety mirrors ``hit_matrix.py``: the subreddit scope is applied in the
shared corpus reader, and the scope hash is recorded in every shard so the
merge can confirm all months agree.

Example (one month, local test):

    python zhu_detection_month.py \\
        --shard-index 0 \\
        --reddit-monthly-dir /data/reddit \\
        --subreddit-file ../../drug_lists/drug_subreddit_list.txt \\
        --answer-file ../zhu_model/data/euphemism_answer_drug.txt \\
        --model-dir ../BERT_model_reddit \\
        --vocab-file ../BERT_model_reddit/vocab.txt \\
        --shard-dir zhu_detect_drug_subreddits/shards
"""

import argparse
import json
import logging
import os
import string
import sys
from collections import defaultdict
from pathlib import Path

# Shared corpus helpers live in shared_utils/. Add both that directory and this
# script's own directory to sys.path before importing them, matching how
# hit_matrix.py locates corpus_stream / subreddit_scope. These imports
# intentionally sit below the path edit rather than at the top of the file.
_HERE = Path(__file__).resolve().parent
_SHARED = _HERE.parent.parent / "shared_utils"
for _p in (_HERE, _SHARED):
    sys.path.insert(0, str(_p))

from corpus_stream import (  # noqa: E402
    resolve_inputs,
    select_month_key,
    stream_corpus,
)
from subreddit_scope import load_subreddits, subreddit_hash  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SHARD_SUFFIX = ".part.json"

# BERT special tokens, same strings Zhu uses.
PAD, MASK, CLS, SEP = "[PAD]", "[MASK]", "[CLS]", "[SEP]"


# ============================================================================
# KEYWORDS
# ============================================================================


def load_keywords_from_answer_file(path):
    """
    Derive the input keywords EXACTLY as Zhu's ``read_file.py`` does, from a
    ``euphemism_answer_<category>.txt`` file.

    Each line is ``formal name: slang1; slang2; ...``. Zhu
    (``read_input_and_ground_truth``) builds ``euphemism_answer[slang] =
    [formal, ...]`` and then takes ``input_keywords = sorted(set(all formal
    names))`` — i.e. the formal target terms (the part before the ``:``),
    lowercased. This reproduces that result token-for-token so the masked
    sentences match a real Zhu detection run.

    Blank lines and ``#`` comments are ignored. Lines without a ``:`` are
    skipped (Zhu's parser assumes the ``key: values`` form).
    """
    formal_names = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                continue
            ans = s.split(":")[0].strip().lower()
            if ans:
                formal_names.add(ans)
    kws = sorted(formal_names)
    if not kws:
        raise ValueError(f"No input keywords derived from answer file {path}.")
    logger.info("Derived %d input keyword(s) from answer file %s (Zhu rule)",
                len(kws), path)
    return kws


def load_keywords_from_list(path):
    """
    Load input keywords from a plain newline-delimited list (one term per line,
    lowercased; blanks and ``#`` comments ignored).

    This is the non-Zhu fallback for running the detection stage against an
    arbitrary target-term list (e.g. ``drug_lists/dea_drug_terms.txt``) that is
    not in Zhu's ``key: values`` answer-file format.
    """
    kws = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip().lower()
            if not s or s.startswith("#"):
                continue
            if s not in seen:
                seen.add(s)
                kws.append(s)
    if not kws:
        raise ValueError(f"No keywords loaded from {path}.")
    logger.info("Loaded %d input keyword(s) from list %s", len(kws), path)
    return kws


# ============================================================================
# MODEL (Zhu uses pytorch_pretrained_bert's BertForMaskedLM)
# ============================================================================


def load_model(model_dir, vocab_file):
    """
    Load the Reddit-tuned BERT masked LM + tokenizer exactly as
    ``zhu_model/detection.py`` does, returning ``(model, tokenizer, device)``.

    Uses ``pytorch_pretrained_bert`` so the forward-pass interface
    (``model(token_idx, segment_idx, mask, masked_lm_labels=None)``) matches
    Zhu's ``single_MLM`` byte-for-byte.
    """
    import torch
    from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initialize BERT vocabulary from %s ...", vocab_file)
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    logger.info("Initialize BERT model from %s on %s ...", model_dir, device)
    model = BertForMaskedLM.from_pretrained(model_dir).to(device)
    model.eval()

    # Fail fast on a vocab/checkpoint mismatch. If the tokenizer's vocab is
    # larger than the model's token-embedding table, some token ids are out of
    # range and the GPU embedding lookup aborts with an opaque async assert
    # (indexSelectLargeIndex ... srcIndex < srcSelectDimSize) deep in a later
    # call. Checking here turns that into a clear, synchronous error.
    try:
        emb_rows = model.bert.embeddings.word_embeddings.weight.shape[0]
        vocab_rows = len(tokenizer.vocab)
        if vocab_rows > emb_rows:
            raise SystemExit(
                f"Tokenizer vocab ({vocab_rows}) is larger than the model's "
                f"token-embedding table ({emb_rows}). The vocab_file and the "
                f"model checkpoint do not match — a token id would be out of "
                f"range and crash the embedding lookup. Use the vocab.txt that "
                f"ships with {model_dir}."
            )
        logger.info("Vocab/model check OK (vocab=%d, embeddings=%d).",
                    vocab_rows, emb_rows)
    except AttributeError:
        # Unexpected model layout; skip the check rather than block the run.
        logger.warning("Could not verify vocab/embedding sizes; skipping check.")

    return model, tokenizer, device


# ============================================================================
# SCORING (faithful port of detection.MLM / single_MLM)
# ============================================================================


def make_single_mlm(model, tokenizer, device, mlm_k, max_length=512):
    """
    Build the per-sentence scorer, a port of Zhu's ``single_MLM``.

    Returns a function ``single_mlm(message) -> [[token, prob], ...]`` of the
    top ``mlm_k`` predictions at the mask position, identical to Zhu's — with
    one safety addition: the token sequence is truncated to BERT's
    ``max_length`` positions (default 512), always keeping the ``[MASK]`` token
    inside the window. Zhu's original corpus was pre-filtered to short lines
    (``len(line) <= 150`` in read_file.read_raw_text), so it never hit the
    limit; the streamed Reddit corpus here has no such cap, and a sequence
    longer than the model's position-embedding table triggers a CUDA
    index-out-of-range assert (``indexSelectLargeIndex ... srcIndex <
    srcSelectDimSize``). Truncating around the mask prevents that while leaving
    short sentences (the overwhelming majority) byte-identical to Zhu.
    """
    import torch

    def to_bert_input(tokens):
        token_idx = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
        sep_idx = tokens.index(SEP)
        segment_idx = token_idx * 0
        segment_idx[(sep_idx + 1):] = 1
        mask = (token_idx != 0)
        return (
            token_idx.unsqueeze(0).to(device),
            segment_idx.unsqueeze(0).to(device),
            mask.unsqueeze(0).to(device),
        )

    def single_mlm(message):
        tokens = tokenizer.tokenize(message)
        if len(tokens) == 0:
            return []
        # Keep the body short enough that, with [CLS] + [SEP] added, the full
        # sequence fits in max_length. Center the window on the [MASK] so the
        # prediction position is never truncated away.
        body_budget = max_length - 2  # room for [CLS] and [SEP]
        if len(tokens) > body_budget:
            try:
                m = tokens.index(MASK)
            except ValueError:
                m = len(tokens) // 2  # no mask (shouldn't happen); center-crop
            half = body_budget // 2
            start = max(0, m - half)
            end = start + body_budget
            if end > len(tokens):
                end = len(tokens)
                start = end - body_budget
            tokens = tokens[start:end]
        if tokens[0] != CLS:
            tokens = [CLS] + tokens
        if tokens[-1] != SEP:
            tokens.append(SEP)
        token_idx, segment_idx, mask = to_bert_input(tokens)
        with torch.no_grad():
            logits = model(token_idx, segment_idx, mask, masked_lm_labels=None)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        topk_prob, topk_tokens = None, None
        for idx, token in enumerate(tokens):
            if token == MASK:
                topk_prob, topk_indices = torch.topk(probs[idx, :], mlm_k)
                topk_tokens = tokenizer.convert_ids_to_tokens(
                    topk_indices.cpu().numpy()
                )
        if topk_tokens is None:  # no mask survived tokenization
            return []
        return [[topk_tokens[i], float(topk_prob[i])] for i in range(mlm_k)]

    return single_mlm


def score_sentence(top_words, input_keyword_set, stopword_set, thres):
    """
    Apply Zhu's informativeness gate + candidate accumulation to one sentence's
    top predictions.

    Returns ``(informative, contributions)`` where ``informative`` is False
    when the sentence is discarded by ``filter_uninformative`` (fewer than two
    input keywords in the top ``thres`` predictions), and ``contributions`` is
    a list of ``(token, prob)`` to add to the running score (empty if not
    informative). The token-exclusion rules are identical to ``detection.MLM``.
    """
    seen_input = 0
    top_thres_tokens = [x[0] for x in top_words[:thres]]
    for kw in input_keyword_set:
        if kw in top_thres_tokens:
            seen_input += 1
    if seen_input < 2:
        return False, []

    contributions = []
    for token, prob in top_words:
        if token in string.punctuation:
            continue
        if token in stopword_set:
            continue
        if token in input_keyword_set:
            continue
        if token in ("drug", "drugs"):  # Zhu excludes these for the drug set.
            continue
        if token[:2] == "##":  # BERT word-piece continuation, not a word.
            continue
        contributions.append((token, prob))
    return True, contributions


# ============================================================================
# MASKED-SENTENCE EXTRACTION
# ============================================================================


def iter_masked_sentences(stream, input_keyword_set, max_chars, dropped,
                          split_sentences=False):
    """
    Yield ``(half_year, masked_sentence)`` for each occurrence of an input
    keyword, masking that occurrence — the temporal analogue of the masked
    sentence construction in ``euphemism_detection``.

    Zhu tokenizes with ``nltk.word_tokenize`` and masks the FIRST index of each
    keyword in the text unit. This reproduces that: one masked sentence per
    (unit, keyword) pair, using the first occurrence of the keyword.

    Text unit (``split_sentences``):
      * ``False`` (default, faithful to Zhu): the unit is the WHOLE record
        (one comment/submission). Zhu's ``euphemism_detection`` iterates
        ``all_text`` and never splits on sentence boundaries — its variable is
        named ``sentence`` but holds a full line. Kept short only by the char
        cap below.
      * ``True``: the record is first split into sentences (shared
        ``text_segmentation.split_sentences``), and each sentence is its own
        unit. A keyword is masked in the sentence that contains it, so long
        records contribute their short sentences instead of being dropped
        whole by the char cap. This is NOT what Zhu's paper does; it is the
        sentence-level reading the hit-matrix pipeline uses.

    Length filter (faithful to Zhu): ``read_file.read_raw_text`` only keeps
    source lines with ``len(line) <= 150`` CHARACTERS before any masking, which
    is why Zhu never over-runs BERT's 512 positions. The same character cap is
    applied here PER UNIT: a unit longer than ``max_chars`` is skipped entirely
    (counted under ``too_long`` in ``dropped``), rather than cropped. With
    ``split_sentences=True`` the cap is per sentence, so a long record is not
    discarded wholesale — only its over-long sentences are. ``max_chars=None``
    disables the filter (the mask-window truncation in ``single_mlm`` is then
    the only guard against the position limit).
    """
    import nltk

    from text_segmentation import iter_sentences

    def mask_units(half_year, unit_text):
        """Yield masked sentences for every keyword in one text unit."""
        if max_chars is not None and len(unit_text) > max_chars:
            dropped["too_long"] += 1
            return
        temp = nltk.word_tokenize(unit_text)
        token_set = set(temp)
        for kw in input_keyword_set:
            if kw not in token_set:
                continue
            temp_index = temp.index(kw)
            masked = (
                " ".join(temp[:temp_index])
                + " [MASK] "
                + " ".join(temp[temp_index + 1:])
            )
            yield half_year, masked

    for half_year, text in stream:
        if split_sentences:
            for sent_text, _s, _e in iter_sentences(text):
                yield from mask_units(half_year, sent_text)
        else:
            yield from mask_units(half_year, text)


# ============================================================================
# SHARD IO
# ============================================================================


def write_shard(shard_dir, shard_name, per_hy, meta):
    """
    Write one month's partial detection scores as a JSON shard.

    ``per_hy`` maps ``half_year -> {"score_sum": {token: float},
    "n_informative": int, "n_masked": int}``. The merge sums ``score_sum``
    across months within each half-year and re-ranks.
    """
    os.makedirs(shard_dir, exist_ok=True)
    final_path = os.path.join(shard_dir, f"{shard_name}{SHARD_SUFFIX}")
    tmp_path = final_path + ".tmp"

    payload = {
        "shard_name": shard_name,
        "half_years": {
            hy: {
                "score_sum": d["score_sum"],
                "n_informative": d["n_informative"],
                "n_masked": d["n_masked"],
            }
            for hy, d in per_hy.items()
        },
        **meta,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, final_path)
    logger.info(
        "Wrote shard %s (%d half-year bucket(s))", final_path, len(per_hy)
    )
    return final_path


# ============================================================================
# MONTH PROCESSING
# ============================================================================


def process_month(stream, input_keywords, single_mlm, args):
    """
    Single pass over a month: build masked sentences, score with BERT, and
    accumulate per-half-year token score sums.

    Returns ``(per_hy, dropped)`` where ``per_hy`` maps ``half_year ->
    {score_sum, n_informative, n_masked}`` and ``dropped`` counts sentences
    skipped before scoring (e.g. ``too_long`` from the char filter).
    """
    import nltk
    from nltk.corpus import stopwords

    stopword_set = set(stopwords.words("english"))
    input_keyword_set = set(input_keywords)

    per_hy = defaultdict(
        lambda: {
            "score_sum": defaultdict(float),
            "n_informative": 0,
            "n_masked": 0,
        }
    )
    dropped = defaultdict(int)
    # Optional per-half-year cap on masked sentences (mirrors Zhu's ms_limit,
    # but applied per half-year so each bucket is comparable).
    cap = args.ms_limit_per_half_year
    hy_counts = defaultdict(int)

    masked_iter = iter_masked_sentences(
        stream, input_keyword_set, args.max_chars, dropped,
        split_sentences=args.split_sentences,
    )
    for half_year, masked in masked_iter:
        if cap is not None and hy_counts[half_year] >= cap:
            continue
        hy_counts[half_year] += 1

        bucket = per_hy[half_year]
        bucket["n_masked"] += 1

        top_words = single_mlm(masked)
        if not top_words:
            continue
        informative, contributions = score_sentence(
            top_words, input_keyword_set, stopword_set, args.thres,
        )
        if not informative:
            continue
        bucket["n_informative"] += 1
        score_sum = bucket["score_sum"]
        for token, prob in contributions:
            score_sum[token] += prob

    # Convert inner defaultdicts to plain dicts for JSON.
    out = {}
    for hy, d in per_hy.items():
        out[hy] = {
            "score_sum": dict(d["score_sum"]),
            "n_informative": d["n_informative"],
            "n_masked": d["n_masked"],
        }
    return out, dict(dropped)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Zhu euphemism detection, one month per task (temporal "
        "version of the detection stage).",
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
                    help="Directory to write this month's shard into.")

    # Scope.
    sc = parser.add_argument_group("scope")
    sc.add_argument("--subreddit-file", default=None,
                    help="One subreddit per line; omit for all of Reddit.")
    sc.add_argument("--start-year", type=int, default=2012)
    sc.add_argument("--end-year", type=int, default=2026)

    # Keywords (Zhu's input/target terms). Prefer --answer-file to reproduce
    # Zhu's input_keywords exactly (the formal names from the euphemism answer
    # key); --keywords-file is the fallback for an arbitrary term list.
    kw = parser.add_argument_group("keywords (choose one)")
    kw.add_argument("--answer-file", default=None,
                    help="Zhu euphemism_answer_<cat>.txt. Input keywords are "
                    "derived exactly as read_file.py does (the formal names "
                    "before each ':'). Preferred for a faithful Zhu run.")
    kw.add_argument("--keywords-file", default=None,
                    help="Fallback: newline-delimited input keywords (one "
                    "target term per line). Use when you do not have Zhu's "
                    "answer-file format.")

    # Model.
    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-dir", required=True,
                     help="BertForMaskedLM directory (Zhu's BERT_model_reddit).")
    mdl.add_argument("--vocab-file", required=True,
                     help="BERT vocab.txt for the tokenizer.")
    mdl.add_argument("--max-length", type=int, default=512,
                     help="Max BERT sequence length. Sentences longer than "
                     "this are truncated to a window around the [MASK] so they "
                     "fit the model's position limit (Zhu's corpus was "
                     "pre-filtered short, so it never truncated).")

    # Zhu scoring knobs (defaults match detection.py).
    z = parser.add_argument_group("zhu scoring")
    z.add_argument("--mlm-k", type=int, default=50,
                   help="Top-k predictions kept per mask (Zhu MLM_k=50).")
    z.add_argument("--thres", type=int, default=5,
                   help="Informativeness threshold: a masked sentence is kept "
                   "only if >=2 input keywords appear in the mask's top-thres "
                   "predictions (Zhu euphemism_detection thres=5).")
    z.add_argument("--ms-limit-per-half-year", type=int, default=None,
                   help="Optional cap on masked sentences scored per half-year "
                   "(analogue of Zhu's ms_limit, per bucket). Default: no cap.")
    z.add_argument("--max-chars", type=int, default=150,
                   help="Skip any text UNIT longer than this many CHARACTERS "
                   "before masking, exactly as Zhu's read_file.read_raw_text "
                   "does (len(line) <= 150). This is why Zhu never over-runs "
                   "BERT's position limit. With --split-sentences the cap is "
                   "per sentence. Set to 0 to disable the filter (then "
                   "--max-length truncation is the only guard).")
    z.add_argument("--split-sentences", action="store_true",
                   help="Split each record into sentences (shared "
                   "text_segmentation.split_sentences) and score each sentence "
                   "as its own unit, instead of the whole record. NOT faithful "
                   "to Zhu (whose unit is the full line), but it lets long "
                   "records contribute their short sentences rather than being "
                   "dropped by --max-chars. Default off (Zhu behaviour).")

    args = parser.parse_args()

    if args.shard_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        args.shard_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.shard_index is None:
        parser.error("--shard-index is required (or set SLURM_ARRAY_TASK_ID).")

    if bool(args.answer_file) == bool(args.keywords_file):
        parser.error(
            "Provide exactly one of --answer-file (Zhu answer key, preferred) "
            "or --keywords-file (plain term list)."
        )

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

    # NLTK data needed by word_tokenize / stopwords; download quietly if absent.
    import nltk
    for pkg in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception as exc:  # noqa: BLE001 - best-effort, offline-safe
            logger.warning("nltk.download(%s) skipped: %s", pkg, exc)

    # --max-chars 0 (or negative) disables the length filter.
    if args.max_chars is not None and args.max_chars <= 0:
        args.max_chars = None

    allowed = load_subreddits(args.subreddit_file)
    if args.answer_file:
        keywords_src = os.path.abspath(args.answer_file)
        keywords_mode = "answer_file"
        input_keywords = load_keywords_from_answer_file(args.answer_file)
    else:
        keywords_src = os.path.abspath(args.keywords_file)
        keywords_mode = "keywords_file"
        input_keywords = load_keywords_from_list(args.keywords_file)

    model, tokenizer, device = load_model(args.model_dir, args.vocab_file)
    single_mlm = make_single_mlm(
        model, tokenizer, device, args.mlm_k, max_length=args.max_length,
    )

    drop_counts = defaultdict(int)
    stream = stream_corpus(
        files, args.start_year, args.end_year, allowed, drop_counts,
    )
    per_hy, sentence_drops = process_month(
        stream, input_keywords, single_mlm, args,
    )

    total_masked = sum(d["n_masked"] for d in per_hy.values())
    total_informative = sum(d["n_informative"] for d in per_hy.values())
    logger.info(
        "Month %s: %d half-year bucket(s), %d masked sentences, %d informative "
        "(%.1f%%), sentence drops=%s, record drops=%s",
        month_key, len(per_hy), total_masked, total_informative,
        (100.0 * total_informative / total_masked) if total_masked else 0.0,
        dict(sentence_drops), dict(drop_counts),
    )

    meta = {
        "scope": {
            "mode": "subreddits" if allowed is not None else "all",
            "file": args.subreddit_file,
            "hash": subreddit_hash(allowed),
        },
        "keywords": {
            "file": keywords_src,
            "source": keywords_mode,
            "n": len(input_keywords),
        },
        "model": {
            "model_dir": os.path.abspath(args.model_dir),
            "vocab_file": os.path.abspath(args.vocab_file),
        },
        "params": {
            "mlm_k": args.mlm_k,
            "thres": args.thres,
            "ms_limit_per_half_year": args.ms_limit_per_half_year,
            "max_chars": args.max_chars,
            "max_length": args.max_length,
            "split_sentences": args.split_sentences,
        },
        "record_drop_counts": dict(drop_counts),
        "sentence_drop_counts": sentence_drops,
        "files": list(files),
    }

    write_shard(args.shard_dir, month_key, per_hy, meta)


if __name__ == "__main__":
    main()
