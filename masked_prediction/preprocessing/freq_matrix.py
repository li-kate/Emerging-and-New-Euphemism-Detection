"""
Build, filter, and save the frequency matrix.

The frequency matrix counts how many times each word occurs in each half-year
of the filtered Reddit corpus (rows = half-years such as ``2012H1``, columns =
words). It is the source of truth for two things downstream:

1. **Target word list** — its columns (after filtering) are the target words
   the hit matrix script scores.
2. **Denominator** — ``hit_rate_k = hits_k / frequency``.

See ``filter.md`` for the full plan. This script:

  * streams the corpus (``.zst`` Reddit dumps or ``.jsonl``), tagging each
    record by half-year and applying the subreddit scope filter first;
  * tokenizes with the same lowercase, word-boundary rule used elsewhere so
    "Molly" and "molly" land in one column;
  * accumulates per (half-year, word) counts, applying the prefilter from
    ``filter.py`` as each unique term is first seen;
  * builds a dense ``[half_years, words]`` matrix, runs the postfilter;
  * saves ``freq.npy`` plus label files, the stop-word list, per-half-year
    totals, and ``freq_meta.json``.

Example:

    python freq_matrix.py \\
        --reddit-monthly-dir /data/reddit \\
        --out-dir freq_all \\
        --dtype int32

    python freq_matrix.py \\
        --input corpus.jsonl \\
        --subreddit-file subreddits.txt \\
        --out-dir freq_subreddits \\
        --dtype int64
"""

import argparse
import glob
import io
import json
import logging
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Imports from the shared filter module (repo root) and the scope helper.
# filter.py lives at the repository root; add it to sys.path so this script
# works no matter where it is invoked from.
# ----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from filter import (
    MAX_WORDS,
    MIN_FREQ,
    MIN_TERM_LENGTH,
    STOP_WORDS,
    log_postfilter_counts,
    log_prefilter_counts,
    postfilter,
    prefilter,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subreddit_scope import (
    keep,
    load_subreddits,
    subreddit_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")

# Tokenization / occurrence matching.
#
# This defines what counts as one occurrence of a word, and must match the
# occurrence-matching rule the hit matrix script reuses (it imports ``tokenize``
# from here). A token is a maximal run of letters/digits, optionally joined by
# a single apostrophe or hyphen (so "don't" and "x-ray" stay whole). Matching
# is case-insensitive; the lowercased form is the column key, mirroring the
# Aho-Corasick matcher's lowercase, alnum-boundary matching in
# collect_instances.py.
TOKEN_RE = re.compile(r"[^\W\d_](?:[\w'\-]*[^\W\d_])?", re.UNICODE)

# ----------------------------------------------------------------------------
# JSON backend: prefer orjson, fall back to stdlib json (mirrors
# collect_instances.py).
# ----------------------------------------------------------------------------
try:
    import orjson

    def _json_loads(s):
        return orjson.loads(s)

    logger.info("Using orjson for JSON parsing")
except ImportError:
    logger.warning("orjson not installed — falling back to stdlib json (slower).")

    def _json_loads(s):
        return json.loads(s)


def tokenize(text):
    """Yield lowercased tokens from ``text`` (the occurrence-matching rule)."""
    for m in TOKEN_RE.finditer(text):
        yield m.group(0).lower()


def half_year_label(dt):
    """Return the half-year label for a datetime, e.g. ``2012H1``."""
    return f"{dt.year}H{1 if dt.month <= 6 else 2}"


# ============================================================================
# CORPUS STREAMING
# ============================================================================


def _parse_record(obj, start_year, end_year, allowed):
    """
    Turn a raw Reddit JSON object into ``(half_year, text)`` or ``None``.

    Applies, in order: subreddit scope filter, text presence, year window.
    Returns ``None`` (and the caller tallies the reason) when the record is
    dropped. The subreddit filter is checked first so later thresholds apply
    to the chosen scope.
    """
    sr = obj.get("subreddit", "")
    if not keep(sr, allowed):
        return None, "subreddit"

    body = obj.get("body", "") or obj.get("selftext", "")
    if not body or body in ("[deleted]", "[removed]"):
        return None, "empty"

    ts_raw = obj.get("created_utc")
    try:
        dt = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None, "bad_timestamp"

    if not start_year <= dt.year <= end_year:
        return None, "year"

    body = URL_RE.sub("", body)
    body = WHITESPACE_RE.sub(" ", body).strip()
    if not body:
        return None, "empty"

    return (half_year_label(dt), body), None


def _open_zst(path):
    """Open a .zst Reddit dump as a decoded text stream."""
    import zstandard as zstd

    fh = open(path, "rb")  # noqa: SIM115 — kept open for the stream's lifetime
    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    reader = dctx.stream_reader(fh, read_size=2**24)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="ignore"), fh


def _iter_lines(path):
    """Yield text lines from a .zst or plain .jsonl/.txt file."""
    if path.endswith(".zst"):
        stream, fh = _open_zst(path)
        try:
            yield from stream
        finally:
            stream.close()
            fh.close()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            yield from f


def stream_corpus(
    paths,
    start_year,
    end_year,
    allowed,
    drop_counts,
) -> Iterator[tuple[str, str]]:
    """
    Stream ``(half_year, text)`` pairs from one or more corpus files.

    ``drop_counts`` is a dict that is updated in place with the number of
    records dropped by each reason (subreddit, empty, year, bad_timestamp,
    parse_error) so the scope can be logged.
    """
    for path in paths:
        logger.info("Streaming %s", path)
        n_read = 0
        for line in _iter_lines(path):
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                obj = _json_loads(line)
            except ValueError:
                drop_counts["parse_error"] += 1
                continue

            parsed, reason = _parse_record(obj, start_year, end_year, allowed)
            if parsed is None:
                drop_counts[reason] += 1
                continue
            yield parsed

            if n_read % 5_000_000 == 0:
                logger.info("  %s: read %s", path, f"{n_read:,}")
        logger.info("  %s: done (%s read)", path, f"{n_read:,}")


# ============================================================================
# COUNTING
# ============================================================================


def build_counts(stream, allowed_term_cache):
    """
    Accumulate per (half-year, word) occurrence counts from the stream.

    Returns:
      counts      - dict[word] -> dict[half_year] -> int
      totals_raw  - dict[half_year] -> total tokens seen (before any
                    prefilter), for the per-half-year totals file.
      prefilter_counts - counts of terms dropped by each prefilter rule,
                    attributed the first time each unique term is seen.
      half_years  - set of half-year labels observed.

    ``allowed_term_cache`` maps a seen token to True/False (passes the
    prefilter); it is populated lazily so the prefilter runs once per unique
    term rather than once per occurrence.
    """
    counts = defaultdict(lambda: defaultdict(int))
    totals_raw = defaultdict(int)
    prefilter_counts = {"short_terms": 0, "multi_word_terms": 0, "stop_words": 0}
    half_years = set()

    n_records = 0
    for half_year, text in stream:
        half_years.add(half_year)
        n_records += 1
        for tok in tokenize(text):
            totals_raw[half_year] += 1

            passes = allowed_term_cache.get(tok)
            if passes is None:
                # First time we see this term: run the single-term prefilter
                # and attribute any drop to the first rule it fails.
                kept, c = prefilter([tok])
                if kept:
                    passes = True
                else:
                    passes = False
                    for key, val in c.items():
                        if val:
                            prefilter_counts[key] += 1
                            break
                allowed_term_cache[tok] = passes

            if passes:
                counts[tok][half_year] += 1

        if n_records % 1_000_000 == 0:
            logger.info(
                "  counted %s records | %s unique kept terms",
                f"{n_records:,}",
                f"{len(counts):,}",
            )

    logger.info("Counting done: %s records, %s unique kept terms",
                f"{n_records:,}", f"{len(counts):,}")
    return counts, totals_raw, prefilter_counts, half_years


def assemble_matrix(counts, half_years, dtype):
    """
    Turn the nested count dict into a dense ``[half_years, words]`` array.

    ``half_years`` is sorted chronologically. Words are sorted alphabetically
    for a stable, reproducible column order. Raises if any count exceeds the
    chosen dtype's maximum (so an int32 run never silently overflows).
    """
    half_years_sorted = sorted(
        half_years, key=lambda h: (int(h[:4]), int(h[5:]))
    )
    words = sorted(counts.keys())
    hy_idx = {h: i for i, h in enumerate(half_years_sorted)}

    freq = np.zeros((len(half_years_sorted), len(words)), dtype=dtype)
    max_val = np.iinfo(dtype).max

    for j, w in enumerate(words):
        for h, c in counts[w].items():
            if c > max_val:
                raise OverflowError(
                    f"Count {c:,} for word {w!r} in {h} exceeds {dtype} max "
                    f"({max_val:,}). Rerun with --dtype int64."
                )
            freq[hy_idx[h], j] = c

    return freq, half_years_sorted, words


# ============================================================================
# SAVING
# ============================================================================


def save_outputs(
    out_dir,
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
):
    """Write the matrix, labels, stop-word list, totals, and metadata."""
    os.makedirs(out_dir, exist_ok=True)

    def p(name):
        return os.path.join(out_dir, name)

    np.save(p(names["matrix"]), freq)
    logger.info("Saved %s %s %s", p(names["matrix"]), freq.shape, freq.dtype)

    pd.Series(half_years, name="half_year").to_csv(
        p(names["half_years"]), index=False
    )
    pd.Series(words, name="word").to_csv(p(names["words"]), index=False)

    with open(p(names["stopwords"]), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(STOP_WORDS)))

    # Per-half-year token totals: before any filter and after all filters.
    totals = pd.DataFrame(
        {
            "half_year": half_years,
            "total_tokens": [totals_raw.get(h, 0) for h in half_years],
            "total_tokens_filtered": [
                totals_filtered.get(h, 0) for h in half_years
            ],
        }
    )
    totals.to_csv(p(names["totals"]), index=False)

    # Copy of the subreddit list used (absent for an all-Reddit run).
    if scope["mode"] == "subreddits" and args.subreddit_file:
        with open(args.subreddit_file, encoding="utf-8") as src:
            content = src.read()
        with open(p(names["subreddits"]), "w", encoding="utf-8") as dst:
            dst.write(content)

    meta = {
        "script": "freq_matrix.py",
        "version": args.version,
        "shape": list(freq.shape),
        "dtype": str(freq.dtype),
        "corpus": "Reddit, URL-stripped, whitespace-normalized bodies/selftext",
        "scope": scope,
        "filters": {
            "min_term_length": MIN_TERM_LENGTH,
            "max_words": MAX_WORDS,
            "stopwords": "nltk english",
            "min_freq_any_half_year": args.min_freq,
        },
        "filter_counts": {**prefilter_counts, **postfilter_counts},
        "matching": (
            "tokens = regex "
            + TOKEN_RE.pattern
            + "; lowercased; one occurrence = one token match"
        ),
        "year_range": [args.start_year, args.end_year],
        "files": names,
    }
    with open(p(names["meta"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved %s", p(names["meta"]))


# ============================================================================
# MAIN
# ============================================================================


def resolve_inputs(args):
    """Expand the input options into a concrete, sorted list of file paths."""
    paths = []
    if args.input:
        paths.extend(args.input)
    if args.reddit_monthly_dir:
        found = sorted(glob.glob(os.path.join(args.reddit_monthly_dir, "RC_*.zst")))
        found += sorted(glob.glob(os.path.join(args.reddit_monthly_dir, "RS_*.zst")))
        logger.info("Found %d dumps in %s", len(found), args.reddit_monthly_dir)
        paths.extend(found)
    if args.input_dir:
        for pat in ("*.zst", "*.jsonl"):
            paths.extend(
                sorted(glob.glob(os.path.join(args.input_dir, "**", pat),
                                 recursive=True))
            )
    # De-duplicate while preserving order.
    seen = set()
    ordered = []
    for pth in paths:
        if pth not in seen:
            seen.add(pth)
            ordered.append(pth)
    return ordered


def build_output_names(stem):
    """Build the set of output filenames from a shared stem (default 'freq')."""
    return {
        "matrix": f"{stem}.npy",
        "half_years": f"{stem}_half_years.csv",
        "words": f"{stem}_words.csv",
        "stopwords": f"{stem}_stopwords.txt",
        "subreddits": f"{stem}_subreddits.txt",
        "totals": f"{stem}_totals.csv",
        "meta": f"{stem}_meta.json",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build, filter, and save the frequency matrix (filter.md).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Inputs (pick one or more) ---
    src = parser.add_argument_group("inputs")
    src.add_argument(
        "--input",
        nargs="+",
        help="One or more corpus files (.zst Reddit dumps or .jsonl).",
    )
    src.add_argument(
        "--input-dir",
        help="Directory searched recursively for *.zst and *.jsonl files.",
    )
    src.add_argument(
        "--reddit-monthly-dir",
        help="Directory of monthly Reddit dumps (RC_YYYY-MM.zst / RS_*.zst).",
    )

    # --- Scope ---
    scope_g = parser.add_argument_group("scope")
    scope_g.add_argument(
        "--subreddit-file",
        default=None,
        help="File listing subreddits to include (one per line). "
        "Omit for all of Reddit.",
    )
    scope_g.add_argument("--start-year", type=int, default=2012)
    scope_g.add_argument("--end-year", type=int, default=2026)

    # --- Filter thresholds ---
    filt = parser.add_argument_group("filters")
    filt.add_argument(
        "--min-freq",
        type=int,
        default=MIN_FREQ,
        help="Postfilter: keep a word if it reaches this count in at least "
        "one half-year.",
    )

    # --- Outputs ---
    out = parser.add_argument_group("outputs")
    out.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (tag by scope, e.g. freq_all/ or "
        "freq_subreddits/).",
    )
    out.add_argument(
        "--stem",
        default="freq",
        help="Filename stem for all outputs (e.g. 'freq' -> freq.npy, "
        "freq_words.csv, ...).",
    )
    out.add_argument(
        "--dtype",
        choices=["int32", "int64"],
        default="int32",
        help="Integer size for the matrix. int32 ~160MB, int64 ~320MB at "
        "~40 half-years x ~1M words.",
    )
    out.add_argument(
        "--version",
        default="unversioned",
        help="Version/commit string recorded in the metadata.",
    )

    args = parser.parse_args()

    paths = resolve_inputs(args)
    if not paths:
        parser.error(
            "Provide at least one input: --input, --input-dir, or "
            "--reddit-monthly-dir."
        )
    logger.info("Resolved %d input file(s)", len(paths))

    dtype = np.int32 if args.dtype == "int32" else np.int64
    names = build_output_names(args.stem)

    # Scope.
    allowed = load_subreddits(args.subreddit_file)
    if allowed is not None:
        logger.info("Scope: %d subreddits from %s", len(allowed),
                    args.subreddit_file)
    else:
        logger.info("Scope: all of Reddit")

    drop_counts = defaultdict(int)
    stream = stream_corpus(
        paths, args.start_year, args.end_year, allowed, drop_counts
    )

    # Count.
    term_cache = {}
    counts, totals_raw, prefilter_counts, half_years = build_counts(
        stream, term_cache
    )

    if not counts:
        logger.error("No terms survived counting. Check inputs and filters.")
        sys.exit(1)

    # Log scope + prefilter.
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

    # Assemble + postfilter.
    freq, half_years, words = assemble_matrix(counts, half_years, dtype)
    logger.info("Pre-postfilter matrix: %s", freq.shape)

    total_before = len(words)
    freq, words, postfilter_counts = postfilter(freq, words, args.min_freq)
    log_postfilter_counts(postfilter_counts, total_before, args.min_freq)

    # Per-half-year filtered totals (sum over kept columns).
    totals_filtered = {
        h: int(freq[i].sum()) for i, h in enumerate(half_years)
    }

    # Scope metadata.
    n_listed = len(allowed) if allowed is not None else None
    scope = {
        "mode": "subreddits" if allowed is not None else "all",
        "file": args.subreddit_file,
        "hash": subreddit_hash(allowed),
        "n_listed": n_listed,
        "n_records_dropped_by_subreddit_filter": int(drop_counts["subreddit"]),
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
        "Final matrix: %s half-years x %s words (%s), saved to %s",
        len(half_years),
        f"{len(words):,}",
        args.dtype,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
