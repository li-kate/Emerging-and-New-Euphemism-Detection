"""
Shared corpus streaming, tokenization, and sharding helpers.

The frequency matrix build (``freq_matrix.py``) and the hit matrix build
(``hit_matrix.py``) read the same Reddit corpus the same way: same subreddit
scope filter, same record parsing, same tokenization / occurrence rule, same
month-based sharding for SLURM arrays. The occurrence rule in particular MUST
match between the two scripts — the hit matrix divides by the frequency
matrix's counts, so a token the two scripts disagree about would break the
denominator (see ``hit_matrix_plan.md`` sections 3b and 8).

This module is the single home for that shared logic so the two scripts cannot
drift apart. It depends only on the standard library (plus the local
``subreddit_scope``), so importing it never pulls in numpy/torch.

What lives here:

  * ``tokenize`` / ``TOKEN_RE`` — the occurrence rule (lowercased maximal
    letter runs, apostrophe/hyphen joins kept). One match = one occurrence.
  * ``half_year_label`` — datetime -> ``YYYYH1`` / ``YYYYH2``.
  * ``parse_record`` / ``stream_corpus`` / ``iter_lines`` — turn raw dump lines
    into ``(half_year, text)`` pairs, applying scope, text-presence, and year
    filters, tallying drop reasons.
  * ``group_by_month`` / ``slice_for_shard`` / ``select_month_key`` — divide
    the input files into SLURM-array tasks (one month per task by default).

Note: ``freq_matrix.py`` predates this module and keeps its own in-file copies
of the counting-side helpers; this module is written to be byte-compatible with
those (same regex, same parse order) so a future consolidation is mechanical.
"""

import glob
import io
import logging
import os
import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from subreddit_scope import keep

logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")

# Occurrence rule. A token is a maximal run of letters, optionally joined by a
# single apostrophe or hyphen (so "don't" / "x-ray" stay whole). Matching is
# case-insensitive; the lowercased form is the key. This is identical to the
# TOKEN_RE in freq_matrix.py and must stay that way.
TOKEN_RE = re.compile(r"[^\W\d_](?:[\w'\-]*[^\W\d_])?", re.UNICODE)

# Month key extractor for RC_/RS_ dumps (parts grouped under one month).
_MONTH_RE = re.compile(r"^(R[CS]_\d{4}-\d{2})")


# ----------------------------------------------------------------------------
# JSON backend: prefer orjson, fall back to stdlib json.
# ----------------------------------------------------------------------------
try:
    import orjson

    def json_loads(s):
        return orjson.loads(s)

except ImportError:  # pragma: no cover - environment dependent
    import json as _json

    def json_loads(s):
        return _json.loads(s)


def tokenize(text):
    """Yield lowercased tokens from ``text`` (the occurrence-matching rule)."""
    for m in TOKEN_RE.finditer(text):
        yield m.group(0).lower()


def iter_token_spans(text):
    """
    Yield ``(token_lower, start, end)`` for each occurrence in ``text``.

    Same rule as ``tokenize`` but also returns character offsets, which the hit
    matrix script needs to mask a specific occurrence. Offsets are into the
    original ``text`` (not lowercased), so slicing ``text[start:end]`` returns
    the surface form.
    """
    for m in TOKEN_RE.finditer(text):
        yield m.group(0).lower(), m.start(), m.end()


def half_year_label(dt):
    """Return the half-year label for a datetime, e.g. ``2012H1``."""
    return f"{dt.year}H{1 if dt.month <= 6 else 2}"


# ============================================================================
# RECORD PARSING / STREAMING
# ============================================================================


def parse_record(obj, start_year, end_year, allowed):
    """
    Turn a raw Reddit JSON object into ``((half_year, text), None)`` or
    ``(None, reason)``.

    Applies, in order: subreddit scope filter, text presence, timestamp
    validity, year window. The subreddit filter is checked first so later
    thresholds apply to the chosen scope. Mirrors ``freq_matrix._parse_record``.
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


def iter_lines(path):
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

    ``drop_counts`` is a dict updated in place with the number of records
    dropped by each reason (subreddit, empty, year, bad_timestamp,
    parse_error).
    """
    for path in paths:
        logger.info("Streaming %s", path)
        n_read = 0
        for line in iter_lines(path):
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                obj = json_loads(line)
            except ValueError:
                drop_counts["parse_error"] += 1
                continue

            parsed, reason = parse_record(obj, start_year, end_year, allowed)
            if parsed is None:
                drop_counts[reason] += 1
                continue
            yield parsed

            if n_read % 5_000_000 == 0:
                logger.info("  %s: read %s", path, f"{n_read:,}")
        logger.info("  %s: done (%s read)", path, f"{n_read:,}")


# ============================================================================
# SHARDING (SLURM array task selection)
# ============================================================================


def resolve_inputs(input=None, input_dir=None, reddit_monthly_dir=None):
    """
    Expand input options into a de-duplicated, ordered list of file paths.

    Mirrors ``freq_matrix.resolve_inputs`` so both scripts accept the same
    ``--input`` / ``--input-dir`` / ``--reddit-monthly-dir`` options.
    """
    paths = []
    if input:
        paths.extend(input)
    if reddit_monthly_dir:
        found = sorted(glob.glob(os.path.join(reddit_monthly_dir, "RC_*.zst")))
        found += sorted(glob.glob(os.path.join(reddit_monthly_dir, "RS_*.zst")))
        logger.info("Found %d dumps in %s", len(found), reddit_monthly_dir)
        paths.extend(found)
    if input_dir:
        for pat in ("*.zst", "*.jsonl"):
            paths.extend(
                sorted(
                    glob.glob(os.path.join(input_dir, "**", pat), recursive=True)
                )
            )
    seen = set()
    ordered = []
    for pth in paths:
        if pth not in seen:
            seen.add(pth)
            ordered.append(pth)
    return ordered


def group_by_month(paths):
    """
    Group dump files by month into ``{month_key: [files]}``, sorted by key.

    A month key is the ``RC_YYYY-MM`` / ``RS_YYYY-MM`` stem; multi-part pieces
    (``RC_2018-07_part000.zst``) group under one key. Non-dump inputs each get
    their own group keyed by their stem. Mirrors ``freq_matrix.group_by_month``
    and the month grouping in ``collect_instances.py``.
    """
    groups = defaultdict(list)
    for path in paths:
        stem = Path(path).stem
        m = _MONTH_RE.match(stem)
        key = m.group(1) if m else stem
        groups[key].append(path)
    return {k: sorted(groups[k]) for k in sorted(groups)}


def slice_for_shard(paths, shard_index, num_shards):
    """Return this shard's round-robin slice ``paths[shard_index::num_shards]``."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index {shard_index} out of range for num_shards {num_shards}"
        )
    return paths[shard_index::num_shards]


def select_month_key(all_paths, shard_index, months=None, max_months=None):
    """
    Pick the month this array task should process.

    Returns ``(month_key, files)`` or ``(None, None)`` if the index is past the
    end of the (optionally restricted) month list. ``months`` restricts to an
    explicit list of keys; ``max_months`` caps to the first N. Mirrors the month
    selection logic in ``freq_matrix.main``.
    """
    grouped = group_by_month(all_paths)
    month_keys = list(grouped)

    if months:
        requested = list(months)
        missing = [m for m in requested if m not in grouped]
        if missing:
            logger.warning(
                "Requested months not found in input: %s", ", ".join(missing)
            )
        month_keys = [m for m in requested if m in grouped]
    elif max_months is not None:
        month_keys = month_keys[:max_months]

    if not month_keys:
        return None, None
    if shard_index >= len(month_keys):
        return None, None

    key = month_keys[shard_index]
    return key, grouped[key]
