"""
============================================================================
EXHAUSTIVE INSTANCE COLLECTION
============================================================================

Finds every occurrence of target words across Reddit dumps or CSV files.
Saves each match with full context, timestamp, source metadata, and
which words were found.

This collects ALL instances — euphemistic, literal, and ambiguous.
The temporal analysis (Stage 2) determines which are which.

Usage:
    # Reddit per-subreddit dumps
    python collect_instances.py \
        --words euphemisms.txt taboo_keywords.txt comparison_words.txt \
        --reddit-dir ./reddit_dumps/ \
        --output ./matches/reddit_matches.jsonl

    # Reddit monthly dumps (full scan)
    python collect_instances.py \
        --words euphemisms.txt \
        --reddit-monthly-dir ./monthly_dumps/ \
        --output ./matches/full_reddit_matches.jsonl

    # CSV files
    python collect_instances.py \
        --words euphemisms.txt \
        --csv-dir ./csv_data/ \
        --output ./matches/csv_matches.jsonl

    # Single Reddit file (for testing)
    python collect_instances.py \
        --words euphemisms.txt \
        --reddit-file ./RC_2020-01.zst \
        --output ./matches/test_matches.jsonl

    # SLURM array (one monthly dump per task)
    python collect_instances.py \
        --words euphemisms.txt \
        --reddit-monthly-dir ./monthly_dumps/ \
        --output ./matches/ \
        --slurm-task-id $SLURM_ARRAY_TASK_ID

    # RUN THIS: python3 collect_instances.py --words drug_words.txt anchors_and_baselines.txt comparison_words.txt --reddit-file ./RC_2020-09.zst --output ./matches/2020-09_matches.jsonl

Required: pip install pyahocorasick zstandard
============================================================================
"""

import argparse
import glob
import io
import json
import logging
import os
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# WORD LIST LOADING
# ============================================================================


def load_word_list(path: str) -> dict[str, str]:
    """
    Load words from a text file. Supports multiple formats:

    Simple (one word per line):
        snow
        ice
        pot

    Tab-separated (word<TAB>drug_ref<TAB>date<TAB>source — only first 2 cols used):
        snow	cocaine	pre-2015	dea_2017
        fenty	fentanyl	2017	dea_2017
        needle	comparison

    Bullet list (markdown-style — strips bullets and metadata after " - "):
        * Skittles - urban dictionary says 2011 - means adderall
        * Fenty (urban dictionary shows earliest 2019)
        - needle
        - pharmacy

    Returns dict: word -> category (or "unknown" if no category)
    """
    words = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Strip markdown bullets
            if line.startswith(("* ", "- ", "+ ")):
                line = line[2:].strip()
            if line.startswith(("*", "-")) and len(line) > 1 and line[1] == " ":
                line = line[2:].strip()

            # Try tab-separated first
            parts = line.split("\t")
            if len(parts) >= 2:
                word = parts[0].strip().lower()
                category = parts[1].strip()
                if word:
                    words[word] = category
                continue

            # Handle bullet-list format: "Skittles - means adderall"
            # or "Fenty (urban dictionary shows earliest 2019)"
            # Extract just the word/phrase before any annotation
            # Remove parenthetical notes
            clean = re.sub(r"\(.*?\)", "", line).strip()
            # Split on " - " to separate word from notes
            if " - " in clean:
                clean = clean.split(" - ")[0].strip()

            word = clean.lower().strip()
            if word:
                words[word] = "unknown"

    logger.info(f"Loaded {len(words)} words from {path}")
    return words


def load_multiple_word_lists(paths: list[str]) -> dict[str, str]:
    """
    Load and merge multiple word list files.

    If a word has no category (plain text file), it gets tagged
    with the filename as its category. This way you can tell
    euphemisms from comparison words in the output.

    Example:
        euphemisms.txt contains "snow" → category = "euphemisms"
        comparison_words.txt contains "needle" → category = "comparison_words"
    """
    merged = {}
    for path in paths:
        # Use filename (without extension) as default category
        file_category = os.path.splitext(os.path.basename(path))[0]
        words = load_word_list(path)
        for word, category in words.items():
            if category == "unknown":
                merged[word] = file_category
            else:
                merged[word] = category
    logger.info(f"Total: {len(merged)} unique words from {len(paths)} files")
    return merged


# ============================================================================
# AHO-CORASICK MATCHER
# ============================================================================


class Matcher:
    """
    Fast multi-pattern string matcher using Aho-Corasick.

    Finds all occurrences of all target words in a single pass
    through the text. Word boundary checking ensures "pot" matches
    in "smoked pot yesterday" but not in "potato" or "spotless".
    """

    def __init__(self, words: dict[str, str]):
        """
        Args:
            words: dict mapping word -> category
        """
        self.words = words
        self._backend = None
        self._automaton = None
        self._build()

    def _build(self):
        try:
            import ahocorasick

            A = ahocorasick.Automaton()
            for word, category in self.words.items():
                A.add_word(word.lower(), (word, category))
            A.make_automaton()
            self._automaton = A
            self._backend = "pyahocorasick"
            logger.info(f"Matcher built ({self._backend}): {len(self.words)} patterns")
        except ImportError:
            logger.warning(
                "pyahocorasick not installed — using naive matching. "
                "This will be MUCH slower. Install: pip install pyahocorasick"
            )
            self._backend = "naive"

    def find(self, text: str) -> list[dict]:
        """
        Find all target word occurrences in text.

        Returns list of:
            {"word": str, "category": str, "start": int, "end": int}
        """
        text_lower = text.lower()
        matches = []

        if self._backend == "pyahocorasick":
            for end_idx, (word, category) in self._automaton.iter(text_lower):
                start = end_idx - len(word) + 1
                end = end_idx + 1
                if self._is_word_boundary(text_lower, start, end):
                    matches.append(
                        {
                            "word": word,
                            "category": category,
                            "start": start,
                            "end": end,
                        }
                    )
        else:
            for word, category in self.words.items():
                idx = 0
                word_lower = word.lower()
                while True:
                    pos = text_lower.find(word_lower, idx)
                    if pos == -1:
                        break
                    end = pos + len(word_lower)
                    if self._is_word_boundary(text_lower, pos, end):
                        matches.append(
                            {
                                "word": word,
                                "category": category,
                                "start": pos,
                                "end": end,
                            }
                        )
                    idx = pos + 1

        return matches

    @staticmethod
    def _is_word_boundary(text: str, start: int, end: int) -> bool:
        if start > 0 and text[start - 1].isalnum():
            return False
        if end < len(text) and text[end].isalnum():
            return False
        return True


# ============================================================================
# REDDIT STREAMING
# ============================================================================


def stream_reddit_file(
    path: str,
    start_year: int = 2015,
    end_year: int = 2026,
    subreddits: set[str] | None = None,
) -> Iterator[dict]:
    """
    Stream Reddit comments from a .zst dump file.

    Yields raw dicts with: text, timestamp, subreddit, permalink, source.
    Minimal processing — no language detection (too slow for full scan).
    """
    import zstandard as zstd

    count = 0
    yielded = 0

    # Try multiple decompression strategies because Reddit dumps
    # vary in how they were compressed across different time periods
    fh = open(path, "rb")

    try:
        # Strategy 1: stream_reader with large window (works for most files)
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        reader = dctx.stream_reader(fh, read_size=2**24)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

        for line in text_stream:
            result = _process_reddit_line(line, start_year, end_year, subreddits)
            count += 1
            if result is not None:
                yielded += 1
                yield result
            if count % 5_000_000 == 0:
                logger.info(f"  {path}: read {count:,}, yielded {yielded:,}")

        text_stream.close()

    except zstd.ZstdError as e:
        logger.warning(f"  stream_reader failed ({e}), trying decompressobj...")
        fh.close()

        # Strategy 2: decompressobj handles concatenated frames and
        # non-standard compression better
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        decompressor = dctx.decompressobj()

        leftover = ""
        count = 0
        yielded = 0

        while True:
            chunk = fh.read(2**24)  # 16 MB chunks
            if not chunk:
                break

            try:
                text = decompressor.decompress(chunk).decode("utf-8", errors="ignore")
            except zstd.ZstdError:
                # Frame boundary — create new decompressor and continue
                try:
                    decompressor = dctx.decompressobj()
                    text = decompressor.decompress(chunk).decode(
                        "utf-8", errors="ignore"
                    )
                except zstd.ZstdError:
                    logger.warning(f"  Skipping unreadable chunk at byte {fh.tell()}")
                    continue

            text = leftover + text
            lines = text.split("\n")
            leftover = lines[-1]  # Last line may be incomplete

            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                result = _process_reddit_line(line, start_year, end_year, subreddits)
                count += 1
                if result is not None:
                    yielded += 1
                    yield result
                if count % 5_000_000 == 0:
                    logger.info(f"  {path}: read {count:,}, yielded {yielded:,}")

        # Process any remaining leftover
        if leftover.strip():
            result = _process_reddit_line(
                leftover.strip(), start_year, end_year, subreddits
            )
            if result is not None:
                yielded += 1

        fh.close()

    finally:
        if not fh.closed:
            fh.close()

    logger.info(f"  {path}: done — {count:,} read, {yielded:,} yielded")


def _process_reddit_line(
    line: str,
    start_year: int,
    end_year: int,
    subreddits: set[str] | None,
) -> dict | None:
    """Process a single JSON line from a Reddit dump. Returns dict or None."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None

    # Subreddit filter
    if subreddits is not None:
        sr = obj.get("subreddit", "")
        if sr not in subreddits:
            return None

    # Get text
    body = obj.get("body", "") or obj.get("selftext", "")
    if not body or body in ("[deleted]", "[removed]"):
        return None

    # Timestamp and year filter
    ts_raw = obj.get("created_utc", "")
    try:
        ts_float = float(ts_raw)
        year = datetime.utcfromtimestamp(ts_float).year
        if year < start_year or year > end_year:
            return None
        timestamp = datetime.utcfromtimestamp(ts_float).isoformat() + "Z"
    except (ValueError, OverflowError, OSError):
        timestamp = str(ts_raw)

    # Minimal cleaning
    body = re.sub(r"https?://\S+", "", body)
    body = re.sub(r"\s+", " ", body).strip()

    if len(body.split()) < 3:
        return None

    return {
        "text": body,
        "timestamp": timestamp,
        "subreddit": obj.get("subreddit", ""),
        "permalink": obj.get("permalink", ""),
        "source": "reddit",
    }


# ============================================================================
# CSV STREAMING
# ============================================================================


def stream_csv_file(
    path: str,
    text_column: str | None = None,
    timestamp_column: str | None = None,
) -> Iterator[dict]:
    """
    Stream text records from a CSV file.

    Auto-detects text and timestamp columns if not specified.
    """
    import csv

    TEXT_CANDIDATES = [
        "text",
        "body",
        "content",
        "comment",
        "message",
        "post",
        "tweet",
        "selftext",
        "full_text",
    ]
    TS_CANDIDATES = [
        "timestamp",
        "date",
        "created_at",
        "created_utc",
        "published_at",
        "datetime",
    ]

    # Detect encoding
    try:
        fh = open(path, "r", encoding="utf-8", newline="")
        fh.readline()
        fh.seek(0)
    except UnicodeDecodeError:
        fh = open(path, "r", encoding="latin-1", newline="")

    reader = csv.DictReader(fh)
    columns = [c.strip().lower() for c in (reader.fieldnames or [])]
    col_map = {c.strip().lower(): c for c in (reader.fieldnames or [])}

    # Auto-detect columns
    if text_column is None:
        for candidate in TEXT_CANDIDATES:
            if candidate in columns:
                text_column = col_map[candidate]
                break
    if text_column is None:
        logger.error(
            f"Cannot detect text column in {path}. Columns: {reader.fieldnames}"
        )
        fh.close()
        return

    if timestamp_column is None:
        for candidate in TS_CANDIDATES:
            if candidate in columns:
                timestamp_column = col_map[candidate]
                break

    logger.info(f"CSV {path}: text={text_column}, timestamp={timestamp_column}")

    count = 0
    for row in reader:
        text = (row.get(text_column) or "").strip()
        if not text or len(text.split()) < 3:
            continue

        ts = ""
        if timestamp_column:
            ts = (row.get(timestamp_column) or "").strip()

        count += 1
        yield {
            "text": text,
            "timestamp": ts,
            "subreddit": "",
            "permalink": "",
            "source": os.path.basename(path),
        }

    fh.close()
    logger.info(f"CSV {path}: {count} records yielded")


# ============================================================================
# CORE: COLLECT ALL INSTANCES
# ============================================================================


def collect_instances(
    stream: Iterator[dict],
    matcher: Matcher,
    output_path: str,
    context_sentences: int = 2,
):
    """
    Stream through text records, find all matches, save to JSONL.

    Each output record contains:
        - word: the matched word
        - category: from the word list (e.g., "cocaine", "comparison")
        - sentence: the sentence containing the match
        - context: surrounding text (full comment/post, up to N sentences)
        - timestamp: from the source
        - subreddit: if from Reddit
        - source: data source identifier

    Note: We save the FULL comment/post text as context rather than
    trying to split into sentences and extract a window. Reddit
    comments are usually short enough that the full text IS the context.
    For longer texts, we truncate to ±context_sentences around the match.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total_records = 0
    total_matches = 0
    match_counts = {}  # word -> count

    with open(output_path, "w", encoding="utf-8") as out:
        for record in stream:
            total_records += 1
            text = record["text"]

            matches = matcher.find(text)
            if not matches:
                continue

            # Deduplicate matches — same word at same position
            seen = set()
            unique_matches = []
            for match in matches:
                key = (match["word"], match["start"])
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(match)

            for match in unique_matches:
                total_matches += 1
                word = match["word"]
                match_counts[word] = match_counts.get(word, 0) + 1

                output_record = {
                    "word": word,
                    "category": match["category"],
                    "sentence": text,  # Full comment as the sentence
                    "timestamp": record["timestamp"],
                    "subreddit": record.get("subreddit", ""),
                    "permalink": record.get("permalink", ""),
                    "source": record["source"],
                }

                out.write(json.dumps(output_record, ensure_ascii=False) + "\n")

            if total_records % 1_000_000 == 0:
                logger.info(
                    f"Processed {total_records:,} records | "
                    f"{total_matches:,} total matches | "
                    f"top words: {_top_n(match_counts, 5)}"
                )

    logger.info(
        f"\nDone: {total_records:,} records processed, "
        f"{total_matches:,} matches saved to {output_path}"
    )
    logger.info(f"Match counts per word:\n{_format_counts(match_counts)}")

    # Save summary stats
    stats_path = output_path.replace(".jsonl", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "total_records": total_records,
                "total_matches": total_matches,
                "unique_words_matched": len(match_counts),
                "match_counts": dict(sorted(match_counts.items(), key=lambda x: -x[1])),
            },
            f,
            indent=2,
        )
    logger.info(f"Stats saved to {stats_path}")


def _top_n(counts: dict, n: int) -> str:
    top = sorted(counts.items(), key=lambda x: -x[1])[:n]
    return ", ".join(f"{w}={c}" for w, c in top)


def _format_counts(counts: dict) -> str:
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
    lines = [f"  {word}: {count:,}" for word, count in sorted_counts[:50]]
    if len(sorted_counts) > 50:
        lines.append(f"  ... and {len(sorted_counts) - 50} more")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Collect all instances of target words from Reddit or CSV data"
    )

    # Word lists
    parser.add_argument(
        "--words",
        nargs="+",
        required=True,
        help="Path(s) to word list files (one word per line, or word<TAB>category)",
    )

    # Data sources (pick one or more)
    parser.add_argument("--reddit-file", help="Single Reddit .zst dump file")
    parser.add_argument("--reddit-dir", help="Directory of per-subreddit Reddit dumps")
    parser.add_argument(
        "--reddit-monthly-dir",
        help="Directory of monthly Reddit dumps (RC_YYYY-MM.zst)",
    )
    parser.add_argument("--csv-file", help="Single CSV file")
    parser.add_argument("--csv-dir", help="Directory of CSV files")

    # Filtering
    parser.add_argument(
        "--subreddits",
        help="Path to file listing subreddits to include (one per line). "
        "If not set, all subreddits are included.",
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)

    # Context
    parser.add_argument(
        "--context-sentences",
        type=int,
        default=2,
        help="Number of surrounding sentences to include",
    )

    # Output
    parser.add_argument("--output", required=True, help="Output JSONL path")

    # SLURM support
    parser.add_argument(
        "--slurm-task-id",
        type=int,
        default=None,
        help="SLURM_ARRAY_TASK_ID — processes one monthly dump file",
    )

    args = parser.parse_args()

    # Load word lists
    words = load_multiple_word_lists(args.words)
    matcher = Matcher(words)

    # Load subreddit filter
    subreddits = None
    if args.subreddits:
        with open(args.subreddits) as f:
            subreddits = {line.strip() for line in f if line.strip()}
        logger.info(f"Filtering to {len(subreddits)} subreddits")

    # Build data stream
    if args.reddit_file:
        stream = stream_reddit_file(
            args.reddit_file,
            start_year=args.start_year,
            end_year=args.end_year,
            subreddits=subreddits,
        )
        collect_instances(stream, matcher, args.output, args.context_sentences)

    elif args.reddit_dir:
        # Per-subreddit dumps: process each subreddit file
        zst_files = sorted(
            glob.glob(os.path.join(args.reddit_dir, "**/*.zst"), recursive=True)
        )
        if not zst_files:
            zst_files = sorted(glob.glob(os.path.join(args.reddit_dir, "*.zst")))
        logger.info(f"Found {len(zst_files)} .zst files in {args.reddit_dir}")

        for zst_file in zst_files:
            name = Path(zst_file).stem
            out_path = args.output.replace(".jsonl", f"_{name}.jsonl")
            logger.info(f"\nProcessing: {zst_file} -> {out_path}")

            stream = stream_reddit_file(
                zst_file,
                start_year=args.start_year,
                end_year=args.end_year,
                subreddits=subreddits,
            )
            collect_instances(stream, matcher, out_path, args.context_sentences)

    elif args.reddit_monthly_dir:
        dump_files = sorted(
            glob.glob(os.path.join(args.reddit_monthly_dir, "RC_*.zst"))
        )
        logger.info(f"Found {len(dump_files)} monthly dumps")

        # Group parts that belong to the same month
        # RC_2018-07.zst, RC_2018-07_part000.zst, RC_2018-07_part001.zst
        # all belong to "RC_2018-07"
        from collections import defaultdict

        month_groups = defaultdict(list)
        for f in dump_files:
            stem = Path(f).stem  # e.g. "RC_2018-07_part000"
            # Extract base month: take first 10 chars "RC_YYYY-MM"
            base = stem[:10] if len(stem) >= 10 else stem
            month_groups[base].append(f)

        logger.info(f"Grouped into {len(month_groups)} months")

        # SLURM array mode: process one month (all its parts)
        if args.slurm_task_id is not None:
            month_keys = sorted(month_groups.keys())
            if args.slurm_task_id >= len(month_keys):
                logger.info(
                    f"Task {args.slurm_task_id} >= {len(month_keys)} months, nothing to do"
                )
                return

            month_key = month_keys[args.slurm_task_id]
            parts = month_groups[month_key]
            out_path = os.path.join(args.output, f"{month_key}_matches.jsonl")
            logger.info(
                f"SLURM task {args.slurm_task_id}: {month_key} ({len(parts)} files)"
            )

            def stream_all_parts():
                for part in sorted(parts):
                    logger.info(f"  Streaming: {Path(part).name}")
                    yield from stream_reddit_file(
                        part,
                        start_year=args.start_year,
                        end_year=args.end_year,
                        subreddits=subreddits,
                    )

            collect_instances(
                stream_all_parts(), matcher, out_path, args.context_sentences
            )
        else:
            # Sequential mode: process all months
            for month_key in sorted(month_groups.keys()):
                parts = month_groups[month_key]
                out_path = os.path.join(args.output, f"{month_key}_matches.jsonl")

                if os.path.exists(out_path):
                    logger.info(f"Skipping {month_key} — already exists")
                    continue

                logger.info(f"\nProcessing: {month_key} ({len(parts)} files)")

                def stream_all_parts(parts=parts):
                    for part in sorted(parts):
                        logger.info(f"  Streaming: {Path(part).name}")
                        yield from stream_reddit_file(
                            part,
                            start_year=args.start_year,
                            end_year=args.end_year,
                            subreddits=subreddits,
                        )

                collect_instances(
                    stream_all_parts(), matcher, out_path, args.context_sentences
                )

    elif args.csv_file:
        stream = stream_csv_file(args.csv_file)
        collect_instances(stream, matcher, args.output, args.context_sentences)

    elif args.csv_dir:
        csv_files = sorted(
            glob.glob(os.path.join(args.csv_dir, "*.csv"))
            + glob.glob(os.path.join(args.csv_dir, "*.tsv"))
        )
        logger.info(f"Found {len(csv_files)} CSV/TSV files")

        for csv_file in csv_files:
            name = Path(csv_file).stem
            out_path = args.output.replace(".jsonl", f"_{name}.jsonl")
            logger.info(f"\nProcessing: {csv_file}")
            stream = stream_csv_file(csv_file)
            collect_instances(stream, matcher, out_path, args.context_sentences)

    else:
        parser.error(
            "Provide at least one data source: --reddit-file, --reddit-dir, "
            "--reddit-monthly-dir, --csv-file, or --csv-dir"
        )


if __name__ == "__main__":
    main()
