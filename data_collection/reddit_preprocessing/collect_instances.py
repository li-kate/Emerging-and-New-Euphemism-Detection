import argparse
import glob
import io
import json
import logging
import os
import re
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")

# ----------------------------------------------------------------------------
# JSON backend: prefer orjson for hot-path speed, fall back to stdlib json.
# ----------------------------------------------------------------------------
try:
    import orjson

    def _json_loads(s: str):
        return orjson.loads(s)

    def _json_dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")

    logger.info("Using orjson for JSON parsing/serialization")
except ImportError:
    logger.warning("orjson not installed — falling back to stdlib json (slower). "
                    "Install with: pip install orjson")

    def _json_loads(s: str):
        return json.loads(s)

    def _json_dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)


# ============================================================================
# WORD LIST LOADING
# ============================================================================


def load_word_lists(paths: list[str]) -> dict[str, str]:
    """
    Load and merge word list files (one word per line).

    Each word is tagged with its source filename (without extension) as
    its category, so you can distinguish euphemisms from comparison
    words in the output — e.g. words in euphemisms.txt get category
    "euphemisms".

    NOTE: if the same word appears in multiple files, the later file's
    category silently wins. Word lists are reviewed manually in this
    workflow, so no collision warning is emitted.
    """
    merged = {}
    for path in paths:
        category = os.path.splitext(os.path.basename(path))[0]
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                merged[line.lower()] = category
                count += 1
        logger.info(f"Loaded {count} words from {path}")

    logger.info(f"Total: {len(merged)} unique words from {len(paths)} files")
    return merged


# ============================================================================
# AHO-CORASICK MATCHER
# ============================================================================


class Matcher:
    """Fast multi-pattern string matcher using Aho-Corasick."""

    def __init__(self, words: dict[str, str]):
        self.words = words
        try:
            import ahocorasick
        except ImportError as exc:
            raise RuntimeError(
                "pyahocorasick is required. Install it with: pip install pyahocorasick"
            ) from exc

        automaton = ahocorasick.Automaton()
        for word, category in words.items():
            automaton.add_word(word.lower(), (word, category))
        automaton.make_automaton()
        self._automaton = automaton
        logger.info("Matcher built (pyahocorasick): %d patterns", len(words))

    def find(self, text: str) -> list[dict]:
        """Find all target word occurrences in text."""
        text_lower = text.lower()
        matches = []

        for end_idx, (word, category) in self._automaton.iter(text_lower):
            start = end_idx - len(word) + 1
            end = end_idx + 1
            if self._is_word_boundary(text_lower, start, end):
                matches.append({
                    "word": word,
                    "category": category,
                    "start": start,
                    "end": end,
                })

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
        # NOTE: this reopens the file from byte 0. Any records already
        # yielded above by strategy 1 will be reprocessed and re-yielded
        # by strategy 2 below — known limitation, not yet deduped upstream.
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

        # Process any remaining leftover (FIXED: this used to increment
        # `yielded` without actually yielding the record, silently
        # dropping the final line of the file).
        if leftover.strip():
            result = _process_reddit_line(
                leftover.strip(), start_year, end_year, subreddits
            )
            count += 1
            if result is not None:
                yielded += 1
                yield result

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
        obj = _json_loads(line)
    except ValueError:
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
    ts_raw = obj.get("created_utc")
    try:
        timestamp_dt = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None

    if not start_year <= timestamp_dt.year <= end_year:
        return None

    timestamp = timestamp_dt.isoformat().replace("+00:00", "Z")

    # Minimal cleaning
    body = URL_RE.sub("", body)
    body = WHITESPACE_RE.sub(" ", body).strip()

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
        - timestamp: from the source
        - subreddit: if from Reddit
        - source: data source identifier

    NOTE: `context_sentences` is currently accepted but not applied —
    the full comment/post text is always written to `sentence` as-is,
    with no sentence-level windowing. See discussion for a proposed
    implementation if tighter context windows are needed.
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

            # Deduplicate matches while preserving output behavior.
            seen = set()
            for match in matches:
                key = (match["word"], match["start"])
                if key in seen:
                    continue
                seen.add(key)
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

                out.write(_json_dumps(output_record) + "\n")

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

    # Save summary stats (stdlib json here — not a hot path, and indent
    # formatting for human readability matters more than speed).
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
# PARALLEL WORKER (module-level so it's picklable for ProcessPoolExecutor)
# ============================================================================


def _process_source_job(job: dict):
    """
    Process one independent unit of work (a single file, or all parts
    of a single month) in isolation. Safe to run in a worker process.

    Rebuilds its own Matcher from the shared `words` dict rather than
    receiving a pre-built Matcher, since the pyahocorasick automaton
    is a C-extension object and isn't reliably picklable across
    process boundaries.

    Any exception is caught and logged here so that one bad/corrupt
    dump file doesn't take down an entire parallel batch.
    """
    label = job.get("label", job["output_path"])
    try:
        if os.path.exists(job["output_path"]):
            logger.info(f"Skipping {label} — output already exists")
            return

        matcher = Matcher(job["words"])

        def stream_all():
            for part in job["parts"]:
                logger.info(f"  [{label}] streaming: {Path(part).name}")
                yield from stream_reddit_file(
                    part,
                    start_year=job["start_year"],
                    end_year=job["end_year"],
                    subreddits=job["subreddits"],
                )

        logger.info(f"Processing: {label} -> {job['output_path']}")
        collect_instances(
            stream_all(), matcher, job["output_path"], job["context_sentences"]
        )

    except Exception:
        logger.exception(f"Failed processing {label} — skipping and continuing")


def _run_jobs(jobs: list[dict], workers: int):
    """Run a list of independent jobs, in parallel if workers > 1."""
    if workers <= 1:
        for job in jobs:
            _process_source_job(job)
        return

    logger.info(f"Running {len(jobs)} jobs with {workers} parallel workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_source_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            label = job.get("label", job["output_path"])
            try:
                future.result()
            except Exception:
                # _process_source_job already catches/logs internally,
                # but guard here too in case of a process-level crash.
                logger.exception(f"Worker crashed while processing {label}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Collect all instances of target words from Reddit data"
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

    # Parallelism (used for --reddit-dir and non-SLURM --reddit-monthly-dir)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for independent "
        "files/months. Ignored in --slurm-task-id mode, since SLURM's "
        "job array already parallelizes at the cluster level.",
    )

    args = parser.parse_args()

    # Load word lists
    words = load_word_lists(args.words)

    # Load subreddit filter
    subreddits = None
    if args.subreddits:
        with open(args.subreddits) as f:
            subreddits = {line.strip() for line in f if line.strip()}
        logger.info(f"Filtering to {len(subreddits)} subreddits")

    common = dict(
        words=words,
        start_year=args.start_year,
        end_year=args.end_year,
        subreddits=subreddits,
        context_sentences=args.context_sentences,
    )

    # Build data stream
    if args.reddit_file:
        job = {
            **common,
            "parts": [args.reddit_file],
            "output_path": args.output,
            "label": args.reddit_file,
        }
        _process_source_job(job)

    elif args.reddit_dir:
        # Per-subreddit dumps: process each subreddit file
        zst_files = sorted(
            glob.glob(os.path.join(args.reddit_dir, "**/*.zst"), recursive=True)
        )
        if not zst_files:
            zst_files = sorted(glob.glob(os.path.join(args.reddit_dir, "*.zst")))
        logger.info(f"Found {len(zst_files)} .zst files in {args.reddit_dir}")

        jobs = []
        for zst_file in zst_files:
            name = Path(zst_file).stem
            out_path = args.output.replace(".jsonl", f"_{name}.jsonl")
            jobs.append({
                **common,
                "parts": [zst_file],
                "output_path": out_path,
                "label": zst_file,
            })

        _run_jobs(jobs, args.workers)

    elif args.reddit_monthly_dir:
        dump_files = sorted(
            glob.glob(os.path.join(args.reddit_monthly_dir, "RC_*.zst"))
        )
        logger.info(f"Found {len(dump_files)} monthly dumps")

        # Group parts that belong to the same month
        # RC_2018-07.zst, RC_2018-07_part000.zst, RC_2018-07_part001.zst
        # all belong to "RC_2018-07"
        month_groups = defaultdict(list)
        for f in dump_files:
            stem = Path(f).stem  # e.g. "RC_2018-07_part000"
            # Extract base month: take first 10 chars "RC_YYYY-MM"
            base = stem[:10] if len(stem) >= 10 else stem
            month_groups[base].append(f)

        logger.info(f"Grouped into {len(month_groups)} months")

        # SLURM array mode: process one month (all its parts).
        # Parallelism is provided by the job array itself, so --workers
        # is not used here.
        if args.slurm_task_id is not None:
            month_keys = sorted(month_groups.keys())
            if args.slurm_task_id >= len(month_keys):
                logger.info(
                    f"Task {args.slurm_task_id} >= {len(month_keys)} months, nothing to do"
                )
                return

            month_key = month_keys[args.slurm_task_id]
            parts = sorted(month_groups[month_key])
            out_path = os.path.join(args.output, f"{month_key}_matches.jsonl")
            job = {
                **common,
                "parts": parts,
                "output_path": out_path,
                "label": f"SLURM task {args.slurm_task_id}: {month_key}",
            }
            _process_source_job(job)
        else:
            # Sequential/parallel mode: process all months
            jobs = []
            for month_key in sorted(month_groups.keys()):
                parts = sorted(month_groups[month_key])
                out_path = os.path.join(args.output, f"{month_key}_matches.jsonl")
                jobs.append({
                    **common,
                    "parts": parts,
                    "output_path": out_path,
                    "label": month_key,
                })

            _run_jobs(jobs, args.workers)

    else:
        parser.error(
            "Provide at least one data source: --reddit-file, --reddit-dir, "
            "--reddit-monthly-dir"
        )


if __name__ == "__main__":
    main()