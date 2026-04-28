"""
============================================================================
DATA SOURCES — Streaming Iterators for Euphemism Detection Pipeline
============================================================================

Each source is a generator that yields standardized TextRecord objects.
The main pipeline consumes these uniformly regardless of origin.

Filtering:
    - Language: English-only by default, using fasttext-langdetect for speed.
      Why fasttext over langdetect or spacy?
        * fasttext: ~1M sentences/sec, 99%+ accuracy on sentences >10 words.
          Single model file (~1MB compressed). Best speed/accuracy tradeoff.
        * langdetect (Google): ~10K sentences/sec, good accuracy but 100x slower.
          Fine for small corpora, unusable at Common Crawl scale.
        * spaCy: Requires loading a full NLP pipeline just for lang ID.
          Overkill and slow for this single task.
      The trade-off: fasttext needs a model download on first run (~1MB).
      We handle this automatically.

    - Temporal: Each source filters by a (start_year, end_year) range.
      Timestamps come from different sources with different semantics:
        * Common Crawl: WARC-Date is the CRAWL date, not publication date.
          A page written in 2016 but crawled in 2024 gets a 2024 timestamp.
          This is a known limitation — note it in your paper. For temporal
          analysis, prefer sources with real publication dates.
        * Wikipedia: Revision timestamps are real edit dates. Reliable.
        * Reddit: created_utc is the post date. Reliable.
        * Twitter: created_at is the tweet date. Reliable.
        * News: published_at is the article date. Reliable.

Adding a new source:
    1. Write a generator function that yields TextRecord objects
    2. Apply the shared filter_record() function before yielding
    3. Import it in the main pipeline
============================================================================
"""

import re
import io
import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# STANDARDIZED RECORD
# ============================================================================

@dataclass
class TextRecord:
    """Standardized record from any source."""
    text: str
    timestamp: str
    source_url: str = ""
    source: str = ""


# ============================================================================
# FILTERING CONFIGURATION
# ============================================================================

@dataclass
class SourceConfig:
    """
    Shared filtering config applied to ALL sources uniformly.

    Centralizing this ensures consistent behavior — you don't want
    one source filtering to 2015-2025 and another to 2016-2024.
    """
    # --- Language ---
    # Target language(s). Set to None to disable language filtering.
    # Uses ISO 639-1 codes: "en" for English, "fr" for French, etc.
    target_languages: Optional[set[str]] = None

    # Minimum confidence for language detection (0.0 - 1.0).
    # 0.7 is conservative — rejects ambiguous short texts.
    # 0.5 is permissive — accepts more but risks non-English leaking through.
    lang_confidence_threshold: float = 0.7

    # --- Temporal ---
    # Inclusive year range. Set either to None to disable that bound.
    start_year: Optional[int] = None
    end_year: Optional[int] = None

    # --- Text length ---
    min_words: int = 5
    max_words: int = 200

    def __post_init__(self):
        # Default to English, 2015-present
        if self.target_languages is None:
            self.target_languages = {"en"}
        if self.start_year is None:
            self.start_year = 2015
        if self.end_year is None:
            self.end_year = datetime.now().year


# Default config instance — import this or create your own
DEFAULT_SOURCE_CONFIG = SourceConfig()


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================
#
# We use fasttext's compressed language ID model (lid.176.ftz, ~1MB).
# It supports 176 languages and runs at ~1M predictions/sec on CPU.
#
# Alternative considered: the `langdetect` library (Google's port).
#   Pros: No model download needed, pure Python.
#   Cons: ~100x slower. At Common Crawl scale (billions of sentences),
#         this is the difference between hours and weeks.
#
# The fasttext model is downloaded once and cached. If you're in an
# environment without internet access, pre-download the model and set
# the path via the FASTTEXT_LANGID_MODEL env var.
#
# Required: pip install fasttext-langdetect
#           (or: pip install fasttext, then download model manually)
# ============================================================================

_lang_model = None


def _get_lang_model():
    """Lazy-load the fasttext language ID model."""
    global _lang_model
    if _lang_model is not None:
        return _lang_model

    try:
        import fasttext
        import os
        import urllib.request

        # Check for pre-downloaded model
        model_path = os.environ.get(
            "FASTTEXT_LANGID_MODEL",
            os.path.expanduser("~/.cache/fasttext/lid.176.ftz"),
        )

        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
            logger.info(f"Downloading fasttext language ID model to {model_path}...")
            urllib.request.urlretrieve(url, model_path)
            logger.info("Download complete.")

        # Suppress fasttext's noisy warning about deprecated loading method
        _lang_model = fasttext.load_model(model_path)
        return _lang_model

    except ImportError:
        logger.warning(
            "fasttext not installed. Language filtering disabled. "
            "Install with: pip install fasttext"
        )
        return None


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect the language of a text string.

    Returns:
        (language_code, confidence) e.g. ("en", 0.95)
        Returns ("unknown", 0.0) if detection fails.
    """
    model = _get_lang_model()
    if model is None:
        return ("unknown", 0.0)

    # fasttext expects single-line input with no newlines
    clean = text.replace("\n", " ").strip()
    if not clean:
        return ("unknown", 0.0)

    predictions = model.predict(clean, k=1)
    # predictions = (('__label__en',), array([0.95]))
    label = predictions[0][0].replace("__label__", "")
    confidence = float(predictions[1][0])

    return (label, confidence)


# ============================================================================
# TIMESTAMP PARSING
# ============================================================================
#
# Different sources use different timestamp formats. We normalize them all
# to extract a year for temporal filtering. The raw timestamp string is
# preserved in TextRecord for the pipeline's temporal analysis.
#
# Formats we handle:
#   - ISO 8601:   "2024-03-15T10:30:00Z" (Common Crawl, Wikipedia)
#   - Unix epoch: "1710489000" (Reddit created_utc)
#   - Twitter:    "2024-03-15T10:30:00.000Z" or "Fri Mar 15 10:30:00 +0000 2024"
#   - Date only:  "2024-03-15" (news dumps)
#   - Fallback:   extract any 4-digit year with regex
# ============================================================================

def extract_year(timestamp: str) -> Optional[int]:
    """
    Extract the year from a timestamp string in any common format.

    Returns None if parsing fails entirely, which means the record
    will be INCLUDED (we err on the side of inclusion when we can't
    determine the date, rather than silently dropping data).
    """
    if not timestamp:
        return None

    timestamp = timestamp.strip()

    # 1. Try Unix epoch (Reddit: "1710489000" or "1710489000.0")
    try:
        epoch = float(timestamp)
        # Sanity: Unix timestamps for 2000-2030 are roughly 9.46e8 to 1.89e9
        if 9e8 < epoch < 2e9:
            return datetime.utcfromtimestamp(epoch).year
    except (ValueError, OverflowError, OSError):
        pass

    # 2. Try ISO 8601 variants
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",       # Common Crawl, Wikipedia
        "%Y-%m-%dT%H:%M:%S.%fZ",    # Twitter v2
        "%Y-%m-%dT%H:%M:%S",        # No timezone
        "%Y-%m-%d",                  # Date only
    ):
        try:
            return datetime.strptime(timestamp, fmt).year
        except ValueError:
            continue

    # 3. Try Twitter v1 format: "Fri Mar 15 10:30:00 +0000 2024"
    try:
        dt = datetime.strptime(timestamp, "%a %b %d %H:%M:%S %z %Y")
        return dt.year
    except ValueError:
        pass

    # 4. Fallback: regex for any 4-digit year between 1990 and 2030
    match = re.search(r"\b(199\d|20[0-2]\d|2030)\b", timestamp)
    if match:
        return int(match.group(1))

    return None


# ============================================================================
# SHARED FILTERING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Minimal cleaning. Preserves casing and punctuation because the
    sentence embedder was trained on natural text.
    """
    text = re.sub(r"https?://\S+", "", text)       # URLs
    text = re.sub(r"@\w+", "", text)                # @handles
    text = re.sub(r"<[^>]+>", "", text)              # residual HTML
    text = re.sub(r"\s+", " ", text).strip()         # whitespace
    return text


def filter_record(
    text: str,
    timestamp: str,
    config: SourceConfig,
) -> bool:
    """
    Apply all shared filters to a candidate record.

    Returns True if the record should be INCLUDED, False if filtered out.

    Filter order is deliberate — cheapest checks first:
        1. Length (free: just count spaces)
        2. Temporal (cheap: parse year from timestamp)
        3. Language (expensive: runs fasttext inference)
    """
    # 1. Length filter
    word_count = len(text.split())
    if not (config.min_words < word_count < config.max_words):
        return False

    # 2. Temporal filter
    if config.start_year is not None or config.end_year is not None:
        year = extract_year(timestamp)
        if year is not None:
            if config.start_year is not None and year < config.start_year:
                return False
            if config.end_year is not None and year > config.end_year:
                return False
        # If year is None (unparseable), we INCLUDE the record.
        # Rationale: dropping records with bad timestamps silently loses data.
        # Better to include and let downstream analysis handle the missing date.

    # 3. Language filter (most expensive — run last)
    if config.target_languages:
        lang, confidence = detect_language(text)
        if lang not in config.target_languages:
            return False
        if confidence < config.lang_confidence_threshold:
            return False

    return True


# ============================================================================
# UTILITY: BATCH RECORDS
# ============================================================================

def batch_records(
    stream: Iterator[TextRecord],
    batch_size: int,
) -> Iterator[list[TextRecord]]:
    """
    Collect records into batches for efficient GPU encoding.

    256 is a good default for all-MiniLM-L6-v2 on a 16GB GPU.
    """
    batch = []
    for record in stream:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ============================================================================
# COMMON CRAWL (WET FILES)
# ============================================================================
#
# Temporal caveat: WARC-Date is the CRAWL date, not the publication date.
# A page published in 2016 but crawled in 2024 will have timestamp 2024.
# This means temporal filtering on Common Crawl data is approximate:
#   - Filtering to 2015+ crawls doesn't guarantee 2015+ content.
#   - A 2024 crawl might include pages unchanged since 2010.
# For precise temporal analysis, prefer Reddit/Twitter/News sources.
# Common Crawl is best used for broad coverage, not temporal precision.
#
# For multi-year studies, use WET files from DIFFERENT crawl epochs:
#   CC-MAIN-2015-06, CC-MAIN-2017-04, CC-MAIN-2019-09, CC-MAIN-2022-05, etc.
#   This gives you a coarse temporal signal (content popular in that era).
#
# WET path index: https://data.commoncrawl.org/crawl-data/CC-MAIN-{YYYY-WW}/wet.paths.gz
# ============================================================================

def stream_common_crawl_wet(
    wet_url: str,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
) -> Iterator[TextRecord]:
    """
    Stream pre-extracted text from a Common Crawl WET file.

    Args:
        wet_url: Full URL to a WET file on data.commoncrawl.org
        config:  Filtering configuration
    """
    import requests
    from warcio.archiveiterator import ArchiveIterator

    try:
        response = requests.get(wet_url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {wet_url}: {e}")
        return

    for record in ArchiveIterator(response.raw):
        if record.rec_type == "conversion":
            uri = record.rec_headers.get_header("WARC-Target-URI") or ""
            timestamp = record.rec_headers.get_header("WARC-Date") or ""

            content = record.content_stream().read().decode("utf-8", errors="ignore")
            if not content:
                continue

            for sentence in re.split(r"(?<=[.!?])\s+", content):
                sentence = clean_text(sentence)
                if not sentence:
                    continue

                if filter_record(sentence, timestamp, config):
                    yield TextRecord(
                        text=sentence,
                        timestamp=timestamp,
                        source_url=uri,
                        source="Common Crawl",
                    )


# ============================================================================
# WIKIPEDIA
# ============================================================================
#
# Temporal note: Revision timestamps are real edit dates, making Wikipedia
# one of the most reliable sources for temporal analysis. However, most
# Wikipedia content is continuously edited, so a sentence present in a 2020
# revision might have been written years earlier. The timestamp tells you
# "this text existed in this form at this date," not "this text was first
# written at this date."
#
# For detecting EMERGING euphemisms over time, Wikipedia's Talk pages and
# edit histories are more useful than article snapshots, since they capture
# real-time community discussion about language choices.
#
# Download: https://dumps.wikimedia.org/enwiki/latest/
#           enwiki-latest-pages-articles.xml.bz2 (~22GB)
#
# Required: pip install mwxml mwparserfromhell
# ============================================================================

def stream_wikipedia_dump(
    dump_path: str,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
) -> Iterator[TextRecord]:
    """
    Stream sentences from a Wikipedia XML dump.

    Args:
        dump_path: Path to enwiki-latest-pages-articles.xml.bz2
        config:    Filtering configuration
    """
    import bz2
    try:
        import mwxml
        import mwparserfromhell
    except ImportError:
        logger.error("pip install mwxml mwparserfromhell")
        return

    dump = mwxml.Dump.from_file(bz2.open(dump_path, "rt", encoding="utf-8"))

    for page in dump:
        for revision in page:
            if revision.text is None:
                continue

            timestamp = str(revision.timestamp) if revision.timestamp else ""

            # Quick temporal pre-filter BEFORE expensive wikitext parsing.
            # If the revision is outside our year range, skip parsing entirely.
            if config.start_year or config.end_year:
                year = extract_year(timestamp)
                if year is not None:
                    if config.start_year and year < config.start_year:
                        continue
                    if config.end_year and year > config.end_year:
                        continue

            try:
                wikicode = mwparserfromhell.parse(revision.text)
                plaintext = wikicode.strip_code()
            except Exception:
                continue

            for line in plaintext.split("\n"):
                line = clean_text(line)
                if not line:
                    continue

                if filter_record(line, timestamp, config):
                    yield TextRecord(
                        text=line,
                        timestamp=timestamp,
                        source_url=f"https://en.wikipedia.org/wiki/{page.title}",
                        source="Wikipedia",
                    )


# ============================================================================
# REDDIT (PUSHSHIFT / ACADEMIC TORRENTS)
# ============================================================================
#
# Temporal note: created_utc is the real post date (Unix epoch seconds).
# Very reliable for temporal analysis. Reddit is arguably the best source
# for tracking colloquial euphemism emergence because:
#   1. High volume of informal language
#   2. Real timestamps with second-level precision
#   3. Subreddit metadata lets you track euphemisms by community
#   4. Comments are threaded (conversational context)
#
# Data access:
#   Academic Torrents: https://academictorrents.com/ (search "reddit")
#   Files: RC_YYYY-MM.zst (comments), RS_YYYY-MM.zst (submissions)
#   Monthly files range from ~1GB (2015) to ~20GB (2023) compressed.
#
# Temporal strategy: Download one file per month for your target range.
#   RC_2015-01.zst through RC_2025-01.zst gives you full coverage.
#   Process each file separately to avoid OOM.
#
# Required for .zst: pip install zstandard
# ============================================================================

def stream_reddit_dump(
    dump_path: str,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
    subreddits: Optional[set[str]] = None,
) -> Iterator[TextRecord]:
    """
    Stream from a Reddit comment dump.

    Args:
        dump_path:  Path to RC_YYYY-MM.zst or .jsonl
        config:     Filtering configuration
        subreddits: Optional whitelist of subreddit names.
                    If None, all subreddits included.
    """
    import zstandard as zstd

    if dump_path.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        fh = open(dump_path, "rb")
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
    else:
        fh = None
        text_stream = open(dump_path, "r", encoding="utf-8", errors="ignore")

    try:
        for line_str in text_stream:
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            # Subreddit filter (before any expensive processing)
            if subreddits and obj.get("subreddit", "") not in subreddits:
                continue

            body = obj.get("body", "")
            if not body or body in ("[deleted]", "[removed]"):
                continue

            # Reddit stores created_utc as int/float epoch
            timestamp = str(obj.get("created_utc", ""))

            # Quick temporal pre-filter before cleaning and language detection
            if config.start_year or config.end_year:
                year = extract_year(timestamp)
                if year is not None:
                    if config.start_year and year < config.start_year:
                        continue
                    if config.end_year and year > config.end_year:
                        continue

            body = clean_text(body)

            for sentence in re.split(r"(?<=[.!?])\s+", body):
                sentence = sentence.strip()
                if not sentence:
                    continue

                if filter_record(sentence, timestamp, config):
                    permalink = obj.get("permalink", "")
                    yield TextRecord(
                        text=sentence,
                        timestamp=timestamp,
                        source_url=f"https://reddit.com{permalink}" if permalink else "",
                        source="Reddit",
                    )
    finally:
        text_stream.close()
        if fh is not None:
            fh.close()


# ============================================================================
# TWITTER / X
# ============================================================================
#
# Temporal note: created_at provides exact tweet timestamps.
# Twitter is valuable for tracking euphemisms that emerge in real-time
# discourse (political euphemisms, viral slang, coded language).
#
# Access is limited post-2023. Existing academic dumps are your best bet.
# Twitter API v2 academic archive search (if accessible) lets you query
# historical tweets by date range directly.
#
# Data format: JSONL with Twitter API v2 schema
# ============================================================================

def stream_twitter_dump(
    dump_path: str,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
) -> Iterator[TextRecord]:
    """
    Stream from a Twitter/X academic API export (JSONL).

    Args:
        dump_path: Path to JSONL file of tweet objects
        config:    Filtering configuration
    """
    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_str in f:
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "")
            if not text:
                continue

            timestamp = obj.get("created_at", "")

            # Temporal pre-filter
            if config.start_year or config.end_year:
                year = extract_year(timestamp)
                if year is not None:
                    if config.start_year and year < config.start_year:
                        continue
                    if config.end_year and year > config.end_year:
                        continue

            text = clean_text(text)

            # Tweets are short — use lower min_words
            tweet_config = SourceConfig(
                target_languages=config.target_languages,
                lang_confidence_threshold=config.lang_confidence_threshold,
                start_year=config.start_year,
                end_year=config.end_year,
                min_words=3,  # Tweets are naturally short
                max_words=config.max_words,
            )

            if filter_record(text, timestamp, tweet_config):
                tweet_id = obj.get("id", "")
                yield TextRecord(
                    text=text,
                    timestamp=timestamp,
                    source_url=f"https://twitter.com/i/status/{tweet_id}" if tweet_id else "",
                    source="Twitter",
                )


# ============================================================================
# NEWS ARTICLES
# ============================================================================
#
# Temporal note: published_at is the article publication date. Reliable.
# News sources are particularly valuable for institutional/political
# euphemisms: "enhanced interrogation", "collateral damage", "restructuring",
# "right-sizing", etc.
#
# Data format: JSONL with fields {title, content/body, published_at/date, url}
# Adapt field names to match your specific dump format.
# ============================================================================

def stream_news_dump(
    dump_path: str,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
) -> Iterator[TextRecord]:
    """
    Stream from a news article dump (JSONL).

    Args:
        dump_path: Path to JSONL file of news articles
        config:    Filtering configuration
    """
    with open(dump_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_str in f:
            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            content = obj.get("content", "") or obj.get("body", "")
            if not content:
                continue

            timestamp = obj.get("published_at", obj.get("date", ""))

            # Temporal pre-filter before sentence splitting and lang detection
            if config.start_year or config.end_year:
                year = extract_year(timestamp)
                if year is not None:
                    if config.start_year and year < config.start_year:
                        continue
                    if config.end_year and year > config.end_year:
                        continue

            content = clean_text(content)

            for sentence in re.split(r"(?<=[.!?])\s+", content):
                sentence = sentence.strip()
                if not sentence:
                    continue

                if filter_record(sentence, timestamp, config):
                    yield TextRecord(
                        text=sentence,
                        timestamp=timestamp,
                        source_url=obj.get("url", ""),
                        source="News",
                    )


# ============================================================================
# GENERIC LOCAL FILE (for testing)
# ============================================================================

# ============================================================================
# CSV / TSV FILES
# ============================================================================
#
# Why this source matters:
#   Many pre-collected datasets come as CSVs: API exports from Twitter/Reddit,
#   annotated corpora from prior research, scraped data saved to tabular format,
#   exported Slack/Discord logs, survey responses, etc.
#
#   The challenge is that CSV schemas vary wildly. Some have a "text" column,
#   others have "body", "content", "snippet", "comment", "message", etc.
#   Same for timestamps: "timestamp", "date", "created_at", "time",
#   "published_at", "created_utc", etc.
#
#   Rather than hardcoding column names, we accept explicit column names
#   and also auto-detect common ones as a fallback.
#
# Multi-sentence handling:
#   A CSV cell might contain a full paragraph or multiple sentences.
#   We split on sentence boundaries (like other sources) so each TextRecord
#   is one sentence. The original cell text is not preserved — if you need
#   it, use the source_url field (which stores the row index).
#
# Encoding:
#   CSVs from the wild are often not UTF-8. We try UTF-8 first, then fall
#   back to latin-1 (which never fails but may garble non-Latin characters).
#   For known encodings, pass encoding= explicitly.
# ============================================================================

# Common column name patterns for auto-detection
_TEXT_COLUMN_NAMES = [
    "text", "body", "content", "snippet", "comment", "message",
    "post", "tweet", "review", "description", "sentence", "utterance",
    "response", "question", "title_and_body", "selftext", "full_text",
    "plain_text"
]

_TIMESTAMP_COLUMN_NAMES = [
    "timestamp", "date", "time", "created_at", "created_utc",
    "published_at", "datetime", "post_date", "created", "updated_at",
    "published_date", "publication_date", "posted_at", "sent_at",
]

_SOURCE_COLUMN_NAMES = [
    "url", "source_url", "link", "permalink", "source", "uri",
]


def _detect_column(
    columns: list[str],
    candidates: list[str],
    label: str,
) -> Optional[str]:
    """
    Auto-detect a column by matching against known common names.
    Case-insensitive, strips whitespace.

    Returns the matched column name or None.
    """
    col_map = {c.strip().lower(): c for c in columns}

    # Exact match first
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]

    # Substring match as fallback (e.g., "comment_text" contains "text")
    for candidate in candidates:
        for col_lower, col_original in col_map.items():
            if candidate.lower() in col_lower:
                return col_original

    return None


def stream_csv(
    path: str,
    text_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    source_url_column: Optional[str] = None,
    source_name: Optional[str] = None,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8",
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
    task_id: int = 0,
    num_tasks: int = 1,
) -> Iterator[TextRecord]:
    """
    Stream sentences from a CSV or TSV file.

    Column detection priority:
        1. Explicit column name passed as argument (highest priority)
        2. Auto-detection from common column name patterns
        3. For text: if only one non-numeric column exists, use it
        4. Raise an error if text column can't be determined

    Row-level sharding (for SLURM arrays):
        When num_tasks > 1, each task processes every num_tasks-th row
        starting at task_id. This lets you parallelize a single large CSV
        across multiple SLURM array tasks.

        Example with num_tasks=4:
            task 0: rows 0, 4, 8, 12, ...
            task 1: rows 1, 5, 9, 13, ...
            task 2: rows 2, 6, 10, 14, ...
            task 3: rows 3, 7, 11, 15, ...

    Args:
        path:               Path to CSV/TSV file
        text_column:        Name of the text/content column. Auto-detected if None.
        timestamp_column:   Name of the timestamp column. Auto-detected if None.
        source_url_column:  Name of the URL/source column. Auto-detected if None.
        source_name:        Label for the source field. Defaults to filename.
        delimiter:          CSV delimiter. Auto-detected if None.
        encoding:           File encoding. Defaults to UTF-8 with latin-1 fallback.
        config:             Filtering configuration (language, temporal, length).
        task_id:            SLURM array task index (0-based). Default 0.
        num_tasks:          Total number of SLURM array tasks. Default 1 (no sharding).

    Yields:
        TextRecord objects, one per sentence extracted from each row.
    """
    import csv
    import os

    if source_name is None:
        source_name = os.path.basename(path)

    # --- Read file with encoding fallback ---
    def open_file():
        try:
            return open(path, "r", encoding=encoding, newline="")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 failed for {path}, falling back to latin-1")
            return open(path, "r", encoding="latin-1", newline="")

    # --- Detect delimiter ---
    if delimiter is None:
        with open_file() as f:
            sample = f.read(8192)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t|;")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","  # default fallback
        logger.info(f"Auto-detected delimiter: {repr(delimiter)}")

    # --- Read header and detect columns ---
    with open_file() as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = reader.fieldnames or []

        if not columns:
            logger.error(f"No columns found in {path}")
            return

        logger.info(f"CSV columns: {columns}")

        # Detect text column
        resolved_text_col = text_column
        if resolved_text_col is None:
            resolved_text_col = _detect_column(columns, _TEXT_COLUMN_NAMES, "text")
        if resolved_text_col is None:
            # Last resort: if there's only one non-numeric-looking column, use it
            non_numeric = [c for c in columns if not c.strip().lower().startswith(("id", "num", "count", "index"))]
            if len(non_numeric) == 1:
                resolved_text_col = non_numeric[0]
        if resolved_text_col is None:
            logger.error(
                f"Could not detect text column in {path}. "
                f"Columns: {columns}. "
                f"Pass text_column= explicitly."
            )
            return

        # Detect timestamp column
        resolved_ts_col = timestamp_column
        if resolved_ts_col is None:
            resolved_ts_col = _detect_column(columns, _TIMESTAMP_COLUMN_NAMES, "timestamp")
        if resolved_ts_col is None:
            logger.info(f"No timestamp column detected in {path}. Temporal filtering skipped.")

        # Detect source URL column
        resolved_url_col = source_url_column
        if resolved_url_col is None:
            resolved_url_col = _detect_column(columns, _SOURCE_COLUMN_NAMES, "source_url")

        logger.info(
            f"Using columns — text: {resolved_text_col}, "
            f"timestamp: {resolved_ts_col}, "
            f"source_url: {resolved_url_col}"
        )

        # --- Stream rows ---
        row_count = 0
        yielded_count = 0

        for row in reader:
            # Row-level sharding for SLURM parallelism
            if num_tasks > 1 and (row_count % num_tasks) != task_id:
                row_count += 1
                continue

            row_count += 1

            text = (row.get(resolved_text_col) or "").strip()
            if not text:
                continue

            timestamp = ""
            if resolved_ts_col:
                timestamp = (row.get(resolved_ts_col) or "").strip()

            source_url = ""
            if resolved_url_col:
                source_url = (row.get(resolved_url_col) or "").strip()

            # Clean the text
            text = clean_text(text)
            if not text:
                continue

            # Split into sentences (a CSV cell may contain paragraphs)
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                if filter_record(sentence, timestamp, config):
                    yielded_count += 1
                    yield TextRecord(
                        text=sentence,
                        timestamp=timestamp,
                        source_url=source_url or f"{path}:row_{row_count}",
                        source=source_name,
                    )

        shard_info = f" (task {task_id}/{num_tasks})" if num_tasks > 1 else ""
        logger.info(
            f"CSV {path}{shard_info}: {row_count} rows read, "
            f"{yielded_count} sentences yielded"
        )


def stream_csv_directory(
    directory: str,
    text_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    source_url_column: Optional[str] = None,
    source_name: Optional[str] = None,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
    glob_pattern: str = "*.csv",
) -> Iterator[TextRecord]:
    """
    Stream from all CSV files in a directory.

    Args:
        directory:    Path to directory containing CSV files
        glob_pattern: File pattern to match. "*.csv" for CSVs, "*.tsv" for TSVs,
                      "*.csv,*.tsv" won't work — use "*.csv" and "*.tsv" separately
                      or "*.*sv" for both.
        (other args): Passed through to stream_csv(). If None, auto-detected
                      independently for each file (since schemas may differ).
    """
    import os
    import glob as globmod

    files = sorted(globmod.glob(os.path.join(directory, glob_pattern)))
    if not files:
        logger.warning(f"No files matching {glob_pattern} in {directory}")
        return

    logger.info(f"Found {len(files)} CSV files in {directory}")

    for filepath in files:
        logger.info(f"Processing: {os.path.basename(filepath)}")
        yield from stream_csv(
            filepath,
            text_column=text_column,
            timestamp_column=timestamp_column,
            source_url_column=source_url_column,
            source_name=source_name or os.path.basename(filepath),
            config=config,
        )


def stream_text_file(
    path: str,
    source_name: str = "local",
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
) -> Iterator[TextRecord]:
    """
    Line-by-line streaming for local test files.

    Note: Local files have no timestamps, so temporal filtering is skipped.
    Language filtering still applies.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_text(line)
            if not line:
                continue

            # Skip temporal filter (no timestamps) but apply length + language
            word_count = len(line.split())
            if not (config.min_words < word_count < config.max_words):
                continue

            if config.target_languages:
                lang, confidence = detect_language(line)
                if lang not in config.target_languages:
                    continue
                if confidence < config.lang_confidence_threshold:
                    continue

            yield TextRecord(
                text=line,
                timestamp="",
                source_url=path,
                source=source_name,
            )


# ============================================================================
# MULTI-FILE HELPERS
# ============================================================================
#
# For multi-year studies you'll often have many files per source
# (one Reddit dump per month, multiple CC WET files per crawl, etc.).
# These helpers make it easy to iterate over a directory of files.
# ============================================================================

def stream_reddit_directory(
    directory: str,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
    subreddits: Optional[set[str]] = None,
) -> Iterator[TextRecord]:
    """
    Stream all Reddit dump files from a directory.

    Expected structure: directory/RC_2015-01.zst, RC_2015-02.zst, ...
    Files are processed in sorted order (chronological).
    """
    import os
    import glob

    files = sorted(glob.glob(os.path.join(directory, "RC_*.zst")))
    if not files:
        files = sorted(glob.glob(os.path.join(directory, "RC_*.jsonl")))

    logger.info(f"Found {len(files)} Reddit dump files in {directory}")

    for filepath in files:
        logger.info(f"Processing: {os.path.basename(filepath)}")
        yield from stream_reddit_dump(filepath, config=config, subreddits=subreddits)


def download_wet_paths(
    crawl_id: str,
    cache_dir: str = "./cc_cache",
) -> str:
    """
    Download and cache the WET file paths index for a Common Crawl crawl.

    Each crawl (~1 month of the web) has ~90,000 WET files listed in
    a gzipped index file. This downloads that index once and caches it
    locally so you don't re-download on every run.

    Args:
        crawl_id:  Crawl identifier, e.g. "CC-MAIN-2024-10"
                   Full list: https://index.commoncrawl.org/collinfo.json
        cache_dir: Directory to cache the paths file

    Returns:
        Path to the local uncompressed paths file
    """
    import gzip
    import requests as req

    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{crawl_id}_wet_paths.txt")

    if os.path.exists(local_path):
        logger.info(f"Using cached WET paths: {local_path}")
        return local_path

    url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/wet.paths.gz"
    logger.info(f"Downloading WET paths index from {url}...")

    response = req.get(url)
    response.raise_for_status()

    content = gzip.decompress(response.content).decode("utf-8")
    with open(local_path, "w") as f:
        f.write(content)

    line_count = len(content.strip().split("\n"))
    logger.info(f"Downloaded {line_count} WET file paths -> {local_path}")

    return local_path


def stream_common_crawl_wet_list(
    wet_paths_file: str,
    base_url: str = "https://data.commoncrawl.org/",
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
    max_files: Optional[int] = None,
) -> Iterator[TextRecord]:
    """
    Stream from a list of WET file paths (from wet.paths.gz).

    Args:
        wet_paths_file: Path to a local file listing WET paths (one per line).
                        Get this via download_wet_paths().
        base_url:       Base URL to prepend to each path
        config:         Filtering configuration
        max_files:      Stop after processing this many files (for testing).
                        None = process all.
    """
    with open(wet_paths_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(paths)} WET files in {wet_paths_file}")

    for i, path in enumerate(paths):
        if max_files is not None and i >= max_files:
            break
        url = base_url + path
        logger.info(f"Processing WET file {i+1}/{min(max_files or len(paths), len(paths))}: {path}")
        yield from stream_common_crawl_wet(url, config=config)


def stream_common_crawl(
    crawl_id: str,
    max_files: int = 1,
    config: SourceConfig = DEFAULT_SOURCE_CONFIG,
    cache_dir: str = "./cc_cache",
) -> Iterator[TextRecord]:
    """
    Convenience function: download the WET index and stream in one call.

    This is the simplest way to get Common Crawl data into the pipeline.

    Args:
        crawl_id:   e.g. "CC-MAIN-2024-10"
        max_files:  How many WET files to process (1 for testing, 10+ for real)
        config:     Filtering configuration
        cache_dir:  Where to cache the paths index

    Example:
        sources = [stream_common_crawl("CC-MAIN-2024-10", max_files=1, config=cfg)]
    """
    paths_file = download_wet_paths(crawl_id, cache_dir)
    yield from stream_common_crawl_wet_list(
        paths_file, config=config, max_files=max_files,
    )

def write_sentences_to_txt(
    stream: Iterator[TextRecord],
    output_path: str,
    include_metadata: bool = False,
):
    import os

    dirpath = os.path.dirname(output_path)
    if dirpath:  # <-- FIX: only create if non-empty
        os.makedirs(dirpath, exist_ok=True)

    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for record in stream:
            if include_metadata:
                line = f"[{record.source} | {record.timestamp}] {record.text}\n"
            else:
                line = record.text + "\n"

            f.write(line)
            count += 1

    logger.info(f"Wrote {count} sentences to {output_path}")