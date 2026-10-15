"""Compute contextual-embedding cosine similarity to the global drug centroid.

One SLURM array task processes one half-year. For each word with positive
frequency in that half-year:
  1. Find every occurrence in the selected Reddit subreddits.
  2. Extract a target-centered BERT window with up to 64 WordPieces
     before and after the target.
  3. Average the target WordPiece hidden states to obtain an occurrence
     embedding.
  4. Average occurrence embeddings for each word and compute cosine
     similarity to the global drug centroid.

Output:
    cosine_YYYYH1.npz or cosine_YYYYH2.npz
    Each file contains scores and embedded occurrence counts aligned to
    the full frequency-matrix vocabulary. Missing words have NaN scores.
"""

import bisect
import csv
import glob
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import torch
import zstandard as zstd
from transformers import AutoModel, AutoTokenizer

# Import the exact tokenizer used by freq_matrix.py.
from freq_matrix import TOKEN_RE


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

REDDIT_MONTHLY_DIR = (
    "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
)

# Change these to the actual directories containing your existing files.
FREQ_DIR = "../frequency_matrix"
DRUG_CENTROID_PATH = (
    "../cosine_similarity_drift/drug_centroid/checkpoints/global_drug_centroid.json"
)
SUBREDDIT_LIST_FILE = "drug_subreddit_list.txt"
OUTPUT_DIR = "./cosine_similarity_results"

MODEL_NAME = "bert-base-uncased"
CONTEXT_WORDPIECES = 64
BATCH_SIZE = 128
MIN_YEAR = 2012
MAX_YEAR = 2026

# SLURM_ARRAY_TASK_ID is zero-based and indexes freq_half_years.csv.
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------
# INPUT HELPERS
# ---------------------------------------------------------------------

def read_csv_column(path, preferred_column):
    """Read a named CSV column, with a fallback for headerless files."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)

        headers = next(csv.reader([first_line]), [])
        if preferred_column in headers:
            reader = csv.DictReader(f)
            return [row[preferred_column] for row in reader]

        # Headerless CSV: use the first column.
        return [
            row[0]
            for row in csv.reader(f)
            if row and row[0].strip()
        ]


def load_frequency_inputs():
    freq = np.load(os.path.join(FREQ_DIR, "freq.npy"), mmap_mode="r")
    half_years = read_csv_column(
        os.path.join(FREQ_DIR, "freq_half_years.csv"), "half_year"
    )
    words = read_csv_column(
        os.path.join(FREQ_DIR, "freq_words.csv"), "word"
    )

    if freq.ndim != 2:
        raise ValueError(f"Expected 2-D freq matrix; got {freq.shape}.")
    if freq.shape != (len(half_years), len(words)):
        raise ValueError(
            "Frequency matrix dimensions do not match CSV files: "
            f"freq={freq.shape}, half-years={len(half_years)}, "
            f"words={len(words)}."
        )
    if len(set(words)) != len(words):
        raise ValueError("freq_words.csv contains duplicate word labels.")
    if len(set(half_years)) != len(half_years):
        raise ValueError("freq_half_years.csv contains duplicate periods.")

    return freq, half_years, words


def load_allowed_subreddits(path):
    with open(path, "r", encoding="utf-8") as f:
        allowed = {
            line.strip().lower()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }

    if not allowed:
        raise ValueError(f"No subreddits loaded from {path!r}.")
    return allowed


def half_year_from_timestamp(timestamp):
    dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    if not MIN_YEAR <= dt.year <= MAX_YEAR:
        return None
    return f"{dt.year}H{1 if dt.month <= 6 else 2}"


def clean_record_text(record):
    """Match freq_matrix.py's body/selftext text-field policy."""
    text = record.get("body") or record.get("selftext") or ""
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text or text in {"[deleted]", "[removed]"}:
        return ""

    text = URL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def iter_reddit_records(path):
    """Stream JSONL records from one zstandard-compressed Reddit dump."""
    with open(path, "rb") as raw:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw) as reader:
            import io

            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line in text_stream:
                    if not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def get_reddit_files():
    files = sorted(
        glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RC_*.zst"))
        + glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RS_*.zst"))
    )
    if not files:
        raise FileNotFoundError(
            f"No RC_*.zst or RS_*.zst files found in {REDDIT_MONTHLY_DIR!r}."
        )
    return files


# ---------------------------------------------------------------------
# CENTROID / MODEL
# ---------------------------------------------------------------------

def load_drug_centroid(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        centroid = np.asarray(
            payload["drug_centroid"]["centroid"], dtype=np.float32
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{path!r} does not contain drug_centroid.centroid."
        ) from exc

    if centroid.ndim != 1 or not np.isfinite(centroid).all():
        raise ValueError("Drug centroid must be a finite 1-D vector.")
    if np.linalg.norm(centroid) == 0:
        raise ValueError("Drug centroid has zero norm.")

    return centroid


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for offset mappings."
        )

    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    return tokenizer, model, device


# ---------------------------------------------------------------------
# EMBEDDING
# ---------------------------------------------------------------------

def process_half_year(
    target_half_year,
    active_words,
    all_words,
    centroid,
    allowed_subreddits,
    tokenizer,
    model,
    device,
):
    """Stream the corpus and calculate scores for one half-year."""

    # Use the original full-vocabulary column positions in the output.
    word_to_local = {word: i for i, word in enumerate(active_words)}
    global_index = {word: i for i, word in enumerate(all_words)}
    local_to_global = {
        i: global_index[word] for i, word in enumerate(active_words)
    }

    hidden_size = int(model.config.hidden_size)
    if centroid.shape[0] != hidden_size:
        raise ValueError(
            f"Centroid dimension {centroid.shape[0]} does not match "
            f"{MODEL_NAME} hidden size {hidden_size}."
        )

    # Float32 keeps memory use manageable. Only this half-year's active
    # vocabulary is allocated.
    embedding_sums = np.zeros(
        (len(active_words), hidden_size), dtype=np.float32
    )
    occurrence_counts = np.zeros(len(active_words), dtype=np.int64)

    pending_ids = []
    pending_targets = []
    records_seen = 0
    records_in_period = 0
    occurrences_queued = 0

    def flush_batch():
        nonlocal pending_ids, pending_targets

        if not pending_ids:
            return

        padded = tokenizer.pad(
            {"input_ids": pending_ids},
            padding=True,
            return_tensors="pt",
        )
        padded = {k: v.to(device) for k, v in padded.items()}

        with torch.inference_mode():
            hidden = model(**padded).last_hidden_state

        hidden = hidden.cpu()

        for batch_i, (local_i, target_start, target_end) in enumerate(
            pending_targets
        ):
            # target_end is exclusive. The target can consist of multiple
            # WordPieces; average them into one occurrence vector.
            vector = hidden[
                batch_i, target_start:target_end, :
            ].mean(dim=0).numpy()

            embedding_sums[local_i] += vector
            occurrence_counts[local_i] += 1

        pending_ids = []
        pending_targets = []

    files = get_reddit_files()

    for file_number, path in enumerate(files, start=1):
        print(
            f"[{target_half_year}] File {file_number}/{len(files)}: "
            f"{os.path.basename(path)}",
            flush=True,
        )

        for record in iter_reddit_records(path):
            records_seen += 1

            subreddit = str(record.get("subreddit", "")).strip().lower()
            if subreddit not in allowed_subreddits:
                continue

            timestamp = record.get("created_utc")
            if timestamp is None:
                continue

            try:
                period = half_year_from_timestamp(timestamp)
            except (TypeError, ValueError, OverflowError):
                continue

            if period != target_half_year:
                continue

            text = clean_record_text(record)
            if not text:
                continue

            records_in_period += 1

            # Match candidate word spans using the exact regex imported
            # from freq_matrix.py.
            candidate_spans = []
            for match in TOKEN_RE.finditer(text):
                word = match.group(0).lower()
                local_i = word_to_local.get(word)
                if local_i is not None:
                    candidate_spans.append(
                        (local_i, match.start(), match.end())
                    )

            if not candidate_spans:
                continue

            # Tokenize the whole record without truncation, then select a
            # separate 64-WordPiece context window around each occurrence.
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            input_ids = encoded["input_ids"]
            offsets = encoded["offset_mapping"]

            if not input_ids:
                continue

            wp_starts = [start for start, end in offsets]
            wp_ends = [end for start, end in offsets]

            for local_i, char_start, char_end in candidate_spans:
                # WordPieces overlap the character span when:
                # wp_end > char_start and wp_start < char_end.
                wp_start = bisect.bisect_right(wp_ends, char_start)
                wp_end = bisect.bisect_left(wp_starts, char_end)

                if wp_start >= wp_end:
                    continue

                window_start = max(0, wp_start - CONTEXT_WORDPIECES)
                window_end = min(
                    len(input_ids), wp_end + CONTEXT_WORDPIECES
                )
                window_ids = input_ids[window_start:window_end]

                prepared = tokenizer.prepare_for_model(
                    window_ids,
                    add_special_tokens=True,
                    truncation=False,
                    return_attention_mask=False,
                )
                # bert-base-uncased adds [CLS] before the window.
                target_start = wp_start - window_start + 1
                target_end = wp_end - window_start + 1

                if (
                    target_start < 1
                    or target_end > len(prepared["input_ids"]) - 1
                    or target_start >= target_end
                ):
                    continue

                pending_ids.append(prepared["input_ids"])
                pending_targets.append(
                    (local_i, target_start, target_end)
                )
                occurrences_queued += 1

                if len(pending_ids) >= BATCH_SIZE:
                    flush_batch()

        print(
            f"[{target_half_year}] Finished {os.path.basename(path)}; "
            f"records scanned so far: {records_seen:,}; "
            f"target-period records: {records_in_period:,}; "
            f"occurrences queued: {occurrences_queued:,}",
            flush=True,
        )

    flush_batch()

    # Full-vocabulary arrays retain exactly the frequency-matrix ordering.
    scores = np.full(len(all_words), np.nan, dtype=np.float32)
    counts_full = np.zeros(len(all_words), dtype=np.int64)

    centroid_norm = float(np.linalg.norm(centroid))

    for local_i, count in enumerate(occurrence_counts):
        global_i = local_to_global[local_i]
        counts_full[global_i] = count

        if count == 0:
            continue

        mean_embedding = embedding_sums[local_i] / float(count)
        norm = float(np.linalg.norm(mean_embedding))

        if norm > 0:
            scores[global_i] = float(
                np.dot(mean_embedding, centroid)
                / (norm * centroid_norm)
            )

    print(
        f"[{target_half_year}] Embedded occurrences: "
        f"{int(occurrence_counts.sum()):,}; "
        f"words with embeddings: "
        f"{int((occurrence_counts > 0).sum()):,}",
        flush=True,
    )

    return scores, counts_full


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    freq, half_years, all_words = load_frequency_inputs()

    if not 0 <= TASK_ID < len(half_years):
        raise ValueError(
            f"TASK_ID={TASK_ID} is out of range for "
            f"{len(half_years)} half-years. Use array indices "
            f"0-{len(half_years) - 1}."
        )

    target_half_year = half_years[TASK_ID]
    frequency_row = np.asarray(freq[TASK_ID])

    # Only words present in this period are embedded.
    active_indices = np.flatnonzero(frequency_row > 0)
    active_words = [all_words[i] for i in active_indices]

    if not active_words:
        raise ValueError(
            f"No positive-frequency words in {target_half_year}."
        )

    print(f"Half-year: {target_half_year}")
    print(f"Full vocabulary size: {len(all_words):,}")
    print(f"Words with positive frequency: {len(active_words):,}")

    centroid = load_drug_centroid(DRUG_CENTROID_PATH)
    allowed_subreddits = load_allowed_subreddits(SUBREDDIT_LIST_FILE)
    tokenizer, model, device = load_model()

    scores, occurrence_counts = process_half_year(
        target_half_year=target_half_year,
        active_words=active_words,
        all_words=all_words,
        centroid=centroid,
        allowed_subreddits=allowed_subreddits,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    output_path = os.path.join(
        OUTPUT_DIR, f"cosine_{target_half_year}.npz"
    )
    np.savez_compressed(
        output_path,
        half_year=np.asarray(target_half_year),
        scores=scores,
        occurrence_counts=occurrence_counts,
    )

    print(f"Saved: {output_path}")


if __name__ == "__main__":
