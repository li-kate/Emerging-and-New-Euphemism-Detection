#!/usr/bin/env python3
"""
Compute monthly Reddit contextual embeddings and cosine similarity to the
global drug centroid.

Performance changes:
  * BERT inference batch size increased to 256.
  * Reddit decompression, subreddit filtering, vocabulary matching, and
    tokenization/context-window preparation run in multiple CPU processes.
  * The main process consumes prepared batches and performs GPU inference.
  * The vocabulary index is created before forking so workers share its
    memory copy-on-write on Linux/PACE.

The monthly NPZ stores sparse sufficient statistics only. It does not store
individual occurrence embeddings.
"""

import bisect
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Local shared modules
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from shared_utils.corpus_stream import resolve_inputs, stream_corpus, iter_token_spans
from shared_utils.subreddit_scope import load_subreddits, subreddit_hash


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDDIT_MONTHLY_DIR = (
    "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
)
FREQ_DIR = (
    "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/freq_drug_subreddits_unfiltered/"
)
VOCAB_FILE = os.path.join(FREQ_DIR, "freq_words.csv")

SUBREDDIT_LIST_FILE = "/storage/scratch1/8/arodrigues46/Emerging-and-New-Euphemism-Detection/drug_lists/drug_subreddit_list.txt"

CENTROID_RESULTS_DIR = "./drug_centroid/drug_centroid_test/centroid_results"
DRUG_CENTROID_PATH = os.path.join(
    CENTROID_RESULTS_DIR, "global_drug_centroid.json"
)

OUTPUT_DIR = "./cosine_similarity_results/shards"
MONTHLY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "monthly")

MODEL_NAME = "bert-base-uncased"
CONTEXT_WORDPIECES = 32

# Requested GPU batch-size increase.
BATCH_SIZE = 256

# Number of CPU preprocessing processes. Match this to --cpus-per-task.
PREPROCESS_WORKERS = 4

# Bounded queue prevents CPU workers from accumulating unlimited prepared
# batches when the GPU is temporarily slower than preprocessing.
PREPROCESS_QUEUE_MAX_BATCHES = 16

MIN_YEAR = 2015
MAX_YEAR = 2015

MONTH_RE = re.compile(r"(\d{4})-(\d{2})")
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

# The vocabulary is large; workers inherit this mapping through fork/COW.
_WORD_TO_LOCAL = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(processName)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input loading and validation
# ---------------------------------------------------------------------------

def read_csv_column(path, preferred_column):
    """Read a named CSV column, falling back to the first column."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        headers = next(csv.reader([first_line]), [])

        if preferred_column in headers:
            values = [
                row.get(preferred_column, "")
                for row in csv.DictReader(f)
            ]
        else:
            values = [
                row[0]
                for row in csv.reader(f)
                if row and row[0].strip()
            ]

    return [value.strip() for value in values if value and value.strip()]


def load_vocabulary(path):
    """
    Load the complete vocabulary from freq_words.csv.

    No frequency matrix is loaded and no positive-frequency filtering is
    performed. The returned order is the original CSV row order.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Vocabulary CSV not found: {path}")

    words = [word.lower() for word in read_csv_column(path, "word")]

    if not words:
        raise ValueError(f"No vocabulary words found in {path}")

    if len(set(words)) != len(words):
        raise ValueError(
            "Vocabulary contains duplicate words after lowercasing; "
            "word-to-index mapping would be ambiguous."
        )

    digest = hashlib.sha256()
    for word in words:
        digest.update(word.encode("utf-8"))
        digest.update(b"\n")

    logger.info("Loaded %s vocabulary terms from %s", f"{len(words):,}", path)
    logger.info("Vocabulary SHA-256: %s", digest.hexdigest())

    return words, digest.hexdigest()


def load_drug_centroid(path):
    """Load and validate the global drug centroid."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        centroid = np.asarray(
            payload["drug_centroid"]["centroid"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path!r} must contain drug_centroid.centroid."
        ) from exc

    if centroid.ndim != 1 or not np.isfinite(centroid).all():
        raise ValueError("Global drug centroid must be a finite 1-D vector.")

    norm = float(np.linalg.norm(centroid))
    if norm == 0.0:
        raise ValueError("Global drug centroid has zero norm.")

    return centroid, payload


def get_month_files(all_paths, month_key):
    """Select RC/RS dump files for one calendar month, e.g. 2015-01."""
    selected = []

    for path in all_paths:
        match = MONTH_RE.search(os.path.basename(path))
        if match and f"{match.group(1)}-{match.group(2)}" == month_key:
            selected.append(path)

    selected = sorted(set(selected))
    if not selected:
        raise FileNotFoundError(
            f"No Reddit dump files found for {month_key} in "
            f"{REDDIT_MONTHLY_DIR}."
        )

    return selected


def get_month_tasks():
    return [
        f"{year}-{month:02d}"
        for year in range(MIN_YEAR, MAX_YEAR + 1)
        for month in range(1, 13)
    ]


# ---------------------------------------------------------------------------
# CPU preprocessing workers
# ---------------------------------------------------------------------------

def _prepare_one_record(text, tokenizer):
    """
    Find vocabulary matches and convert each occurrence to a BERT input.

    Returns a list of (input_ids, (vocab_index, target_start, target_end)).
    Tokenization of the original comment happens once, even when it contains
    multiple target-word occurrences.
    """
    candidate_spans = []

    for token, char_start, char_end in iter_token_spans(text):
        local_index = _WORD_TO_LOCAL.get(token.lower())
        if local_index is not None:
            candidate_spans.append((local_index, char_start, char_end))

    if not candidate_spans:
        return [], 0

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if not input_ids:
        return [], len(candidate_spans)

    wp_starts = [start for start, end in offsets]
    wp_ends = [end for start, end in offsets]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise RuntimeError("Tokenizer is missing CLS or SEP token IDs.")

    prepared = []
    without_wordpieces = 0

    for local_index, char_start, char_end in candidate_spans:
        # Locate the WordPiece span overlapping this character span.
        wp_start = bisect.bisect_right(wp_ends, char_start)
        wp_end = bisect.bisect_left(wp_starts, char_end)

        if wp_start >= wp_end:
            without_wordpieces += 1
            continue

        window_start = max(0, wp_start - CONTEXT_WORDPIECES)
        window_end = min(len(input_ids), wp_end + CONTEXT_WORDPIECES)
        window_ids = input_ids[window_start:window_end]

        # Target positions are shifted by one because [CLS] is prepended.
        prepared_ids = [cls_id] + window_ids + [sep_id]
        target_start = wp_start - window_start + 1
        target_end = wp_end - window_start + 1

        if (
            target_start < 1
            or target_end > len(prepared_ids) - 1
            or target_start >= target_end
        ):
            without_wordpieces += 1
            continue

        prepared.append(
            (
                prepared_ids,
                (local_index, target_start, target_end),
            )
        )

    return prepared, without_wordpieces


def _preprocessing_worker(task_queue, output_queue, allowed_subreddits):
    """
    Read/decompress assigned Reddit files and prepare batches on a CPU worker.

    Each worker creates its own fast tokenizer. The large vocabulary mapping
    is inherited from the parent process using Linux fork copy-on-write.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from transformers import BertTokenizerFast

        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
        if not tokenizer.is_fast:
            raise RuntimeError("A fast BERT tokenizer is required.")

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            raise RuntimeError("Tokenizer is missing CLS/SEP token IDs.")

        while True:
            path = task_queue.get()
            if path is None:
                break

            file_records = 0
            file_matches = 0
            file_queued = 0
            file_without_wordpieces = 0
            local_drop_counts = defaultdict(int)

            pending_ids = []
            pending_targets = []

            try:
                stream = stream_corpus(
                    [path],
                    MIN_YEAR,
                    MAX_YEAR,
                    allowed_subreddits,
                    local_drop_counts,
                )

                for _record_half_year, text_value in stream:
                    file_records += 1
                    if not text_value:
                        continue

                    prepared, missing = _prepare_one_record(
                        text_value, tokenizer
                    )
                    file_without_wordpieces += missing

                    if prepared:
                        file_matches += 1

                    for input_ids, target in prepared:
                        pending_ids.append(input_ids)
                        pending_targets.append(target)
                        file_queued += 1

                        if len(pending_ids) >= BATCH_SIZE:
                            output_queue.put(
                                ("batch", pending_ids, pending_targets)
                            )
                            pending_ids = []
                            pending_targets = []

                if pending_ids:
                    output_queue.put(
                        ("batch", pending_ids, pending_targets)
                    )

                output_queue.put(
                    (
                        "file_done",
                        {
                            "file": path,
                            "records_seen": file_records,
                            "records_with_matches": file_matches,
                            "occurrences_queued": file_queued,
                            "occurrences_without_wordpieces":
                                file_without_wordpieces,
                            "drop_counts": dict(local_drop_counts),
                        },
                    )
                )

            except Exception:
                output_queue.put(
                    (
                        "error",
                        path,
                        traceback.format_exc(),
                    )
                )
                # Stop this worker after a file-level failure. The parent
                # receives the error and terminates the run.
                break

    except Exception:
        output_queue.put(
            ("worker_error", traceback.format_exc())
        )

    finally:
        output_queue.put(("worker_done",))


# ---------------------------------------------------------------------------
# GPU inference and sparse accumulation
# ---------------------------------------------------------------------------

def process_month_parallel(
    files,
    allowed_subreddits,
    centroid,
    vocabulary_size,
):
    """
    Run CPU preprocessing workers concurrently with GPU BERT inference.

    The GPU process consumes bounded batches from the workers and maintains
    sparse sufficient statistics keyed by vocabulary index.
    """

    # Sparse accumulators: only words with actual occurrences use memory.
    embedding_sums = {}
    n_occurrences = defaultdict(int)
    cosine_sum = defaultdict(float)
    cosine_sq_sum = defaultdict(float)

    # Fork is intentional: the multi-million-entry vocabulary dictionary
    # is shared copy-on-write instead of being pickled to every worker.
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "This PACE implementation requires multiprocessing 'fork' "
            "to share the large vocabulary mapping efficiently."
        )

    ctx = mp.get_context("fork")
    task_queue = ctx.Queue()
    output_queue = ctx.Queue(
        maxsize=PREPROCESS_QUEUE_MAX_BATCHES
    )

    workers = []
    for worker_index in range(PREPROCESS_WORKERS):
        process = ctx.Process(
            target=_preprocessing_worker,
            args=(task_queue, output_queue, allowed_subreddits),
            name=f"reddit-preprocess-{worker_index}",
        )
        process.start()
        workers.append(process)

    for path in files:
        task_queue.put(path)

    for _ in workers:
        task_queue.put(None)

    # IMPORTANT: fork all CPU workers before initializing CUDA in the
    # parent. Workers perform CPU-only decompression/tokenization.
    import torch
    from transformers import AutoModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast BERT tokenizer is required.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else None
    model = AutoModel.from_pretrained(
        MODEL_NAME, torch_dtype=dtype
    ).to(device)
    model.eval()

    hidden_size = int(model.config.hidden_size)
    if centroid.shape[0] != hidden_size:
        for process in workers:
            if process.is_alive():
                process.terminate()
            process.join()
        raise ValueError(
            f"Centroid dimension {centroid.shape[0]} does not match "
            f"model hidden size {hidden_size}."
        )

    logger.info("Loaded %s on %s", MODEL_NAME, device)
    logger.info("GPU inference batch size: %d", BATCH_SIZE)

    centroid_norm = float(np.linalg.norm(centroid))

    occurrences_embedded = 0
    occurrences_queued = 0
    records_seen = 0
    records_with_matches = 0
    files_completed = 0
    workers_completed = 0
    drop_counts = defaultdict(int)

    started_at = time.monotonic()
    last_progress_at = started_at

    def run_gpu_batch(input_ids, targets):
        nonlocal occurrences_embedded

        if not input_ids:
            return

        padded = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        padded = {
            key: value.to(device, non_blocking=True)
            for key, value in padded.items()
        }

        with torch.inference_mode():
            hidden = model(**padded).last_hidden_state.float()

            starts = torch.tensor(
                [target[1] for target in targets],
                device=device,
                dtype=torch.long,
            )
            ends = torch.tensor(
                [target[2] for target in targets],
                device=device,
                dtype=torch.long,
            )
            batch_indices = torch.arange(
                len(targets), device=device
            )

            prefix = torch.cat(
                [
                    torch.zeros_like(hidden[:, :1, :]),
                    hidden.cumsum(dim=1),
                ],
                dim=1,
            )

            vectors = (
                prefix[batch_indices, ends]
                - prefix[batch_indices, starts]
            ) / (ends - starts).unsqueeze(1).to(hidden.dtype)

            vectors = vectors.cpu().numpy().astype(np.float64)

        word_indices = np.asarray(
            [target[0] for target in targets],
            dtype=np.int64,
        )

        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(norms) & (norms > 0)

        if not np.any(valid):
            return

        vectors = vectors[valid]
        word_indices = word_indices[valid]
        norms = norms[valid]

        cosines = (vectors @ centroid) / (norms * centroid_norm)

        # Aggregate by distinct word within this batch. This avoids calling
        # np.add.at against a dense (vocabulary_size x hidden_size) matrix.
        for word_index in np.unique(word_indices):
            mask = word_indices == word_index
            vector_sum = vectors[mask].sum(axis=0)

            if int(word_index) in embedding_sums:
                embedding_sums[int(word_index)] += vector_sum
            else:
                embedding_sums[int(word_index)] = vector_sum

            n_occurrences[int(word_index)] += int(mask.sum())
            cosine_sum[int(word_index)] += float(cosines[mask].sum())
            cosine_sq_sum[int(word_index)] += float(
                np.square(cosines[mask]).sum()
            )

        occurrences_embedded += int(valid.sum())

    try:
        while workers_completed < len(workers):
            try:
                message = output_queue.get(timeout=5)
            except queue.Empty:
                dead = [
                    p.name for p in workers
                    if not p.is_alive() and p.exitcode not in (0, None)
                ]
                if dead:
                    raise RuntimeError(
                        "Preprocessing workers exited unexpectedly: "
                        + ", ".join(dead)
                    )
                continue

            kind = message[0]

            if kind == "batch":
                _, input_ids, targets = message
                occurrences_queued += len(targets)
                run_gpu_batch(input_ids, targets)

            elif kind == "file_done":
                stats = message[1]
                files_completed += 1
                records_seen += stats["records_seen"]
                records_with_matches += stats["records_with_matches"]

                for key, value in stats["drop_counts"].items():
                    drop_counts[key] += value

                logger.info(
                    "Finished preprocessing %s: records=%s, "
                    "matched_records=%s, queued_occurrences=%s",
                    os.path.basename(stats["file"]),
                    f'{stats["records_seen"]:,}',
                    f'{stats["records_with_matches"]:,}',
                    f'{stats["occurrences_queued"]:,}',
                )

            elif kind == "error":
                _, path, details = message
                raise RuntimeError(
                    f"Preprocessing failed for {path}:\n{details}"
                )

            elif kind == "worker_error":
                raise RuntimeError(
                    "A preprocessing worker failed:\n" + message[1]
                )

            elif kind == "worker_done":
                workers_completed += 1

            now = time.monotonic()
            if now - last_progress_at >= 60:
                logger.info(
                    "Pipeline progress: files=%d/%d, records_seen=%s, "
                    "occurrences_queued=%s, occurrences_embedded=%s, "
                    "unique_words=%s, elapsed=%.1f min",
                    files_completed,
                    len(files),
                    f"{records_seen:,}",
                    f"{occurrences_queued:,}",
                    f"{occurrences_embedded:,}",
                    f"{len(n_occurrences):,}",
                    (now - started_at) / 60.0,
                )
                last_progress_at = now

        # Ensure all worker processes have exited cleanly.
        for process in workers:
            process.join()

        bad_workers = [
            (process.name, process.exitcode)
            for process in workers
            if process.exitcode != 0
        ]
        if bad_workers:
            raise RuntimeError(
                f"Preprocessing worker exit errors: {bad_workers}"
            )

    except Exception:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            process.join()
        raise

    finally:
        task_queue.close()
        output_queue.close()

    run_stats = {
        "files_seen": len(files),
        "files_completed": files_completed,
        "records_seen": records_seen,
        "records_with_matches": records_with_matches,
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
        "occurrences_queued": occurrences_queued,
        "occurrences_embedded": occurrences_embedded,
        "words_with_embeddings": len(n_occurrences),
        "vocabulary_size": vocabulary_size,
        "preprocessing_workers": PREPROCESS_WORKERS,
        "batch_size": BATCH_SIZE,
        "corpus_drop_counts": dict(drop_counts),
    }

    logger.info(
        "Embedded %s occurrences for %s distinct words in %.1f minutes.",
        f"{occurrences_embedded:,}",
        f"{len(n_occurrences):,}",
        run_stats["elapsed_seconds"] / 60.0,
    )

    aggregates = {
        "embedding_sums": embedding_sums,
        "n_occurrences": n_occurrences,
        "cosine_sum": cosine_sum,
        "cosine_sq_sum": cosine_sq_sum,
    }

    return run_stats, aggregates


# ---------------------------------------------------------------------------
# Monthly sparse shard output
# ---------------------------------------------------------------------------

def write_monthly_shard(
    month_key,
    half_year,
    vocabulary_hash,
    vocabulary_size,
    aggregates,
    run_stats,
    files,
    scope,
    centroid_payload,
):
    """Write sparse monthly sufficient statistics and JSON metadata."""
    os.makedirs(MONTHLY_OUTPUT_DIR, exist_ok=True)

    counts = aggregates["n_occurrences"]
    indices = np.asarray(sorted(counts), dtype=np.int32)

    if len(indices):
        embedding_matrix = np.vstack(
            [aggregates["embedding_sums"][int(i)] for i in indices]
        ).astype(np.float64)
        occurrence_counts = np.asarray(
            [counts[int(i)] for i in indices], dtype=np.int64
        )
        cosine_sums = np.asarray(
            [aggregates["cosine_sum"][int(i)] for i in indices],
            dtype=np.float64,
        )
        cosine_sq_sums = np.asarray(
            [aggregates["cosine_sq_sum"][int(i)] for i in indices],
            dtype=np.float64,
        )
    else:
        hidden_size = len(
            centroid_payload["drug_centroid"]["centroid"]
        )
        embedding_matrix = np.empty((0, hidden_size), dtype=np.float64)
        occurrence_counts = np.empty(0, dtype=np.int64)
        cosine_sums = np.empty(0, dtype=np.float64)
        cosine_sq_sums = np.empty(0, dtype=np.float64)

    npz_path = os.path.join(
        MONTHLY_OUTPUT_DIR, f"month_{month_key}.npz"
    )
    npz_tmp = npz_path + ".tmp"

    with open(npz_tmp, "wb") as f:
        np.savez_compressed(
            f,
            indices=indices,
            embedding_sums=embedding_matrix,
            n_occurrences=occurrence_counts,
            cosine_sum=cosine_sums,
            cosine_sq_sum=cosine_sq_sums,
        )

    os.replace(npz_tmp, npz_path)

    centroid_meta = centroid_payload.get("drug_centroid", {})
    metadata = {
        "month": month_key,
        "half_year": half_year,
        "model_name": MODEL_NAME,
        "embedding_dimension": int(embedding_matrix.shape[1]),
        "context_wordpieces_each_side": CONTEXT_WORDPIECES,
        "batch_size": BATCH_SIZE,
        "preprocessing_workers": PREPROCESS_WORKERS,
        "vocabulary_file": os.path.abspath(VOCAB_FILE),
        "vocabulary_size": int(vocabulary_size),
        "vocabulary_sha256": vocabulary_hash,
        "sufficient_statistics": [
            "embedding_sums",
            "n_occurrences",
            "cosine_sum",
            "cosine_sq_sum",
        ],
        "similarity_target": "global drug centroid",
        "occurrence_cosine_std": "population standard deviation",
        "scope": scope,
        "drug_centroid": {
            "file": os.path.abspath(DRUG_CENTROID_PATH),
            "metadata": {
                key: value
                for key, value in centroid_meta.items()
                if key != "centroid"
            },
        },
        "files": list(files),
        "statistics": run_stats,
        "active_words_with_occurrences": int(len(indices)),
        "npz_file": os.path.basename(npz_path),
    }

    json_path = os.path.join(
        MONTHLY_OUTPUT_DIR, f"month_{month_key}.json"
    )
    json_tmp = json_path + ".tmp"

    with open(json_tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, allow_nan=False, indent=2)

    os.replace(json_tmp, json_path)

    logger.info("Wrote monthly sufficient statistics: %s", npz_path)
    logger.info("Wrote monthly metadata: %s", json_path)

    return npz_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _WORD_TO_LOCAL

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    month_tasks = get_month_tasks()
    if not 0 <= TASK_ID < len(month_tasks):
        raise ValueError(
            f"TASK_ID={TASK_ID} is invalid; valid IDs are "
            f"0-{len(month_tasks) - 1}."
        )

    month_key = month_tasks[TASK_ID]
    year, month = map(int, month_key.split("-"))
    half_year = f"{year}H{1 if month <= 6 else 2}"

    logger.info("Month: %s", month_key)
    logger.info("Half-year: %s", half_year)
    logger.info("BERT batch size: %d", BATCH_SIZE)
    logger.info("CPU preprocessing workers: %d", PREPROCESS_WORKERS)
    logger.info("Context WordPieces per side: %d", CONTEXT_WORDPIECES)

    # Load the full vocabulary only. No freq.npy, half-year CSV, or freq_meta.
    words, vocabulary_hash = load_vocabulary(VOCAB_FILE)

    # This dictionary is inherited copy-on-write by forked CPU workers.
    _WORD_TO_LOCAL = {word: index for index, word in enumerate(words)}
    logger.info(
        "Built vocabulary lookup with %s entries.",
        f"{len(_WORD_TO_LOCAL):,}",
    )

    all_paths = resolve_inputs(
        input=None,
        input_dir=None,
        reddit_monthly_dir=REDDIT_MONTHLY_DIR,
    )
    files = get_month_files(all_paths, month_key)

    logger.info(
        "Selected %d Reddit dump files for %s: %s",
        len(files),
        month_key,
        ", ".join(os.path.basename(path) for path in files),
    )

    allowed = load_subreddits(SUBREDDIT_LIST_FILE)
    if allowed is None:
        raise ValueError(
            "The subreddit allowlist loader returned None. "
            "Refusing to process the unrestricted corpus."
        )

    scope = {
        "mode": "subreddits",
        "file": os.path.abspath(SUBREDDIT_LIST_FILE),
        "hash": subreddit_hash(allowed),
    }

    centroid, centroid_payload = load_drug_centroid(
        DRUG_CENTROID_PATH
    )

    # process_month_parallel starts the CPU workers first, then initializes
    # the parent tokenizer/model. This ordering avoids forking after CUDA
    # initialization.
    run_stats, aggregates = process_month_parallel(
        files=files,
        allowed_subreddits=allowed,
        centroid=centroid,
        vocabulary_size=len(words),
    )

    run_stats["month"] = month_key
    run_stats["half_year"] = half_year

    output_path = write_monthly_shard(
        month_key=month_key,
        half_year=half_year,
        vocabulary_hash=vocabulary_hash,
        vocabulary_size=len(words),
        aggregates=aggregates,
        run_stats=run_stats,
        files=files,
        scope=scope,
        centroid_payload=centroid_payload,
    )

    logger.info("Saved monthly shard: %s", output_path)


if __name__ == "__main__":
    main()
