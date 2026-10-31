import bisect
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Local shared modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_utils.corpus_stream import resolve_inputs, stream_corpus, iter_token_spans
from shared_utils.subreddit_scope import load_subreddits, subreddit_hash

# CONFIGURATION

REDDIT_MONTHLY_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
FREQ_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/freq_drug_subreddits_unfiltered"
FREQ_STEM = "freq"
SUBREDDIT_LIST_FILE = "../drug_lists/drug_subreddit_list.txt"

CENTROID_RESULTS_DIR = "./drug_centroid/drug_centroid_test/centroid_results"
DRUG_CENTROID_PATH = os.path.join(CENTROID_RESULTS_DIR, "global_drug_centroid.json")
OUTPUT_DIR = "./cosine_similarity_results/shards"
MONTHLY_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "monthly")

MODEL_NAME = "bert-base-uncased"
CONTEXT_WORDPIECES = 64
BATCH_SIZE = 128
MIN_YEAR = 2015
MAX_YEAR = 2016

TOP_K_DRUG_CENTROIDS = 3
MONTH_RE = re.compile(r"(\d{4})-(\d{2})")
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
SHARD_SUFFIX = ".json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# LOADERS

def read_csv_column(path, preferred_column):
    """Read a named CSV column, with a fallback for headerless files."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        headers = next(csv.reader([first_line]), [])

        if preferred_column in headers:
            return [
                row[preferred_column]
                for row in csv.DictReader(f)
            ]

        return [
            row[0]
            for row in csv.reader(f)
            if row and row[0].strip()
        ]


def load_frequency_inputs(freq_dir, stem="freq"):
    """Load the frequency matrix and its row/column labels."""
    freq_path = os.path.join(freq_dir, f"{stem}.npy")
    half_years_path = os.path.join(
        freq_dir,
        f"{stem}_half_years.csv",
    )
    words_path = os.path.join(
        freq_dir,
        f"{stem}_words.csv",
    )

    for path in (freq_path, half_years_path, words_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required frequency input not found: {path}"
            )

    freq = np.load(freq_path, mmap_mode="r")
    half_years = read_csv_column(half_years_path, "half_year")
    words = read_csv_column(words_path, "word")

    if freq.ndim != 2:
        raise ValueError(
            f"Expected a 2-D frequency matrix; got {freq.shape}."
        )

    if freq.shape != (len(half_years), len(words)):
        raise ValueError(
            "Frequency matrix dimensions do not match CSV files: "
            f"freq={freq.shape}, half-years={len(half_years)}, "
            f"words={len(words)}."
        )

    if len(set(words)) != len(words):
        raise ValueError("Frequency word list contains duplicate labels.")

    if len(set(half_years)) != len(half_years):
        raise ValueError("Frequency half-year list contains duplicate labels.")

    logger.info(
        "Loaded frequency matrix %s with shape %s",
        freq_path,
        freq.shape,
    )

    return freq, half_years, words, freq_path


def load_freq_meta(freq_dir, stem="freq"):
    """Load frequency-matrix metadata for the subreddit-scope check."""
    path = os.path.join(freq_dir, f"{stem}_meta.json")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Frequency metadata not found: {path}. "
            "The scope check requires this file."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_scope(freq_meta, allowed, subreddit_file):
    """Abort if the subreddit scope differs from the frequency matrix."""
    freq_scope = freq_meta.get("scope", {})
    this_mode = "subreddits" if allowed is not None else "all"
    this_hash = subreddit_hash(allowed)

    if (
        freq_scope.get("mode") != this_mode
        or freq_scope.get("hash") != this_hash
    ):
        raise SystemExit(
            "Scope mismatch with the frequency matrix.\n"
            f"  freq_meta: mode={freq_scope.get('mode')} "
            f"hash={freq_scope.get('hash')}\n"
            f"  this run:  mode={this_mode} hash={this_hash} "
            f"(subreddit-file={subreddit_file})\n"
            "Use the same subreddit scope as the frequency-matrix run."
        )

    logger.info("Scope check passed (mode=%s).", this_mode)


def file_sha256(path):
    """Return a reproducible SHA-256 fingerprint for an input file."""
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_drug_centroid(path):
    """Load the precomputed global drug centroid."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    try:
        centroid = np.asarray(
            payload["drug_centroid"]["centroid"],
            dtype=np.float64,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"{path!r} must contain drug_centroid.centroid."
        ) from exc

    if centroid.ndim != 1 or not np.isfinite(centroid).all():
        raise ValueError(
            "Global drug centroid must be a finite 1-D vector."
        )

    if float(np.linalg.norm(centroid)) == 0.0:
        raise ValueError("Global drug centroid has zero norm.")

    return centroid, payload


def load_individual_centroids(path, hidden_size):
    """Load and L2-normalize individual drug-term centroid vectors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Individual drug-centroid file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, (dict, list)):
        raise ValueError(
            "Unsupported individual-centroid JSON format. "
            "Expected a JSON object or list."
        )

    if isinstance(payload, dict):
        if "term_centroids" in payload:
            container = payload["term_centroids"]
        elif any(
            key in payload
            for key in (
                "individual_centroids",
                "drug_term_centroids",
                "drug_centroids",
                "centroids",
            )
        ):
            container = next(
                payload[key]
                for key in (
                    "individual_centroids",
                    "drug_term_centroids",
                    "drug_centroids",
                    "centroids",
                )
                if key in payload
            )
        else:
            container = payload
    else:
        container = payload

    raw_centroids = {}

    if isinstance(container, dict):
        for term, value in container.items():
            if isinstance(value, dict):
                vector = (
                    value.get("centroid")
                    or value.get("vector")
                    or value.get("embedding")
                )
            else:
                vector = value

            if vector is not None:
                raw_centroids[str(term)] = vector

    elif isinstance(container, list):
        for entry in container:
            if not isinstance(entry, dict):
                continue

            term = (
                entry.get("term")
                or entry.get("name")
                or entry.get("word")
            )
            vector = (
                entry.get("centroid")
                or entry.get("vector")
                or entry.get("embedding")
            )

            if term is not None and vector is not None:
                raw_centroids[str(term)] = vector
    else:
        raise ValueError(
            "Individual centroid data must be a dictionary or list."
        )

    if not raw_centroids:
        raise ValueError(
            f"No individual centroid vectors could be parsed from {path}. "
            "Expected a term_centroids mapping."
        )

    normalized_centroids = {}

    for term, raw_vector in raw_centroids.items():
        try:
            vector = np.asarray(raw_vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not convert centroid for term {term!r} "
                "to a numeric vector."
            ) from exc

        if vector.ndim != 1:
            raise ValueError(
                f"Centroid for term {term!r} must be 1-D; "
                f"got shape {vector.shape}."
            )

        if vector.shape[0] != hidden_size:
            raise ValueError(
                f"Centroid for term {term!r} has dimension "
                f"{vector.shape[0]}, but model hidden size is "
                f"{hidden_size}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Centroid for term {term!r} contains non-finite values."
            )

        norm = float(np.linalg.norm(vector))

        if norm == 0.0:
            raise ValueError(
                f"Centroid for term {term!r} has zero norm."
            )

        normalized_centroids[term] = vector / norm

    logger.info(
        "Loaded %d individual drug-term centroids from %s",
        len(normalized_centroids),
        path,
    )
    logger.info(
        "Individual centroid terms: %s",
        ", ".join(sorted(normalized_centroids)),
    )

    return normalized_centroids, payload


def get_half_year_files(all_paths, half_year):
    """Select the monthly Reddit dump files belonging to one half-year."""
    year = int(half_year[:4])
    half = int(half_year[-1])

    if half == 1:
        allowed_months = {
            f"{year}-{month:02d}" for month in range(1, 7)
        }
    elif half == 2:
        allowed_months = {
            f"{year}-{month:02d}" for month in range(7, 13)
        }
    else:
        raise ValueError(f"Invalid half-year label: {half_year!r}")

    selected = []

    for path in all_paths:
        match = MONTH_RE.search(os.path.basename(path))
        if not match:
            continue

        month_key = f"{match.group(1)}-{match.group(2)}"
        if month_key in allowed_months:
            selected.append(path)

    selected = sorted(set(selected))

    if not selected:
        raise FileNotFoundError(
            f"No monthly Reddit dumps found for {half_year}. "
            f"Expected one or more files for: {sorted(allowed_months)}."
        )

    return selected


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
            f"No Reddit dump files found for {month_key} "
            f"in {REDDIT_MONTHLY_DIR}."
        )

    return selected


# MODEL

def load_model(model_name):
    """Load the BERT encoder and fast tokenizer used for drug centroids."""
    import torch
    from transformers import AutoModel, BertTokenizerFast

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info("Loading %s on %s", model_name, device)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    if not tokenizer.is_fast:
        raise RuntimeError(
            f"Expected a fast tokenizer, got {type(tokenizer).__name__}"
        )

    dtype = torch.float16 if device.type == "cuda" else None

    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    logger.info(
        "Loaded tokenizer class: %s (fast=%s)",
        type(tokenizer).__name__,
        tokenizer.is_fast,
    )
    logger.info(
        "Tokenizer diagnostic: class=%s, is_fast=%s, "
        "CLS=%s, SEP=%s, offset_mapping=%s",
        type(tokenizer).__name__,
        tokenizer.is_fast,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        callable(getattr(tokenizer, "__call__", None)),
    )

    return tokenizer, model, device


# EMBEDDING AND AGGREGATION

def process_half_year(
    stream,
    target_half_year,
    active_words,
    frequency_by_word,
    centroid,
    individual_centroids,
    top_k,
    model,
    tokenizer,
    device,
    batch_size,
):
    """
    Stream a month's Reddit records and aggregate target-word embeddings.

    The function name is retained for compatibility, but the caller supplies
    only the files for one month. Raw occurrence embeddings are not saved.
    """
    import torch

    if top_k < 1:
        raise ValueError(f"top_k must be at least 1; got {top_k}.")

    word_to_local = {
        word: i for i, word in enumerate(active_words)
    }
    hidden_size = int(model.config.hidden_size)

    if centroid.shape[0] != hidden_size:
        raise ValueError(
            f"Global centroid dimension {centroid.shape[0]} does not "
            f"match model hidden size {hidden_size}."
        )

    if not individual_centroids:
        raise ValueError("No individual drug centroids were loaded.")

    centroid_norm = float(np.linalg.norm(centroid))
    individual_terms = list(individual_centroids.keys())

    embedding_sums = np.zeros(
        (len(active_words), hidden_size),
        dtype=np.float64,
    )
    n_occurrences = np.zeros(len(active_words), dtype=np.int64)
    cosine_sum = np.zeros(len(active_words), dtype=np.float64)
    cosine_sq_sum = np.zeros(len(active_words), dtype=np.float64)

    pending_ids = []
    pending_targets = []

    occurrences_queued = 0
    occurrences_embedded = 0
    occurrences_without_wordpieces = 0
    records_in_period = 0
    records_seen = 0
    records_with_candidates = 0
    started_at = time.monotonic()
    last_progress_at = started_at
    last_progress_records = 0
    progress_interval_seconds = 60
    progress_interval_records = 10_000

    
    def flush_batch():
        nonlocal pending_ids, pending_targets, occurrences_embedded

        if not pending_ids:
            return

        padded = tokenizer.pad(
            {"input_ids": pending_ids},
            padding=True,
            return_tensors="pt",
        )
        padded = {
            key: value.to(device)
            for key, value in padded.items()
        }

        with torch.inference_mode():
            hidden = model(**padded).last_hidden_state.float()

            # Target spans are [start, end), excluding the end position.
            starts = torch.tensor(
                [target[1] for target in pending_targets],
                device=device,
                dtype=torch.long,
            )
            ends = torch.tensor(
                [target[2] for target in pending_targets],
                device=device,
                dtype=torch.long,
            )
            batch_idx = torch.arange(len(pending_targets), device=device)

            # Prefix sums let us average each target span in parallel.
            prefix = torch.cat(
                [
                    torch.zeros_like(hidden[:, :1, :]),
                    hidden.cumsum(dim=1),
                ],
                dim=1,
            )

            vectors = (
                prefix[batch_idx, ends] - prefix[batch_idx, starts]
            ) / (ends - starts).unsqueeze(1).to(hidden.dtype)

            vectors = vectors.cpu().numpy().astype(np.float64)

        local_indices = np.asarray(
            [target[0] for target in pending_targets],
            dtype=np.intp,
        )

        norms = np.linalg.norm(vectors, axis=1)
        valid = np.isfinite(norms) & (norms > 0)

        if np.any(valid):
            vectors = vectors[valid]
            indices = local_indices[valid]
            norms = norms[valid]

            cosines = (
                vectors @ centroid
            ) / (norms * centroid_norm)

            # add.at correctly accumulates repeated words in one batch.
            np.add.at(embedding_sums, indices, vectors)
            np.add.at(n_occurrences, indices, 1)
            np.add.at(cosine_sum, indices, cosines)
            np.add.at(cosine_sq_sum, indices, cosines * cosines)

            occurrences_embedded += int(valid.sum())

        pending_ids = []
        pending_targets = []


    for record_half_year, text in stream:
        records_seen += 1
        if record_half_year != target_half_year:
            continue
        if not text:
            continue

        records_in_period += 1
        candidate_spans = []

        for token, char_start, char_end in iter_token_spans(text):
            local_i = word_to_local.get(token.lower())
            if local_i is not None:
                candidate_spans.append((local_i, char_start, char_end))

        if not candidate_spans:
            now = time.monotonic()
            if (
                now - last_progress_at >= progress_interval_seconds
                or records_seen - last_progress_records >= progress_interval_records
            ):
                logger.info(
                    "[%s] Progress: records_seen=%s, records_in_month=%s, "
                    "records_with_matches=%s, occurrences_queued=%s, "
                    "occurrences_embedded=%s, elapsed=%.1f min",
                    target_half_year,
                    f"{records_seen:,}",
                    f"{records_in_period:,}",
                    f"{records_with_candidates:,}",
                    f"{occurrences_queued:,}",
                    f"{occurrences_embedded:,}",
                    (now - started_at) / 60.0,
                )
                last_progress_at = now
                last_progress_records = records_seen
            continue

        records_with_candidates += 1
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
            occurrences_without_wordpieces += len(candidate_spans)
            continue

        wp_starts = [start for start, end in offsets]
        wp_ends = [end for start, end in offsets]

        for local_i, char_start, char_end in candidate_spans:
            wp_start = bisect.bisect_right(wp_ends, char_start)
            wp_end = bisect.bisect_left(wp_starts, char_end)

            if wp_start >= wp_end:
                occurrences_without_wordpieces += 1
                continue

            window_start = max(0, wp_start - CONTEXT_WORDPIECES)
            window_end = min(
                len(input_ids),
                wp_end + CONTEXT_WORDPIECES,
            )
            window_ids = input_ids[window_start:window_end]

            cls_id = tokenizer.cls_token_id
            sep_id = tokenizer.sep_token_id
            if cls_id is None or sep_id is None:
                raise RuntimeError(
                    "BERT tokenizer is missing CLS or SEP token IDs."
                )

            prepared_ids = [cls_id] + window_ids + [sep_id]
            target_start = wp_start - window_start + 1
            target_end = wp_end - window_start + 1

            if (
                target_start < 1
                or target_end > len(prepared_ids) - 1
                or target_start >= target_end
            ):
                occurrences_without_wordpieces += 1
                continue

            pending_ids.append(prepared_ids)
            pending_targets.append((local_i, target_start, target_end))
            occurrences_queued += 1

            if len(pending_ids) >= batch_size:
                flush_batch()

        now = time.monotonic()
        if (
            now - last_progress_at >= progress_interval_seconds
            or records_seen - last_progress_records >= progress_interval_records
        ):
            logger.info(
                "[%s] Progress: records_seen=%s, records_in_month=%s, "
                "records_with_matches=%s, occurrences_queued=%s, "
                "occurrences_embedded=%s, elapsed=%.1f min",
                target_half_year,
                f"{records_seen:,}",
                f"{records_in_period:,}",
                f"{records_with_candidates:,}",
                f"{occurrences_queued:,}",
                f"{occurrences_embedded:,}",
                (now - started_at) / 60.0,
            )
            last_progress_at = now
            last_progress_records = records_seen

    flush_batch()

    run_stats = {
        "records_seen": records_seen,
        "records_in_period": records_in_period,
        "records_with_candidates": records_with_candidates,
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
        "occurrences_queued": occurrences_queued,
        "occurrences_embedded": occurrences_embedded,
        "occurrences_without_wordpieces": occurrences_without_wordpieces,
        "words_with_embeddings": int((n_occurrences > 0).sum()),
        "active_words": len(active_words),
        "individual_centroids_compared": len(individual_terms),
        "top_k_drug_centroids": min(top_k, len(individual_terms)),
    }

    logger.info(
        "[%s] Embedded %s occurrences for %s words.",
        target_half_year,
        f"{occurrences_embedded:,}",
        f"{run_stats['words_with_embeddings']:,}",
    )

    aggregates = {
        "embedding_sums": embedding_sums,
        "n_occurrences": n_occurrences,
        "cosine_sum": cosine_sum,
        "cosine_sq_sum": cosine_sq_sum,
    }
    return None, run_stats, aggregates


# SHARDS

def write_shard(
    shard_dir,
    half_year,
    cells,
    run_stats,
    files,
    scope,
    freq_path,
    centroid_path,
    centroid_payload,
    individual_centroids,
    top_k,
    model_name,
):
    """Write the half-year's aggregate similarity statistics as JSON."""
    os.makedirs(shard_dir, exist_ok=True)

    final_path = os.path.join(
        shard_dir,
        f"cosine_{half_year}{SHARD_SUFFIX}",
    )
    tmp_path = final_path + ".tmp"
    centroid_meta = centroid_payload.get("drug_centroid", {})

    payload = {
        "shard_name": half_year,
        "model_name": model_name,
        "embedding_dimension": int(
            len(centroid_payload["drug_centroid"]["centroid"])
        ),
        "context_wordpieces_each_side": CONTEXT_WORDPIECES,
        "similarity_target": (
            "global drug centroid and individual drug-term centroids"
        ),
        "embedding_method": (
            "Mean target WordPiece hidden state per occurrence; "
            "aggregate occurrence embeddings by word and half-year"
        ),
        "individual_centroid_similarity_method": (
            "Compare each word's mean contextual embedding for the "
            "half-year against every L2-normalized individual drug-term "
            "centroid; rank by cosine similarity"
        ),
        "top_k_drug_centroids": min(top_k, len(individual_centroids)),
        "occurrence_cosine_std": "population standard deviation",
        "scope": scope,
        "frequency_matrix": {
            "file": os.path.abspath(freq_path),
            "half_year": half_year,
        },
        "drug_centroid": {
            "file": os.path.abspath(centroid_path),
            "metadata": {
                key: value
                for key, value in centroid_meta.items()
                if key != "centroid"
            },
        },
        "individual_drug_centroids": {
            "file": os.path.abspath(centroid_path),
            "count": len(individual_centroids),
            "terms": list(individual_centroids.keys()),
        },
        "statistics": run_stats,
        "cells": cells,
        "files": list(files),
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False)

    os.replace(tmp_path, final_path)
    logger.info(
        "Wrote shard %s (%d word cells)",
        final_path,
        len(cells),
    )
    return final_path


def write_monthly_shard(
    month_key,
    half_year,
    active_words,
    aggregates,
    run_stats,
    files,
    scope,
    freq_path,
):
    """Write sparse sufficient statistics for one month, for exact merging."""
    os.makedirs(MONTHLY_OUTPUT_DIR, exist_ok=True)

    counts = aggregates["n_occurrences"]
    indices = np.flatnonzero(counts > 0).astype(np.int32)

    npz_path = os.path.join(MONTHLY_OUTPUT_DIR, f"month_{month_key}.npz")
    npz_tmp = npz_path + ".tmp"

    with open(npz_tmp, "wb") as f:
        np.savez_compressed(
            f,
            indices=indices,
            embedding_sums=aggregates["embedding_sums"][indices].astype(
                np.float64
            ),
            n_occurrences=counts[indices].astype(np.int64),
            cosine_sum=aggregates["cosine_sum"][indices].astype(np.float64),
            cosine_sq_sum=aggregates["cosine_sq_sum"][indices].astype(
                np.float64
            ),
        )

    os.replace(npz_tmp, npz_path)

    metadata = {
        "month": month_key,
        "half_year": half_year,
        "model_name": MODEL_NAME,
        "embedding_dimension": int(
            aggregates["embedding_sums"].shape[1]
        ),
        "context_wordpieces_each_side": CONTEXT_WORDPIECES,
        "sufficient_statistics": [
            "embedding_sums",
            "n_occurrences",
            "cosine_sum",
            "cosine_sq_sum",
        ],
        "frequency_matrix": {
            "file": os.path.abspath(freq_path),
            "half_year": half_year,
        },
        "scope": scope,
        "files": list(files),
        "statistics": run_stats,
        "active_words_with_occurrences": int(len(indices)),
        "npz_file": os.path.basename(npz_path),
    }

    json_path = os.path.join(MONTHLY_OUTPUT_DIR, f"month_{month_key}.json")
    tmp_path = json_path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, allow_nan=False, indent=2)

    os.replace(tmp_path, json_path)

    logger.info("Wrote monthly sufficient statistics: %s", npz_path)
    logger.info("Wrote monthly metadata: %s", json_path)
    return npz_path


# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    freq, half_years, all_words, freq_path = load_frequency_inputs(
        FREQ_DIR,
        FREQ_STEM,
    )

    # One SLURM array task per month from 2015-01 through 2016-12.
    month_tasks = [
        f"{year}-{month:02d}"
        for year in range(MIN_YEAR, MAX_YEAR + 1)
        for month in range(1, 13)
    ]

    if not 0 <= TASK_ID < len(month_tasks):
        raise ValueError(
            f"TASK_ID={TASK_ID} is invalid. Valid task IDs: "
            f"0-{len(month_tasks) - 1}."
        )

    month_key = month_tasks[TASK_ID]
    year, month = map(int, month_key.split("-"))
    target_half_year = f"{year}H{1 if month <= 6 else 2}"

    if target_half_year not in half_years:
        raise ValueError(
            f"Half-year {target_half_year} is missing from frequency axis."
        )

    freq_row_idx = half_years.index(target_half_year)
    frequency_row = np.asarray(freq[freq_row_idx])

    active_indices = np.flatnonzero(frequency_row > 0)
    active_words = [all_words[i].lower() for i in active_indices]

    if len(set(active_words)) != len(active_words):
        raise ValueError(
            "Lowercasing the frequency vocabulary produced duplicate words."
        )

    frequency_by_word = {
        all_words[i].lower(): int(frequency_row[i])
        for i in active_indices
    }

    logger.info("Month: %s", month_key)
    logger.info("Half-year: %s", target_half_year)
    logger.info("Full vocabulary size: %s", f"{len(all_words):,}")
    logger.info("Positive-frequency words: %s", f"{len(active_words):,}")

    all_paths = resolve_inputs(
        input=None,
        input_dir=None,
        reddit_monthly_dir=REDDIT_MONTHLY_DIR,
    )
    files = get_month_files(all_paths, month_key)

    logger.info(
        "[%s / %s] Selected %d Reddit dump files.",
        month_key,
        target_half_year,
        len(files),
    )

    allowed = load_subreddits(SUBREDDIT_LIST_FILE)
    freq_meta = load_freq_meta(FREQ_DIR, FREQ_STEM)
    verify_scope(freq_meta, allowed, SUBREDDIT_LIST_FILE)

    centroid, centroid_payload = load_drug_centroid(DRUG_CENTROID_PATH)
    tokenizer, model, device = load_model(MODEL_NAME)
    hidden_size = int(model.config.hidden_size)

    if centroid.shape[0] != hidden_size:
        raise ValueError(
            f"Global centroid dimension {centroid.shape[0]} does not "
            f"match {MODEL_NAME} hidden size {hidden_size}."
        )

    individual_centroids, _ = load_individual_centroids(
        DRUG_CENTROID_PATH,
        hidden_size=hidden_size,
    )

    drop_counts = defaultdict(int)
    stream = stream_corpus(
        files,
        MIN_YEAR,
        MAX_YEAR,
        allowed,
        drop_counts,
    )

    _, run_stats, aggregates = process_half_year(
        stream=stream,
        target_half_year=target_half_year,
        active_words=active_words,
        frequency_by_word=frequency_by_word,
        centroid=centroid,
        individual_centroids=individual_centroids,
        top_k=TOP_K_DRUG_CENTROIDS,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=BATCH_SIZE,
    )

    run_stats["corpus_drop_counts"] = dict(drop_counts)

    scope = {
        "mode": "subreddits" if allowed is not None else "all",
        "file": os.path.abspath(SUBREDDIT_LIST_FILE),
        "hash": subreddit_hash(allowed),
    }

    output_path = write_monthly_shard(
        month_key=month_key,
        half_year=target_half_year,
        active_words=active_words,
        aggregates=aggregates,
        run_stats=run_stats,
        files=files,
        scope=scope,
        freq_path=freq_path,
    )

    logger.info("Saved monthly sufficient statistics: %s", output_path)


if __name__ == "__main__":
    main()