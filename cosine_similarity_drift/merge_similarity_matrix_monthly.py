
#!/usr/bin/env python3
"""Merge monthly sparse sufficient statistics into half-year JSON shards."""

import json
import logging
import math
import os
from collections import defaultdict

import numpy as np

import similarity_matrix as sm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MONTHLY_DIR = sm.MONTHLY_OUTPUT_DIR
OUTPUT_DIR = sm.OUTPUT_DIR

SKIP_HALF_YEARS = {
    "2024H2",
    "2026H2",
}


def sum_stat(metadata_list, key):
    return sum(
        float(meta.get("statistics", {}).get(key, 0))
        for meta in metadata_list
    )


def load_monthly_files(
    month_key,
    half_year,
    expected_scope_hash,
    expected_vocabulary_hash,
):
    """Load and validate one month's JSON metadata and NPZ arrays."""

    json_path = os.path.join(MONTHLY_DIR, f"month_{month_key}.json")
    npz_path = os.path.join(MONTHLY_DIR, f"month_{month_key}.npz")

    if not os.path.isfile(json_path) or not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Missing monthly output for {month_key}: expected "
            f"{json_path} and {npz_path}."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("month") != month_key:
        raise ValueError(
            f"{json_path} declares month={metadata.get('month')!r}."
        )

    if metadata.get("half_year") != half_year:
        raise ValueError(
            f"{json_path} declares half_year="
            f"{metadata.get('half_year')!r}; expected {half_year!r}."
        )

    if metadata.get("centroid_half_year") != half_year:
        raise ValueError(
            f"{json_path} used centroid "
            f"{metadata.get('centroid_half_year')!r}; "
            f"expected {half_year!r}."
        )

    if metadata.get("model_name") != sm.MODEL_NAME:
        raise ValueError(f"Model mismatch in {json_path}.")

    if metadata.get("scope", {}).get("hash") != expected_scope_hash:
        raise ValueError(f"Subreddit scope mismatch in {json_path}.")

    if metadata.get("vocabulary_sha256") != expected_vocabulary_hash:
        raise ValueError(f"Vocabulary hash mismatch in {json_path}.")

    with np.load(npz_path) as data:
        arrays = {
            "indices": data["indices"].astype(np.int64, copy=True),
            "embedding_sums": data["embedding_sums"].astype(
                np.float64, copy=True
            ),
            "n_occurrences": data["n_occurrences"].astype(
                np.int64, copy=True
            ),
            "cosine_sum": data["cosine_sum"].astype(
                np.float64, copy=True
            ),
            "cosine_sq_sum": data["cosine_sq_sum"].astype(
                np.float64, copy=True
            ),
        }

    indices = arrays["indices"]
    embedding_sums = arrays["embedding_sums"]

    if indices.ndim != 1:
        raise ValueError(f"Invalid indices shape in {npz_path}.")

    if embedding_sums.ndim != 2:
        raise ValueError(f"Invalid embedding_sums shape in {npz_path}.")

    if embedding_sums.shape[0] != len(indices):
        raise ValueError(f"Embedding/index length mismatch in {npz_path}.")

    for key in ("n_occurrences", "cosine_sum", "cosine_sq_sum"):
        if arrays[key].ndim != 1 or len(arrays[key]) != len(indices):
            raise ValueError(f"Invalid {key} shape in {npz_path}.")

    if np.any(arrays["n_occurrences"] < 0):
        raise ValueError(f"Negative occurrence count in {npz_path}.")

    if not all(
        np.isfinite(arrays[key]).all()
        for key in ("embedding_sums", "cosine_sum", "cosine_sq_sum")
    ):
        raise ValueError(f"Non-finite values found in {npz_path}.")

    return metadata, arrays


def write_half_year_shard(
    half_year,
    cells,
    run_stats,
    files,
    scope,
    vocabulary_hash,
    vocabulary_size,
    centroid_payload,
):
    """Write a half-year JSON shard for the matrix assembly stage."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    centroid_meta = centroid_payload.get("drug_centroid", {})

    payload = {
        "half_year": half_year,
        "model_name": sm.MODEL_NAME,
        "embedding_dimension": int(
            len(centroid_meta.get("centroid", []))
        ),
        "similarity_target": "half-year drug centroid",
        "centroid_half_year": half_year,
        "occurrence_cosine_std": "population standard deviation",
        "drug_centroid": {
            "file": os.path.abspath(sm.DRUG_CENTROID_PATH),
            "half_year": half_year,
            "count": centroid_meta.get("count"),
        },
        "vocabulary_file": os.path.abspath(sm.VOCAB_FILE),
        "vocabulary_size": int(vocabulary_size),
        "vocabulary_sha256": vocabulary_hash,
        "scope": scope,
        "metrics": [
            "n_occurrences",
            "cosine_mean_embedding",
            "std_occurrence_cosine",
            "mean_embedding_norm",
        ],
        "statistics": run_stats,
        "files": list(files),
        "active_words_with_occurrences": len(cells),
        "cells": cells,
    }

    output_path = os.path.join(
        OUTPUT_DIR,
        f"cosine_{half_year}.json",
    )
    temp_path = output_path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, allow_nan=False, indent=2)

    os.replace(temp_path, output_path)

    return output_path


def merge_half_year(
    half_year,
    all_words,
    vocabulary_hash,
    scope,
):
    """Merge the six monthly shards for one half-year."""

    centroid, centroid_payload = sm.load_drug_centroid(
        sm.DRUG_CENTROID_PATH,
        half_year,
    )

    year = int(half_year[:4])
    month_numbers = (
        range(1, 7) if half_year.endswith("H1") else range(7, 13)
    )
    month_keys = [
        f"{year}-{month:02d}"
        for month in month_numbers
    ]

    metadata_list = []

    embedding_sums = {}
    n_occurrences = defaultdict(int)
    cosine_sum = defaultdict(float)
    cosine_sq_sum = defaultdict(float)

    for month_key in month_keys:
        metadata, arrays = load_monthly_files(
            month_key=month_key,
            half_year=half_year,
            expected_scope_hash=scope["hash"],
            expected_vocabulary_hash=vocabulary_hash,
        )
        metadata_list.append(metadata)

        indices = arrays["indices"]
        sums = arrays["embedding_sums"]

        if np.any(indices < 0) or np.any(indices >= len(all_words)):
            raise ValueError(
                f"{month_key} contains vocabulary indices outside "
                f"[0, {len(all_words)})."
            )

        for index, vector, count, cos_sum, cos_sq_sum in zip(
            indices,
            sums,
            arrays["n_occurrences"],
            arrays["cosine_sum"],
            arrays["cosine_sq_sum"],
        ):
            index = int(index)

            if index not in embedding_sums:
                embedding_sums[index] = np.zeros_like(vector)

            embedding_sums[index] += vector
            n_occurrences[index] += int(count)
            cosine_sum[index] += float(cos_sum)
            cosine_sq_sum[index] += float(cos_sq_sum)

    centroid_norm = float(np.linalg.norm(centroid))
    cells = {}

    for index in sorted(n_occurrences):
        count = int(n_occurrences[index])

        if count <= 0:
            continue

        word = all_words[index]
        mean_embedding = embedding_sums[index] / count
        mean_norm = float(np.linalg.norm(mean_embedding))

        if (
            mean_norm > 0.0
            and math.isfinite(mean_norm)
            and centroid_norm > 0.0
        ):
            cosine_mean_embedding = float(
                np.dot(mean_embedding, centroid)
                / (mean_norm * centroid_norm)
            )
        else:
            cosine_mean_embedding = None

        # Population standard deviation of occurrence-level cosine values.
        mean_occurrence_cosine = cosine_sum[index] / count
        variance = (
            cosine_sq_sum[index] / count
            - mean_occurrence_cosine ** 2
        )

        cells[word] = {
            "n_occurrences": count,
            "cosine_mean_embedding": cosine_mean_embedding,
            "std_occurrence_cosine": math.sqrt(
                max(0.0, float(variance))
            ),
            "mean_embedding_norm": mean_norm,
        }

    drop_counts = defaultdict(int)

    for metadata in metadata_list:
        for key, value in metadata.get(
            "statistics", {}
        ).get("corpus_drop_counts", {}).items():
            drop_counts[key] += int(value)

    run_stats = {
        "months_merged": month_keys,
        "monthly_tasks_merged": len(metadata_list),
        "files_seen": int(sum_stat(metadata_list, "files_seen")),
        "files_completed": int(sum_stat(metadata_list, "files_completed")),
        "records_seen": int(sum_stat(metadata_list, "records_seen")),
        "records_with_matches": int(
            sum_stat(metadata_list, "records_with_matches")
        ),
        "occurrences_queued": int(
            sum_stat(metadata_list, "occurrences_queued")
        ),
        "occurrences_embedded": int(
            sum_stat(metadata_list, "occurrences_embedded")
        ),
        "words_with_embeddings": len(cells),
        "vocabulary_size": len(all_words),
        "elapsed_seconds": round(
            sum_stat(metadata_list, "elapsed_seconds"), 2
        ),
        "corpus_drop_counts": dict(drop_counts),
    }

    files = sorted({
        path
        for metadata in metadata_list
        for path in metadata.get("files", [])
    })

    output_path = write_half_year_shard(
        half_year=half_year,
        cells=cells,
        run_stats=run_stats,
        files=files,
        scope=scope,
        vocabulary_hash=vocabulary_hash,
        vocabulary_size=len(all_words),
        centroid_payload=centroid_payload,
    )

    logger.info(
        "Merged %s from six months -> %s (%s words)",
        half_year,
        output_path,
        f"{len(cells):,}",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # similarity_matrix.load_vocabulary takes VOCAB_FILE as its only argument.
    all_words, vocabulary_hash = sm.load_vocabulary(sm.VOCAB_FILE)

    allowed = sm.load_subreddits(sm.SUBREDDIT_LIST_FILE)

    if not allowed:
        raise ValueError(
            "The subreddit allowlist is empty. Check SUBREDDIT_LIST_FILE."
        )

    scope = {
        "mode": "subreddits",
        "file": os.path.abspath(sm.SUBREDDIT_LIST_FILE),
        "hash": sm.subreddit_hash(allowed),
    }

    # Only process half-years for which a centroid was actually generated.
    with open(sm.DRUG_CENTROID_PATH, "r", encoding="utf-8") as f:
        centroid_data = json.load(f)

    available_half_years = sorted(
        half_year
        for half_year in centroid_data.get("half_years", {})
        if (
            len(half_year) == 6
            and half_year[:4].isdigit()
            and sm.MIN_YEAR <= int(half_year[:4]) <= sm.MAX_YEAR
            and half_year not in SKIP_HALF_YEARS
        )
    )

    if not available_half_years:
        raise ValueError(
            "No available half-year centroids fall within "
            f"{sm.MIN_YEAR}-{sm.MAX_YEAR}."
        )

    logger.info("Half-years to merge: %s", available_half_years)

    for half_year in available_half_years:
        merge_half_year(
            half_year=half_year,
            all_words=all_words,
            vocabulary_hash=vocabulary_hash,
            scope=scope,
        )


if __name__ == "__main__":
    main()
