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
YEARS = range(sm.MIN_YEAR, sm.MAX_YEAR + 1)


def sum_stat(metadata_list, key):
    return sum(
        int(meta.get("statistics", {}).get(key, 0))
        for meta in metadata_list
    )


def load_monthly_files(month_key, half_year, expected_scope_hash):
    json_path = os.path.join(MONTHLY_DIR, f"month_{month_key}.json")
    npz_path = os.path.join(MONTHLY_DIR, f"month_{month_key}.npz")

    if not os.path.isfile(json_path) or not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Missing monthly output for {month_key}: expected {json_path} "
            f"and {npz_path}."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("half_year") != half_year:
        raise ValueError(
            f"{json_path} declares half_year={metadata.get('half_year')!r}."
        )

    if metadata.get("model_name") != sm.MODEL_NAME:
        raise ValueError(f"Model mismatch in {json_path}.")

    if metadata.get("scope", {}).get("hash") != expected_scope_hash:
        raise ValueError(f"Subreddit scope mismatch in {json_path}.")

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

    lengths = {len(value) for value in arrays.values() if value.ndim == 1}
    if len(lengths) > 1:
        raise ValueError(f"Inconsistent sparse-array lengths in {npz_path}.")

    if arrays["embedding_sums"].shape[0] != len(arrays["indices"]):
        raise ValueError(f"Embedding/index length mismatch in {npz_path}.")

    return metadata, arrays


def merge_half_year(half_year, all_words, scope, centroid, centroid_payload):
    month_numbers = range(1, 7) if half_year.endswith("H1") else range(7, 13)
    year = int(half_year[:4])
    month_keys = [f"{year}-{month:02d}" for month in month_numbers]

    metadata_list = []
    embedding_sums = {}
    n_occurrences = defaultdict(int)
    cosine_sum = defaultdict(float)
    cosine_sq_sum = defaultdict(float)

    for month_key in month_keys:
        metadata, arrays = load_monthly_files(
            month_key,
            half_year,
            scope["hash"],
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

        # Used only to derive the variance; not written as an output metric.
        mean_occurrence_cosine = float(cosine_sum[index] / count)
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
        for key, value in metadata.get("statistics", {}).get(
            "corpus_drop_counts", {}
        ).items():
            drop_counts[key] += int(value)

    run_stats = {
        "records_seen": sum_stat(metadata_list, "records_seen"),
        "records_in_period": sum_stat(metadata_list, "records_in_period"),
        "records_with_candidates": sum_stat(
            metadata_list, "records_with_candidates"
        ),
        "occurrences_queued": sum_stat(metadata_list, "occurrences_queued"),
        "occurrences_embedded": sum_stat(
            metadata_list, "occurrences_embedded"
        ),
        "occurrences_without_wordpieces": sum_stat(
            metadata_list, "occurrences_without_wordpieces"
        ),
        "words_with_embeddings": len(cells),
        "vocabulary_words": len(all_words),
        "corpus_drop_counts": dict(drop_counts),
        "monthly_tasks_merged": len(metadata_list),
    }

    files = sorted({
        path
        for metadata in metadata_list
        for path in metadata.get("files", [])
    })

    output_path = sm.write_shard(
        shard_dir=OUTPUT_DIR,
        half_year=half_year,
        cells=cells,
        run_stats=run_stats,
        files=files,
        scope=scope,
        centroid_path=sm.DRUG_CENTROID_PATH,
        centroid_payload=centroid_payload,
        model_name=sm.MODEL_NAME,
    )

    logger.info(
        "Merged %s from six months -> %s (%s words with occurrences)",
        half_year,
        output_path,
        f"{len(cells):,}",
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_words = sm.load_vocabulary(sm.FREQ_DIR, sm.FREQ_STEM)

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

    centroid, centroid_payload = sm.load_drug_centroid(
        sm.DRUG_CENTROID_PATH
    )

    for year in YEARS:
        for half in (1, 2):
            merge_half_year(
                f"{year}H{half}",
                all_words,
                scope,
                centroid,
                centroid_payload,
            )


if __name__ == "__main__":
    main()
