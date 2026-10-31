#!/usr/bin/env python3
"""Merge monthly cosine-similarity sufficient statistics into half-year shards."""

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
    return sum(int(meta.get("statistics", {}).get(key, 0)) for meta in metadata_list)


def merge_half_year(
    half_year,
    freq,
    half_years,
    all_words,
    freq_path,
    allowed,
    scope,
    centroid,
    centroid_payload,
    individual_centroids,
):
    """Merge six months of sparse sufficient statistics for one half-year."""
    month_numbers = range(1, 7) if half_year.endswith("H1") else range(7, 13)
    year = int(half_year[:4])
    month_keys = [f"{year}-{month:02d}" for month in month_numbers]

    metadata_list = []
    loaded_arrays = []

    for month_key in month_keys:
        json_path = os.path.join(MONTHLY_DIR, f"month_{month_key}.json")
        npz_path = os.path.join(MONTHLY_DIR, f"month_{month_key}.npz")

        if not os.path.isfile(json_path) or not os.path.isfile(npz_path):
            raise FileNotFoundError(
                f"Missing monthly output for {month_key}: expected both "
                f"{json_path} and {npz_path}. Refusing to merge incomplete data."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("half_year") != half_year:
            raise ValueError(
                f"{json_path} declares half_year={meta.get('half_year')!r}."
            )

        if meta.get("model_name") != sm.MODEL_NAME:
            raise ValueError(f"Model mismatch in {json_path}.")

        if meta.get("scope", {}).get("hash") != scope["hash"]:
            raise ValueError(f"Subreddit scope mismatch in {json_path}.")

        metadata_list.append(meta)

        with np.load(npz_path) as data:
            loaded_arrays.append({
                "indices": data["indices"].copy(),
                "embedding_sums": data["embedding_sums"].copy(),
                "n_occurrences": data["n_occurrences"].copy(),
                "cosine_sum": data["cosine_sum"].copy(),
                "cosine_sq_sum": data["cosine_sq_sum"].copy(),
            })

    if half_year not in half_years:
        raise ValueError(f"{half_year} is absent from the frequency matrix.")

    row_idx = half_years.index(half_year)
    frequency_row = np.asarray(freq[row_idx])
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

    hidden_size = len(centroid)
    embedding_sums = np.zeros(
        (len(active_words), hidden_size), dtype=np.float64
    )
    n_occurrences = np.zeros(len(active_words), dtype=np.int64)
    cosine_sum = np.zeros(len(active_words), dtype=np.float64)
    cosine_sq_sum = np.zeros(len(active_words), dtype=np.float64)

    for arrays in loaded_arrays:
        indices = arrays["indices"].astype(np.int64, copy=False)
        embedding_sums[indices] += arrays["embedding_sums"]
        n_occurrences[indices] += arrays["n_occurrences"]
        cosine_sum[indices] += arrays["cosine_sum"]
        cosine_sq_sum[indices] += arrays["cosine_sq_sum"]

    centroid_norm = float(np.linalg.norm(centroid))
    individual_terms = list(individual_centroids.keys())
    individual_vectors = np.stack(
        [individual_centroids[term] for term in individual_terms]
    ).astype(np.float64)

    cells = {}

    for i, word in enumerate(active_words):
        count = int(n_occurrences[i])
        frequency = int(frequency_by_word[word])

        if count == 0:
            cells[word] = {
                "n_occurrences": 0,
                "cosine_mean_embedding": None,
                "mean_occurrence_cosine": None,
                "std_occurrence_cosine": None,
                "frequency": frequency,
                "coverage": 0.0 if frequency > 0 else None,
                "mean_embedding_norm": None,
                "top_k_drug_terms": [],
            }
            continue

        mean_embedding = embedding_sums[i] / count
        mean_norm = float(np.linalg.norm(mean_embedding))

        if mean_norm > 0.0 and math.isfinite(mean_norm):
            cosine_mean_embedding = float(
                np.dot(mean_embedding, centroid) / (mean_norm * centroid_norm)
            )

            normalized_mean_embedding = mean_embedding / mean_norm
            individual_scores = individual_vectors @ normalized_mean_embedding
            ranked_indices = np.argsort(individual_scores)[::-1]
            k = min(sm.TOP_K_DRUG_CENTROIDS, len(individual_terms))

            top_k_drug_terms = [
                {
                    "term": individual_terms[int(j)],
                    "cosine_similarity": float(individual_scores[j]),
                }
                for j in ranked_indices[:k]
            ]
        else:
            cosine_mean_embedding = None
            top_k_drug_terms = []

        mean_occurrence_cosine = float(cosine_sum[i] / count)
        variance = cosine_sq_sum[i] / count - mean_occurrence_cosine ** 2

        cells[word] = {
            "n_occurrences": count,
            "cosine_mean_embedding": cosine_mean_embedding,
            "mean_occurrence_cosine": mean_occurrence_cosine,
            "std_occurrence_cosine": math.sqrt(max(0.0, float(variance))),
            "frequency": frequency,
            "coverage": float(count / frequency) if frequency > 0 else None,
            "mean_embedding_norm": mean_norm,
            "top_k_drug_terms": top_k_drug_terms,
        }

    drop_counts = defaultdict(int)

    for meta in metadata_list:
        for key, value in meta.get("statistics", {}).get(
            "corpus_drop_counts", {}
        ).items():
            drop_counts[key] += int(value)

    run_stats = {
        "records_in_period": sum_stat(metadata_list, "records_in_period"),
        "occurrences_queued": sum_stat(metadata_list, "occurrences_queued"),
        "occurrences_embedded": int(n_occurrences.sum()),
        "occurrences_without_wordpieces": sum_stat(
            metadata_list, "occurrences_without_wordpieces"
        ),
        "words_with_embeddings": int((n_occurrences > 0).sum()),
        "active_words": len(active_words),
        "individual_centroids_compared": len(individual_terms),
        "top_k_drug_centroids": min(
            sm.TOP_K_DRUG_CENTROIDS, len(individual_terms)
        ),
        "corpus_drop_counts": dict(drop_counts),
        "monthly_tasks_merged": len(metadata_list),
    }

    files = sorted({
        path
        for meta in metadata_list
        for path in meta.get("files", [])
    })

    output_path = sm.write_shard(
        shard_dir=OUTPUT_DIR,
        half_year=half_year,
        cells=cells,
        run_stats=run_stats,
        files=files,
        scope=scope,
        freq_path=freq_path,
        centroid_path=sm.DRUG_CENTROID_PATH,
        centroid_payload=centroid_payload,
        individual_centroids=individual_centroids,
        top_k=sm.TOP_K_DRUG_CENTROIDS,
        model_name=sm.MODEL_NAME,
    )

    logger.info("Merged %s from six months -> %s", half_year, output_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    freq, half_years, all_words, freq_path = sm.load_frequency_inputs(
        sm.FREQ_DIR, sm.FREQ_STEM
    )

    allowed = sm.load_subreddits(sm.SUBREDDIT_LIST_FILE)
    freq_meta = sm.load_freq_meta(sm.FREQ_DIR, sm.FREQ_STEM)
    sm.verify_scope(freq_meta, allowed, sm.SUBREDDIT_LIST_FILE)

    scope = {
        "mode": "subreddits" if allowed is not None else "all",
        "file": os.path.abspath(sm.SUBREDDIT_LIST_FILE),
        "hash": sm.subreddit_hash(allowed),
    }

    centroid, centroid_payload = sm.load_drug_centroid(sm.DRUG_CENTROID_PATH)

    # The merge only needs centroid vectors, not the transformer model.
    individual_centroids, _ = sm.load_individual_centroids(
        sm.DRUG_CENTROID_PATH, hidden_size=len(centroid)
    )

    for year in YEARS:
        for half in (1, 2):
            half_year = f"{year}H{half}"
            merge_half_year(
                half_year,
                freq,
                half_years,
                all_words,
                freq_path,
                allowed,
                scope,
                centroid,
                centroid_payload,
                individual_centroids,
            )


if __name__ == "__main__":
    main()