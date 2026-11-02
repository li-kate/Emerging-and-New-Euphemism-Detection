
#!/usr/bin/env python3
"""Assemble half-year JSON shards into vocabulary-by-half-year NumPy matrices."""

import json
import logging
import os

import numpy as np

import similarity_matrix as sm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


SHARD_DIR = sm.OUTPUT_DIR
MATRIX_DIR = "./cosine_similarity_results/matrices"

YEARS = range(sm.MIN_YEAR, sm.MAX_YEAR + 1)

METRICS = (
    "cosine_mean_embedding",
    "std_occurrence_cosine",
    "mean_embedding_norm",
)


SKIP_HALF_YEARS = {
    "2024H2",
    "2026H2",
}


def load_available_half_years():
    """Return half-years present in the centroid JSON and configured year range."""

    with open(sm.DRUG_CENTROID_PATH, "r", encoding="utf-8") as f:
        centroid_data = json.load(f)

    available = centroid_data.get("half_years", {})

    half_years = [
        f"{year}H{half}"
        for year in YEARS
        for half in (1, 2)
        if (
            f"{year}H{half}" in available
            and f"{year}H{half}" not in SKIP_HALF_YEARS
        )
    ]

    if not half_years:
        raise ValueError(
            "No available half-years found in centroid JSON "
            f"for years {sm.MIN_YEAR}-{sm.MAX_YEAR}."
        )

    return half_years


def validate_shard(payload, half_year, vocabulary_hash, vocabulary_size):
    """Validate shard metadata before inserting its values into the matrices."""

    if payload.get("half_year") != half_year:
        raise ValueError(
            f"{half_year}: incorrect half_year metadata: "
            f"{payload.get('half_year')}"
        )

    if payload.get("centroid_half_year") != half_year:
        raise ValueError(
            f"{half_year}: incorrect centroid_half_year metadata: "
            f"{payload.get('centroid_half_year')}"
        )

    if payload.get("similarity_target") != "half-year drug centroid":
        raise ValueError(
            f"{half_year}: unexpected similarity target: "
            f"{payload.get('similarity_target')}"
        )

    if payload.get("model_name") != sm.MODEL_NAME:
        raise ValueError(
            f"{half_year}: model mismatch: {payload.get('model_name')}"
        )

    if payload.get("embedding_dimension") != 768:
        raise ValueError(
            f"{half_year}: unexpected embedding dimension: "
            f"{payload.get('embedding_dimension')}"
        )

    if payload.get("vocabulary_size") != vocabulary_size:
        raise ValueError(
            f"{half_year}: vocabulary size mismatch: "
            f"{payload.get('vocabulary_size')} != {vocabulary_size}"
        )

    if payload.get("vocabulary_sha256") != vocabulary_hash:
        raise ValueError(
            f"{half_year}: vocabulary hash mismatch.\n"
            f"Shard:   {payload.get('vocabulary_sha256')}\n"
            f"Current: {vocabulary_hash}"
        )

    cells = payload.get("cells")
    if not isinstance(cells, dict):
        raise ValueError(f"{half_year}: missing or invalid cells dictionary.")

    expected_active = payload.get("active_words_with_occurrences")
    if expected_active != len(cells):
        raise ValueError(
            f"{half_year}: active word count mismatch: "
            f"{expected_active} != {len(cells)}"
        )

    return cells


def main():
    os.makedirs(MATRIX_DIR, exist_ok=True)

    logger.info("Loading vocabulary...")
    words, vocabulary_hash = sm.load_vocabulary(sm.VOCAB_FILE)

    vocabulary_size = len(words)
    word_to_col = {word: i for i, word in enumerate(words)}

    if len(word_to_col) != vocabulary_size:
        raise ValueError("Vocabulary contains duplicate words.")

    logger.info("Vocabulary size: %s", vocabulary_size)
    logger.info("Vocabulary SHA-256: %s", vocabulary_hash)

    half_years = load_available_half_years()

    logger.info("Half-years to assemble: %s", half_years)

    shape = (len(half_years), vocabulary_size)

    matrices = {
        metric: np.full(shape, np.nan, dtype=np.float32)
        for metric in METRICS
    }

    n_occurrences = np.zeros(shape, dtype=np.int64)

    for row, half_year in enumerate(half_years):
        shard_path = os.path.join(
            SHARD_DIR,
            f"cosine_{half_year}.json",
        )

        if not os.path.isfile(shard_path):
            raise FileNotFoundError(
                f"Missing half-year shard: {shard_path}"
            )

        logger.info("Loading %s...", shard_path)

        with open(shard_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        cells = validate_shard(
            payload,
            half_year,
            vocabulary_hash,
            vocabulary_size,
        )

        inserted = 0

        for word, cell in cells.items():
            normalized_word = word.lower()
            col = word_to_col.get(normalized_word)

            if col is None:
                raise ValueError(
                    f"{half_year}: word not found in vocabulary: {word!r}"
                )

            count = int(cell.get("n_occurrences", 0))

            if count <= 0:
                raise ValueError(
                    f"{half_year}: invalid occurrence count for {word!r}: "
                    f"{count}"
                )

            n_occurrences[row, col] = count

            for metric in METRICS:
                value = cell.get(metric)

                if value is None:
                    continue

                value = float(value)

                if not np.isfinite(value):
                    raise ValueError(
                        f"{half_year}: non-finite {metric} for {word!r}"
                    )

                matrices[metric][row, col] = value

            inserted += 1

        logger.info(
            "%s: inserted %s words; total occurrences: %s",
            half_year,
            inserted,
            int(n_occurrences[row].sum()),
        )

    # Save metric matrices.
    for metric, matrix in matrices.items():
        output_path = os.path.join(
            MATRIX_DIR,
            f"similarity_{metric}.npy",
        )

        np.save(output_path, matrix)
        logger.info("Saved %s", output_path)

    # Save occurrence-count matrix.
    count_path = os.path.join(
        MATRIX_DIR,
        "similarity_n_occurrences.npy",
    )

    np.save(count_path, n_occurrences)
    logger.info("Saved %s", count_path)

    # Save metadata describing matrix dimensions and ordering.
    metadata = {
        "half_years": half_years,
        "vocabulary_file": sm.VOCAB_FILE,
        "vocabulary_size": vocabulary_size,
        "vocabulary_sha256": vocabulary_hash,
        "model_name": sm.MODEL_NAME,
        "embedding_dimension": 768,
        "similarity_target": "half-year drug centroid",
        "metrics": list(METRICS),
        "matrix_shape": list(shape),
        "matrix_dtype": "float32",
        "occurrence_count_dtype": "int64",
        "missing_metric_value": "NaN",
        "missing_occurrence_count": 0,
        "matrix_files": {
            metric: f"similarity_{metric}.npy"
            for metric in METRICS
        },
        "occurrence_count_file": "similarity_n_occurrences.npy",
    }

    metadata_path = os.path.join(MATRIX_DIR, "metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved %s", metadata_path)
    logger.info("Matrix assembly complete.")
    logger.info("Final matrix shape: %s", shape)


if __name__ == "__main__":
    main()
