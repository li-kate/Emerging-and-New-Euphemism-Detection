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


def main():
    os.makedirs(MATRIX_DIR, exist_ok=True)

    words = sm.load_vocabulary(sm.FREQ_DIR, sm.FREQ_STEM)
    word_to_col = {word: i for i, word in enumerate(words)}

    half_years = [
        f"{year}H{half}"
        for year in YEARS
        for half in (1, 2)
    ]

    shape = (len(half_years), len(words))
    logger.info(
        "Creating output matrices with shape %s (%s vocabulary words).",
        shape,
        f"{len(words):,}",
    )

    # NaN means that no embedding was produced for that word/period.
    matrices = {
        metric: np.full(shape, np.nan, dtype=np.float32)
        for metric in METRICS
    }
    n_occurrences = np.zeros(shape, dtype=np.int32)

    for row, half_year in enumerate(half_years):
        shard_path = os.path.join(
            SHARD_DIR,
            f"cosine_{half_year}.json",
        )

        if not os.path.isfile(shard_path):
            raise FileNotFoundError(
                f"Missing half-year shard: {shard_path}"
            )

        with open(shard_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("model_name") != sm.MODEL_NAME:
            raise ValueError(f"Model mismatch in {shard_path}.")

        if payload.get("context_wordpieces_each_side") != sm.CONTEXT_WORDPIECES:
            raise ValueError(f"Context-window mismatch in {shard_path}.")

        cells = payload.get("cells", {})

        for word, cell in cells.items():
            normalized_word = word.lower()
            col = word_to_col.get(normalized_word)

            if col is None:
                raise ValueError(
                    f"Word {word!r} in {shard_path} is not in freq_words.csv."
                )

            count = int(cell.get("n_occurrences", 0))
            if count < 0:
                raise ValueError(
                    f"Negative occurrence count for {word!r} in {shard_path}."
                )

            n_occurrences[row, col] = count

            for metric in METRICS:
                value = cell.get(metric)
                if value is not None:
                    matrices[metric][row, col] = float(value)

        logger.info(
            "Loaded %s: %s words with embeddings",
            half_year,
            f"{len(cells):,}",
        )

    for metric, matrix in matrices.items():
        path = os.path.join(MATRIX_DIR, f"similarity_{metric}.npy")
        np.save(path, matrix)
        logger.info("Saved %s", path)

    counts_path = os.path.join(
        MATRIX_DIR,
        "similarity_n_occurrences.npy",
    )
    np.save(counts_path, n_occurrences)
    logger.info("Saved %s", counts_path)

    # Save labels so matrix rows and columns can be interpreted.
    labels_path = os.path.join(MATRIX_DIR, "similarity_matrix_metadata.json")
    metadata = {
        "model_name": sm.MODEL_NAME,
        "context_wordpieces_each_side": sm.CONTEXT_WORDPIECES,
        "half_years": half_years,
        "vocabulary_file": os.path.abspath(
            os.path.join(sm.FREQ_DIR, f"{sm.FREQ_STEM}_words.csv")
        ),
        "vocabulary_size": len(words),
        "matrix_shape": list(shape),
        "metrics": list(METRICS),
        "occurrence_count_file": os.path.basename(counts_path),
        "missing_value": "NaN means no occurrences were embedded.",
    }

    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved matrix metadata: %s", labels_path)


if __name__ == "__main__":
    main()
