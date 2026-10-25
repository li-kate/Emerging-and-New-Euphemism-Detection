
"""Merge per-half-year cosine similarity JSON shards into aligned matrices."""

import csv
import json
import logging
import math
import os

import numpy as np


# CONFIGURATION
FREQ_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301"
OUTPUT_DIR = "./cosine_similarity_results"
SHARD_DIR = os.path.join(OUTPUT_DIR, "shards")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def read_csv_column(path, preferred_column):
    """Read a named CSV column, with support for headerless files."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)

        headers = next(csv.reader([first_line]), [])

        if preferred_column in headers:
            return [
                row[preferred_column].strip()
                for row in csv.DictReader(f)
            ]

        return [
            row[0].strip()
            for row in csv.reader(f)
            if row and row[0].strip()
        ]


def load_frequency_inputs():
    """Load the reference matrix and its row/column labels."""
    freq_path = os.path.join(FREQ_DIR, "freq.npy")
    half_years_path = os.path.join(
        FREQ_DIR, "freq_half_years.csv"
    )
    words_path = os.path.join(
        FREQ_DIR, "freq_words.csv"
    )

    for path in (freq_path, half_years_path, words_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    freq = np.load(freq_path, mmap_mode="r")
    half_years = read_csv_column(half_years_path, "half_year")
    words = read_csv_column(words_path, "word")

    if freq.ndim != 2:
        raise ValueError(
            f"Expected a 2-D frequency matrix; got {freq.shape}."
        )

    expected_shape = (len(half_years), len(words))
    if freq.shape != expected_shape:
        raise ValueError(
            f"Frequency matrix shape {freq.shape} does not match "
            f"the CSV axes {expected_shape}."
        )

    if len(set(half_years)) != len(half_years):
        raise ValueError("Duplicate half-year labels.")

    normalized_words = [word.lower() for word in words]
    if len(set(normalized_words)) != len(normalized_words):
        raise ValueError(
            "Frequency vocabulary contains duplicate words after "
            "lowercasing, so shard cells cannot be mapped uniquely."
        )

    if np.any(np.asarray(freq) < 0):
        raise ValueError("Frequency matrix contains negative counts.")

    logger.info(
        "Loaded frequency matrix %s with shape %s",
        freq_path,
        freq.shape,
    )

    return freq, half_years, words, normalized_words, freq_path


def load_shard(path, expected_half_year):
    """Load and validate one similarity_matrix.py JSON shard."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object.")

    actual_half_year = payload.get("shard_name")
    if actual_half_year != expected_half_year:
        raise ValueError(
            f"{path}: shard_name={actual_half_year!r}; "
            f"expected {expected_half_year!r}."
        )

    cells = payload.get("cells")
    if not isinstance(cells, dict):
        raise ValueError(f"{path}: missing dictionary 'cells'.")

    for word, record in cells.items():
        if not isinstance(record, dict):
            raise ValueError(
                f"{path}: cell {word!r} must contain a JSON object."
            )

    return payload, cells


def numeric_value(record, key):
    """Return a finite numeric value, or NaN for JSON null/missing data."""
    value = record.get(key)
    if value is None:
        return math.nan
    if isinstance(value, bool):
        raise ValueError(
            f"Expected a numeric statistic for {key!r}, got bool."
        )
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric value for {key!r}: {value!r}"
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Non-finite value for {key!r}: {value!r}"
        )
    return value

def get_count(record, word, half_year):
    """Read the exact occurrence-count field emitted by the producer."""
    value = numeric_value(record, "n_occurrences")

    if not math.isfinite(value):
        raise ValueError(
            f"{half_year}/{word}: n_occurrences is missing or null."
        )

    if value < 0 or not value.is_integer():
        raise ValueError(
            f"{half_year}/{word}: invalid n_occurrences={value}."
        )

    return int(value)

def get_top_k(record, word, half_year):
    """Validate the individual drug-term matches stored in each cell."""
    matches = record.get("top_k_drug_terms", [])

    if matches is None:
        return []

    if not isinstance(matches, list):
        raise ValueError(
            f"{half_year}/{word}: top_k_drug_terms must be a list."
        )

    for match in matches:
        if not isinstance(match, dict):
            raise ValueError(
                f"{half_year}/{word}: invalid top-k entry."
            )
        if "term" not in match or "cosine_similarity" not in match:
            raise ValueError(
                f"{half_year}/{word}: top-k entries must contain "
                "'term' and 'cosine_similarity'."
            )

    return matches


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    (
        freq,
        half_years,
        words,
        normalized_words,
        freq_path,
    ) = load_frequency_inputs()

    TEST_HALF_YEARS = ["2015H1", "2015H2"]

    selected_indices = []
    for period in TEST_HALF_YEARS:
        if period not in half_years:
            raise ValueError(
                f"{period} is missing from freq_half_years.csv"
            )
        selected_indices.append(half_years.index(period))

    freq = freq[selected_indices, :]
    half_years = [half_years[i] for i in selected_indices]

    shape = (len(half_years), len(words))
    word_to_col = {
        word: i for i, word in enumerate(normalized_words)
    }

    metric_names = (
        "cosine_mean_embedding",
        "mean_occurrence_cosine",
        "std_occurrence_cosine",
        "mean_embedding_norm",
        "coverage",
    )

    matrices = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in metric_names
    }
    occurrence_matrix = np.zeros(shape, dtype=np.int64)
    frequency_matrix = np.asarray(freq, dtype=np.float64).copy()

    summary_path = os.path.join(
        OUTPUT_DIR, "similarity_summary.csv"
    )

    summary_columns = [
        "half_year",
        "word",
        "frequency",
        "n_occurrences",
        "cosine_mean_embedding",
        "mean_occurrence_cosine",
        "std_occurrence_cosine",
        "mean_embedding_norm",
        "coverage",
        "top_k_drug_terms",
    ]

    input_files = []
    shard_metadata = []
    missing_positive_frequency_cells = []
    unexpected_cells = []
    extra_words = set()
    finite_cosine_cells = 0

    # Stream the summary CSV to avoid keeping every row in memory.
    with open(
        summary_path, "w", encoding="utf-8", newline=""
    ) as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_columns,
        )
        writer.writeheader()

        for row_idx, half_year in enumerate(half_years):
            shard_path = os.path.join(
                SHARD_DIR, f"cosine_{half_year}.json"
            )

            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Missing shard for {half_year}: {shard_path}"
                )

            payload, cells = load_shard(shard_path, half_year)
            input_files.append(os.path.basename(shard_path))

            # Confirm the shard identifies the expected frequency row.
            freq_meta = payload.get("frequency_matrix", {})
            if isinstance(freq_meta, dict):
                saved_half_year = freq_meta.get("half_year")
                if (
                    saved_half_year is not None
                    and saved_half_year != half_year
                ):
                    raise ValueError(
                        f"{shard_path}: frequency_matrix.half_year "
                        f"is {saved_half_year!r}, expected {half_year!r}."
                    )

            shard_metadata.append({
                "shard_name": payload.get("shard_name"),
                "model_name": payload.get("model_name"),
                "embedding_dimension": payload.get(
                    "embedding_dimension"
                ),
                "context_wordpieces_each_side": payload.get(
                    "context_wordpieces_each_side"
                ),
                "top_k_drug_centroids": payload.get(
                    "top_k_drug_centroids"
                ),
                "scope": payload.get("scope"),
                "statistics": payload.get("statistics"),
            })

            extra_words.update(
                set(cells) - set(normalized_words)
            )

            row_frequency = np.asarray(freq[row_idx])
            expected_active_words = {
                normalized_words[col]
                for col in np.flatnonzero(row_frequency > 0)
            }

            unexpected_cells.extend(
                (half_year, word)
                for word in cells
                if word not in expected_active_words
            )

            row_finite_cosines = 0

            for col_idx, word in enumerate(words):
                normalized_word = normalized_words[col_idx]
                frequency = float(freq[row_idx, col_idx])
                record = cells.get(normalized_word)

                if record is None:
                    if frequency > 0:
                        missing_positive_frequency_cells.append(
                            (half_year, normalized_word)
                        )

                    writer.writerow({
                        "half_year": half_year,
                        "word": word,
                        "frequency": frequency,
                        "n_occurrences": 0,
                        "cosine_mean_embedding": "",
                        "mean_occurrence_cosine": "",
                        "std_occurrence_cosine": "",
                        "mean_embedding_norm": "",
                        "coverage": "",
                        "top_k_drug_terms": "[]",
                    })
                    continue

                count = get_count(record, normalized_word, half_year)
                occurrence_matrix[row_idx, col_idx] = count

                if count == 0:
                    for metric in metric_names:
                        if record.get(metric) is not None:
                            raise ValueError(
                                f"{half_year}/{normalized_word}: "
                                f"{metric} should be null when "
                                "n_occurrences is zero."
                            )
                    top_k = get_top_k(
                        record, normalized_word, half_year
                    )
                    if top_k:
                        raise ValueError(
                            f"{half_year}/{normalized_word}: expected "
                            "an empty top-k list for zero occurrences."
                        )
                else:
                    for metric in metric_names:
                        value = numeric_value(record, metric)
                        if math.isfinite(value):
                            matrices[metric][row_idx, col_idx] = value

                    if math.isfinite(
                        matrices["cosine_mean_embedding"][
                            row_idx, col_idx
                        ]
                    ):
                        row_finite_cosines += 1
                        finite_cosine_cells += 1

                    top_k = get_top_k(
                        record, normalized_word, half_year
                    )
                cell_frequency = numeric_value(record, "frequency")
                if (
                    not math.isfinite(cell_frequency)
                    or cell_frequency != frequency
                ):
                    raise ValueError(
                        f"{half_year}/{normalized_word}: cell frequency "
                        f"{cell_frequency} != frequency matrix value "
                        f"{frequency}."
                    )

                writer.writerow({
                    "half_year": half_year,
                    "word": word,
                    "frequency": frequency,
                    "n_occurrences": count,
                    **{
                        metric: (
                            float(matrices[metric][row_idx, col_idx])
                            if math.isfinite(
                                matrices[metric][row_idx, col_idx]
                            )
                            else ""
                        )
                        for metric in metric_names
                    },
                    "top_k_drug_terms": json.dumps(
                        top_k, ensure_ascii=False
                    ),
                })

            logger.info(
                "Loaded %s: %d cells, %d finite mean-embedding cosines",
                half_year,
                len(cells),
                row_finite_cosines,
            )

    # Save one matrix per statistic.
    output_files = {}

    for metric, matrix in matrices.items():
        filename = f"similarity_{metric}.npy"
        np.save(os.path.join(OUTPUT_DIR, filename), matrix)
        output_files[metric] = filename

    count_filename = "similarity_n_occurrences.npy"
    np.save(
        os.path.join(OUTPUT_DIR, count_filename),
        occurrence_matrix,
    )
    output_files["n_occurrences"] = count_filename

    frequency_filename = "similarity_frequency.npy"
    np.save(
        os.path.join(OUTPUT_DIR, frequency_filename),
        frequency_matrix,
    )
    output_files["frequency"] = frequency_filename

    # Sanity checks.
    cosine = matrices["cosine_mean_embedding"]
    bad_cosines = np.isfinite(cosine) & (
        (cosine < -1.0001) | (cosine > 1.0001)
    )

    models = {
        item["model_name"]
        for item in shard_metadata
        if item["model_name"] is not None
    }
    dimensions = {
        item["embedding_dimension"]
        for item in shard_metadata
        if item["embedding_dimension"] is not None
    }
    scopes = {
        json.dumps(item["scope"], sort_keys=True)
        for item in shard_metadata
        if item["scope"] is not None
    }

    if len(models) > 1:
        raise ValueError(f"Shards disagree on model: {models}")
    if len(dimensions) > 1:
        raise ValueError(
            f"Shards disagree on embedding dimensions: {dimensions}"
        )
    if len(scopes) > 1:
        raise ValueError("Shards disagree on subreddit scope.")

    metadata = {
        "script": "merge_similarity_matrix.py",
        "model_name": next(iter(models), None),
        "embedding_dimension": next(iter(dimensions), None),
        "frequency_matrix": os.path.abspath(freq_path),
        "half_years": half_years,
        "num_words": len(words),
        "matrix_shape": list(shape),
        "input_files": input_files,
        "output_files": output_files,
        "summary_csv": os.path.basename(summary_path),
        "shard_metadata": shard_metadata,
        "sanity_checks": {
            "expected_shards": len(half_years),
            "loaded_shards": len(input_files),
            "missing_positive_frequency_cells": len(
                missing_positive_frequency_cells
            ),
            "extra_words_not_in_frequency_vocabulary": len(extra_words),
            "cells_for_nonpositive_frequency_words": len(
                unexpected_cells
            ),
            "cosine_values_outside_bounds": int(bad_cosines.sum()),
            "finite_cosine_cells": finite_cosine_cells,
            "all_matrix_shapes_match_frequency": all(
                matrix.shape == shape
                for matrix in matrices.values()
            ) and occurrence_matrix.shape == shape,
        },
        "missing_similarity_value": "NaN in NumPy matrices; blank in CSV",
    }

    metadata_path = os.path.join(
        OUTPUT_DIR, "similarity_meta.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)

    if missing_positive_frequency_cells:
        logger.warning(
            "%d positive-frequency word cells were missing. "
            "See the sanity-check count in similarity_meta.json.",
            len(missing_positive_frequency_cells),
        )

    if unexpected_cells:
        logger.warning(
            "%d cells correspond to words with nonpositive frequency.",
            len(unexpected_cells),
        )

    if bad_cosines.any():
        logger.warning(
            "%d mean-embedding cosine values fell outside [-1, 1].",
            int(bad_cosines.sum()),
        )

    logger.info("Merge complete.")
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Summary CSV: %s", summary_path)
    logger.info("Metadata: %s", metadata_path)
    logger.info("Matrix shape: %s", shape)


if __name__ == "__main__":
    main()