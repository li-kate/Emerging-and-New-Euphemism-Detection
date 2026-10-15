import csv
import json
import os

import numpy as np


FREQ_DIR = "../frequency_matrix"
OUTPUT_DIR = "./cosine_similarity_results"


def read_csv_column(path, preferred_column):
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


def main():
    freq_path = os.path.join(FREQ_DIR, "freq.npy")
    freq = np.load(freq_path, mmap_mode="r")

    half_years = read_csv_column(
        os.path.join(FREQ_DIR, "freq_half_years.csv"),
        "half_year",
    )
    words = read_csv_column(
        os.path.join(FREQ_DIR, "freq_words.csv"),
        "word",
    )

    expected_shape = (len(half_years), len(words))
    if freq.shape != expected_shape:
        raise ValueError(
            f"Frequency matrix shape {freq.shape} != {expected_shape}"
        )

    # NaN means no valid contextual embedding was produced.
    cosine_matrix = np.full(expected_shape, np.nan, dtype=np.float32)
    occurrence_matrix = np.zeros(expected_shape, dtype=np.int64)

    for row_idx, half_year in enumerate(half_years):
        path = os.path.join(
            OUTPUT_DIR, f"cosine_{half_year}.npz"
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing output for {half_year}: {path}"
            )

        with np.load(path, allow_pickle=False) as result:
            saved_half_year = str(result["half_year"].item())
            scores = result["scores"]
            counts = result["occurrence_counts"]

        if saved_half_year != half_year:
            raise ValueError(
                f"{path} contains {saved_half_year}, "
                f"but row {row_idx} expects {half_year}."
            )

        if scores.shape != (len(words),):
            raise ValueError(
                f"Wrong score shape in {path}: {scores.shape}"
            )
        if counts.shape != (len(words),):
            raise ValueError(
                f"Wrong occurrence-count shape in {path}: {counts.shape}"
            )

        cosine_matrix[row_idx] = scores
        occurrence_matrix[row_idx] = counts

        print(
            f"Loaded {half_year}: "
            f"{np.isfinite(scores).sum():,} scored words"
        )

    cosine_path = os.path.join(
        OUTPUT_DIR, "cosine_similarity.npy"
    )
    occurrence_path = os.path.join(
        OUTPUT_DIR, "cosine_occurrence_counts.npy"
    )

    np.save(cosine_path, cosine_matrix)
    np.save(occurrence_path, occurrence_matrix)

    metadata = {
        "model": "bert-base-uncased",
        "similarity_target": "global drug centroid",
        "embedding_method": (
            "Mean target WordPiece hidden state per occurrence, "
            "then mean across occurrences for each word and half-year"
        ),
        "context_wordpieces_each_side": 64,
        "frequency_matrix": freq_path,
        "half_years": half_years,
        "num_words": len(words),
        "matrix_shape": list(cosine_matrix.shape),
        "cosine_file": cosine_path,
        "occurrence_counts_file": occurrence_path,
        "missing_score_value": "NaN",
    }

    metadata_path = os.path.join(
        OUTPUT_DIR, "cosine_meta.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nMerge complete.")
    print(f"Cosine matrix: {cosine_path}")
    print(f"Occurrence counts: {occurrence_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Matrix shape: {cosine_matrix.shape}")


if __name__ == "__main__":
    main()