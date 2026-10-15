"""
Merge partial embedding outputs into ONE global drug centroid.

Input files are produced by drug_centroid_embeddings_fixed.py and have the
structure:
    {
        "data": {
            "<drug_term>": {
                "YYYY-MM": {
                    "sum_embedding": [ ... ],
                    "count": 123
                }
            }
        },
        "stats": { ... }
    }

For each drug term, this script first aggregates its monthly/task sums and
counts. It then combines all drug terms using their occurrence counts to
produce one frequency-weighted centroid across all drug-term occurrences.

Mathematically:
    term_sum[t]   = sum of monthly/task sum_embedding vectors for term t
    term_count[t] = sum of monthly/task counts for term t

    drug_sum      = sum_t term_sum[t]
    drug_count    = sum_t term_count[t]
    drug_centroid = drug_sum / drug_count

This is equivalent to averaging all embedded occurrences across all drug
terms, with each occurrence contributing equally.

By default, reads partial_results_*.json from the current directory and writes
global_drug_centroid.json.
"""

import glob
import json
import os

import numpy as np


RESULTS_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/drug_centroids"
OUTPUT_PATH = os.path.join(RESULTS_DIR, "global_drug_centroid.json")
INPUT_PATTERN = "partial_results_*.json"


def merge_partial_results(
    results_dir=RESULTS_DIR,
    output_path=OUTPUT_PATH,
):
    partial_files = sorted(
        glob.glob(os.path.join(results_dir, INPUT_PATTERN))
    )

    if not partial_files:
        raise FileNotFoundError(
            f"No files matching {INPUT_PATTERN!r} found in {results_dir!r}."
        )

    print(f"Found {len(partial_files)} partial result file(s).")

    # First aggregate per term across all months and partial files.
    # term_agg[word] = {"sum_embedding": vector sum, "count": occurrence count}
    term_agg = {}

    for file_path in partial_files:
        print(f"Reading: {os.path.basename(file_path)}")

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        word_data = payload.get("data")
        if not isinstance(word_data, dict):
            raise ValueError(
                f"{file_path} does not contain a dictionary at key 'data'."
            )

        for term, months in word_data.items():
            if not isinstance(months, dict):
                raise ValueError(
                    f"Expected monthly data for term {term!r} in {file_path}."
                )

            for month, entry in months.items():
                if "sum_embedding" not in entry or "count" not in entry:
                    raise ValueError(
                        f"Missing 'sum_embedding' or 'count' for "
                        f"{term!r}/{month!r} in {file_path}."
                    )

                sum_embedding = np.asarray(
                    entry["sum_embedding"], dtype=np.float64
                )
                count = int(entry["count"])

                if sum_embedding.ndim != 1:
                    raise ValueError(
                        f"Expected a 1-D embedding sum for {term!r}/{month!r} "
                        f"in {file_path}; got shape {sum_embedding.shape}."
                    )
                if count < 0:
                    raise ValueError(
                        f"Negative occurrence count for {term!r}/{month!r}."
                    )
                if count == 0:
                    continue

                if term not in term_agg:
                    term_agg[term] = {
                        "sum_embedding": np.zeros_like(
                            sum_embedding, dtype=np.float64
                        ),
                        "count": 0,
                    }
                elif term_agg[term]["sum_embedding"].shape != sum_embedding.shape:
                    raise ValueError(
                        f"Embedding dimension mismatch for term {term!r}: "
                        f"expected {term_agg[term]['sum_embedding'].shape}, "
                        f"got {sum_embedding.shape}."
                    )

                term_agg[term]["sum_embedding"] += sum_embedding
                term_agg[term]["count"] += count

    # Next combine all per-term sums and counts into one drug-level centroid.
    drug_sum = None
    drug_count = 0

    per_term_summary = {}
    for term, agg in sorted(term_agg.items()):
        if agg["count"] == 0:
            continue

        if drug_sum is None:
            drug_sum = np.zeros_like(agg["sum_embedding"], dtype=np.float64)
        elif drug_sum.shape != agg["sum_embedding"].shape:
            raise ValueError(
                f"Embedding dimension mismatch while combining term {term!r}: "
                f"expected {drug_sum.shape}, got {agg['sum_embedding'].shape}."
            )

        drug_sum += agg["sum_embedding"]
        drug_count += agg["count"]

        # Keep per-term statistics in the output for auditing; this does not
        # change the fact that the primary centroid is one global drug centroid.
        per_term_summary[term] = {
            "sum_embedding": agg["sum_embedding"].tolist(),
            "count": agg["count"],
            "centroid": (
                agg["sum_embedding"] / agg["count"]
            ).tolist(),
        }

    if drug_count == 0 or drug_sum is None:
        raise ValueError(
            "No embedded drug-term occurrences with positive counts were found."
        )

    drug_centroid = drug_sum / drug_count

    output_payload = {
        "drug_centroid": {
            "centroid": drug_centroid.tolist(),
            "sum_embedding": drug_sum.tolist(),
            "count": drug_count,
        },
        "per_term": per_term_summary,
        "metadata": {
            "source_files": [os.path.basename(p) for p in partial_files],
            "num_source_files": len(partial_files),
            "num_terms": len(per_term_summary),
            "total_occurrences": drug_count,
            "aggregation": "frequency-weighted across all drug-term occurrences",
        },
    }

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n--- Global Drug Centroid ---")
    print(f"Drug terms included: {len(per_term_summary):,}")
    print(f"Total embedded occurrences: {drug_count:,}")
    print(f"Centroid dimension: {drug_centroid.size}")
    print(f"Saved global drug centroid to: {output_path}")
    print(
        "\nWarning: make sure the input directory contains only the intended "
        "partial result files. Including duplicate task outputs will double-count "
        "their embeddings."
    )


if __name__ == "__main__":
    merge_partial_results()
