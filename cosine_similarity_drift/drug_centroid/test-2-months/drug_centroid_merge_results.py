import glob
import json
import os

import numpy as np


RESULTS_DIR = "/storage/ice1/0/1/kli605/Emerging-and-New-Euphemism-Detection/cosine_similarity_drift/drug_centroid/checkpoints"
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

    if drug_count == 0 or drug_sum is None:
        raise ValueError(
            "No embedded drug-term occurrences with positive counts were found."
        )

    drug_centroid = drug_sum / drug_count

    # Save ONLY the global drug centroid structure requested
    output_payload = {
        "drug_centroid": {
            "centroid": drug_centroid.tolist(),
            "sum_embedding": drug_sum.tolist(),
            "count": drug_count,
        }
    }

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print("\n--- Global Drug Centroid ---")
    print(f"Total embedded occurrences: {drug_count:,}")
    print(f"Centroid dimension: {drug_centroid.size}")
    print(f"Saved global drug centroid to: {output_path}")


if __name__ == "__main__":
    merge_partial_results()