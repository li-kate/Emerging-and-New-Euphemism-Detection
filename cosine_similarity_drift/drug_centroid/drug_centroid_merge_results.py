import glob
import json
import os

import numpy as np


REDDIT_MONTHLY_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SYNONYMS_FILE = "/storage/scratch1/8/arodrigues46/Emerging-and-New-Euphemism-Detection/drug_lists/drug_synonyms.txt"
RESULTS_DIR = "./drug_centroid_test/checkpoints"
OUTPUT_DIR = "./drug_centroid_test/centroid_results"

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "global_drug_centroid.json")
INPUT_PATTERN = "partial_results_*.json"
HIDDEN_DIM = 768
TOP_N = 10


def load_target_words(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Synonyms file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        words = {
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }

    if not words:
        raise ValueError(f"No target terms found in {path}")

    return sorted(words)


def discover_expected_tasks():
    dump_files = (
        glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RC_*.zst"))
        + glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RS_*.zst"))
    )
    return set(range(len(dump_files))), len(dump_files)


def merge_partial_results(
    results_dir=RESULTS_DIR,
    output_path=OUTPUT_PATH,
):
    target_words = load_target_words(SYNONYMS_FILE)
    target_word_set = set(target_words)

    partial_files = sorted(
        glob.glob(os.path.join(results_dir, INPUT_PATTERN))
    )

    if not partial_files:
        raise FileNotFoundError(
            f"No partial result files found in {results_dir}"
        )

    expected_tasks, expected_task_count = discover_expected_tasks()
    task_files = {}

    for path in partial_files:
        filename = os.path.basename(path)

        try:
            task_id = int(
                filename.removeprefix("partial_results_").removesuffix(".json")
            )
        except ValueError:
            raise ValueError(f"Unexpected partial filename: {filename}")

        if task_id in task_files:
            raise ValueError(f"Duplicate partial result for task {task_id}")

        task_files[task_id] = path

    found_tasks = set(task_files)
    missing_tasks = sorted(expected_tasks - found_tasks)
    unexpected_tasks = sorted(found_tasks - expected_tasks)

    if missing_tasks:
        print(f"WARNING: Missing task IDs: {missing_tasks}")
    if unexpected_tasks:
        print(f"WARNING: Unexpected task IDs: {unexpected_tasks}")

    term_agg = {}
    monthly_counts = {}

    total_stats = {
        "valid_records_processed": 0,
        "records_with_matches": 0,
        "matches_found": 0,
        "embedded": 0,
        "skipped_no_tokens": 0,
    }

    for task_id in sorted(task_files):
        file_path = task_files[task_id]

        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        word_data = payload.get("data")
        if not isinstance(word_data, dict):
            raise ValueError(f"Invalid data structure in {file_path}")

        for key in total_stats:
            total_stats[key] += int(payload.get("stats", {}).get(key, 0))

        for term, months in word_data.items():
            if term not in target_word_set:
                raise ValueError(
                    f"Term {term!r} in {file_path} is not in the current "
                    "synonyms file. Check whether the partial results are stale."
                )

            if not isinstance(months, dict):
                raise ValueError(
                    f"Expected monthly data for {term!r} in {file_path}"
                )

            if term not in term_agg:
                term_agg[term] = {
                    "sum_embedding": np.zeros(HIDDEN_DIM, dtype=np.float64),
                    "count": 0,
                }

            for month, entry in months.items():
                if "sum_embedding" not in entry or "count" not in entry:
                    raise ValueError(
                        f"Missing sum_embedding or count for {term!r}/{month}"
                    )

                embedding_sum = np.asarray(
                    entry["sum_embedding"], dtype=np.float64
                )
                count = int(entry["count"])

                if embedding_sum.shape != (HIDDEN_DIM,):
                    raise ValueError(
                        f"Unexpected embedding shape for {term!r}/{month}: "
                        f"{embedding_sum.shape}"
                    )

                if count < 0:
                    raise ValueError(
                        f"Negative count for {term!r}/{month}"
                    )

                if not np.all(np.isfinite(embedding_sum)):
                    raise ValueError(
                        f"Non-finite embedding values for {term!r}/{month}"
                    )

                if count == 0:
                    if np.any(embedding_sum != 0):
                        raise ValueError(
                            f"Nonzero embedding sum with zero count for "
                            f"{term!r}/{month}"
                        )
                    continue

                term_agg[term]["sum_embedding"] += embedding_sum
                term_agg[term]["count"] += count

                monthly_counts[month] = (
                    monthly_counts.get(month, 0) + count
                )

    if total_stats["matches_found"] != (
        total_stats["embedded"] + total_stats["skipped_no_tokens"]
    ):
        print(
            "WARNING: Matched occurrences do not equal embedded plus "
            "skipped occurrences. Check the partial results."
        )

    term_counts = {}
    term_centroids = {}
    drug_sum = np.zeros(HIDDEN_DIM, dtype=np.float64)
    drug_count = 0

    for term in target_words:
        agg = term_agg.get(term)

        if agg is None or agg["count"] == 0:
            continue

        count = agg["count"]
        embedding_sum = agg["sum_embedding"]

        term_counts[term] = count
        term_centroids[term] = (embedding_sum / count).tolist()

        drug_sum += embedding_sum
        drug_count += count

    if drug_count == 0:
        raise ValueError("No embedded drug-term occurrences were found.")

    drug_centroid = drug_sum / drug_count

    missing_terms = sorted(target_word_set - set(term_centroids))
    represented_months = sorted(monthly_counts)

    term_count_values = np.asarray(
        list(term_counts.values()), dtype=np.int64
    )

    stats = {
        "partial_files_read": len(task_files),
        "expected_tasks": expected_task_count,
        "missing_task_ids": missing_tasks,
        "unexpected_task_ids": unexpected_tasks,
        "complete_task_coverage": (
            not missing_tasks and not unexpected_tasks
        ),
        "valid_records_processed": total_stats["valid_records_processed"],
        "records_with_matches": total_stats["records_with_matches"],
        "matched_occurrences": total_stats["matches_found"],
        "embedded_occurrences": drug_count,
        "skipped_no_tokens": total_stats["skipped_no_tokens"],
        "target_terms": len(target_words),
        "terms_with_embeddings": len(term_centroids),
        "terms_without_embeddings": len(missing_terms),
        "terms_without_embeddings_list": missing_terms,
        "first_month": represented_months[0] if represented_months else None,
        "last_month": represented_months[-1] if represented_months else None,
        "months_represented": len(represented_months),
        "occurrences_per_term": {
            "min": int(term_count_values.min()),
            "median": float(np.median(term_count_values)),
            "mean": float(np.mean(term_count_values)),
            "max": int(term_count_values.max()),
        },
        "top_terms": [
            {"term": term, "count": count}
            for term, count in sorted(
                term_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:TOP_N]
        ],
        "weighting": "equal_weight_per_embedded_occurrence",
        "model": "bert-base-uncased",
        "embedding_dimension": HIDDEN_DIM,
    }

    output_payload = {
        "drug_centroid": {
            "centroid": drug_centroid.tolist(),
            "sum_embedding": drug_sum.tolist(),
            "count": drug_count,
        },
        "term_centroids": term_centroids,
        "term_counts": term_counts,
        "monthly_counts": monthly_counts,
        "statistics": stats,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    tmp_output = output_path + ".tmp"
    with open(tmp_output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f)
    os.replace(tmp_output, output_path)

    print("\n--- Global Drug Centroid ---")
    print(f"Embedded occurrences: {drug_count:,}")
    print(f"Unique terms represented: {len(term_centroids):,}/{len(target_words):,}")
    print(f"Terms without embeddings: {len(missing_terms):,}")
    print(f"Months represented: {len(represented_months):,}")
    print(f"Month range: {stats['first_month']} to {stats['last_month']}")
    print(f"Complete task coverage: {stats['complete_task_coverage']}")

    print("\n--- Most Frequent Terms ---")
    for item in stats["top_terms"]:
        print(f"{item['term']}: {item['count']:,}")

    print(f"\nSaved centroid and term embeddings to: {output_path}")


if __name__ == "__main__":
    merge_partial_results()