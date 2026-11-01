import glob
import json
import os
import re

import numpy as np


REDDIT_MONTHLY_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SYNONYMS_FILE = "/storage/scratch1/8/arodrigues46/Emerging-and-New-Euphemism-Detection/drug_lists/drug_synonyms.txt"
RESULTS_DIR = "./drug_centroid_hy/checkpoints"
OUTPUT_DIR = "./drug_centroid_hy/centroid_results"

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR, "drug_centroids_by_half_year.json"
)

INPUT_PATTERN = "partial_results_*.json"
HIDDEN_DIM = 768
TOP_N = 10

SKIP_HALF_YEARS = {"2015H1", "2024H2"}

try:
    import orjson

    def load_json(path):
        with open(path, "rb") as f:
            return orjson.loads(f.read())

    def dump_json(payload, path):
        with open(path, "wb") as f:
            f.write(orjson.dumps(payload))
except ImportError:
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def dump_json(payload, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)


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
        sorted(glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RC_*.zst")))
        + sorted(glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RS_*.zst")))
    )

    expected_tasks = set()
    skipped_tasks = set()

    for task_id, path in enumerate(dump_files):
        filename = os.path.basename(path)
        match = re.search(r"(\d{4}-\d{2})", filename)

        if not match:
            raise ValueError(
                f"Could not extract YYYY-MM from dump filename: {filename}"
            )

        half_year = month_to_half_year(match.group(1))

        if half_year in SKIP_HALF_YEARS:
            skipped_tasks.add(task_id)
        else:
            expected_tasks.add(task_id)

    return expected_tasks, skipped_tasks, len(dump_files)


def month_to_half_year(month):
    """Convert YYYY-MM into YYYYH1 or YYYYH2."""
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"Invalid month format: {month}")

    year, month_number = month.split("-")
    half = "H1" if int(month_number) <= 6 else "H2"
    return f"{year}{half}"


def new_half_year_accumulator():
    return {
        "terms": {},
        "monthly_counts": {},
        "drug_sum": np.zeros(HIDDEN_DIM, dtype=np.float64),
        "drug_count": 0,
    }


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

    expected_tasks, skipped_tasks, expected_task_count = (
        discover_expected_tasks()
    )
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
    
    all_task_ids = set(range(expected_task_count))
    
    missing_tasks = sorted(expected_tasks - found_tasks)
    unexpected_tasks = sorted(found_tasks - all_task_ids)
    existing_skipped_tasks = sorted(skipped_tasks & found_tasks)

    if missing_tasks:
        print(f"WARNING: Missing task IDs: {missing_tasks}")
    if unexpected_tasks:
        print(f"WARNING: Unexpected task IDs: {unexpected_tasks}")

    half_year_agg = {}
    total_stats = {
        "valid_records_processed": 0,
        "records_with_matches": 0,
        "matches_found": 0,
        "embedded": 0,
        "skipped_no_tokens": 0,
    }

    processed_task_count = 0

    for task_id in sorted(task_files):
    
        # Do not merge results from deliberately skipped half-years.
        if task_id in skipped_tasks:
            print(f"Skipping task {task_id}: belongs to an excluded half-year")
            continue
    
        file_path = task_files[task_id]
        print(f"Reading task {task_id}: {os.path.basename(file_path)}")
    
        payload = load_json(file_path)
        processed_task_count += 1

        word_data = payload.get("data")
        if not isinstance(word_data, dict):
            raise ValueError(f"Invalid data structure in {file_path}")

        payload_stats = payload.get("stats", {})
        for key in total_stats:
            total_stats[key] += int(payload_stats.get(key, 0))

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

            for month, entry in months.items():
                half_year = month_to_half_year(month)

                if half_year not in half_year_agg:
                    half_year_agg[half_year] = new_half_year_accumulator()

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
                    raise ValueError(f"Negative count for {term!r}/{month}")
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

                accumulator = half_year_agg[half_year]
                if term not in accumulator["terms"]:
                    accumulator["terms"][term] = {
                        "sum_embedding": np.zeros(
                            HIDDEN_DIM, dtype=np.float64
                        ),
                        "count": 0,
                    }

                accumulator["terms"][term]["sum_embedding"] += embedding_sum
                accumulator["terms"][term]["count"] += count
                accumulator["drug_sum"] += embedding_sum
                accumulator["drug_count"] += count
                accumulator["monthly_counts"][month] = (
                    accumulator["monthly_counts"].get(month, 0) + count
                )

    if total_stats["matches_found"] != (
        total_stats["embedded"] + total_stats["skipped_no_tokens"]
    ):
        print(
            "WARNING: Matched occurrences do not equal embedded plus "
            "skipped occurrences. Check the partial results."
        )

    output_half_years = {}

    for half_year in sorted(half_year_agg):
        accumulator = half_year_agg[half_year]
        drug_sum = accumulator["drug_sum"]
        drug_count = accumulator["drug_count"]

        if drug_count == 0:
            print(f"Skipping {half_year}: no embedded occurrences")
            continue

        drug_centroid = drug_sum / drug_count
        term_centroids = {}
        term_counts = {}

        for term, agg in accumulator["terms"].items():
            count = agg["count"]
            if count == 0:
                continue
            term_counts[term] = count
            term_centroids[term] = (
                agg["sum_embedding"] / count
            ).tolist()

        missing_terms = sorted(target_word_set - set(term_centroids))
        represented_months = sorted(accumulator["monthly_counts"])
        term_count_values = np.asarray(
            list(term_counts.values()), dtype=np.int64
        )

        stats = {
            "embedded_occurrences": drug_count,
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

        output_half_years[half_year] = {
            "drug_centroid": {
                "centroid": drug_centroid.tolist(),
                "sum_embedding": drug_sum.tolist(),
                "count": drug_count,
            },
            "term_centroids": term_centroids,
            "term_counts": term_counts,
            "monthly_counts": accumulator["monthly_counts"],
            "statistics": stats,
        }

    if not output_half_years:
        raise ValueError(
            "No half-years contained embedded drug-term occurrences."
        )

    global_stats = {
        "partial_files_read": processed_task_count,
        "skipped_half_years": sorted(SKIP_HALF_YEARS),
        "skipped_task_ids": sorted(skipped_tasks),
        "existing_skipped_partial_files": existing_skipped_tasks,
        "expected_tasks": expected_task_count,
        "missing_task_ids": missing_tasks,
        "unexpected_task_ids": unexpected_tasks,
        "complete_task_coverage": (
            not missing_tasks and not unexpected_tasks
        ),
        "valid_records_processed": total_stats["valid_records_processed"],
        "records_with_matches": total_stats["records_with_matches"],
        "matched_occurrences": total_stats["matches_found"],
        "embedded_occurrences": total_stats["embedded"],
        "skipped_no_tokens": total_stats["skipped_no_tokens"],
        "half_years_created": sorted(output_half_years),
        "number_of_half_years": len(output_half_years),
        "weighting": "equal_weight_per_embedded_occurrence",
        "model": "bert-base-uncased",
        "embedding_dimension": HIDDEN_DIM,
    }

    output_payload = {
        "half_years": output_half_years,
        "global_statistics": global_stats,
    }

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True,
    )
    tmp_output = output_path + ".tmp"
    dump_json(output_payload, tmp_output)
    os.replace(tmp_output, output_path)

    print("\n" + "=" * 60)
    print("HALF-YEAR DRUG CENTROIDS")
    print("=" * 60)

    for half_year, data in output_half_years.items():
        drug_data = data["drug_centroid"]
        stats = data["statistics"]

        print(f"\n--- {half_year} ---")
        print(f"Embedded occurrences: {drug_data['count']:,}")
        print(
            f"Unique terms represented: {stats['terms_with_embeddings']:,}/"
            f"{stats['target_terms']:,}"
        )
        print(f"Terms without embeddings: {stats['terms_without_embeddings']:,}")
        print(f"Months represented: {stats['months_represented']:,}")
        print(f"Month range: {stats['first_month']} to {stats['last_month']}")
        print("Most frequent terms:")
        for item in stats["top_terms"]:
            print(f"  {item['term']}: {item['count']:,}")

    print("\n--- Global Processing Statistics ---")
    print(f"Partial files read: {len(task_files):,}")
    print(
        "Complete task coverage: "
        f"{global_stats['complete_task_coverage']}"
    )
    print(f"Matched occurrences: {total_stats['matches_found']:,}")
    print(f"Embedded occurrences: {total_stats['embedded']:,}")
    print(f"Half-years created: {len(output_half_years):,}")
    print(f"\nSaved half-year centroids to: {output_path}")


if __name__ == "__main__":
    merge_partial_results()