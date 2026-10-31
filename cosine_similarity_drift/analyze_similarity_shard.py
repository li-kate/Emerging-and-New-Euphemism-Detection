
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SHARD_PATH = Path(
    "cosine_similarity_results/shards/monthly/month_2015-01.json"
)

OUTPUT_DIR = Path(
    "cosine_similarity_results/similarity_analysis/month_2015-01"
)

FREQ_DIR = Path(
    "/storage/cedar/cedar0/cedarp-smussmann3-0/"
    "kmehta301/freq_drug_subreddits_unfiltered"
)

FREQ_WORDS_PATH = FREQ_DIR / "freq_words.csv"
FREQ_HALF_YEARS_PATH = FREQ_DIR / "freq_half_years.csv"
FREQ_MATRIX_PATH = FREQ_DIR / "freq.npy"

TOP_N = 100
MIN_OCCURRENCES = 20
OCCURRENCE_THRESHOLDS = [1, 5, 10, 20, 50, 100]


def load_frequency_words(path):
    """Load vocabulary in the same order as the frequency matrix."""
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Vocabulary file is empty: {path}")

    candidates = ["word", "words", "token", "term", "vocabulary"]
    column_lookup = {str(c).lower(): c for c in df.columns}

    word_column = next(
        (column_lookup[c] for c in candidates if c in column_lookup),
        None,
    )

    if word_column is None:
        if len(df.columns) == 1:
            word_column = df.columns[0]
        else:
            raise ValueError(
                f"Could not identify vocabulary column in {path}. "
                f"Columns: {list(df.columns)}"
            )

    return df[word_column].astype(str).to_numpy()


def load_frequency_row(half_year):
    """Return vocabulary and frequency values for the requested half-year."""
    required = (
        FREQ_WORDS_PATH,
        FREQ_HALF_YEARS_PATH,
        FREQ_MATRIX_PATH,
    )

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required frequency files are missing:\n" + "\n".join(missing)
        )

    periods_df = pd.read_csv(FREQ_HALF_YEARS_PATH)
    period_column = next(
        (
            c for c in periods_df.columns
            if str(c).lower() in {"half_year", "period", "year_half"}
        ),
        periods_df.columns[0],
    )
    periods = periods_df[period_column].astype(str).tolist()

    if half_year not in periods:
        raise ValueError(
            f"{half_year!r} not found in {FREQ_HALF_YEARS_PATH}. "
            f"Available examples: {periods[:5]}"
        )

    frequencies_matrix = np.load(FREQ_MATRIX_PATH, mmap_mode="r")
    words = load_frequency_words(FREQ_WORDS_PATH)

    if len(words) != frequencies_matrix.shape[1]:
        raise ValueError(
            f"Vocabulary has {len(words):,} words but freq.npy has "
            f"{frequencies_matrix.shape[1]:,} columns."
        )

    return words, np.asarray(
        frequencies_matrix[periods.index(half_year)]
    )


def main():
    json_path = SHARD_PATH
    npz_path = json_path.with_suffix(".npz")

    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    metadata = json.loads(json_path.read_text())
    half_year = metadata["half_year"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the shard's sufficient statistics.
    with np.load(npz_path, allow_pickle=False) as arrays:
        indices = arrays["indices"].astype(np.int64)
        embedding_sums = arrays["embedding_sums"]
        counts = arrays["n_occurrences"].astype(np.int64)
        cosine_sum = arrays["cosine_sum"]
        cosine_sq_sum = arrays["cosine_sq_sum"]

        mean_cosine = np.divide(
            cosine_sum,
            counts,
            out=np.full(counts.shape, np.nan, dtype=np.float64),
            where=counts > 0,
        )

        # Population variance: E[x^2] - E[x]^2.
        variance = np.divide(
            cosine_sq_sum,
            counts,
            out=np.full(counts.shape, np.nan, dtype=np.float64),
            where=counts > 0,
        ) - mean_cosine**2

        std_cosine = np.sqrt(np.maximum(variance, 0.0))

        # Norm of the average contextual embedding.
        mean_embedding_norm = np.divide(
            np.linalg.norm(embedding_sums, axis=1),
            counts,
            out=np.full(counts.shape, np.nan, dtype=np.float64),
            where=counts > 0,
        )

        # Copy arrays before closing the NPZ file.
        indices = indices.copy()
        counts = counts.copy()
        mean_cosine = mean_cosine.copy()
        std_cosine = std_cosine.copy()
        mean_embedding_norm = mean_embedding_norm.copy()

    # Map vocabulary indices to words and frequency values.
    words, frequencies = load_frequency_row(half_year)

    if indices.size and (
        indices.min() < 0 or indices.max() >= len(words)
    ):
        raise ValueError("Shard vocabulary indices are out of range.")

    if not (
        len(indices)
        == len(counts)
        == len(mean_cosine)
        == len(std_cosine)
        == len(mean_embedding_norm)
    ):
        raise ValueError("Shard arrays have inconsistent lengths.")

    results = pd.DataFrame({
        "word": words[indices],
        "vocabulary_index": indices,
        "n_occurrences": counts,
        "mean_occurrence_cosine": mean_cosine,
        "std_occurrence_cosine": std_cosine,
        "mean_embedding_norm": mean_embedding_norm,
        "frequency": frequencies[indices],
    })

    results["coverage"] = np.divide(
        results["n_occurrences"].to_numpy(dtype=np.float64),
        results["frequency"].to_numpy(dtype=np.float64),
        out=np.full(len(results), np.nan, dtype=np.float64),
        where=results["frequency"].to_numpy() > 0,
    )

    # Preserve original counts before applying any thresholds.
    total_words_before_filter = len(results)
    total_occurrences_before_filter = int(results["n_occurrences"].sum())

    # Generate each threshold ranking from the UNFILTERED results.
    threshold_summary = []

    for threshold in OCCURRENCE_THRESHOLDS:
        filtered = results.loc[
            results["n_occurrences"] >= threshold
        ].copy()

        filtered = filtered.sort_values(
            "mean_occurrence_cosine",
            ascending=False,
            na_position="last",
        )

        output_path = (
            OUTPUT_DIR / f"top_{TOP_N}_min_occ_{threshold}.csv"
        )
        filtered.head(TOP_N).to_csv(output_path, index=False)

        eligible = len(filtered)
        threshold_summary.append({
            "minimum_occurrences": threshold,
            "eligible_words": eligible,
            "top_words_csv": str(output_path),
        })

        print(
            f"Minimum occurrences >= {threshold:3}: "
            f"{eligible:,} words eligible"
        )

    pd.DataFrame(threshold_summary).to_csv(
        OUTPUT_DIR / "threshold_summary.csv",
        index=False,
    )

    # Apply the configured minimum threshold for the main outputs.
    results = results.loc[
        results["n_occurrences"] >= MIN_OCCURRENCES
    ].copy()

    results = results.sort_values(
        "mean_occurrence_cosine",
        ascending=False,
        na_position="last",
    )

    all_path = OUTPUT_DIR / "all_words_ranked.csv"
    results.to_csv(all_path, index=False)

    top_path = OUTPUT_DIR / f"top_{TOP_N}_words.csv"
    results.head(TOP_N).to_csv(top_path, index=False)

    frequent_path = OUTPUT_DIR / "top_words_by_occurrences.csv"
    results.sort_values(
        "n_occurrences", ascending=False
    ).head(TOP_N).to_csv(frequent_path, index=False)

    summary = {
        "month": metadata["month"],
        "half_year": half_year,
        "model_name": metadata["model_name"],
        "original_words_in_shard": total_words_before_filter,
        "original_total_occurrences": total_occurrences_before_filter,
        "minimum_occurrences_for_main_outputs": MIN_OCCURRENCES,
        "words_meeting_main_threshold": len(results),
        "occurrences_among_words_meeting_threshold": int(
            results["n_occurrences"].sum()
        ),
        "ranking": "mean_occurrence_cosine, descending",
        "interpretation": (
            "Mean cosine similarity of individual contextual occurrence "
            "embeddings to the global drug centroid."
        ),
        "source_statistics": metadata["statistics"],
        "threshold_results": threshold_summary,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    display_columns = [
        "word",
        "mean_occurrence_cosine",
        "std_occurrence_cosine",
        "n_occurrences",
        "frequency",
        "coverage",
    ]

    print(f"\nMonth: {metadata['month']}")
    print(f"Words in original shard: {total_words_before_filter:,}")
    print(f"Words meeting minimum of {MIN_OCCURRENCES}: {len(results):,}")
    print(
        "Occurrences among eligible words: "
        f"{results['n_occurrences'].sum():,}"
    )
    print(f"\nTop {min(TOP_N, len(results))} by mean occurrence cosine:")
    print(results.head(TOP_N)[display_columns].to_string(index=False))

    print("\nSaved:")
    print(f"  Main ranking:    {all_path}")
    print(f"  Top cosine:      {top_path}")
    print(f"  Top occurrences: {frequent_path}")
    print(f"  Thresholds:     {OUTPUT_DIR / 'threshold_summary.csv'}")
    print(f"  Summary:         {summary_path}")


if __name__ == "__main__":
    main()
