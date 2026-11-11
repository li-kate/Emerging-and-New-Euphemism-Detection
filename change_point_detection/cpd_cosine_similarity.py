"""
PELT change point detection on the cosine similarity output.

Reads the folder written by the cosine similarity run (``metadata.json`` plus
one ``[half_years x words]`` matrix per metric and
``similarity_n_occurrences.npy``) and finds, for each word, upward shifts in
how close its contextual BERT embedding is to that half-year's drug centroid.
A word whose usage BECOMES more drug-like shows an upward change point; a word
that was always drug-like (e.g. "heroin") stays flat and is not flagged.

The detection itself (PELT, permutation test, Benjamini-Hochberg, plots, and
the evaluation table) is shared with ``change_point_detection.py``, so results
from the two scripts are directly comparable.

Differences from the frequency run:

  * Half-years come from ``metadata.json``, not the frequency matrix. Gaps in
    the half-year sequence (e.g. a failed task) are logged and recorded in
    ``cpd_meta.json``; PELT runs on the half-years that are present.
  * Cells built from fewer than ``--min-occurrences`` occurrences are set to
    NaN, since an average of a handful of embeddings is mostly noise.
  * A word is analyzed only if at least ``--min-valid-periods`` half-years
    survive that mask.
  * ``--metric`` selects the matrix: ``cosine_mean_embedding`` (default),
    ``std_occurrence_cosine``, or ``mean_embedding_norm``. Use
    ``--decreasing`` when a drop is the direction of interest.

Outputs (in ``--out-dir``) use the same names and columns as the frequency
run: ``cpd_results.csv``, ``cpd_changepoints.csv``, ``cpd_eval.csv``,
``cpd_meta.json``, and ``plots/``. ``cpd_results.csv`` also has
``total_occurrences``, ``first_valid_half_year``, and ``last_valid_half_year``.

Example:

    python cpd_cosine_similarity.py \\
        --cosine-dir ../cosine_similarity_output \\
        --words-csv ../other_outputs/freq_words.csv \\
        --out-dir results_cosine
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from change_point_detection import (
    analyze_word,
    benjamini_hochberg,
    evaluate,
    plot_words,
    robust_noise,
)

logger = logging.getLogger(__name__)

METRICS = ["cosine_mean_embedding", "std_occurrence_cosine", "mean_embedding_norm"]
COLUMN_CHUNK = 500_000


# ============================================================================
# LOADING
# ============================================================================


def load_metadata(cosine_dir):
    with open(os.path.join(cosine_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    shape = tuple(meta["matrix_shape"])
    if shape[0] != len(meta["half_years"]):
        raise ValueError(
            f"metadata.json lists {len(meta['half_years'])} half-years but "
            f"matrix_shape is {shape}."
        )
    return meta


def load_words(words_csv, expected_size):
    words = pd.read_csv(words_csv, keep_default_na=False)["word"].astype(str).tolist()
    if len(words) != expected_size:
        raise ValueError(
            f"{words_csv} has {len(words):,} words but the cosine matrices have "
            f"{expected_size:,} columns. Use the frequency matrix the cosine run "
            f"was built from."
        )
    return words


def load_matrix(cosine_dir, meta, name, file_name):
    matrix = np.load(os.path.join(cosine_dir, file_name), mmap_mode="r")
    if matrix.shape != tuple(meta["matrix_shape"]):
        raise ValueError(
            f"{file_name} has shape {matrix.shape}, expected {meta['matrix_shape']}."
        )
    logger.info("Opened %s %s (%s)", name, matrix.shape, file_name)
    return matrix


def missing_half_years(half_years):
    """Half-years absent between the first and last label (e.g. failed tasks)."""

    def parse(label):
        m = re.fullmatch(r"(\d{4})H([12])", label)
        if not m:
            raise ValueError(f"Unexpected half-year label {label!r}.")
        return int(m.group(1)) * 2 + int(m.group(2)) - 1

    present = {parse(h) for h in half_years}
    lo, hi = min(present), max(present)
    return [f"{i // 2}H{i % 2 + 1}" for i in range(lo, hi + 1) if i not in present]


def eligible_columns(n_occurrences, rows, min_occurrences, min_valid_periods):
    """Columns with >= ``min_valid_periods`` half-years of >= ``min_occurrences``."""
    cols = []
    for start in range(0, n_occurrences.shape[1], COLUMN_CHUNK):
        block = np.asarray(n_occurrences[:, start:start + COLUMN_CHUNK])[rows]
        ok = (block >= min_occurrences).sum(axis=0) >= min_valid_periods
        cols.append(np.flatnonzero(ok) + start)
    cols = np.concatenate(cols)
    logger.info(
        "Eligible words: %s (>= %d occurrences in >= %d half-years)",
        f"{len(cols):,}", min_occurrences, min_valid_periods,
    )
    return cols


# ============================================================================
# MAIN
# ============================================================================


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    parser = argparse.ArgumentParser(
        description="PELT change point detection on the cosine similarity output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cosine-dir",
                        default=os.path.join(repo, "cosine_similarity_output"))
    parser.add_argument("--words-csv",
                        default=os.path.join(repo, "other_outputs", "freq_words.csv"),
                        help="Word labels for the matrix columns (the frequency "
                        "matrix's freq_words.csv).")
    parser.add_argument("--metric", choices=METRICS, default="cosine_mean_embedding")
    parser.add_argument("--decreasing", action="store_true",
                        help="Treat a decrease as the direction of interest.")
    parser.add_argument("--out-dir", default=os.path.join(here, "results_cosine"))

    parser.add_argument("--min-occurrences", type=int, default=30,
                        help="Cells with fewer occurrences are set to NaN.")
    parser.add_argument("--min-valid-periods", type=int, default=6,
                        help="Minimum half-years left after masking.")
    parser.add_argument("--exclude-half-years", nargs="*", default=[],
                        help="Half-years to drop entirely (e.g. a partial one).")

    parser.add_argument("--min-size", type=int, default=3,
                        help="Minimum segment length in half-years.")
    parser.add_argument("--pen-mult", type=float, default=3.0,
                        help="PELT penalty = pen_mult * log(n).")
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="q-value cutoff for flagging a word as emerging.")
    parser.add_argument("--min-shift", type=float, default=0.0,
                        help="Minimum upward jump in the metric's units to flag a "
                        "word as emerging.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eval-words", default=os.path.join(here, "eval_words.json"))
    parser.add_argument("--n-plots", type=int, default=30,
                        help="Plot this many top-ranked words (plus eval words).")
    parser.add_argument("--max-words", type=int, default=None,
                        help="Only analyze the first N eligible words (test runs).")

    args = parser.parse_args()
    if args.min_valid_periods < 2 * args.min_size:
        parser.error("--min-valid-periods must be at least 2 * --min-size.")
    return args


def main():
    # ------------------------------------------------------------------
    # 1. Setup: logging, command-line options, output folder, and a seeded
    #    random generator (the permutation test shuffles data, so a fixed
    #    seed makes every run give the same p-values).
    # ------------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 2. Read the labels. metadata.json says which half-year each matrix
    #    row is; freq_words.csv says which word each column is (the cosine
    #    run uses the frequency matrix's column order).
    # ------------------------------------------------------------------
    meta = load_metadata(args.cosine_dir)
    half_years = list(meta["half_years"])
    words = load_words(args.words_csv, meta["vocabulary_size"])

    # Warn about half-years missing from the middle of the timeline (for
    # example 2024H2 if its cluster job failed). PELT still runs, but a gap
    # can hide a change or make one look sharper than it was.
    gaps = missing_half_years(half_years)
    if gaps:
        logger.warning(
            "Half-years missing from the cosine output: %s. PELT will run on the "
            "half-years that are present; rerun the missing ones if possible.",
            ", ".join(gaps),
        )

    # Drop any half-years the user asked to exclude. `rows` are the matrix
    # row numbers kept; `periods` are their labels, in the same order.
    unknown = [h for h in args.exclude_half_years if h not in half_years]
    if unknown:
        raise ValueError(f"--exclude-half-years not in the output: {unknown}")
    rows = [i for i, h in enumerate(half_years) if h not in args.exclude_half_years]
    periods = [half_years[i] for i in rows]

    # ------------------------------------------------------------------
    # 3. Open the two matrices this run needs (memory-mapped, so nothing is
    #    loaded yet): the chosen metric (cosine similarity by default) and
    #    the number of occurrences behind each cell.
    # ------------------------------------------------------------------
    metric = load_matrix(args.cosine_dir, meta, args.metric,
                         meta["matrix_files"][args.metric])
    n_occurrences = load_matrix(args.cosine_dir, meta, "n_occurrences",
                                meta["occurrence_count_file"])

    # ------------------------------------------------------------------
    # 4. Pick the words worth analyzing: those with at least
    #    --min-occurrences in at least --min-valid-periods half-years. This
    #    cuts 5.4 million columns down to tens of thousands.
    # ------------------------------------------------------------------
    cols = eligible_columns(n_occurrences, rows, args.min_occurrences,
                            args.min_valid_periods)
    if args.max_words is not None:
        cols = cols[: args.max_words]

    # Load only those columns. `signal` is [half-years x eligible words].
    # Cells averaged from too few occurrences are too noisy to trust, so
    # they become NaN and are skipped below.
    counts = np.asarray(n_occurrences[:, cols][rows], dtype=np.int64)
    signal = np.asarray(metric[:, cols][rows], dtype=np.float64)
    signal[counts < args.min_occurrences] = np.nan

    # A lower limit on each word's noise estimate. A very smooth series
    # would otherwise get a near-zero noise estimate, and then PELT would
    # treat tiny wiggles as change points. The limit is a quarter of the
    # typical noise level, estimated from the first 2,000 words.
    sigmas = [robust_noise(s[np.isfinite(s)]) for s in signal.T[:2000]]
    noise_floor = max(0.25 * np.nanmedian(sigmas), 1e-6)

    # ------------------------------------------------------------------
    # 5. Run change point detection on every eligible word.
    #    results:    one summary dict per word (becomes cpd_results.csv)
    #    cp_rows:    one row per detected change point (cpd_changepoints.csv)
    #    *_of dicts: what each word's plot needs (series, labels, changes)
    # ------------------------------------------------------------------
    results, cp_rows = [], []
    signal_of, cps_of, periods_of = {}, {}, {}
    for i, col in enumerate(cols):
        word = words[col]

        # Keep only this word's usable half-years (non-NaN), in time order.
        column = signal[:, i]
        valid = np.isfinite(column)
        series = column[valid]
        valid_periods = [p for p, v in zip(periods, valid) if v]

        # PELT + permutation test for this one word (shared with the
        # frequency script). Returns None if too few half-years remain.
        out = analyze_word(series, valid_periods, args, noise_floor, rng)
        if out is None:
            continue
        result, cp = out

        # Add the cosine-specific columns to the word's summary row.
        result.update(
            word=word,
            total_occurrences=int(counts[:, i].sum()),
            first_valid_half_year=valid_periods[0],
            last_valid_half_year=valid_periods[-1],
        )
        results.append(result)
        cp_rows.extend({"word": word, **r} for r in cp)

        # Save what the plot needs. Change points are stored as half-year
        # labels ("2020H1;2022H2"); the plot needs their positions in the
        # series, so convert labels back to indices.
        signal_of[word] = series
        periods_of[word] = valid_periods
        cps_of[word] = [valid_periods.index(h) for h in
                        (result["changepoints"].split(";") if result["changepoints"] else [])]

        if (i + 1) % 10000 == 0:
            logger.info("  analyzed %s / %s words", f"{i + 1:,}", f"{len(cols):,}")

    if not results:
        raise SystemExit("No words had enough valid half-years to analyze.")

    # ------------------------------------------------------------------
    # 6. Decide which words are "emerging" and rank them.
    # ------------------------------------------------------------------
    results = pd.DataFrame(results)

    # Tens of thousands of words are tested, so some would look significant
    # by chance. Benjamini-Hochberg turns each p-value into a q-value that
    # controls the share of false discoveries across the whole list.
    results["q_value"] = benjamini_hochberg(results["p_value"])

    # Emerging = significant (q below --alpha) AND the largest upward jump
    # is bigger than --min-shift. With the default of 0, any significant
    # upward shift counts.
    results["emerging"] = (
        (results["q_value"] < args.alpha)
        & (results["shift"] > max(args.min_shift, 0.0))
    )

    # Ranking: emerging words first, then by jump size (biggest first),
    # then by q-value. Rank 1 is the strongest emerging candidate.
    results = results.sort_values(
        ["emerging", "shift", "q_value"], ascending=[False, False, True]
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))

    # Move the most useful columns to the front of the CSV.
    front = ["rank", "word", "emerging", "onset_half_year", "shift", "q_value"]
    results = results[front + [c for c in results.columns if c not in front]]

    # ------------------------------------------------------------------
    # 7. Write the outputs.
    # ------------------------------------------------------------------
    # The ranked word list and every individual change point.
    results.to_csv(os.path.join(args.out_dir, "cpd_results.csv"), index=False)
    pd.DataFrame(cp_rows).to_csv(
        os.path.join(args.out_dir, "cpd_changepoints.csv"), index=False
    )
    logger.info(
        "Analyzed %s words; %s flagged as emerging (q < %.2f, shift > %.3f).",
        f"{len(results):,}", f"{int(results['emerging'].sum()):,}",
        args.alpha, args.min_shift,
    )

    # Evaluation: look up where the known word groups (candidate
    # euphemisms, established euphemisms, comparison words, drug names)
    # landed in the ranking, and print a one-line summary per group.
    eval_df = evaluate(results, args.eval_words)
    eval_df.to_csv(os.path.join(args.out_dir, "cpd_eval.csv"), index=False)
    for group, g in eval_df.groupby("group", sort=False):
        elig = g[g["eligible"]]
        logger.info(
            "  eval %-22s eligible %2d/%2d, emerging %2d, median percentile %s",
            group, len(elig), len(g), int(elig["emerging"].sum()) if len(elig) else 0,
            f"{elig['percentile'].median():.1f}" if len(elig) else "n/a",
        )

    # Plots: the top --n-plots words plus every evaluation word that was
    # analyzed (without plotting any word twice).
    ylabel = args.metric.replace("_", " ")
    to_plot = list(results["word"].head(args.n_plots))
    to_plot += [w for w in eval_df.loc[eval_df["eligible"], "word"] if w not in to_plot]
    plot_words(to_plot, signal_of, cps_of, periods_of, ylabel,
               os.path.join(args.out_dir, "plots"))
    logger.info("Saved %d plots to %s", len(to_plot), os.path.join(args.out_dir, "plots"))

    # Record every setting used, so the run can be reproduced and described
    # exactly in the paper's methods section.
    run_meta = {
        "script": "cpd_cosine_similarity.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cosine_dir": os.path.abspath(args.cosine_dir),
        "words_csv": os.path.abspath(args.words_csv),
        "metric": args.metric,
        "decreasing": args.decreasing,
        "similarity_target": meta.get("similarity_target"),
        "model_name": meta.get("model_name"),
        "half_years_used": periods,
        "half_years_excluded": list(args.exclude_half_years),
        "half_years_missing_from_output": gaps,
        "eligibility": {
            "min_occurrences_per_cell": args.min_occurrences,
            "min_valid_periods": args.min_valid_periods,
        },
        "pelt": {
            "model": "l2",
            "min_size": args.min_size,
            "penalty": f"{args.pen_mult} * log(n)",
            "standardization": "robust noise (MAD of first differences)",
            "noise_floor": noise_floor,
        },
        "significance": {
            "test": "one-sided permutation test of max standardized upward mean shift",
            "n_perm": args.n_perm,
            "multiple_testing": "Benjamini-Hochberg",
            "alpha": args.alpha,
            "min_shift": args.min_shift,
            "seed": args.seed,
        },
        "counts": {
            "eligible_words": int(len(cols)),
            "analyzed_words": int(len(results)),
            "emerging_words": int(results["emerging"].sum()),
        },
    }
    with open(os.path.join(args.out_dir, "cpd_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
