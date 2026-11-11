"""PELT change point detection on hit-matrix mean log rank.

Reads the per-month shards written by ``hit_matrix.py`` (``*.part.json``),
sums each word's ``n_scored`` and ``log_rank_sum`` within a half-year, and
detects decreases in

    mean_log_rank = log_rank_sum / n_scored

``log`` is the natural log. ``exp(mean_log_rank)`` is the geometric-mean rank
of the best DEA term: 1 is the top guess, and a smaller value means BERT
more often filled the mask with a drug name. Emergence is a *decrease*.
The shared ``analyze_word`` routine is run with ``decreasing=True``, so
``shift`` in the output is the size of the largest drop (a positive number).
A shift of ln(2) ≈ 0.693 means the geometric-mean rank halved.

Ranks are merged by summing the logs and the counts, never by averaging
per-month means. A half-year with fewer than ``--min-scored`` contexts is
masked, because a rank estimated from a handful of contexts is mostly noise.

The month array in ``hit_matrix_shards/`` is incomplete. The script still
runs on whatever months are present, records which months each half-year is
missing, and leaves partial half-years in unless ``--exclude-half-years``
drops them. It does not rescore BERT.

Example (not run by default):

    python cpd_hit_logrank.py --max-words 2000
    python cpd_hit_logrank.py
"""

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
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
from cpd_cosine_similarity import missing_half_years

logger = logging.getLogger(__name__)

SHARD_SUFFIX = ".part.json"
SHARD_MONTH = re.compile(r"RC_(\d{4})-(\d{2})$")


# ============================================================================
# SHARDS -> HALF-YEAR MEAN LOG RANK
# ============================================================================


def shard_month(path):
    """``(year, month)`` from ``RC_YYYY-MM.part.json``, or None."""
    stem = os.path.basename(path)[: -len(SHARD_SUFFIX)]
    match = SHARD_MONTH.fullmatch(stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def half_year_label(year, month):
    return f"{year}H{1 if month <= 6 else 2}"


def month_coverage(shard_paths):
    """
    Which calendar months each half-year actually has a shard for.

    A half-year has six months. Missing months do not remove the half-year;
    they are recorded so a partial half-year is not treated as complete.
    """
    present = defaultdict(set)
    for path in shard_paths:
        found = shard_month(path)
        if found is None:
            logger.warning("Shard name is not RC_YYYY-MM, skipped for coverage: %s", path)
            continue
        year, month = found
        present[half_year_label(year, month)].add(f"{year}-{month:02d}")

    coverage = {}
    for label in sorted(present, key=_half_year_key):
        year = int(label[:4])
        months = range(1, 7) if label.endswith("H1") else range(7, 13)
        expected = [f"{year}-{m:02d}" for m in months]
        have = sorted(present[label])
        coverage[label] = {
            "months_present": have,
            "months_missing": [m for m in expected if m not in present[label]],
        }
    return coverage


def _half_year_key(label):
    match = re.fullmatch(r"(\d{4})H([12])", label)
    if not match:
        raise ValueError(f"Unexpected half-year label {label!r}.")
    return int(match.group(1)) * 2 + int(match.group(2))


def load_logrank(shard_paths):
    """
    Sum ``n_scored`` and ``log_rank_sum`` into one row per (half-year, word).

    Returns the table plus the scope and DEA metadata the shards agree on.
    """
    frames = []
    scopes, deas, ks_seen = [], [], []
    for i, path in enumerate(shard_paths, 1):
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)
        scopes.append(shard.get("scope"))
        deas.append(shard.get("dea"))
        ks_seen.append(shard.get("ks"))

        rows = []
        file_hy = None
        found = shard_month(path)
        if found is not None:
            file_hy = half_year_label(*found)
        for key, cell in shard["cells"].items():
            hy, word = key.split("\t", 1)
            if file_hy is not None and hy != file_hy:
                raise ValueError(
                    f"{os.path.basename(path)} is {file_hy} but contains a "
                    f"{hy!r} cell ({word!r})."
                )
            n = int(cell["n_scored"])
            if n <= 0:
                continue
            rows.append((hy, word, n, float(cell["log_rank_sum"])))
        frames.append(pd.DataFrame(rows, columns=["half_year", "word", "n_scored", "log_rank_sum"]))
        logger.info(
            "Read shard %d/%d: %s (%s words)",
            i, len(shard_paths), os.path.basename(path), f"{len(rows):,}",
        )

    _require_agreement("scope", scopes)
    _require_agreement("ks", ks_seen)
    # The DEA *set* must be the same run to run. Compare the fields that
    # decide which tokens counted as a hit; the file path may differ by machine.
    dea_keys = [
        (d.get("n_single_token"), d.get("n_dropped"), d.get("model_name"))
        if isinstance(d, dict) else None
        for d in deas
    ]
    _require_agreement("dea token set", dea_keys)

    table = (
        pd.concat(frames, ignore_index=True)
        .groupby(["half_year", "word"], as_index=False)[["n_scored", "log_rank_sum"]]
        .sum()
    )
    logger.info(
        "Merged %s (half-year, word) cells from %d shards.",
        f"{len(table):,}", len(shard_paths),
    )
    return table, scopes[0], deas[0], ks_seen[0]


def _require_agreement(name, values):
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise ValueError(
                f"Shards disagree on {name}: {first!r} vs {value!r}. "
                "PELT needs one DEA set and one subreddit scope."
            )


def build_signal(table, periods, min_scored, min_valid_periods, max_words):
    """
    ``signal`` is mean log rank, NaN where ``n_scored < min_scored``.

    A word is eligible when at least ``min_valid_periods`` half-years survive
    that mask. Columns are alphabetical so ``--max-words`` is deterministic.
    """
    usable = table[
        table["half_year"].isin(periods) & (table["n_scored"] >= min_scored)
    ]
    n_periods = usable.groupby("word").size()
    words = sorted(n_periods[n_periods >= min_valid_periods].index)
    logger.info(
        "Eligible words: %s (>= %d scored contexts in >= %d half-years)",
        f"{len(words):,}", min_scored, min_valid_periods,
    )
    if max_words is not None:
        words = words[:max_words]

    word_index = {w: i for i, w in enumerate(words)}
    hy_index = {h: i for i, h in enumerate(periods)}
    signal = np.full((len(periods), len(words)), np.nan, dtype=np.float64)
    counts = np.zeros((len(periods), len(words)), dtype=np.int64)

    sub = table[table["word"].isin(set(words)) & table["half_year"].isin(periods)]
    for hy, word, n, log_sum in sub.itertuples(index=False):
        i, j = hy_index[hy], word_index[word]
        counts[i, j] = int(n)
        if n >= min_scored:
            signal[i, j] = log_sum / n
    return words, signal, counts


# ============================================================================
# MAIN
# ============================================================================


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    parser = argparse.ArgumentParser(
        description="PELT on a decrease in hit-matrix mean log rank.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shard-dir", default=os.path.join(repo, "hit_matrix_shards"))
    parser.add_argument("--out-dir", default=os.path.join(here, "results_hit_logrank"))
    parser.add_argument("--min-scored", type=int, default=30,
                        help="Half-years with fewer scored contexts are set to NaN.")
    parser.add_argument("--min-valid-periods", type=int, default=6,
                        help="Minimum half-years left after masking.")
    parser.add_argument("--exclude-half-years", nargs="*", default=[],
                        help="Half-years to drop, e.g. a partial one such as 2020H2.")
    parser.add_argument("--min-size", type=int, default=3,
                        help="Minimum segment length in half-years.")
    parser.add_argument("--pen-mult", type=float, default=3.0,
                        help="PELT penalty = pen_mult * log(n).")
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="q-value cutoff for flagging a word as emerging.")
    parser.add_argument("--min-shift", type=float, default=0.0,
                        help="Minimum drop in mean log rank. ln(2) ≈ 0.693 "
                        "requires the geometric-mean rank to at least halve.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-words", default=os.path.join(here, "eval_words.json"))
    parser.add_argument("--n-plots", type=int, default=30,
                        help="Plot this many top-ranked words (plus eval words).")
    parser.add_argument("--max-words", type=int, default=None,
                        help="Only analyze the first N eligible words, alphabetically.")
    args = parser.parse_args()
    if args.min_valid_periods < 2 * args.min_size:
        parser.error("--min-valid-periods must be at least 2 * --min-size.")
    # Emergence is a fall in rank. analyze_word expects this flag.
    args.decreasing = True
    return args


def main():
    # ------------------------------------------------------------------
    # 1. Setup. The permutation test shuffles half-years, so the seed makes
    #    p-values repeatable. decreasing=True is fixed: a drop in mean log
    #    rank is the direction that means "more drug-like".
    # ------------------------------------------------------------------
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, f"*{SHARD_SUFFIX}")))
    if not shard_paths:
        raise SystemExit(f"No *{SHARD_SUFFIX} files in {args.shard_dir}.")

    # ------------------------------------------------------------------
    # 2. Record which months each half-year is missing. PELT only sees the
    #    half-years that have at least one shard; a half-year built from one
    #    month is noisier than one built from six, but the per-cell
    #    --min-scored mask is what drops it for a given word.
    # ------------------------------------------------------------------
    coverage = month_coverage(shard_paths)
    for label, info in coverage.items():
        missing = info["months_missing"]
        if missing:
            logger.warning("%s is partial, missing %s.", label, ", ".join(missing))

    # ------------------------------------------------------------------
    # 3. Merge the shards. Each cell already stores sum(log(best_rank)), so
    #    the half-year value is the sum of those sums divided by the sum of
    #    the counts. Averaging the monthly means would overweight a month
    #    with few contexts.
    # ------------------------------------------------------------------
    table, scope, dea, ks = load_logrank(shard_paths)
    half_years = sorted(table["half_year"].unique(), key=_half_year_key)
    gaps = missing_half_years(half_years)
    if gaps:
        logger.warning(
            "No shards at all for %s. Those half-years are absent from the "
            "series, so a change there is dated to the nearest present one.",
            ", ".join(gaps),
        )

    unknown = [h for h in args.exclude_half_years if h not in half_years]
    if unknown:
        raise ValueError(f"--exclude-half-years not in the shards: {unknown}")
    periods = [h for h in half_years if h not in args.exclude_half_years]
    if len(periods) < args.min_valid_periods:
        raise SystemExit(
            f"Only {len(periods)} half-years available ({', '.join(periods)}), "
            f"fewer than --min-valid-periods {args.min_valid_periods}. Lower "
            "it, or wait for more months."
        )

    # ------------------------------------------------------------------
    # 4. Build the series. signal[t, j] is word j's mean log rank in
    #    half-year t. NaN cells are half-years with too few contexts, and
    #    PELT skips them rather than treating the gap as a zero.
    # ------------------------------------------------------------------
    words, signal, counts = build_signal(
        table, periods, args.min_scored, args.min_valid_periods, args.max_words,
    )
    if len(words) == 0:
        raise SystemExit("No word had enough half-years above --min-scored.")

    finite = signal[:, : min(2000, signal.shape[1])]
    sigmas = [robust_noise(s[np.isfinite(s)]) for s in finite.T]
    noise_floor = max(0.25 * float(np.nanmedian(sigmas)), 1e-6)

    # ------------------------------------------------------------------
    # 5. PELT plus a permutation test, one word at a time. The plotted
    #    series stays in mean-log-rank units (so a real change slopes down).
    #    analyze_word flips the sign internally when it measures the drop.
    # ------------------------------------------------------------------
    results, cp_rows = [], []
    signal_of, cps_of, periods_of = {}, {}, {}
    for j, word in enumerate(words):
        column = signal[:, j]
        valid = np.isfinite(column)
        series = column[valid]
        valid_periods = [p for p, keep in zip(periods, valid) if keep]

        out = analyze_word(series, valid_periods, args, noise_floor, rng)
        if out is None:
            continue
        result, cp = out
        result.update(
            word=word,
            total_scored=int(counts[:, j].sum()),
            first_valid_half_year=valid_periods[0],
            last_valid_half_year=valid_periods[-1],
        )
        results.append(result)
        cp_rows.extend({"word": word, **row} for row in cp)

        signal_of[word] = series
        periods_of[word] = valid_periods
        cps_of[word] = [
            valid_periods.index(h)
            for h in (result["changepoints"].split(";") if result["changepoints"] else [])
        ]
        if (j + 1) % 10000 == 0:
            logger.info("  analyzed %s / %s words", f"{j + 1:,}", f"{len(words):,}")

    if not results:
        raise SystemExit("No words had enough valid half-years to analyze.")

    # ------------------------------------------------------------------
    # 6. Flag and rank. shift is the largest drop in mean log rank, so a
    #    larger shift is a stronger move toward drug-name contexts.
    #    geomean_rank_* is exp(level): the typical best DEA rank before and
    #    after that drop. rank_ratio < 1 means the rank got better.
    # ------------------------------------------------------------------
    results = pd.DataFrame(results)
    results["q_value"] = benjamini_hochberg(results["p_value"])
    results["emerging"] = (
        (results["q_value"] < args.alpha)
        & (results["shift"] > max(args.min_shift, 0.0))
    )
    results = results.sort_values(
        ["emerging", "shift", "q_value"], ascending=[False, False, True]
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    results["geomean_rank_before"] = np.exp(results["level_before"])
    results["geomean_rank_after"] = np.exp(results["level_after"])
    results["rank_ratio"] = np.exp(-results["shift"])

    front = ["rank", "word", "emerging", "onset_half_year", "shift", "q_value",
             "geomean_rank_before", "geomean_rank_after", "rank_ratio"]
    results = results[front + [c for c in results.columns if c not in front]]

    # ------------------------------------------------------------------
    # 7. Outputs: the ranking, every change point, the eval-word scorecard,
    #    plots of the top words and the eval words, and the run metadata.
    # ------------------------------------------------------------------
    results.to_csv(os.path.join(args.out_dir, "cpd_results.csv"), index=False)
    pd.DataFrame(cp_rows).to_csv(os.path.join(args.out_dir, "cpd_changepoints.csv"), index=False)
    logger.info(
        "Analyzed %s words; %s flagged (q < %.2f, drop in mean log rank > %.3f).",
        f"{len(results):,}", f"{int(results['emerging'].sum()):,}",
        args.alpha, args.min_shift,
    )

    eval_df = evaluate(results, args.eval_words)
    eval_df.to_csv(os.path.join(args.out_dir, "cpd_eval.csv"), index=False)
    for group, g in eval_df.groupby("group", sort=False):
        elig = g[g["eligible"]]
        logger.info(
            "  eval %-22s eligible %2d/%2d, emerging %2d, median percentile %s",
            group, len(elig), len(g), int(elig["emerging"].sum()) if len(elig) else 0,
            f"{elig['percentile'].median():.1f}" if len(elig) else "n/a",
        )

    to_plot = list(results["word"].head(args.n_plots))
    to_plot += [w for w in eval_df.loc[eval_df["eligible"], "word"] if w not in to_plot]
    plot_words(
        to_plot, signal_of, cps_of, periods_of,
        "mean log rank", os.path.join(args.out_dir, "plots"),
    )
    logger.info("Saved %d plots to %s", len(to_plot), os.path.join(args.out_dir, "plots"))

    run_meta = {
        "script": "cpd_hit_logrank.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "shard_dir": os.path.abspath(args.shard_dir),
        "signal": "mean_log_rank = sum(log(best_rank)) / n_scored",
        "log_base": "natural",
        "direction": "decrease (DEA term ranks higher)",
        "shift_units": "drop in mean log rank; exp(-shift) is the factor applied to the geometric-mean rank",
        "scope": scope,
        "dea": dea,
        "ks_in_shards": ks,
        "half_years_used": periods,
        "half_years_excluded": list(args.exclude_half_years),
        "half_years_missing_from_output": gaps,
        "month_coverage": coverage,
        "eligibility": {
            "min_scored_per_cell": args.min_scored,
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
            "test": "one-sided permutation test of max standardized improvement in rank",
            "n_perm": args.n_perm,
            "multiple_testing": "Benjamini-Hochberg",
            "alpha": args.alpha,
            "min_shift": args.min_shift,
            "seed": args.seed,
        },
        "counts": {
            "shards": len(shard_paths),
            "eligible_words": len(words),
            "analyzed_words": int(len(results)),
            "emerging_words": int(results["emerging"].sum()),
        },
    }
    with open(os.path.join(args.out_dir, "cpd_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    logger.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
