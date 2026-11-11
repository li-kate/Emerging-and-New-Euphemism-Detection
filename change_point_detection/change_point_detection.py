"""
Change point detection on per-word time series (half-years x words).

An emerging euphemism is a word whose usage shifts relative to its OWN earlier
half-years, not a word that is always drug-related. This script finds those
shifts with PELT (ruptures) and ranks words by the size of the upward shift.

Any [half_years x words] matrix aligned to the frequency matrix can be used:

  * ``--signal frequency`` (default): log occurrences per million tokens,
    computed from ``freq.npy`` and ``freq_totals.csv``.
  * ``--signal matrix --signal-npy PATH``: a precomputed matrix such as the hit
    matrix's ``hits_rate_5.npy`` / ``hits_mean_log_rank.npy`` or the cosine
    similarity matrix ``cosine_similarity.npy``. NaN cells are skipped. Pass
    ``--decreasing`` for signals where a DROP means more drug-like (for example
    ``mean_log_rank``).

Per word:

  1. Keep half-years whose corpus is complete (the partial last half-year is
     dropped; see ``--partial-threshold``).
  2. Eligibility: at least ``--min-freq`` occurrences in at least
     ``--min-periods`` half-years (frequency matrix counts).
  3. Standardize by a robust noise estimate (MAD of first differences) and run
     PELT with an L2 (mean-shift) cost, penalty ``--pen-mult * log(n)``, and a
     minimum segment length of ``--min-size`` half-years.
  4. The emergence point is the change point with the largest upward jump
     between adjacent segment means; ``shift`` is that jump in signal units
     (for frequency, a log fold change in rate).
  5. Significance: one-sided permutation test of the maximum standardized
     upward mean shift over all splits (time order shuffled ``--n-perm``
     times), then Benjamini-Hochberg q-values across words. Shuffling ignores
     autocorrelation, so a smooth trend can also come out significant.

Outputs (in ``--out-dir``):

  * ``cpd_results.csv``: one row per eligible word, ranked.
  * ``cpd_changepoints.csv``: every detected change point (long format).
  * ``cpd_eval.csv``: results for the evaluation words, with rank/percentile.
  * ``cpd_meta.json``: parameters, dropped half-years, and summary counts.
  * ``plots/``: series with segment means and change points for the top
    words and the evaluation words.

Example:

    python change_point_detection.py \\
        --freq-dir ../other_outputs \\
        --out-dir results_frequency
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ruptures as rpt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Additive smoothing (in counts) before taking the log rate, so zero-count
# half-years have a finite value and tiny counts are not over-weighted.
COUNT_PSEUDOCOUNT = 0.5


# ============================================================================
# LOADING
# ============================================================================


def load_frequency(freq_dir, stem="freq"):
    """Load the frequency matrix, its labels, and per-half-year token totals."""
    freq = np.load(os.path.join(freq_dir, f"{stem}.npy"), mmap_mode="r")
    half_years = pd.read_csv(
        os.path.join(freq_dir, f"{stem}_half_years.csv")
    )["half_year"].astype(str).tolist()
    words = pd.read_csv(
        os.path.join(freq_dir, f"{stem}_words.csv"),
        keep_default_na=False,
    )["word"].astype(str).tolist()
    totals = pd.read_csv(os.path.join(freq_dir, f"{stem}_totals.csv"))
    totals["half_year"] = totals["half_year"].astype(str)
    totals = totals.set_index("half_year").reindex(half_years)

    if freq.shape != (len(half_years), len(words)):
        raise ValueError(
            f"freq.npy shape {freq.shape} does not match labels "
            f"({len(half_years)} half-years, {len(words)} words)."
        )
    if totals["total_tokens"].isna().any():
        raise ValueError(f"{stem}_totals.csv is missing some half-years.")

    logger.info("Loaded frequency matrix %s from %s", freq.shape, freq_dir)
    return freq, half_years, words, totals["total_tokens"].to_numpy(np.float64)


def complete_periods(half_years, total_tokens, threshold):
    """
    Mask of half-years with a complete corpus.

    A half-year is treated as partial when its token total is below
    ``threshold`` times the larger of its neighbors' totals. Comparing to
    neighbors rather than a global median keeps early half-years, which are
    smaller only because the corpus grew. The last half-year of a dump that
    is still being collected is the usual case.
    """
    keep = np.ones(len(half_years), dtype=bool)
    for i in range(len(half_years)):
        neighbors = total_tokens[max(0, i - 1): i + 2]
        neighbors = np.delete(neighbors, min(i, 1))
        if len(neighbors) and total_tokens[i] < threshold * neighbors.max():
            keep[i] = False
    dropped = [h for h, k in zip(half_years, keep) if not k]
    if dropped:
        logger.info("Dropping partial half-year(s): %s", ", ".join(dropped))
    return keep


def eligible_columns(counts, min_freq, min_periods):
    """Columns with >= ``min_freq`` occurrences in >= ``min_periods`` half-years."""
    cols = np.flatnonzero((counts >= min_freq).sum(axis=0) >= min_periods)
    logger.info(
        "Eligible words: %s (>= %d occurrences in >= %d half-years)",
        f"{len(cols):,}", min_freq, min_periods,
    )
    return cols


def build_signal(args, counts, total_tokens):
    """Return the [half_years x eligible words] signal and its display name."""
    if args.signal == "frequency":
        rate = (counts + COUNT_PSEUDOCOUNT) / total_tokens[:, None] * 1e6
        return np.log(rate), "log occurrences per million tokens"

    signal = np.load(args.signal_npy, mmap_mode="r")
    return signal, args.signal_name or os.path.basename(args.signal_npy)


# ============================================================================
# CHANGE POINT DETECTION
# ============================================================================


def robust_noise(series):
    """Noise scale from the MAD of first differences (insensitive to shifts)."""
    diffs = np.diff(series)
    if len(diffs) == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2)


def detect_changepoints(z, min_size, penalty):
    """PELT on a standardized series; returns change indices (segment starts)."""
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(z.reshape(-1, 1))
    return algo.predict(pen=penalty)[:-1]


def segment_means(series, changepoints):
    bounds = [0, *changepoints, len(series)]
    return [float(series[a:b].mean()) for a, b in zip(bounds[:-1], bounds[1:])]


def max_upward_stat(x, min_size):
    """
    Maximum standardized upward mean shift over all valid splits.

    For a split at ``t`` the statistic is
    ``(mean(x[t:]) - mean(x[:t])) * sqrt(t * (n - t) / n)``. ``x`` may be 2-D
    (one series per row), which is how the permutations are scored at once.
    """
    x = np.atleast_2d(x)
    n = x.shape[1]
    csum = np.cumsum(x, axis=1)
    total = csum[:, -1:]
    t = np.arange(min_size, n - min_size + 1)
    before = csum[:, t - 1] / t
    after = (total - csum[:, t - 1]) / (n - t)
    stat = (after - before) * np.sqrt(t * (n - t) / n)
    return stat.max(axis=1)


def permutation_pvalue(x, min_size, n_perm, rng):
    observed = max_upward_stat(x, min_size)[0]
    perms = rng.permuted(np.tile(x, (n_perm, 1)), axis=1)
    null = max_upward_stat(perms, min_size)
    return float(observed), float((1 + np.sum(null >= observed)) / (n_perm + 1))


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def analyze_word(series, periods, args, noise_floor, rng):
    """
    Run change point detection for one word.

    ``series`` holds only the valid (finite) half-years, in time order, and
    ``periods`` their labels. Returns a result dict and the change point rows,
    or ``None`` if there are too few valid half-years.
    """
    n = len(series)
    if n < 2 * args.min_size:
        return None

    oriented = -series if args.decreasing else series
    sigma = max(robust_noise(oriented), noise_floor)
    z = (oriented - oriented.mean()) / sigma

    changepoints = detect_changepoints(z, args.min_size, args.pen_mult * np.log(n))
    means = segment_means(oriented, changepoints)
    jumps = np.diff(means)

    up_stat, p_value = permutation_pvalue(z, args.min_size, args.n_perm, rng)

    result = {
        "n_valid": n,
        "n_changepoints": len(changepoints),
        "changepoints": ";".join(periods[c] for c in changepoints),
        "onset_half_year": "",
        "level_before": np.nan,
        "level_after": np.nan,
        "shift": 0.0,
        "max_up_stat": up_stat,
        "p_value": p_value,
    }
    if len(jumps) and jumps.max() > 0:
        j = int(np.argmax(jumps))
        sign = -1.0 if args.decreasing else 1.0
        result.update(
            onset_half_year=periods[changepoints[j]],
            level_before=sign * means[j],
            level_after=sign * means[j + 1],
            shift=float(jumps[j]),
        )

    rows = [
        {
            "half_year": periods[c],
            "level_before": means[k] * (-1.0 if args.decreasing else 1.0),
            "level_after": means[k + 1] * (-1.0 if args.decreasing else 1.0),
            "jump": float(jumps[k]),
        }
        for k, c in enumerate(changepoints)
    ]
    return result, rows


# ============================================================================
# OUTPUTS
# ============================================================================


def evaluate(results, eval_file):
    """Look up each evaluation word's rank and detection outcome."""
    with open(eval_file, encoding="utf-8") as f:
        groups = json.load(f)["groups"]

    by_word = results.set_index("word")
    n = len(results)
    rows = []
    for group, words in groups.items():
        for word in words:
            row = {"group": group, "word": word}
            if word in by_word.index:
                r = by_word.loc[word]
                row.update(
                    eligible=True,
                    rank=int(r["rank"]),
                    percentile=round(100 * (1 - (r["rank"] - 1) / n), 2),
                    emerging=bool(r["emerging"]),
                    onset_half_year=r["onset_half_year"],
                    shift=r["shift"],
                    q_value=r["q_value"],
                    changepoints=r["changepoints"],
                )
            else:
                row["eligible"] = False
            rows.append(row)
    return pd.DataFrame(rows)


def plot_words(words, signal_of, cps_of, periods_of, ylabel, out_dir, titles=None,
               ylim=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    for word in words:
        series, periods = signal_of[word], periods_of[word]
        changepoints = cps_of[word]
        x = np.arange(len(series))

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(x, series, marker="o", ms=3, lw=1, color="0.35")
        bounds = [0, *changepoints, len(series)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            ax.hlines(series[a:b].mean(), a - 0.4, b - 0.6, color="tab:red", lw=2)
        for c in changepoints:
            ax.axvline(c - 0.5, color="tab:blue", ls="--", lw=1)
        step = max(1, len(periods) // 12)
        ax.set_xticks(x[::step], [periods[i] for i in x[::step]], rotation=45)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(titles.get(word, word) if titles else word)
        fig.tight_layout()
        safe = re.sub(r"[^\w\-]+", "_", word)
        fig.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=120)
        plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="PELT change point detection on per-word time series.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--freq-dir", required=True,
                        help="Frequency matrix directory (freq.npy + labels + totals).")
    parser.add_argument("--freq-stem", default="freq")
    parser.add_argument("--signal", choices=["frequency", "matrix"],
                        default="frequency")
    parser.add_argument("--signal-npy",
                        help="[half_years x words] matrix aligned to the frequency "
                        "matrix (required with --signal matrix).")
    parser.add_argument("--signal-name", help="Label for plots and metadata.")
    parser.add_argument("--decreasing", action="store_true",
                        help="Treat a decrease as the emergence direction "
                        "(e.g. mean_log_rank).")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--min-freq", type=int, default=30)
    parser.add_argument("--min-periods", type=int, default=2)
    parser.add_argument("--partial-threshold", type=float, default=0.5,
                        help="Drop half-years with fewer tokens than this "
                        "fraction of the median half-year.")

    parser.add_argument("--min-size", type=int, default=3,
                        help="Minimum segment length in half-years.")
    parser.add_argument("--pen-mult", type=float, default=3.0,
                        help="PELT penalty = pen_mult * log(n).")
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="q-value cutoff for flagging a word as emerging.")
    parser.add_argument("--min-shift", type=float, default=None,
                        help="Minimum upward jump (signal units) to flag a word "
                        "as emerging. Defaults to log(2), a doubling of the "
                        "rate, for the frequency signal and 0 otherwise.")
    parser.add_argument("--seed", type=int, default=0)

    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--eval-words", default=os.path.join(here, "eval_words.json"))
    parser.add_argument("--n-plots", type=int, default=30,
                        help="Plot this many top-ranked words (plus eval words).")
    parser.add_argument("--max-words", type=int, default=None,
                        help="Only analyze the first N eligible words (test runs).")

    args = parser.parse_args()
    if args.signal == "matrix" and not args.signal_npy:
        parser.error("--signal matrix requires --signal-npy.")
    if args.decreasing and args.signal == "frequency":
        parser.error("--decreasing does not apply to the frequency signal.")
    if args.min_shift is None:
        args.min_shift = float(np.log(2)) if args.signal == "frequency" else 0.0
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    freq, half_years, words, total_tokens = load_frequency(args.freq_dir, args.freq_stem)
    keep = complete_periods(half_years, total_tokens, args.partial_threshold)
    periods = [h for h, k in zip(half_years, keep) if k]

    counts_all = np.asarray(freq[keep])
    cols = eligible_columns(counts_all, args.min_freq, args.min_periods)
    if args.max_words is not None:
        cols = cols[: args.max_words]
    counts = counts_all[:, cols].astype(np.float64)
    del counts_all

    signal, ylabel = build_signal(args, counts, total_tokens[keep])
    if args.signal == "matrix":
        if signal.shape != freq.shape:
            raise ValueError(
                f"Signal shape {signal.shape} does not match the frequency "
                f"matrix {freq.shape}; both must use the same half-years and words."
            )
        signal = np.asarray(signal[keep][:, cols], dtype=np.float64)

    sigmas = [robust_noise(s[np.isfinite(s)]) for s in signal.T[:2000]]
    noise_floor = max(0.25 * np.nanmedian(sigmas), 1e-6)

    results, cp_rows = [], []
    signal_of, cps_of, periods_of = {}, {}, {}
    for i, col in enumerate(cols):
        word = words[col]
        column = signal[:, i]
        valid = np.isfinite(column)
        series = column[valid]
        valid_periods = [p for p, v in zip(periods, valid) if v]

        out = analyze_word(series, valid_periods, args, noise_floor, rng)
        if out is None:
            continue
        result, rows = out
        result.update(
            word=word,
            total_count=int(counts[:, i].sum()),
            peak_count=int(counts[:, i].max()),
        )
        results.append(result)
        cp_rows.extend({"word": word, **r} for r in rows)

        signal_of[word] = series
        periods_of[word] = valid_periods
        cps_of[word] = [valid_periods.index(h) for h in
                        (result["changepoints"].split(";") if result["changepoints"] else [])]

        if (i + 1) % 10000 == 0:
            logger.info("  analyzed %s / %s words", f"{i + 1:,}", f"{len(cols):,}")

    results = pd.DataFrame(results)
    results["q_value"] = benjamini_hochberg(results["p_value"])
    results["emerging"] = (
        (results["q_value"] < args.alpha)
        & (results["shift"] > max(args.min_shift, 0.0))
    )
    if args.signal == "frequency":
        results["rate_before_per_million"] = np.exp(results["level_before"])
        results["rate_after_per_million"] = np.exp(results["level_after"])
        results["fold_change"] = np.exp(results["shift"])

    results = results.sort_values(
        ["emerging", "shift", "q_value"], ascending=[False, False, True]
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    front = ["rank", "word", "emerging", "onset_half_year", "shift", "q_value"]
    results = results[front + [c for c in results.columns if c not in front]]

    results.to_csv(os.path.join(args.out_dir, "cpd_results.csv"), index=False)
    pd.DataFrame(cp_rows).to_csv(
        os.path.join(args.out_dir, "cpd_changepoints.csv"), index=False
    )
    logger.info(
        "Analyzed %s words; %s flagged as emerging (q < %.2f, shift > %.3f).",
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
    plot_words(to_plot, signal_of, cps_of, periods_of, ylabel,
               os.path.join(args.out_dir, "plots"))
    logger.info("Saved %d plots to %s", len(to_plot), os.path.join(args.out_dir, "plots"))

    meta = {
        "script": "change_point_detection.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "signal": args.signal,
        "signal_npy": os.path.abspath(args.signal_npy) if args.signal_npy else None,
        "signal_label": ylabel,
        "decreasing": args.decreasing,
        "freq_dir": os.path.abspath(args.freq_dir),
        "half_years_used": periods,
        "half_years_dropped_partial": [h for h, k in zip(half_years, keep) if not k],
        "eligibility": {"min_freq": args.min_freq, "min_periods": args.min_periods},
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
        json.dump(meta, f, indent=2)
    logger.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
