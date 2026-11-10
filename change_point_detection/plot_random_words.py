"""Plot words from a finished cosine change point run without rerunning PELT.

Reads the saved results (cpd_results.csv, cpd_meta.json) and rebuilds each
word's series from the cosine matrix. Each plot's title shows the word's
rank, whether it was flagged as emerging, its shift and q-value.

Two modes:
    Random sample (default): plot --n random words that are not already in
    <results-dir>/plots, saved to <results-dir>/plots_random.

        python plot_random_words.py --n 1000

    Unscaled copies (--unscaled): redraw every word in <results-dir>/plots and
    <results-dir>/plots_random with the y-axis fixed to --ylim, saved to
    <results-dir>/unscaled_plots/plots and .../plots_random.

        python plot_random_words.py --unscaled
"""

import argparse
import json
import logging
import os
import re

import numpy as np
import pandas as pd

from change_point_detection import plot_words
from cpd_cosine_similarity import load_matrix, load_metadata, load_words

logger = logging.getLogger(__name__)


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Plot words from a cosine CPD run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", default=os.path.join(here, "results_cosine"))
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of random words to plot.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unscaled", action="store_true",
                        help="Redraw the existing plots with a fixed y-axis instead "
                        "of sampling new words.")
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.0, 1.0],
                        help="y-axis range used with --unscaled.")
    return parser.parse_args()


def safe_name(word):
    """File name stem used by plot_words."""
    return re.sub(r"[^\w\-]+", "_", word)


def words_in_plot_dir(plot_dir, results):
    """Map the PNGs in ``plot_dir`` back to words (best-ranked word if names collide)."""
    stems = {os.path.splitext(f)[0] for f in os.listdir(plot_dir) if f.endswith(".png")}
    picked = {}
    for word in results["word"]:
        stem = safe_name(word)
        if stem in stems and stem not in picked:
            picked[stem] = word
    return list(picked.values())


class SeriesLoader:
    """Rebuilds per-word series exactly as the original run did."""

    def __init__(self, run):
        self.run = run
        self.meta = load_metadata(run["cosine_dir"])
        self.words = load_words(run["words_csv"], self.meta["vocabulary_size"])
        self.rows = [self.meta["half_years"].index(h) for h in run["half_years_used"]]
        self.periods = run["half_years_used"]
        self.min_occurrences = run["eligibility"]["min_occurrences_per_cell"]
        self.metric = load_matrix(run["cosine_dir"], self.meta, run["metric"],
                                  self.meta["matrix_files"][run["metric"]])
        self.n_occurrences = load_matrix(run["cosine_dir"], self.meta, "n_occurrences",
                                         self.meta["occurrence_count_file"])

    def load(self, word_list):
        wanted = set(word_list)
        col_of = {}
        for col, w in enumerate(self.words):
            if w in wanted and w not in col_of:
                col_of[w] = col
        cols = np.array([col_of[w] for w in word_list])
        order = np.argsort(cols)
        signal = np.empty((len(self.rows), len(cols)))
        counts = np.empty((len(self.rows), len(cols)), dtype=np.int64)
        signal[:, order] = np.asarray(self.metric[:, cols[order]][self.rows], dtype=np.float64)
        counts[:, order] = np.asarray(self.n_occurrences[:, cols[order]][self.rows],
                                      dtype=np.int64)
        signal[counts < self.min_occurrences] = np.nan
        return signal


def plot_selection(selection, loader, n_total, out_dir, ylim=None):
    """Plot the words in ``selection`` (rows of cpd_results.csv) into ``out_dir``."""
    signal = loader.load(list(selection["word"]))
    signal_of, cps_of, periods_of, titles = {}, {}, {}, {}
    for i, r in enumerate(selection.itertuples(index=False)):
        valid = np.isfinite(signal[:, i])
        valid_periods = [p for p, v in zip(loader.periods, valid) if v]
        signal_of[r.word] = signal[valid, i]
        periods_of[r.word] = valid_periods
        cps_of[r.word] = [valid_periods.index(h) for h in
                          (r.changepoints.split(";") if r.changepoints else [])]
        flag = "EMERGING" if str(r.emerging) == "True" else "not flagged"
        titles[r.word] = (f"{r.word}  |  rank {r.rank:,}/{n_total:,}  |  {flag}  |  "
                          f"shift {float(r.shift):.3f}, q {float(r.q_value):.3g}")

    plot_words(list(selection["word"]), signal_of, cps_of, periods_of,
               loader.run["metric"].replace("_", " "), out_dir, titles=titles, ylim=ylim)
    selection.sort_values("rank").to_csv(os.path.join(out_dir, "plotted_words.csv"),
                                         index=False)
    logger.info("Saved %d plots to %s", len(selection), out_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    with open(os.path.join(args.results_dir, "cpd_meta.json"), encoding="utf-8") as f:
        run = json.load(f)
    results = pd.read_csv(os.path.join(args.results_dir, "cpd_results.csv"),
                          keep_default_na=False, low_memory=False)
    n_total = len(results)
    loader = SeriesLoader(run)

    if args.unscaled:
        for name in ["plots", "plots_random"]:
            src = os.path.join(args.results_dir, name)
            if not os.path.isdir(src):
                logger.warning("Skipping %s (not found)", src)
                continue
            selection = results[results["word"].isin(words_in_plot_dir(src, results))]
            plot_selection(selection, loader, n_total,
                           os.path.join(args.results_dir, "unscaled_plots", name),
                           ylim=args.ylim)
        return

    plotted_dir = os.path.join(args.results_dir, "plots")
    already = set(words_in_plot_dir(plotted_dir, results)) if os.path.isdir(plotted_dir) else set()
    pool = results[~results["word"].isin(already)]
    sample = pool.sample(n=min(args.n, len(pool)), random_state=args.seed)
    logger.info("Sampled %d of %d unplotted words", len(sample), len(pool))
    plot_selection(sample, loader, n_total, os.path.join(args.results_dir, "plots_random"))


if __name__ == "__main__":
    main()
