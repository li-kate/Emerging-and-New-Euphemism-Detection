"""
Reduce step for the hit matrix: sum the per-month shards from ``hit_matrix.py``
into the final summary table and matrices described in ``hit_matrix_plan.md``
sections 7 and 8.

``hit_matrix.py`` (one SLURM array task per month) writes one ``*.part.json``
shard holding running per (half_year, word) counts. This script sums the cells
across all months and writes the **raw hit matrix** (counts + diagnostics)
plus the frequency denominator — but NOT the hit *rate*. Hit rates are left for
downstream analysis, which can compute ``hits_k / frequency`` directly and
experiment with different reliability thresholds (the old ``--min-contexts``)
without re-running the merge.

This merge is **matrix-only**, mirroring the matrix-only ``hit_matrix.py``. The
per-context pilot table and the manual-spot-check word sample are produced by
their own experiment pipelines (``pilot_hits.py`` / ``merge_pilot_hits.py`` and
``sample_contexts.py`` / ``merge_sample_contexts.py``), so a plain matrix merge
never has to know about them. Outputs:

  * ``<stem>_summary.csv`` — one row per (half_year, word): ``frequency``,
    ``n_scored``, ``hits_k`` for each k, ``min_rank``, ``mean_log_rank``,
    ``mean_soft``. (No ``hit_rate_k``; divide ``hits_k`` by ``frequency``
    downstream.)
  * ``<stem>_hits_k.npy`` — hit-count matrices ``[half_years x words]`` for each
    k, plus ``mean_log_rank`` / ``min_rank`` / ``mean_soft`` matrices, all
    aligned EXACTLY to the frequency matrix (same half-years, same words, same
    order). No rate matrices are written.
  * ``<stem>_meta.json`` — scope, DEA, k list, and the sanity-check results.

The frequency matrix is still loaded, for three non-rate purposes: the matrix
axis template (half-years x words order), the ``frequency`` column (so
downstream can compute rates without reloading it), and the ``n_scored`` vs
``frequency`` sanity check.

Cell states in the matrices (cumulative causal ``--min-freq`` rule from
``hit_matrix.py``): a cell is left ``NaN`` where the word was not eligible yet
(not scored), ``0`` where it was eligible/scored with no hit, and the scored
value otherwise — so "not yet emerged" stays distinguishable from "scored,
zero hits".

Example:

    python merge_hit_matrix.py \\
        --shard-dir hits_drug_subreddits/shards \\
        --freq-dir freq_drug_subreddits \\
        --out-dir hits_drug_subreddits
"""

import argparse
import glob
import json
import logging
import math
import os
from collections import defaultdict
from itertools import pairwise

import numpy as np
import pandas as pd
from hit_matrix_core import KS, SHARD_SUFFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# FREQUENCY MATRIX (denominator + axis alignment)
# ============================================================================


def load_freq(freq_dir, stem="freq"):
    """Load the frequency matrix as a DataFrame (half_years x words)."""
    matrix = np.load(os.path.join(freq_dir, f"{stem}.npy"))
    half_years = pd.read_csv(
        os.path.join(freq_dir, f"{stem}_half_years.csv")
    )["half_year"].astype(str).tolist()
    words = pd.read_csv(
        os.path.join(freq_dir, f"{stem}_words.csv")
    )["word"].astype(str).tolist()
    df = pd.DataFrame(matrix, index=half_years, columns=words)
    logger.info("Loaded frequency matrix %s from %s", df.shape, freq_dir)
    return df


# ============================================================================
# SHARD MERGE
# ============================================================================


def find_shards(shard_dir):
    shards = sorted(glob.glob(os.path.join(shard_dir, f"*{SHARD_SUFFIX}")))
    if not shards:
        raise FileNotFoundError(
            f"No shards matching *{SHARD_SUFFIX} in {shard_dir}. Did the month "
            f"array run and finish?"
        )
    return shards


def _empty_cell():
    cell = {"n_scored": 0, "min_rank": None, "log_rank_sum": 0.0, "soft_sum": 0.0}
    for k in KS:
        cell[f"hits_{k}"] = 0
    return cell


def merge_shards(shard_files):
    """
    Sum all month shards.

    Returns:
      cells       - dict[(half_year, word)] -> merged running counts
      drop_counts - summed record drop reasons
      ctx_drops   - summed context-construction drop reasons
      scopes      - list of per-shard scope dicts (checked for agreement)
      dea_metas   - list of per-shard DEA metadata dicts
      target_sels - list of per-shard target-selection dicts (min_freq +
                    kept/dropped counts; checked for agreement on min_freq)
    """
    cells = defaultdict(_empty_cell)
    drop_counts = defaultdict(int)
    ctx_drops = defaultdict(int)
    scopes, dea_metas, target_sels = [], [], []

    for i, path in enumerate(shard_files, 1):
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)

        if shard.get("ks") != KS:
            raise ValueError(
                f"Shard {path} was built with ks={shard.get('ks')} but this "
                f"merge expects KS={KS}. Rebuild the shards or align KS."
            )

        for key, src in shard["cells"].items():
            hy, w = key.split("\t", 1)
            dst = cells[(hy, w)]
            dst["n_scored"] += src["n_scored"]
            dst["log_rank_sum"] += src["log_rank_sum"]
            dst["soft_sum"] += src["soft_sum"]
            smr = src["min_rank"]
            if smr is not None:
                dst["min_rank"] = smr if dst["min_rank"] is None \
                    else min(dst["min_rank"], smr)
            for k in KS:
                dst[f"hits_{k}"] += src[f"hits_{k}"]

        for reason, c in shard.get("drop_counts", {}).items():
            drop_counts[reason] += c
        for reason, c in shard.get("dropped_contexts", {}).items():
            ctx_drops[reason] += c
        if "scope" in shard:
            scopes.append(shard["scope"])
        if "dea" in shard:
            dea_metas.append(shard["dea"])
        if "target_selection" in shard:
            target_sels.append(shard["target_selection"])

        logger.info("Merged shard %d/%d: %s (running cells: %s)",
                    i, len(shard_files), os.path.basename(path),
                    f"{len(cells):,}")

    return (cells, drop_counts, ctx_drops, scopes, dea_metas, target_sels)


# ============================================================================
# SUMMARY TABLE
# ============================================================================


def build_summary(cells, freq):
    """
    Build the per (half_year, word) summary DataFrame (plan section 7): the raw
    hit matrix, NOT rates.

    ``frequency`` comes from the external frequency matrix and is included so
    downstream analysis can compute ``hits_k / frequency`` itself (and choose a
    reliability threshold then). ``mean_log_rank`` and ``mean_soft`` divide by
    ``n_scored``. No ``hit_rate_k`` columns are produced here.
    """
    freq_lookup = freq.stack()  # (half_year, word) -> count

    rows = []
    for (hy, w), cell in cells.items():
        n = cell["n_scored"]
        frequency = int(freq_lookup.get((hy, w), 0))
        row = {
            "half_year": hy,
            "word": w,
            "frequency": frequency,
            "n_scored": n,
            "min_rank": cell["min_rank"],
            "mean_log_rank": (cell["log_rank_sum"] / n) if n else math.nan,
            "mean_soft": (cell["soft_sum"] / n) if n else math.nan,
        }
        for k in KS:
            row[f"hits_{k}"] = cell[f"hits_{k}"]
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["half_year", "word"]) \
        .reset_index(drop=True)
    return summary


# ============================================================================
# MATRICES (aligned to the frequency matrix)
# ============================================================================


def pivot_aligned(summary, value_col, freq):
    """Pivot ``value_col`` to [half_years x words], reindexed to ``freq``'s axes."""
    m = summary.pivot(index="half_year", columns="word", values=value_col)
    return m.reindex(index=freq.index, columns=freq.columns)


def eligibility_mask(freq, min_freq):
    """
    Boolean [half_years x words] mask of the cells that ``hit_matrix.py`` would
    have scored under the cumulative causal (real-time) ``--min-freq`` rule:
    True for ``(hy, word)`` iff the word has reached ``min_freq`` in ``hy`` or
    an earlier half-year.

    This mirrors ``hit_matrix.load_target_words`` so the merge can tell a cell
    that was deliberately **not scored** (word had not emerged yet) from a cell
    that was scored and simply had zero hits. ``min_freq <= 1`` makes every
    cell eligible (no floor). Rows are assumed chronological, matching the
    frequency matrix / ``freq_half_years.csv`` order.
    """
    reaches = freq.to_numpy() >= max(int(min_freq), 1)
    if min_freq is None or min_freq <= 1:
        eligible = np.ones_like(reaches, dtype=bool)
    else:
        # Cumulative down each column: once True, stays True (emerged onward).
        eligible = np.maximum.accumulate(reaches, axis=0)
    return pd.DataFrame(eligible, index=freq.index, columns=freq.columns)


def save_matrices(out_dir, stem, summary, freq, eligible):
    """
    Write the hit-count and diagnostic matrices aligned to the frequency
    matrix. No rate matrices are written — hit rates are computed downstream.

    All matrices keep the full frequency-matrix axes (same half-years, same
    words, same order) so downstream code can index them identically to the
    frequency matrix.

    Three cell states are kept distinct, which matters under the cumulative
    causal ``--min-freq`` rule (a word is scored only in half-years at or after
    it first reached the floor):
      * **not eligible** — the word had not emerged in that half-year, so it
        was never scored. Left ``NaN`` in every matrix, including ``hits_k``.
      * **eligible, zero hits** — the word was scored there and produced no DEA
        hit. A genuine ``0`` in ``hits_k``.
      * **eligible, some hits** — the scored value.

    ``eligible`` is the [half_years x words] mask from ``eligibility_mask``.
    Only eligible cells with no summary row are zero-filled in ``hits_k``;
    non-eligible cells stay ``NaN`` so "not scored (not yet emerged)" is never
    confused with "scored, zero hits". The diagnostic matrices
    (``mean_log_rank``/``min_rank``/``mean_soft``) and the rate matrices are
    never zero-filled.
    """
    eligible_np = eligible.to_numpy()

    def save(name, df, zero_fill_eligible=False):
        if zero_fill_eligible:
            arr = np.array(df.to_numpy(dtype=np.float64), copy=True)
            # Fill 0 only where the cell was eligible (scored) but has no value;
            # non-eligible cells keep NaN.
            fill_here = eligible_np & np.isnan(arr)
            arr[fill_here] = 0.0
            df = pd.DataFrame(arr, index=df.index, columns=df.columns)
        path = os.path.join(out_dir, f"{stem}_{name}.npy")
        np.save(path, df.to_numpy(dtype=np.float64))
        logger.info("Saved %s %s", path, df.shape)

    for k in KS:
        # hits: eligible cells with no hit are 0; non-eligible cells stay NaN.
        save(f"hits_{k}", pivot_aligned(summary, f"hits_{k}", freq),
             zero_fill_eligible=True)
    save("mean_log_rank", pivot_aligned(summary, "mean_log_rank", freq))
    save("min_rank", pivot_aligned(summary, "min_rank", freq))
    save("mean_soft", pivot_aligned(summary, "mean_soft", freq))


# ============================================================================
# SANITY CHECKS (plan section 8)
# ============================================================================


def run_sanity_checks(summary, freq):
    """
    Validate the merged results. Returns a dict of findings (also logged).

    Checks:
      * hit nesting: hits_5 <= hits_10 <= ... <= hits_50 per row.
      * n_scored vs frequency: how many cells disagree, and the total gap.
    """
    findings = {}

    # Nesting.
    nesting_violations = 0
    ks_sorted = sorted(KS)
    for a, b in pairwise(ks_sorted):
        bad = (summary[f"hits_{a}"] > summary[f"hits_{b}"]).sum()
        nesting_violations += int(bad)
    findings["nesting_violations"] = nesting_violations
    if nesting_violations:
        logger.error("Hit nesting violated in %d comparisons (bug).",
                     nesting_violations)
    else:
        logger.info("Hit nesting OK (hits_5 <= ... <= hits_50).")

    # n_scored vs frequency.
    mismatch = summary[summary["n_scored"] != summary["frequency"]]
    findings["n_scored_mismatch_cells"] = len(mismatch)
    findings["n_scored_total_gap"] = int(
        (summary["frequency"] - summary["n_scored"]).abs().sum()
    )
    if len(mismatch):
        logger.warning(
            "n_scored != frequency in %d cells (total abs gap %d). Likely "
            "dropped contexts (too long / tokenization / empty). See plan 8.",
            len(mismatch), findings["n_scored_total_gap"],
        )
    else:
        logger.info("n_scored matches frequency for every cell.")

    return findings


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Merge hit matrix month shards (hit_matrix_plan.md 7/8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shard-dir", required=True,
                        help="Directory with the *.part.json month shards.")
    parser.add_argument("--freq-dir", required=True,
                        help="Frequency matrix dir (denominator + axes).")
    parser.add_argument("--freq-stem", default="freq")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="hits")
    parser.add_argument("--version", default="unversioned")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    freq = load_freq(args.freq_dir, args.freq_stem)

    shard_files = find_shards(args.shard_dir)
    logger.info("Found %d shard(s) in %s", len(shard_files), args.shard_dir)
    (cells, drop_counts, ctx_drops, scopes, dea_metas,
     target_sels) = merge_shards(shard_files)
    if not cells:
        logger.error("No cells after merging shards. Check the map outputs.")
        raise SystemExit(1)

    # Target-selection agreement: every month must have been scored at the
    # same pre-scoring floor, or the scored subset (and thus which words have
    # cells) differs across months.
    min_freqs = {s.get("min_freq") for s in target_sels if s}
    if len(min_freqs) > 1:
        raise SystemExit(
            f"Shards disagree on the --min-freq floor ({sorted(min_freqs)}). "
            f"All months must be scored with the same floor."
        )
    target_selection = (
        max(target_sels, key=lambda s: s.get("n_total", 0))
        if target_sels else None
    )

    # Scope agreement across shards (all months must share one scope).
    uniq_scopes = {json.dumps(s, sort_keys=True) for s in scopes}
    if len(uniq_scopes) > 1:
        raise SystemExit(
            f"Shards disagree on scope ({len(uniq_scopes)} distinct). All "
            f"months must be scored with the same subreddit scope."
        )

    summary = build_summary(cells, freq)
    summary_path = os.path.join(args.out_dir, f"{args.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote summary %s (%d rows)", summary_path, len(summary))

    # Reconstruct the cumulative causal eligibility mask that hit_matrix.py
    # applied, so non-eligible (not-yet-emerged) cells stay NaN rather than
    # being zero-filled in the hits_k matrices. min_freq comes from the shard
    # metadata (all shards agree, checked above); default to 1 (no floor) if a
    # shard predates the field.
    merge_min_freq = (target_selection or {}).get("min_freq", 1)
    eligible = eligibility_mask(freq, merge_min_freq)
    logger.info(
        "Eligibility mask at min_freq=%s (cumulative causal): %d of %d cells "
        "eligible.",
        merge_min_freq, int(eligible.to_numpy().sum()), eligible.size,
    )

    save_matrices(args.out_dir, args.stem, summary, freq, eligible)

    findings = run_sanity_checks(summary, freq)

    meta = {
        "script": "merge_hit_matrix.py",
        "version": args.version,
        "ks": KS,
        # Pre-scoring cumulative-causal cost/noise floor used by hit_matrix.py
        # (--min-freq) and the resulting scored-word counts. The merge no
        # longer computes rates, so there is no reliability (min_contexts)
        # mask here; pick that downstream when computing hits_k / frequency.
        "target_selection": target_selection,
        "n_shards_merged": len(shard_files),
        "scope": scopes[0] if scopes else None,
        "dea": dea_metas[0] if dea_metas else None,
        "freq_dir": os.path.abspath(args.freq_dir),
        "freq_shape": list(freq.shape),
        "record_drop_counts": dict(drop_counts),
        "context_drop_counts": dict(ctx_drops),
        "sanity_checks": findings,
        "files": {
            "summary": f"{args.stem}_summary.csv",
            "hits_k": [f"{args.stem}_hits_{k}.npy" for k in KS],
            "mean_log_rank": f"{args.stem}_mean_log_rank.npy",
            "min_rank": f"{args.stem}_min_rank.npy",
            "mean_soft": f"{args.stem}_mean_soft.npy",
        },
    }
    meta_path = os.path.join(args.out_dir, f"{args.stem}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)

    logger.info("Hit matrix merge done: %d cells over %s, saved to %s",
                len(summary), tuple(freq.shape), args.out_dir)


if __name__ == "__main__":
    main()
