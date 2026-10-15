"""
Reduce step for the hit matrix: sum the per-month shards from ``hit_matrix.py``
into the final summary table and matrices described in ``hit_matrix_plan.md``
sections 7 and 8.

``hit_matrix.py`` (one SLURM array task per month) writes one ``*.part.json``
shard holding running per (half_year, word) counts plus per-context pilot rows.
This script sums the cells across all months, concatenates the pilot rows,
derives the rate/mean columns, and writes:

  * ``<stem>_summary.csv`` — one row per (half_year, word): ``frequency``,
    ``n_scored``, ``hits_k``/``hit_rate_k`` for each k, ``min_rank``,
    ``mean_log_rank``, ``mean_soft``.
  * ``<stem>_hits_k.npy`` and ``<stem>_rate_k.npy`` — matrices
    ``[half_years x words]`` for each k, plus ``mean_log_rank`` / ``min_rank`` /
    ``mean_soft`` matrices, all aligned EXACTLY to the frequency matrix (same
    half-years, same words, same order).
  * ``<stem>_pilot.csv`` — the per-context pilot table.
  * ``<stem>_meta.json`` — scope, DEA, k list, and the sanity-check results.

Rates divide by the external frequency matrix (the denominator). Cells with
frequency below ``--min-contexts`` are reported as ``NaN``, not 0 (plan 8).

Example:

    python merge_hit_matrix.py \\
        --shard-dir hits_drug_subreddits/shards \\
        --freq-dir freq_drug_subreddits \\
        --out-dir hits_drug_subreddits \\
        --min-contexts 30
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
from hit_matrix import KS, SHARD_SUFFIX

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
      pilot_rows  - concatenated list of per-context pilot rows
      drop_counts - summed record drop reasons
      ctx_drops   - summed context-construction drop reasons
      scopes      - list of per-shard scope dicts (checked for agreement)
      dea_metas   - list of per-shard DEA metadata dicts
    """
    cells = defaultdict(_empty_cell)
    pilot_rows = []
    drop_counts = defaultdict(int)
    ctx_drops = defaultdict(int)
    scopes, dea_metas = [], []

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

        pilot_rows.extend(shard.get("pilot_rows", []))

        for reason, c in shard.get("drop_counts", {}).items():
            drop_counts[reason] += c
        for reason, c in shard.get("dropped_contexts", {}).items():
            ctx_drops[reason] += c
        if "scope" in shard:
            scopes.append(shard["scope"])
        if "dea" in shard:
            dea_metas.append(shard["dea"])

        logger.info("Merged shard %d/%d: %s (running cells: %s)",
                    i, len(shard_files), os.path.basename(path),
                    f"{len(cells):,}")

    return cells, pilot_rows, drop_counts, ctx_drops, scopes, dea_metas


# ============================================================================
# SUMMARY TABLE
# ============================================================================


def build_summary(cells, freq, min_contexts):
    """
    Build the per (half_year, word) summary DataFrame (plan section 7).

    ``frequency`` comes from the external frequency matrix. Rates are
    ``hits_k / frequency`` but reported as NaN where frequency < ``min_contexts``
    (plan 8: do not report a rate on too few contexts). ``mean_log_rank`` and
    ``mean_soft`` divide by ``n_scored``.
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
        enough = frequency >= min_contexts
        for k in KS:
            hits = cell[f"hits_{k}"]
            row[f"hits_{k}"] = hits
            row[f"hit_rate_{k}"] = (hits / frequency) if (frequency and enough) \
                else math.nan
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


def save_matrices(out_dir, stem, summary, freq):
    """Write the hit/rate/diagnostic matrices aligned to the frequency matrix."""
    def save(name, df):
        path = os.path.join(out_dir, f"{stem}_{name}.npy")
        np.save(path, df.to_numpy(dtype=np.float64))
        logger.info("Saved %s %s", path, df.shape)

    for k in KS:
        save(f"hits_{k}", pivot_aligned(summary, f"hits_{k}", freq).fillna(0))
        save(f"rate_{k}", pivot_aligned(summary, f"hit_rate_{k}", freq))
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
    parser.add_argument("--min-contexts", type=int, default=30,
                        help="Mask (NaN) rates for cells with frequency below "
                        "this (plan 8).")
    parser.add_argument("--version", default="unversioned")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    freq = load_freq(args.freq_dir, args.freq_stem)

    shard_files = find_shards(args.shard_dir)
    logger.info("Found %d shard(s) in %s", len(shard_files), args.shard_dir)
    cells, pilot_rows, drop_counts, ctx_drops, scopes, dea_metas = merge_shards(
        shard_files
    )
    if not cells:
        logger.error("No cells after merging shards. Check the map outputs.")
        raise SystemExit(1)

    # Scope agreement across shards (all months must share one scope).
    uniq_scopes = {json.dumps(s, sort_keys=True) for s in scopes}
    if len(uniq_scopes) > 1:
        raise SystemExit(
            f"Shards disagree on scope ({len(uniq_scopes)} distinct). All "
            f"months must be scored with the same subreddit scope."
        )

    summary = build_summary(cells, freq, args.min_contexts)
    summary_path = os.path.join(args.out_dir, f"{args.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote summary %s (%d rows)", summary_path, len(summary))

    save_matrices(args.out_dir, args.stem, summary, freq)

    # Pilot table.
    pilot_df = pd.DataFrame(
        pilot_rows,
        columns=["word", "half_year", "context_id", "best_rank", "soft_score"],
    )
    pilot_path = os.path.join(args.out_dir, f"{args.stem}_pilot.csv")
    pilot_df.to_csv(pilot_path, index=False)
    logger.info("Wrote pilot table %s (%d rows)", pilot_path, len(pilot_df))

    findings = run_sanity_checks(summary, freq)

    meta = {
        "script": "merge_hit_matrix.py",
        "version": args.version,
        "ks": KS,
        "min_contexts": args.min_contexts,
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
            "pilot": f"{args.stem}_pilot.csv",
            "hits_k": [f"{args.stem}_hits_{k}.npy" for k in KS],
            "rate_k": [f"{args.stem}_rate_{k}.npy" for k in KS],
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
