"""
Pilot experiment (reduce step): concatenate the per-month pilot shards from
``pilot_hits.py`` into one per-context pilot table.

``pilot_hits.py`` (one SLURM array task per month) writes one ``*.pilot.json``
shard holding the per-context rows ``(word, half_year, context_id, best_rank,
soft_score)`` for the pilot/control words. This script concatenates them into
``<stem>_pilot.csv`` and writes a small ``<stem>_pilot_meta.json``.

The ``context_id`` is per-month (each map task numbers its own contexts from
0), so it is only unique within a (month, word). That matches the previous
behaviour, where pilot rows were likewise numbered per shard. Downstream uses
the pilot table for descriptive statistics on the control set (bootstrap CIs,
other k, median) and for choosing k (``hit_matrix_plan.md`` section 6), none of
which need a globally unique context id.

Example:

    python merge_pilot_hits.py \\
        --shard-dir pilot_drug_subreddits/shards \\
        --out-dir pilot_drug_subreddits
"""

import argparse
import glob
import json
import logging
import os
from collections import defaultdict

import pandas as pd

from pilot_hits import PILOT_SHARD_SUFFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_shards(shard_dir):
    shards = sorted(glob.glob(os.path.join(shard_dir, f"*{PILOT_SHARD_SUFFIX}")))
    if not shards:
        raise FileNotFoundError(
            f"No pilot shards matching *{PILOT_SHARD_SUFFIX} in {shard_dir}. "
            f"Did the pilot_hits.py month array run and finish?"
        )
    return shards


def merge_shards(shard_files):
    """
    Concatenate all month pilot shards.

    Returns ``(pilot_rows, drop_counts, ctx_drops, scopes, dea_metas,
    target_sels)``.
    """
    pilot_rows = []
    drop_counts = defaultdict(int)
    ctx_drops = defaultdict(int)
    scopes, dea_metas, target_sels = [], [], []

    for i, path in enumerate(shard_files, 1):
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)

        pilot_rows.extend(shard.get("pilot_rows", []))

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

        logger.info("Merged pilot shard %d/%d: %s (running rows: %s)",
                    i, len(shard_files), os.path.basename(path),
                    f"{len(pilot_rows):,}")

    return pilot_rows, drop_counts, ctx_drops, scopes, dea_metas, target_sels


def main():
    parser = argparse.ArgumentParser(
        description="Merge pilot per-context shards (hit_matrix_plan.md 5/6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shard-dir", required=True,
                        help="Directory with the *.pilot.json month shards.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="hits")
    parser.add_argument("--version", default="unversioned")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shard_files = find_shards(args.shard_dir)
    logger.info("Found %d pilot shard(s) in %s", len(shard_files), args.shard_dir)
    (pilot_rows, drop_counts, ctx_drops, scopes, dea_metas,
     target_sels) = merge_shards(shard_files)

    # Scope agreement across shards (all months must share one scope).
    uniq_scopes = {json.dumps(s, sort_keys=True) for s in scopes}
    if len(uniq_scopes) > 1:
        raise SystemExit(
            f"Pilot shards disagree on scope ({len(uniq_scopes)} distinct). "
            f"All months must be scored with the same subreddit scope."
        )

    # min_freq agreement (so the half-years scored per word are consistent).
    min_freqs = {s.get("min_freq") for s in target_sels if s}
    if len(min_freqs) > 1:
        raise SystemExit(
            f"Pilot shards disagree on the --min-freq floor "
            f"({sorted(min_freqs)}). All months must use the same floor."
        )

    pilot_df = pd.DataFrame(
        pilot_rows,
        columns=["word", "half_year", "context_id", "best_rank", "soft_score"],
    )
    pilot_path = os.path.join(args.out_dir, f"{args.stem}_pilot.csv")
    pilot_df.to_csv(pilot_path, index=False)
    logger.info("Wrote pilot table %s (%d rows)", pilot_path, len(pilot_df))

    meta = {
        "script": "merge_pilot_hits.py",
        "version": args.version,
        "n_shards_merged": len(shard_files),
        "n_rows": len(pilot_df),
        "n_pilot_words": int(pilot_df["word"].nunique()) if len(pilot_df) else 0,
        "scope": scopes[0] if scopes else None,
        "dea": dea_metas[0] if dea_metas else None,
        "target_selection": (
            max(target_sels, key=lambda s: s.get("n_total", 0))
            if target_sels else None
        ),
        "record_drop_counts": dict(drop_counts),
        "context_drop_counts": dict(ctx_drops),
        "files": {"pilot": f"{args.stem}_pilot.csv"},
    }
    meta_path = os.path.join(args.out_dir, f"{args.stem}_pilot_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)

    logger.info("Pilot merge done: %d rows over %d word(s), saved to %s",
                len(pilot_df),
                int(pilot_df["word"].nunique()) if len(pilot_df) else 0,
                args.out_dir)


if __name__ == "__main__":
    main()
