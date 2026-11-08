"""
Reduce step for the temporal Zhu euphemism detection: sum the per-month shards
from ``zhu_detection_month.py`` into a ranked euphemism-candidate list **per
half-year**.

``zhu_detection_month.py`` (one SLURM array task per month) writes one
``*.part.json`` shard holding, for each half-year the month touched, a
per-token ``score_sum`` plus the masked/informative sentence counts. This
script sums those token scores across every month that shares a half-year and
re-ranks candidates within each half-year, reproducing what Zhu's
``euphemism_detection`` -> ``MLM`` would produce if run on that half-year's
slice of the corpus.

Ranking rule (faithful to ``detection.MLM``): candidates are sorted by total
``MLM_score`` (the summed softmax probability mass they received across the
half-year's informative masked sentences), highest first.

Outputs (one directory, ``--stem`` prefix):

  * ``<stem>_ranked.csv`` — one row per (half_year, rank, word, score), the
    full ranked candidate list for every half-year. This is the temporal table
    you read top candidates off, per period.
  * ``<stem>_top_words.json`` — ``{half_year: [word, ...]}`` giving just the
    ordered candidate words per half-year (the ``top_words`` Zhu's
    identification stage consumes), capped at ``--top-n`` if set.
  * ``<stem>_meta.json`` — scope, keywords, model, Zhu params, per-half-year
    masked/informative counts, number of shards merged, and the agreement
    checks.

Like the hit-matrix merge, this fails only if there are no shards at all; a
single failed month just contributes nothing to its half-years.

Example:

    python merge_zhu_detection.py \\
        --shard-dir zhu_detect_drug_subreddits/shards \\
        --out-dir zhu_detect_drug_subreddits \\
        --stem zhu_detect
"""

import argparse
import glob
import json
import logging
import os
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SHARD_SUFFIX = ".part.json"


# ============================================================================
# SHARD DISCOVERY + MERGE
# ============================================================================


def find_shards(shard_dir):
    shards = sorted(glob.glob(os.path.join(shard_dir, f"*{SHARD_SUFFIX}")))
    if not shards:
        raise FileNotFoundError(
            f"No shards matching *{SHARD_SUFFIX} in {shard_dir}. Did the month "
            f"array run and finish?"
        )
    return shards


def merge_shards(shard_files):
    """
    Sum all month shards into per-half-year token score totals.

    Returns:
      scores       - dict[half_year] -> dict[token] -> summed MLM_score
      hy_counts    - dict[half_year] -> {"n_masked": int, "n_informative": int}
      scopes       - list of per-shard scope dicts (checked for agreement)
      keyword_metas- list of per-shard keyword metadata dicts
      param_metas  - list of per-shard Zhu-param dicts (checked for agreement)
      model_metas  - list of per-shard model dicts
      drop_counts  - summed record-drop reasons across months
    """
    scores = defaultdict(lambda: defaultdict(float))
    hy_counts = defaultdict(lambda: {"n_masked": 0, "n_informative": 0})
    scopes, keyword_metas, param_metas, model_metas = [], [], [], []
    drop_counts = defaultdict(int)

    for i, path in enumerate(shard_files, 1):
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)

        for hy, bucket in shard.get("half_years", {}).items():
            dst = scores[hy]
            for token, s in bucket.get("score_sum", {}).items():
                dst[token] += s
            hy_counts[hy]["n_masked"] += bucket.get("n_masked", 0)
            hy_counts[hy]["n_informative"] += bucket.get("n_informative", 0)

        if "scope" in shard:
            scopes.append(shard["scope"])
        if "keywords" in shard:
            keyword_metas.append(shard["keywords"])
        if "params" in shard:
            param_metas.append(shard["params"])
        if "model" in shard:
            model_metas.append(shard["model"])
        for reason, c in shard.get("record_drop_counts", {}).items():
            drop_counts[reason] += c

        logger.info("Merged shard %d/%d: %s (running half-years: %d)",
                    i, len(shard_files), os.path.basename(path), len(scores))

    return scores, hy_counts, scopes, keyword_metas, param_metas, \
        model_metas, drop_counts


# ============================================================================
# RANKING
# ============================================================================


def rank_half_year(token_scores):
    """
    Rank one half-year's candidates by total MLM_score, highest first
    (``sorted(..., reverse=True)`` on the score, matching ``detection.MLM``).

    Returns a list of ``(word, score)`` tuples.
    """
    return sorted(token_scores.items(), key=lambda kv: kv[1], reverse=True)


def build_ranked_rows(scores, top_n=None):
    """
    Build the flat ranked table rows and the per-half-year ordered word lists.

    Returns ``(rows, top_words)`` where ``rows`` is a list of dicts
    (half_year, rank, word, score) over every half-year in chronological order,
    and ``top_words`` maps ``half_year -> [word, ...]`` (capped at ``top_n``).
    """
    rows = []
    top_words = {}
    for hy in sorted(scores):
        ranked = rank_half_year(scores[hy])
        words = []
        for rank, (word, score) in enumerate(ranked, 1):
            if top_n is None or rank <= top_n:
                rows.append({
                    "half_year": hy,
                    "rank": rank,
                    "word": word,
                    "score": score,
                })
                words.append(word)
        top_words[hy] = words
    return rows, top_words


# ============================================================================
# AGREEMENT CHECKS
# ============================================================================


def check_agreement(scopes, param_metas):
    """
    Confirm every month was scored under one scope and one set of Zhu params,
    so the summed half-years are comparable. Aborts on disagreement, mirroring
    the hit-matrix merge's scope/min_freq checks.
    """
    uniq_scopes = {json.dumps(s, sort_keys=True) for s in scopes}
    if len(uniq_scopes) > 1:
        raise SystemExit(
            f"Shards disagree on scope ({len(uniq_scopes)} distinct). All "
            f"months must be scored with the same subreddit scope."
        )
    uniq_params = {json.dumps(p, sort_keys=True) for p in param_metas}
    if len(uniq_params) > 1:
        raise SystemExit(
            f"Shards disagree on Zhu scoring params ({len(uniq_params)} "
            f"distinct: mlm_k / thres / ms_limit_per_half_year). Re-run all "
            f"months with the same values so the rankings are comparable."
        )


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Merge Zhu detection month shards into per-half-year "
        "ranked candidate lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shard-dir", required=True,
                        help="Directory with the *.part.json month shards.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="zhu_detect")
    parser.add_argument("--top-n", type=int, default=None,
                        help="Keep only the top N candidates per half-year in "
                        "the outputs. Default: keep all.")
    parser.add_argument("--version", default="unversioned")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shard_files = find_shards(args.shard_dir)
    logger.info("Found %d shard(s) in %s", len(shard_files), args.shard_dir)

    (scores, hy_counts, scopes, keyword_metas, param_metas,
     model_metas, drop_counts) = merge_shards(shard_files)
    if not scores:
        logger.error("No candidate scores after merging shards. Check the map "
                     "outputs (were any masked sentences informative?).")
        raise SystemExit(1)

    check_agreement(scopes, param_metas)

    rows, top_words = build_ranked_rows(scores, top_n=args.top_n)

    # Ranked CSV (no pandas dependency needed; write directly).
    import csv

    ranked_path = os.path.join(args.out_dir, f"{args.stem}_ranked.csv")
    with open(ranked_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["half_year", "rank", "word", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote ranked candidates %s (%d rows over %d half-year(s))",
                ranked_path, len(rows), len(top_words))

    top_path = os.path.join(args.out_dir, f"{args.stem}_top_words.json")
    with open(top_path, "w", encoding="utf-8") as f:
        json.dump(top_words, f, indent=2)
    logger.info("Wrote per-half-year top words %s", top_path)

    meta = {
        "script": "merge_zhu_detection.py",
        "version": args.version,
        "n_shards_merged": len(shard_files),
        "n_half_years": len(top_words),
        "half_years": sorted(top_words),
        "top_n": args.top_n,
        "scope": scopes[0] if scopes else None,
        "keywords": keyword_metas[0] if keyword_metas else None,
        "model": model_metas[0] if model_metas else None,
        "params": param_metas[0] if param_metas else None,
        "half_year_counts": {
            hy: hy_counts[hy] for hy in sorted(hy_counts)
        },
        "n_candidates_per_half_year": {
            hy: len(scores[hy]) for hy in sorted(scores)
        },
        "record_drop_counts": dict(drop_counts),
        "files": {
            "ranked": f"{args.stem}_ranked.csv",
            "top_words": f"{args.stem}_top_words.json",
        },
    }
    meta_path = os.path.join(args.out_dir, f"{args.stem}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)

    logger.info("Zhu detection merge done: %d half-year(s), saved to %s",
                len(top_words), args.out_dir)


if __name__ == "__main__":
    main()
