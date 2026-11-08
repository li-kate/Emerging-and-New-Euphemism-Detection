"""
Context-sample experiment (reduce step): union the per-month word-sample shards
from ``sample_contexts.py`` into one sample per half-year.

``sample_contexts.py`` (one SLURM array task per month) writes one
``*.sample.json`` shard holding a uniform reservoir of up to ``k`` contexts for
the chosen word per half-year, along with ``n_seen`` (how many contexts that
month's half-year actually had). A half-year is split across the month shards
(e.g. 2015H1 = Jan-Jun 2015), so the shards must be combined to get one sample
for the whole half-year. The union is written to ``<stem>_sample.json``.

Example:

    python merge_sample_contexts.py \\
        --shard-dir sample_drug_subreddits/shards \\
        --out-dir sample_drug_subreddits
"""

import argparse
import glob
import json
import logging
import os
import random
from collections import defaultdict

from sample_contexts import SAMPLE_SHARD_SUFFIX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_shards(shard_dir):
    shards = sorted(glob.glob(os.path.join(shard_dir, f"*{SAMPLE_SHARD_SUFFIX}")))
    if not shards:
        raise FileNotFoundError(
            f"No sample shards matching *{SAMPLE_SHARD_SUFFIX} in {shard_dir}. "
            f"Did the sample_contexts.py month array run and finish?"
        )
    return shards


def merge_word_sample(shard_files, seed=0):
    """
    Union each shard's per-half-year ``word_sample`` into one sample per
    half-year, spanning all months.

    The union is itself a (near) uniform size-``k`` sample of the pooled
    contexts. A shard's reservoir is a uniform sample of that shard's
    ``n_seen`` contexts, so each of its ``r`` sampled items stands in for
    ``n_i / r`` of the pooled stream. To draw ``k`` items from the pool so that
    each original context has roughly equal probability, every candidate item
    is given weight ``w = n_i / r`` and we take a weighted random sample of
    size ``k`` without replacement across all candidates for the half-year.

    This is done with Efraimidis-Spirakis A-Res weighted reservoir sampling:
    assign each item the key ``u ** (1 / w)`` with ``u`` uniform in (0, 1), and
    keep the ``k`` items with the largest keys. A shard contributing ``n_i`` of
    the ``N = sum n_i`` total contexts then wins about ``n_i / N`` of the
    slots. It is exact when every shard's reservoir is unsaturated
    (``n_i <= k``, so no information was lost at the map step) and an unbiased
    approximation otherwise — the standard limit of merging fixed-size
    reservoirs without the discarded items.

    Returns ``None`` if no shard carried a ``word_sample``. Otherwise returns
    ``{"word", "k", "seed", "samples": {hy: [records]}, "n_seen": {hy: total}}``.
    """
    rng = random.Random(seed)

    word = None
    k = None
    # Per-half-year list of (key, record): key is the A-Res weighted key; we
    # keep the top-k keys at the end.
    candidates = defaultdict(list)
    n_seen_total = defaultdict(int)
    any_sample = False

    for path in shard_files:
        with open(path, encoding="utf-8") as f:
            shard = json.load(f)
        ws = shard.get("word_sample")
        if not ws:
            continue
        any_sample = True

        if word is None:
            word = ws.get("word")
            k = ws.get("k")
        else:
            if ws.get("word") != word:
                raise SystemExit(
                    f"Shards disagree on the sampled word "
                    f"({word!r} vs {ws.get('word')!r} in {path}). All months "
                    f"must sample the same word."
                )
            if ws.get("k") != k:
                raise SystemExit(
                    f"Shards disagree on --sample-size (k={k} vs "
                    f"{ws.get('k')} in {path}). All months must use the same "
                    f"sample size."
                )

        shard_seen = ws.get("n_seen", {})
        for hy, records in ws.get("samples", {}).items():
            if not records:
                continue
            n_i = int(shard_seen.get(hy, len(records)))
            n_seen_total[hy] += n_i
            # Each sampled item represents n_i / r of the shard's contexts.
            w = n_i / len(records)
            for rec in records:
                # A-Res key; larger weight -> key closer to 1 in expectation.
                key = rng.random() ** (1.0 / w) if w > 0 else 0.0
                candidates[hy].append((key, rec))

    if not any_sample:
        return None

    samples = {}
    for hy, items in candidates.items():
        # Keep the k items with the largest keys (weighted sample w/o replacement).
        items.sort(key=lambda t: t[0], reverse=True)
        samples[hy] = [rec for _key, rec in items[:k]]

    return {
        "word": word,
        "k": k,
        "seed": seed,
        "samples": samples,
        "n_seen": dict(n_seen_total),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-month word-sample shards (hit_matrix_plan.md 8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shard-dir", required=True,
                        help="Directory with the *.sample.json month shards.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stem", default="hits")
    parser.add_argument(
        "--sample-seed", type=int, default=0,
        help="Seed for merging the per-half-year reservoirs across months.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    shard_files = find_shards(args.shard_dir)
    logger.info("Found %d sample shard(s) in %s", len(shard_files), args.shard_dir)

    word_sample = merge_word_sample(shard_files, seed=args.sample_seed)
    if word_sample is None:
        logger.error(
            "No word_sample found in any shard. Did sample_contexts.py write "
            "samples (the word may not have been eligible in any month)?"
        )
        raise SystemExit(1)

    sample_path = os.path.join(args.out_dir, f"{args.stem}_sample.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(word_sample, f, indent=2, ensure_ascii=False)
    n_kept = sum(len(v) for v in word_sample["samples"].values())
    logger.info(
        "Wrote word sample %s: word=%r, %d contexts across %d half-year(s).",
        sample_path, word_sample["word"], n_kept, len(word_sample["samples"]),
    )


if __name__ == "__main__":
    main()
