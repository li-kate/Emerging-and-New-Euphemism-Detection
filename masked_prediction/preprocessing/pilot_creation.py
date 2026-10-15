"""
Select the pilot (control) word set for the hit matrix script.

See ``hit_matrix_plan.md`` section 5b. The pilot set is a small, labeled
control group of known positives (DEA drug euphemisms) and frequency-matched
negatives. The hit matrix script keeps *per-context* rows for pilot words only
(everything else is aggregated on the fly), and section 6 uses the pilot set to
choose ``k``. A separate held-out set of positives is reserved for final
detector evaluation and must stay OUT of the pilot list.

This script turns the plan's automated steps into a reproducible run:

  1. Load the frequency matrix (``freq.npy`` + ``freq_words.csv`` +
     ``freq_half_years.csv``) built by ``freq_matrix.py`` / ``merge_freq_matrix.py``.
  2. Load the DEA term list (one term per line).
  3. Eligibility filter: a word must reach ``--min-freq`` occurrences in at
     least ``--min-periods`` half-years so hit rates are stable.
  4. Seeded split of eligible positives into pilot vs. held-out.
  5. Frequency-matched negatives: for each pilot positive, pick a non-DEA word
     whose log-frequency profile over half-years is closest (sampled from the
     nearest ``--neg-top``), never reusing a word.
  6. Write ``pilot_words.json`` recording the seed, thresholds, and the final
     pilot positives, pilot negatives, and held-out positives, so the pilot
     set is reproducible and consumable by the hit matrix script.

The manual steps from the plan (labeling positives ambiguous vs. drug-sense,
screening negatives, adding hand-picked hard negatives) are left to a human.
This script writes editable fields so those labels can be filled in by hand:
each positive carries a ``"sense"`` field defaulting to ``"unreviewed"``, and
an empty ``hard_negatives`` list is included for hand-picked additions.

Example:

    python pilot_creation.py \\
        --freq-dir freq_drug_subreddits \\
        --dea-file ../../data_collection/word_lists/drug_synonyms.txt \\
        --out pilot_words.json \\
        --seed 0 \\
        --min-freq 50 --min-periods 8 \\
        --n-pilot 30
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# LOADERS (single functions so the source can change — see plan section 2)
# ============================================================================


def load_frequency_matrix(freq_dir, stem="freq"):
    """
    Load the frequency matrix saved by ``freq_matrix.py`` into a DataFrame.

    Reads ``<stem>.npy`` (shape ``[half_years, words]``), ``<stem>_half_years.csv``
    (one ``half_year`` column) and ``<stem>_words.csv`` (one ``word`` column).

    Returns a ``pandas.DataFrame`` indexed by half-year, columns = words, so the
    plan's ``freq.columns`` / ``freq[w]`` / ``(freq >= MIN_FREQ).sum(axis=0)``
    expressions all work directly.
    """
    matrix_path = os.path.join(freq_dir, f"{stem}.npy")
    hy_path = os.path.join(freq_dir, f"{stem}_half_years.csv")
    words_path = os.path.join(freq_dir, f"{stem}_words.csv")

    for pth in (matrix_path, hy_path, words_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(
                f"Expected frequency output not found: {pth}. Did freq_matrix.py "
                f"/ merge_freq_matrix.py run with --out-dir {freq_dir} "
                f"--stem {stem}?"
            )

    freq = np.load(matrix_path)
    half_years = pd.read_csv(hy_path)["half_year"].astype(str).tolist()
    words = pd.read_csv(words_path)["word"].astype(str).tolist()

    if freq.shape != (len(half_years), len(words)):
        raise ValueError(
            f"Matrix shape {freq.shape} does not match labels "
            f"({len(half_years)} half-years x {len(words)} words). "
            f"Mismatched or stale label files?"
        )

    df = pd.DataFrame(freq, index=half_years, columns=words)
    logger.info(
        "Loaded frequency matrix: %d half-years x %d words from %s",
        len(half_years),
        len(words),
        freq_dir,
    )
    return df


def load_dea_terms(dea_file):
    """
    Load DEA terms (one per line), lowercased.

    Blank lines and ``#`` comments are ignored, mirroring the other word-list
    loaders in the repo. Returns a set of lowercased terms.
    """
    terms = set()
    with open(dea_file, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            terms.add(s.lower())
    logger.info("Loaded %d DEA terms from %s", len(terms), dea_file)
    return terms


# ============================================================================
# SELECTION (plan section 5b, automated steps)
# ============================================================================


def eligible_mask(freq, min_freq, min_periods):
    """
    Boolean Series over words: reaches ``min_freq`` in >= ``min_periods``
    half-years. Index aligns with ``freq.columns``.
    """
    return (freq >= min_freq).sum(axis=0) >= min_periods


def split_positives(eligible_pos, n_pilot, rng):
    """Seeded split of eligible positives into (pilot, held-out)."""
    shuffled = list(rng.permutation(np.array(eligible_pos, dtype=object)))
    return shuffled[:n_pilot], shuffled[n_pilot:]


def pick_negatives(positives, pool, logf, rng, per_pos=1, top=20):
    """
    Frequency-matched negatives.

    For each positive, compute the Euclidean distance between its
    log-frequency profile over half-years and every available pool word's,
    take the ``top`` nearest, and sample ``per_pos`` of them (so a tie or
    near-tie does not always resolve to the same word). A chosen word is never
    reused for a later positive.

    ``logf`` is a DataFrame (half-years x words) of ``log1p(freq)``. Returns a
    list of negative words aligned 1:1 with ``positives`` when ``per_pos == 1``.
    """
    chosen = []
    avail = list(pool)
    for w in positives:
        if not avail:
            logger.warning(
                "Negative pool exhausted before matching all positives; "
                "stopping at %d negatives.",
                len(chosen),
            )
            break
        target = logf[w].to_numpy()
        X = logf[avail].to_numpy().T  # [n_avail, n_half_years]
        d = np.linalg.norm(X - target, axis=1)
        n_near = min(top, len(avail))
        near = [avail[i] for i in np.argsort(d)[:n_near]]
        k = min(per_pos, len(near))
        picks = list(rng.choice(near, size=k, replace=False))
        chosen += picks
        picked = set(picks)
        avail = [a for a in avail if a not in picked]
    return chosen


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Select the reproducible pilot word set (hit_matrix_plan.md 5b).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--freq-dir",
        required=True,
        help="Directory holding the frequency outputs (freq.npy, "
        "freq_words.csv, freq_half_years.csv).",
    )
    parser.add_argument(
        "--stem",
        default="freq",
        help="Filename stem used by freq_matrix.py for its outputs.",
    )
    parser.add_argument(
        "--dea-file",
        required=True,
        help="DEA term list (one term per line). Positives are the "
        "intersection of this list with the matrix columns.",
    )
    parser.add_argument(
        "--out",
        default="pilot_words.json",
        help="Output config path.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-freq",
        type=int,
        default=50,
        help="Eligibility: a word must reach this count in >= --min-periods "
        "half-years (plan's pilot MIN_FREQ; distinct from the frequency "
        "postfilter threshold).",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=8,
        help="Eligibility: number of half-years in which --min-freq must be met.",
    )
    parser.add_argument(
        "--n-pilot",
        type=int,
        default=30,
        help="Number of positives placed in the pilot set; the rest are "
        "held out for final evaluation.",
    )
    parser.add_argument(
        "--neg-per-pos",
        type=int,
        default=1,
        help="Negatives to draw per pilot positive.",
    )
    parser.add_argument(
        "--neg-top",
        type=int,
        default=20,
        help="Sample each negative from this many nearest log-frequency "
        "matches (instead of always the single closest).",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    freq = load_frequency_matrix(args.freq_dir, args.stem)
    dea = load_dea_terms(args.dea_file)

    # Eligibility.
    ok = eligible_mask(freq, args.min_freq, args.min_periods)
    cols = list(freq.columns)
    eligible_pos = [w for w in cols if w in dea and bool(ok[w])]
    pool = [w for w in cols if w not in dea and bool(ok[w])]

    logger.info(
        "Eligible positives (DEA & frequent enough): %d; negative pool: %d",
        len(eligible_pos),
        len(pool),
    )

    dea_in_matrix = sum(1 for w in cols if w in dea)
    logger.info(
        "DEA terms present in matrix columns: %d / %d (rest never appeared or "
        "were filtered out of the matrix).",
        dea_in_matrix,
        len(dea),
    )

    if not eligible_pos:
        logger.error(
            "No eligible positives. Loosen --min-freq / --min-periods, or "
            "check that the DEA terms match the matrix's tokenization "
            "(lowercase, single word)."
        )
        raise SystemExit(1)
    if len(eligible_pos) < args.n_pilot:
        logger.warning(
            "Only %d eligible positives < --n-pilot %d; all go to the pilot "
            "set and the held-out set will be empty. Consider loosening the "
            "eligibility thresholds.",
            len(eligible_pos),
            args.n_pilot,
        )

    # Split positives (seeded) and match negatives.
    pilot_pos, heldout_pos = split_positives(eligible_pos, args.n_pilot, rng)

    logf = np.log1p(freq)
    pilot_neg = pick_negatives(
        pilot_pos, pool, logf, rng, per_pos=args.neg_per_pos, top=args.neg_top
    )

    logger.info(
        "Pilot positives: %d | pilot negatives: %d | held-out positives: %d",
        len(pilot_pos),
        len(pilot_neg),
        len(heldout_pos),
    )

    # Build the config. Positives carry an editable manual-review field; the
    # plan asks for ambiguous (e.g. "ice", "tea") vs. drug-sense (e.g. "molly")
    # labels, filled in by hand after this runs.
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source": {
            "freq_dir": os.path.abspath(args.freq_dir),
            "stem": args.stem,
            "dea_file": os.path.abspath(args.dea_file),
        },
        "eligibility": {
            "min_freq": args.min_freq,
            "min_periods": args.min_periods,
        },
        "selection": {
            "n_pilot": args.n_pilot,
            "neg_per_pos": args.neg_per_pos,
            "neg_top": args.neg_top,
        },
        "counts": {
            "eligible_positives": len(eligible_pos),
            "negative_pool": len(pool),
            "pilot_positives": len(pilot_pos),
            "pilot_negatives": len(pilot_neg),
            "heldout_positives": len(heldout_pos),
        },
        # Final sets consumed by the hit matrix script. pilot_words = the union
        # of pilot_positives + pilot_negatives (+ any hand-added hard_negatives).
        "pilot_positives": [
            {"word": w, "sense": "unreviewed"} for w in sorted(pilot_pos)
        ],
        "pilot_negatives": sorted(pilot_neg),
        # Hand-picked hard negatives (e.g. "dealer", "overdose"): add by hand,
        # then re-derive pilot_words if you consume this field downstream.
        "hard_negatives": [],
        # Reserved for final detector evaluation — keep OUT of the pilot set.
        "heldout_positives": sorted(heldout_pos),
        # Convenience: the flat set the hit script loads as PILOT_WORDS.
        "pilot_words": sorted(set(pilot_pos) | set(pilot_neg)),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Wrote pilot config to %s", args.out)

    logger.info(
        "Next (manual, plan 5b): label pilot_positives 'sense' as ambiguous vs. "
        "drug-sense, screen pilot_negatives, and optionally fill hard_negatives."
    )


if __name__ == "__main__":
    main()
