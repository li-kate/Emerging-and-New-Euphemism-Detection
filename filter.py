# Prefilter
# < 4 characters
# > 1 word
# get rid of stop words

# Postfilter
# Frequency < 30 in all half years in created matrix

# and in the logs, count how many words were filtered out for each filter


import re

import numpy as np
from nltk.corpus import stopwords

# PREFILTER SETTINGS

# Minimum number of characters allowed in a term.
MIN_TERM_LENGTH = 4
# Maximum number of words allowed in a term.
MAX_WORDS = 1

# STOP WORDS

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    # The NLTK stopwords corpus has not been downloaded yet.
    import nltk

    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


def prefilter(
    words,
    min_term_length=MIN_TERM_LENGTH,
    max_words=MAX_WORDS,
):
    """
    Apply universal prefilters to an iterable of terms.

    Filter rules (in order):
      1. Removes terms with fewer than min_term_length characters.
      2. Removes terms containing more than max_words words.
      3. Removes terms that are stop words.

    Args:
      words            - iterable of terms to filter.
      min_term_length  - minimum number of characters a term must have
                         (defaults to MIN_TERM_LENGTH).
      max_words        - maximum number of words a term may contain
                         (defaults to MAX_WORDS).

    Returns:
      kept    - list of terms that passed every filter.
      counts  - dict mapping each filter name to the number of terms
                it removed (counted against the first filter a term
                failed, so the counts sum to the number of dropped terms).
    """
    kept = []

    counts = {
        "short_terms": 0,
        "multi_word_terms": 0,
        "stop_words": 0,
    }

    for word in words:
        if word is None:
            counts["short_terms"] += 1
            continue

        stripped = word.strip()

        # 1. Reject terms that are too short (ignoring whitespace).
        character_count = len(
            re.sub(r"\s+", "", stripped)
        )
        if character_count < min_term_length:
            counts["short_terms"] += 1
            continue

        # 2. Reject terms containing more than max_words words.
        word_parts = re.findall(r"\S+", stripped)
        if len(word_parts) > max_words:
            counts["multi_word_terms"] += 1
            continue

        # 3. Reject stop words.
        if stripped.lower() in STOP_WORDS:
            counts["stop_words"] += 1
            continue

        kept.append(word)

    return kept, counts


def log_prefilter_counts(
    counts,
    total,
    min_term_length=MIN_TERM_LENGTH,
    max_words=MAX_WORDS,
    samples=None,
):
    """
    Print how many terms each prefilter removed.

    If ``samples`` is given (a dict mapping each bucket name to a list of
    example removed terms), a sample of removed words is printed under each
    bucket. Samples are examples only, not an exhaustive list.
    """
    removed = sum(counts.values())

    print("\n========================================")
    print("PREFILTER")
    print("========================================")

    print(
        f"Input terms:                   "
        f"{total:,}"
    )
    print(
        f"Removed <{min_term_length}-character terms:    "
        f"{counts['short_terms']:,}"
    )
    print(
        f"Removed >{max_words}-word terms:         "
        f"{counts['multi_word_terms']:,}"
    )
    print(
        f"Removed stop words:            "
        f"{counts['stop_words']:,}"
    )
    print(
        f"Total removed:                 "
        f"{removed:,}"
    )
    print(
        f"Kept terms:                    "
        f"{total - removed:,}"
    )

    if samples:
        _print_removed_samples(
            samples,
            labels={
                "short_terms": f"<{min_term_length}-character terms",
                "multi_word_terms": f">{max_words}-word terms",
                "stop_words": "stop words",
            },
        )


def _print_removed_samples(samples, labels):
    """Print a sample of removed terms for each bucket (examples only)."""
    print("\n--- sample of removed terms (examples only) ---")
    for key, label in labels.items():
        sample = samples.get(key) or []
        if not sample:
            continue
        print(f"{label} ({len(sample)} shown):")
        print("  " + ", ".join(sample))

# POSTFILTER SETTINGS

# Minimum frequency a word must reach in at least one half-year to be kept.
MIN_FREQ = 30


def postfilter(freq, words, min_freq=MIN_FREQ, sample_removed=0, rng=None):
    """
    Apply the postfilter to a built frequency matrix.

    A word is kept if its count reaches ``min_freq`` in at least one
    half-year (i.e. removed only if it is below ``min_freq`` in *every*
    half-year). Keeping words that are frequent in only a few half-years
    is deliberate: newly emerging words may only be frequent in recent
    half-years.

    Args:
      freq            - 2D array of shape [half_years, words] with occurrence
                        counts.
      words           - sequence of word labels in column order; must line up
                        with ``freq``'s columns.
      min_freq        - minimum count a word must reach in at least one
                        half-year (defaults to MIN_FREQ).
      sample_removed  - if > 0, also return a uniform random sample of up to
                        this many removed word labels.
      rng             - optional ``random.Random`` for a reproducible sample.

    Returns:
      freq_kept   - ``freq`` with only the kept columns.
      kept_words  - list of the kept word labels, in the same order.
      counts      - dict with the number of words removed under the
                    "low_frequency" key.
      samples     - list of up to ``sample_removed`` removed word labels
                    (empty when ``sample_removed == 0``).
    """
    import random as _random

    keep = freq.max(axis=0) >= min_freq
    kept_words = [w for w, k in zip(words, keep) if k]
    counts = {"low_frequency": int((~keep).sum())}

    samples = []
    if sample_removed > 0:
        removed_words = [w for w, k in zip(words, keep) if not k]
        r = rng or _random.Random()
        if len(removed_words) <= sample_removed:
            samples = list(removed_words)
        else:
            samples = r.sample(removed_words, sample_removed)

    return freq[:, keep], kept_words, counts, samples


def log_postfilter_counts(counts, total, min_freq=MIN_FREQ, samples=None):
    """
    Print how many words the postfilter removed.

    If ``samples`` is a list of example removed words, it is printed below the
    counts. Samples are examples only, not an exhaustive list.
    """
    removed = counts["low_frequency"]

    print("\n========================================")
    print("POSTFILTER")
    print("========================================")

    print(
        f"Input terms:                      "
        f"{total:,}"
    )
    print(
        f"Removed freq <{min_freq} in all half-years: "
        f"{removed:,}"
    )
    print(
        f"Kept terms:                       "
        f"{total - removed:,}"
    )

    if samples:
        print(f"\n--- sample of removed terms (examples only, {len(samples)} "
              f"shown) ---")
        print("  " + ", ".join(samples))


def postfilter_per_half_year(
    freq, words, min_freq=MIN_FREQ, sample_removed=0, rng=None
):
    """
    Apply the postfilter independently within each half-year (leakage-safe).

    Unlike :func:`postfilter`, which makes one keep/drop decision per word
    using the maximum count across *all* half-years, this variant applies the
    threshold to every cell on its own:

      * A cell ``freq[h, w]`` below ``min_freq`` is **nulled** (set to 0).
      * A word (column) is dropped only if it was nulled in *every*
        half-year — i.e. it never reached ``min_freq`` in any single
        half-year.

    Because each cell's keep/null decision depends only on that half-year's
    count, the result never carries information from future half-years into
    an earlier one: a word that only becomes frequent in a later half-year
    keeps its later cells but has its earlier (sub-threshold) cells zeroed.

    Args:
      freq            - 2D array of shape [half_years, words] with occurrence
                        counts.
      words           - sequence of word labels in column order; must line up
                        with ``freq``'s columns.
      min_freq        - minimum count a cell must reach in its own half-year
                        to be kept (defaults to MIN_FREQ).
      sample_removed  - if > 0, also return a uniform random sample of up to
                        this many *dropped column* labels (words nulled in
                        every half-year).
      rng             - optional ``random.Random`` for a reproducible sample.

    Returns:
      freq_kept   - the per-cell-nulled matrix restricted to the kept columns.
      kept_words  - list of the kept word labels, in the same order.
      counts      - dict with ``low_frequency`` (number of dropped columns)
                    and ``cells_nulled`` (number of sub-threshold cells zeroed
                    among the *kept* columns).
      samples     - list of up to ``sample_removed`` dropped column labels
                    (empty when ``sample_removed == 0``).
    """
    import random as _random

    # Per-cell keep mask: a cell survives only if its own half-year count
    # reaches the threshold.
    cell_keep = freq >= min_freq

    # Null every sub-threshold cell. Work on a copy so the caller's array is
    # left untouched.
    nulled = np.where(cell_keep, freq, 0)

    # A column survives if at least one of its cells survived.
    col_keep = cell_keep.any(axis=0)
    kept_words = [w for w, k in zip(words, col_keep) if k]

    freq_kept = nulled[:, col_keep]

    # Cells nulled among the kept columns (dropped columns are gone entirely,
    # so they are not counted here).
    cells_nulled = int((~cell_keep[:, col_keep]).sum())
    counts = {
        "low_frequency": int((~col_keep).sum()),
        "cells_nulled": cells_nulled,
    }

    samples = []
    if sample_removed > 0:
        removed_words = [w for w, k in zip(words, col_keep) if not k]
        r = rng or _random.Random()
        if len(removed_words) <= sample_removed:
            samples = list(removed_words)
        else:
            samples = r.sample(removed_words, sample_removed)

    return freq_kept, kept_words, counts, samples


def log_postfilter_per_half_year_counts(
    counts, total, min_freq=MIN_FREQ, samples=None
):
    """
    Print how many words/cells the per-half-year postfilter removed.

    ``counts`` is the dict returned by :func:`postfilter_per_half_year`
    (keys ``low_frequency`` and ``cells_nulled``). If ``samples`` is a list of
    example dropped words, it is printed below the counts. Samples are
    examples only, not an exhaustive list.
    """
    removed = counts["low_frequency"]
    cells_nulled = counts.get("cells_nulled", 0)

    print("\n========================================")
    print("POSTFILTER (per half-year)")
    print("========================================")

    print(
        f"Input terms:                      "
        f"{total:,}"
    )
    print(
        f"Removed freq <{min_freq} in every half-year: "
        f"{removed:,}"
    )
    print(
        f"Cells nulled (kept cols, freq <{min_freq}): "
        f"{cells_nulled:,}"
    )
    print(
        f"Kept terms:                       "
        f"{total - removed:,}"
    )

    if samples:
        print(f"\n--- sample of removed terms (examples only, {len(samples)} "
              f"shown) ---")
        print("  " + ", ".join(samples))
