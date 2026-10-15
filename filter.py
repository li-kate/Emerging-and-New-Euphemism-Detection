# Prefilter
# < 4 characters
# > 1 word
# get rid of stop words

# Postfilter
# Frequency < 30 in all half years in created matrix

# and in the logs, count how many words were filtered out for each filter


import re

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
):
    """Print how many terms each prefilter removed."""
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

# POSTFILTER SETTINGS

# Minimum frequency a word must reach in at least one half-year to be kept.
MIN_FREQ = 30


def postfilter(freq, words, min_freq=MIN_FREQ):
    """
    Apply the postfilter to a built frequency matrix.

    A word is kept if its count reaches ``min_freq`` in at least one
    half-year (i.e. removed only if it is below ``min_freq`` in *every*
    half-year). Keeping words that are frequent in only a few half-years
    is deliberate: newly emerging words may only be frequent in recent
    half-years.

    Args:
      freq      - 2D array of shape [half_years, words] with occurrence
                  counts.
      words     - sequence of word labels in column order; must line up
                  with ``freq``'s columns.
      min_freq  - minimum count a word must reach in at least one
                  half-year (defaults to MIN_FREQ).

    Returns:
      freq_kept   - ``freq`` with only the kept columns.
      kept_words  - list of the kept word labels, in the same order.
      counts      - dict with the number of words removed under the
                    "low_frequency" key.
    """
    keep = freq.max(axis=0) >= min_freq
    kept_words = [w for w, k in zip(words, keep) if k]
    counts = {"low_frequency": int((~keep).sum())}
    return freq[:, keep], kept_words, counts


def log_postfilter_counts(counts, total, min_freq=MIN_FREQ):
    """Print how many words the postfilter removed."""
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
