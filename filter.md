# Plan: Building, Filtering, and Saving the Frequency Matrix

## 1. Purpose

The frequency matrix counts how many times each word occurs in each half-year of the filtered Reddit corpus. It serves two roles:

1. **Target word list:** its columns (after filtering) are the final target words for the hit matrix script.
2. **Denominator:** `hit_rate_k = hits_k / frequency`.

This plan covers filtering, saving, loading, and keeping it consistent with the hit matrix outputs.

## 2. Contents

- **Axes:** rows are half-years (e.g. `2012H1`, about 40 of them), columns are words (about 1M before filtering; fewer after).
- **Cell value:** number of occurrences of that word in that half-year (one occurrence = one context in the hit script).
- **Filtered:** words removed by the prefilter or postfilter below do not appear in the matrix, so they are never target words.

## 2b. Corpus scope: subreddit list or all of Reddit

The script has one setting, `SUBREDDIT_FILE`:

- `SUBREDDIT_FILE = "subreddits.txt"`: count only text from the listed subreddits.
- `SUBREDDIT_FILE = None`: count the whole corpus.

**File format:** one subreddit per line. Names are normalized (lowercased, `r/` or `/r/` prefix stripped, whitespace trimmed); blank lines and lines starting with `#` are ignored. Reddit subreddit names are case-insensitive, so match on the lowercase form.

```python
import re, hashlib

def load_subreddits(path):
    """Return a set of normalized subreddit names, or None for all of Reddit."""
    if path is None:
        return None
    subs = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):          # skip blanks and comments
                continue
            s = re.sub(r"^/?r/", "", s, flags=re.I).lower()
            subs.add(s)
    return subs

def subreddit_hash(subs):
    return None if subs is None else hashlib.sha256(
        "\n".join(sorted(subs)).encode()).hexdigest()

# in the corpus reader, before any counting or scoring
def keep(record_subreddit, allowed):
    return allowed is None or record_subreddit.lower() in allowed
```

Rules:

- **The subreddit filter is applied first,** at the corpus reader, before the prefilter, counting, and the postfilter. Every later threshold (including `MIN_FREQ = 30`) then applies to the chosen scope, so a subreddit run will keep far fewer words than an all-Reddit run.
- **One shared function.** The hit script imports the same `load_subreddits` / `keep` code (from a shared module), so both scripts always see the same subset.
- **Separate outputs per scope.** Write outputs to a directory tagged by scope (e.g. `freq_all/` or `freq_subreddits/`, using the file stem), so a subset run never overwrites an all-Reddit run.
- **Log the scope:** number of subreddits listed, number found in the corpus, the listed names with no data (renamed, banned, or private subreddits are often missing from dumps), and the number of records dropped by the filter.
- **Record the scope** in `freq_meta.json` (see section 4), including the hash of the normalized list so the hit script can verify it is using the same one.

## 3. Filtering (`filter.py`)

### Prefilter (applied to the term list before counting)

Rules, applied in order. A term is counted against the **first** filter it fails, so the counts sum to the number of dropped terms.

| # | Rule | Setting | Log key |
|---|---|---|---|
| 1 | Remove terms shorter than `MIN_TERM_LENGTH` characters (whitespace ignored) | `MIN_TERM_LENGTH = 4` | `short_terms` |
| 2 | Remove terms with more than `MAX_WORDS` words | `MAX_WORDS = 1` | `multi_word_terms` |
| 3 | Remove stop words (NLTK English list, case-insensitive) | NLTK `stopwords` | `stop_words` |

`None` terms are counted under `short_terms`.

### Postfilter (applied to the created matrix)

The postfilter has three modes, chosen with `--postfilter-mode` and
`--no-postfilter`. The default is **per-half-year**.

**`per-half-year` (default, leakage-safe).** Apply `MIN_FREQ` to each cell
independently instead of once per word over the whole timeline:

| # | Rule | Setting | Log key |
|---|---|---|---|
| 4a | Null (set to 0) any cell whose count in its own half-year is below `MIN_FREQ` | `MIN_FREQ = 30` | `cells_nulled` |
| 4b | After nulling, drop a word only if it was nulled in **every** half-year (never reached `MIN_FREQ` in any single half-year) | `MIN_FREQ = 30` | `low_frequency` |

Because each cell's keep/null decision uses only that half-year's own count,
the matrix never carries information from a future half-year into an earlier
one. A word that only becomes frequent in a recent half-year keeps its recent
cells but has its earlier, sub-threshold cells zeroed — so a reader of an
early row never learns the word later became common. This is the fix for
frequency leaking information from the future.

```python
MIN_FREQ = 30

def postfilter_per_half_year(freq, words, min_freq=MIN_FREQ):
    """freq: array [half_years, words]. Null sub-threshold cells; drop a word
    only if nulled in every half-year."""
    cell_keep = freq >= min_freq
    nulled = np.where(cell_keep, freq, 0)
    col_keep = cell_keep.any(axis=0)
    kept_words = [w for w, k in zip(words, col_keep) if k]
    counts = {
        "low_frequency": int((~col_keep).sum()),
        "cells_nulled": int((~cell_keep[:, col_keep]).sum()),
    }
    return nulled[:, col_keep], kept_words, counts
```

**`global` (legacy).** One keep/drop decision per word, using its maximum
count over all half-years. A word is kept if at least one half-year reaches
`MIN_FREQ`; kept cells are left as-is (nothing nulled). This is simpler but
not leakage-safe: the mere presence of a column reflects the whole timeline.

| # | Rule | Setting | Log key |
|---|---|---|---|
| 4 | Remove a word if its frequency is below `MIN_FREQ` in **every** half-year | `MIN_FREQ = 30` | `low_frequency` |

```python
def postfilter(freq, words, min_freq=MIN_FREQ):
    """freq: array [half_years, words]. Keep words with max count >= min_freq."""
    keep = freq.max(axis=0) >= min_freq
    kept_words = [w for w, k in zip(words, keep) if k]
    counts = {"low_frequency": int((~keep).sum())}
    return freq[:, keep], kept_words, counts
```

**`--no-postfilter` (skip).** Keep every prefiltered word and every raw count
— no cells nulled, no columns dropped. Use this to build an unfiltered matrix
and defer the postfilter to a later step.

### Logging

- Log how many words each filter removed (prefilter: `log_prefilter_counts`; postfilter: `log_postfilter_counts`).
- Also write the counts to `freq_meta.json` (`filter_counts`), so they persist beyond console output.

### Notes on the filters

- **Order and attribution:** because stop words are mostly short, most are removed by the length filter first, so `stop_words` will be a small count. That reflects the order, not that few stop words existed.
- **Normalization:** the prefilter lowercases only for the stop-word check and keeps the original string. Use the same normalization as the frequency script's matching function, or "Molly" and "molly" can end up as separate columns.
- **Stop-word list:** save the exact list used (`freq_stopwords.txt`) and record the NLTK version, so the filter is reproducible. Run the NLTK download once during setup rather than at import time, in case the environment has no network.
- **Short DEA terms (check this):** a 4-character minimum removes 3-letter words such as "ice", "tea", and "pot", which have well-known drug senses. Any emerging euphemism of 3 letters or fewer would also never be examined, which is a recall loss in a recall-oriented filter. Before fixing the threshold, count how many words in the DEA euphemism list each filter removes (see checks). Options if many are lost: lower `MIN_TERM_LENGTH` to 3, or exempt a DEA allowlist from the prefilter (this protects only known words).
- **Postfilter threshold:** dropping a word only when it is below 30 in *every* half-year does not penalize newly emerging words, which may only be frequent in recent half-years. In the default `per-half-year` mode, cells below 30 are nulled rather than left in place, so a kept word can still have zeroed (sub-threshold) cells — the hit script's minimum contexts per cell is still needed. The pilot selection thresholds (`MIN_FREQ`, `MIN_PERIODS` in the hit plan) are stricter and apply on top of this. In the `global` mode, sub-threshold cells of a kept word are left as-is.
- **Totals:** filtering changes column sums, so they no longer equal total tokens. Save per-half-year total tokens before any filter and after all filters (`freq_totals.csv`) in case relative frequency is needed.

## 4. Files

| File | Contents |
|---|---|
| `freq.npy` | 2D array, shape `[half_years, words]`, dtype `int64` |
| `freq_half_years.csv` | half-year labels, in row order |
| `freq_words.csv` | word labels, in column order (all filters applied) |
| `freq_stopwords.txt` | the exact stop-word list that was applied |
| `freq_subreddits.txt` | copy of the subreddit list used (absent for an all-Reddit run) |

**Why a dense `.npy`:** at about 40 half-years and up to ~1M words, one matrix is roughly 160 MB as `int32` or 320 MB as `int64`. It loads in seconds, supports constant-time lookups, and maps directly onto the hit script's accumulator arrays. Sparse storage and long-format Parquet add complexity without a speed gain at this size. A long-format Parquet copy can be exported later for joins with the hit summary table, but the `.npy` is the source of truth.

**dtype:** use `int32`

## 5. Saving

```python
import numpy as np, pandas as pd, json

np.save("freq.npy", freq.astype("int64"))
pd.Series(half_years, name="half_year").to_csv("freq_half_years.csv", index=False)
pd.Series(words, name="word").to_csv("freq_words.csv", index=False)
open("freq_stopwords.txt", "w").write("\n".join(sorted(STOP_WORDS)))

totals.to_csv("freq_totals.csv")   # half_year, total_tokens, total_tokens_filtered

json.dump({
    "script": "freq_script.py", "version": "<commit or version>",
    "shape": list(freq.shape), "dtype": "int64",
    "corpus": "Reddit, <filtering description>",
    "scope": {"mode": "subreddits" if subs else "all",
              "file": SUBREDDIT_FILE, "hash": subreddit_hash(subs),
              "n_listed": len(subs) if subs else None,
              "n_found": n_found, "missing": sorted(missing)},
    "filters": {"min_term_length": 4, "max_words": 1,
                "stopwords": "nltk english", "min_freq_any_half_year": 30},
    "filter_counts": {**prefilter_counts, **postfilter_counts},
    "matching": "<tokenization and normalization rules>",
}, open("freq_meta.json", "w"), indent=2)
```

## 6. Loading and use in the hit script

```python
freq = np.load("freq.npy", mmap_mode="r")
half_years = pd.read_csv("freq_half_years.csv")["half_year"].tolist()
words = pd.read_csv("freq_words.csv")["word"].tolist()

hy_idx = {h: i for i, h in enumerate(half_years)}
w_idx  = {w: j for j, w in enumerate(words)}
TARGET_WORDS = set(words)
```

- Use the two dicts for per-context lookups in the scoring loop. Do not use pandas label lookups per context.
- Build every hit accumulator (`n_scored`, `hits_k`, `log_rank_sum`, `soft_sum`, `min_rank`) as a NumPy array of the same shape and label order, and save them the same way, so outputs line up with the frequency matrix cell for cell.
- Hit rate: `np.where(freq > 0, hits_k / freq, np.nan)`, so empty cells are `NaN`, not 0.

## 8. Checklist

- [ ] Add `SUBREDDIT_FILE` option (file or `None`) using the shared `load_subreddits` / `keep` module; log scope counts
- [ ] Run prefilter on the term list; log counts per filter
- [ ] Build the matrix for the kept terms
- [ ] Run postfilter (default `per-half-year`: null cells `< MIN_FREQ = 30`, drop words nulled in every half-year; `--postfilter-mode global` for legacy; `--no-postfilter` to skip); log counts
- [ ] Save `freq.npy`, label files, stop-word list, totals, and `freq_meta.json` (with filter settings and counts)
- [ ] Run the consistency checks above
- [ ] Load via the shared index dicts in the hit script
