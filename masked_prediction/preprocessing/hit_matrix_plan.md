# Plan: Hit Matrix Script for Emerging Euphemism Candidates

## 1. Purpose

Preprocessing step that flags candidate words for a separate downstream euphemism detector. For each (half-year, word), measure how often a masked language model (MLM) predicts a DEA drug term as a substitute for the word. The output is a set of matrices (half-years x words) that the detector will consume.

Because this is a candidate filter, **recall matters more than precision**. A false positive costs the detector extra work; a false negative means a real emerging euphemism is never examined.

## 2. Inputs

| Input | Notes |
|---|---|
| Corpus | Reddit, tagged by half-year (e.g. `2012H1`). Scope is set by `SUBREDDIT_FILE`: a `subreddits.txt` list, or `None` for all of Reddit (see section 2b) |
| Frequency matrix | Built by a separate process. Rows are half-years, columns are words. **The words (columns) in this matrix are the final target words.** Also serves as the denominator for hit rate |
| DEA term list | DEA slang/code-word list, **single-token terms only** (as seen by the BERT tokenizer) |
| MLM | BERT. Needed: tokenizer and logits at the mask position |

Loaders (target words, frequency matrix) should each be a single function so the source can change.

## 2b. Corpus scope: subreddit list or all of Reddit

One setting, `SUBREDDIT_FILE`: a path to `subreddits.txt` (one subreddit per line; `r/` prefix, case, and whitespace normalized; blank lines and `#` comments ignored) or `None` for all of Reddit.

- **Must match the frequency matrix.** The frequency matrix is built on a specific scope, and hit rate divides by it. Load the scope with the **same shared function** the frequency script uses (`load_subreddits` / `keep`; code in the frequency plan, section 2b), and apply it in the corpus reader before any scoring.
- **Verify at startup:** compare the subreddit mode and list hash with `freq_meta.json`. If they differ, stop with an error instead of scoring against the wrong denominator.
- **Separate outputs per scope** (e.g. `hits_all/`, `hits_subreddits/`), so runs never overwrite each other.
- **Log:** subreddits listed, found, and missing; contexts dropped by the filter.
- **Subset effects:** a smaller corpus means fewer words survive the frequency filters, fewer contexts per cell, and noisier hit rates. Redo pilot selection (section 5b) and the control-set k choice on the subset matrix. A k chosen on all of Reddit may not transfer, and some pilot words may no longer be frequent enough.

## 3. Definitions

- **Context:** one occurrence of the target word, with that occurrence replaced by the mask token. The text around it is the sentence containing it (see "Context construction" below). Each occurrence is its own context, so a word appearing twice in one sentence gives two contexts.
- **best_rank:** rank (1 = top prediction) of the highest-ranked DEA term over the full vocabulary, after excluding the target word and junk tokens.
- **Hit at k:** a context where `best_rank <= k`. Each context counts at most once, even if several DEA terms are in the top k.
- **Hits (per half-year, word):** number of contexts in that cell that were hits at k.
- **Frequency:** number of times the word occurs in that half-year, taken from the external frequency matrix. This counts occurrences, so it equals the number of contexts.
- **n_scored:** number of contexts this script actually scored in that cell (should equal frequency; see section 8).
- **Hit rate:** `hits / frequency`.
- **mean_log_rank:** average of `log(best_rank)` over a cell's contexts (natural log); exponentiating gives the geometric mean rank. Lower means DEA terms rank higher on average. The log is used because ranks are heavily skewed (most in the thousands, a few in single digits), so a raw mean is dominated by the tail.
- **soft_score:** total softmax probability on DEA terms at the mask position (optional graded signal).

## 3b. Context construction

Decision: **sentence level**, with a minimum and maximum length.

- **Unit:** the sentence containing the occurrence.
- **Too short** (under about 5 to 8 words): extend with neighboring text until the minimum is reached.
- **Too long or unpunctuated** (Reddit run-ons): truncate to a window around the target (about 32 words on each side, or whatever fits comfortably under BERT's 512-token limit).
- **Multiple occurrences in one sentence:** mask one occurrence at a time and leave the others visible. Each masked version counts as one context.
- **Why sentence level:** it matches the sentence-sized contexts used in the substitution literature (verify in Zhu et al. and Card), costs far less than a 64-word window at Reddit scale, and avoids topic contamination from surrounding sentences that mention drugs.
- **Matching the frequency matrix:** an existing frequency script already defines what counts as an occurrence (case, punctuation, word boundaries, variants). Reference that script: import and reuse its tokenization and matching function rather than reimplementing it, so the occurrences found here are the same ones it counted. Do not duplicate its rules in this plan.
- **No silent drops:** if an occurrence is skipped (too long, tokenization failure, empty after cleaning), log the reason. Do not deduplicate reposts unless the frequency matrix did.
- **Empirical check:** run sentence versus ±64-word window on the pilot words (recall on positives, flag rate on negatives, time per context). If recall is similar, keep the sentence.

## 4. Per-context computation

For each batch of masked contexts:

1. Run the model, slice logits at the mask position only: `logits` of shape `[B, V]`.
2. Cast to float32.
3. Apply exclusions by setting logits to `-inf`:
   - the target word (and optionally trivial variants: plural, inflections, case)
   - subword pieces, punctuation, stopwords
4. Remove the target word from the DEA id set for that context (a target that is itself a DEA term must not produce a trivial hit).
5. Compute:

```python
log_probs  = torch.log_softmax(logits, dim=-1)
soft_score = log_probs[:, dea_ids].exp().sum(-1)             # [B]

best_dea_logit = logits[:, dea_ids].max(dim=-1).values       # [B]
best_rank = (logits > best_dea_logit[:, None]).sum(-1) + 1   # [B]
```

6. Move only `best_rank` and `soft_score` to CPU. Never store full logits or top-k lists.

Notes:
- Ranking is over the full vocabulary, so any k can be evaluated later.
- Apply the same exclusion rules to every context so scores are comparable.
- Strict `>` means tied logits share a rank.
- DEA terms: single-token terms only, applied consistently.
- A target word that BERT splits into several wordpieces is still replaced by a single mask token.

## 5. Single-run workflow (summary for all words, per-context for pilot words)

Everything happens in one pass over the corpus. There are no separate pilot and full phases.

Setup: load the frequency matrix. Its columns define the target words; only these words are scored. Its non-zero cells define which (half-year, word) pairs to expect.

For each scored batch:

1. **All words:** update running counts per (half_year, word): `n_scored`, `hits_k` for every k in the kept list (5, 10, 15, 20, 50), the running `min_rank`, the running sum of `log(best_rank)`, and the running sum of `soft_score` (divide by `n_scored` at the end for `mean_log_rank` and `mean_soft`).
2. **Pilot words only** (the control set of known positives and negatives, defined in a small config list): additionally append the per-context row to a separate pilot table.

Pilot table (small, pilot words only):

| word | half_year | context_id | best_rank | soft_score |
|---|---|---|---|---|

```python
TARGET_WORDS = set(freq_matrix.columns)   # final target words come from the frequency matrix
PILOT_WORDS = set(load_pilot_words())     # DEA euphemism positives + frequency-matched negatives
KS = [5, 10, 15, 20, 50]

# per batch, after computing best_rank and soft_score
for w, hy, cid, r, s in zip(words, half_years, context_ids, best_rank, soft_score):
    cell = counts[(hy, w)]
    cell["n_scored"] += 1
    cell["min_rank"] = min(cell["min_rank"], r)
    cell["log_rank_sum"] += math.log(r)
    cell["soft_sum"] += s
    for k in KS:
        cell[f"hits_{k}"] += int(r <= k)
    if w in PILOT_WORDS:
        pilot_rows.append((w, hy, cid, r, s))
```

Why this design:
- k can be chosen after the run from the summary columns, because all k values are stored for every word.
- The pilot table additionally supports finer analysis on the control words: bootstrap confidence intervals, trying k values not in `KS`, and re-checking the hit definition or DEA list for those words only.
- Storage stays small, since per-context rows exist only for the pilot words. This matters at Reddit scale.

Limits to keep in mind:
- For non-pilot words you can only use the k values in `KS`, and you cannot change the DEA list or hit definition without rerunning. Choose `KS` generously (include a large value like 50 or 100).
- Per-context rows are kept for pilot words only (decided), so bootstrap intervals are available for pilot words only.

## 5b. Selecting the pilot words (code plus manual review)

The frequency matrix supports most of this step in code. A few judgment calls stay manual.

**Automated steps:**

1. **Eligible positives:** intersect the DEA euphemism list with the frequency matrix columns, keeping words with at least `MIN_FREQ` occurrences in at least `MIN_PERIODS` half-years (so hit rates are stable).
2. **Split:** randomly divide eligible positives into a pilot set and a held-out set with a fixed seed. Consider splitting within frequency bins so both sets have a similar mix.
3. **Frequency-matched negatives:** from non-DEA words that pass the same eligibility filter, find words whose log-frequency profile over half-years is closest to each positive. Sample from the nearest ~20 instead of always taking the single closest, and do not reuse a word.
4. **Save** the seed and final lists to a config file so the pilot set is reproducible.

```python
import numpy as np, pandas as pd

rng = np.random.default_rng(0)
logf = np.log1p(freq)                      # half-years x words
dea = set(dea_words)

MIN_FREQ, MIN_PERIODS = 50, 8              # tune to the data
ok = (freq >= MIN_FREQ).sum(axis=0) >= MIN_PERIODS

eligible_pos = [w for w in freq.columns if w in dea and ok[w]]
pool = [w for w in freq.columns if w not in dea and ok[w]]

shuffled = list(rng.permutation(eligible_pos))
n_pilot = 30
pilot_pos, heldout_pos = shuffled[:n_pilot], shuffled[n_pilot:]

def pick_negatives(positives, per_pos=1, top=20):
    chosen, avail = [], list(pool)
    for w in positives:
        X = logf[avail].T.values
        d = np.linalg.norm(X - logf[w].values, axis=1)
        near = [avail[i] for i in np.argsort(d)[:top]]
        picks = list(rng.choice(near, size=per_pos, replace=False))
        chosen += picks
        avail = [a for a in avail if a not in picks]
    return chosen

pilot_neg = pick_negatives(pilot_pos)
PILOT_WORDS = set(pilot_pos) | set(pilot_neg)
```

**Manual steps:**

- **Review positives:** label each as ambiguous everyday word (e.g. "ice", "tea", "lean") or mostly drug-sense (e.g. "molly"). Include both kinds in the pilot set; stratify the split by this label if possible.
- **Screen negatives:** frequency matching cannot detect an unlisted drug sense. Skim the chosen negatives, and after the run inspect any with high hit rates (hidden euphemism, topic contamination, or a bug).
- **Optional extras:** add a few hand-picked hard negatives (words that appear near drug discussion but are not euphemisms, e.g. "dealer", "overdose"), and check part of speech if a tagger is available.

**Notes:**

- Set `MIN_FREQ` and `MIN_PERIODS` so enough positives remain; loosen them if too few survive.
- Aim for roughly 20 to 50 positives and a similar number of negatives in the pilot set.
- Keep the held-out DEA words out of the pilot list; use them only for final evaluation.

## 6. Choosing k

Compute hits at k in {5, 10, 15, 20, 50} for the pilot words (pilot table and/or their summary columns from the same run; no separate pilot run is needed).

1. Build a labeled control set (these are the pilot words):
   - **Positives:** words from the DEA euphemism list (e.g. molly, snow, ice, tea, grass, crack, lean), with the half-years where the sense was established if known.
   - **Negatives:** words with the same frequency (matched using the frequency matrix) that are not known drug euphemisms.
2. For each k, define "flagged" (e.g. hit rate above a threshold in at least one half-year, with a minimum context count).
3. Compute recall on positives, flag rate on negatives, and the fraction of the vocabulary passing the filter. Plot against k.
4. Choose the smallest k where recall plateaus or reaches the target (about 90%), keeping the pass-through rate manageable.
5. Vary k and the hit-rate threshold together (small grid), since they interact.
6. Hold out a separate set of words for final detector evaluation; do not tune k on the same words used to evaluate the detector. Since positives come from the DEA list, reserve some DEA words that are not in the pilot list.

Defaults if no tuning is done: k = 10 (lexical substitution convention), k = 50 (generous, recall-oriented), k = 20 as a single compromise.

## 7. Outputs

### Summary table (one row per half-year x word)

| Column | Meaning |
|---|---|
| `half_year`, `word` | keys |
| `frequency` | occurrences, from the external frequency matrix |
| `n_scored` | contexts actually scored by this script |
| `hits_k` | contexts with a DEA term in top k (one column per kept k) |
| `hit_rate_k` | `hits_k / frequency` |
| `min_rank` | best rank any DEA term reached in the cell (rough diagnostic; one lucky context sets it) |
| `mean_log_rank` | mean of `log(best_rank)` over the cell's contexts; more stable than `min_rank` and registers movement below the top-k cutoff |
| `mean_soft` | mean soft score |

### Matrices (half-years x words)

The frequency matrix already exists, so only the hit and derived matrices are new:

```python
hits_k = summary.pivot(index="half_year", columns="word", values="hits_k")
rate_k = hits_k / freq_matrix.reindex_like(hits_k)    # NaN where freq == 0
```

Also optionally: `mean_log_rank`, `min_rank` and `mean_soft` matrices. Axes must match the frequency matrix exactly (same half-years, same words, same order).

## 8. Validation and sanity checks

- **Frequency consistency:** `n_scored` should equal the frequency matrix count for every cell. If contexts are skipped (too long, tokenization failures, sampling), the rate is no longer hits / true frequency; either fix the mismatch or divide by `n_scored` instead and record the difference. Since matching is reused from the frequency script, a mismatch most likely means a dropped context (too long, tokenization failure, empty after cleaning) or a bug in how the shared function is called.
- **Scope check:** the subreddit mode and list hash match the frequency matrix's `freq_meta.json`.
- Hits are nested: `hits_5 <= hits_10 <= hits_15 <= hits_20 <= hits_50`. A violation indicates a bug.
- Known positives should light up at a reasonable k; ordinary words should not.
- Mask cells with frequency below a minimum when computing rates; report `NaN`, not 0.
- Spot-check a few contexts by hand: print the text, the top 10 predictions, and the computed `best_rank`.
- Confirm a target that is itself a DEA term does not hit trivially.

## 9. Efficiency notes

- The forward pass dominates the cost; rank and soft score are O(V) post-processing.
- Batch contexts; slice the mask position before any vocabulary-wide operation.
- Use half precision for the forward pass if tolerated; cast logits to float32 before softmax.
- Compute `best_rank` and `soft_score` on the GPU and transfer only those two values.
- BERT's input limit is 512 tokens; sentence-level contexts (plus the length cap) fit easily and are several times cheaper than a 64-word window.
- At Reddit scale, aggregate on the fly and checkpoint the running counts periodically so a crash does not lose the run.

## 10. Downstream interpretation (for the detector)

- **mean_log_rank vs. hit rate:** hit rate only sees contexts that cross a threshold; mean_log_rank also registers contexts that moved from, say, rank 5000 to rank 200. It is a useful early-movement signal, but mask low-frequency cells before using it.
- **Median is not a summary here:** it only moves once more than half of a word's contexts rank DEA terms high, so it misses emerging words with a small share of drug-sense contexts. Compute exact medians and percentiles for pilot words only (from the pilot table), as descriptive statistics when choosing k.
- **Optional finer thresholds:** adding a few log-spaced counts (e.g. 100, 500, 1000) beyond `KS` gives a coarse rank histogram per cell for every word, from which approximate quantiles can be read.

- A word that always sits near drug terms (e.g. "heroin") is not emerging. Emergence is a rise relative to the word's own earlier half-years, so keep the full time series.
- Frequent words get more chance hits; the frequency-matched negatives help calibrate this.
- Hit rate needs enough contexts per cell; bootstrap confidence intervals are available for pilot words only.

## 11. Implementation checklist

- [ ] Loader for the frequency matrix; target words = its columns
- [ ] `SUBREDDIT_FILE` option via the shared module; startup check that the scope hash matches `freq_meta.json`
- [ ] Loader for DEA terms; keep single-token terms only; map to BERT token ids
- [ ] Corpus reader yielding (word, half_year, context text), with the target replaced by the mask token
- [ ] Sentence splitter with min-length extension, max-length truncation, and one-mask-per-occurrence handling
- [ ] Import the frequency script's occurrence-matching function (no reimplementation)
- [ ] Logging of any skipped occurrences with reasons
- [ ] Exclusion mask (target, variants, junk tokens)
- [ ] Batched scoring function returning `best_rank` and `soft_score`
- [ ] Pilot selection script: eligible positives, seeded pilot/held-out split, frequency-matched negatives (section 5b)
- [ ] Manual review of positives (ambiguous vs. drug-sense) and negatives; save the final pilot list and seed to a config file
- [ ] Single run: on-the-fly aggregation of all words with multiple k columns, plus per-context rows for pilot words only
- [ ] Pilot comparison of sentence vs. ±64-word window (optional sanity check)
- [ ] Control-set evaluation to choose k
- [ ] Summary table and matrix export aligned to the frequency matrix (including `mean_log_rank`)
- [ ] Sanity checks (frequency consistency, nesting, spot checks, trivial-hit test)

## 12. Decisions and open questions

Decided:
- MLM: BERT. Corpus: all of Reddit.
- Target words: the columns of the externally built frequency matrix.
- DEA terms: single-token terms only.
- Pilot list: positives are words from the DEA euphemism list; negatives are frequency-matched words that are not known drug euphemisms.
- Per-context rows: pilot words only.
- `KS` = 5, 10, 15, 20, 50.
- Context unit: sentence level, with min-length extension and max-length truncation; one context per occurrence.
- Frequency matrix counts occurrences (word count per half-year), so frequency = number of contexts.
- Occurrence matching: reuse the existing frequency script's logic; no separate rules defined here.

Still open:
- Which subreddits go in `subreddits.txt`; run on the list, all of Reddit, or both?
- Minimum contexts per cell for computing a hit rate?
- Values for `MIN_FREQ`, `MIN_PERIODS`, and the pilot size?
- Which positives are labeled ambiguous vs. drug-sense after manual review?
- Exact min and max context lengths for the sentence rule.

