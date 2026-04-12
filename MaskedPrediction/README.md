# Masked prediction shift — `masked_prediction.ipynb` reference

This document describes the **masked prediction** Jupyter notebook in this repository: what problem it addresses, how it connects to the euphemism pipeline, what each step does, and how to interpret or extend the results.

---

## 1. Purpose in one paragraph

The notebook implements a **diachronic masked-language-model probe**. For each **euphemism candidate** (a phrase discovered upstream), it takes **real corpus sentences** where that phrase appears, **replaces the matched span with BERT `[MASK]` tokens**, and inspects the model’s **top‑k** predictions at those positions. It then checks whether those predictions overlap a **taboo-related lexicon** derived from `taboo_words_refined.py`. Finally, it **groups instances by time period** (year or decade) and measures whether the **fraction of contexts where a taboo word appears in the top‑k predictions** rises over time. An upward trend is interpreted as **evidence that contextual usage is shifting toward taboo-denoting language**—the same intuition as “if you mask the euphemism, the model increasingly fills in blunt/taboo words.”

---

## 2. Scientific framing

### 2.1 What is being measured?

- **Not** a classifier accuracy or a single scalar “euphemism score.”
- **Yes** a **usage-shift signal**: relative frequency (over sampled contexts) with which a **fixed masked LM**, given masked spans in **period‑tagged** sentences, proposes **lexical items that belong to a taboo anchor list**.

### 2.2 Why masked LM at all?

Masked language models encode **distributional preferences**: “what word is most natural here?” If a phrase historically appeared in **neutral** contexts, masking it might yield **neutral** completions. If the **same surface phrase** increasingly appears in contexts **aligned with taboo semantics**, the model—trained on large general corpora—may assign higher probability to **taboo-denoting** fillers at the mask. That is a **complementary** signal to embedding similarity (first pass) and string co-occurrence (second pass).

### 2.3 What this does *not* prove by itself

- It does **not** establish speaker intent or offensiveness.
- It does **not** separate **sense change** from **sampling bias** (e.g., more news about a topic in later years).
- The **taboo lexicon** is a **proxy** for “taboo-related”; overlap can be **noisy** (polysemy, subword pieces, short tokens).

These caveats belong in any paper or report that cites this analysis.

---

## 3. Where this notebook sits in the pipeline

```
Corpus (time-stamped text)
    → first_pass.py   → first_pass_results.jsonl   (candidate hits + similarity)
    → second_pass.py  → second_pass_results.jsonl  (exhaustive instances + offsets)
    → masked_prediction.ipynb  (this notebook)
```

The notebook **consumes** second-pass JSONL by default. It does **not** re-run FAISS or Aho–Corasick matching.

---

## 4. Upstream output schemas (exact fields)

Understanding these fields matters for **reproducibility** and for **adapting** the notebook to other files.

### 4.1 `first_pass.py` → `first_pass_results.jsonl`

Each line is one JSON object (a serialized `EuphemismCandidate`).

| Field | Meaning |
|--------|---------|
| `text` | Extracted **phrase** (candidate surface form from perturbation / windowing). |
| `context` | **Full sentence** (or record text) containing the hit. |
| `taboo_anchor` | Nearest taboo anchor label from embedding similarity. |
| `taboo_category` | Category of that anchor. |
| `similarity_score` | Similarity to that anchor. |
| `phrase_extraction_method` | e.g. perturbation vs contrastive masking. |
| `phrase_drop_score` | Score from phrase extraction. |
| `timestamp` | From the source record (string). |
| `source_url`, `source` | Provenance. |
| `context_window_size` | Optional; windowed vs full-sentence mode. |
| `all_anchor_matches` | Optional list of multiple anchor hits. |

**Note:** First-pass rows do **not** include **character offsets** for the phrase inside `context`. To use this notebook with first-pass data alone, you must **locate** `text` inside `context` (e.g., first match, case-insensitive), then pass the resulting `(start, end)` into the same masking functions.

### 4.2 `second_pass.py` → `second_pass_results.jsonl` (default input)

Each line is one JSON object (a serialized `EuphemismInstance`).

| Field | Meaning |
|--------|---------|
| `phrase` | **Surface** phrase matched in the sentence. |
| `canonical_phrase` | Canonical form from first-pass deduplication. |
| `sentence` | The **sentence** containing the match (used for masking). |
| `before_context`, `after_context` | Lists of adjacent sentences (not used in the current notebook logic). |
| `char_offset_start`, `char_offset_end` | **Character indices** into `sentence` for the matched span **[start, end)**. |
| `taboo_anchors` | List of `{anchor, category, score}` from first pass. |
| `primary_category` | Category of the strongest anchor association. |
| `first_pass_similarity`, `first_pass_phrase_drop` | Aggregated first-pass scores. |
| `timestamp` | From the source record. |
| `source_url`, `source` | Provenance. |
| `match_mode` | `exact`, `lemma`, or `stem`. |
| `is_variant` | Whether the surface form differed from the canonical phrase. |

The notebook uses **`sentence`**, **`char_offset_start`**, **`char_offset_end`**, **`canonical_phrase`**, **`timestamp`**, and optionally **`primary_category`** for analysis and plotting.

---

## 5. Why the notebook prefers second-pass data

1. **Exact span alignment:** Second pass records **character-level** offsets produced by the matcher. Masking uses tokenizer **offset_mapping** so that every subword piece overlapping that span becomes `[MASK]`. That avoids fragile string searches when punctuation or casing differs.
2. **Exhaustive instances:** Second pass is designed to list **many** occurrences per candidate, which supports **stable** per-period rates.
3. **Same sentence field:** The offsets are guaranteed to refer to `sentence`, which is exactly what gets tokenized.

---

## 6. Dependencies and environment

The notebook expects (see the first code cell):

- **Python** with `torch`, `transformers`, `pandas`, `matplotlib`.
- A typical install:

```bash
pip install torch transformers pandas matplotlib
```

**GPU:** Optional but recommended for large JSONL files; CPU works for smaller samples.

**Working directory:** Cells use `Path(".").resolve()` as `REPO_ROOT`, so run Jupyter from the **repository root** (or adjust paths).

---

## 7. Configuration variables (notebook cell 1)

These are the main **knobs** you should document alongside any experiment.

| Variable | Role |
|----------|------|
| `SECOND_PASS_JSONL` | Path to `second_pass_results.jsonl` (or equivalent). |
| `TABOO_WORDS_PY` | Path to `taboo_words_refined.py` (must define `TABOO_ANCHORS`). |
| `MODEL_NAME` | Default `bert-base-uncased` (uncased BERT MLM). |
| `TOP_K` | How many logits to keep **per mask position** before unioning (default 15). |
| `MAX_INSTANCES` | Random cap on rows after time parsing (`None` = all rows; can be slow). |
| `MAX_PER_GROUP` | After grouping by `(canonical_phrase, period)`, keep at most this many rows per group (default 50; `None` = no cap). |
| `TIME_BUCKET` | `"year"` or `"decade"` for aggregating timestamps. |

---

## 8. Taboo vocabulary construction

Function: `load_taboo_vocab(path)`.

1. **Executes** `taboo_words_refined.py` in a namespace (same pattern as `first_pass.load_taboo_anchors`).
2. Reads **`TABOO_ANCHORS`**: a dict from **category name → list of anchor strings**.
3. Normalizes each string to **lowercase**, adds the full string to a set.
4. **Splits** on whitespace and hyphens and adds **parts** longer than **2 characters** (so multi-word anchors contribute useful fragments).

The **hit test** (`prediction_hits_taboo`) does **not** use embeddings. It takes the **decoded** top‑k strings from the LM, extracts **ASCII letter runs** with a regex, lowercases them, and returns **true** if **any** such word is **in** the taboo set.

**Implications:**

- Short or ambiguous tokens in the lexicon can add noise.
- **Subword** outputs are decoded with `tokenizer.decode([id])`; multi-token taboo phrases are **not** fully matched unless a **single** decoded piece equals a lexicon word.

---

## 9. Timestamp parsing and time buckets

Function: `parse_timestamp(raw)`.

Order of attempts:

1. `datetime.fromisoformat` after normalizing `Z` to `+00:00`.
2. Several `strptime` formats (`%Y-%m-%d`, etc.).
3. A regex that grabs a **4-digit year** `19xx` or `20xx` and sets **January 1** of that year.

Rows with **no** parseable timestamp are **dropped** before analysis.

Function: `bucket_label(dt, mode)`:

- **`year`:** string of the calendar year, e.g. `"2019"`.
- **`decade`:** e.g. `"2010s"`.

Period labels are **strings**; plots use them on the x-axis in lexical order (fine for year strings; for decades, same).

---

## 10. Masking algorithm (core)

### 10.1 `mask_phrase_spans(tokenizer, sentence, char_start, char_end, device)`

1. Tokenizes `sentence` with **`return_offsets_mapping=True`**, special tokens on, **truncation** at **512** tokens (BERT limit).
2. For each **offset pair** `(s, e)` for each token:
   - Skips pairs `(0, 0)` (special tokens in typical HF behavior).
   - If the token span **overlaps** the half-open interval **[char_start, char_end)** in character space, that token’s id is replaced by **`tokenizer.mask_token_id`**.
3. Returns **`input_ids`** on the chosen **device** and the list of **masked sequence indices**.

**Multi-word phrases** become **multiple** `[MASK]` tokens—one per affected WordPiece piece. That matches standard practice for phrase MLM probes.

### 10.2 Edge cases

- **No overlapping tokens** (e.g., bad offsets, empty span, tokenizer mismatch): masked list is empty → the row contributes **`taboo_in_topk = False`** (conservative).
- **Truncation:** If the sentence is very long, only the **first 512 tokens** are modeled; offsets still refer to the **original** string, so if the match falls **beyond** the truncated region, overlap can be empty—again yielding no masks.

---

## 11. Top‑k extraction and “hit” definition

### 11.1 `topk_at_masked_positions`

For **each** masked index, takes **`torch.topk`** on the vocabulary logits at that position (k clipped by vocab size). Converts each token id to a string via **`tokenizer.decode([tid])`**, strips whitespace, and **appends** to a flat list.

So for **M** mask positions, you can have up to **M × k** decoded strings (with duplicates possible).

### 11.2 `row_taboo_hit`

Returns **true** if **`prediction_hits_taboo`** finds **any** lexicon hit in **any** of those decoded strings.

**Interpretation:** A single **high-ranking** taboo word at **any** mask position counts as a **positive** for that **sentence instance**. This is **permissive** by design for exploratory shift detection; you can tighten it (e.g., require hits at **all** positions, or only the **first** mask) in a fork of the notebook.

---

## 12. Data loading and filtering (cell 4)

1. Loads all valid JSON lines into a **pandas** `DataFrame`.
2. Fails fast if the JSONL file is **missing** (with a clear `FileNotFoundError`).
3. Parses timestamps → drops rows without dates.
4. Adds **`period`** column.
5. Optionally **subsamples** globally (`MAX_INSTANCES`).
6. Optionally **caps** each `(canonical_phrase, period)` group with **`groupby(...).head(MAX_PER_GROUP)`** (first rows in file order within each group—not random).

For **balanced** sampling across periods, you may want to replace `.head` with stratified or random sampling inside each group.

---

## 13. Model inference loop (cell 5)

1. Loads **`AutoTokenizer`** and **`AutoModelForMaskedLM`** for `MODEL_NAME`.
2. Sets **`model.eval()`** and uses **`torch.inference_mode()`** in the top‑k helper.
3. Iterates **`df`** rows; for each row calls **`row_taboo_hit`** with `sentence`, `char_offset_start`, `char_offset_end`, and **`TOP_K`**.
4. Builds a **`records`** list with `canonical_phrase`, `period`, `taboo_in_topk`, `primary_category`.

### 13.1 Aggregation

`agg` groups by **`canonical_phrase`** and **`period`**:

- **`taboo_rate`:** mean of `taboo_in_topk` → **fraction of instances** in that group with at least one taboo hit in the union of top‑k predictions.
- **`n`:** count of instances in the group.

---

## 14. Visualization (cell 6)

1. Pivots **`taboo_rate`** to a matrix: **index = period**, **columns = canonical_phrase**.
2. Builds a **count** matrix for sample sizes per `(phrase, period)`.
3. **`phrases_ok`:** keeps phrases where **at least two** periods have count **≥ `MIN_N_PER_PERIOD`** (default 3). If none qualify, falls back to the **first eight** phrases (so the plot is not empty on tiny data).
4. Plots up to **12** phrases as lines with markers.

**Y-axis label:** `Fraction of contexts with taboo token in top-{TOP_K}`.

---

## 15. How to read the plot

- **Rising line:** For that **canonical phrase**, a larger share of **masked contexts** in **later** periods produced at least one taboo lexicon hit in the model’s top‑k predictions.
- **Flat or noisy lines:** May indicate stable usage, insufficient **n**, or **LM + lexicon** noise.
- **Compare `n` in `agg`:** Always cross-check rates against counts; **sparse** periods exaggerate variance.

---

## 16. Limitations and design choices (summary)

| Topic | Limitation |
|--------|------------|
| **LM** | `bert-base-uncased` is **not** time-stamped; it is a **single** snapshot. “Shift” is measured **relative to your corpus periods**, not relative to BERT’s training cut-off in a causal sense. |
| **Lexicon** | Taboo “hit” is **string overlap** with anchor lists, not semantic entailment. |
| **Multi-mask** | Union of top‑k across positions inflates positives vs single-mask policies. |
| **Truncation** | Long sentences may **lose** the match beyond position 512. |
| **Sampling** | `MAX_INSTANCES` and `head` per group introduce **selection bias** unless carefully designed. |
| **Period confounds** | Topic drift in the news domain can correlate with calendar time. |

---

## 17. Adapting to `first_pass_results.jsonl`

1. Load JSONL into a DataFrame.
2. For each row, set `sentence = context` and find `(start, end)` such that `sentence[start:end]` matches `text` (decide **case** and **which occurrence** if multiple).
3. Use the same **`mask_phrase_spans` → topk_at_masked_positions → prediction_hits_taboo`** pipeline.
4. Keep **`timestamp`** for period bucketing; **`text`** or a normalized key can stand in for **`canonical_phrase`** in groupbys.

---

## 18. Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| `FileNotFoundError` for JSONL | Path wrong or second pass not run; set `SECOND_PASS_JSONL` to an absolute path. |
| All-zero or very low taboo rates | `TOP_K` too small; lexicon mismatch with BERT’s vocabulary; masks not aligned (check offsets). |
| Empty plot | Too few periods or phrases; relax `MIN_N_PER_PERIOD` or increase data. |
| Slow runs | Reduce `MAX_INSTANCES`, use GPU, or batch inference (not implemented in current notebook). |

---

## 19. Cell-by-cell map

| Cell | Content |
|------|---------|
| 0 | Markdown: goal + upstream schema summary (also note: fix accidental triple-quoted string in markdown if you want it to render as prose only). |
| 1 | Imports and **configuration**. |
| 2 | Taboo load, **timestamp** parsing, **period** labels, **hit** predicate. |
| 3 | **Masking**, **top‑k**, **`row_taboo_hit`**. |
| 4 | Load JSONL, filter, **sample caps**. |
| 5 | Load BERT, **inference loop**, **`agg`** table. |
| 6 | **Line plot** of taboo rate over period. |

---

## 20. File reference

- **Notebook:** `masked_prediction.ipynb`
- **Default inputs:** `second_pass_results.jsonl`, `taboo_words_refined.py`
- **Related code:** `first_pass.py` (`EuphemismCandidate`), `second_pass.py` (`EuphemismInstance`)

This document is meant to stand alone for collaborators and for **Methods** / **Appendix** sections in academic write-ups. Update the **Configuration** and **Limitations** sections if you change the notebook’s behavior.
