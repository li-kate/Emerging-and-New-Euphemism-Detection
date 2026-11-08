# Temporal Zhu euphemism detection (month → half-year)

Run the **detection** stage of Zhu et al.'s euphemism model over time: one
ranked euphemism-candidate list **per half-year** instead of a single list for
the whole corpus, so emerging candidates can be tracked period by period.

This mirrors the structure of `masked_prediction/preprocessing/hit_matrix.py`:
a per-month **map** step (one SLURM array task per month) writes shards, and a
**merge** step reduces them — here, bucketed and re-ranked per half-year.

`zhu_model/` is **not modified**. The detection scoring (`detection.MLM` /
`single_MLM` and the informativeness filter) is reproduced here, parameterized,
and the Reddit-tuned BERT model directory + vocab are passed in. The scripts
live outside `zhu_model/` because that folder's `.gitignore` ignores everything
in it.

## Files

| File | Role |
|---|---|
| `zhu_detection_month.py` | Map: score one month, write a per-half-year candidate-score shard |
| `merge_zhu_detection.py` | Reduce: sum shards per half-year, write ranked candidate lists |
| `submit_zhu_detection.sbatch` | Standalone single-month job (debugging) |
| `_zhu_batch_controller.sh` | Self-chaining batch controller (submit-cap aware) |
| `submit_zhu_detection_pipeline.sh` | Launches the full map-array → merge pipeline |
| `submit_zhu_missing.sh` | Re-maps only months that have no shard (failed/never ran), then re-merges all |

## What each half-year result is

For every sentence containing an input keyword (Zhu's known target terms — the
formal drug names from the euphemism answer key), the keyword is masked and
scored with BERT. A masked
sentence is kept only if at least two input keywords appear in the mask's top
`--thres` predictions (Zhu's noise gate). Each surviving prediction's
probability is added to a running score for that token, after dropping
punctuation, English stopwords, the input keywords, `drug`/`drugs`, and `##`
word-pieces. Candidates are ranked by total score, highest first — identical to
`detection.MLM`, but the score sums are bucketed by the sentence's half-year so
each half-year ranks independently.

Difference from Zhu's one-shot run: no global `ms_limit` random sample and no
shuffle — every eligible masked sentence in a month is scored. A per-half-year
cap is available via `--ms-limit-per-half-year` if a bucket is too large.

## Outputs (in `OUT_DIR`)

- `<stem>_ranked.csv` — rows of `half_year, rank, word, score` (the ranked
  candidate list for every half-year).
- `<stem>_top_words.json` — `{half_year: [word, ...]}`, the ordered candidate
  words per half-year (what Zhu's identification stage would consume), capped
  at `--top-n`.
- `<stem>_meta.json` — scope, keywords, model, Zhu params, per-half-year
  masked/informative counts, shard count, and agreement checks.

## Run it

Full pipeline (edit the config block at the top of the launcher first — dump
dir, subreddit list, keywords file, model dir/vocab, year range):

```bash
chmod +x submit_zhu_detection_pipeline.sh _zhu_batch_controller.sh
./submit_zhu_detection_pipeline.sh
```

The launcher globs the Reddit monthly dumps, keeps the months inside
`[START_YEAR, END_YEAR]`, and splits them into batches of at most
`MAX_SUBMIT - 1` tasks so the cluster's QOS submit cap is never exceeded. The
controller submits one batch at a time (each later batch `afterany` the
previous) and finally the merge. At most one map batch plus one
controller/merge job is queued at any moment.

Single month (debugging):

```bash
sbatch --array=0 submit_zhu_detection.sbatch
```

Local smoke test on a few months (no SLURM), e.g. two months into shards then
merge:

```bash
python zhu_detection_month.py \
    --input-dir /path/to/jsonl_or_zst \
    --max-months 2 --shard-index 0 \
    --answer-file ../zhu_model/data/euphemism_answer_drug.txt \
    --model-dir ../BERT_model_reddit \
    --vocab-file ../BERT_model_reddit/vocab.txt \
    --shard-dir zhu_detect_test/shards

python merge_zhu_detection.py \
    --shard-dir zhu_detect_test/shards \
    --out-dir zhu_detect_test \
    --stem zhu_detect
```

## Failed months (GPU busy, etc.)

Each month is a separate GPU array task. A task can land on a node whose GPU is
momentarily busy or unavailable (`CUDA error: CUDA-capable device(s) is/are
busy or unavailable`) and die before the model loads. This is a cluster/GPU
condition, not a data problem, and it hits different months on different runs.
Because batches chain with `afterany`, a failed month does not stall the rest —
it just leaves no shard, and the merge silently omits it.

To recover, fill the gaps (same idea as the frequency matrix's
`submit_freq_missing.sh`):

```bash
./submit_zhu_missing.sh --dry-run   # list months with no shard
./submit_zhu_missing.sh             # re-map only those, then re-merge all shards
```

It maps only months that have a dump but no shard, leaves existing shards
untouched, and re-sums everything. Keep its config identical to the original
run (the merge rejects shards scored with different params). A month with a
partial/bad shard counts as "present" — delete that shard to force a rescore.

## Input keywords

Pass exactly one of:

- `--answer-file` (preferred): Zhu's `data/euphemism_answer_<cat>.txt`. The
  input keywords are derived from it **exactly** as `zhu_model/read_file.py`
  does — the formal drug names before each `:`, lowercased, de-duplicated,
  sorted. For `euphemism_answer_drug.txt` this is the same 57 keywords a real
  Zhu detection run uses.
- `--keywords-file`: a plain newline-delimited term list (one target term per
  line). Fallback for running against an arbitrary list that is not in Zhu's
  `key: values` answer-file format.

## Key parameters (defaults match `zhu_model/detection.py`)

- `--mlm-k` (50): top-k predictions kept per mask.
- `--thres` (5): a masked sentence is kept only if ≥2 input keywords are in the
  mask's top-`thres` predictions.
- `--ms-limit-per-half-year` (none): optional cap on masked sentences scored
  per half-year.
- `--max-chars` (150): skip source texts longer than this many characters
  before masking, exactly as Zhu's `read_file.read_raw_text` does
  (`len(line) <= 150`). Set `0` to disable.
- `--max-length` (512): safety truncation around `[MASK]` for any text that
  still over-runs BERT's position limit (rare once `--max-chars` is on).
- `--top-n` (merge, 500): keep only the top-N candidates per half-year in the
  outputs.

## Notes

- Scope (`--subreddit-file`) is applied in the shared corpus reader
  (`shared_utils/corpus_stream.py`), the same reader the frequency and hit
  matrices use, so "occurrence" and half-year labels line up with those builds.
  The scope hash and Zhu params are recorded in every shard and the merge
  aborts if months disagree.
- Dependencies: `numpy nltk zstandard orjson torch pytorch_pretrained_bert`.
  `pytorch_pretrained_bert` is the same BERT library `zhu_model/detection.py`
  uses, so the forward pass matches byte-for-byte. NLTK `punkt` / `stopwords`
  are downloaded on first run if missing.
