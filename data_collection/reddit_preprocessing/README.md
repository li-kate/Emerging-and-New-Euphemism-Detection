# collect_instances.py

Scans Reddit comment/post dumps (`.zst`) for occurrences of target words
(e.g. drug slang, euphemisms, comparison terms) and saves every match —
with its full source text, timestamp, subreddit, and permalink — to a
JSONL file for downstream analysis.

## What it does

1. Loads one or more word list `.txt` files and merges them into a single
   word → category lookup.
2. Builds a fast multi-pattern matcher (Aho-Corasick) from that word list.
3. Streams comments/posts out of `.zst` Reddit dumps, filtering by year
   range and (optionally) subreddit.
4. For every record, finds all word-boundary matches and writes one JSONL
   row per match to the output file, plus a `_stats.json` summary of match
   counts per word.

## Requirements

```bash
pip install pyahocorasick zstandard orjson
```

- `pyahocorasick` and `zstandard` are required — the script raises a clear
  error if they're missing.
- `orjson` is optional but recommended for speed on large runs; the script
  falls back to the standard library `json` module if it isn't installed.

## Word list format

One word per line, plain text:

```
snow
ice
pot
```

- Blank lines and lines starting with `#` are skipped.
- Words are lowercased on load (matching is case-insensitive).
- Each file's words are tagged with a **category equal to the filename**
  (without extension). For example, a file called `euphemisms.txt`
  containing `snow` tags that word with category `"euphemisms"`.
- If you pass multiple files and the same word appears in more than one,
  the **last file wins** silently — no warning is logged. Word lists are
  expected to be reviewed manually, so this is treated as an accepted
  trade-off, not a bug.

## Usage

### Single file

```bash
python collect_instances.py \
  --words euphemisms.txt comparison_words.txt \
  --reddit-file RC_2020-01.zst \
  --output matches.jsonl
```

### Directory of per-subreddit dumps

```bash
python collect_instances.py \
  --words euphemisms.txt \
  --reddit-dir /data/reddit/by_subreddit \
  --output matches.jsonl \
  --workers 4
```

Produces one output file per input `.zst`, named
`matches_<subreddit_filename>.jsonl`. `--workers N` processes files in
parallel across N worker processes; each worker rebuilds its own matcher
and writes to its own output file, so there's no shared state to
coordinate. If an output file already exists, that source file is skipped
(safe to re-run after a partial failure).

### Directory of monthly dumps (`RC_YYYY-MM.zst`)

```bash
python collect_instances.py \
  --words euphemisms.txt \
  --reddit-monthly-dir /data/reddit/monthly \
  --output /data/matches/ \
  --workers 8
```

Files are grouped by month — `RC_2018-07.zst`, `RC_2018-07_part000.zst`,
and `RC_2018-07_part001.zst` are all treated as parts of `RC_2018-07` and
streamed into a single `RC_2018-07_matches.jsonl`. Without
`--slurm-task-id`, all months are processed, in parallel if `--workers` is
set. Already-completed months (existing output file) are skipped.

### SLURM array mode

```bash
#SBATCH --array=0-59
python collect_instances.py \
  --words euphemisms.txt \
  --reddit-monthly-dir /data/reddit/monthly \
  --output /data/matches/ \
  --slurm-task-id $SLURM_ARRAY_TASK_ID
```

Each array task processes exactly one month (sorted, indexed by task ID)
and exits early if the task ID is out of range. `--workers` is ignored in
this mode — parallelism comes from the job array itself, one month per
task.

## Other options

| Flag | Default | Description |
|---|---|---|
| `--subreddits PATH` | none (all subreddits) | Text file, one subreddit name per line, to restrict matching to |
| `--start-year` / `--end-year` | 2015 / 2026 | Year range filter on `created_utc` |
| `--context-sentences` | 2 | **Currently unused** — accepted but has no effect; see Known limitations |
| `--output` | required | Output JSONL path (or directory, for `--reddit-monthly-dir`) |

## Output format

Each line of the output JSONL:

```json
{
  "word": "snow",
  "category": "euphemisms",
  "sentence": "man it was snowing like crazy last night lol",
  "timestamp": "2019-03-14T02:11:07Z",
  "subreddit": "some_subreddit",
  "permalink": "/r/some_subreddit/comments/.../",
  "source": "reddit"
}
```

`sentence` is currently the **full comment/post body**, not a windowed
excerpt — see Known limitations below. A companion `<output>_stats.json`
is written alongside each output file with total record/match counts and
per-word match counts.

## Known limitations (not yet fixed)

- **`context_sentences` does nothing.** The parameter is threaded through
  the code but `sentence` is always the entire comment/post text,
  regardless of length. True sentence-window extraction (splitting on
  sentence boundaries and taking ±N sentences around the match) isn't
  implemented yet.
- **Decompression fallback can double-count.** If a `.zst` file fails
  partway through the primary decompression strategy, the fallback
  strategy re-reads the file from byte 0. Any records already yielded by
  the primary strategy before the failure get reprocessed and re-yielded,
  which can inflate match counts for that file. This only triggers on
  files that hit a decompression error — most files never take this path.
- **Silent category collisions.** If the same word appears in two input
  word lists, the later file's category wins with no warning. Fine as
  long as word lists are reviewed by hand before a run.

## Notes on performance

- Matching uses Aho-Corasick (`pyahocorasick`), so match time is
  effectively linear in text length regardless of word list size.
- JSON parsing/writing uses `orjson` when available, which is
  significantly faster than the standard library on the high line-volume
  Reddit dumps this is designed for.
- `--workers` parallelizes across independent files/months, not within a
  single file — each worker is a separate process with its own memory
  and its own Aho-Corasick automaton, so pick a worker count with your
  available RAM and CPU cores in mind, not just "as many as possible."