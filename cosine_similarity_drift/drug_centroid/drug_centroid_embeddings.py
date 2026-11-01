import glob
import io
import json
import os
import re
import time
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


REDDIT_MONTHLY_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SYNONYMS_FILE = "/storage/scratch1/8/arodrigues46/Emerging-and-New-Euphemism-Detection/drug_lists/drug_synonyms.txt"
SUBREDDIT_LIST_FILE = "/storage/scratch1/8/arodrigues46/Emerging-and-New-Euphemism-Detection/drug_lists/drug_subreddit_list.txt"
OUTPUT_DIR = "./drug_centroid_hy/checkpoints"
MODEL_NAME = "bert-base-uncased"

# Performance settings. BATCH_SIZE remains unchanged.
BATCH_SIZE = 256
USE_MIXED_PRECISION = True
PAD_TO_MULTIPLE_OF = 8

HIDDEN_DIM = 768
SAVE_EVERY = 100
WORDPIECES_BEFORE = 64
WORDPIECES_AFTER = 64

TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

URL_RE = re.compile(r"https?://\S+")
WHITESPACE_RE = re.compile(r"\s+")

try:
    import orjson

    def json_loads(s):
        return orjson.loads(s)
except ImportError:
    def json_loads(s):
        return json.loads(s)


def load_target_words(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Synonyms file not found: {path}")

    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith("#"):
                words.add(word)

    if not words:
        raise ValueError(f"No target words found in {path}")

    result = sorted(words, key=lambda w: (-len(w), w))
    print(f"Loaded {len(result):,} unique target terms from {path}")
    return result


def load_allowed_subreddits(path: str) -> set[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Subreddit list file not found: {path}")

    subreddits = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if (
                not line
                or line.startswith("#")
                or line.startswith("—")
                or line.startswith("-")
            ):
                continue

            if line.startswith("r/"):
                line = line[2:]
            elif line.startswith("/r/"):
                line = line[3:]
            elif line.startswith("/"):
                line = line[1:]

            if line and " " not in line:
                subreddits.add(line)

    if not subreddits:
        raise ValueError(f"No valid subreddits found in {path}")

    print(f"Loaded {len(subreddits):,} allowed subreddits from {path}")
    return subreddits


def build_matcher(words: list[str]):
    try:
        import ahocorasick
    except ImportError as exc:
        raise RuntimeError(
            "pyahocorasick is required. Install it with: pip install pyahocorasick"
        ) from exc

    automaton = ahocorasick.Automaton()
    for word in words:
        automaton.add_word(word, word)
    automaton.make_automaton()
    return automaton


def find_all_spans(text: str, automaton) -> list[tuple[str, int, int]]:
    lowered = text.lower()
    matches = []

    for end_inclusive, word in automaton.iter(lowered):
        start = end_inclusive - len(word) + 1
        end = end_inclusive + 1

        if start > 0 and lowered[start - 1].isalnum():
            continue
        if end < len(lowered) and lowered[end].isalnum():
            continue

        matches.append((word, start, end))

    return matches


def process_line(obj: dict) -> dict | None:
    body = obj.get("body", "")
    selftext = obj.get("selftext", "")
    title = obj.get("title", "")

    if body in ("[deleted]", "[removed]"):
        body = ""
    if selftext in ("[deleted]", "[removed]"):
        selftext = ""

    parts = []
    for part in (title, selftext, body):
        if isinstance(part, str) and part.strip():
            cleaned = WHITESPACE_RE.sub(" ", URL_RE.sub("", part)).strip()
            if cleaned:
                parts.append(cleaned)

    text = " ".join(parts)
    if not text or len(text.split()) < 3:
        return None

    try:
        timestamp_dt = datetime.fromtimestamp(
            float(obj.get("created_utc")), tz=timezone.utc
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return None

    # Calculate the month once per record.
    month = timestamp_dt.strftime("%Y-%m")

    return {"text": text, "month": month}


def stream_reddit_file(
    path: str,
    allowed_subreddits: set[str],
) -> Iterator[dict]:
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise RuntimeError(
            "zstandard is required. Install it with: pip install zstandard"
        ) from exc

    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
        with dctx.stream_reader(fh, read_size=2**20) as reader:
            with io.TextIOWrapper(
                reader, encoding="utf-8", errors="replace"
            ) as text_stream:
                for line in text_stream:
                    try:
                        obj = json_loads(line)
                    except (ValueError, TypeError):
                        continue

                    if not isinstance(obj, dict):
                        continue

                    subreddit = obj.get("subreddit", "")
                    if not isinstance(subreddit, str):
                        continue
                    if subreddit.lower() not in allowed_subreddits:
                        continue

                    record = process_line(obj)
                    if record is not None:
                        yield record


def prepare_sentence_windows(
    sentence: str,
    spans: list[tuple[str, int, int]],
    tokenizer,
) -> list[dict]:
    """Tokenize a sentence once, then prepare one BERT window per match."""
    encoded = tokenizer(
        sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )

    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    # Token offsets are ordered by character position.
    # Precompute token ends so bisect_right can locate the first token
    # that could overlap each target occurrence.
    token_ends = [end for _, end in offsets]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if cls_id is None or sep_id is None:
        raise RuntimeError("Tokenizer must define CLS and SEP token IDs")

    prepared = []

    for word, char_start, char_end in spans:
        # Find the first token whose end offset is greater than char_start.
        first_possible = bisect_right(token_ends, char_start)

        target_positions = []
        token_index = first_possible

        # Only inspect tokens that could overlap this target span.
        while (
            token_index < len(offsets)
            and offsets[token_index][0] < char_end
        ):
            start, end = offsets[token_index]

            if end > char_start:
                target_positions.append(token_index)

            token_index += 1

        if not target_positions:
            prepared.append({
                "candidate": word,
                "input_ids": None,
                "target_positions": None,
            })
            continue

        first_target = target_positions[0]
        last_target = target_positions[-1]

        left = max(0, first_target - WORDPIECES_BEFORE)
        right = min(
            len(token_ids),
            last_target + 1 + WORDPIECES_AFTER,
        )

        window_ids = token_ids[left:right]
        input_ids = [cls_id] + window_ids + [sep_id]

        if len(input_ids) > tokenizer.model_max_length:
            raise ValueError(
                f"Target window has {len(input_ids)} tokens; "
                f"maximum is {tokenizer.model_max_length}"
            )

        target_local = [position - left + 1 for position in target_positions]

        prepared.append({
            "candidate": word,
            "input_ids": input_ids,
            "target_positions": target_local,
        })

    return prepared


def get_batch_span_embeddings(batch_data, tokenizer, model, device):
    """Run BERT on already-tokenized target windows."""
    valid_indices = [
        i for i, item in enumerate(batch_data)
        if item["input_ids"] is not None
    ]
    results = [None] * len(batch_data)

    if not valid_indices:
        return results

    sequences = [batch_data[i]["input_ids"] for i in valid_indices]
    max_len = max(len(ids) for ids in sequences)

    if PAD_TO_MULTIPLE_OF:
        max_len = min(
            ((max_len + PAD_TO_MULTIPLE_OF - 1) // PAD_TO_MULTIPLE_OF)
            * PAD_TO_MULTIPLE_OF,
            tokenizer.model_max_length,
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise RuntimeError("Tokenizer must define a pad_token_id")

    input_ids = torch.full(
        (len(sequences), max_len),
        pad_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_len),
        dtype=torch.long,
    )

    for row, ids in enumerate(sequences):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1

    if device.type == "cuda":
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        if device.type == "cuda" and USE_MIXED_PRECISION:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                hidden = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
        else:
            hidden = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

    # Convert back to float32 before pooling/accumulation for stable sums.
    hidden = hidden.float().cpu()

    for row, original_index in enumerate(valid_indices):
        target_positions = batch_data[original_index]["target_positions"]
        results[original_index] = (
            hidden[row, target_positions].mean(dim=0).numpy()
        )

    return results


def new_accumulator():
    return defaultdict(
        lambda: defaultdict(
            lambda: {
                "sum_embedding": np.zeros(HIDDEN_DIM, dtype=np.float64),
                "count": 0,
            }
        )
    )


def save_checkpoint(path, candidate_slices, stats, rows_processed):
    data = {
        "rows_processed": rows_processed,
        "stats": stats,
        "candidate_slices": {},
    }

    for word, slices in candidate_slices.items():
        data["candidate_slices"][word] = {
            month: {
                "sum_embedding": entry["sum_embedding"].tolist(),
                "count": entry["count"],
            }
            for month, entry in slices.items()
        }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        acc = new_accumulator()
        for word, slices in data["candidate_slices"].items():
            for month, entry in slices.items():
                vector = np.asarray(
                    entry["sum_embedding"], dtype=np.float64
                )
                if vector.shape != (HIDDEN_DIM,):
                    raise ValueError(
                        f"Unexpected embedding shape for {word}/{month}: "
                        f"{vector.shape}"
                    )
                acc[word][month]["sum_embedding"] = vector
                acc[word][month]["count"] = int(entry["count"])

        print(f"Resuming from valid record {data['rows_processed']:,}")
        return acc, data["stats"], int(data["rows_processed"])

    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint {path} is unreadable; move/delete it to restart: {exc}"
        ) from exc


def main():
    if TASK_ID < 0:
        raise ValueError("TASK_ID must be >= 0")

    run_start = time.perf_counter()
    timings = {
        "read_parse_filter_seconds": 0.0,
        "matching_seconds": 0.0,
        "tokenization_seconds": 0.0,
        "model_batch_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }

    words = load_target_words(SYNONYMS_FILE)
    allowed_subreddits = load_allowed_subreddits(SUBREDDIT_LIST_FILE)
    automaton = build_matcher(words)

    # Preserve the same RC-then-RS ordering used by the SLURM task mapping.
    dump_files = (
        sorted(glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RC_*.zst")))
        + sorted(glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RS_*.zst")))
    )

    if not dump_files:
        raise FileNotFoundError(
            f"No RC_*.zst or RS_*.zst files found in {REDDIT_MONTHLY_DIR}"
        )
    if TASK_ID >= len(dump_files):
        raise IndexError(
            f"TASK_ID {TASK_ID} is outside 0..{len(dump_files)-1}; "
            f"found {len(dump_files)} dump files"
        )

    target_file = dump_files[TASK_ID]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(
        OUTPUT_DIR, f"partial_results_{TASK_ID}.json"
    )
    checkpoint_file = os.path.join(
        OUTPUT_DIR, f"checkpoint_{TASK_ID}.json"
    )

    print(f"Task {TASK_ID}/{len(dump_files)-1}: {Path(target_file).name}")
    print(f"Output: {output_file}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Checkpoint interval: {SAVE_EVERY} batches")
    print(f"Mixed precision: {USE_MIXED_PRECISION and torch.cuda.is_available()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required")

    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    if int(model.config.hidden_size) != HIDDEN_DIM:
        raise ValueError(
            f"Expected hidden size {HIDDEN_DIM}, "
            f"model reports {model.config.hidden_size}"
        )

    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        candidate_slices, stats, rows_to_skip = checkpoint
    else:
        candidate_slices = new_accumulator()
        stats = {
            "valid_records_processed": 0,
            "records_with_matches": 0,
            "matches_found": 0,
            "embedded": 0,
            "skipped_no_tokens": 0,
        }
        rows_to_skip = 0

    batch_queue = []
    batches_since_save = 0
    valid_record_index = 0

    def flush_batch(queue):
        batch_start = time.perf_counter()
        embeddings = get_batch_span_embeddings(
            queue, tokenizer, model, device
        )
        timings["model_batch_seconds"] += time.perf_counter() - batch_start

        for item, embedding in zip(queue, embeddings):
            if embedding is None:
                stats["skipped_no_tokens"] += 1
                continue

            month = item["month"]
            entry = candidate_slices[item["candidate"]][month]
            entry["sum_embedding"] += embedding.astype(
                np.float64, copy=False
            )
            entry["count"] += 1
            stats["embedded"] += 1

    # Explicit next() calls let us measure time spent reading/decompressing,
    # parsing JSON, filtering subreddits, and preparing valid records.
    record_iterator = iter(stream_reddit_file(target_file, allowed_subreddits))

    while True:
        read_start = time.perf_counter()
        try:
            record = next(record_iterator)
        except StopIteration:
            break
        timings["read_parse_filter_seconds"] += (
            time.perf_counter() - read_start
        )

        valid_record_index += 1

        if valid_record_index <= rows_to_skip:
            continue

        stats["valid_records_processed"] += 1
        sentence = record["text"]
        record_month = record["month"]

        matching_start = time.perf_counter()
        matches = find_all_spans(sentence, automaton)
        timings["matching_seconds"] += time.perf_counter() - matching_start

        if matches:
            stats["records_with_matches"] += 1
            stats["matches_found"] += len(matches)

            tokenization_start = time.perf_counter()
            prepared_matches = prepare_sentence_windows(
                sentence, matches, tokenizer
            )
            timings["tokenization_seconds"] += (
                time.perf_counter() - tokenization_start
            )

            for prepared in prepared_matches:
                if prepared["input_ids"] is None:
                    stats["skipped_no_tokens"] += 1
                    continue

                batch_queue.append({
                    "candidate": prepared["candidate"],
                    "input_ids": prepared["input_ids"],
                    "target_positions": prepared["target_positions"],
                    "month": record_month,
                })

                while len(batch_queue) >= BATCH_SIZE:
                    current_batch = batch_queue[:BATCH_SIZE]
                    del batch_queue[:BATCH_SIZE]
                    flush_batch(current_batch)
                    batches_since_save += 1

        if batches_since_save >= SAVE_EVERY:
            if batch_queue:
                flush_batch(batch_queue)
                batch_queue = []

            checkpoint_start = time.perf_counter()
            save_checkpoint(
                checkpoint_file,
                candidate_slices,
                stats,
                valid_record_index,
            )
            timings["checkpoint_seconds"] += (
                time.perf_counter() - checkpoint_start
            )
            batches_since_save = 0

        if valid_record_index % 1_000_000 == 0:
            elapsed = time.perf_counter() - run_start
            print(
                f"Read {valid_record_index:,} valid records; "
                f"matched {stats['matches_found']:,} occurrences; "
                f"embedded {stats['embedded']:,}; "
                f"elapsed {elapsed / 60:.1f} min"
            )

    if batch_queue:
        flush_batch(batch_queue)

    output = {"data": {}, "stats": stats}
    for word, slices in candidate_slices.items():
        output["data"][word] = {
            month: {
                "sum_embedding": entry["sum_embedding"].tolist(),
                "count": entry["count"],
            }
            for month, entry in slices.items()
        }

    tmp_output = output_file + ".tmp"
    with open(tmp_output, "w", encoding="utf-8") as f:
        json.dump(output, f)
    os.replace(tmp_output, output_file)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    term_counts = {
        word: sum(entry["count"] for entry in months.values())
        for word, months in output["data"].items()
    }
    terms_with_embeddings = sum(
        count > 0 for count in term_counts.values()
    )
    months = sorted({
        month
        for slices in output["data"].values()
        for month in slices
    })

    total_elapsed = time.perf_counter() - run_start

    print("\n--- Task Statistics ---")
    print(f"Input file: {Path(target_file).name}")
    print(f"Valid records processed: {stats['valid_records_processed']:,}")
    print(f"Records with matches: {stats['records_with_matches']:,}")
    print(f"Matched occurrences: {stats['matches_found']:,}")
    print(f"Successfully embedded occurrences: {stats['embedded']:,}")
    print(f"Occurrences skipped during embedding: {stats['skipped_no_tokens']:,}")
    print(f"Unique terms matched: {len(term_counts):,}")
    print(f"Terms with embeddings: {terms_with_embeddings:,}")
    print(f"Terms without embeddings: {len(words) - terms_with_embeddings:,}")
    print(f"Months represented: {len(months):,}")
    print(
        f"Month range: {months[0] if months else 'N/A'} to "
        f"{months[-1] if months else 'N/A'}"
    )

    print("\n--- Timing ---")
    for name, seconds in timings.items():
        print(f"{name}: {seconds / 60:.2f} minutes")
    print(f"Total elapsed: {total_elapsed / 60:.2f} minutes")

    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()