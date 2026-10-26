
import glob
import io
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


REDDIT_MONTHLY_DIR = "/storage/cedar/cedar0/cedarp-smussmann3-0/kmehta301/reddit_zst"
SYNONYMS_FILE = "../../drug_lists/drug_synonyms.txt"
SUBREDDIT_LIST_FILE = "../../drug_lists/drug_subreddit_list.txt"
OUTPUT_DIR = "/storage/ice1/0/1/kli605/Emerging-and-New-Euphemism-Detection/cosine_similarity_drift/drug_centroid/checkpoints"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 128
HIDDEN_DIM = 768
SAVE_EVERY = 20
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

    return {
        "text": text,
        "timestamp_dt": timestamp_dt,
    }


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


def prepare_target_window(
    sentence: str,
    char_start: int,
    char_end: int,
    tokenizer,
):
    encoded = tokenizer(
        sentence,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )

    offsets = encoded["offset_mapping"]
    target_positions = [
        i
        for i, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]

    if not target_positions:
        return None

    first_target = target_positions[0]
    last_target = target_positions[-1]
    left = max(0, first_target - WORDPIECES_BEFORE)
    right = min(
        len(encoded["input_ids"]),
        last_target + 1 + WORDPIECES_AFTER,
    )

    ids = encoded["input_ids"][left:right]
    input_ids = tokenizer.build_inputs_with_special_tokens(ids)
    target_local = [p - left + 1 for p in target_positions]

    return input_ids, target_local


def get_batch_span_embeddings(batch_data, tokenizer, model, device):
    prepared = [
        prepare_target_window(
            item["sentence"],
            item["span"][0],
            item["span"][1],
            tokenizer,
        )
        for item in batch_data
    ]

    valid_indices = [
        i for i, item in enumerate(prepared) if item is not None
    ]
    results = [None] * len(batch_data)

    if not valid_indices:
        return results

    sequences = [prepared[i][0] for i in valid_indices]
    max_len = max(len(ids) for ids in sequences)
    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        raise RuntimeError("Tokenizer must define a pad_token_id")

    input_ids = torch.full(
        (len(sequences), max_len), pad_id, dtype=torch.long
    )
    attention_mask = torch.zeros(
        (len(sequences), max_len), dtype=torch.long
    )

    for row, ids in enumerate(sequences):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state.cpu()

    for row, original_index in enumerate(valid_indices):
        target_positions = prepared[original_index][1]
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

    words = load_target_words(SYNONYMS_FILE)
    allowed_subreddits = load_allowed_subreddits(SUBREDDIT_LIST_FILE)
    automaton = build_matcher(words)

    dump_files = sorted(
        glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RC_*.zst"))
        + glob.glob(os.path.join(REDDIT_MONTHLY_DIR, "RS_*.zst"))
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
        embeddings = get_batch_span_embeddings(
            queue, tokenizer, model, device
        )

        for item, embedding in zip(queue, embeddings):
            if embedding is None:
                stats["skipped_no_tokens"] += 1
                continue

            month = item["timestamp_dt"].strftime("%Y-%m")
            entry = candidate_slices[item["candidate"]][month]
            entry["sum_embedding"] += embedding.astype(
                np.float64, copy=False
            )
            entry["count"] += 1
            stats["embedded"] += 1

    for record in stream_reddit_file(target_file, allowed_subreddits):
        valid_record_index += 1

        if valid_record_index <= rows_to_skip:
            continue

        stats["valid_records_processed"] += 1
        sentence = record["text"]
        matches = find_all_spans(sentence, automaton)

        if matches:
            stats["records_with_matches"] += 1

        for word, start, end in matches:
            stats["matches_found"] += 1
            batch_queue.append({
                "sentence": sentence,
                "span": (start, end),
                "candidate": word,
                "timestamp_dt": record["timestamp_dt"],
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

            save_checkpoint(
                checkpoint_file,
                candidate_slices,
                stats,
                valid_record_index,
            )
            batches_since_save = 0

        if valid_record_index % 1_000_000 == 0:
            print(
                f"Read {valid_record_index:,} valid records; "
                f"matched {stats['matches_found']:,} occurrences; "
                f"embedded {stats['embedded']:,}"
            )

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
    print(f"Month range: {months[0] if months else 'N/A'} to {months[-1] if months else 'N/A'}")
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
