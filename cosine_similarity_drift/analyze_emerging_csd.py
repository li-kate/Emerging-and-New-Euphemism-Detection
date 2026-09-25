"""
Per-file BERT embedding script for SLURM array jobs.
Extracts target-centered contextual span embeddings for all words (anchors,
euphemism candidates, established euphemisms, and comparison words).

For corpus occurrences, the script uses exact character offsets emitted by the
collector, keeps +/- CONTEXT_TOKENS_EACH_SIDE BERT wordpieces around the target,
and averages ONLY the target wordpiece hidden states to form the occurrence
vector. This prevents long-comment truncation from dropping the target and
prevents repeated target words from being confused with the first occurrence.

INCREMENTAL SAVING: Checkpoints to disk every SAVE_EVERY batches.
If the job crashes or times out, restarting picks up from the last
checkpoint — no work is lost.

Saves:
  - Per-word, per-month sum embeddings + counts (for averaging in merge)
  - Per-anchor template span embeddings (for per-anchor similarity in merge)
  - Corpus anchor partial sums (for corpus-derived anchor embeddings in merge)
  - Word group labels so the merge script can categorize results

Usage: Run as SLURM array job, one task per JSONL file.
Output: partial_results_{task_id}.json
"""

import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../matches"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 256
HIDDEN_DIM = 768
SAVE_EVERY = 20  # Save checkpoint every N batches (~5120 rows)

# Context strategy: keep exactly this many BERT wordpiece tokens on each side
# of the target occurrence. The target's own wordpieces are always retained.
CONTEXT_TOKENS_EACH_SIDE = 64
EMBEDDING_VERSION = "target_centered_wordpiece_window_v1"

# Output paths
OUTPUT_FILE = f"partial_results_{task_id}.json"
CHECKPOINT_FILE = f"checkpoint_{task_id}.json"

# ──────────────────────────────────────────────────────────────
# WORD GROUPS
# ──────────────────────────────────────────────────────────────
ANCHORS = [
    "cathinones",
    "cocaine",
    "heroin",
    "marijuana",
    "fentanyl",
    "methamphetamine",
    "meth",
    "amphetamine",
    "oxycodone",
    "xanax",
    "adderall",
    "mdma",
    "ecstasy",
    "lsd",
    "pcp",
    "codeine",
    "ketamine",
    "bath salts",
]

ESTABLISHED_EUPHEMISMS = ["molly", "coke", "crystal", "ping"]

EUPHEMISM_CANDIDATES = [
    "zing",
    "zaza",
    "flakka",
    "yart",
    "fein",
    "fenty",
    "pressed",
    "penjamin",
    "ouid",
    "oui'd",
    "tranq",
    "gray death",
    "grey death",
    "usb stick",
    "fetty",
    "tusi",
    "stamps",
    "tucibi",
    "happy water",
]

COMPARISON_WORDS = ["needle", "pharmacy", "prescription", "overdose"]

WORD_GROUPS = {}
for w in ANCHORS:
    WORD_GROUPS[w.lower()] = "anchor"
for w in ESTABLISHED_EUPHEMISMS:
    WORD_GROUPS[w.lower()] = "established_euphemism"
for w in EUPHEMISM_CANDIDATES:
    WORD_GROUPS[w.lower()] = "euphemism_candidate"
for w in COMPARISON_WORDS:
    WORD_GROUPS[w.lower()] = "comparison"

# Spelling variants that should be treated as the same word
WORD_ALIASES = {
    "grey death": "gray death",
    "oui'd": "ouid",
}

ANCHOR_TEMPLATES = [
    "They were caught with {}.",
    "The effects of {} can be dangerous.",
    "People use {} recreationally at parties.",
    "The police seized a large amount of {}.",
]


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def get_month_slice(timestamp):
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        return dt.strftime("%Y-%m")
    except Exception:
        return "unknown"


def find_span(text, word):
    """Legacy fallback for old JSONL files that do not contain offsets."""
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, text, re.IGNORECASE)
    return (match.start(), match.end()) if match else (None, None)


def embedding_config():
    """Configuration that must match before a checkpoint can be resumed."""
    return {
        "embedding_version": EMBEDDING_VERSION,
        "model_name": MODEL_NAME,
        "context_tokens_each_side": CONTEXT_TOKENS_EACH_SIDE,
        "layer_strategy": "last_hidden_state",
        "target_pooling": "mean_target_wordpieces",
    }


def crop_target_context(text, char_start, char_end):
    """
    Create a target-centered BERT-wordpiece context window.

    Returns:
        (context_text, new_char_start, new_char_end) or None

    The returned character offsets are relative to `context_text`. We tokenize
    the full source text only to determine wordpiece boundaries; BERT itself is
    run later on the much smaller cropped context.
    """
    if not (0 <= char_start < char_end <= len(text)):
        return None

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoded["offset_mapping"]
    if not offsets:
        return None

    # A token belongs to the target if its character span overlaps the target
    # span. Overlap is safer than requiring full containment for punctuation or
    # unusual tokenization edge cases.
    target_token_indices = [
        i
        for i, (ts, te) in enumerate(offsets)
        if te > char_start and ts < char_end
    ]
    if not target_token_indices:
        return None

    first_target = target_token_indices[0]
    last_target = target_token_indices[-1]

    left_token = max(0, first_target - CONTEXT_TOKENS_EACH_SIDE)
    right_token = min(
        len(offsets), last_target + 1 + CONTEXT_TOKENS_EACH_SIDE
    )

    context_char_start = offsets[left_token][0]
    context_char_end = offsets[right_token - 1][1]
    context_text = text[context_char_start:context_char_end]

    new_start = char_start - context_char_start
    new_end = char_end - context_char_start
    if not (0 <= new_start < new_end <= len(context_text)):
        return None

    return context_text, new_start, new_end


# ──────────────────────────────────────────────────────────────
# CHECKPOINT SAVE / LOAD
# ──────────────────────────────────────────────────────────────
def save_checkpoint(candidate_slices, corpus_anchor_sums, stats, rows_processed):
    """Save current state to disk. Atomic write via rename."""
    data = {
        "embedding_config": embedding_config(),
        "rows_processed": rows_processed,
        "stats": stats,
        "corpus_anchor_sums": {
            anchor: {"sum": d["sum"].tolist(), "count": d["count"]}
            for anchor, d in corpus_anchor_sums.items()
        },
        "candidate_slices": {},
    }
    for cand, slices in candidate_slices.items():
        data["candidate_slices"][cand] = {}
        for month, entry in slices.items():
            data["candidate_slices"][cand][month] = {
                "sum_embedding": entry["sum_embedding"].tolist(),
                "count": entry["count"],
            }

    # Write to temp file then rename — prevents corruption if killed mid-write
    tmp_file = CHECKPOINT_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(data, f)
    os.replace(tmp_file, CHECKPOINT_FILE)


def load_checkpoint():
    """Load checkpoint if it exists. Returns (candidate_slices, corpus_anchor_sums, stats, rows_to_skip) or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    print(f"Found checkpoint: {CHECKPOINT_FILE}")
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)

        if data.get("embedding_config") != embedding_config():
            print(
                "  [WARN] Checkpoint was created with a different embedding "
                "configuration; starting fresh to avoid mixing incompatible vectors."
            )
            return None

        # Rebuild candidate_slices
        candidate_slices = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "sum_embedding": np.zeros(HIDDEN_DIM),
                    "count": 0,
                }
            )
        )
        for cand, slices in data["candidate_slices"].items():
            for month, entry in slices.items():
                candidate_slices[cand][month]["sum_embedding"] = np.array(
                    entry["sum_embedding"]
                )
                candidate_slices[cand][month]["count"] = entry["count"]

        # Rebuild corpus_anchor_sums
        corpus_anchor_sums = {}
        for anchor, d in data["corpus_anchor_sums"].items():
            corpus_anchor_sums[anchor] = {
                "sum": np.array(d["sum"]),
                "count": d["count"],
            }

        rows_processed = data["rows_processed"]
        stats = data["stats"]
        stats.setdefault("skipped_invalid_offsets", 0)
        stats.setdefault("skipped_context_build", 0)
        stats.setdefault("legacy_span_fallback", 0)
        print(f"  Resuming from row {rows_processed}")
        print(
            f"  Stats so far: embedded={stats['embedded']}, skipped={stats['skipped_no_tokens']}"
        )
        return candidate_slices, corpus_anchor_sums, stats, rows_processed

    except Exception as e:
        print(f"  [WARN] Checkpoint corrupted ({e}), starting fresh.")
        return None


# ──────────────────────────────────────────────────────────────
# MODEL SETUP
# ──────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def extract_span_embedding(text, char_start, char_end):
    """Embed one target span; used for the short anchor templates."""
    inputs = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**{k: v for k, v in inputs.items() if k != "offset_mapping"})

    embeddings = outputs.last_hidden_state.squeeze(0).cpu()
    offsets = inputs["offset_mapping"].squeeze(0).cpu()

    token_indices = []
    for j, (ts, te) in enumerate(offsets):
        ts, te = ts.item(), te.item()
        if ts == 0 and te == 0:
            continue
        if te > char_start and ts < char_end:
            token_indices.append(j)

    if not token_indices:
        return None
    return embeddings[token_indices].mean(dim=0).numpy()


def get_batch_span_embeddings(batch_data):
    """
    Run BERT on already-cropped target-centered contexts and return one vector
    per target occurrence by averaging only the target's BERT wordpieces.
    """
    texts = [d["text"] for d in batch_data]

    inputs = tokenizer(
        texts,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    offset_mapping = inputs.pop("offset_mapping").cpu()

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.cpu()
    results = []

    for i in range(len(batch_data)):
        char_start, char_end = batch_data[i]["span"]
        offsets = offset_mapping[i]

        token_indices = []
        for j, (ts, te) in enumerate(offsets):
            ts, te = ts.item(), te.item()
            if ts == 0 and te == 0:
                continue
            if te > char_start and ts < char_end:
                token_indices.append(j)

        if not token_indices:
            results.append(None)
        else:
            results.append(embeddings[i][token_indices].mean(dim=0).numpy())

    return results


# ──────────────────────────────────────────────────────────────
# BUILD PER-ANCHOR TEMPLATE EMBEDDINGS (Option A)
# Always recomputed (fast, deterministic, no checkpoint needed)
# ──────────────────────────────────────────────────────────────
print("Building per-anchor template embeddings (Option A)...")
anchor_template_embeddings = {}

for anchor in ANCHORS:
    word_vectors = []
    for template in ANCHOR_TEMPLATES:
        sentence = template.format(anchor)
        start, end = find_span(sentence, anchor)
        if start is None:
            continue
        vec = extract_span_embedding(sentence, start, end)
        if vec is not None:
            word_vectors.append(vec)
    if word_vectors:
        anchor_template_embeddings[anchor] = np.mean(word_vectors, axis=0)
        print(f"  ✓ {anchor} ({len(word_vectors)} templates)")
    else:
        print(f"  [WARN] No vectors for '{anchor}', skipping.")

print(
    f"Template embeddings for {len(anchor_template_embeddings)}/{len(ANCHORS)} anchors.\n"
)


# ──────────────────────────────────────────────────────────────
# IDENTIFY TARGET FILE
# ──────────────────────────────────────────────────────────────
jsonl_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl")))
if task_id >= len(jsonl_files):
    print(f"Task ID {task_id} exceeds file count ({len(jsonl_files)}). Exiting.")
    exit()

target_file = jsonl_files[task_id]
print(f"Task {task_id}: processing {os.path.basename(target_file)}")


# ──────────────────────────────────────────────────────────────
# LOAD CHECKPOINT OR INITIALIZE FRESH
# ──────────────────────────────────────────────────────────────
checkpoint = load_checkpoint()

if checkpoint:
    candidate_slices, corpus_anchor_sums, stats, rows_to_skip = checkpoint
else:
    candidate_slices = defaultdict(
        lambda: defaultdict(
            lambda: {
                "sum_embedding": np.zeros(HIDDEN_DIM),
                "count": 0,
            }
        )
    )
    corpus_anchor_sums = {
        anchor: {"sum": np.zeros(HIDDEN_DIM), "count": 0} for anchor in ANCHORS
    }
    stats = {
        "rows_read": 0,
        "span_found": 0,
        "embedded": 0,
        "skipped_no_span": 0,
        "skipped_invalid_offsets": 0,
        "skipped_context_build": 0,
        "skipped_no_tokens": 0,
        "legacy_span_fallback": 0,
    }
    rows_to_skip = 0

anchor_lower_map = {a.lower(): a for a in ANCHORS}


# ──────────────────────────────────────────────────────────────
# PROCESS FILE
# ──────────────────────────────────────────────────────────────
batch_queue = []
batches_since_save = 0
current_row = 0


def flush_batch(queue):
    """Process batch and accumulate results."""
    if not queue:
        return
    embs = get_batch_span_embeddings(queue)
    for i, emb in enumerate(embs):
        if emb is not None:
            item = queue[i]
            month = get_month_slice(item["timestamp"])
            word_lower = item["candidate"].lower()
            word_lower = WORD_ALIASES.get(word_lower, word_lower)

            entry = candidate_slices[word_lower][month]
            entry["sum_embedding"] += emb
            entry["count"] += 1
            stats["embedded"] += 1

            if word_lower in anchor_lower_map:
                anchor_key = anchor_lower_map[word_lower]
                corpus_anchor_sums[anchor_key]["sum"] += emb
                corpus_anchor_sums[anchor_key]["count"] += 1
        else:
            stats["skipped_no_tokens"] += 1


with open(target_file, "r") as f:
    for line in f:
        if not line.strip():
            continue

        current_row += 1

        # Skip rows already processed (from checkpoint)
        if current_row <= rows_to_skip:
            continue

        stats["rows_read"] += 1

        row = json.loads(line)
        # New collector uses `text`; `sentence` is accepted only so older files
        # can still be read. Older files do not identify repeated occurrences
        # reliably, so fresh schema-v2 collection is preferred for final runs.
        text = row.get("text") or row.get("sentence", "")
        candidate = row.get("word", "")
        timestamp = row.get("timestamp", "")

        if not text or not candidate:
            continue

        if "match_start" in row and "match_end" in row:
            try:
                start = int(row["match_start"])
                end = int(row["match_end"])
            except (TypeError, ValueError):
                stats["skipped_invalid_offsets"] += 1
                continue

            # Do not silently fall back when schema-v2 offsets are malformed;
            # that could embed the wrong occurrence in a repeated-word comment.
            if not (0 <= start < end <= len(text)):
                stats["skipped_invalid_offsets"] += 1
                continue
            if text[start:end].casefold() != candidate.casefold():
                stats["skipped_invalid_offsets"] += 1
                continue
        else:
            # Backward compatibility only. re.search returns the first
            # occurrence, so this is not occurrence-safe for repeated targets.
            start, end = find_span(text, candidate)
            if start is None:
                stats["skipped_no_span"] += 1
                continue
            stats["legacy_span_fallback"] += 1

        stats["span_found"] += 1

        cropped = crop_target_context(text, start, end)
        if cropped is None:
            stats["skipped_context_build"] += 1
            continue

        context_text, context_start, context_end = cropped
        batch_queue.append(
            {
                "text": context_text,
                "span": (context_start, context_end),
                "candidate": candidate,
                "timestamp": timestamp,
            }
        )

        if len(batch_queue) >= BATCH_SIZE:
            flush_batch(batch_queue)
            batch_queue = []
            batches_since_save += 1

            # Incremental save
            if batches_since_save >= SAVE_EVERY:
                save_checkpoint(
                    candidate_slices, corpus_anchor_sums, stats, current_row
                )
                batches_since_save = 0
                print(
                    f"  [CHECKPOINT] Row {current_row} | "
                    f"Embedded: {stats['embedded']} | "
                    f"Skipped: {stats['skipped_no_tokens']}"
                )

# Flush final batch
flush_batch(batch_queue)
batch_queue = []

print("\nProcessing complete.")
print(f"  Rows read:             {stats['rows_read']}")
print(f"  Spans found:           {stats['span_found']}")
print(f"  Successfully embedded: {stats['embedded']}")
print(f"  Skipped (no span):     {stats['skipped_no_span']}")
print(f"  Skipped (bad offsets): {stats['skipped_invalid_offsets']}")
print(f"  Skipped (context):     {stats['skipped_context_build']}")
print(f"  Skipped (no tokens):   {stats['skipped_no_tokens']}")
print(f"  Legacy span fallback:  {stats['legacy_span_fallback']}")


# ──────────────────────────────────────────────────────────────
# SAVE FINAL RESULTS
# ──────────────────────────────────────────────────────────────
output = {
    "embedding_config": embedding_config(),
    "data": {},
    "anchor_template_embeddings": {
        k: v.tolist() for k, v in anchor_template_embeddings.items()
    },
    "corpus_anchor_sums": {
        anchor: {"sum": data["sum"].tolist(), "count": data["count"]}
        for anchor, data in corpus_anchor_sums.items()
    },
    "word_groups": WORD_GROUPS,
    "stats": stats,
}

for cand, slices in candidate_slices.items():
    output["data"][cand] = {}
    for month, entry in slices.items():
        output["data"][cand][month] = {
            "sum_embedding": entry["sum_embedding"].tolist(),
            "count": entry["count"],
        }

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f)

# Clean up checkpoint now that final output is saved
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint cleaned up.")

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Unique words in this file: {len(candidate_slices)}")
corpus_with_data = sum(1 for v in corpus_anchor_sums.values() if v["count"] > 0)
print(f"Anchors with corpus data: {corpus_with_data}/{len(ANCHORS)}")
