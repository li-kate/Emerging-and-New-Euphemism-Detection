"""
Per-file BERT embedding script for SLURM array jobs.
Extracts contextual span embeddings for all words (anchors, euphemism
candidates, established euphemisms, and comparison words).

Saves:
  - Per-word, per-month sum embeddings + counts (for averaging in merge)
  - Per-anchor template span embeddings (for per-anchor similarity in merge)
  - Corpus anchor partial sums (for corpus-derived anchor embeddings in merge)
  - Word group labels so the merge script can categorize results

Usage: Run as SLURM array job, one task per JSONL file.
Output: partial_results_{task_id}.json
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
import os
import glob
import re

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../matches"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 256
HIDDEN_DIM = 768

# ──────────────────────────────────────────────────────────────
# WORD GROUPS
# These define the three-way comparison at the heart of the analysis.
# ──────────────────────────────────────────────────────────────

# Direct drug names — these are the taboo anchors.
# Each candidate will be compared against EACH of these individually.
ANCHORS = [
    "cathinones", "cocaine", "heroin", "marijuana", "fentanyl",
    "methamphetamine", "meth", "amphetamine", "oxycodone", "xanax",
    "adderall", "mdma", "ecstasy", "lsd", "pcp", "codeine", "ketamine",
    "bath salts",
]

# Established euphemisms — should show STABLE HIGH similarity to their anchor.
# These are your positive controls / sanity checks.
ESTABLISHED_EUPHEMISMS = ["molly", "coke", "crystal", "ping"]

# Euphemism candidates — the words you're actually studying.
# These are what you expect to show INCREASING similarity over time.
EUPHEMISM_CANDIDATES = [
    "study", "skittles", "zing", "zaza", "christina", "flakka",
    "yart", "zoom", "fein", "flower", "yimyom", "fenty", "pressed",
]

# Comparison words — drug-adjacent but NOT euphemistic.
# Should show FLAT, LOW similarity. Negative controls.
COMPARISON_WORDS = ["needle", "pharmacy", "prescription", "overdose"]

# Build a lookup: word -> group label
WORD_GROUPS = {}
for w in ANCHORS:
    WORD_GROUPS[w.lower()] = "anchor"
for w in ESTABLISHED_EUPHEMISMS:
    WORD_GROUPS[w.lower()] = "established_euphemism"
for w in EUPHEMISM_CANDIDATES:
    WORD_GROUPS[w.lower()] = "euphemism_candidate"
for w in COMPARISON_WORDS:
    WORD_GROUPS[w.lower()] = "comparison"

# Templates for building per-anchor reference embeddings (Option A).
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
    """Extract YYYY-MM from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        return dt.strftime("%Y-%m")
    except Exception:
        return "unknown"


def find_span(sentence, word):
    """Find character-level (start, end) of `word` in `sentence`."""
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence, re.IGNORECASE)
    return (match.start(), match.end()) if match else (None, None)


# ──────────────────────────────────────────────────────────────
# MODEL SETUP
# ──────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def extract_span_embedding(sentence, char_start, char_end):
    """
    Get the contextual embedding of a character span within a sentence.
    Returns numpy vector of shape (HIDDEN_DIM,), or None on failure.

    This is the SINGLE source of truth for turning a word-in-context
    into a vector. Candidates, anchors, and controls all use this.
    """
    inputs = tokenizer(
        sentence,
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
        if ts >= char_start and te <= char_end:
            token_indices.append(j)

    if not token_indices:
        return None
    return embeddings[token_indices].mean(dim=0).numpy()


def get_batch_span_embeddings(batch_data):
    """Batch version of span extraction for GPU efficiency."""
    sentences = [d["sentence"] for d in batch_data]

    inputs = tokenizer(
        sentences,
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
            if ts >= char_start and te <= char_end:
                token_indices.append(j)

        if not token_indices:
            results.append(None)
        else:
            results.append(embeddings[i][token_indices].mean(dim=0).numpy())

    return results


# ──────────────────────────────────────────────────────────────
# BUILD PER-ANCHOR TEMPLATE EMBEDDINGS (Option A)
# For each anchor word, we get a span embedding averaged across
# multiple template sentences. These are FIXED reference vectors.
# ──────────────────────────────────────────────────────────────
print("Building per-anchor template embeddings (Option A)...")
anchor_template_embeddings = {}  # anchor_word -> numpy vector

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

print(f"Template embeddings for {len(anchor_template_embeddings)}/{len(ANCHORS)} anchors.\n")


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
# PROCESS FILE
# ──────────────────────────────────────────────────────────────
# Accumulator: word -> month -> {sum_embedding, count}
candidate_slices = defaultdict(lambda: defaultdict(lambda: {
    "sum_embedding": np.zeros(HIDDEN_DIM),
    "count": 0,
}))

# Option B: per-anchor corpus embedding accumulators
# anchor_word -> {sum, count}
corpus_anchor_sums = {
    anchor: {"sum": np.zeros(HIDDEN_DIM), "count": 0}
    for anchor in ANCHORS
}
# Build a fast lowercase lookup for anchor matching
anchor_lower_map = {a.lower(): a for a in ANCHORS}

batch_queue = []
stats = {
    "rows_read": 0, "span_found": 0, "embedded": 0,
    "skipped_no_span": 0, "skipped_no_tokens": 0,
}


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

            # Track monthly embeddings for ALL words
            entry = candidate_slices[item["candidate"]][month]
            entry["sum_embedding"] += emb
            entry["count"] += 1
            stats["embedded"] += 1

            # If this word is an anchor, also accumulate for corpus centroid
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
        stats["rows_read"] += 1

        row = json.loads(line)
        sentence = row.get("sentence", "")
        candidate = row.get("word", "")
        timestamp = row.get("timestamp", "")

        if not sentence or not candidate:
            continue

        start, end = find_span(sentence, candidate)
        if start is None:
            stats["skipped_no_span"] += 1
            continue

        stats["span_found"] += 1

        batch_queue.append({
            "sentence": sentence,
            "span": (start, end),
            "candidate": candidate,
            "timestamp": timestamp,
        })

        if len(batch_queue) >= BATCH_SIZE:
            flush_batch(batch_queue)
            batch_queue = []

            if stats["rows_read"] % 10000 == 0:
                print(
                    f"  Rows: {stats['rows_read']} | "
                    f"Embedded: {stats['embedded']} | "
                    f"Skipped: {stats['skipped_no_tokens']}"
                )

# Flush final batch
flush_batch(batch_queue)
batch_queue = []

print(f"\nProcessing complete.")
print(f"  Rows read:             {stats['rows_read']}")
print(f"  Spans found:           {stats['span_found']}")
print(f"  Successfully embedded: {stats['embedded']}")
print(f"  Skipped (no span):     {stats['skipped_no_span']}")
print(f"  Skipped (no tokens):   {stats['skipped_no_tokens']}")


# ──────────────────────────────────────────────────────────────
# SAVE PARTIAL RESULTS
# ──────────────────────────────────────────────────────────────
output = {
    # Per-word, per-month sum embeddings
    "data": {},
    # Option A: per-anchor template embeddings (fixed, same across files)
    "anchor_template_embeddings": {
        k: v.tolist() for k, v in anchor_template_embeddings.items()
    },
    # Option B: per-anchor corpus sums (to be averaged in merge)
    "corpus_anchor_sums": {
        anchor: {
            "sum": data["sum"].tolist(),
            "count": data["count"],
        }
        for anchor, data in corpus_anchor_sums.items()
    },
    # Word group labels
    "word_groups": WORD_GROUPS,
    # Processing stats
    "stats": stats,
}

for cand, slices in candidate_slices.items():
    output["data"][cand] = {}
    for month, entry in slices.items():
        output["data"][cand][month] = {
            "sum_embedding": entry["sum_embedding"].tolist(),
            "count": entry["count"],
        }

output_file = f"partial_results_{task_id}.json"
with open(output_file, "w") as f:
    json.dump(output, f)

print(f"\nSaved: {output_file}")
print(f"Unique words in this file: {len(candidate_slices)}")
corpus_with_data = sum(1 for v in corpus_anchor_sums.values() if v["count"] > 0)
print(f"Anchors with corpus data: {corpus_with_data}/{len(ANCHORS)}")