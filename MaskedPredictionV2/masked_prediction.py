"""
Masked Prediction Shift Analysis for Emerging Euphemism Detection.

For each word in each time period, masks it in its corpus contexts and
checks BERT's top-k predictions for taboo-related tokens. If the taboo
hit rate increases over time, the word's usage is shifting toward drug
meaning — a signal of emerging euphemistic usage.

KEY DIFFERENCE FROM ZHU ET AL: We track this rate OVER TIME. A word that
always triggers drug predictions (high but flat) is not emerging — it was
always drug-related. A word that goes from low to high IS emerging. The
temporal dimension eliminates false positives from static analysis.

Works with the same JSONL format as the cosine similarity pipeline:
  {"word": "flower", "sentence": "...", "timestamp": "2015-06-01T...", "category": "drug_words"}

Three-group comparison (same as cosine pipeline):
  - Euphemism candidates: expect INCREASING taboo hit rate
  - Established euphemisms: expect STABLE HIGH taboo hit rate
  - Comparison words: expect FLAT LOW taboo hit rate

Usage:
  # Single file test
  python masked_prediction.py --data-dir ../matches --task-id 0

  # SLURM array (one task per JSONL file)
  sbatch --array=0-99 run_masked.sh

Output: masked_results_{task_id}.json (partial), then merge with merge_masked.py
"""

import json
import re
import os
import glob
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "../matches"
MODEL_NAME = "bert-base-uncased"
TOP_K = 15              # Number of predictions to check per masked position
SAVE_EVERY = 5000       # Checkpoint every N rows

OUTPUT_FILE = f"masked_results_{task_id}.json"
CHECKPOINT_FILE = f"masked_checkpoint_{task_id}.json"

# ──────────────────────────────────────────────────────────────
# WORD GROUPS (identical to cosine pipeline)
# ──────────────────────────────────────────────────────────────
ANCHORS = [
    "cathinones", "cocaine", "heroin", "marijuana", "fentanyl",
    "methamphetamine", "meth", "amphetamine", "oxycodone", "xanax",
    "adderall", "mdma", "ecstasy", "lsd", "pcp", "codeine", "ketamine",
    "bath salts",
]

ESTABLISHED_EUPHEMISMS = ["molly", "coke", "crystal", "ping"]

EUPHEMISM_CANDIDATES = [
    "study", "skittles", "zing", "zaza", "christina", "flakka",
    "yart", "zoom", "fein", "flower", "yimyom", "fenty", "pressed",
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

# ──────────────────────────────────────────────────────────────
# TABOO VOCABULARY
# Words that indicate drug-related meaning when they appear in
# BERT's top-k predictions for a masked position.
# This includes the anchor drugs plus common drug slang/terms.
# ──────────────────────────────────────────────────────────────
TABOO_VOCAB = set()
# Add all anchor words and their subwords
for word in ANCHORS:
    TABOO_VOCAB.add(word.lower())
    for part in re.split(r"[\s\-]+", word.lower()):
        if len(part) > 2:
            TABOO_VOCAB.add(part)

# Add common drug-related terms that BERT might predict
EXTRA_TABOO = [
    "drug", "drugs", "narcotic", "narcotics", "substance",
    "cannabis", "weed", "pot", "hash", "hashish",
    "opiate", "opiates", "opioid", "opioids",
    "stimulant", "depressant", "hallucinogen",
    "overdose", "addiction", "addictive", "addicted",
    "snort", "inject", "smoke", "inhale",
    "dealer", "trafficking", "cartel", "dea",
    "stash", "dose", "dosage", "high", "stoned",
    "crack", "crystal", "ice", "speed",
    "pill", "pills", "tablet", "capsule",
    "powder", "rock", "rocks",
    "syringe", "needle","percocet", "suboxone", "klonopin", "ritalin", "concerta",
    "vicodin", "norco", "dilaudid", "rohypnol", "oxycontin",
    "roxicodone", "ketalar", "daytrana", "morphine", "opium",
    "ghb", "peyote", "mescaline", "psilocybin", "mushrooms",
    "steroids", "khat", "hash", "hydrocodone", "hydromorphone",
    "promethazine", "buprenorphine", "naloxone", "clonazepam",
    "alprazolam", "flunitrazepam", "methylphenidate",
]
for w in EXTRA_TABOO:
    TABOO_VOCAB.add(w.lower())

print(f"Taboo vocabulary: {len(TABOO_VOCAB)} terms")


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def get_month_slice(timestamp):
    """Extract YYYY-MM from ISO timestamp (same as cosine pipeline)."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        return dt.strftime("%Y-%m")
    except Exception:
        return "unknown"


def find_span(sentence, word):
    """Find character-level (start, end) of word in sentence."""
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence, re.IGNORECASE)
    return (match.start(), match.end()) if match else (None, None)


# ──────────────────────────────────────────────────────────────
# CHECKPOINT SAVE / LOAD
# ──────────────────────────────────────────────────────────────
def save_checkpoint(accumulator, stats, rows_processed):
    """Atomic checkpoint save."""
    data = {
        "rows_processed": rows_processed,
        "stats": stats,
        "accumulator": {},
    }
    for word, months in accumulator.items():
        data["accumulator"][word] = {}
        for month, counts in months.items():
            data["accumulator"][word][month] = {
                "hits": counts["hits"],
                "total": counts["total"],
            }

    tmp_file = CHECKPOINT_FILE + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(data, f)
    os.replace(tmp_file, CHECKPOINT_FILE)


def load_checkpoint():
    """Load checkpoint if it exists."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    print(f"Found checkpoint: {CHECKPOINT_FILE}")
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)

        accumulator = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))
        for word, months in data["accumulator"].items():
            for month, counts in months.items():
                accumulator[word][month]["hits"] = counts["hits"]
                accumulator[word][month]["total"] = counts["total"]

        rows_processed = data["rows_processed"]
        stats = data["stats"]
        print(f"  Resuming from row {rows_processed}")
        print(f"  Stats so far: processed={stats['processed']}, hits={stats['total_hits']}")
        return accumulator, stats, rows_processed

    except Exception as e:
        print(f"  [WARN] Checkpoint corrupted ({e}), starting fresh.")
        return None


# ──────────────────────────────────────────────────────────────
# MODEL SETUP
# ──────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

MASK_ID = tokenizer.mask_token_id


@torch.inference_mode()
def check_taboo_in_predictions(sentence, char_start, char_end):
    """
    Mask the target word in the sentence, get BERT's top-k predictions
    for each masked token position, and check if any taboo words appear.

    Returns: (is_hit, n_taboo_tokens_found)
      - is_hit: True if ANY taboo word appears in predictions
      - n_taboo_tokens_found: count of taboo tokens across all positions
    """
    # Tokenize with offset mapping
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )

    input_ids = enc["input_ids"].to(device).clone()
    offsets = enc["offset_mapping"][0].tolist()

    # Find token positions that overlap with our target character span
    masked_positions = []
    for i, (ts, te) in enumerate(offsets):
        if ts == 0 and te == 0:  # Special tokens
            continue
        if ts >= char_end:
            break
        if te <= char_start:
            continue
        # This token overlaps with the target word
        input_ids[0, i] = MASK_ID
        masked_positions.append(i)

    if not masked_positions:
        return False, 0

    # Get predictions
    logits = model(input_ids).logits[0]  # (seq_len, vocab_size)

    taboo_count = 0
    for pos in masked_positions:
        _, top_indices = torch.topk(logits[pos], k=min(TOP_K, logits.shape[-1]))
        for tid in top_indices.tolist():
            token = tokenizer.decode([tid]).strip().lower()
            if token in TABOO_VOCAB:
                taboo_count += 1

    return taboo_count > 0, taboo_count


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
# LOAD CHECKPOINT OR INITIALIZE
# ──────────────────────────────────────────────────────────────
checkpoint = load_checkpoint()

if checkpoint:
    accumulator, stats, rows_to_skip = checkpoint
else:
    # word -> month -> {hits: int, total: int}
    # hits = number of instances where taboo appeared in top-k
    # total = number of instances checked
    accumulator = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))
    stats = {
        "rows_read": 0,
        "processed": 0,
        "skipped_no_span": 0,
        "total_hits": 0,
    }
    rows_to_skip = 0


# ──────────────────────────────────────────────────────────────
# PROCESS FILE
# ──────────────────────────────────────────────────────────────
current_row = 0
rows_since_save = 0

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
        sentence = row.get("sentence", "")
        word = row.get("word", "")
        timestamp = row.get("timestamp", "")

        if not sentence or not word:
            continue

        # Find the word in the sentence
        start, end = find_span(sentence, word)
        if start is None:
            stats["skipped_no_span"] += 1
            continue

        month = get_month_slice(timestamp)

        # Mask the word and check predictions
        is_hit, taboo_count = check_taboo_in_predictions(sentence, start, end)

        # Accumulate
        accumulator[word][month]["total"] += 1
        if is_hit:
            accumulator[word][month]["hits"] += 1
            stats["total_hits"] += 1
        stats["processed"] += 1

        rows_since_save += 1

        # Periodic checkpoint
        if rows_since_save >= SAVE_EVERY:
            save_checkpoint(accumulator, stats, current_row)
            rows_since_save = 0
            print(
                f"  [CHECKPOINT] Row {current_row} | "
                f"Processed: {stats['processed']} | "
                f"Hits: {stats['total_hits']} | "
                f"Hit rate: {stats['total_hits']/max(1,stats['processed']):.3f}"
            )

print(f"\nProcessing complete.")
print(f"  Rows read:        {stats['rows_read']}")
print(f"  Processed:        {stats['processed']}")
print(f"  Skipped (no span): {stats['skipped_no_span']}")
print(f"  Total hits:       {stats['total_hits']}")
print(f"  Overall hit rate: {stats['total_hits']/max(1,stats['processed']):.4f}")


# ──────────────────────────────────────────────────────────────
# SAVE FINAL RESULTS
# ──────────────────────────────────────────────────────────────
output = {
    "data": {},
    "word_groups": WORD_GROUPS,
    "stats": stats,
    "config": {
        "model": MODEL_NAME,
        "top_k": TOP_K,
        "taboo_vocab_size": len(TABOO_VOCAB),
    },
}

for word, months in accumulator.items():
    output["data"][word] = {}
    for month, counts in months.items():
        output["data"][word][month] = {
            "hits": counts["hits"],
            "total": counts["total"],
        }

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f)

# Clean up checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint cleaned up.")

print(f"\nSaved: {OUTPUT_FILE}")
print(f"Unique words: {len(accumulator)}")