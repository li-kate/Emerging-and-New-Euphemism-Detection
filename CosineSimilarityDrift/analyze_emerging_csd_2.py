# DON'T USE THIS - NOT PARALLELIZED

# this code is for analyzing emerging euphemisms
# uses the cosine similarity drift method
# Pipeline:
#    1. Organize the data from the second by the candidate euphemism, and then by timestamp.
#    2. Use BERT to create contextual embeddings for each row in the data.
#    3. For each candidate, take the mean of all of its embeddings in the time slice (half a year).
#    4. Compare each average embedding to the embedding for the taboo topic it was labeled as.
#    5. Output a graph of how the cosine similarity score changes over time.

# from the second pass, we have rows of potential euphemisms (candidates), the context around the candidates, and
# their corresponding taboo category

# time_slice: monthly for now
# also a method to get half yearly time slices - just switch line 127: slice_key = get_month_slice(row["timestamp"])

import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg') # Required for headless cluster environments
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
import re

os.makedirs("plots", exist_ok=True)

# CHECK DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CHANGE PATH BEFORE RUNNING
DATA_DIR = "../SecondPass/matches/"
print(f"Searching for data in: {os.path.abspath(DATA_DIR)}")
jsonl_files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
print(f"Found {len(jsonl_files)} .jsonl files.")

if not jsonl_files:
    print("WARNING: No files found. Check your DATA_DIR path.")

# Create Monthly Time Slices
def get_month_slice(timestamp):
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        return dt.strftime("%Y-%m")
    except:
        return "unknown"

# Load BERT
MODEL_NAME = "bert-base-uncased"
print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print("Model loaded successfully.")

# Find word span in sentence
def find_span(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence, re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None, None

# Batch Embedding Function
def get_batch_phrase_embeddings(batch_data):
    sentences = [d['sentence'] for d in batch_data]
    inputs = tokenizer(
        sentences,
        return_offsets_mapping=True,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    
    embeddings = outputs.last_hidden_state.cpu()
    offsets = inputs["offset_mapping"].cpu()
    
    results = []
    for i in range(len(batch_data)):
        start, end = batch_data[i]['span']
        token_indices = []
        for j, (token_start, token_end) in enumerate(offsets[i]):
            if token_start >= start and token_end <= end and (token_start != 0 or token_end != 0):
                token_indices.append(j)
        
        if not token_indices:
            results.append(None)
        else:
            phrase_emb = embeddings[i][token_indices].mean(dim=0).numpy()
            results.append(phrase_emb)
    return results

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# Processing Logic
candidate_slices = defaultdict(lambda: defaultdict(list))
taboo_embeddings = {}
BATCH_SIZE = 16 
batch_queue = []
skipped = 0
total_processed_rows = 0
matches_found = 0

print("Starting streaming processing...")

for file_idx, file_path in enumerate(jsonl_files):
    print(f"Processing file {file_idx + 1}/{len(jsonl_files)}: {os.path.basename(file_path)}")
    with open(file_path, "r") as f:
        for line_idx, line in enumerate(f):
            if not line.strip(): continue
            try:
                total_processed_rows += 1
                row = json.loads(line)
                sentence = row.get("sentence", "")
                candidate = row.get("word", "")
                category = row.get("primary_category", row.get("category"))
                
                if not sentence or not candidate: continue
                
                start, end = find_span(sentence, candidate)
                if start is None: continue
                
                matches_found += 1
                
                # Pre-embed category if new
                if category not in taboo_embeddings:
                    taboo_embeddings[category] = embed_text(category)
                
                batch_queue.append({
                    "sentence": sentence,
                    "span": (start, end),
                    "candidate": candidate,
                    "category": category,
                    "timestamp": row.get("timestamp", "")
                })

                if len(batch_queue) >= BATCH_SIZE:
                    embs = get_batch_phrase_embeddings(batch_queue)
                    for i, emb in enumerate(embs):
                        if emb is None:
                            skipped += 1
                        else:
                            item = batch_queue[i]
                            slice_key = get_month_slice(item["timestamp"])
                            candidate_slices[item["candidate"]][slice_key].append({
                                "embedding": emb,
                                "category": item["category"]
                            })
                    batch_queue = []
                
                # Periodic progress update every 10,000 lines
                if total_processed_rows % 10000 == 0:
                    print(f"Rows read: {total_processed_rows} | Matches found: {matches_found} | Skipped: {skipped}")
                    
            except Exception as e:
                continue

# Finalize last batch
if batch_queue:
    print(f"Finalizing remaining {len(batch_queue)} items in queue...")
    embs = get_batch_phrase_embeddings(batch_queue)
    for i, emb in enumerate(embs):
        if emb is not None:
            item = batch_queue[i]
            slice_key = get_month_slice(item["timestamp"])
            candidate_slices[item["candidate"]][slice_key].append({"embedding": emb, "category": item["category"]})

print(f"Streaming complete. Total rows read: {total_processed_rows}")
print(f"Total valid candidate matches found: {matches_found}")
print(f"Skipped rows (BERT tokenization issues): {skipped}")

# Compute Mean Embeddings + Similarity
print("Computing mean embeddings and cosine similarity...")
results = {}
for candidate, slices in candidate_slices.items():
    results[candidate] = []
    for slice_key in sorted(slices.keys()):
        entries = slices[slice_key]
        mean_embedding = np.array([e["embedding"] for e in entries]).mean(axis=0)
        category = entries[0]["category"]
        taboo_embedding = taboo_embeddings[category]
        
        similarity = cosine_similarity(
            mean_embedding.reshape(1, -1),
            taboo_embedding.reshape(1, -1)
        )[0][0]

        results[candidate].append({
            "time": slice_key,
            "similarity": float(similarity),
            "category": category
        })

# Plot Results
print(f"Generating plots for {len(results)} candidates...")
plots_saved = 0
for candidate, data_points in results.items():
    times = [r["time"] for r in data_points]
    sims = [r["similarity"] for r in data_points]
    if len(times) < 2: 
        continue

    plt.figure(figsize=(10, 5))
    plt.plot(times, sims, marker="o")
    plt.title(f"Semantic Drift: {candidate}")
    plt.xlabel("Time Slice")
    plt.ylabel("Cosine Similarity to Taboo Topic")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    safe_candidate = candidate.replace(" ", "_").replace("/", "_")
    plt.savefig(f"plots/{safe_candidate}.png")
    plt.close()
    plots_saved += 1

with open("cosine_drift_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. {plots_saved} plots generated. Results saved to cosine_drift_results.json")