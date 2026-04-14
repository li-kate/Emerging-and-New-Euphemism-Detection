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
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob

os.makedirs("plots", exist_ok=True)

# CHECK DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CHANGE PATH BEFORE RUNNING
DATA_DIR = "../SecondPass/matches/"
jsonl_files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))

# Load Data
data = []
for file_path in jsonl_files:
    print(f"Loading {file_path}...")
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

if not data:
    print("No data found! Check your DATA_DIR path.")
    exit()

# Sort by candidate then timestamp
data = sorted(
    data,
    key=lambda x: (x["word"], x["timestamp"])
)

# Create Half-Year Time Slices
def get_half_year(timestamp):
    dt = datetime.fromisoformat(timestamp.replace("Z", ""))
    year = dt.year
    half = 1 if dt.month <= 6 else 2
    return f"{year}_H{half}"

# Create Monthly Time Slices
def get_month_slice(timestamp):
    dt = datetime.fromisoformat(timestamp.replace("Z", ""))
    # Format as YYYY-MM to ensure alphabetical sorting is also chronological
    return dt.strftime("%Y-%m")

# Load BERT
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device) # Moved to GPU

model.eval()

# Get Phrase Embedding
def get_phrase_embedding(sentence, start, end):
    inputs = tokenizer(
        sentence,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True
    ).to(device) # Moved to GPU

    offsets = inputs["offset_mapping"][0].cpu() # Move offsets back for indexing logic

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    # Use .cpu() before converting to numpy
    embeddings = outputs.last_hidden_state[0].cpu()

    token_indices = []
    for i, (token_start, token_end) in enumerate(offsets):
        if token_start >= start and token_end <= end:
            token_indices.append(i)

    if not token_indices:
        return None

    phrase_embedding = embeddings[token_indices].mean(dim=0)
    return phrase_embedding.numpy()

# Embed Taboo Topics
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

taboo_embeddings = {}
for row in data:
    category = row["primary_category"]
    if category not in taboo_embeddings:
        taboo_embeddings[category] = embed_text(category)

# Group by Candidate + Time Slice
candidate_slices = defaultdict(lambda: defaultdict(list))

for row in tqdm(data):
    embedding = get_phrase_embedding(
        row["sentence"],
        row["start"],
        row["end"]
    )
    if embedding is None:
        continue

    candidate = row["word"]
    slice_key = get_month_slice(row["timestamp"])

    candidate_slices[candidate][slice_key].append({
        "embedding": embedding,
        "category": row["category"]
    })

# Compute Mean Embeddings + Similarity
results = {}
for candidate in candidate_slices:
    results[candidate] = []
    for slice_key in sorted(candidate_slices[candidate].keys()):
        entries = candidate_slices[candidate][slice_key]
        embeddings = np.array([e["embedding"] for e in entries])
        mean_embedding = embeddings.mean(axis=0)
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
for candidate in results:
    times = [r["time"] for r in results[candidate]]
    sims = [r["similarity"] for r in results[candidate]]
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

# Save Results
with open("cosine_drift_results.json", "w") as f:
    json.dump(results, f, indent=2)