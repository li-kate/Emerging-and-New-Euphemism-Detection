import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import re

# GET SLURM ARRAY INDEX
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# SETTINGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../SecondPass/matches"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 256 # Increased for H100 efficiency

# 1. Identify the specific file for this task
jsonl_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl")))
if task_id >= len(jsonl_files):
    print(f"Task ID {task_id} exceeds file count. Exiting.")
    exit()

target_file = jsonl_files[task_id]
print(f"Task {task_id} processing: {os.path.basename(target_file)}")

# 2. Setup BERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def get_month_slice(timestamp):
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        return dt.strftime("%Y-%m")
    except: return "unknown"

def find_span(sentence, word):
    pattern = r"\b" + re.escape(word) + r"\b"
    match = re.search(pattern, sentence, re.IGNORECASE)
    return (match.start(), match.end()) if match else (None, None)

def get_batch_phrase_embeddings(batch_data):
    sentences = [d['sentence'] for d in batch_data]
    inputs = tokenizer(sentences, return_offsets_mapping=True, return_tensors="pt", 
                       padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.cpu()
    offsets = inputs["offset_mapping"].cpu()
    results = []
    
    for i in range(len(batch_data)):
        start, end = batch_data[i]['span']
        token_indices = [j for j, (ts, te) in enumerate(offsets[i]) 
                         if ts >= start and te <= end and (ts != 0 or te != 0)]
        if not token_indices:
            results.append(None)
        else:
            results.append(embeddings[i][token_indices].mean(dim=0).numpy())
    return results

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

# 3. Processing
candidate_slices = defaultdict(lambda: defaultdict(list))
taboo_embeddings = {}
batch_queue = []

with open(target_file, "r") as f:
    for line in f:
        if not line.strip(): continue
        row = json.loads(line)
        sentence, candidate = row.get("sentence", ""), row.get("word", "")
        category = row.get("primary_category", row.get("category"))
        
        if not sentence or not candidate: continue
        start, end = find_span(sentence, candidate)
        if start is None: continue
        
        if category not in taboo_embeddings:
            taboo_embeddings[category] = embed_text(category)
            
        batch_queue.append({
            "sentence": sentence, "span": (start, end), "candidate": candidate,
            "category": category, "timestamp": row.get("timestamp", "")
        })

        if len(batch_queue) >= BATCH_SIZE:
            embs = get_batch_phrase_embeddings(batch_queue)
            for i, emb in enumerate(embs):
                if emb is not None:
                    item = batch_queue[i]
                    slice_key = get_month_slice(item["timestamp"])
                    candidate_slices[item["candidate"]][slice_key].append({
                        "embedding": emb.tolist(), # Convert to list for JSON
                        "category": item["category"]
                    })
            batch_queue = []

# Finalize and Save
# We save raw embeddings to be averaged in the merge script
summarized_data = defaultdict(dict)
for cand, slices in candidate_slices.items():
    for t_slice, entries in slices.items():
        # Stack all embeddings for this month in this file
        embs = np.array([e["embedding"] for e in entries])
        
        summarized_data[cand][t_slice] = {
            "sum_embedding": np.sum(embs, axis=0).tolist(), # Sum of all vectors
            "count": int(len(entries)),                    # Number of vectors
            "category": entries[0]["category"]
        }

output_file = f"partial_results_{task_id}.json"
with open(output_file, "w") as f:
    json.dump({"data": summarized_data, "taboos": {k: v.tolist() for k, v in taboo_embeddings.items()}}, f)