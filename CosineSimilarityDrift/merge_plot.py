# USE THIS AFTER PARALLELIZED SCRIPT - MERGES ALL DATA FROM SEPARATE ARRAYS AND THEN PLOTS FOR EACH WORD


import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os

os.makedirs("plots", exist_ok=True)
all_data = defaultdict(lambda: defaultdict(list))
all_taboos = {}

# 1. Aggregate all partial files
partial_files = sorted(glob.glob("partial_results_*.json"))
print(f"Found {len(partial_files)} partial files. Starting merge...")

for f_path in partial_files:
    print(f"Reading {f_path}...", end=" ", flush=True)
    try:
        with open(f_path, 'r') as f:
            content = json.load(f)
            
            # Merge embeddings
            for cand, slices in content["data"].items():
                for t_slice, entries in slices.items():
                    all_data[cand][t_slice].extend(entries)
            
            # Merge taboo reference embeddings
            for cat, emb in content["taboos"].items():
                all_taboos[cat] = np.array(emb)
        print("Success.")
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] {f_path} is corrupted and will be skipped.")
        print(f"Details: {e}")
        continue

if not all_data:
    print("No data was successfully loaded. Check your partial_results files.")
    exit()

# 2. Compute Mean & Plot
print(f"Generating plots for {len(all_data)} candidates...")
for candidate, slices in all_data.items():
    time_keys = sorted(slices.keys())
    if len(time_keys) < 2: 
        continue 
    
    similarities = []
    valid_times = []
    
    for t_slice in time_keys:
        entries = slices[t_slice]
        # Calculate mean of all embeddings for this word in this month
        try:
            embs = np.array([e["embedding"] for e in entries])
            mean_emb = np.mean(embs, axis=0).reshape(1, -1)
            
            category = entries[0]["category"]
            taboo_emb = all_taboos[category].reshape(1, -1)
            
            sim = cosine_similarity(mean_emb, taboo_emb)[0][0]
            similarities.append(sim)
            valid_times.append(t_slice)
        except Exception as e:
            print(f"Error processing {candidate} for {t_slice}: {e}")
            continue
    
    if len(valid_times) < 2: continue

    plt.figure(figsize=(12, 6))
    plt.plot(valid_times, similarities, marker='o', linestyle='-', color='teal')
    plt.title(f"Semantic Drift: {candidate}")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Cosine Similarity to Taboo Topic")
    plt.tight_layout()
    
    safe_name = candidate.replace(' ', '_').replace('/', '_')
    plt.savefig(f"plots/{safe_name}.png")
    plt.close()

print(f"Done! Plots are in the /plots folder.")