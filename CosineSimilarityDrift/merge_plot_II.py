import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
from transformers import BertTokenizer, BertModel

# 0. Initialize BERT to generate the standard "drug" reference
print("Loading BERT for reference embedding generation...", end=" ", flush=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_reference_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the mean of the last hidden state for the standard "drug" concept
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Create the standard comparison target
DRUG_REF = get_reference_vector("drug").reshape(1, -1)
print("Success. Target set to 'drug'.")

os.makedirs("plotsII", exist_ok=True)
all_data = defaultdict(lambda: defaultdict(list))

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
                for t_slice, stats in slices.items():
                    if t_slice not in all_data[cand]:
                        all_data[cand][t_slice] = {
                            "total_sum": np.zeros(768), 
                            "total_count": 0
                        }
                    
                    all_data[cand][t_slice]["total_sum"] += np.array(stats["sum_embedding"])
                    all_data[cand][t_slice]["total_count"] += stats["count"]
        print("Success.")
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] {f_path} is corrupted and will be skipped.")
        print(f"Details: {e}")
        continue
    except Exception as e:
        print(f"\n[ERROR] Unexpected error with {f_path}: {e}")
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
        data = slices[t_slice]
        # Calculate mean of all embeddings for this word in this month
        try:
            true_mean = (data["total_sum"] / data["total_count"]).reshape(1, -1)
            
            # CORE CHANGE: Comparing directly to the "drug" vector
            sim = cosine_similarity(true_mean, DRUG_REF)[0][0]
            
            similarities.append(sim)
            valid_times.append(t_slice)
        except Exception as e:
            print(f"Error processing {candidate} for {t_slice}: {e}")
            continue
    
    if len(valid_times) < 2: 
        continue

    plt.figure(figsize=(12, 6))
    plt.plot(valid_times, similarities, marker='o', linestyle='-', color='teal')
    plt.title(f"Semantic Drift: {candidate} (Relative to 'drug')")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Cosine Similarity to 'drug'")
    plt.tight_layout()
    
    # Clean name for saving
    safe_name = candidate.replace(' ', '_').replace('/', '_')
    plt.savefig(f"plotsII/{safe_name}.png")
    plt.close()

print(f"Done! Plots are in the /plots folder.")