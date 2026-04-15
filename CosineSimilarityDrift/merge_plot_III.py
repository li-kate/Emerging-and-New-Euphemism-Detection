import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
from transformers import BertTokenizer, BertModel

# 0. Initialize BERT
print("Loading BERT for centroid generation...", end=" ", flush=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_bert_vector(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean of the last hidden state
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# 1. Create a "Centroid" Reference Vector
# We use a variety of unambiguous drugs to define the "Drug Zone"
anchor_words = ["cathinones", "cocaine", "heroin", "marijuana", "fentanyl", "methamphetamine", "meth", "amphetamine", "oxycodone", "xanax", "adderall", "mdma", "ecstasy", "lsd", "pcp", "codeine", "ketamine", "bath salts", "molly", "coke", "crystal", "ping"]
print(f"\nCalculating centroid using anchors: {', '.join(anchor_words)}...")

anchor_vectors = []
for word in anchor_words:
    anchor_vectors.append(get_bert_vector(word))

# The DRUG_REF is now the average of all these concepts
DRUG_REF = np.mean(anchor_vectors, axis=0).reshape(1, -1)
print("Centroid successfully calculated.")

# 2. Setup output directory
PLOT_DIR = "plots_III"
os.makedirs(PLOT_DIR, exist_ok=True)
all_data = defaultdict(lambda: defaultdict(list))

# 3. Aggregate all partial files
partial_files = sorted(glob.glob("partial_results_*.json"))
print(f"Found {len(partial_files)} partial files. Starting merge...")

for f_path in partial_files:
    print(f"Reading {f_path}...", end=" ", flush=True)
    try:
        with open(f_path, 'r') as f:
            content = json.load(f)
            for cand, slices in content["data"].items():
                for t_slice, stats in slices.items():
                    if t_slice not in all_data[cand]:
                        all_data[cand][t_slice] = {"total_sum": np.zeros(768), "total_count": 0}
                    all_data[cand][t_slice]["total_sum"] += np.array(stats["sum_embedding"])
                    all_data[cand][t_slice]["total_count"] += stats["count"]
        print("Success.")
    except Exception as e:
        print(f"\n[ERROR] Skipping {f_path}: {e}")

# 4. Compute Mean & Plot
print(f"Generating plots for {len(all_data)} candidates...")
for candidate, slices in all_data.items():
    time_keys = sorted(slices.keys())
    if len(time_keys) < 2: continue 
    
    similarities = []
    valid_times = []
    
    for t_slice in time_keys:
        data = slices[t_slice]
        try:
            # Monthly mean embedding of the word being studied
            true_mean = (data["total_sum"] / data["total_count"]).reshape(1, -1)
            
            # Compare mean word usage to the Centroid
            sim = cosine_similarity(true_mean, DRUG_REF)[0][0]
            
            similarities.append(sim)
            valid_times.append(t_slice)
        except:
            continue
    
    if not valid_times: continue

    plt.figure(figsize=(12, 6))
    plt.plot(valid_times, similarities, marker='o', linestyle='-', color='royalblue')
    plt.title(f"Semantic Drift: {candidate} (Relative to Drug Centroid)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylabel("Cosine Similarity to Drug Concept")
    plt.tight_layout()
    
    safe_name = candidate.replace(' ', '_').replace('/', '_')
    plt.savefig(f"{PLOT_DIR}/{safe_name}.png")
    plt.close()

print(f"Done! All plots saved to the /{PLOT_DIR} folder.")