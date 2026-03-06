import pandas as pd
import demoji
import re
import html
import torch
import os
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build
from datetime import datetime

# --- INITIALIZATION ---
# 1. Replace with your actual YouTube API Key
API_KEY = 'AIzaSyAFHTcRHuSEnhlniS7xyjzZT_gXP7oO4Gs' 

# 2. Setup YouTube API and AI Model
print("Loading Machine Learning Model... (This takes a moment on the first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')
youtube = build('youtube', 'v3', developerKey=API_KEY)
demoji.download_codes()
print("Model Loaded. Starting Research Pipeline...")

# --- TABOO TOPICS ---
TABOO_CLUSTERS = {
    "Substances": ["drugs addiction heroin", "cocaine overdose", "alcohol drunk hangover"],
    "Sexuality": ["sex pregnancy prostitution", "pornography masturbation genitals", "orgasm erection"],
    "Mortality": ["death funeral suicide", "kill euthanasia murder"],
    "Medical": ["disease cancer terminal illness", "dementia venereal disease", "mental illness psychotic schizophrenia"],
    "Bodily": ["urinate defecate vomit", "flatulence blood semen", "toilet diarrhea menstruation"],
    "Physical": ["injury wound amputation", "disabled blind deaf", "fat obese ugly bald body odor"],
    "Economic": ["poverty wealth bankrupt", "debt homeless eviction", "fired laid off illiterate expelled"],
    "Conflict": ["crime theft prison", "bribery corruption propaganda", "war bombing civilian death", "torture genocide"],
    "Social": ["racism segregation discrimination", "immigrant refugee deportation", "violence riot assault rape"],
    "Personal": ["anger depression anxiety", "grief divorce abuse", "domestic violence abandonment", "infidelity cheating rejection"]
}

# Flatten list for whole-word checking
ALL_TABOO_WORDS = set([word for sublist in [q.split() for cluster in TABOO_CLUSTERS.values() for q in cluster] for word in sublist])
TABOO_VECTORS = model.encode(list(ALL_TABOO_WORDS), convert_to_tensor=True)

# Basic Stopwords to filter out "the", "and", etc., during Discovery
STOPWORDS = {"the", "and", "this", "that", "with", "from", "they", "have", "been", "would", "about", "their", "there", "what", "some"}

def clean_text(text):
    """Sanitizes text and converts emojis to descriptors."""
    text = html.unescape(text)
    text = demoji.replace_with_desc(text, sep=":")
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    return " ".join(text.split()).lower()

def run_extraction():
    all_comments = []
    
    # --- DATA COLLECTION ---
    for category, queries in TABOO_CLUSTERS.items():
        for query in queries:
            print(f"Collecting: {query}...")
            try:
                search = youtube.search().list(q=query, part="snippet", type="video", maxResults=5).execute()
                for v in search['items']:
                    v_id, v_title = v['id']['videoId'], v['snippet']['title']
                    
                    try:
                        comments = youtube.commentThreads().list(part="snippet", videoId=v_id, maxResults=100, textFormat="plainText").execute()
                        for item in comments['items']:
                            c = item['snippet']['topLevelComment']['snippet']
                            clean = clean_text(c['textDisplay'])
                            if len(clean) > 10:
                                all_comments.append({
                                    'category': category,
                                    'video_title': v_title,
                                    'content': clean,
                                    'timestamp': pd.to_datetime(c['publishedAt']),
                                    'author_id': c.get('authorChannelId', {}).get('value', 'unknown')
                                })
                    except: continue
            except: continue

    df = pd.DataFrame(all_comments)
    if df.empty: return print("No data collected. Check API key/quota.")

    # --- PASS 1: DISCOVERY (2010s & 2020s) ---
    print("Running Pass 1: Discovering Euphemisms in 2010-2026 data...")
    modern_df = df[df['timestamp'].dt.year >= 2010].copy()
    
    # Encode all modern comments at once for speed
    contents = modern_df['content'].tolist()
    content_vecs = model.encode(contents, convert_to_tensor=True, show_progress_bar=True)
    
    discovered_words = set()
    
    for i, vec in enumerate(content_vecs):
        scores = util.cos_sim(vec, TABOO_VECTORS)
        max_score = torch.max(scores).item()
        text = contents[i]
        
        # Whole-word taboo check
        has_taboo = any(re.search(rf'\b{re.escape(t)}\b', text) for t in ALL_TABOO_WORDS)
        
        # If conceptually taboo but words are missing -> Extract new words
        if max_score > 0.45 and not has_taboo:
            # Tokenize and find unique words not in taboo list or stopwords
            words = re.findall(r'\b[a-z]{4,}\b', text) # Only words 4+ letters long
            for w in words:
                if w not in ALL_TABOO_WORDS and w not in STOPWORDS:
                    discovered_words.add(w)

    print(f"Discovered {len(discovered_words)} candidate euphemism words.")

    # --- PASS 2: DETECTION (Everything) ---
    print("Running Pass 2: Tagging all instances across all time periods...")
    df = df.sort_values('timestamp')
    
    is_euphemism_list = []
    first_time_list = []
    seen_euphemisms = set()

    for _, row in df.iterrows():
        text = row['content']
        # Find if any discovered word is in this comment
        found_words = [w for w in discovered_words if re.search(rf'\b{re.escape(w)}\b', text)]
        
        if found_words:
            is_euphemism_list.append(True)
            # Mark 'First Time' if any of these words are new to the memory
            is_new = any(w not in seen_euphemisms for w in found_words)
            first_time_list.append(is_new)
            for w in found_words: seen_euphemisms.add(w)
        else:
            is_euphemism_list.append(False)
            first_time_list.append(False)

    df['is_euphemism'] = is_euphemism_list
    df['is_first_time_appearance'] = first_time_list
    
    # Save results
    df.to_csv("final_euphemism_research.csv", index=False)
    print(f"Success! Saved {len(df)} rows to final_euphemism_research.csv")

# Run the full pipeline
run_extraction()