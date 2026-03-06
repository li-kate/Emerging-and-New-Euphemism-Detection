import pandas as pd
import demoji
import re
import html
import torch
import os
from sentence_transformers import SentenceTransformer, util
from googleapiclient.discovery import build

# --- INITIALIZATION ---
API_KEY = 'AIzaSyAFHTcRHuSEnhlniS7xyjzZT_gXP7oO4Gs'
youtube = build('youtube', 'v3', developerKey=API_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- ORGANIZED TABOO TOPICS ---
# Grouped to maximize search relevancy on YouTube
TABOO_CLUSTERS = {
    "Substances": ["drugs addiction heroin", "cocaine overdose", "alcohol drunk hangover"],
    "Sexuality": ["sex pregnancy prostitution", "pornography masturbation genitals", "orgasm erection"],
    "Mortality": ["death funeral suicide", "kill euthanasia murder"],
    "Medical_Conditions": ["disease cancer terminal illness", "dementia venereal disease", "mental illness psychotic schizophrenia"],
    "Bodily_Functions": ["urinate defecate vomit", "flatulence blood semen", "toilet diarrhea menstruation"],
    "Physical_State": ["injury wound amputation", "disabled blind deaf", "fat obese ugly bald body odor"],
    "Socio_Economic": ["poverty wealth bankrupt", "debt homeless eviction", "fired laid off illiterate expelled"],
    "Crime_Conflict": ["crime theft prison", "bribery corruption propaganda", "war bombing civilian death", "torture genocide"],
    "Social_Issues": ["racism segregation discrimination", "immigrant refugee deportation", "violence riot assault rape"],
    "Personal_Relational": ["anger depression anxiety", "grief divorce abuse", "domestic violence abandonment", "infidelity cheating rejection"]
}

# Flattened list for similarity checking
ALL_TABOO_WORDS = [word for sublist in [q.split() for cluster in TABOO_CLUSTERS.values() for q in cluster] for word in sublist]
TABOO_VECTORS = model.encode(ALL_TABOO_WORDS, convert_to_tensor=True)

def clean_text(text):
    """Sanitizes text and converts emojis for model readability."""
    text = html.unescape(text)
    text = demoji.replace_with_desc(text, sep=":")
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    return " ".join(text.split()).lower()

print("Loading Machine Learning Model... (This may take a minute on the first run)")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded. Starting YouTube Search...")

def get_data():
    """PASS 1: Data Collection & Semantic Scoring"""
    all_comments = []
    
    for category, queries in TABOO_CLUSTERS.items():
        for query in queries:
            print(f"Searching: {category} -> {query}")
            try:
                # Search for videos
                search_req = youtube.search().list(
                    q=query, part="snippet", type="video", maxResults=5
                ).execute()
                
                for v in search_req['items']:
                    v_id = v['id']['videoId']
                    v_title = v['snippet']['title']
                    
                    # Fetch comments
                    comment_req = youtube.commentThreads().list(
                        part="snippet", videoId=v_id, maxResults=100, textFormat="plainText"
                    ).execute()
                    
                    for item in comment_req['items']:
                        c = item['snippet']['topLevelComment']['snippet']
                        raw_text = c['textDisplay']
                        clean = clean_text(raw_text)
                        
                        if len(clean) > 10:
                            # Calculate Similarity
                            comment_vec = model.encode(clean, convert_to_tensor=True)
                            scores = util.cos_sim(comment_vec, TABOO_VECTORS)
                            max_score = torch.max(scores).item()
                            
                            # Check if it's a euphemism (High similarity but no taboo words used)
                            has_taboo = any(t in clean for t in ALL_TABOO_WORDS)
                            is_euphemism = (max_score > 0.6) and not has_taboo
                            
                            all_comments.append({
                                'category': category,
                                'video_title': v_title,
                                'content': clean,
                                'similarity_score': round(max_score, 4),
                                'is_euphemism': is_euphemism,
                                'timestamp': c['publishedAt'],
                                'author_id': c.get('authorChannelId', {}).get('value', 'unknown')
                            })
            except Exception as e:
                print(f"Error skipping: {e}")
                continue
                
    return all_comments

def process_results(data_list):
    """PASS 2: Chronological Analysis & First-Time Detection"""
    df = pd.DataFrame(data_list)
    if df.empty: return df
    
    # Sort by time to track origin
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Track the first time a specific semantic pattern (euphemism) appears
    seen_content = set()
    first_time_flags = []
    
    for _, row in df.iterrows():
        if row['is_euphemism'] and row['content'] not in seen_content:
            first_time_flags.append(True)
            seen_content.add(row['content'])
        else:
            first_time_flags.append(False)
            
    df['is_first_time_appearance'] = first_time_flags
    return df

# --- EXECUTION ---
raw_results = get_data()
final_df = process_results(raw_results)
final_df.to_csv("taboo_research_dataset.csv", index=False)
print(f"Dataset complete. Saved {len(final_df)} rows to taboo_research_dataset.csv")