import pandas as pd
import demoji
import re
import html
from googleapiclient.discovery import build

# --- CONFIGURATION ---
API_KEY = '<API_KEY>' 
DAY_NUMBER = 5  # Change this (1, 2, 3, 4, 5) to process different words each day
COMMENTS_PER_VIDEO = 500 
VIDEOS_PER_WORD = 30

# --- TABOO WORDS LIST ---
ALL_TABOO_WORDS_FULL = [
    "drugs", "addiction", "heroin", "cocaine", "overdose", "alcohol", "drunk", "hangover",
    "sex", "pregnancy", "prostitution", "pornography", "masturbation", "genitals", "orgasm",
    "erection", "death", "funeral", "suicide", "kill", "euthanasia", "disease", "cancer",
    "terminal illness", "dementia", "venereal disease", "menstruation", "aging", "elderly",
    "urinate", "defecate", "vomit", "flatulence", "blood", "semen", "toilet", "diarrhea",
    "injury", "wound", "amputation", "stupid", "ignorant", "anger", "depression", "anxiety",
    "grief", "homosexual", "transgender", "divorce", "abuse", "domestic violence",
    "abandonment", "infidelity", "poverty", "wealth", "bankrupt", "debt", "homeless",
    "eviction", "fired", "laid off", "illiterate", "expelled", "crime", "theft", "murder",
    "prison", "bribery", "corruption", "propaganda", "censorship", "dictator", "lying",
    "war", "bombing", "civilian death", "torture", "genocide", "racism", "segregation",
    "discrimination", "immigrant", "refugee", "deportation", "violence", "riot", "assault",
    "rape", "surveillance", "spying", "pollution", "extinction", "sin", "cheating",
    "disabled", "blind", "deaf", "mental illness", "psychotic", "schizophrenia", "god",
    "hell", "damn", "fat", "obese", "ugly", "bald", "body odor", "fuck", "shit", "ass",
    "lie", "plastic surgery", "rejection"
]

# 5-Day Split Logic (approx 23 words per day)
WORDS_PER_DAY = 23
start_idx = (DAY_NUMBER - 1) * WORDS_PER_DAY
end_idx = start_idx + WORDS_PER_DAY
CURRENT_BATCH = ALL_TABOO_WORDS_FULL[start_idx:end_idx]

# --- INITIALIZATION ---
print(f"Starting Day {DAY_NUMBER} Research for: {CURRENT_BATCH}")
youtube = build('youtube', 'v3', developerKey=API_KEY)

def clean_simple(text):
    """Basic cleaning for checking word presence."""
    text = html.unescape(text)
    return text.lower()

def get_all_comments(video_id, max_results):
    comments = []
    nextPageToken = None
    while len(comments) < max_results:
        try:
            res = youtube.commentThreads().list(
                part="snippet", 
                videoId=video_id, 
                maxResults=min(100, max_results - len(comments)),
                pageToken=nextPageToken,
                textFormat="plainText"
            ).execute()
            
            for item in res['items']:
                c = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'text': c['textDisplay'],
                    'timestamp': c['publishedAt']
                })
            
            nextPageToken = res.get('nextPageToken')
            if not nextPageToken: break
        except: break
    return comments

def run_extraction():
    final_data = []

    for query in CURRENT_BATCH:
        print(f"Searching YouTube for: {query}...")
        try:
            # Search for videos related to the taboo word
            search = youtube.search().list(q=query, part="snippet", type="video", maxResults=VIDEOS_PER_WORD).execute()
            
            for v in search['items']:
                v_id = v['id']['videoId']
                video_url = f"https://www.youtube.com/watch?v={v_id}"
                
                raw_comments = get_all_comments(v_id, COMMENTS_PER_VIDEO)
                
                for c_data in raw_comments:
                    original_text = c_data['text'].replace('\n', ' ')
                    clean = clean_simple(original_text)
                    
                    # LOGIC: We want the context, but we are looking for EUPHEMISMS.
                    # Therefore, we collect comments from these videos that DON'T 
                    # necessarily use the raw taboo word, but are in the discussion.
                    has_taboo = any(re.search(rf'\b{re.escape(t)}\b', clean) for t in [query])
                    
                    # We save the comment regardless, as it provides the 'context' 
                    # you need for your research.
                    final_data.append({
                        'context': original_text,
                        'timestamp': c_data['timestamp'],
                        'source_url': video_url,
                        'source': 'Youtube'
                    })
        except Exception as e:
            print(f"Error processing {query}: {e}")
            continue

    if not final_data:
        return print("No data collected. Check API key or Query batch.")

    # Save to CSV
    df = pd.DataFrame(final_data)
    # Remove exact duplicates if multiple searches find the same video/comment
    df = df.drop_duplicates(subset=['context', 'source_url'])
    
    filename = f"euphemism_results_day_{DAY_NUMBER}.csv"
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Success! {len(df)} rows saved to {filename}")

if __name__ == "__main__":
    run_extraction()
