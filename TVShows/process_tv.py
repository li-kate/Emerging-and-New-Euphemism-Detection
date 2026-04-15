# https://huggingface.co/datasets/sedthh/tv_dialogue/viewer/default/train?p=13
import argparse, re, os, json, ast, dateparser, time
import tmdbsimple as tmdb
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
tmdb.API_KEY = 'cead083b0fad60a30f64c99a717149ab'
CACHE_FILE = "tv_metadata_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_airdate_fallback(show_name, ep_title, cache):
    """Deep search TMDB for the EXACT episode air date."""
    if not show_name or show_name == "Unknown": return None
    
    # Strip parentheses/years for the search query
    clean_query = re.sub(r'\(.*?\)', '', show_name).strip()
    
    cache_key = f"{clean_query}||{ep_title}".lower()
    if cache_key in cache: return cache[cache_key]
    
    try:
        time.sleep(0.3) 
        search = tmdb.Search()
        show_results = search.tv(query=clean_query)['results']
        if not show_results: return None
        
        show_id = show_results[0]['id']
        show_detail = tmdb.TV(show_id).info()
        num_seasons = show_detail.get('number_of_seasons', 1)

        for s_num in range(1, num_seasons + 1):
            season = tmdb.TV_Seasons(show_id, s_num).info()
            for ep in season['episodes']:
                s_ep_name = re.sub(r'[^a-z0-9]', '', ep['name'].lower())
                target = re.sub(r'[^a-z0-9]', '', ep_title.lower())
                if target in s_ep_name or s_ep_name in target:
                    date = ep.get('air_date')
                    cache[cache_key] = date
                    return date
        
        # Fallback to series start date
        date = show_results[0].get('first_air_date')
        cache[cache_key] = date
        return date
    except: return None

def extract_date_from_string(text):
    """Robust helper to find dates in text (handles Nov., 23rd, etc)."""
    if not text: return None
    
    # Clean noise like semicolons
    text = text.replace(';', ' ')
    
    # Pattern 1: Day Month Year (23 Nov 1963, 23rd November 1963)
    p1 = r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]{2,}\.?\s*,?\s+\d{4})"
    # Pattern 2: Month Day Year (Nov 23, 1963, November 23rd 1963)
    p2 = r"([A-Z][a-z]{2,}\.?\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s+\d{4})"
    
    for p in [p1, p2]:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            # Remove ordinals for dateparser
            date_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str, flags=re.IGNORECASE)
            dt = dateparser.parse(date_str)
            if dt: return dt.strftime('%Y-%m-%d')
    return None

def parse_row(item, cache):
    raw_text = str(item.get('TEXT', item.get('text', '')))
    raw_metadata = item.get('METADATA', item.get('metadata', ''))
    
    show_name, ep_title, timestamp, is_invalid = "Unknown", "Unknown", None, False
    
    # 1. Parse Metadata
    if raw_metadata:
        try:
            m = ast.literal_eval(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
            show_name = m.get('show', m.get('series', 'Unknown'))
            ep_title = m.get('title', m.get('episode', 'Unknown'))
            if m.get('type') == 'movie' or show_name == ep_title: is_invalid = True
        except: pass

    # 2. PRIORITY 1: 'Original Airdate' header (User confirmed works)
    airdate_match = re.search(r"Original Airdate:\s*([A-Za-z0-9\s,]+)", raw_text[:2000])
    if airdate_match:
        timestamp = extract_date_from_string(airdate_match.group(1))

    # 3. PRIORITY 2: Metadata Title/Show strings (Fixes Doctor Who)
    if not timestamp:
        timestamp = extract_date_from_string(show_name) or extract_date_from_string(ep_title)

    # 4. PRIORITY 3: General Text Header Scan (Backup if metadata was missing date)
    if not timestamp:
        timestamp = extract_date_from_string(raw_text[:3000])

    # 5. PRIORITY 4: TMDB API Fallback
    if not timestamp and show_name != "Unknown" and not is_invalid:
        timestamp = get_airdate_fallback(show_name, ep_title, cache)

    # Filtering logic
    if show_name == "Unknown" or ep_title == "Unknown": is_invalid = True

    context = raw_text
    if "'TEXT':" in raw_text:
        try: context = raw_text.split("'TEXT':", 1)[1].strip().strip("'\"{}")
        except: pass

    return {
        "context": context, 
        "timestamp": timestamp, 
        "show_episode": f"{show_name} - {ep_title}", 
        "is_invalid": is_invalid
    }

def main():
    dataset = load_dataset("sedthh/tv_dialogue", split="train")
    cache = load_cache()
    processed = []
    
    for item in tqdm(dataset, desc="Final TV Cleaning"):
        res = parse_row(item, cache)
        if not res['is_invalid'] and len(res['context']) > 150:
            del res['is_invalid']
            processed.append(res)
        
    df = pd.DataFrame(processed)
    df.to_csv("tv_data_v16.csv", index=False)
    save_cache(cache)

if __name__ == "__main__":
    main()