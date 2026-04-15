import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime

def apply_newness_logic(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp') # Critical: Must be in chronological order
    
    seen_euphemisms = set()
    is_new_list = []
    
    for term in df['euphemism']:
        term = str(term).strip().lower()
        if term == "" or term == "nan" or term == "none":
            is_new_list.append(0)
        elif term in seen_euphemisms:
            is_new_list.append(0)
        else:
            seen_euphemisms.add(term)
            is_new_list.append(1)
            
    df['is_new_euphemism'] = is_new_list
    df.to_csv("youtube_labeled_dataset.csv", index=False)