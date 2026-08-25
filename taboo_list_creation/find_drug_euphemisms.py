import pandas as pd
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import re

###################################
# CONFIG
###################################

OLD_DATASET = "urban_dictionary_2016.csv"
DRUG_FILE = "drug_synonyms.txt"

BASE_URL = "https://unofficialurbandictionaryapi.com/api/date"

START_DATE = datetime(2017, 1, 1)
END_DATE = datetime.today()

###################################
# HELPERS
###################################

def normalize_word(word):
    return str(word).lower().strip()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text

###################################
# LOAD DRUG TERMS
###################################

with open(DRUG_FILE, encoding="utf-8") as f:
    drug_terms = [
        line.strip().lower()
        for line in f
        if line.strip()
    ]

# match whole words only
drug_pattern = re.compile(
    r'\b(?:' + '|'.join(map(re.escape, drug_terms)) + r')\b',
    re.IGNORECASE
)

###################################
# STEP 1:
# BUILD 2016 WORD -> DEFINITIONS
###################################

print("Loading 2016 dataset...")

df2016 = pd.read_csv(OLD_DATASET, engine='python', on_bad_lines='skip')

old_drug_dict = defaultdict(list)

for _, row in tqdm(df2016.iterrows(), total=len(df2016)):

    word = normalize_word(row["word"])
    definition = clean_text(row["definition"])

    if drug_pattern.search(definition):

        old_drug_dict[word].append({
            "definition": definition
        })

print("2016 drug words:", len(old_drug_dict))

###################################
# STEP 2:
# BUILD 2017+ WORD -> DEFINITIONS + DATES
###################################

new_drug_dict = defaultdict(list)

current_date = START_DATE

while current_date <= END_DATE:

    date_string = current_date.strftime("%Y-%m-%d")

    try:

        response = requests.get(
            BASE_URL,
            params={"date": date_string},
            timeout=20
        )

        if response.status_code == 200:

            entries = response.json().get("data", [])

            for entry in entries:

                definition = clean_text(
                    entry.get("definition", "")
                )

                example = clean_text(
                    entry.get("example", "")
                )

                combined = definition + " " + example

                if drug_pattern.search(combined):

                    word = normalize_word(entry["word"])

                    new_drug_dict[word].append({
                        "definition": definition,
                        "example": example,
                        "date": date_string
                    })

    except Exception as e:
        print("Failed:", date_string, e)

    current_date += timedelta(days=1)

print("2017+ drug words:", len(new_drug_dict))

###################################
# STEP 3:
# FIND WORDS WITH NEW DRUG SENSES
###################################

candidate_rows = []

for word, definitions in new_drug_dict.items():

    # word was not a drug term in 2016
    if word not in old_drug_dict:

        for entry in definitions:

            candidate_rows.append({
                "word": word,
                "date": entry["date"],
                "definition": entry["definition"],
                "example": entry["example"]
            })

###################################
# SAVE
###################################

candidate_df = pd.DataFrame(candidate_rows)

candidate_df.to_csv(
    "candidate_drug_euphemisms.csv",
    index=False
)

print("Found", len(candidate_df), "candidate entries.")