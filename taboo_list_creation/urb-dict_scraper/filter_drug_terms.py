"""
Filter Rules:
  1. Filter for words with drug-related terms (from drugs.txt)
  2. Removes terms from the database that are more than 2 words.
  3. Removes words that are less than 4 characters.
  4. Removes words with less than 100 upvotes.
  5. If a drug-related definition exists before 2015, exclude that word.

Drug Matching:
  - Checks definitions for terms listed in drug_synonyms.txt.
  - Saves the matching drug terms for each normalized word.
  - If a definition containing a drug term is before 2015,
    the ENTIRE normalized word is excluded.

Additional Normalization:
  - Tries to combine words that are the same.
      - ex: <word> & <word>s
      - ex: <word> & a <word>
"""


import sqlite3
import json
import re
import unicodedata
from functools import lru_cache
from datetime import datetime

from nltk.stem import WordNetLemmatizer


# FILE SETTINGS

DB_PATH = "urban-dict-drugs.db"
DRUG_FILE = "drug_synonyms.txt"
OUTPUT_FILE = "filtered_drug_terms.json"

# FILTER SETTINGS

# Maximum number of words allowed in a term.
MAX_WORDS = 2
# Minimum number of characters allowed in a term.
MIN_TERM_LENGTH = 4
# Remove a word entirely if its earliest definition is before this date.
MIN_FIRST_MENTION_DATE = datetime(2015, 1, 1)
# Only save definitions with at least this many upvotes.
MIN_UPVOTES = 100

# WORD NORMALIZATION

lemmatizer = WordNetLemmatizer()

def parse_date(date_string):
    """Parse Urban Dictionary dates for chronological sorting."""
    if not date_string:
        return None

    try:
        return datetime.strptime(
            date_string.strip(),
            "%B %d, %Y"
        )

    except ValueError:
        print(
            f"WARNING: Could not parse date: "
            f"{date_string!r}"
        )
        return None

@lru_cache(maxsize=500_000)
def normalize_word(word):
    if not word:
        return None

    # Unicode normalization
    word = unicodedata.normalize(
        "NFKC",
        word
    )
    # Lowercase
    word = word.lower()

    # Normalize whitespace
    word = re.sub(
        r"\s+",
        " ",
        word
    ).strip()

    if not word:
        return None

    # Remove surrounding punctuation
    word = word.strip(
        " \t\n\r.,!?;:\"“”‘’()[]{}<>"
    )

    if not word:
        return None

    # Treat "a <word>" as equivalent to "<word>"
    if word.startswith("a "):

        word = word[2:].strip()

    if not word:
        return None

    # Normalize possessives
    word = re.sub(
        r"(['’])s$",
        "",
        word
    )

    word = re.sub(
        r"(['’])$",
        "",
        word
    )

    # Split into tokens
    tokens = word.split()
    normalized_tokens = []

    for token in tokens:
        token = token.strip(
            ".,!?;:\"“”‘’()[]{}<>"
        )

        if not token:
            continue

        # Morphological normalization
        if token.isalpha():
            lemma = lemmatizer.lemmatize(
                token,
                pos="n"
            )

            if lemma != token:

                if (
                    token.endswith("s")
                    and len(lemma) >= 3
                    and len(token) - len(lemma) <= 3
                ):
                    token = lemma
            else:
                if (
                    len(token) > 4
                    and token.endswith("ies")
                ):
                    candidate = (
                        token[:-3] + "y"
                    )

                    if len(candidate) >= 3:
                        token = candidate

                # Common -es plurals
                elif (
                    len(token) > 4
                    and token.endswith("es")
                ):
                    candidate = token[:-2]

                    if len(candidate) >= 3:
                        token = candidate

                # Simple plural s
                elif (
                    len(token) > 4
                    and token.endswith("s")
                    and not token.endswith(
                        ("ss", "us", "is")
                    )
                ):
                    candidate = token[:-1]

                    if len(candidate) >= 3:
                        token = candidate

        normalized_tokens.append(token)

    normalized = " ".join(
        normalized_tokens
    )

    return normalized if normalized else None

# LOAD DRUG SYNONYMS

with open(
    DRUG_FILE,
    "r",
    encoding="utf-8"
) as f:
    drugs = [
        line.strip().lower()
        for line in f
        if line.strip()
        and not line.startswith("#")
    ]


# Processing (longest drug terms first).
drugs.sort(
    key=len,
    reverse=True
)

# Create case-insensitive regex patterns.
patterns = [
    (
        drug,
        re.compile(
            r"\b" + re.escape(drug) + r"\b",
            re.IGNORECASE
        )
    )
    for drug in drugs
]

print(
    f"Loaded {len(drugs):,} drug terms "
    f"from {DRUG_FILE}"
)

# DATABASE

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("""
SELECT
    word.word,
    definition.definition,
    definition.date,
    definition.upvotes,
    definition.downvotes
FROM definition
JOIN word
    ON definition.word_id = word.word
""")

# RESULTS

results_by_word = {}
excluded_words = set()

processed = 0
matched_definitions = 0
pre_2015_matches = 0

skipped_long_words = 0
skipped_empty_words = 0
skipped_short_words = 0
skipped_low_upvotes = 0

saved_definitions = 0

# PROCESS ALL DEFINITIONS

for row in cur:
    processed += 1
    original_word = row["word"]

    # Reject empty words
    if not original_word:
        skipped_empty_words += 1
        continue

    # Reject terms containing more than MAX_WORDS words
    word_parts = re.findall(
        r"\S+",
        original_word.strip()
    )
    if len(word_parts) > MAX_WORDS:
        skipped_long_words += 1
        continue

    # Normalize word
    normalized_word = normalize_word(
        original_word
    )
    if not normalized_word:
        skipped_empty_words += 1
        continue

    # Reject terms that are too short
    character_count = len(
        re.sub(
            r"\s+",
            "",
            normalized_word
        )
    )
    if character_count < MIN_TERM_LENGTH:
        skipped_short_words += 1
        continue

    # Get definition
    definition = row["definition"]
    if not definition:
        continue

    # CHECK DEFINITION FOR DRUG TERMS

    matches = []

    for drug, pattern in patterns:
        if pattern.search(definition):
            matches.append(drug)

    if not matches:
        continue

    # Get date
    definition_date = row["date"]
    parsed_date = parse_date(
        definition_date
    )

    # If a definition containing a drug term is before 2015, exclude the ENTIRE normalized word.
    if (
        matches
        and parsed_date is not None
        and parsed_date < MIN_FIRST_MENTION_DATE
    ):
        excluded_words.add(
            normalized_word
        )
        pre_2015_matches += 1
        continue

    # Get votes
    upvotes = row["upvotes"]
    downvotes = row["downvotes"]

    # Convert NULL values to zero.
    if upvotes is None:
        upvotes = 0
    if downvotes is None:
        downvotes = 0

    # Filter definitions with insufficient upvotes
    if upvotes < MIN_UPVOTES:
        skipped_low_upvotes += 1
        continue

    # Create word entry if necessary
    if normalized_word not in results_by_word:
        results_by_word[normalized_word] = {
            "word": original_word,
            # Store all drug terms found across definitions.
            "matched_drugs": set(),
            "definitions": []
        }

    result = results_by_word[normalized_word]

    # Save drug matches
    result["matched_drugs"].update(matches)

    # Save definition
    result["definitions"].append({
        "definition": definition,
        "date": definition_date,
        "upvotes": upvotes,
        "downvotes": downvotes
    })
    saved_definitions += 1

conn.close()

for normalized_word in excluded_words:
    results_by_word.pop(normalized_word, None)

# PREPARE FINAL JSON

results = []

for normalized_word, result in results_by_word.items():
    definitions = result["definitions"]

    # Remove exact duplicate definitions
    unique_definitions = []
    seen_definitions = set()
    for item in definitions:
        key = (
            item["definition"],
            item["date"],
            item["upvotes"],
            item["downvotes"]
        )

        if key in seen_definitions:
            continue
        seen_definitions.add(
            key
        )
        unique_definitions.append(
            item
        )

    for item in unique_definitions:
        parsed_date = parse_date(item["date"])

    # Sort definitions chronologically
    unique_definitions.sort(
        key=lambda x: (
            parse_date(
                x["date"]
            ) is None,

            parse_date(
                x["date"]
            )
            if parse_date(
                x["date"]
            ) is not None
            else datetime.max
        )
    )

    # Final object

    results.append({
        "word": result["word"],
        "matched_drugs": sorted(result["matched_drugs"]),
        "definitions": unique_definitions
    })

# SORT FINAL RESULTS

results.sort(
    key=lambda x: x["word"].lower()
)

# WRITE JSON

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

# SUMMARY

print("\n========================================")
print("DONE")
print("========================================")

print(
    f"Loaded drug terms:            "
    f"{len(drugs):,}"
)
print(
    f"Processed definitions:         "
    f"{processed:,}"
)
print(
    f"Drug-matching definitions:     "
    f"{matched_definitions:,}"
)
print(
    f"Pre-2015 drug matches:         "
    f"{pre_2015_matches:,}"
)
print(
    f"Words excluded from drug match:"
    f"{len(excluded_words):,}"
)
print(
    f"Skipped >2-word terms:         "
    f"{skipped_long_words:,}"
)
print(
    f"Skipped empty terms:           "
    f"{skipped_empty_words:,}"
)
print(
    f"Skipped <=3-character terms:   "
    f"{skipped_short_words:,}"
)
print(
    f"Skipped low-upvote defs:       "
    f"{skipped_low_upvotes:,}"
)
print(
    f"Saved definitions:             "
    f"{saved_definitions:,}"
)
print(
    f"Final unique words:            "
    f"{len(results):,}"
)
print(
    f"Wrote:                         "
    f"{OUTPUT_FILE}"
)