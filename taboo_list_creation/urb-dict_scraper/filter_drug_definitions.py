import sqlite3
import json
import re
import unicodedata
from functools import lru_cache
from datetime import datetime

from nltk.stem import WordNetLemmatizer


DB_PATH = "urban-dict.db"
DRUG_FILE = "drug_synonyms.txt"
OUTPUT_FILE = "drug_matches.json"

# Definitions containing drug terms before this date
# cause the ENTIRE normalized word to be excluded.
CUTOFF_DATE = datetime(2015, 1, 1)

# Maximum number of words allowed in a term.
MAX_WORDS = 2


# ============================================================
# WORD NORMALIZATION
# ============================================================

lemmatizer = WordNetLemmatizer()

def parse_date(date_string):
    if not date_string:
        return None

    try:
        return datetime.strptime(
            date_string.strip(),
            "%B %d, %Y"
        )
    except ValueError:
        print(f"WARNING: Could not parse date: {date_string!r}")
        return None

@lru_cache(maxsize=500_000)
def normalize_word(word):
    """
    Normalize terms for deduplication.

    Examples:
        Coke       -> coke
        COKE       -> coke
        coke       -> coke
        coke's     -> coke
        cokes      -> coke
        Coca Cola  -> coca cola
    """

    if not word:
        return None

    # Unicode normalization
    word = unicodedata.normalize("NFKC", word)

    # Lowercase
    word = word.lower()

    # Normalize whitespace
    word = re.sub(r"\s+", " ", word).strip()

    if not word:
        return None

    # Remove punctuation surrounding the whole term
    word = word.strip(
        " \t\n\r.,!?;:\"“”‘’()[]{}<>"
    )

    if not word:
        return None

    # Normalize possessives
    #
    # coke's -> coke
    # coke'  -> coke
    word = re.sub(r"(['’])s$", "", word)
    word = re.sub(r"(['’])$", "", word)

    tokens = word.split()

    normalized_tokens = []

    for token in tokens:

        token = token.strip(
            ".,!?;:\"“”‘’()[]{}<>"
        )

        if not token:
            continue

        # Only perform morphological normalization on
        # purely alphabetic tokens.
        if token.isalpha():

            original = token

            # WordNet noun lemmatization
            lemma = lemmatizer.lemmatize(
                token,
                pos="n"
            )

            if lemma != token:

                # Conservative acceptance of WordNet result
                if (
                    token.endswith("s")
                    and len(lemma) >= 3
                    and len(token) - len(lemma) <= 3
                ):
                    token = lemma

            else:

                # --------------------------------------------
                # Conservative plural handling
                # --------------------------------------------

                # babies -> baby
                if (
                    len(token) > 4
                    and token.endswith("ies")
                ):
                    candidate = token[:-3] + "y"

                    if len(candidate) >= 3:
                        token = candidate

                # --------------------------------------------
                # Common -es plurals
                # --------------------------------------------
                elif (
                    len(token) > 4
                    and token.endswith("es")
                ):
                    candidate = token[:-2]

                    if len(candidate) >= 3:
                        token = candidate

                # --------------------------------------------
                # Simple plural s
                #
                # Avoid words ending in:
                # ss, us, is
                # --------------------------------------------
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

    normalized = " ".join(normalized_tokens)

    return normalized if normalized else None


# ============================================================
# LOAD DRUG SYNONYMS
# ============================================================

with open(DRUG_FILE, "r", encoding="utf-8") as f:

    drugs = [
        line.strip().lower()
        for line in f
        if line.strip()
        and not line.startswith("#")
    ]

# Longest first
drugs.sort(key=len, reverse=True)

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


# ============================================================
# DATABASE
# ============================================================

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

cur = conn.cursor()

cur.execute("""
SELECT
    word.word,
    definition.definition,
    definition.date
FROM definition
JOIN word
    ON definition.word_id = word.word
""")


# ============================================================
# RESULTS
# ============================================================

results_by_word = {}

# Normalized words that have a drug-matching definition
# before 2018.
excluded_words = set()

processed = 0
matched_definitions = 0
pre_2018_matches = 0
skipped_long_words = 0
skipped_empty_words = 0


# ============================================================
# PROCESS DEFINITIONS
# ============================================================

for row in cur:

    processed += 1

    original_word = row["word"]

    if not original_word:
        skipped_empty_words += 1
        continue


    # --------------------------------------------------------
    # Reject terms containing more than 2 words
    # --------------------------------------------------------

    word_parts = re.findall(
        r"\S+",
        original_word.strip()
    )

    if len(word_parts) > MAX_WORDS:
        skipped_long_words += 1
        continue


    # --------------------------------------------------------
    # Normalize word
    # --------------------------------------------------------

    normalized_word = normalize_word(
        original_word
    )

    if not normalized_word:
        skipped_empty_words += 1
        continue


    # --------------------------------------------------------
    # Check definition for drug terms
    # --------------------------------------------------------

    definition = row["definition"]

    if not definition:
        continue

    matches = []

    for drug, pattern in patterns:

        if pattern.search(definition):
            matches.append(drug)

    if not matches:
        continue


    matched_definitions += 1


    # ========================================================
    # CRITICAL RULE:
    #
    # Any drug-matching definition before 2018 causes the
    # ENTIRE normalized word to be excluded.
    # ========================================================

    definition_date = row["date"]
    parsed_date = parse_date(definition_date)

    if parsed_date is not None and parsed_date < CUTOFF_DATE:
        excluded_words.add(normalized_word)
        pre_2018_matches += 1
        continue


    # --------------------------------------------------------
    # This is a valid 2018+ drug-matching definition.
    # Store it temporarily.
    # --------------------------------------------------------

    if normalized_word not in results_by_word:

        results_by_word[normalized_word] = {

            "word": original_word,

            "matched_drugs": set(matches),

            "definitions": [
                {
                    "definition": definition,
                    "date": definition_date
                }
            ],

            "earliest_date": parsed_date

        }

    else:

        result = results_by_word[
            normalized_word
        ]

        # Add drug matches
        result["matched_drugs"].update(
            matches
        )

        # Add definition
        result["definitions"].append({

            "definition": definition,

            "date": definition_date

        })


        # Update earliest valid date
        current_date = result["earliest_date"]

        if definition_date is not None:

            if (
                parsed_date is not None
                and (
                    current_date is None
                    or parsed_date < current_date
                )
            ):
                result["earliest_date"] = parsed_date
                result["word"] = original_word


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if processed % 100_000 == 0:

        print(
            f"Processed: {processed:,} | "
            f"Matching definitions: "
            f"{matched_definitions:,} | "
            f"Excluded words: "
            f"{len(excluded_words):,} | "
            f"Candidate words: "
            f"{len(results_by_word):,}"
        )


conn.close()


# ============================================================
# REMOVE WORDS WITH PRE-2018 DRUG MATCHES
# ============================================================

for normalized_word in excluded_words:

    results_by_word.pop(
        normalized_word,
        None
    )


# ============================================================
# PREPARE FINAL JSON
# ============================================================

results = []


for normalized_word, result in results_by_word.items():

    definitions = result["definitions"]


    # --------------------------------------------------------
    # Sort definitions chronologically
    # --------------------------------------------------------

    definitions.sort(
        key=lambda x: (
            x["date"] is None,
            str(x["date"])
            if x["date"] is not None
            else ""
        )
    )


    # --------------------------------------------------------
    # Remove exact duplicate definition/date pairs
    # --------------------------------------------------------

    unique_definitions = []

    seen_definitions = set()

    for item in definitions:

        key = (
            item["definition"],
            item["date"]
        )

        if key in seen_definitions:
            continue

        seen_definitions.add(key)

        unique_definitions.append(item)


    # --------------------------------------------------------
    # Final object
    # --------------------------------------------------------

    results.append({

        "word": result["word"],

        "matched_drugs": sorted(
            result["matched_drugs"]
        ),

        "definitions": unique_definitions,

        "date": (
            result["earliest_date"].strftime("%B %d, %Y")
            if result["earliest_date"] is not None
            else None
        )

    })


# ============================================================
# SORT FINAL RESULTS
# ============================================================

results.sort(
    key=lambda x: x["word"].lower()
)


# ============================================================
# WRITE JSON
# ============================================================

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


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("DONE")
print("========================================")

print(
    f"Processed definitions:     "
    f"{processed:,}"
)

print(
    f"Matching definitions:      "
    f"{matched_definitions:,}"
)

print(
    f"Pre-2018 drug matches:     "
    f"{pre_2018_matches:,}"
)

print(
    f"Words excluded entirely:   "
    f"{len(excluded_words):,}"
)

print(
    f"Skipped >2-word terms:     "
    f"{skipped_long_words:,}"
)

print(
    f"Skipped empty terms:       "
    f"{skipped_empty_words:,}"
)

print(
    f"Final unique words:        "
    f"{len(results):,}"
)

print(
    f"Wrote:                     "
    f"{OUTPUT_FILE}"
)

print("========================================")