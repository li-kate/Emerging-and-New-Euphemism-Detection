import re
import time
import sqlite3
from urllib.parse import unquote
from sqlite3 import IntegrityError
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup

from .constants import DRUGS_URL
from .db import initialize_db


CON = initialize_db()


def write_drug_words():
    page_num = CON.execute(
        'SELECT MAX(page_num) FROM word'
    ).fetchone()[0]

    if not page_num:
        page_num = 1
    else:
        page_num += 1

    while True:

        if page_num == 1:
            url = DRUGS_URL
        else:
            url = f'{DRUGS_URL}?page={page_num}'

        print(f'Fetching Drugs page {page_num}: {url}')

        for attempt in range(5):
            try:
                req = requests.get(
                    url,
                    timeout=30,
                    headers={
                        'User-Agent': 'Mozilla/5.0'
                    }
                )
                req.raise_for_status()
                break

            except requests.exceptions.RequestException as e:
                print(f'Attempt {attempt + 1} failed: {e}')

                if attempt == 4:
                    raise

                time.sleep(2 ** attempt)

        soup = BeautifulSoup(req.text, features='html.parser')

        # Find links to Urban Dictionary definitions.
        a_tags = soup.find_all(
            'a',
            href=re.compile(r'/define\.php')
        )

        words = []

        for tag in a_tags:
            href = tag.get('href')

            if not href:
                continue

            match = re.search(
                r'/define\.php\?term=([^&]+)',
                href
            )

            if not match:
                continue

            word = unquote(match.group(1)).strip()

            if valid_word(word) and word not in words:
                words.append(word)

        # If there are no words, we've reached the end.
        if not words:
            print(
                f'No words found on Drugs page {page_num}. '
                'Finished scraping the Drugs category.'
            )
            break

        # Determine the letter just for compatibility with the
        # existing database schema.
        formatted_words: List[Tuple[str, int, int, str]] = [
            (
                word,
                0,
                page_num,
                word[0].upper() if word[0].isalpha() else '*'
            )
            for word in words
        ]

        before = CON.total_changes

        try:
            CON.executemany(
                '''
                INSERT OR IGNORE INTO word
                (word, complete, page_num, letter)
                VALUES (?, ?, ?, ?)
                ''',
                formatted_words
            )

            CON.commit()

        except sqlite3.Error as e:
            print(f'Database error: {e}')
            raise

        after = CON.total_changes
        inserted = after - before

        print(
            f'Page {page_num}: found {len(words)} words; '
            f'inserted {inserted} new words.'
        )

        # If the page contained no new words, we've probably reached
        # the end or encountered a repeated page.
        if inserted == 0:
            print(
                'No new words inserted. '
                'Stopping category scrape.'
            )
            break

        page_num += 1

        # Avoid hammering Urban Dictionary.
        time.sleep(1)

def valid_word(word: str) -> bool:
    word = word.strip()

    # Maximum of 2 words
    if len(word.split()) > 2:
        return False

    # Each word may contain only letters and dashes.
    # Spaces between words are allowed.
    for part in word.split():
        if not re.fullmatch(r'[A-Za-z-]+', part):
            return False

    return True


def write_all_words():
    write_drug_words()