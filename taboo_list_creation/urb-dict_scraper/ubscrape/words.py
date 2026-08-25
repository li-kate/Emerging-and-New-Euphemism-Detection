# internal to python
import re
import time
from urllib.parse import unquote
from string import ascii_uppercase
from sqlite3 import IntegrityError
from typing import List, Tuple

# external
import requests
from bs4 import BeautifulSoup


from .constants import BASE_URL
from .db import initialize_db

CON = initialize_db()


def write_words_for_letter(prefix: str):
    if not prefix:
        raise ValueError(f'Prefix {prefix} needs to be at least one letter.')

    def make_url():
        if page_num > 1:
            return f'{BASE_URL}/browse.php?character={letter}&page={page_num}'
        return f'{BASE_URL}/browse.php?character={letter}'

    letter = prefix.upper()

    page_num: int = CON.execute(
        'SELECT max(page_num) FROM word WHERE letter = ?', (letter,)).fetchone()[0]

    if not page_num:
        page_num = 1

    url = make_url()
    for attempt in range(5):
        try:
            req = requests.get(url, timeout=30)
            req.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

    while req.url != 'https://www.urbandictionary.com/':
        soup = BeautifulSoup(req.text, features="html.parser")
        a_tags = soup.find_all('a', href=re.compile(r'/define.php'))

        pattern = re.compile(
            r'\/define\.php\?term=(.*)')

        links = [l['href'] for l in a_tags]

        encoded_words: List[str] = [pattern.search(l).group(1)
                                    for l in links if pattern.search(l)]

        words: List[str] = [unquote(w) for w in encoded_words]

        #if not words:
        #   break

        formatted_words: List[Tuple[str, int, int, str]] = [
            (w, 0, page_num, letter) for w in words]

        try:
            before = CON.total_changes
            CON.executemany(
                'INSERT INTO word(word, complete, page_num, letter) VALUES (?, ?, ?, ?)',
                formatted_words)
            CON.commit()
            after = CON.total_changes

            if after == before:
                print("No new words inserted; stopping.")
                break

        except IntegrityError:
            # IntegrityError normally occurs when we try to
            # insert words that are already in the database.
            pass

        print(
            f'Working on page {page_num} for {letter}. Total {140 * (page_num - 1) + len(words)} {letter} words.')

        page_num += 1
        url = make_url()
        for attempt in range(5):
            try:
                req = requests.get(url, timeout=30)
                req.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1} failed: {e}")
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)


def write_all_words():
    for letter in ascii_uppercase + '*':
        write_words_for_letter(letter)
