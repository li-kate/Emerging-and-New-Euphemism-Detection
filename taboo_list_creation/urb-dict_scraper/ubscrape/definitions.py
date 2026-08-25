import multiprocessing as mp
from typing import List, Tuple

from bs4 import BeautifulSoup
import requests

from .constants import BASE_URL
from .db import initialize_db
import re

CON = initialize_db()


def define_word(word: str) -> List[str]:
    if not word:
        raise ValueError('Must pass a word.')

    url = f'{BASE_URL}/define.php'

    req = requests.get(url, params={'term': word})

    soup = BeautifulSoup(req.text, features="html.parser")

    # meaning_tags = soup.find_all('div', {'class': 'meaning'})
    definition_cards = soup.select(".definition")

    results = []

    for card in definition_cards:

        meaning = card.find("div", class_="meaning")
        if meaning is None:
            continue

        contributor = card.find("div", class_="font-medium")

        definition = meaning.get_text(" ", strip=True)

        date = None
        if contributor:
            text = contributor.get_text(" ", strip=True)

            m = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
                text,
            )

            if m:
                date = m.group(0)

        results.append((definition, date))

    return results


def write_definition(word_t: Tuple[str]) -> List[str]:
    # word will always be a tuple when this function is called from define_all_words().
    # so in `cli.py`, we make word a tuple to match the type signature.
    word = word_t[0]

    # Note: this code will always make a network request.
    # If offline support for definitions was required, it
    # could check the local db for any definitions.
    defs: List[Tuple[str, str]] = define_word(word)
    formatted_defs: List[Tuple[str, str, str]] = [
        (definition, date, word)
        for definition, date in defs
    ]

    CON.executemany(
        'INSERT INTO definition(definition, date, word_id) VALUES (?, ?, ?)', formatted_defs)
    CON.execute('UPDATE word SET complete = 1 WHERE word = ?', word_t)
    CON.commit()

    return defs


def define_all_words():
    pool = mp.Pool(mp.cpu_count())

    words = CON.execute(
        'SELECT word FROM word WHERE complete = 0').fetchall()

    pool.map(write_definition, words, chunksize=200)
