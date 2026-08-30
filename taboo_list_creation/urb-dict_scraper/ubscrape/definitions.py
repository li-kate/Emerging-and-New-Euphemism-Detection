import multiprocessing as mp
import time
import re

from typing import List, Tuple, Dict

import requests
from bs4 import BeautifulSoup

from .constants import BASE_URL
from .db import initialize_db


CON = initialize_db()


def extract_number(text):
    """
    Extract an integer from text such as:
        '123'
        '1,234'
        '123 thumbs up!'
    """

    if not text:
        return 0

    match = re.search(r'[\d,]+', text)

    if not match:
        return 0

    try:
        return int(match.group(0).replace(',', ''))
    except ValueError:
        return 0


def define_word(word: str) -> List[Dict]:

    if not word:
        raise ValueError('Must pass a word.')

    url = f'{BASE_URL}/define.php'

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/150.0.0.0 Safari/537.36'
        ),
        'Accept': (
            'text/html,application/xhtml+xml,application/xml;'
            'q=0.9,*/*;q=0.8'
        )
    }

    # ---------------------------------------------------------
    # GET DEFINITION PAGE
    # ---------------------------------------------------------

    for attempt in range(5):

        try:
            req = requests.get(
                url,
                params={'term': word},
                timeout=30,
                headers=headers
            )

            req.raise_for_status()
            break

        except requests.exceptions.RequestException as e:

            print(
                f'Attempt {attempt + 1} failed for "{word}": {e}'
            )

            if attempt == 4:
                raise

            time.sleep(2 ** attempt)

    soup = BeautifulSoup(
        req.text,
        features='html.parser'
    )

    definition_cards = [
        card for card in soup.select('.definition')
        if card.get('data-word', '').strip().lower() == word.strip().lower()
    ]

    results = []

    # ---------------------------------------------------------
    # FIRST PASS:
    # Extract definitions, dates, and defids
    # ---------------------------------------------------------

    for card in definition_cards:

        meaning = card.find(
            'div',
            class_='meaning'
        )

        if meaning is None:
            continue

        definition = meaning.get_text(
            ' ',
            strip=True
        )

        # -----------------------------------------------------
        # DEFID
        # -----------------------------------------------------

        vote_div = card.find(
            id=re.compile(r'^vote-buttons-\d+$')
        )

        if vote_div:
            match = re.search(
                r'vote-buttons-(\d+)',
                vote_div.get('id', '')
            )

            if match:
                defid = match.group(1)
            else:
                defid = None
        else:
            defid = None

        # -----------------------------------------------------
        # DATE
        # -----------------------------------------------------

        date = None

        contributor = card.find(
            'div',
            class_='font-medium'
        )

        if contributor:

            text = contributor.get_text(
                ' ',
                strip=True
            )

            match = re.search(
                r'(January|February|March|April|May|June|July|'
                r'August|September|October|November|December)'
                r'\s+\d{1,2},\s+\d{4}',
                text
            )

            if match:
                date = match.group(0)

        results.append({
            'definition': definition,
            'date': date,
            'defid': defid,
            'upvotes': 0,
            'downvotes': 0
        })

    # ---------------------------------------------------------
    # SECOND PASS:
    # Get actual vote counts from /ui/votes
    # ---------------------------------------------------------

    defids = [
        item['defid']
        for item in results
        if item['defid'] is not None
    ]

    if defids:

        defid_string = ','.join(defids)

        vote_headers = {
            **headers,
            'Accept': '*/*',
            'Referer': (
                f'{BASE_URL}/define.php?term='
                + requests.utils.quote(word)
            ),
            'X-Up-Context': '{}',
            'X-Up-Fail-Context': '{}',
            'X-Up-Fail-Mode': 'root',
            'X-Up-Mode': 'root',
            'X-Up-Origin-Mode': 'root',
            'X-Up-Target': ', '.join(
                f'#vote-buttons-{d}'
                for d in defids
            ),
            'X-Up-Version': '3.14.2'
        }

        votes_url = f'{BASE_URL}/ui/votes'

        for attempt in range(5):

            try:

                vote_req = requests.get(
                    votes_url,
                    params={'defids': defid_string},
                    timeout=30,
                    headers=vote_headers
                )

                vote_req.raise_for_status()
                break

            except requests.exceptions.RequestException as e:

                print(
                    f'Vote request attempt {attempt + 1} '
                    f'failed for "{word}": {e}'
                )

                if attempt == 4:
                    print(
                        f'Could not retrieve votes for "{word}". '
                        'Using 0 for missing vote counts.'
                    )
                    vote_req = None
                    break

                time.sleep(2 ** attempt)

        # -----------------------------------------------------
        # Parse vote response
        # -----------------------------------------------------

        if vote_req is not None:

            vote_soup = BeautifulSoup(
                vote_req.text,
                features='html.parser'
            )

            for vote_div in vote_soup.find_all(
                id=re.compile(r'^vote-buttons-\d+$')
            ):

                match = re.search(
                    r'vote-buttons-(\d+)',
                    vote_div.get('id', '')
                )

                if not match:
                    continue

                defid = match.group(1)

                upvotes = 0
                downvotes = 0

                # Upvote
                upvote_button = vote_div.find(
                    'button',
                    attrs={
                        'aria-label': re.compile(
                            r'^Upvote'
                        )
                    }
                )

                if upvote_button:

                    label = upvote_button.get(
                        'aria-label',
                        ''
                    )

                    match = re.search(
                        r'\(([\d,]+)\)',
                        label
                    )

                    if match:
                        upvotes = int(
                            match.group(1).replace(',', '')
                        )

                # Downvote
                downvote_button = vote_div.find(
                    'button',
                    attrs={
                        'aria-label': re.compile(
                            r'^Downvote'
                        )
                    }
                )

                if downvote_button:

                    label = downvote_button.get(
                        'aria-label',
                        ''
                    )

                    match = re.search(
                        r'\(([\d,]+)\)',
                        label
                    )

                    if match:
                        downvotes = int(
                            match.group(1).replace(',', '')
                        )

                # -------------------------------------------------
                # Match votes back to the definition
                # -------------------------------------------------

                for item in results:

                    if item['defid'] == defid:

                        item['upvotes'] = upvotes
                        item['downvotes'] = downvotes
                        break

    # ---------------------------------------------------------
    # Remove defid before returning
    # ---------------------------------------------------------

    for item in results:
        item.pop('defid', None)

    return results

def write_definition(word_t: Tuple[str]):

    word = word_t[0]

    definitions = define_word(word)

    formatted_defs = []

    for item in definitions:

        formatted_defs.append(
            (
                item['definition'],
                item['date'],
                item['upvotes'],
                item['downvotes'],
                word
            )
        )

    if formatted_defs:

        CON.executemany(
            '''
            INSERT INTO definition
            (
                definition,
                date,
                upvotes,
                downvotes,
                word_id
            )
            VALUES (?, ?, ?, ?, ?)
            ''',
            formatted_defs
        )

    CON.execute(
        'UPDATE word SET complete = 1 WHERE word = ?',
        word_t
    )

    CON.commit()

    return definitions


def define_all_words():

    pool = mp.Pool(mp.cpu_count())

    words = CON.execute(
        'SELECT word FROM word WHERE complete = 0'
    ).fetchall()

    print(
        f'{len(words)} words need definitions.'
    )

    pool.map(
        write_definition,
        words,
        chunksize=200
    )

    pool.close()
    pool.join()