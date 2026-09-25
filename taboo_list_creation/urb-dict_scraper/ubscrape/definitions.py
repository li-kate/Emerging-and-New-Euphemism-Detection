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

def fetch_definition(word_t: Tuple[str]):

    """
    Worker function.

    This runs in parallel and ONLY makes HTTP requests.
    It does NOT access SQLite.
    """

    word = word_t[0]

    try:
        definitions = define_word(word)

        return {
            'word': word,
            'definitions': definitions,
            'success': True
        }

    except Exception as e:

        print(
            f'ERROR fetching "{word}": {e}'
        )

        return {
            'word': word,
            'definitions': [],
            'success': False
        }


def save_definition(result):

    """
    Runs only in the main process.

    This is the ONLY function that writes to SQLite,
    preventing 'database is locked' errors.
    """

    word = result['word']
    definitions = result['definitions']

    # Do not mark the word complete if the request failed.
    # This allows it to be retried in a future run.
    if not result['success']:
        return

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

    try:

        # Save all definitions for this word.
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

        # Only mark the word complete after successfully
        # saving its definitions.
        CON.execute(
            '''
            UPDATE word
            SET complete = 1
            WHERE word = ?
            ''',
            (word,)
        )

        CON.commit()

    except Exception as e:

        CON.rollback()

        print(
            f'ERROR saving "{word}": {e}'
        )


def define_all_words():

    words = CON.execute(
        '''
        SELECT word
        FROM word
        WHERE complete = 0
        '''
    ).fetchall()

    print(
        f'{len(words)} words need definitions.'
    )

    if not words:

        print(
            'All words already have definitions.'
        )

        return

    # Don't use every CPU automatically.
    #
    # Too many simultaneous requests may cause Urban
    # Dictionary to rate-limit or block the scraper.
    workers = min(
        8,
        mp.cpu_count()
    )

    print(
        f'Using {workers} worker processes.'
    )

    completed = 0

    with mp.Pool(workers) as pool:

        # imap_unordered returns results as soon as workers
        # finish, allowing the MAIN process to write each
        # result safely to SQLite.
        for result in pool.imap_unordered(
            fetch_definition,
            words,
            chunksize=10
        ):

            save_definition(result)

            completed += 1

            if completed % 100 == 0:

                remaining = len(words) - completed

                print(
                    f'Processed {completed}/{len(words)} '
                    f'words. '
                    f'{remaining} remaining.'
                )

    print(
        'Finished processing all remaining words.'
    )