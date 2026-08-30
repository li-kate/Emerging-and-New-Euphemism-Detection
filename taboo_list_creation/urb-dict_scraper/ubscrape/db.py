import sqlite3

from .jsonwriter import JsonWriter
from .csvwriter import CsvWriter

DB_FILE_NAME = 'urban-dict-drugs.db'


def get_connection():
    return sqlite3.connect(DB_FILE_NAME)


def initialize_db():
    con = get_connection()

    con.execute('''CREATE TABLE IF NOT EXISTS word (
    word text PRIMARY KEY,
    letter text NOT NULL,
    complete integer NOT NULL,
    page_num integer NOT NULL
    );''')

    con.execute('''CREATE TABLE IF NOT EXISTS definition (
    id integer PRIMARY KEY,
    word_id text NOT NULL,
    definition text NOT NULL,
    date text,
    upvotes integer,
    downvotes integer,
    FOREIGN KEY (word_id) REFERENCES word (word)
    );''')

    con.commit()

    return con


def clear_database():
    con = get_connection()

    con.execute('DROP TABLE definition')
    con.execute('DROP TABLE word')

    con.commit()

    con.close()


def dump_database(arg, csv=False):
    con = get_connection()

    if csv:
        writer = CsvWriter()

        if isinstance(arg, str):
            writer = CsvWriter(out=arg)
    else:
        writer = JsonWriter()

        if isinstance(arg, str):
            writer = JsonWriter(out=arg)

    print(f'Dumping to: {writer.path}')

    prev_word = ''
    definition_list = []

    query = '''
            SELECT
                word.word,
                definition.definition,
                definition.date,
                definition.upvotes,
                definition.downvotes
            FROM definition
            INNER JOIN word
                ON definition.word_id = word.word
            ORDER BY word.word ASC;
        '''
    
    for (word, definition, date, upvotes, downvotes) in con.execute(query).fetchall():
        if word == prev_word:
            definition_list.append({
                "definition": definition,
                "date": date,
                "upvotes": upvotes,
                "downvotes": downvotes
            })

        if word != prev_word:
            # dump this definition and start a new set
            writer.write_word(prev_word, definition_list)

            prev_word = word
            definition_list = [{
                "definition": definition,
                "date": date,
                "upvotes": upvotes,
                "downvotes": downvotes
            }]

    writer.write_word(prev_word, definition_list)
    writer.dump_pool()
