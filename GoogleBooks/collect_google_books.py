#!/usr/bin/env python3
"""
Collect Google Books API results into a local dataset (JSONL + SQLite).

What you get per row:
- query used
- volume_id
- title, authors, publisher, publishedDate, language, categories
- text snippet (when available via searchInfo.textSnippet)
- timestamps: fetched_at_utc + publishedDate (as provided by Google)

Notes:
- You do NOT get full book text. Only metadata + (sometimes) a snippet.
- For "natural placement", use lots of neutral queries (the, people, life, etc.)
- Respect Google's terms + quotas.

Usage example:
  python collect_google_books.py \
    --api-key YOUR_KEY \
    --queries queries.txt \
    --out-jsonl books.jsonl \
    --out-sqlite books.sqlite \
    --lang en \
    --max-requests 200 \
    --sleep 0.2

queries.txt: one query per line (can include quotes)
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

GOOGLE_BOOKS_VOLUMES_URL = "https://www.googleapis.com/books/v1/volumes"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_queries(path: str) -> List[str]:
    qs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            qs.append(q)
    if not qs:
        raise ValueError(f"No queries found in {path}")
    return qs


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalize_snippet(snippet: Optional[str]) -> Optional[str]:
    if not snippet:
        return None
    # Google returns HTML-escaped snippets like "... &quot;word&quot; ..."
    # Unescape and strip.
    return html.unescape(snippet).strip()


def books_api_request(
    api_key: Optional[str],
    q: str,
    start_index: int,
    max_results: int,
    lang: Optional[str],
    print_type: str,
) -> Dict[str, Any]:
    params = {
        "q": q,
        "startIndex": start_index,
        "maxResults": max_results,  # max 40
        "printType": print_type,    # books | magazines | all
    }
    if lang:
        params["langRestrict"] = lang
    if api_key:
        params["key"] = api_key

    r = requests.get(GOOGLE_BOOKS_VOLUMES_URL, params=params, timeout=30)
    # 429 / 503 can happen. Caller handles backoff.
    r.raise_for_status()
    return r.json()


def extract_rows(payload: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = payload.get("items") or []
    for it in items:
        volume_id = it.get("id")
        vinfo = it.get("volumeInfo") or {}
        sinfo = it.get("searchInfo") or {}

        row = {
            "fetched_at_utc": utc_now_iso(),
            "query": query,
            "volume_id": volume_id,
            "title": vinfo.get("title"),
            "subtitle": vinfo.get("subtitle"),
            "authors": vinfo.get("authors") or [],
            "publisher": vinfo.get("publisher"),
            "publishedDate": vinfo.get("publishedDate"),  # can be YYYY, YYYY-MM, or YYYY-MM-DD
            "language": vinfo.get("language"),
            "categories": vinfo.get("categories") or [],
            "pageCount": vinfo.get("pageCount"),
            "printType": vinfo.get("printType"),
            "maturityRating": vinfo.get("maturityRating"),
            "snippet": normalize_snippet(sinfo.get("textSnippet")),
            "infoLink": vinfo.get("infoLink"),
            "previewLink": vinfo.get("previewLink"),
            "canonicalVolumeLink": vinfo.get("canonicalVolumeLink"),
        }
        rows.append(row)
    return rows


def init_sqlite(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at_utc TEXT,
            query TEXT,
            volume_id TEXT,
            title TEXT,
            subtitle TEXT,
            authors_json TEXT,
            publisher TEXT,
            publishedDate TEXT,
            language TEXT,
            categories_json TEXT,
            pageCount INTEGER,
            printType TEXT,
            maturityRating TEXT,
            snippet TEXT,
            infoLink TEXT,
            previewLink TEXT,
            canonicalVolumeLink TEXT,
            UNIQUE(volume_id, query, snippet)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_books_volume_id ON books(volume_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_books_publishedDate ON books(publishedDate)")
    con.commit()
    con.close()


def insert_sqlite(db_path: str, rows: List[Dict[str, Any]]) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    inserted = 0
    for r in rows:
        try:
            cur.execute(
                """
                INSERT OR IGNORE INTO books (
                    fetched_at_utc, query, volume_id, title, subtitle, authors_json,
                    publisher, publishedDate, language, categories_json, pageCount,
                    printType, maturityRating, snippet, infoLink, previewLink, canonicalVolumeLink
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("fetched_at_utc"),
                    r.get("query"),
                    r.get("volume_id"),
                    r.get("title"),
                    r.get("subtitle"),
                    json.dumps(r.get("authors") or [], ensure_ascii=False),
                    r.get("publisher"),
                    r.get("publishedDate"),
                    r.get("language"),
                    json.dumps(r.get("categories") or [], ensure_ascii=False),
                    r.get("pageCount"),
                    r.get("printType"),
                    r.get("maturityRating"),
                    r.get("snippet"),
                    r.get("infoLink"),
                    r.get("previewLink"),
                    r.get("canonicalVolumeLink"),
                ),
            )
            if cur.rowcount == 1:
                inserted += 1
        except sqlite3.Error:
            # Keep going; the row is likely malformed/too long in rare cases
            continue
    con.commit()
    con.close()
    return inserted


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def backoff_sleep(base: float, attempt: int, jitter: float = 0.2) -> None:
    # Exponential backoff with jitter
    wait = base * (2 ** attempt)
    wait = wait * (1 + random.uniform(-jitter, jitter))
    time.sleep(max(0.0, wait))


def collect(
    api_key: Optional[str],
    queries: List[str],
    out_jsonl: Optional[str],
    out_sqlite: Optional[str],
    lang: Optional[str],
    print_type: str,
    max_results_per_request: int,
    max_requests: int,
    per_query_pages: int,
    sleep_s: float,
    shuffle_queries: bool,
) -> None:
    if out_sqlite:
        init_sqlite(out_sqlite)

    if shuffle_queries:
        random.shuffle(queries)

    req_count = 0
    total_rows = 0
    total_inserted = 0

    for q in queries:
        # Each page is a separate request; startIndex increments by maxResults
        for page in range(per_query_pages):
            if req_count >= max_requests:
                print(f"Reached max_requests={max_requests}. Stopping.")
                print(f"Total rows collected: {total_rows}")
                if out_sqlite:
                    print(f"Total inserted into SQLite: {total_inserted}")
                return

            start_index = page * max_results_per_request

            # Retry loop for transient errors
            attempt = 0
            while True:
                try:
                    payload = books_api_request(
                        api_key=api_key,
                        q=q,
                        start_index=start_index,
                        max_results=max_results_per_request,
                        lang=lang,
                        print_type=print_type,
                    )
                    break
                except requests.HTTPError as e:
                    status = getattr(e.response, "status_code", None)
                    # Common transient statuses: 429 Too Many Requests, 500/503
                    if status in (429, 500, 503) and attempt < 6:
                        backoff_sleep(base=1.0, attempt=attempt)
                        attempt += 1
                        continue
                    raise
                except requests.RequestException:
                    if attempt < 6:
                        backoff_sleep(base=1.0, attempt=attempt)
                        attempt += 1
                        continue
                    raise

            req_count += 1

            rows = extract_rows(payload, query=q)
            total_rows += len(rows)

            if out_jsonl and rows:
                append_jsonl(out_jsonl, rows)

            if out_sqlite and rows:
                inserted = insert_sqlite(out_sqlite, rows)
                total_inserted += inserted

            print(
                f"[{req_count}/{max_requests}] q={q!r} page={page+1}/{per_query_pages} "
                f"startIndex={start_index} rows={len(rows)} inserted={total_inserted if out_sqlite else 'n/a'}"
            )

            time.sleep(sleep_s)

    print("Done.")
    print(f"Total rows collected: {total_rows}")
    if out_sqlite:
        print(f"Total inserted into SQLite: {total_inserted}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None, help="Google API key (recommended).")
    ap.add_argument("--queries", required=True, help="Path to a text file: one query per line.")
    ap.add_argument("--out-jsonl", default="books.jsonl", help="Output JSONL file path (append).")
    ap.add_argument("--out-sqlite", default="books.sqlite", help="Output SQLite DB path.")
    ap.add_argument("--lang", default="en", help="langRestrict, e.g., en. Use empty to disable.")
    ap.add_argument("--print-type", default="books", choices=["books", "magazines", "all"])
    ap.add_argument("--max-results", type=int, default=40, help="Max results per request (<=40).")
    ap.add_argument("--max-requests", type=int, default=200, help="Total API requests to make.")
    ap.add_argument("--per-query-pages", type=int, default=3, help="How many pages per query to fetch.")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests (seconds).")
    ap.add_argument("--shuffle-queries", action="store_true", help="Shuffle query order.")
    args = ap.parse_args()

    if args.max_results < 1 or args.max_results > 40:
        raise ValueError("--max-results must be between 1 and 40")

    lang = args.lang.strip() or None
    queries = read_queries(args.queries)

    collect(
        api_key=args.api_key,
        queries=queries,
        out_jsonl=args.out_jsonl if args.out_jsonl else None,
        out_sqlite=args.out_sqlite if args.out_sqlite else None,
        lang=lang,
        print_type=args.print_type,
        max_results_per_request=args.max_results,
        max_requests=args.max_requests,
        per_query_pages=args.per_query_pages,
        sleep_s=args.sleep,
        shuffle_queries=args.shuffle_queries,
    )


if __name__ == "__main__":
    main()