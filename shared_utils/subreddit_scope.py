"""
Shared subreddit-scope helpers for the frequency and hit matrix scripts.

Both scripts import ``load_subreddits`` / ``keep`` from here so they always
see the same subset of the corpus. The subreddit filter is applied first, at
the corpus reader, before any prefilter, counting, or postfilter — so every
later threshold applies to the chosen scope.

Reddit subreddit names are case-insensitive, so all matching is done on the
lowercase form.
"""

import hashlib
import re

__all__ = ["keep", "load_subreddits", "normalize_subreddit", "subreddit_hash"]

# Strips a leading "r/" or "/r/" (any case) from a subreddit name.
_PREFIX_RE = re.compile(r"^/?r/", flags=re.IGNORECASE)


def normalize_subreddit(name):
    """Normalize a subreddit name: strip an r/ prefix, trim, lowercase."""
    return _PREFIX_RE.sub("", name.strip()).lower()


def load_subreddits(path):
    """
    Return a set of normalized subreddit names, or ``None`` for all of Reddit.

    File format: one subreddit per line. Blank lines and lines starting with
    ``#`` are ignored. Names are normalized (``r/``/``/r/`` prefix stripped,
    whitespace trimmed, lowercased).

    Passing ``path=None`` means "count the whole corpus" and returns ``None``.
    """
    if path is None:
        return None

    subs = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):  # skip blanks and comments
                continue
            subs.add(normalize_subreddit(s))
    return subs


def subreddit_hash(subs):
    """
    Stable hash of a normalized subreddit set, or ``None`` for all of Reddit.

    The hit script records this in its own metadata and compares it against
    the frequency matrix's ``freq_meta.json`` to confirm both ran on the same
    scope.
    """
    if subs is None:
        return None
    joined = "\n".join(sorted(subs)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def keep(record_subreddit, allowed):
    """
    Return True if a record's subreddit is in scope.

    ``allowed`` is the set returned by ``load_subreddits`` (or ``None`` for
    all of Reddit). Matching is case-insensitive.
    """
    if allowed is None:
        return True
    if record_subreddit is None:
        return False
    return normalize_subreddit(record_subreddit) in allowed
