"""AI-powered related history search using semantic similarity via local embeddings.

Embeds page title and domain of current page and history entries in-memory using
fastembed.

fastembed uses the default model BAAI/bge-small-en-v1.5, which is downloaded once
on first use.
"""

from typing import List
from urllib.parse import urlparse
import numpy as np
from qutebrowser.browser.history import web_history


HISTORY_LIMIT = 100  # Max most recent number of history entries for similarity search
TOP_K = 5  # Number of top similar entries to return
SIMILARITY_THRESHOLD = 0.7  # Minimum cosine similarity to consider an entry relevant


def get_domain(url: str) -> str:
    """Extract the domain name from a URL string.

    >>> get_domain("https://docs.python.org/3/library/os.html")
    'docs.python.org'
    >>> domain("not-a-url")
    'not-a-url'
    """
    try:
        host = urlparse(url).hostname
        return host if host else url
    except Exception:
        return url


def rank_entries(
    current_vec: np.ndarray, entry_vecs: list[np.ndarray], entries: list[dict]
) -> list[dict]:
    """Rank history entries by cosine similarity to the current page vector.

    Only returns entries with similarity above SIMILARITY_THRESHOLD and
    at most TOP_K entries.

    Args:
        current_vec (np.ndarray): Embedding vector of the current page.
        entry_vecs (list[np.ndarray]): List of embedding vectors for history entries.
        entries (list[dict]): List of history entry dicts corresponding to entry_vecs.
        Returns:
            list[dict]: List of top similar history entries with added "score" key.
    """
    scored = sorted(
        ((np.dot(current_vec, vec), entry) for vec, entry in zip(entry_vecs, entries)),
        key=lambda x: x[0],
        reverse=True,
    )
    return [
        {**entry, "score": round(score, 3)}
        for score, entry in scored[:TOP_K]
        if score >= SIMILARITY_THRESHOLD
    ]


def fetch_history_entries(exclude_url: str, limit=HISTORY_LIMIT) -> List[dict]:
    """Fetch max HISTORY_LIMIT most recent history entries, excluding those with the
    same URL as the current page.

    Returns:
        List[dict]: List of history entries with keys "title", "url", and "domain".
    """
    entries = web_history.select(sort_by="atime", sort_order="desc", limit=limit)

    seen_keys = set()
    history_entries = []

    for entry in entries:
        if not entry.title:
            continue
        if entry.url == exclude_url:
            continue
        domain = get_domain(entry.url)
        key = (entry.title, domain)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        history_entries.append(
            {
                "title": entry.title,
                "url": entry.url,
                "domain": domain,
            }
        )
    return history_entries
