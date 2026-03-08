"""AI-powered related history search using semantic similarity via local embeddings.

Embeds page title and domain of current page and history entries in-memory using
fastembed.

fastembed uses the default model BAAI/bge-small-en-v1.5, which is downloaded once
on first use.
"""

import logging
import threading
from typing import Any, Callable, List
from urllib.parse import urlparse
import numpy as np
from qutebrowser.browser import history
from qutebrowser.qt.core import pyqtSignal, QObject

# Model cache — loaded lazily on first command invocation.
_model = None
_model_lock = threading.Lock()
log = logging.getLogger('ai_history')


HISTORY_LIMIT = 100  # Max most recent number of history entries for similarity search
TOP_K = 5  # Number of top similar entries to return
SIMILARITY_THRESHOLD = 0.7  # Minimum cosine similarity to consider an entry relevant


# Shared state written by the command, read by the qute:// handler.
ai_related_result: dict | None = None


# Model cache — loaded lazily on first command invocation.
_model = None
_model_lock = threading.Lock()


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
    entries = history.web_history.select(sort_by="atime", sort_order="desc", limit=limit)

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


class _Dispatcher(QObject):
    """Ferries results from a background thread to the Qt main thread."""

    done = pyqtSignal(object)
    error = pyqtSignal(str)


def format_text(title: str, url_domain: str) -> str:
    return f"{title} {url_domain}".strip()


def _get_model() -> Any:
    """Return the cached model, loading it on first call (thread-safe)."""
    global _model
    with _model_lock:
        if _model is not None:
            return _model

    log.debug("ai_history: importing fastembed")
    from fastembed import TextEmbedding

    log.debug("ai_history: instantiating TextEmbedding (may download model)")
    model = TextEmbedding()

    with _model_lock:
        _model = model
    log.debug("ai_history: embedding model ready")
    return model


def find_related(
    current_title: str,
    current_domain: str,
    current_url: str,
    on_done: Callable[[list[dict]], None],
    on_error: Callable[[str], None],
) -> None:
    """Embed the current page + recent history, then call on_done with top-k.

    Runs in a daemon thread so the Qt event loop is never blocked.
    on_done / on_error are always called on the Qt main thread.

    Args:
        current_title:  Title of the currently open page.
        current_domain: Domain of the currently open page.
        current_url:    Full URL of the current page (used to exclude it from
                        results).
        on_done:  Called with a list of result dicts on success.
        on_error: Called with an error string on failure.
    """
    dispatcher = _Dispatcher()
    dispatcher.done.connect(on_done)
    dispatcher.error.connect(on_error)

    # Fetch history on the main thread because QSqlDatabase connections are
    # thread-local and cannot be used from a background thread.
    try:
        entries = fetch_history_entries(exclude_url=current_url)
    except Exception as exc:
        log.exception("ai_history: fetch_history_entries failed")
        on_error(str(exc))
        return

    log.debug("ai_history: acquired %d history entries", len(entries))

    def _thread_fn() -> None:
        try:
            model = _get_model()

            current_text = format_text(current_title, current_domain)
            history_texts = [format_text(e["title"], e["domain"]) for e in entries]
            all_vecs = list(model.embed([current_text] + history_texts))

            results = rank_entries(all_vecs[0], all_vecs[1:], entries)
            log.debug("ai_history: %d results", len(results))
            dispatcher.done.emit(results)
        except ImportError:
            dispatcher.error.emit(
                "fastembed is not installed. Run: pip install fastembed"
            )
        except Exception as exc:
            log.exception("ai_history: find_related failed")
            dispatcher.error.emit(str(exc))

    threading.Thread(target=_thread_fn, daemon=True, name="ai-history").start()