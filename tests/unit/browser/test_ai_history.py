import logging
import threading
from collections import namedtuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qutebrowser.browser import ai_history

HistoryEntry = namedtuple("HistoryEntry", ["url", "title"])


class TestDomain:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://docs.python.org/3/library/os.html", "docs.python.org"),
            ("http://example.com", "example.com"),
            ("https://example.com/path?q=1#frag", "example.com"),
            ("not-a-url", "not-a-url"),
            ("", ""),
        ],
    )
    def test_extracts_hostname(self, url, expected):
        assert ai_history.get_domain(url) == expected


class TestFetchHistory:
    def test_fetch_history_entries(self, mocker):
        """Regression: entries_before() requires offset — must not be omitted."""
        mock_web_history = mocker.MagicMock()
        mock_web_history.web_history.select.return_value = [
            HistoryEntry(url="https://example.com", title="Example"),
            HistoryEntry(url="https://example.com", title="Example Other"),
            HistoryEntry(url="https://example.com/subpage", title="Example Subpage"),
            HistoryEntry(url="https://example.org", title="Example Org"),
            HistoryEntry(url="https://example.net", title="Example Net"),
        ]
        mocker.patch.object(ai_history, "history", mock_web_history)

        entries = ai_history.fetch_history_entries(exclude_url="https://example.com")
        assert entries == [
            {
                "url": "https://example.com/subpage",
                "title": "Example Subpage",
                "domain": "example.com",
            },
            {
                "url": "https://example.org",
                "title": "Example Org",
                "domain": "example.org",
            },
            {
                "url": "https://example.net",
                "title": "Example Net",
                "domain": "example.net",
            },
        ]


class TestRankEntries:
    """Tests for rank_entries cosine-similarity ranking."""

    def _make_vec(self, *components):
        """Create a normalized vector from components."""
        v = np.array(components, dtype=float)
        return v / np.linalg.norm(v)

    def test_returns_entries_sorted_by_similarity(self):
        current = self._make_vec(1, 0, 0)
        vecs = [
            self._make_vec(0.5, 0.5, 0),  # medium similarity
            self._make_vec(1, 0, 0),  # perfect similarity
            self._make_vec(0.9, 0.1, 0),  # high similarity
        ]
        entries = [
            {"title": "Medium", "url": "http://m.com", "domain": "m.com"},
            {"title": "Perfect", "url": "http://p.com", "domain": "p.com"},
            {"title": "High", "url": "http://h.com", "domain": "h.com"},
        ]

        result = ai_history.rank_entries(current, vecs, entries)

        assert result[0]["title"] == "Perfect"
        assert result[1]["title"] == "High"
        assert result[2]["title"] == "Medium"

    def test_adds_score_key(self):
        current = self._make_vec(1, 0, 0)
        vecs = [self._make_vec(1, 0, 0)]
        entries = [{"title": "A", "url": "http://a.com", "domain": "a.com"}]

        result = ai_history.rank_entries(current, vecs, entries)

        assert len(result) == 1
        assert "score" in result[0]
        assert result[0]["score"] == 1.0

    def test_filters_below_threshold(self):
        current = self._make_vec(1, 0, 0)
        vecs = [
            self._make_vec(1, 0, 0),  # score ~1.0, above threshold
            self._make_vec(0, 1, 0),  # score ~0.0, below threshold
            self._make_vec(0, 0, 1),  # score ~0.0, below threshold
        ]
        entries = [
            {"title": "Match", "url": "http://a.com", "domain": "a.com"},
            {"title": "Unrelated1", "url": "http://b.com", "domain": "b.com"},
            {"title": "Unrelated2", "url": "http://c.com", "domain": "c.com"},
        ]

        result = ai_history.rank_entries(current, vecs, entries)

        assert len(result) == 1
        assert result[0]["title"] == "Match"

    def test_limits_to_top_k(self, monkeypatch):
        monkeypatch.setattr(ai_history, "TOP_K", 2)
        current = self._make_vec(1, 0, 0)
        # All vectors similar to current
        vecs = [self._make_vec(1, 0.01 * i, 0) for i in range(5)]
        entries = [
            {"title": f"E{i}", "url": f"http://e{i}.com", "domain": f"e{i}.com"}
            for i in range(5)
        ]

        result = ai_history.rank_entries(current, vecs, entries)

        assert len(result) <= 2

    def test_empty_entries(self):
        current = self._make_vec(1, 0, 0)

        result = ai_history.rank_entries(current, [], [])

        assert result == []

    def test_all_below_threshold_returns_empty(self):
        current = self._make_vec(1, 0, 0)
        vecs = [
            self._make_vec(0, 1, 0),
            self._make_vec(0, 0, 1),
        ]
        entries = [
            {"title": "A", "url": "http://a.com", "domain": "a.com"},
            {"title": "B", "url": "http://b.com", "domain": "b.com"},
        ]

        result = ai_history.rank_entries(current, vecs, entries)

        assert result == []

    def test_score_is_rounded_to_three_decimals(self):
        current = self._make_vec(1, 0, 0)
        vecs = [self._make_vec(0.9, 0.1, 0)]
        entries = [{"title": "A", "url": "http://a.com", "domain": "a.com"}]

        result = ai_history.rank_entries(current, vecs, entries)

        assert len(result) == 1
        score_str = str(result[0]["score"])
        # At most 3 decimal places
        if "." in score_str:
            assert len(score_str.split(".")[1]) <= 3

    def test_preserves_original_entry_fields(self):
        current = self._make_vec(1, 0, 0)
        vecs = [self._make_vec(1, 0, 0)]
        entries = [{"title": "My Page", "url": "http://x.com/p", "domain": "x.com"}]

        result = ai_history.rank_entries(current, vecs, entries)

        assert result[0]["title"] == "My Page"
        assert result[0]["url"] == "http://x.com/p"
        assert result[0]["domain"] == "x.com"


class TestFindRelated:
    """Tests for find_related — the main entry point that orchestrates
    history fetching, embedding, ranking, and callback delivery."""

    @pytest.fixture(autouse=True)
    def sync_dispatcher(self, mocker):
        """Replace Qt _Dispatcher with a synchronous stub.

        Signals emitted from the background thread are delivered directly to
        connected callbacks without needing a running Qt event loop.
        """
        class _FakeSignal:
            def __init__(self):
                self._callbacks = []

            def connect(self, cb):
                self._callbacks.append(cb)

            def emit(self, *args):
                for cb in self._callbacks:
                    cb(*args)

        class _FakeDispatcher:
            def __init__(self):
                self.done = _FakeSignal()
                self.error = _FakeSignal()

        mocker.patch.object(ai_history, "_Dispatcher", _FakeDispatcher)

    @pytest.fixture()
    def history_entries(self):
        """A small list of pre-built history entry dicts."""
        return [
            {"title": "Python Docs", "url": "https://docs.python.org", "domain": "docs.python.org"},
            {"title": "NumPy", "url": "https://numpy.org", "domain": "numpy.org"},
        ]

    @pytest.fixture()
    def fake_model(self):
        """A mock embedding model whose embed() returns normalised unit vectors."""
        model = MagicMock()

        def _embed(texts):
            """Return one normalised vector per text."""
            for i, _ in enumerate(texts):
                v = np.zeros(3)
                v[i % 3] = 1.0
                yield v

        model.embed.side_effect = _embed
        return model

    @pytest.fixture()
    def callbacks(self):
        """A pair of (on_done, on_error) callbacks that record their calls
        and signal a threading.Event so the test can wait."""
        done_event = threading.Event()
        error_event = threading.Event()

        results = {}

        def on_done(data):
            results["done"] = data
            done_event.set()

        def on_error(msg):
            results["error"] = msg
            error_event.set()

        return on_done, on_error, done_event, error_event, results

    # ── success path ────────────────────────────────────────────────

    def test_success_calls_on_done(self, mocker, callbacks, history_entries, fake_model):
        """Happy path: on_done is invoked with ranked results."""
        on_done, on_error, done_event, error_event, results = callbacks

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=history_entries)
        mocker.patch.object(ai_history, "_get_model", return_value=fake_model)

        # rank_entries is deterministic; stub it to return entries unchanged
        expected = [{"title": "Python Docs", "score": 0.95}]
        mocker.patch.object(ai_history, "rank_entries", return_value=expected)

        ai_history.find_related(
            current_title="Test Page",
            current_domain="test.com",
            current_url="https://test.com",
            on_done=on_done,
            on_error=on_error,
        )

        assert done_event.wait(timeout=5), "on_done was never called"
        assert results["done"] == expected

    def test_calls_fetch_history_with_current_url(self, mocker, callbacks, fake_model):
        """fetch_history_entries must receive the current URL to exclude it."""
        on_done, on_error, done_event, _, _ = callbacks

        fetch_mock = mocker.patch.object(
            ai_history, "fetch_history_entries", return_value=[]
        )
        mocker.patch.object(ai_history, "_get_model", return_value=fake_model)
        mocker.patch.object(ai_history, "rank_entries", return_value=[])

        ai_history.find_related(
            current_title="T",
            current_domain="d.com",
            current_url="https://d.com/page",
            on_done=on_done,
            on_error=on_error,
        )

        done_event.wait(timeout=5)
        fetch_mock.assert_called_once_with(exclude_url="https://d.com/page")

    def test_embeds_current_and_history_texts(self, mocker, callbacks, history_entries, fake_model):
        """The model should receive [current_text] + history_texts."""
        on_done, on_error, done_event, _, _ = callbacks

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=history_entries)
        mocker.patch.object(ai_history, "_get_model", return_value=fake_model)
        mocker.patch.object(ai_history, "rank_entries", return_value=[])

        ai_history.find_related(
            current_title="My Page",
            current_domain="my.com",
            current_url="https://my.com",
            on_done=on_done,
            on_error=on_error,
        )

        done_event.wait(timeout=5)
        call_args = fake_model.embed.call_args[0][0]
        assert call_args[0] == "My Page my.com"
        assert call_args[1] == "Python Docs docs.python.org"
        assert call_args[2] == "NumPy numpy.org"

    def test_passes_vectors_to_rank_entries(self, mocker, callbacks, history_entries):
        """rank_entries must receive (current_vec, entry_vecs, entries)."""
        on_done, on_error, done_event, _, _ = callbacks

        # Produce distinct, stable vectors
        vectors = [np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0])]

        model = MagicMock()
        model.embed.return_value = iter(vectors)

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=history_entries)
        mocker.patch.object(ai_history, "_get_model", return_value=model)
        rank_mock = mocker.patch.object(ai_history, "rank_entries", return_value=[])

        ai_history.find_related(
            current_title="T", current_domain="d.com", current_url="https://d.com",
            on_done=on_done, on_error=on_error,
        )

        done_event.wait(timeout=5)
        args = rank_mock.call_args
        np.testing.assert_array_equal(args[0][0], vectors[0])  # current_vec
        assert len(args[0][1]) == 2  # entry_vecs (one per history entry)
        assert args[0][2] is history_entries  # entries passed through

    # ── fetch failure ───────────────────────────────────────────────

    def test_fetch_failure_calls_on_error(self, mocker, callbacks, caplog):
        """If fetch_history_entries raises, on_error is called immediately."""
        on_done, on_error, done_event, error_event, results = callbacks
        caplog.set_level(logging.ERROR, logger="ai_history")

        mocker.patch.object(
            ai_history, "fetch_history_entries",
            side_effect=RuntimeError("db locked"),
        )

        ai_history.find_related(
            current_title="T", current_domain="d.com", current_url="https://d.com",
            on_done=on_done, on_error=on_error,
        )

        # on_error is called synchronously for fetch failures, but check via event
        assert error_event.wait(timeout=5), "on_error was never called"
        assert "db locked" in results["error"]

    def test_fetch_failure_does_not_call_on_done(self, mocker, callbacks, caplog):
        """on_done must NOT be called when fetching fails."""
        on_done, on_error, done_event, error_event, results = callbacks
        caplog.set_level(logging.ERROR, logger="ai_history")

        mocker.patch.object(
            ai_history, "fetch_history_entries",
            side_effect=RuntimeError("fail"),
        )

        ai_history.find_related(
            current_title="T", current_domain="d.com", current_url="https://d.com",
            on_done=on_done, on_error=on_error,
        )

        error_event.wait(timeout=5)
        assert "done" not in results

    # ── import error (fastembed not installed) ──────────────────────

    def test_import_error_calls_on_error_with_install_hint(self, mocker, callbacks):
        """If fastembed is missing, on_error should mention pip install."""
        on_done, on_error, done_event, error_event, results = callbacks

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=[])
        mocker.patch.object(
            ai_history, "_get_model", side_effect=ImportError("no fastembed"),
        )

        ai_history.find_related(
            current_title="T", current_domain="d.com", current_url="https://d.com",
            on_done=on_done, on_error=on_error,
        )

        assert error_event.wait(timeout=5)
        assert "fastembed" in results["error"].lower()
        assert "pip install" in results["error"].lower()

    # ── generic thread exception ────────────────────────────────────

    def test_thread_exception_calls_on_error(self, mocker, callbacks, caplog):
        """An unexpected exception inside the thread calls on_error."""
        on_done, on_error, done_event, error_event, results = callbacks
        caplog.set_level(logging.ERROR, logger="ai_history")

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=[])
        mocker.patch.object(
            ai_history, "_get_model", side_effect=ValueError("boom"),
        )

        ai_history.find_related(
            current_title="T", current_domain="d.com", current_url="https://d.com",
            on_done=on_done, on_error=on_error,
        )

        assert error_event.wait(timeout=5)
        assert "boom" in results["error"]

    # ── empty history ───────────────────────────────────────────────

    def test_empty_history_returns_empty_results(self, mocker, callbacks):
        """When there are no history entries, on_done receives an empty list."""
        on_done, on_error, done_event, error_event, results = callbacks

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=[])

        model = MagicMock()
        # Only the current page vector (no history texts)
        model.embed.return_value = iter([np.array([1.0, 0.0, 0.0])])
        mocker.patch.object(ai_history, "_get_model", return_value=model)
        mocker.patch.object(ai_history, "rank_entries", return_value=[])

        ai_history.find_related(
            current_title="T", current_domain="d.com", current_url="https://d.com",
            on_done=on_done, on_error=on_error,
        )

        assert done_event.wait(timeout=5)
        assert results["done"] == []

    # ── thread is a daemon ──────────────────────────────────────────

    def test_spawns_daemon_thread(self, mocker, callbacks, fake_model):
        """The background thread should be a daemon so it doesn't block exit."""
        on_done, on_error, done_event, _, _ = callbacks

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=[])
        mocker.patch.object(ai_history, "_get_model", return_value=fake_model)
        mocker.patch.object(ai_history, "rank_entries", return_value=[])

        thread_start = mocker.patch.object(threading.Thread, "start")

        ai_history.find_related(
            current_title="T", current_domain="d.com", current_url="https://d.com",
            on_done=on_done, on_error=on_error,
        )

        thread_start.assert_called_once()
        # Grab the Thread instance from the most recent Thread() call
        thread_cls_call = mocker.patch.object(threading, "Thread")
        # Re-invoke to capture constructor kwargs
        mocker.stopall()

        # Instead, inspect via the original approach
        spawned = []
        original_init = threading.Thread.__init__

        def capture_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            spawned.append(self)

        mocker.patch.object(ai_history, "fetch_history_entries", return_value=[])
        mocker.patch.object(ai_history, "_get_model", return_value=fake_model)
        mocker.patch.object(ai_history, "rank_entries", return_value=[])

        with patch.object(threading.Thread, "__init__", capture_init):
            ai_history.find_related(
                current_title="T", current_domain="d.com", current_url="https://d.com",
                on_done=on_done, on_error=on_error,
            )

        assert len(spawned) == 1
        assert spawned[0].daemon is True
        assert spawned[0].name == "ai-history"
