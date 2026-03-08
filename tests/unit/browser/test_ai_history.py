import pytest
import numpy as np
from qutebrowser.browser import ai_history
from collections import namedtuple

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
        mock_web_history.select.return_value = [
            HistoryEntry(url="https://example.com", title="Example"),
            HistoryEntry(url="https://example.com", title="Example Other"),
            HistoryEntry(url="https://example.com/subpage", title="Example Subpage"),
            HistoryEntry(url="https://example.org", title="Example Org"),
            HistoryEntry(url="https://example.net", title="Example Net"),
        ]
        mocker.patch.object(ai_history, "web_history", mock_web_history)

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
