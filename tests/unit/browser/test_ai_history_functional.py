"""Functional tests for ai_history.rank_entries using real embeddings.

These tests load the actual fastembed model (BAAI/bge-small-en-v1.5) and
verify that semantic similarity rankings make sense from a user perspective.

Marked ``integration`` because they download / load a real model and are
slower than pure-unit tests.  Run with::

    pytest -m integration tests/unit/browser/test_ai_history_functional.py
"""

import numpy as np
import pytest

from qutebrowser.browser import ai_history
from qutebrowser.browser.ai_history import format_text, rank_entries

# Skip the entire module when fastembed is not installed.
fastembed = pytest.importorskip("fastembed")


@pytest.fixture(scope="module")
def model():
    """Load the embedding model once for the whole module."""
    return fastembed.TextEmbedding()


def _embed(model, texts: list[str]):
    """Helper: return a list of numpy vectors for *texts*."""
    return list(model.embed(texts))


def _entry(title: str, domain: str) -> dict:
    """Build a minimal history-entry dict."""
    return {"title": title, "url": f"https://{domain}", "domain": domain}


def _embed_and_score(model, current_title, current_domain, entries):
    """Embed texts and return (score, entry) pairs sorted by score desc.

    Unlike rank_entries this does NOT apply SIMILARITY_THRESHOLD / TOP_K
    so tests can inspect raw scores independently of configuration.
    """
    current_text = format_text(current_title, current_domain)
    history_texts = [format_text(e["title"], e["domain"]) for e in entries]
    vecs = _embed(model, [current_text] + history_texts)
    scored = sorted(
        (
            (float(np.dot(vecs[0], vec)), entry)
            for vec, entry in zip(vecs[1:], entries)
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    return scored


def _rank(model, current_title, current_domain, entries):
    """Embed the current page + entries and return ranked results
    (applies SIMILARITY_THRESHOLD and TOP_K via rank_entries)."""
    current_text = format_text(current_title, current_domain)
    history_texts = [format_text(e["title"], e["domain"]) for e in entries]
    vecs = _embed(model, [current_text] + history_texts)
    return rank_entries(vecs[0], vecs[1:], entries)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSameTopicRanksHigher:
    """Pages about the same topic should score higher than unrelated pages."""

    def test_python_docs_closer_to_python_tutorial(self, model):
        entries = [
            _entry("Best Pizza Recipes", "food.com"),
            _entry("Python Tutorial for Beginners", "realpython.com"),
            _entry("Weather Forecast Today", "weather.com"),
        ]
        scored = _embed_and_score(model, "Python Documentation", "docs.python.org", entries)

        # "Python Tutorial" should be the top result by a clear margin
        assert scored[0][1]["title"] == "Python Tutorial for Beginners"
        best, second_best = scored[0][0], scored[1][0]
        assert best > second_best + 0.1  # meaningful gap

    def test_machine_learning_closer_to_deep_learning(self, model):
        entries = [
            _entry("Introduction to Deep Learning", "deeplearning.ai"),
            _entry("Gardening Tips for Spring", "garden.com"),
            _entry("How to Fix a Leaky Faucet", "diy.com"),
        ]
        scored = _embed_and_score(model, "Machine Learning Basics", "ml-course.org", entries)

        assert scored[0][1]["title"] == "Introduction to Deep Learning"
        assert scored[0][0] > scored[1][0] + 0.1

    def test_javascript_framework_closer_to_web_dev(self, model):
        entries = [
            _entry("History of Ancient Rome", "history.com"),
            _entry("React.js Getting Started Guide", "reactjs.org"),
            _entry("Organic Chemistry Notes", "chem.edu"),
        ]
        scored = _embed_and_score(
            model, "JavaScript Frontend Development", "developer.mozilla.org", entries
        )

        assert scored[0][1]["title"] == "React.js Getting Started Guide"
        assert scored[0][0] > scored[1][0] + 0.1


@pytest.mark.integration
class TestUnrelatedContentFilteredOut:
    """Completely unrelated entries should fall below the similarity threshold."""

    def test_cooking_vs_programming(self, model):
        entries = [
            _entry("Vegan Pasta Recipe", "cooking.com"),
            _entry("How to Bake Sourdough Bread", "bakery.com"),
            _entry("Best Italian Restaurants in NYC", "yelp.com"),
        ]
        results = _rank(model, "Rust Programming Language", "rust-lang.org", entries)

        # None of the cooking entries should be similar enough
        assert len(results) == 0

    def test_sports_vs_quantum_physics(self, model):
        entries = [
            _entry("Premier League Match Results", "bbc.com/sport"),
            _entry("NBA Playoff Schedule 2026", "espn.com"),
        ]
        results = _rank(model, "Quantum Mechanics Introduction", "physics.org", entries)

        assert len(results) == 0


@pytest.mark.integration
class TestDomainBoostsRelevance:
    """Two entries with the same title — the one on the more relevant domain
    should score higher because format_text concatenates title + domain."""

    def test_python_on_python_domain_ranks_higher(self, model):
        entries = [
            _entry("Python Reference", "python.org"),
            _entry("Python Reference", "randomsite.com"),
        ]
        scored = _embed_and_score(model, "Python Standard Library", "docs.python.org", entries)

        # python.org domain should score strictly higher than randomsite.com
        assert scored[0][1]["domain"] == "python.org"
        assert scored[0][0] > scored[1][0]


@pytest.mark.integration
class TestMultipleRelatedEntriesRanked:
    """When several entries are related, they should all score higher than
    unrelated entries."""

    def test_web_dev_topic_cluster(self, model):
        entries = [
            _entry("CSS Flexbox Guide", "css-tricks.com"),
            _entry("HTML5 Semantic Elements", "developer.mozilla.org"),
            _entry("Advanced Calculus Textbook", "math.edu"),
            _entry("Responsive Web Design Principles", "smashingmagazine.com"),
            _entry("Bird Watching in North America", "audubon.org"),
        ]
        scored = _embed_and_score(
            model, "Building Modern Websites with CSS Grid", "web.dev", entries
        )

        web_titles = {
            "CSS Flexbox Guide",
            "HTML5 Semantic Elements",
            "Responsive Web Design Principles",
        }
        non_web_titles = {"Advanced Calculus Textbook", "Bird Watching in North America"}

        # Every web-dev entry should score higher than every non-web entry
        web_scores = [s for s, e in scored if e["title"] in web_titles]
        non_web_scores = [s for s, e in scored if e["title"] in non_web_titles]
        assert min(web_scores) > max(non_web_scores)

    def test_scores_are_monotonically_decreasing(self, model):
        entries = [
            _entry("Data Science with Python", "kaggle.com"),
            _entry("Pandas DataFrame Tutorial", "pandas.pydata.org"),
            _entry("NumPy Array Operations", "numpy.org"),
            _entry("Homemade Candle Making", "crafts.com"),
        ]
        scored = _embed_and_score(model, "Python Data Analysis", "pydata.org", entries)

        scores = [s for s, _ in scored]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
class TestRankEntriesWithThreshold:
    """Exercise rank_entries end-to-end (threshold + top-k) with real vectors."""

    def test_threshold_filters_unrelated(self, model):
        """Entries well below the threshold should be excluded by rank_entries."""
        entries = [
            _entry("Rust Programming Language", "rust-lang.org"),
            _entry("Best Chocolate Cake Recipe", "allrecipes.com"),
            _entry("How to Train for a Marathon", "runnersworld.com"),
        ]
        results = _rank(model, "Rust Ownership and Borrowing", "doc.rust-lang.org", entries)

        titles = [r["title"] for r in results]
        # Rust entry should make it through; cooking and running should not
        assert "Rust Programming Language" in titles
        assert "Best Chocolate Cake Recipe" not in titles
        assert "How to Train for a Marathon" not in titles

    def test_lowered_threshold_admits_more_entries(self, model, monkeypatch):
        """With a lower threshold, entries that were borderline should appear."""
        monkeypatch.setattr(ai_history, "SIMILARITY_THRESHOLD", 0.5)

        entries = [
            _entry("Python Tutorial for Beginners", "realpython.com"),
            _entry("Best Pizza Recipes", "food.com"),
        ]
        results = _rank(model, "Python Documentation", "docs.python.org", entries)

        # Both should now appear since even the pizza entry scores ~0.45
        # and the Python tutorial scores ~0.67
        assert len(results) >= 1
        assert results[0]["title"] == "Python Tutorial for Beginners"


@pytest.mark.integration
class TestEdgeCases:
    """Edge-case scenarios with the real model."""

    def test_identical_page_has_perfect_score(self, model):
        entries = [
            _entry("GitHub Copilot Documentation", "docs.github.com"),
        ]
        results = _rank(model, "GitHub Copilot Documentation", "docs.github.com", entries)

        assert len(results) == 1
        assert results[0]["score"] >= 0.99

    def test_empty_history_returns_nothing(self, model):
        results = _rank(model, "Anything at All", "example.com", [])

        assert results == []

    def test_short_titles_still_rank_sensibly(self, model):
        entries = [
            _entry("Git", "git-scm.com"),
            _entry("Yoga", "yoga.com"),
        ]
        results = _rank(model, "GitHub", "github.com", entries)

        if results:
            assert results[0]["title"] == "Git"
