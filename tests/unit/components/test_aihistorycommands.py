"""Tests for qutebrowser.components.aihistorycommands."""

import logging
from unittest import mock

import pytest

from qutebrowser.qt.core import QUrl

from qutebrowser.api import cmdutils
from qutebrowser.browser import ai_history
from qutebrowser.components import aihistorycommands
from qutebrowser.utils import usertypes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tab(url="https://example.com/page", title="Example Page"):
    """Return a lightweight mock that quacks like an apitypes.Tab."""
    tab = mock.Mock()
    qurl = QUrl(url)
    tab.url.return_value = qurl
    tab.title.return_value = title
    return tab


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestAiRelatedHistoryErrors:

    def test_no_tab_raises(self):
        """Calling with tab=None should raise CommandError."""
        with pytest.raises(
            cmdutils.CommandError,
            match="No current tab",
        ):
            aihistorycommands.ai_related_history(tab=None, win_id=0)

    def test_empty_title_and_domain_raises(self):
        """A tab whose title AND domain are both empty should raise."""
        tab = _make_tab(url="", title="")
        with pytest.raises(
            cmdutils.CommandError,
            match="no title or URL",
        ):
            aihistorycommands.ai_related_history(tab=tab, win_id=0)


# ---------------------------------------------------------------------------
# Happy path – find_related is called correctly
# ---------------------------------------------------------------------------

class TestAiRelatedHistoryCallsFindRelated:

    def test_find_related_called_with_correct_args(self, mocker, message_mock):
        """Verify that ai_history.find_related receives the right kwargs."""
        find_related = mocker.patch.object(ai_history, "find_related")

        tab = _make_tab(
            url="https://docs.python.org/3/library/os.html",
            title="os — Miscellaneous",
        )

        aihistorycommands.ai_related_history(tab=tab, win_id=0)

        find_related.assert_called_once()
        kwargs = find_related.call_args[1]
        assert kwargs["current_title"] == "os — Miscellaneous"
        assert kwargs["current_domain"] == "docs.python.org"
        assert kwargs["current_url"] == "https://docs.python.org/3/library/os.html"
        assert callable(kwargs["on_done"])
        assert callable(kwargs["on_error"])

    def test_info_message_shown(self, mocker, message_mock):
        """An info message should be displayed while results load."""
        mocker.patch.object(ai_history, "find_related")
        tab = _make_tab()

        aihistorycommands.ai_related_history(tab=tab, win_id=0)

        msg = message_mock.getmsg(usertypes.MessageLevel.info)
        assert "related history" in msg.text.lower() or "embedding" in msg.text.lower()


# ---------------------------------------------------------------------------
# on_done callback
# ---------------------------------------------------------------------------

class TestOnDoneCallback:

    def _invoke_on_done(self, mocker, results, win_id=0):
        """Call ai_related_history, capture the on_done callback, invoke it."""
        find_related = mocker.patch.object(ai_history, "find_related")
        tab = _make_tab(
            url="https://example.com/page",
            title="Example Page",
        )
        aihistorycommands.ai_related_history(tab=tab, win_id=win_id)

        on_done = find_related.call_args[1]["on_done"]
        on_done(results)
        return on_done

    def test_sets_ai_related_result(self, mocker, message_mock,
                                     tabbed_browser_stubs):
        """on_done should populate ai_history.ai_related_result."""
        results = [{"title": "Hit", "url": "https://hit.example", "score": 0.9}]
        self._invoke_on_done(mocker, results, win_id=0)

        assert ai_history.ai_related_result is not None
        assert ai_history.ai_related_result["current_title"] == "Example Page"
        assert ai_history.ai_related_result["current_url"] == "https://example.com/page"
        assert ai_history.ai_related_result["results"] is results

    def test_opens_qute_url_in_new_tab(self, mocker, message_mock,
                                        tabbed_browser_stubs):
        """on_done should open qute://ai-related-history/ in a new tab."""
        self._invoke_on_done(mocker, [], win_id=0)

        loaded = tabbed_browser_stubs[0].loaded_url
        assert loaded is not None
        assert loaded.toString() == "qute://ai-related-history/"

    def test_on_done_exception_shows_error(self, mocker, message_mock, caplog):
        """If objreg.get fails inside on_done, an error message is shown."""
        find_related = mocker.patch.object(ai_history, "find_related")
        # Don't register a tabbed-browser so objreg.get will raise
        tab = _make_tab()
        aihistorycommands.ai_related_history(tab=tab, win_id=999)

        on_done = find_related.call_args[1]["on_done"]
        with caplog.at_level(logging.ERROR, 'message'):
            with caplog.at_level(logging.ERROR, 'ai_history'):
                on_done([])

        # First message is the info toast; the error comes second.
        errors = [
            m for m in message_mock.messages
            if m.level == usertypes.MessageLevel.error
        ]
        assert len(errors) == 1
        assert "ai-related-history" in errors[0].text


# ---------------------------------------------------------------------------
# on_error callback
# ---------------------------------------------------------------------------

class TestOnErrorCallback:

    def test_on_error_shows_error_message(self, mocker, message_mock, caplog):
        """on_error should relay the error string to message.error."""
        find_related = mocker.patch.object(ai_history, "find_related")
        tab = _make_tab()
        aihistorycommands.ai_related_history(tab=tab, win_id=0)

        on_error = find_related.call_args[1]["on_error"]
        with caplog.at_level(logging.ERROR, 'message'):
            on_error("something broke")

        errors = [
            m for m in message_mock.messages
            if m.level == usertypes.MessageLevel.error
        ]
        assert len(errors) == 1
        assert "something broke" in errors[0].text
