"""Command: :ai-related-history — find similar pages in browsing history."""

import logging

from qutebrowser.qt.core import QUrl

from qutebrowser.api import cmdutils, apitypes, message
from qutebrowser.browser import ai_history
from qutebrowser.utils import objreg

log = logging.getLogger('ai_history')


@cmdutils.register()
@cmdutils.argument("tab", value=cmdutils.Value.cur_tab)
@cmdutils.argument("win_id", value=cmdutils.Value.win_id)
def ai_related_history(tab: apitypes.Tab, win_id: int) -> None:
    """Find the most similar pages to the current one in your history.

    Uses a local embedding model (fastembed) — no API key required.
    The model is downloaded once on first use (~130 MB).
    Results open in a new tab at qute://ai-related-history/.
    """
    if tab is None:
        raise cmdutils.CommandError("No current tab to find related history for.")

    current_url = tab.url()
    current_title = tab.title()
    current_domain = current_url.host()
    current_url_str = current_url.toDisplayString()

    if not current_title and not current_domain:
        raise cmdutils.CommandError("Current page has no title or URL to match against.")

    message.info("Finding related history… (embedding model may download on first run)")

    def on_done(results: list[dict]) -> None:
        try:
            ai_history.ai_related_result = {
                "current_title": current_title,
                "current_url": current_url_str,
                "results": results,
            }
            tabbed_browser = objreg.get("tabbed-browser", scope="window", window=win_id)
            tabbed_browser.load_url(QUrl("qute://ai-related-history/"), newtab=True)
        except Exception as exc:
            log.exception("ai_history: on_done failed")
            message.error(f"ai-related-history: {exc}")

    def on_error(err: str) -> None:
        message.error(f"ai-related-history: {err}")

    ai_history.find_related(
        current_title=current_title,
        current_domain=current_domain,
        current_url=current_url_str,
        on_done=on_done,
        on_error=on_error,
    )