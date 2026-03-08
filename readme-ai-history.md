# ai-related-history

A qutebrowser command that finds pages in your browsing history that are semantically similar to the page you're currently on — using a local embedding model, no API key, no network calls.

## User Story

> As a user researching a topic, I want to find pages I've previously visited that are related to what I'm currently reading, so that I can quickly revisit relevant sources without having to remember exact titles or URLs.

Browser history search is keyword-based: if you don't remember a specific word from the title or URL, you won't find it. This feature closes that gap, i.e. the current page itself becomes the search query, and results surface by topic rather than by text match.

---

## Usage

Run `:ai-related-history` on any page. The command embeds the current page's title and domain, compares it against the 100 most recent history entries using cosine similarity, and opens a new tab at `qute://ai-related-history/` showing up to 5 matches ranked by similarity score. Only results with a similarity score of 0.7 or higher are shown — pages below that threshold are excluded.

The embedding model ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) via [fastembed](https://github.com/qdrant/fastembed), ~130 MB) is downloaded once on first run and cached locally. To hide the startup latency, it pre-loads in a background thread when the browser starts.

> `fastembed` is used and is included in the project's dependencies and will be installed automatically during setup.

---

## How to Run It

### Setup

Clone or pull the latest version of the code, then create a virtualenv and install dependencies from the repo directory:

```bash
git pull
python scripts/mkvenv.py
```

Start the browser with:

```bash
.venv/bin/python3 -m qutebrowser
```

### Using the feature

Visit any website of your choice. Once the page has loaded, open the qutebrowser command prompt and run:

```
:ai-related-history
```

This will open a new tab with history results ranked by semantic similarity to the current page.

> **Note:** The embedding model (~130 MB) is downloaded the first time the command is executed. This is a one-time download and may take 1–2 minutes depending on your connection. Subsequent runs use the cached model and are fast.

### Tested behavior

Functional behavior is covered by:

```
tests/unit/browser/test_ai_history_functional.py
```

---

## Design Decisions & Tradeoffs

### Local-only embeddings via fastembed (ONNX)

The feature runs fully offline — no API key, no data leaves the machine. This is consistent with qutebrowser's privacy ethos. The tradeoff is a one-time ~130 MB model download and a short startup cost on the first run.

### Title + domain, not full page text

Embedding the full page body would provide a much better semantic signal. However, that would require extracting body text for every history entry, and that data isn't currently stored in the history DB. Since titles are already available, we can use them to provide the semantic signal needed to distinguish content categories.

### No embedding persistence

Caching vectors to disk adds a storage schema, invalidation logic, and migration concerns. Re-embedding 100 short strings takes under a second on commodity hardware — not worth the complexity at this scale.

### Background threads + Qt signals for main-thread delivery

All embedding work runs in daemon threads to avoid blocking the Qt event loop. Results are delivered back to the main thread via `pyqtSignal` emission, which is thread-safe in PyQt6. The alternative (`QTimer.singleShot` from a background thread) is silently dropped in PyQt6 — this was a real bug that informed the design.

### Global `ai_related_result` as the command↔handler mailbox

The `qute://` handler is a stateless function; it can't receive arguments directly. The simplest bridge is a module-level variable that the command writes before navigating to the result URL. The tradeoff: only one result exists at a time. If the user triggers the command twice in quick succession, the second result clobbers the first before the tab opens. Acceptable for v1, but fragile.

### Similarity threshold (0.7) filters noise

Without a floor, unrelated pages fill the results just because they're the closest available. The threshold is a hardcoded constant for now — the right value depends on the model and the user's history. Could be tuned with user feedback.

---

## What's Next

**Per-request result IDs** — Pass a UUID as a query parameter in the `qute://` URL and store results in a dict keyed by ID. Eliminates the global state race condition and makes multiple simultaneous queries safe.

**Full page text embedding** — Inject a JS snippet to extract `document.body.innerText` from the current tab and embed that instead of (or alongside) the title. Would dramatically improve match quality for pages with generic titles.

**Persist embeddings** — Cache computed embeddings to disk (keyed by URL + title). Re-embedding on every invocation is fast today but would become a bottleneck with larger history or if a larger model is used.

**Configurable knobs** — Expose `HISTORY_LIMIT`, `TOP_K`, and `SIMILARITY_THRESHOLD` as qutebrowser config options so users can tune behaviour without touching source.