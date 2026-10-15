"""TranscriptViewer — a loopback HTTP server that hands a finished transcript
to a browser.

Why this exists: the MCP client is a chat window.  Returning a 3,000-word
transcript through it costs the calling agent most of its context and is
awkward to read besides, which is why ``transcript_get`` defaults to a preview.
But the user still wants the whole thing, and telling them to run a command is
a poor answer when the process holding the transcript is already running.  So
a job that finishes also produces a URL: click it, the browser downloads the
file, and it is theirs to read anywhere.

Three properties keep this from being a hole in a project whose premise is that
nothing leaves the machine:

* **Loopback only by default.** ``viewer_host`` is ``127.0.0.1``; nothing off
  this machine can reach it.
* **Unguessable links.** Every URL carries a token generated fresh at startup.
  Job ids are UUIDs and hard to guess already, but a token means another local
  process cannot enumerate transcripts even if it learns one.
* **Nothing new on disk.** The content is rendered from the database on request,
  exactly as ``transcript_get`` renders it. Rule N9 holds: the only file
  Transcriptor writes is the one ``transcript_export`` is explicitly asked for.

The adapter takes a ``fetch`` callable rather than a service, so it depends on
nothing above the adapter layer (N1). ``server.py`` supplies one that closes
over ``ReadService``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import socket
import urllib.error
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote

log = logging.getLogger(__name__)

# What a browser should do with each rendered format. ``json`` is the odd one:
# browsers render it inline rather than downloading, which is usually what you
# want when you are inspecting it.
_MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json; charset=utf-8",
}

# Returned by /healthz so a second instance can tell "another Transcriptor has
# the port" from "something unrelated has the port". The two cases need
# opposite responses and are indistinguishable from the bind error alone.
_MARKER = "transcriptor-viewer"

Fetch = Callable[[str, str], Awaitable[tuple[str, str]]]
"""``(job_id, fmt) -> (content, suggested_filename)``. Raises TranscriptorError."""


class TranscriptViewer:
    """Serves rendered transcripts over loopback HTTP.

    Auxiliary by design: if the port is taken or Starlette is missing, the
    viewer reports itself unavailable and the MCP server carries on. A download
    link is a convenience, and losing it should never cost a transcription.
    """

    def __init__(
        self,
        *,
        fetch: Fetch,
        host: str = "127.0.0.1",
        port: int = 8766,
        token: str | None = None,
    ) -> None:
        self._fetch = fetch
        self._host = host
        self._port = port
        # Deliberately NOT random per process. Claude Desktop runs two
        # instances of this server, only one of which can hold the port; the
        # other must emit links the holder will accept, or the feature works
        # only by luck of which instance the client happens to ask. Derived
        # from the DSN, so every instance pointed at the same database agrees,
        # and nothing has to be written to disk to coordinate (N9).
        self._token = token or secrets.token_urlsafe(16)
        self._ready = asyncio.Event()
        self._error: str | None = None
        # True when another Transcriptor already holds the port. Its links work
        # for us: it reads the same database and shares our derived token.
        self._peer = False

    # ------------------------------------------------------------------ urls

    @property
    def available(self) -> bool:
        """Serving, or a peer Transcriptor is serving on our behalf."""
        return self._ready.is_set() and (self._error is None or self._peer)

    @property
    def serving_via_peer(self) -> bool:
        return self._peer

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def url_for(self, job_id: str, fmt: str = "txt") -> str | None:
        """A download link, or ``None`` when the viewer is not serving.

        ``None`` rather than a broken URL: a link that 404s is worse than no
        link, because the user tries it before they disbelieve it.
        """
        if not self.available:
            return None
        return f"{self.base_url}/t/{quote(job_id)}.{fmt}?k={self._token}"

    def links_for(self, job_id: str) -> dict[str, str] | None:
        """Every format at once — what a finished job reports."""
        if not self.available:
            return None
        return {fmt: self.url_for(job_id, fmt) or "" for fmt in _MEDIA_TYPES}

    # ----------------------------------------------------------------- serve

    async def run(self) -> None:
        """Serve until cancelled. Safe to supervise: a bind failure disables
        the viewer instead of restarting forever against a port that is not
        going to free itself."""
        try:
            app = self._build_app()
        except ImportError as exc:  # pragma: no cover - starlette ships with fastmcp
            self._error = f"starlette unavailable: {exc}"
            self._ready.set()
            log.warning("transcript viewer disabled: %s", self._error)
            await self._park()
            return

        import uvicorn

        # Bind here rather than letting uvicorn do it. Uvicorn catches the
        # OSError itself and calls sys.exit(1) — SystemExit is a BaseException,
        # so `except OSError` never sees it and the whole MCP server dies with
        # it. Claude Desktop starts two instances of this server per launch, so
        # the second one hitting a taken port is the normal case, not an edge.
        taken = await asyncio.to_thread(self._port_in_use)
        log.info(
            "transcript viewer: %s:%s %s",
            self._host,
            self._port,
            "already in use" if taken else "free",
        )
        if taken:
            await self._resolve_conflict()
            self._ready.set()
            await self._park()
            return

        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
            # stdout is the MCP transport in stdio mode. Uvicorn must never
            # touch it or the client sees protocol garbage.
            log_config=None,
        )
        server = uvicorn.Server(config)
        try:
            self._ready.set()
            await server.serve()
        except asyncio.CancelledError:
            raise
        except (OSError, SystemExit) as exc:
            # SystemExit covers the race between the check above and uvicorn's
            # own bind. Losing a download link must never cost a transcription.
            await self._resolve_conflict(str(exc))
            await self._park()

    async def _park(self) -> None:
        """Wait for cancellation instead of returning.

        ``supervise`` restarts its factory the moment it returns, with no delay
        on a clean return — it assumes a task either loops forever or raises.
        A viewer that gives up and returns therefore spins: bind, fail, log,
        return, restart, at about a thousand times a second. Parking is how a
        task says "I am done, but not crashed".
        """
        await asyncio.Event().wait()

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            return probe.connect_ex((self._host, self._port)) == 0

    async def _resolve_conflict(self, detail: str = "address already in use") -> None:
        """Decide whether the occupant is one of us.

        It matters which: another Transcriptor serves the same database and
        derives the same token, so its URLs work and we should keep advertising
        them. Anything else means our links would 404, and a dead link is worse
        than no link.
        """
        if await asyncio.to_thread(self._peer_responds):
            self._peer = True
            log.info(
                "transcript viewer: another Transcriptor holds %s:%s — "
                "using its links",
                self._host,
                self._port,
            )
            return
        self._error = f"could not bind {self._host}:{self._port}: {detail}"
        log.warning("transcript viewer disabled: %s", self._error)

    def _peer_responds(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/healthz", timeout=2
            ) as response:
                return json.loads(response.read()).get("service") == _MARKER
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _build_app(self) -> Any:
        from starlette.applications import Starlette
        from starlette.responses import (
            HTMLResponse,
            JSONResponse,
            PlainTextResponse,
            Response,
        )
        from starlette.routing import Route

        def _authorised(request: Any) -> bool:
            # secrets.compare_digest, not ==: token comparison should not leak
            # its answer through timing, cheap as that attack is on loopback.
            return secrets.compare_digest(
                request.query_params.get("k", ""), self._token
            )

        async def download(request: Any) -> Response:
            if not _authorised(request):
                return PlainTextResponse("forbidden", status_code=403)
            job_id = request.path_params["job_id"]
            fmt = request.path_params["fmt"]
            if fmt not in _MEDIA_TYPES:
                return PlainTextResponse(f"unknown format: {fmt}", status_code=404)
            try:
                content, filename = await self._fetch(job_id, fmt)
            except Exception as exc:  # noqa: BLE001 - one boundary, typed below
                return PlainTextResponse(_reason(exc), status_code=404)
            return Response(
                content,
                media_type=_MEDIA_TYPES[fmt],
                headers={
                    # attachment, so a click saves the file rather than
                    # dumping subtitle markup into a browser tab.
                    "Content-Disposition": content_disposition(filename),
                    "Cache-Control": "no-store",
                },
            )

        async def view(request: Any) -> Response:
            """A readable page, for when you want to look rather than save."""
            if not _authorised(request):
                return PlainTextResponse("forbidden", status_code=403)
            job_id = request.path_params["job_id"]
            try:
                content, filename = await self._fetch(job_id, "txt")
            except Exception as exc:  # noqa: BLE001
                return PlainTextResponse(_reason(exc), status_code=404)
            return HTMLResponse(_page(job_id, filename, content, self._token))

        async def healthz(_request: Any) -> Response:
            """Unauthenticated on purpose: it reveals only that a Transcriptor
            viewer is here, which a second instance needs to know before it
            decides whether its own links will work."""
            return JSONResponse({"service": _MARKER})

        return Starlette(
            routes=[
                Route("/healthz", healthz),
                Route("/t/{job_id}.{fmt}", download),
                Route("/v/{job_id}", view),
            ]
        )


def derive_token(secret: str) -> str:
    """A link token every instance sharing a database agrees on.

    One-way, so the DSN cannot be recovered from a URL, and stable, so a link
    survives a restart. It is a capability for loopback URLs, not an
    authentication system — the viewer binds to 127.0.0.1 for the actual
    security boundary.
    """
    return hashlib.blake2b(
        secret.encode("utf-8"), person=b"viewer-tok", digest_size=16
    ).hexdigest()


def safe_filename(title: str | None, job_id: str, fmt: str) -> str:
    """A filename a person can find again in their Downloads folder.

    A bare UUID is technically correct and useless the moment the file leaves
    this process, so the source title is used when there is one. Only
    characters that are safe in a filename on every platform survive, and the
    job id is kept as a suffix so two videos with the same title stay distinct.
    """
    stem = ""
    if title:
        kept = [c if (c.isalnum() or c in " -_") else " " for c in title]
        stem = " ".join("".join(kept).split())[:60].strip().replace(" ", "-")
    short = job_id.split("-")[0]
    return f"{stem}-{short}.{fmt}" if stem else f"{job_id}.{fmt}"


def content_disposition(filename: str) -> str:
    """``Content-Disposition`` that survives a non-English title.

    HTTP header values are latin-1. A Hindi, Japanese or Arabic video title in
    a plain ``filename=`` therefore raises ``UnicodeEncodeError`` and the
    download 500s — the same English-shaped string assumption that once split
    "नमस्ते" in the tokeniser. RFC 5987 is the fix: an ASCII fallback for old
    clients, and ``filename*`` carrying the real name percent-encoded as UTF-8.
    """
    ascii_name = filename.encode("ascii", "ignore").decode() or "transcript"
    quoted = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quoted}"


def _reason(exc: Exception) -> str:
    """Adapters raise typed errors; this is the one place a viewer response
    turns one into text. Nothing here parses a message to decide anything."""
    code = getattr(exc, "code", None)
    message = getattr(exc, "user_message", None) or str(exc)
    return f"{code}: {message}" if code else message


def _page(job_id: str, filename: str, content: str, token: str) -> str:
    """The readable page.

    This is the single link worth handing a user — it shows the transcript and
    links every download format — so it carries the affordances they actually
    reach for: read it, copy it, save it.

    Styling is inline and self-contained. No CDN, no webfont: the premise of
    this project is that nothing leaves the machine, and that does not stop
    being true because the thing leaving would only be a stylesheet.

    The copy button uses the async clipboard API, which needs a secure context.
    ``http://127.0.0.1`` qualifies (browsers treat loopback as trustworthy), but
    a permission can still be refused, so the button selects the text and says
    so rather than silently doing nothing — a copy button that lies is worse
    than no copy button.
    """
    body = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    words = len(content.split())
    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{filename}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #fbfbfa;
    --fg: #1a1a18;
    --muted: #6b6b66;
    --line: #e4e4e0;
    --card: #ffffff;
    --accent: #1a1a18;
    --accent-fg: #ffffff;
    --ok: #147a3d;
    --warn: #a63d2f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #121211;
      --fg: #ececea;
      --muted: #96968f;
      --line: #2c2c29;
      --card: #191918;
      --accent: #ececea;
      --accent-fg: #121211;
      --ok: #4ec27d;
      --warn: #e08573;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--fg);
    font: 16px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
          sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  header {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: color-mix(in srgb, var(--bg) 88%, transparent);
    -webkit-backdrop-filter: saturate(180%) blur(12px);
    backdrop-filter: saturate(180%) blur(12px);
    border-bottom: 1px solid var(--line);
  }}
  .inner {{ max-width: 46rem; margin: 0 auto; padding: 1.15rem 1.5rem; }}
  h1 {{
    margin: 0;
    font-size: .95rem;
    font-weight: 600;
    letter-spacing: -.01em;
    word-break: break-word;
  }}
  .meta {{ color: var(--muted); font-size: .8rem; margin-top: .15rem; }}
  .bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: .45rem;
    margin-top: .9rem;
  }}
  .btn {{
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    font: inherit;
    font-size: .82rem;
    font-weight: 500;
    line-height: 1;
    padding: .5rem .85rem;
    border-radius: 8px;
    border: 1px solid transparent;
    cursor: pointer;
    text-decoration: none;
    white-space: nowrap;
    transition: background .14s ease, border-color .14s ease,
                color .14s ease, transform .06s ease;
  }}
  .btn:active {{ transform: translateY(1px); }}
  .btn:focus-visible {{ outline: 2px solid var(--fg); outline-offset: 2px; }}
  .primary {{ background: var(--accent); color: var(--accent-fg); }}
  .primary:hover {{ opacity: .88; }}
  .primary[data-state="ok"] {{ background: var(--ok); color: #fff; }}
  .primary[data-state="fail"] {{ background: var(--warn); color: #fff; }}
  .ghost {{
    background: var(--card);
    border-color: var(--line);
    color: var(--muted);
  }}
  .ghost:hover {{ color: var(--fg); border-color: var(--muted); }}
  .label {{
    color: var(--muted);
    font-size: .74rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-left: .4rem;
  }}
  main {{ max-width: 46rem; margin: 0 auto; padding: 2rem 1.5rem 6rem; }}
  pre {{
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    font: inherit;
  }}
</style>
<header>
  <div class="inner">
    <h1>{filename}</h1>
    <div class="meta">{words:,} words</div>
    <div class="bar">
      <button id="copy" class="btn primary" type="button">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.2" stroke-linecap="round"
             stroke-linejoin="round" aria-hidden="true">
          <rect x="9" y="9" width="12" height="12" rx="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span id="copy-label">Copy all</span>
      </button>
      <span class="label">download</span>
      <a class="btn ghost" href="/t/{job_id}.txt?k={token}">txt</a>
      <a class="btn ghost" href="/t/{job_id}.srt?k={token}">srt</a>
      <a class="btn ghost" href="/t/{job_id}.vtt?k={token}">vtt</a>
      <a class="btn ghost" href="/t/{job_id}.json?k={token}">json</a>
    </div>
  </div>
</header>
<main><pre id="transcript">{body}</pre></main>
<script>
  const button = document.getElementById("copy");
  const label = document.getElementById("copy-label");
  const source = document.getElementById("transcript");

  function flash(text, state) {{
    label.textContent = text;
    button.dataset.state = state;
    setTimeout(() => {{
      label.textContent = "Copy all";
      delete button.dataset.state;
    }}, 1900);
  }}

  button.addEventListener("click", async () => {{
    try {{
      await navigator.clipboard.writeText(source.textContent);
      flash("Copied", "ok");
    }} catch (err) {{
      // Selecting the text is the honest fallback: the user can still press
      // cmd-C, and they are told the automatic path was refused rather than
      // being left wondering whether the click registered.
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      flash("Selected \u2014 press \u2318C", "fail");
    }}
  }});
</script>
"""
