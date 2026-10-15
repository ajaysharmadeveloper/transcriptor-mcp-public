"""The transcript download viewer.

Tested against the real Starlette app through a real client, not by calling the
handlers directly: what matters is the response a browser actually receives —
the status code, the Content-Disposition header that makes a click save a file
rather than render markup, and the 403 that keeps another local process out.
"""

from __future__ import annotations

import pytest

from transcriptor.adapters.http_viewer import (
    TranscriptViewer,
    content_disposition,
    safe_filename,
)
from transcriptor.domain.errors import TranscriptorError


async def _fetch(job_id: str, fmt: str) -> tuple[str, str]:
    if job_id == "missing":
        raise TranscriptorError("no such job: missing", code="job_not_found")
    if job_id == "hindi":
        return "नमस्ते", f"नमस्ते-दुनिया.{fmt}"
    return f"content of {job_id} as {fmt}", f"{job_id}.{fmt}"


@pytest.fixture
def client():
    starlette_testclient = pytest.importorskip("starlette.testclient")
    viewer = TranscriptViewer(fetch=_fetch, token="test-token")
    # The app is what the routes are on; `available` normally flips in run(),
    # which we are deliberately not starting — no port is bound in these tests.
    app = viewer._build_app()
    return viewer, starlette_testclient.TestClient(app)


def test_download_sets_attachment_header(client):
    """A click must save a file. Without this header the browser renders SRT
    markup in a tab, which is the wrong answer to 'give me the transcript'."""
    _, c = client
    r = c.get("/t/abc.srt?k=test-token")
    assert r.status_code == 200
    assert r.text == "content of abc as srt"
    assert r.headers["content-disposition"].startswith("attachment;")
    assert 'filename="abc.srt"' in r.headers["content-disposition"]
    assert r.headers["content-type"].startswith("application/x-subrip")


def test_non_ascii_title_does_not_break_the_download():
    """HTTP headers are latin-1, so a Hindi or Japanese video title in a plain
    `filename=` raises UnicodeEncodeError and the download 500s. Caught by
    trying it, not by reading the code — the same English-shaped string
    assumption that once split नमस्ते in the tokeniser."""
    for name in ("नमस्ते-दुनिया.srt", "日本語.txt", "مرحبا.vtt"):
        header = content_disposition(name)
        header.encode("latin-1")  # raises if we regressed
        assert "filename*=UTF-8''" in header


def test_non_ascii_download_is_served(client):
    _, c = client
    r = c.get("/t/hindi.srt?k=test-token")
    assert r.status_code == 200
    assert r.text == "नमस्ते"


@pytest.mark.parametrize(
    "title,expected_stem",
    [
        ("Browser Side v/s Server Side", "Browser-Side-v-s-Server-Side"),
        ("  spaced   out  ", "spaced-out"),
        (None, None),
    ],
)
def test_filenames_are_findable_in_a_downloads_folder(title, expected_stem):
    """A bare UUID is correct and useless the moment the file leaves the
    process. The job id stays as a suffix so two videos with the same title
    do not collide."""
    name = safe_filename(title, "34052bb0-663f-4083-85e6-d9b16948d0dc", "srt")
    if expected_stem is None:
        assert name == "34052bb0-663f-4083-85e6-d9b16948d0dc.srt"
    else:
        assert name == f"{expected_stem}-34052bb0.srt"


@pytest.mark.parametrize("fmt", ["txt", "srt", "vtt", "json"])
def test_every_advertised_format_is_served(client, fmt):
    """links_for() advertises four formats; each has to actually work, or the
    link we hand the user 404s after they click it."""
    _, c = client
    assert c.get(f"/t/abc.{fmt}?k=test-token").status_code == 200


def test_unknown_format_is_404_not_500(client):
    _, c = client
    r = c.get("/t/abc.docx?k=test-token")
    assert r.status_code == 404


def test_wrong_token_is_forbidden(client):
    """Job ids are UUIDs and hard to guess, but the token is what stops another
    process on this machine enumerating transcripts once it learns one."""
    _, c = client
    assert c.get("/t/abc.txt?k=wrong").status_code == 403
    assert c.get("/t/abc.txt").status_code == 403
    assert c.get("/v/abc?k=wrong").status_code == 403


def test_missing_job_reports_the_error_code(client):
    """Typed errors reach the browser as text; nothing parses a message to
    decide anything."""
    _, c = client
    r = c.get("/t/missing.txt?k=test-token")
    assert r.status_code == 404
    assert "job_not_found" in r.text


def test_view_page_escapes_transcript_text():
    """Transcript text is model output over arbitrary audio. It is never
    trusted as markup."""
    from transcriptor.adapters.http_viewer import _page

    html = _page("abc", "abc.txt", "<script>alert(1)</script>", "tok")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_view_page_offers_copy_and_every_download_format():
    """This is the single link worth handing a user, so it has to carry what
    they actually reach for: read it, copy it, or save it."""
    from transcriptor.adapters.http_viewer import _page

    html = _page("abc", "abc.txt", "one two three", "tok")
    assert 'id="copy"' in html
    assert "navigator.clipboard.writeText" in html
    assert "3 words" in html
    for fmt in ("txt", "srt", "vtt", "json"):
        assert f'href="/t/abc.{fmt}?k=tok"' in html


def test_copy_button_reports_failure_rather_than_lying():
    """A copy button that silently does nothing is worse than no button. If
    the clipboard permission is refused the text is selected instead and the
    user is told to press cmd-C."""
    from transcriptor.adapters.http_viewer import _page

    html = _page("abc", "abc.txt", "text", "tok")
    assert "catch" in html
    assert "selectNodeContents" in html


def test_urls_are_none_until_the_viewer_is_serving():
    """A dead link is worse than no link: the user tries it before they
    disbelieve it."""
    viewer = TranscriptViewer(fetch=_fetch)
    assert viewer.available is False
    assert viewer.url_for("abc") is None
    assert viewer.links_for("abc") is None


def test_urls_carry_the_token_and_the_format():
    viewer = TranscriptViewer(fetch=_fetch, port=9999, token="tok")
    viewer._ready.set()  # what run() does once it is about to serve
    assert viewer.url_for("abc", "srt") == "http://127.0.0.1:9999/t/abc.srt?k=tok"
    assert set(viewer.links_for("abc")) == {"txt", "srt", "vtt", "json"}


async def test_taken_port_does_not_kill_the_process():
    """The bug this replaced a fake test for.

    uvicorn catches the bind OSError itself and calls sys.exit(1). SystemExit
    is a BaseException, so `except OSError` never saw it and the whole MCP
    server died — Claude Desktop showed "Server disconnected" on every launch,
    because Desktop starts two instances and only one can hold the port.

    The previous version of this test set `_error` by hand and asserted the
    consequence, so it passed while the real path crashed. A fake that cannot
    express the failure tests nothing.
    """
    import asyncio
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        port = squatter.getsockname()[1]

        viewer = TranscriptViewer(fetch=_fetch, port=port, token="tok")
        task = asyncio.create_task(viewer.run())
        await asyncio.wait_for(viewer._ready.wait(), timeout=5)
        await asyncio.sleep(0)

        # It must park, not return: `supervise` restarts its factory the moment
        # it returns, with no delay on a clean return, so a returning viewer
        # spins at about a thousand bind attempts a second.
        assert not task.done(), "viewer returned instead of parking"
        task.cancel()

    # A plain socket is not a Transcriptor, so no links are advertised.
    assert viewer.available is False
    assert viewer.serving_via_peer is False
    assert viewer.url_for("abc") is None
    assert "already in use" in (viewer.error or "").lower()


async def test_a_peer_transcriptor_keeps_links_working():
    """Two instances, one port. The one that lost still advertises links,
    because the winner reads the same database and derives the same token."""
    import asyncio

    winner = TranscriptViewer(fetch=_fetch, port=0, token="shared")
    # Serve the winner on a real ephemeral port.
    import uvicorn

    config = uvicorn.Config(
        winner._build_app(), host="127.0.0.1", port=0,
        log_level="critical", access_log=False, log_config=None,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(100):
        if server.started:
            break
        await asyncio.sleep(0.02)
    assert server.started, "test server never came up"
    port = server.servers[0].sockets[0].getsockname()[1]

    loser = TranscriptViewer(fetch=_fetch, port=port, token="shared")
    loser_task = asyncio.create_task(loser.run())
    await asyncio.wait_for(loser._ready.wait(), timeout=5)
    await asyncio.sleep(0)
    assert not loser_task.done(), "viewer returned instead of parking"
    loser_task.cancel()

    assert loser.serving_via_peer is True
    assert loser.available is True
    assert loser.url_for("abc", "srt") == (
        f"http://127.0.0.1:{port}/t/abc.srt?k=shared"
    )

    server.should_exit = True
    await task


def test_healthz_identifies_the_service(client):
    """How a second instance tells a peer Transcriptor from something else on
    the port. Unauthenticated on purpose — it reveals nothing else."""
    _, c = client
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["service"] == "transcriptor-viewer"


def test_token_is_shared_by_instances_on_the_same_database():
    """Random-per-process would mean the instance that lost the port emits
    links the winner rejects with 403 — the feature working by luck."""
    from transcriptor.adapters.http_viewer import derive_token

    dsn = "postgresql://u:p@localhost:5432/transcriptor_db"
    assert derive_token(dsn) == derive_token(dsn)
    assert derive_token(dsn) != derive_token(dsn + "_other")
    assert dsn not in derive_token(dsn)


# --------------------------------------------------- the link surface itself


class _StubViewer:
    """Stands in for a serving viewer, so the link surface can be checked
    without binding a port."""

    available = True
    base_url = "http://127.0.0.1:8766"

    def links_for(self, job_id: str) -> dict[str, str]:
        return {
            fmt: f"{self.base_url}/t/{job_id}.{fmt}?k=tok"
            for fmt in ("txt", "srt", "vtt", "json")
        }


def test_every_tool_that_hands_over_a_transcript_carries_links():
    """The gap this closes: `transcript_export` returned no links, so a client
    asked for a URL exported a file, saw no link in the response, and told the
    user that no URL existed — a confident claim its own tool output would have
    contradicted, had the tool carried one.

    Any handler a user might reasonably reach by asking "give me the
    transcript" has to be inside the link surface. This asserts the list rather
    than the mechanism, because the mechanism was never the problem.
    """
    import inspect

    from transcriptor import tools

    source = inspect.getsource(tools)
    must_link = [
        "transcript_status",
        "transcript_get",
        "transcript_export",
        "transcript_find_clips",
    ]
    for name in must_link:
        start = source.index(f"async def {name}(")
        end = source.find("\n    @mcp.tool", start)
        body = source[start : end if end != -1 else len(source)]
        assert "_links(" in body, (
            f"{name} hands the user a transcript but returns no download link"
        )


def test_links_helper_omits_keys_when_nothing_is_servable():
    from transcriptor import tools

    deps = tools.Deps(transcription=None, reads=None, health=None)
    assert tools._links(deps, "abc") == {}

    deps.viewer = _StubViewer()
    assert tools._links(deps, "abc", status="queued") == {}
    assert tools._links(deps, "abc", status="failed") == {}

    links = tools._links(deps, "abc", status="done")
    assert set(links["download_urls"]) == {"txt", "srt", "vtt", "json"}
    assert links["view_url"] == "http://127.0.0.1:8766/v/abc?k=tok"
