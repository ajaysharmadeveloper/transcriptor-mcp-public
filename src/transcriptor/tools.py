"""MCP tool definitions.

Phase 1 exposes four: ``transcript_create``, ``transcript_status``,
``transcript_get``, ``transcript_health``.  The other seven land in phase 2.

Handlers are thin by rule: schema validation, params → command, result →
response.  No business logic, no engine, no filesystem, no subprocess — the
database is the only thing they reach, and only through services.  An
import-linter contract enforces it.

The ``transcript_`` prefix stays even without a shared namespace to protect: a
client with several servers connected can end up with bare ``search``, ``list``
and ``status`` colliding, and prefixed names stay unambiguous in tool-selection
prompts.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass
from typing import Any

from transcriptor.domain.errors import TranscriptorError
from transcriptor.domain.source import CaptionPolicy
from transcriptor.services.reads import ReadService
from transcriptor.services.transcription import TranscriptionService

log = logging.getLogger(__name__)


@dataclass
class Deps:
    """What the handlers are allowed to touch.

    Constructed in ``server.py`` and passed in, so ``tools.py`` never reaches
    for a global and never constructs an adapter itself.
    """

    transcription: TranscriptionService
    reads: ReadService
    health: Any  # HealthService — duck-typed to keep tools free of adapters
    lifecycle: Any | None = None  # LifecycleService
    clips: Any | None = None  # ClipService
    requests: Any | None = None  # RequestLog
    viewer: Any | None = None  # TranscriptViewer — duck-typed, may be absent


def _links(deps: "Deps", job_id: str, *, status: str | None = None) -> dict[str, Any]:
    """Download links for a job, when there is anything to download.

    Returned as part of a normal response rather than behind a separate tool:
    the moment the user wants the transcript is the moment they are told it is
    ready, and making them ask again for the link wastes a turn. Empty when the
    viewer is off or the job has produced nothing yet — an absent key is
    honest, a dead link is not.
    """
    viewer = deps.viewer
    if viewer is None or not getattr(viewer, "available", False):
        return {}
    if status is not None and status in {"queued", "failed", "cancelled"}:
        return {}
    links = viewer.links_for(job_id)
    if not links:
        return {}
    return {
        "download_urls": links,
        "view_url": f"{viewer.base_url}/v/{job_id}?k={links['txt'].rsplit('k=', 1)[1]}",
    }


def _error(exc: TranscriptorError) -> dict[str, Any]:
    """One mapping point.  Clients read ``error_code`` and ``resumable``; they
    never parse ``error`` to decide whether to retry."""
    return {
        "ok": False,
        "error": exc.user_message,
        "error_code": exc.code,
        "resumable": exc.resumable,
        **({"context": exc.context} if exc.context else {}),
    }


def register(mcp, deps: Deps) -> None:
    """Attach the tools to a FastMCP server."""

    async def _record(tool: str, params: dict, outcome: str, started: float) -> None:
        if deps.requests is None:
            return
        try:
            await deps.requests.record(
                tool=tool,
                params=params,
                outcome=outcome,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception:  # noqa: BLE001
            # The audit trail must never be why a tool call fails.
            log.exception("could not record request", extra={"tool": tool})

    @mcp.tool
    async def transcript_create(
        source: str,
        language: str | None = None,
        captions: str = "manual_only",
        force: bool = False,
    ) -> dict[str, Any]:
        """Start a transcription job and return its id immediately.

        `source` is either an absolute local path (`~` is expanded) or a URL —
        YouTube, Instagram, TikTok, X, Vimeo, Reddit and anything else yt-dlp
        supports, plus direct media URLs. Relative paths are refused, and a
        playlist or channel URL is refused as ambiguous: pass one video.

        The call never waits for work. Poll `transcript_status`, and read
        partial output with `transcript_get` while it runs.

        `captions` decides whether to use a platform's existing subtitles
        instead of transcribing: `manual_only` (default) takes human-written
        captions when present, which are usually better than Whisper;
        `prefer` also accepts auto-generated ones, which are usually worse;
        `never` always transcribes. The result reports `origin` so you know
        which you got.

        `language` is an ISO code to force; omit it for automatic detection.
        `force` bypasses the cache and re-transcribes.

        Sources longer than four hours are refused. Login-walled, private,
        members-only or age-restricted URLs fail with `auth_required` — set
        TRANSCRIPTOR_COOKIES_FROM_BROWSER (chrome, safari, firefox, brave) or
        TRANSCRIPTOR_COOKIES_FILE in the server's .env and retry.
        """
        started = time.monotonic()
        params = {
            "source": source,
            "language": language,
            "captions": captions,
            "force": force,
        }
        try:
            result = await deps.transcription.create(
                source,
                language=language,
                captions=CaptionPolicy(captions),
                force=force,
            )
        except TranscriptorError as exc:
            await _record("transcript_create", params, exc.code, started)
            return _error(exc)
        except ValueError:
            return _error(
                TranscriptorError(
                    f"captions must be one of manual_only, never, prefer — got "
                    f"{captions!r}",
                    code="invalid_parameter",
                    resumable=False,
                )
            )

        await _record("transcript_create", params, "ok", started)
        payload = dataclasses.asdict(result)
        # A reused transcript is finished work, so the link is useful now.
        # A fresh job has nothing to serve yet; transcript_status will carry it.
        if payload.get("reuse_reason") in {"exact_source", "same_audio"}:
            payload.update(_links(deps, str(payload.get("job_id", ""))))
        return {"ok": True, **payload}

    @mcp.tool
    async def transcript_probe(
        source: str,
        language: str | None = None,
        captions: str = "manual_only",
    ) -> dict[str, Any]:
        """Inspect a source without transcribing it, and without downloading it.

        Says how long it is, whether captions already exist (which would make it
        near-instant), whether it needs cookies, and whether you have already
        transcribed it under these exact settings. Pass the same `language` and
        `captions` you would pass to transcript_create, so `existing` reports
        what *would* happen rather than what might.
        """
        started = time.monotonic()
        params = {"source": source, "language": language, "captions": captions}
        try:
            result = await deps.transcription.probe(
                source, language=language, captions=CaptionPolicy(captions)
            )
        except TranscriptorError as exc:
            await _record("transcript_probe", params, exc.code, started)
            return _error(exc)
        except ValueError:
            return _error(
                TranscriptorError(
                    f"captions must be manual_only, never or prefer — got {captions!r}",
                    code="invalid_parameter",
                    resumable=False,
                )
            )
        await _record("transcript_probe", params, "ok", started)
        return {"ok": True, **dataclasses.asdict(result)}

    @mcp.tool
    async def transcript_status(job_id: str) -> dict[str, Any]:
        """Where a job stands: status, stage, progress, elapsed and ETA.

        ``status`` is one of queued, running, paused, done, failed, cancelled,
        interrupted. ``resumable`` says whether retrying helps — do not parse
        the error message to decide.

        Once the job has produced anything, the response also carries
        ``download_urls`` (txt, srt, vtt, json) and ``view_url`` — loopback
        links served by this process. Give them to the user when a job
        finishes: clicking one downloads the transcript, which is a better
        answer than pushing thousands of words through the chat.

        **Present a link as a markdown link inside a sentence, not inside a
        code block** — fenced blocks are not clickable, so the user has to
        copy and paste. Prefer offering ``view_url`` alone: that page already
        links to all four formats, so one link beats five.
        """
        started = time.monotonic()
        try:
            view = await deps.reads.status(job_id)
        except TranscriptorError as exc:
            await _record("transcript_status", {"job_id": job_id}, exc.code, started)
            return _error(exc)
        await _record("transcript_status", {"job_id": job_id}, "ok", started)
        return {
            "ok": True,
            **dataclasses.asdict(view),
            **_links(deps, job_id, status=view.status),
        }

    @mcp.tool
    async def transcript_get(
        job_id: str,
        format: str = "txt",
        mode: str = "preview",
        max_chars: int = 2000,
    ) -> dict[str, Any]:
        """Transcript text, rendered from segments on demand.

        `format` is txt, srt, vtt or json. `mode="preview"` is the default
        deliberately: a long transcript returned whole floods the calling
        agent's context. Use `mode="full"` when the user has actually asked for
        the whole transcript — the default exists to stop unprompted flooding,
        not to refuse a direct request.

        The response also carries ``download_urls`` and ``view_url``, which are
        usually the better answer for a long transcript: the user gets a file
        they can keep, and your context stays free for the work. Offer them as
        markdown links in a sentence — a URL in a code block cannot be clicked.

        Works on a running job — you get what is committed so far, with
        ``partial: true``.
        """
        started = time.monotonic()
        params = {
            "job_id": job_id,
            "format": format,
            "mode": mode,
            "max_chars": max_chars,
        }
        try:
            view = await deps.reads.get(
                job_id, fmt=format, mode=mode, max_chars=max_chars
            )
        except TranscriptorError as exc:
            await _record("transcript_get", params, exc.code, started)
            return _error(exc)
        await _record("transcript_get", params, "ok", started)
        return {
            "ok": True,
            **dataclasses.asdict(view),
            **_links(deps, job_id, status=view.status),
        }

    @mcp.tool
    async def transcript_search(
        job_id: str,
        query: str,
        context_seconds: float = 5.0,
        max_results: int = 10,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Find a phrase in a transcript, with timestamps.

        The intended way into a long transcript, together with
        transcript_segments — pulling the whole thing would flood your context.
        Works on a running job: it searches what has been committed so far.
        Falls back to fuzzy matching when nothing matches whole words, which is
        what helps when you half-remember a phrase.
        """
        started = time.monotonic()
        params = {"job_id": job_id, "query": query, "max_results": max_results}
        try:
            hits = await deps.reads.search(
                job_id, query, context_seconds=context_seconds,
                max_results=max_results,
            )
        except TranscriptorError as exc:
            await _record("transcript_search", params, exc.code, started)
            return _error(exc)
        await _record("transcript_search", params, "ok", started)
        return hits

    @mcp.tool
    async def transcript_segments(
        job_id: str,
        start_s: float,
        end_s: float,
        include_words: bool = False,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Pull a time window of the transcript.

        Segments overlapping the window are returned, not only those contained
        by it, so a segment straddling the boundary still appears.
        `include_words` roughly triples the response size, so it is off by
        default.
        """
        started = time.monotonic()
        params = {"job_id": job_id, "start_s": start_s, "end_s": end_s}
        try:
            found = await deps.reads.segments(
                job_id, start_s, end_s, include_words=include_words
            )
        except TranscriptorError as exc:
            await _record("transcript_segments", params, exc.code, started)
            return _error(exc)
        await _record("transcript_segments", params, "ok", started)
        return found

    @mcp.tool
    async def transcript_list(
        limit: int = 20,
        query: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """Recent jobs, newest first.

        Filter by `status` (queued, running, paused, done, failed, cancelled,
        interrupted) for a board-level view of what is in flight, what is stuck
        and what is finished.
        """
        started = time.monotonic()
        params = {"limit": limit, "query": query, "status": status}
        try:
            jobs = await deps.reads.list_jobs(limit=limit, query=query, status=status)
        except TranscriptorError as exc:
            await _record("transcript_list", params, exc.code, started)
            return _error(exc)
        await _record("transcript_list", params, "ok", started)
        return [dataclasses.asdict(j) for j in jobs]

    @mcp.tool
    async def transcript_export(
        job_id: str,
        dest_path: str,
        format: str = "srt",
    ) -> dict[str, Any]:
        """Write a transcript to a file at a path you choose.

        The only thing Transcriptor writes to disk, and only on request.
        `dest_path` must be absolute. Nothing partial is left behind if the
        write fails.

        The response also carries ``download_urls`` and ``view_url``. If the
        user asked for a link rather than a path, give them those — they are
        served by this process and need no file at all. Write them as markdown
        links in a sentence; a URL fenced in a code block has to be copied by
        hand.
        """
        started = time.monotonic()
        params = {"job_id": job_id, "dest_path": dest_path, "format": format}
        try:
            result = await deps.lifecycle.export(
                job_id, fmt=format, dest_path=dest_path
            )
        except TranscriptorError as exc:
            await _record("transcript_export", params, exc.code, started)
            return _error(exc)
        await _record("transcript_export", params, "ok", started)
        return {
            "ok": True,
            **dataclasses.asdict(result),
            **_links(deps, job_id),
        }

    @mcp.tool
    async def transcript_cancel(job_id: str) -> dict[str, Any]:
        """Stop a job.

        A queued job stops at once; a running one stops at the next chunk
        boundary, so allow up to about two minutes. Everything already
        transcribed survives and the job can be resumed later.
        """
        started = time.monotonic()
        try:
            result = await deps.lifecycle.cancel(job_id)
        except TranscriptorError as exc:
            await _record("transcript_cancel", {"job_id": job_id}, exc.code, started)
            return _error(exc)
        await _record("transcript_cancel", {"job_id": job_id}, "ok", started)
        return {"ok": True, **result}

    @mcp.tool
    async def transcript_delete(
        job_id: str, delete_media: bool = False
    ) -> dict[str, Any]:
        """Delete a transcript. Destructive and not reversible.

        Removes the job and its segments, chunks, events and cached renders.
        `delete_media=True` also removes any scratch audio still on disk.
        Refuses while the job is running — cancel it first.
        """
        started = time.monotonic()
        params = {"job_id": job_id, "delete_media": delete_media}
        try:
            result = await deps.lifecycle.delete(job_id, delete_media=delete_media)
        except TranscriptorError as exc:
            await _record("transcript_delete", params, exc.code, started)
            return _error(exc)
        await _record("transcript_delete", params, "ok", started)
        return {"ok": True, **dataclasses.asdict(result)}

    @mcp.tool
    async def transcript_batch(
        folder: str,
        pattern: str = "*",
        language: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Queue every media file in a folder, one job each.

        `folder` must be absolute. Not recursive — recursing would silently pull
        in a backup directory nobody meant to transcribe. Files already
        transcribed under these settings come back as reused rather than being
        queued again. Poll the returned job_ids with transcript_status, or watch
        the whole set with transcript_list.
        """
        started = time.monotonic()
        params = {"folder": folder, "pattern": pattern, "limit": limit}
        try:
            result = await deps.lifecycle.batch(
                folder, pattern=pattern, language=language, limit=limit
            )
        except TranscriptorError as exc:
            await _record("transcript_batch", params, exc.code, started)
            return _error(exc)
        await _record("transcript_batch", params, "ok", started)
        return {"ok": True, **result}

    @mcp.tool
    async def transcript_find_clips(
        job_id: str,
        count: int = 5,
        min_s: float = 15.0,
        max_s: float = 60.0,
        dest_dir: str | None = None,
    ) -> dict[str, Any]:
        """Find self-contained, high-density moments worth cutting into clips.

        Scored on words per second, but only runs that begin at the start of a
        sentence and end at the end of one are considered — a moment that opens
        mid-clause is worthless however dense it is. Selected clips never
        overlap.

        Returns, per clip: the ffmpeg command to cut it, the command to crop it
        to 9:16, and an SRT rebased to the clip's own clock for burning in.
        **The commands are returned, not run** — nothing is written to disk.
        """
        started = time.monotonic()
        params = {"job_id": job_id, "count": count, "min_s": min_s, "max_s": max_s}
        try:
            result = await deps.clips.find(
                job_id, count=count, min_s=min_s, max_s=max_s, dest_dir=dest_dir
            )
        except TranscriptorError as exc:
            await _record("transcript_find_clips", params, exc.code, started)
            return _error(exc)
        await _record("transcript_find_clips", params, "ok", started)
        return {"ok": True, **result, **_links(deps, job_id)}

    @mcp.tool
    async def transcript_health() -> dict[str, Any]:
        """Engine state, real memory pressure, queue depth, degraded tasks.

        ``engine`` and ``device`` say what is actually running here — `mlx` on
        Metal, or `faster-whisper` on `cuda` or `cpu`. Worth reading before
        promising a user how long something will take: the same model differs
        by an order of magnitude between them. On a machine with a discrete
        GPU the report also carries ``vram_*``; on unified memory those keys
        are absent, because there the GPU spends the memory
        ``system_available_gb`` already accounts for.

        Also reports whether the transcript download viewer is serving. If
        ``viewer.available`` is false, ``viewer.error`` says why — usually the
        port is already taken by another Transcriptor.

        Memory numbers come from the monitor task's most recent sample rather
        than being measured inline, so this stays a database-and-memory read.
        """
        started = time.monotonic()
        try:
            report = await deps.health.report()
        except TranscriptorError as exc:
            await _record("transcript_health", {}, exc.code, started)
            return _error(exc)
        await _record("transcript_health", {}, "ok", started)
        viewer = deps.viewer
        if viewer is not None:
            # A viewer that failed to bind is not a crashed task, so the
            # supervisor never hears about it. Say so here instead of leaving
            # the user to wonder why no link ever appears.
            report["viewer"] = {
                "available": bool(getattr(viewer, "available", False)),
                "base_url": getattr(viewer, "base_url", None),
                "error": getattr(viewer, "error", None),
            }
        return {"ok": True, **report}
