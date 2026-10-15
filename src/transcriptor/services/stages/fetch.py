"""Fetch — get the media onto local disk, or take the captions shortcut.

For a local file there is nothing to fetch: the source *is* the media.  For a
URL, yt-dlp pulls ``bestaudio`` — a 40-minute 4K video is roughly 30MB of audio
instead of 2GB of video.

The captions shortcut is the interesting part.  Many videos already carry
captions, and fetching them takes seconds instead of ten minutes of inference.
But the two kinds are not equivalent: human-uploaded captions are usually
*better* than Whisper, auto-generated ones are *worse*.  The policy decides, and
``origin`` always reports which was used.

When captions are taken they are committed here as chunk 0.  Extract, Detect and
Transcribe then all report satisfied, and the pipeline's ordinary skip loop does
the rest — no branching.
"""

from __future__ import annotations

import logging

from transcriptor.domain.job import Origin
from transcriptor.domain.job import Stage as JobStage
from transcriptor.domain.source import CaptionPolicy, SourceKind
from transcriptor.domain.transcript import Chunk, ChunkStatus, Segment
from transcriptor.services.pipeline import JobContext

log = logging.getLogger(__name__)

DOWNLOAD_STEM = "source"


class FetchStage:
    name = "fetch"
    job_stage = JobStage.DOWNLOADING

    def __init__(self, *, gateway=None, captions: CaptionPolicy = CaptionPolicy.MANUAL_ONLY):
        self._gateway = gateway
        self._captions = captions

    async def satisfied(self, ctx: JobContext) -> bool:
        if ctx.from_captions:
            return True
        if ctx.ref is None or ctx.ref.kind is SourceKind.FILE:
            ctx.media_path = ctx.source_path
            return ctx.media_path is not None and ctx.media_path.exists()

        existing = sorted(ctx.job_scratch.glob(DOWNLOAD_STEM + ".*"))
        if not existing:
            return False
        ctx.media_path = existing[0]
        return True

    async def run(self, ctx: JobContext) -> None:
        if ctx.ref is None or ctx.ref.kind is SourceKind.FILE:
            ctx.media_path = ctx.source_path
            return

        if self._gateway is None:
            from transcriptor.domain.errors import ExtractorFailed

            raise ExtractorFailed(
                "URL sources need yt-dlp — run: uv sync --extra media",
                source=ctx.ref.key,
            )

        if await self._try_captions(ctx):
            return

        ctx.media_path = await self._gateway.fetch_audio(
            ctx.ref, ctx.job_scratch / DOWNLOAD_STEM, cancel=ctx.cancel
        )
        log.info(
            "audio downloaded",
            extra={
                "job_id": str(ctx.job_id),
                "bytes": ctx.media_path.stat().st_size,
                "path": str(ctx.media_path),
            },
        )

    async def _try_captions(self, ctx: JobContext) -> bool:
        """Returns whether captions answered the request outright."""
        policy = ctx.captions or self._captions
        if policy is CaptionPolicy.NEVER:
            # Do not make a request whose answer is already decided.
            return False

        result = await self._gateway.fetch_captions(ctx.ref, policy)
        if result is None:
            return False

        segments, origin = result
        renumbered = [
            Segment(idx=n, start=s.start, end=s.end, text=s.text)
            for n, s in enumerate(segments)
        ]
        span = renumbered[-1].end if renumbered else (ctx.job.duration_s or 0.0)

        await ctx.transcripts.commit_chunk(
            ctx.job_id,
            Chunk(idx=0, start_s=0.0, end_s=span, status=ChunkStatus.DONE, attempts=1),
            renumbered,
        )
        await ctx.jobs.set_job_fields(ctx.job_id, origin=Origin(origin).value)
        await ctx.refresh()

        log.info(
            "captions used instead of inference",
            extra={
                "job_id": str(ctx.job_id),
                "origin": origin,
                "segments": len(renumbered),
            },
        )
        return True
