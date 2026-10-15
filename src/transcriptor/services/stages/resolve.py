"""Resolve — turn the source row into known, validated media.

Satisfied when the job carries a duration, because that is this stage's
checkpoint.  Phase 1 sources are local files, already resolved at create time;
the work left here is probing and validating them.
"""

from __future__ import annotations

from pathlib import Path

from transcriptor.domain.errors import (
    DurationExceedsLimit,
    ExtractorFailed,
    SourceNotFound,
)
from transcriptor.domain.job import Stage as JobStage
from transcriptor.domain.source import SourceKind, SourceRef
from transcriptor.services.pipeline import JobContext


class ResolveStage:
    name = "resolve"
    job_stage = JobStage.RESOLVING

    def __init__(self, *, gateway=None) -> None:
        self._gateway = gateway

    async def satisfied(self, ctx: JobContext) -> bool:
        if ctx.job.duration_s is None:
            return False
        # Later stages need the source row and the ref whether or not this
        # stage ran, so they are loaded on the probe too.
        await self._load_ref(ctx)
        return True

    async def _load_ref(self, ctx: JobContext) -> None:
        ctx.source = ctx.source or await ctx.jobs.get_source(ctx.job.source_id) or {}
        if ctx.ref is None and ctx.source:
            ctx.ref = SourceRef(
                kind=SourceKind(ctx.source["kind"]),
                raw=ctx.source["raw"],
                key=ctx.source["source_key"],
                platform=ctx.source.get("platform") or "local",
            )

    async def run(self, ctx: JobContext) -> None:
        await self._load_ref(ctx)
        meta = await self._probe(ctx)

        # Guards against an accidental 12-hour stream VOD. Chunking removed the
        # phase 1 reason for a low cap, so this is now the documented 4-hour
        # limit rather than ten minutes.
        if meta.duration_s > ctx.max_duration_s:
            raise DurationExceedsLimit(
                f"source is {meta.duration_s:.0f}s, the limit is "
                f"{ctx.max_duration_s}s",
                duration_s=meta.duration_s,
                limit_s=ctx.max_duration_s,
            )

        ctx.meta = meta
        await ctx.jobs.set_source_meta(
            ctx.job.source_id, duration_s=meta.duration_s, title=meta.title
        )
        # duration_s is this stage's checkpoint. Written directly rather than
        # through a status transition, because the status is not changing.
        await ctx.jobs.set_job_fields(ctx.job_id, duration_s=meta.duration_s)
        await ctx.refresh()

    async def _probe(self, ctx: JobContext):
        """Cheap metadata, before committing to a download or an hour of GPU."""
        if ctx.ref is not None and ctx.ref.kind is SourceKind.URL:
            if self._gateway is None:
                raise ExtractorFailed(
                    "URL sources need yt-dlp — run: uv sync --extra media",
                    source=ctx.ref.key,
                )
            return await self._gateway.probe(ctx.ref)

        path = Path(ctx.source["source_key"])
        if not path.exists():
            raise SourceNotFound(f"source file no longer exists: {path}", path=str(path))
        return await ctx.extractor.probe(path)
