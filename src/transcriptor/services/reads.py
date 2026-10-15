"""Read-only paths: status, get, health.

These bypass the runner entirely and hit the database directly, which is why
``transcript_status`` stays under 300ms during a long job and why a running job
is queryable rather than opaque.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from transcriptor.domain.errors import JobNotFound, TranscriptorError
from transcriptor.domain.job import JobStatus
from transcriptor.domain.transcript import Format
from transcriptor.ports.stores import JobStore, TranscriptStore


def parse_job_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError, TypeError):
        raise JobNotFound(f"not a job id: {raw!r}", job_id=str(raw)) from None


def parse_format(raw: str) -> Format:
    try:
        return Format(raw.lower())
    except ValueError:
        raise TranscriptorError(
            f"unknown format {raw!r} — phase 1 supports txt and srt",
            code="unknown_format",
            resumable=False,
        ) from None


@dataclass(frozen=True, slots=True)
class StatusView:
    status: str
    stage: str | None
    progress_pct: float
    elapsed_s: float | None
    eta_s: float | None
    partial: bool
    covered_until_s: float
    resumable: bool
    error: str | None
    error_code: str | None
    heartbeat_age_s: float | None
    duration_s: float | None
    language: str | None
    word_count: int | None
    # Why a queued job has not started. None when nothing is holding it.
    queued_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptView:
    content: str
    total_chars: int
    truncated: bool
    partial: bool
    covered_until_s: float
    duration_s: float | None
    language: str | None
    word_count: int | None
    status: str


@dataclass(frozen=True, slots=True)
class JobSummary:
    job_id: str
    source: str
    title: str | None
    status: str
    created_at: str | None
    duration_s: float | None
    language: str | None
    resumable: bool
    word_count: int | None


class ReadService:
    def __init__(
        self, jobs: JobStore, transcripts: TranscriptStore, *, runner=None
    ) -> None:
        self._jobs = jobs
        self._transcripts = transcripts
        # Optional: without it, status simply omits the queued_reason.
        self._runner = runner

    async def status(self, job_id: str) -> StatusView:
        job = await self._jobs.get(parse_job_id(job_id))
        if job is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)

        now = dt.datetime.now(dt.UTC)
        elapsed = (
            ((job.finished_at or now) - job.started_at).total_seconds()
            if job.started_at
            else None
        )

        eta = None
        if job.status is JobStatus.RUNNING and elapsed and job.progress_pct > 1.0:
            eta = max(0.0, elapsed * (100.0 - job.progress_pct) / job.progress_pct)

        return StatusView(
            status=job.status.value,
            stage=job.stage.value if job.stage else None,
            progress_pct=round(job.progress_pct, 1),
            elapsed_s=round(elapsed, 1) if elapsed is not None else None,
            eta_s=round(eta, 1) if eta is not None else None,
            partial=job.partial,
            covered_until_s=job.covered_until_s,
            resumable=job.resumable,
            error=job.error,
            error_code=job.error_code,
            heartbeat_age_s=(
                round((now - job.heartbeat_at).total_seconds(), 1)
                if job.heartbeat_at
                else None
            ),
            duration_s=job.duration_s,
            language=job.language,
            word_count=job.word_count,
            queued_reason=(
                self._runner.hold_reason
                if self._runner is not None and job.status is JobStatus.QUEUED
                else None
            ),
        )

    async def search(
        self, job_id: str, query: str, *, context_seconds: float = 5.0,
        max_results: int = 10,
    ) -> list[dict]:
        """Find phrases with timestamps.

        Works on a running job: it searches what is committed so far, which
        falls out of per-chunk commits for free.
        """
        uid = parse_job_id(job_id)
        if await self._jobs.get(uid) is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)
        if not query.strip():
            raise TranscriptorError(
                "query is empty", code="invalid_parameter", resumable=False
            )
        hits = await self._transcripts.search(uid, query, limit=max_results)
        for hit in hits:
            hit["context_start"] = max(0.0, hit["start"] - context_seconds)
            hit["context_end"] = hit["end"] + context_seconds
        return hits

    async def segments(
        self, job_id: str, start_s: float, end_s: float, *, include_words: bool = False
    ) -> list[dict]:
        """Pull a time window.

        The intended access path for a long transcript, together with search —
        returning the whole thing would flood the caller's context.
        """
        uid = parse_job_id(job_id)
        if await self._jobs.get(uid) is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)
        if end_s < start_s:
            raise TranscriptorError(
                f"end_s ({end_s}) is before start_s ({start_s})",
                code="invalid_parameter",
                resumable=False,
            )
        found = await self._transcripts.segments_between(
            uid, start_s, end_s, include_words=include_words
        )
        return [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                **(
                    {"words": [
                        {"start": w.start, "end": w.end, "text": w.text}
                        for w in s.words
                    ]}
                    if include_words and s.words
                    else {}
                ),
            }
            for s in found
        ]

    async def list_jobs(
        self, *, limit: int = 20, query: str | None = None, status: str | None = None
    ) -> list[JobSummary]:
        """Recent jobs — a board-level view of what is in flight, stuck, done."""
        parsed = None
        if status is not None:
            try:
                parsed = JobStatus(status)
            except ValueError:
                raise TranscriptorError(
                    f"unknown status {status!r} — one of "
                    + ", ".join(s.value for s in JobStatus),
                    code="invalid_parameter",
                    resumable=False,
                ) from None

        jobs = await self._jobs.recent(limit=limit, status=parsed)
        summaries: list[JobSummary] = []
        for job in jobs:
            source = await self._jobs.get_source(job.source_id) or {}
            summary = JobSummary(
                job_id=str(job.id),
                source=source.get("raw") or source.get("source_key") or "",
                title=source.get("title"),
                status=job.status.value,
                created_at=job.created_at.isoformat() if job.created_at else None,
                duration_s=job.duration_s,
                language=job.language,
                resumable=job.resumable,
                word_count=job.word_count,
            )
            if query and query.lower() not in (
                f"{summary.source} {summary.title or ''}".lower()
            ):
                continue
            summaries.append(summary)
        return summaries

    async def get(
        self,
        job_id: str,
        *,
        fmt: str = "txt",
        mode: str = "preview",
        max_chars: int = 2000,
    ) -> TranscriptView:
        """``preview`` is the default for a reason: a 2-hour transcript is
        40,000+ tokens, and returning it whole makes the tool actively harmful
        to the calling agent."""
        uid = parse_job_id(job_id)
        job = await self._jobs.get(uid)
        if job is None:
            raise JobNotFound(f"no such job: {job_id}", job_id=job_id)

        content = await self._transcripts.render(uid, parse_format(fmt))
        total = len(content)

        truncated = mode != "full" and total > max_chars
        if truncated:
            content = content[:max_chars]

        return TranscriptView(
            content=content,
            total_chars=total,
            truncated=truncated,
            partial=job.partial and job.status is not JobStatus.DONE,
            covered_until_s=job.covered_until_s,
            duration_s=job.duration_s,
            language=job.language,
            word_count=job.word_count,
            status=job.status.value,
        )
