"""Persistence contracts: JobStore, TranscriptStore, RequestLog.

All async, because every one of these is IO against the database.  The stores are
the only place SQL lives; services speak in domain types.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from transcriptor.domain.job import ClaimResult, Job, JobStatus, Origin, Stage
from transcriptor.domain.source import CaptionPolicy, SourceRef
from transcriptor.domain.transcript import Chunk, Format, Segment


@runtime_checkable
class JobStore(Protocol):
    """Job identity, claim, lifecycle and the queue.

    The queue *is* this table: ``status = 'queued'``.  No Redis, no Celery, no
    in-memory queue — so it survives restart with no reconciliation step, and
    the dedupe invariant lives on the same rows as the queue.
    """

    async def upsert_source(
        self, ref: SourceRef, *, fingerprint: str | None = None, **meta: Any
    ) -> uuid.UUID:
        """Insert or refresh the source row, returning its id."""
        ...

    async def claim_or_attach(
        self,
        key: str,
        source_id: uuid.UUID,
        *,
        model: str,
        language: str | None,
        captions: CaptionPolicy,
        fingerprint: str | None = None,
        force: bool = False,
    ) -> ClaimResult:
        """Atomically create, attach to, or reuse a job for ``key``.

        Runs inside one transaction holding ``pg_advisory_xact_lock`` and uses
        ``INSERT ... ON CONFLICT DO NOTHING`` against the partial unique index.
        **Never check-then-insert** — two concurrent callers can both read "no
        active job" before either writes.
        """
        ...

    async def get(self, job_id: uuid.UUID) -> Job | None: ...

    async def get_source(self, source_id: uuid.UUID) -> dict[str, Any] | None: ...

    async def set_source_meta(
        self, source_id: uuid.UUID, *, duration_s: float | None = None,
        title: str | None = None,
    ) -> None: ...

    async def set_audio_hash(self, source_id: uuid.UUID, audio_hash: str) -> None:
        """Layer-2 dedupe key, filled after extraction."""
        ...

    async def set_stage(self, job_id: uuid.UUID, stage: Stage) -> None:
        """Advance the pipeline step without changing status."""
        ...

    async def set_job_fields(self, job_id: uuid.UUID, **fields: Any) -> None:
        """Write a stage checkpoint onto the job row, without a transition."""
        ...

    async def next_queued(self) -> Job | None:
        """Claim the oldest queued job with ``FOR UPDATE SKIP LOCKED``."""
        ...

    async def transition(
        self,
        job_id: uuid.UUID,
        to: JobStatus,
        *,
        stage: Stage | None = None,
        note: str | None = None,
        **fields: Any,
    ) -> None:
        """Move the job, rejecting illegal transitions, and write a
        ``job_events`` row in the same transaction."""
        ...

    async def delete(self, job_id: uuid.UUID) -> bool:
        """Remove a job and everything that cascades from it."""
        ...

    async def heartbeat(self, job_id: uuid.UUID) -> None:
        """Touch ``heartbeat_at``.  A job whose heartbeat goes stale is what
        the reaper converts to ``interrupted``."""
        ...

    async def reap_stale(self, older_than_s: int) -> list[uuid.UUID]:
        """Mark jobs whose heartbeat has gone stale as ``interrupted``.

        Runs at startup and every 60s.  Startup matters most: after a crash the
        first thing to do is release orphaned jobs, or every later request
        attaches to a job that will never finish.
        """
        ...

    async def find_by_result_key(self, key: str) -> dict[str, Any] | None:
        """The newest job for a result_key, for transcript_probe."""
        ...

    async def find_done_by_audio_hash(
        self,
        audio_hash: str,
        *,
        model: str,
        language: str | None,
        exclude_job_id: uuid.UUID | None = None,
    ) -> uuid.UUID | None:
        """Layer 2 — a completed job over byte-identical decoded audio."""
        ...

    async def queue_depth(self) -> int: ...

    async def recent(
        self, *, limit: int = 20, status: JobStatus | None = None
    ) -> list[Job]: ...


@runtime_checkable
class TranscriptStore(Protocol):
    """Segments, chunks, and everything derived from them."""

    async def plan_chunks(self, job_id: uuid.UUID, chunks: list[Chunk]) -> None:
        """Write the pending chunk rows for a job."""
        ...

    async def commit_chunk(
        self, job_id: uuid.UUID, chunk: Chunk, segments: list[Segment]
    ) -> None:
        """Chunk row, segments and job progress, in **one transaction**.

        Partial chunks never exist, so ``segments`` can never contain rows whose
        chunk is not ``done`` (invariant I3).  This is also what makes partial
        results and resume possible.
        """
        ...

    async def chunks(self, job_id: uuid.UUID) -> list[Chunk]: ...

    async def segments(
        self, job_id: uuid.UUID, *, include_words: bool = False
    ) -> list[Segment]: ...

    async def segments_between(
        self, job_id: uuid.UUID, a: float, b: float, *, include_words: bool = False
    ) -> list[Segment]: ...

    async def replace_segments(
        self, job_id: uuid.UUID, segments: list[Segment]
    ) -> int:
        """Rewrite segment text in place. Callers must drop renders after."""
        ...

    async def copy_from(
        self, source_job_id: uuid.UUID, dest_job_id: uuid.UUID
    ) -> int:
        """Copy a finished transcript onto another job, for a layer-2 hit."""
        ...

    async def next_segment_idx(self, job_id: uuid.UUID) -> int:
        """The next free segment index, so a resumed job continues numbering."""
        ...

    async def word_count(self, job_id: uuid.UUID) -> int: ...

    async def render(self, job_id: uuid.UUID, fmt: Format) -> str:
        """Render a format from segments, caching it once the job is final."""
        ...

    async def drop_renders(self, job_id: uuid.UUID) -> int:
        """Invalidate the render cache — renders are always regenerable."""
        ...

    async def search(
        self, job_id: uuid.UUID, query: str, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Full-text with a trigram fallback for a half-remembered phrase."""
        ...


@runtime_checkable
class RequestLog(Protocol):
    """Every tool call that started or touched a job.

    The audit trail: what was asked for, when, by which tool, and what came back.
    """

    async def record(
        self,
        *,
        tool: str,
        params: dict[str, Any],
        source_id: uuid.UUID | None = None,
        job_id: uuid.UUID | None = None,
        outcome: str | None = None,
        latency_ms: int | None = None,
    ) -> None: ...
