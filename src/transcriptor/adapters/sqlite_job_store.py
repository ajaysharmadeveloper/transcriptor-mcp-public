"""SqliteJobStore and SqliteRequestLog — SQLAlchemy 2.0 async ORM over SQLite.

Every method is one unit of work: writes open ``db.begin()`` (commit on success,
rollback on any error), reads open ``db.session()``.  No session is held on the
store, so two concurrent callers never share one.

ORM rows never escape this module: each method maps to a domain dataclass before
returning, which is what keeps the ORM behind the adapter boundary.
"""

from __future__ import annotations

import datetime as dt

import uuid
from typing import Any

from sqlalchemy import delete, func, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from transcriptor.adapters import orm
from transcriptor.adapters.orm import _utcnow
from transcriptor.adapters.db import Database
from transcriptor.domain.job import (
    Attached,
    ClaimResult,
    Created,
    Job,
    JobEvent,
    JobStatus,
    Origin,
    ReuseReason,
    Reused,
    Stage,
    assert_transition,
)
from transcriptor.domain.source import CaptionPolicy, SourceRef
from transcriptor.domain.transcript import Timeline

ACTIVE = [s.value for s in JobStatus if s.is_active]


def _to_job(row: orm.Job) -> Job:
    """ORM row → domain dataclass.  Nothing mapped leaves this module."""
    return Job(
        id=row.id,
        source_id=row.source_id,
        result_key=row.result_key,
        origin=Origin(row.origin) if row.origin else None,
        status=JobStatus(row.status),
        stage=Stage(row.stage) if row.stage else None,
        resumable=row.resumable,
        forced=row.forced,
        heartbeat_at=row.heartbeat_at,
        progress_pct=row.progress_pct,
        partial=row.partial,
        covered_until_s=row.covered_until_s,
        last_chunk_idx=row.last_chunk_idx,
        error=row.error,
        error_code=row.error_code,
        language=row.language,
        requested_language=row.requested_language,
        model=row.model,
        duration_s=row.duration_s,
        word_count=row.word_count,
        transcribe_ms=row.transcribe_ms,
        speech_timeline=Timeline.from_json(row.speech_timeline),
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )


class SqliteJobStore:
    """Implements :class:`transcriptor.ports.stores.JobStore`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------- sources

    async def upsert_source(
        self, ref: SourceRef, *, fingerprint: str | None = None, **meta: Any
    ) -> uuid.UUID:
        """Insert or refresh the source row.

        ``source_key`` is unique, so a repeat request refreshes the metadata on
        the existing row rather than creating a second one.

        Every refreshed field is wrapped in ``coalesce``, which is the point:
        a later probe that learned less than an earlier one must not erase what
        is already known. A URL resolved from a cache has no duration; the row
        that already had one keeps it.
        """
        stmt = sqlite_insert(orm.Source).values(
            id=uuid.uuid4(),
            kind=ref.kind.value,
            raw=ref.raw,
            source_key=ref.key,
            file_fingerprint=fingerprint,
            platform=ref.platform,
            title=meta.get("title"),
            uploader=meta.get("uploader"),
            duration_s=meta.get("duration_s"),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[orm.Source.source_key],
            set_={
                "raw": stmt.excluded.raw,
                "file_fingerprint": func.coalesce(
                    stmt.excluded.file_fingerprint, orm.Source.file_fingerprint
                ),
                "title": func.coalesce(stmt.excluded.title, orm.Source.title),
                "uploader": func.coalesce(stmt.excluded.uploader, orm.Source.uploader),
                "duration_s": func.coalesce(
                    stmt.excluded.duration_s, orm.Source.duration_s
                ),
            },
        ).returning(orm.Source.id)

        async with self._db.begin() as session:
            return (await session.execute(stmt)).scalar_one()

    async def get_source(self, source_id: uuid.UUID) -> dict[str, Any] | None:
        async with self._db.session() as session:
            row = await session.get(orm.Source, source_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "kind": row.kind,
                "raw": row.raw,
                "source_key": row.source_key,
                "file_fingerprint": row.file_fingerprint,
                "audio_hash": row.audio_hash,
                "title": row.title,
                "uploader": row.uploader,
                "platform": row.platform,
                "duration_s": row.duration_s,
            }

    async def set_source_meta(
        self,
        source_id: uuid.UUID,
        *,
        duration_s: float | None = None,
        title: str | None = None,
    ) -> None:
        async with self._db.begin() as session:
            await session.execute(
                update(orm.Source)
                .where(orm.Source.id == source_id)
                .values(
                    duration_s=func.coalesce(duration_s, orm.Source.duration_s),
                    title=func.coalesce(title, orm.Source.title),
                )
            )

    async def set_audio_hash(self, source_id: uuid.UUID, audio_hash: str) -> None:
        """Layer-2 dedupe key, filled after extraction."""
        async with self._db.begin() as session:
            await session.execute(
                update(orm.Source)
                .where(orm.Source.id == source_id)
                .values(audio_hash=audio_hash)
            )

    # --------------------------------------------------------------- claim

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
        """Create, attach to, or reuse a job for ``result_key``.

        Order: reuse a completed job, resume a resumable one, else insert; a
        conflict means an active job already owns the key, so attach to it.

        The Postgres version took ``pg_advisory_xact_lock`` around this, so the
        reuse lookup and the insert could not interleave with another caller.
        There is no lock here, and losing it costs nothing — because the lock
        was never what made this correct. **The partial unique index is**, and
        the conflict branch below is what handles the race: if another caller
        inserted between our lookup and our insert, ``ON CONFLICT DO NOTHING``
        returns no row and we attach to theirs instead of starting a second
        forty-minute inference.

        SQLite serialises writers in any case, so the window the lock protected
        does not exist here. What remains is one transaction that either wins
        the insert or discovers it lost, which is the same answer either way.
        """
        async with self._db.begin() as session:
            if not force:
                # Layer 1 — a completed job for this exact key.
                done = (
                    await session.execute(
                        select(orm.Job.id)
                        .where(orm.Job.result_key == key, orm.Job.status == "done")
                        .order_by(orm.Job.finished_at.desc().nullslast())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if done is not None:
                    return Reused(done, ReuseReason.EXACT_SOURCE)

                # A resumable terminal job re-enters at its last committed
                # chunk rather than starting over.  Same row, back to queued.
                resumable = (
                    await session.execute(
                        select(orm.Job)
                        .where(
                            orm.Job.result_key == key,
                            orm.Job.status.in_(["interrupted", "failed"]),
                            orm.Job.resumable.is_(True),
                        )
                        .order_by(orm.Job.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if resumable is not None:
                    frm = JobStatus(resumable.status)
                    assert_transition(frm, JobStatus.QUEUED)
                    resumable.status = JobStatus.QUEUED.value
                    resumable.stage = None
                    resumable.error = None
                    resumable.error_code = None
                    resumable.finished_at = None
                    resumable.heartbeat_at = None
                    session.add(
                        orm.JobEvent(
                            job_id=resumable.id,
                            from_status=frm.value,
                            to_status=JobStatus.QUEUED.value,
                            note="resumed",
                        )
                    )
                    return Reused(resumable.id, ReuseReason.RESUMED)

            job_id = uuid.uuid4()
            created = (
                await session.execute(
                    sqlite_insert(orm.Job)
                    .values(
                        id=job_id,
                        source_id=source_id,
                        result_key=key,
                        status=JobStatus.QUEUED.value,
                        language=language,
                        requested_language=language,
                        model=model,
                        forced=force,
                    )
                    .on_conflict_do_nothing()
                    .returning(orm.Job.id)
                )
            ).scalar_one_or_none()

            if created is not None:
                session.add(
                    orm.JobEvent(
                        job_id=job_id,
                        to_status=JobStatus.QUEUED.value,
                        note="created",
                    )
                )
                return Created(job_id)

            # No row returned means an active job already owns this key.
            # Layer 3: attach rather than start a second transcription.
            active = (
                await session.execute(
                    select(orm.Job.id).where(
                        orm.Job.result_key == key, orm.Job.status.in_(ACTIVE)
                    )
                )
            ).scalar_one_or_none()
            if active is None:  # pragma: no cover - the lock makes this unreachable
                raise RuntimeError(
                    "insert conflicted but no active job owns the key — "
                    "the advisory lock or the partial index is not doing its job"
                )
            return Attached(active, ReuseReason.ALREADY_RUNNING)

    async def find_by_result_key(self, key: str) -> dict[str, Any] | None:
        """The newest job for a result_key, whatever its status.

        What ``transcript_probe`` reports as ``existing``: an exact match, so it
        says what *would* happen rather than what might.
        """
        async with self._db.session() as session:
            row = (
                await session.execute(
                    select(orm.Job)
                    .where(orm.Job.result_key == key)
                    .order_by(orm.Job.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "job_id": str(row.id),
            "status": row.status,
            "origin": row.origin,
            "model": row.model,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "resumable": row.resumable,
            "covered_until_s": row.covered_until_s,
            "word_count": row.word_count,
        }

    async def find_done_by_audio_hash(
        self,
        audio_hash: str,
        *,
        model: str,
        language: str | None,
        exclude_job_id: uuid.UUID | None = None,
    ) -> uuid.UUID | None:
        """Layer 2 — a completed job over byte-identical decoded audio.

        ``audio_hash`` is blake2b over the decoded 16kHz mono PCM, so it is the
        same regardless of container, bitrate, or where the file came from.  It
        catches what layer 1 cannot: the same video from a shortened URL and a
        full one, a local copy of something already pulled from YouTube, a
        re-upload at a different resolution.

        Matched on model and the *requested* language, not the detected one:
        a completed auto-detect job ends up with language='en', so matching on
        that would stop a fresh auto-detect request from ever hitting.
        Caption *policy* is not stored anywhere, so it is approximated by
        requiring the existing job to be ASR-sourced — a documented limitation
        rather than a silent one.
        """
        stmt = (
            select(orm.Job.id)
            .join(orm.Source, orm.Source.id == orm.Job.source_id)
            .where(
                orm.Source.audio_hash == audio_hash,
                orm.Job.status == JobStatus.DONE.value,
                orm.Job.model == model,
                orm.Job.requested_language.is_not_distinct_from(language),
                orm.Job.origin == Origin.ASR.value,
            )
            .order_by(orm.Job.finished_at.desc().nullslast())
            .limit(1)
        )
        if exclude_job_id is not None:
            stmt = stmt.where(orm.Job.id != exclude_job_id)

        async with self._db.session() as session:
            return (await session.execute(stmt)).scalar_one_or_none()

    # ------------------------------------------------------------ lifecycle

    async def get(self, job_id: uuid.UUID) -> Job | None:
        async with self._db.session() as session:
            row = await session.get(orm.Job, job_id)
            return _to_job(row) if row else None

    async def next_queued(self) -> Job | None:
        """Claim the oldest queued job and move it to ``running``.

        There is no ``SKIP LOCKED`` here and none is needed: SQLite serialises
        writers, so the claim and the transition are one transaction that no
        other writer can interleave with. A second process — and Claude Desktop
        really does start two (§18a) — waits out its ``busy_timeout`` and then
        finds the job already ``running``. The row is durably ``running``
        before the transaction commits, so a job cannot be handed out twice.
        """
        async with self._db.begin() as session:
            row = (
                await session.execute(
                    select(orm.Job)
                    .where(orm.Job.status == JobStatus.QUEUED.value)
                    .order_by(orm.Job.created_at)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            assert_transition(JobStatus.QUEUED, JobStatus.RUNNING)
            row.status = JobStatus.RUNNING.value
            row.stage = Stage.RESOLVING.value
            row.heartbeat_at = _utcnow()
            # In Python rather than SQL coalesce: the row is already loaded,
            # and a column expression assigned to a loaded attribute would be
            # written back as a literal on the next flush.
            row.started_at = row.started_at or _utcnow()
            session.add(
                orm.JobEvent(
                    job_id=row.id,
                    from_status=JobStatus.QUEUED.value,
                    to_status=JobStatus.RUNNING.value,
                    stage=Stage.RESOLVING.value,
                    note="claimed",
                )
            )
            await session.flush()
            await session.refresh(row)
            return _to_job(row)

    async def transition(
        self,
        job_id: uuid.UUID,
        to: JobStatus,
        *,
        stage: Stage | None = None,
        note: str | None = None,
        **fields: Any,
    ) -> None:
        """Move the job and log the transition in the same transaction.

        Illegal transitions raise rather than being silently absorbed — a job
        that moves somewhere the state machine forbids is a bug in the caller.
        """
        async with self._db.begin() as session:
            row = (
                await session.execute(
                    # No row lock: the whole database is locked to one
                    # writer for the life of this transaction, which is a
                    # stronger guarantee than SELECT FOR UPDATE gave.
                    select(orm.Job).where(orm.Job.id == job_id)
                )
            ).scalar_one_or_none()
            if row is None:
                raise LookupError(f"no such job: {job_id}")

            frm = JobStatus(row.status)
            assert_transition(frm, to)

            row.status = to.value
            if stage is not None:
                row.stage = stage.value
            elif to.is_terminal:
                row.stage = None

            for column, value in fields.items():
                setattr(row, column, value)

            if to is JobStatus.RUNNING:
                row.started_at = row.started_at or _utcnow()
                row.heartbeat_at = _utcnow()
            if to.is_terminal:
                row.finished_at = _utcnow()

            session.add(
                orm.JobEvent(
                    job_id=job_id,
                    from_status=frm.value,
                    to_status=to.value,
                    stage=stage.value if stage else None,
                    note=note,
                )
            )

    async def set_job_fields(self, job_id: uuid.UUID, **fields: Any) -> None:
        """Write a stage's checkpoint onto the job row.

        Deliberately separate from :meth:`transition`: a checkpoint is not a
        status change, and routing it through the state machine would mean
        inventing self-transitions the transition table rightly forbids.
        """
        if not fields:
            return
        async with self._db.begin() as session:
            await session.execute(
                update(orm.Job).where(orm.Job.id == job_id).values(**fields)
            )

    async def set_stage(self, job_id: uuid.UUID, stage: Stage) -> None:
        """Advance the pipeline step without changing status.

        A stage change is not a status transition, so it does not go through
        :meth:`transition` — but it still writes a ``job_events`` row, because
        "which stage is this job stuck in" should be a query, not a guess.
        """
        async with self._db.begin() as session:
            row = (
                await session.execute(
                    # No row lock: the whole database is locked to one
                    # writer for the life of this transaction, which is a
                    # stronger guarantee than SELECT FOR UPDATE gave.
                    select(orm.Job).where(orm.Job.id == job_id)
                )
            ).scalar_one_or_none()
            if row is None:
                raise LookupError(f"no such job: {job_id}")
            status = row.status
            row.stage = stage.value
            row.heartbeat_at = _utcnow()
            session.add(
                orm.JobEvent(
                    job_id=job_id,
                    from_status=status,
                    to_status=status,
                    stage=stage.value,
                    note="stage",
                )
            )

    async def delete(self, job_id: uuid.UUID) -> bool:
        """Remove a job.  chunks, segments, job_events and renders cascade."""
        async with self._db.begin() as session:
            result = await session.execute(
                delete(orm.Job).where(orm.Job.id == job_id)
            )
        return bool(result.rowcount)

    async def heartbeat(self, job_id: uuid.UUID) -> None:
        async with self._db.begin() as session:
            await session.execute(
                update(orm.Job)
                .where(orm.Job.id == job_id, orm.Job.status.in_(ACTIVE))
                .values(heartbeat_at=_utcnow())
            )

    async def reap_stale(self, older_than_s: int) -> list[uuid.UUID]:
        """Release jobs orphaned by a dead process.

        Runs at startup and every 60s.  Startup matters most: after a crash, the
        first thing to do is release orphaned jobs, or every later request
        attaches to a job that will never finish — which would undo the whole
        dedupe design.
        """
        async with self._db.begin() as session:
            rows = (
                await session.execute(
                    update(orm.Job)
                    .where(
                        orm.Job.status.in_(ACTIVE),
                        orm.Job.heartbeat_at.is_not(None),
                        # The cutoff is computed in Python and bound as a
                        # value. SQLite has no interval type, and its date
                        # functions work on strings — doing this in SQL would
                        # mean string-comparing timestamps and hoping the
                        # formats match.
                        orm.Job.heartbeat_at
                        < _utcnow() - dt.timedelta(seconds=older_than_s),
                    )
                    .values(
                        status=JobStatus.INTERRUPTED.value,
                        resumable=True,
                        stage=None,
                        finished_at=_utcnow(),
                        error_code=func.coalesce(orm.Job.error_code, "process_died"),
                        error=func.coalesce(orm.Job.error, "heartbeat went stale"),
                    )
                    .returning(orm.Job.id)
                )
            ).scalars().all()

            for job_id in rows:
                session.add(
                    orm.JobEvent(
                        job_id=job_id,
                        to_status=JobStatus.INTERRUPTED.value,
                        note=f"heartbeat older than {older_than_s}s",
                    )
                )
            return list(rows)

    # ---------------------------------------------------------------- reads

    async def queue_depth(self) -> int:
        async with self._db.session() as session:
            return (
                await session.execute(
                    select(func.count())
                    .select_from(orm.Job)
                    .where(orm.Job.status == JobStatus.QUEUED.value)
                )
            ).scalar_one()

    async def active_count(self) -> int:
        async with self._db.session() as session:
            return (
                await session.execute(
                    select(func.count())
                    .select_from(orm.Job)
                    .where(orm.Job.status.in_(["running", "paused"]))
                )
            ).scalar_one()

    async def recent(
        self, *, limit: int = 20, status: JobStatus | None = None
    ) -> list[Job]:
        stmt = select(orm.Job).order_by(orm.Job.created_at.desc()).limit(limit)
        if status is not None:
            stmt = stmt.where(orm.Job.status == status.value)
        async with self._db.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [_to_job(r) for r in rows]

    async def events(self, job_id: uuid.UUID) -> list[JobEvent]:
        async with self._db.session() as session:
            rows = (
                await session.execute(
                    select(orm.JobEvent)
                    .where(orm.JobEvent.job_id == job_id)
                    .order_by(orm.JobEvent.at, orm.JobEvent.id)
                )
            ).scalars().all()
        return [
            JobEvent(
                job_id=r.job_id,
                from_status=JobStatus(r.from_status) if r.from_status else None,
                to_status=JobStatus(r.to_status),
                stage=Stage(r.stage) if r.stage else None,
                note=r.note,
                at=r.at,
            )
            for r in rows
        ]


class SqliteRequestLog:
    """Implements :class:`transcriptor.ports.stores.RequestLog`."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def record(
        self,
        *,
        tool: str,
        params: dict[str, Any],
        source_id: uuid.UUID | None = None,
        job_id: uuid.UUID | None = None,
        outcome: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        async with self._db.begin() as session:
            session.add(
                orm.Request(
                    tool=tool,
                    source_id=source_id,
                    job_id=job_id,
                    params=_jsonable(params),
                    outcome=outcome,
                    latency_ms=latency_ms,
                )
            )


def _jsonable(value: Any) -> Any:
    """JSONB takes the params as received — but a Path or UUID is not JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
