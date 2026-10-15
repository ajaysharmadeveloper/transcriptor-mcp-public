"""SQLAlchemy 2.0 declarative models.

These live in ``adapters`` on purpose.  The domain layer stays plain dataclasses
with no persistence awareness — the stores map between the two — so the
dependency rule holds and the ORM cannot leak upwards.  An import-linter
contract enforces it.

The models mirror ``migrations/001_initial.sql``, which remains the source of
truth for the schema: the partial unique index and the composite foreign key
are declared here so queries can reason about them, but they are *created* by
the migration, not by ``create_all``.  ``test_orm_matches_migrated_schema``
asserts the two never drift.

Full-text search is **not** modelled here. On Postgres it was a generated
``tsvector`` column and therefore part of the row; on SQLite it is two FTS5
virtual tables maintained by triggers, which the ORM cannot usefully describe.
The store queries them with hand-written SQL, which is honest about what they
are.
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    Text,
    TypeDecorator,
    Uuid,
    text as sa_text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class UtcDateTime(TypeDecorator):
    """A timezone-aware UTC datetime, stored as ISO-8601 text.

    SQLite has no date or time type, and SQLAlchemy's own ``DateTime`` on this
    dialect stores a naive string and hands back a naive datetime. That is not
    a cosmetic loss: the stale-heartbeat reaper compares stored times against
    ``datetime.now(UTC)``, and comparing naive to aware raises — while anything
    that silently coerced would make the reaper's ninety-second window depend
    on the machine's timezone, and wrong twice a year at the daylight-saving
    boundaries.

    So the conversion is explicit and lives in exactly one place. Naive values
    going in are *assumed* UTC rather than rejected, because the alternative is
    a crash on a value some future caller forgot to localise, and every writer
    in this codebase already uses ``dt.datetime.now(dt.UTC)``.
    """

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC).isoformat()

    def process_result_value(self, value, dialect):  # noqa: ANN001
        if value is None:
            return None
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        return parsed.astimezone(dt.UTC)


def _utcnow() -> dt.datetime:
    """Defaults are applied in Python, not by the database.

    SQLite's ``CURRENT_TIMESTAMP`` is naive, second-resolution and in UTC
    without saying so — three different ways to disagree with what
    :class:`UtcDateTime` reads back.
    """
    return dt.datetime.now(dt.UTC)

# Mirrors the `jobs_one_active_idx` predicate and `JobStatus.is_active`.
ACTIVE_STATUSES = ("queued", "running", "paused")
TERMINAL_STATUSES = ("done", "failed", "cancelled", "interrupted")


class Base(DeclarativeBase):
    """No schema is set on the models — SQLite has none.

    Tests get isolation from a database file in a temporary directory, which is
    simpler than the throwaway Postgres schema it replaces and needs no running
    service at all.
    """


class Source(Base):
    """One row per distinct video or file ever seen."""

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    kind: Mapped[str] = mapped_column(Text)
    raw: Mapped[str] = mapped_column(Text)
    # Normalised URL, or absolute resolved path.  Unique: layer-1 dedupe.
    source_key: Mapped[str] = mapped_column(Text, unique=True)
    # Local files: size + mtime.  Detects a file edited in place.
    file_fingerprint: Mapped[str | None] = mapped_column(Text, default=None)
    # blake2b over decoded PCM.  Layer-2 dedupe, filled after extraction.
    audio_hash: Mapped[str | None] = mapped_column(Text, default=None)
    title: Mapped[str | None] = mapped_column(Text, default=None)
    uploader: Mapped[str | None] = mapped_column(Text, default=None)
    platform: Mapped[str | None] = mapped_column(Text, default=None)
    duration_s: Mapped[float | None] = mapped_column(Float, default=None)
    has_manual_captions: Mapped[bool | None] = mapped_column(Boolean, default=None)
    has_auto_captions: Mapped[bool | None] = mapped_column(Boolean, default=None)
    # first_seen_at, last_seen_at and request_count used to live here. Audited
    # before the SQLite port: the first was read by nothing at all, the other
    # two were written on every repeat request and read by nothing. Dropped
    # rather than carried across out of habit.

    __table_args__ = (
        CheckConstraint("kind IN ('file','url')", name="sources_kind_check"),
        Index("sources_audio_hash_idx", "audio_hash"),
        Index("sources_platform_idx", "platform"),
    )


class Request(Base):
    """Every tool call that started or touched a job — the audit trail."""

    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tool: Mapped[str] = mapped_column(Text)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("sources.id", ondelete="SET NULL"), default=None
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), default=None
    )
    params: Mapped[dict] = mapped_column(JSON)
    outcome: Mapped[str | None] = mapped_column(Text, default=None)
    latency_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=_utcnow)

    __table_args__ = (
        Index("requests_created_idx", sa_text("created_at DESC")),
        Index("requests_source_idx", "source_id", sa_text("created_at DESC")),
    )


class Job(Base):
    """One transcription attempt.  The queue is this table."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    source_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE")
    )
    result_key: Mapped[str] = mapped_column(Text)
    origin: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(Text)
    stage: Mapped[str | None] = mapped_column(Text, default=None)
    resumable: Mapped[bool] = mapped_column(Boolean, server_default=sa_text("0"))
    # force=true: skip dedupe layers 1 and 2. Stored rather than passed as an
    # argument, because layer 2 runs a stage later than the create call — and so
    # a resumed forced job stays forced.
    forced: Mapped[bool] = mapped_column(Boolean, server_default=sa_text("0"))
    heartbeat_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)
    progress_pct: Mapped[float] = mapped_column(Float, server_default=sa_text("0"))
    partial: Mapped[bool] = mapped_column(Boolean, server_default=sa_text("0"))
    covered_until_s: Mapped[float] = mapped_column(Float, server_default=sa_text("0"))
    last_chunk_idx: Mapped[int] = mapped_column(Integer, server_default=sa_text("-1"))
    error: Mapped[str | None] = mapped_column(Text, default=None)
    error_code: Mapped[str | None] = mapped_column(Text, default=None)
    # The effective language of the transcript, detected or forced.
    language: Mapped[str | None] = mapped_column(Text, default=None)
    # What the caller asked for; NULL means auto-detect. Part of result_key, and
    # what layer-2 dedupe matches on — see migration 003.
    requested_language: Mapped[str | None] = mapped_column(Text, default=None)
    model: Mapped[str | None] = mapped_column(Text, default=None)
    duration_s: Mapped[float | None] = mapped_column(Float, default=None)
    word_count: Mapped[int | None] = mapped_column(Integer, default=None)
    transcribe_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    # VAD regions as [[start, end], ...] on the original clock.  NULL means not
    # detected yet; [] means detected and silent — a distinction the Detect
    # stage's `satisfied` probe depends on.
    speech_timeline: Mapped[list | None] = mapped_column(JSON, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=_utcnow)
    started_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)
    finished_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)

    events: Mapped[list[JobEvent]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="raise"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued','running','paused','done','failed','cancelled','interrupted')",
            name="jobs_status_check",
        ),
        Index("jobs_source_idx", "source_id", sa_text("created_at DESC")),
        Index("jobs_status_idx", "status", sa_text("created_at DESC")),
        # At most ONE active job per result_key — enforced by the database, not
        # by application logic, because check-then-act in Python races.
        # SQLite supports partial indexes, so invariant I1 keeps exactly the
        # guarantee it had on Postgres.
        Index(
            "jobs_one_active_idx",
            "result_key",
            unique=True,
            sqlite_where=sa_text("status IN ('queued','running','paused')"),
        ),
        Index(
            "jobs_reuse_idx",
            "result_key",
            sa_text("finished_at DESC"),
            sqlite_where=sa_text("status = 'done'"),
        ),
    )


class JobEvent(Base):
    """Every status transition.  The audit trail, without making current state a
    fold over history."""

    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE")
    )
    from_status: Mapped[str | None] = mapped_column(Text, default=None)
    to_status: Mapped[str] = mapped_column(Text)
    stage: Mapped[str | None] = mapped_column(Text, default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=_utcnow)

    job: Mapped[Job] = relationship(back_populates="events", lazy="raise")

    __table_args__ = (Index("job_events_job_idx", "job_id", "at"),)


class Chunk(Base):
    """One unit of work within a job — and the checkpoint unit."""

    __tablename__ = "chunks"

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, server_default=sa_text("0"))
    engine_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    committed_at: Mapped[dt.datetime | None] = mapped_column(UtcDateTime, default=None)

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','done','failed')", name="chunks_status_check"
        ),
    )


class Render(Base):
    """A lazily cached output format.

    Regenerable by definition: any correction to a segment deletes the affected
    rows and the next read rebuilds them (invariant I7).
    """

    __tablename__ = "renders"

    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    format: Mapped[str] = mapped_column(Text, primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    bytes: Mapped[int] = mapped_column(Integer)
    generated_at: Mapped[dt.datetime] = mapped_column(UtcDateTime, default=_utcnow)

    __table_args__ = (
        CheckConstraint(
            "format IN ('txt','srt','vtt','json')", name="renders_format_check"
        ),
    )


class Segment(Base):
    """The transcript itself — the source of truth for every output format.

    The composite foreign key into ``chunks`` is what enforces invariant I3: a
    segment cannot exist without its chunk row.
    """

    __tablename__ = "segments"

    job_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_idx: Mapped[int] = mapped_column(Integer)
    start_s: Mapped[float] = mapped_column(Float)
    end_s: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    words: Mapped[list | None] = mapped_column(JSON, default=None)
    # No `tsv` column. On Postgres full-text search was a generated column and
    # so part of the row; here it is two FTS5 virtual tables kept in step by
    # triggers, which the ORM has no way to describe. See the module docstring
    # and `migrations/001_initial.sql`.

    __table_args__ = (
        PrimaryKeyConstraint("job_id", "idx"),
        ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        ForeignKeyConstraint(
            ["job_id", "chunk_idx"], ["chunks.job_id", "chunks.idx"], ondelete="CASCADE"
        ),
        Index("segments_time_idx", "job_id", "start_s"),
    )
