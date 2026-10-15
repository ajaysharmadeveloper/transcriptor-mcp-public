"""A one-off import from the Postgres database this project used to keep.

Transcriptor moved to SQLite so that installing it needs no database server.
That left anyone who had already been using it with
transcripts in a Postgres nobody reads any more — hours of inference, still
sitting there, unreachable. This brings them across.

**It is deliberately not part of the running server.** It is a CLI subcommand,
it runs once, and ``asyncpg`` is an optional extra rather than a dependency:
a tool for a migration that most people will never perform should not be in
everybody's install.

## What has to be converted, and why each one bites

The two schemas describe the same data, but the types underneath do not match:

``UUID`` → 32 hex characters
    SQLAlchemy's ``Uuid`` stores UUIDs on SQLite without dashes. Insert them
    with dashes and every foreign key still *looks* right while joining to
    nothing — the failure is silent, which is the worst kind.

``timestamptz`` → ISO-8601 text
    Matching ``orm.UtcDateTime``. Postgres hands back aware datetimes; SQLite
    stores what it is given. A naive value here would make the reaper's
    heartbeat comparison raise later, a long way from the cause.

``jsonb`` → JSON text
    asyncpg returns JSONB as a string unless a codec is registered, so it is
    passed through as text — but a dict or list is handled too, because that
    depends on driver configuration this module does not control.

``boolean`` → 0/1
    sqlite3 does this itself. Listed because it is the one that looks like it
    needs work and does not.

Three columns are dropped rather than converted: ``sources.first_seen_at``,
``last_seen_at`` and ``request_count`` do not exist in the new schema. They
were audited as dead before the port.

## Order matters

Rows go in parent-first — sources, jobs, chunks, segments, job_events,
renders, requests — because ``PRAGMA foreign_keys = ON`` is set on every
connection and will reject a child whose parent is not there yet. ``requests``
is last because it points at both sources and jobs.

Every insert is ``INSERT OR IGNORE``, so running this twice is safe and a
half-finished run can simply be repeated.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger(__name__)

# Parent before child. The FK pragma is on, so this order is not cosmetic.
TABLES: tuple[str, ...] = (
    "sources",
    "jobs",
    "chunks",
    "segments",
    "job_events",
    "renders",
    "requests",
)

# Columns to read from Postgres, per table. Written out rather than `SELECT *`
# so that a column added to one schema and not the other fails here, loudly,
# instead of being silently dropped or silently rejected.
#
# `segments.tsv` is absent on purpose: it was a generated column, and its
# replacement is an FTS5 index that the triggers rebuild as rows arrive.
COLUMNS: dict[str, tuple[str, ...]] = {
    "sources": (
        "id", "kind", "raw", "source_key", "file_fingerprint", "audio_hash",
        "title", "uploader", "platform", "duration_s", "has_manual_captions",
        "has_auto_captions",
    ),
    "jobs": (
        "id", "source_id", "result_key", "origin", "status", "stage",
        "resumable", "forced", "heartbeat_at", "progress_pct", "partial",
        "covered_until_s", "last_chunk_idx", "error", "error_code", "language",
        "requested_language", "model", "duration_s", "word_count",
        "transcribe_ms", "speech_timeline", "created_at", "started_at",
        "finished_at",
    ),
    "chunks": (
        "job_id", "idx", "start_s", "end_s", "status", "attempts", "engine_ms",
        "committed_at",
    ),
    "segments": ("job_id", "idx", "chunk_idx", "start_s", "end_s", "text", "words"),
    "job_events": ("id", "job_id", "from_status", "to_status", "stage", "note", "at"),
    "renders": ("job_id", "format", "content", "bytes", "generated_at"),
    "requests": (
        "id", "tool", "source_id", "job_id", "params", "outcome", "latency_ms",
        "created_at",
    ),
}


@dataclass
class ImportReport:
    """What happened, per table.  ``skipped`` means the row was already here."""

    imported: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    source_rows: dict[str, int] = field(default_factory=dict)

    @property
    def total_imported(self) -> int:
        return sum(self.imported.values())

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())


def _convert(value: object) -> object:
    """One Postgres value, in the form SQLite and the ORM agree on."""
    if isinstance(value, uuid.UUID):
        # 32 hex characters, no dashes — what SQLAlchemy's Uuid type writes.
        return value.hex
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.UTC)
        return value.astimezone(dt.UTC).isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


async def import_from_postgres(
    dsn: str, engine: AsyncEngine, *, dry_run: bool = False
) -> ImportReport:
    """Copy every row from the Postgres at ``dsn`` into the SQLite ``engine``.

    The target must already have its schema — run ``migrate`` first. Returns a
    report rather than printing, so the CLI owns presentation and a test can
    assert on the numbers.
    """
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "asyncpg is not installed — this is a one-off tool, so it is an "
            'optional extra: uv sync --extra pgimport'
        ) from exc

    report = ImportReport()
    source = await asyncpg.connect(dsn)
    try:
        for table in TABLES:
            columns = COLUMNS[table]
            rows = await source.fetch(
                f"SELECT {', '.join(columns)} FROM {table}"  # noqa: S608 - fixed names
            )
            report.source_rows[table] = len(rows)
            if not rows:
                report.imported[table] = 0
                report.skipped[table] = 0
                continue

            values = [tuple(_convert(row[c]) for c in columns) for row in rows]

            if dry_run:
                report.imported[table] = 0
                report.skipped[table] = 0
                continue

            placeholders = ", ".join("?" for _ in columns)
            statement = (
                f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "  # noqa: S608
                f"VALUES ({placeholders})"
            )

            async with engine.begin() as conn:
                before = await _count(conn, table)
                raw = await conn.get_raw_connection()
                await raw.driver_connection.executemany(statement, values)
                after = await _count(conn, table)

            inserted = after - before
            report.imported[table] = inserted
            report.skipped[table] = len(rows) - inserted
            log.info(
                "imported table",
                extra={"table": table, "inserted": inserted, "read": len(rows)},
            )
    finally:
        await source.close()

    return report


async def _count(conn, table: str) -> int:  # noqa: ANN001
    return (
        await conn.execute(text(f"SELECT count(*) FROM {table}"))  # noqa: S608
    ).scalar_one()
