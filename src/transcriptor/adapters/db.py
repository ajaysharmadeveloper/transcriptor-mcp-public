"""Engine, session management, and the migration runner.

The database is a file inside the project. There is no server to install, no
password to protect, no DSN to hand around, and no container to start before
the first transcript — which is most of what stood between someone finding this
and running it.

Session policy — the whole of it, in one place so no call site has to improvise:

* **One session per unit of work.**  ``Database.begin()`` opens a session *and*
  a transaction, commits on clean exit and rolls back on any exception.  Every
  write goes through it.  ``Database.session()`` is the read-only counterpart:
  no explicit transaction, nothing to commit.
* **Sessions are never shared across tasks.**  ``async_sessionmaker`` hands out
  a fresh one each time; none is stored on the store objects, so two concurrent
  callers can never interleave on one session's connection.
* **``expire_on_commit=False``.**  Rows are mapped to domain dataclasses inside
  the session and returned as plain data, so nothing is lazy-loaded after
  commit — and ``lazy="raise"`` on the relationships turns any accidental
  attempt into an error rather than a silent extra query.
* **Migrations stay raw ``.sql``.**  The schema is the design here — the partial
  unique index that enforces I1, the FTS5 tables and the triggers that maintain
  them — so it is written out explicitly rather than generated.

## The four pragmas, and why each is not optional

SQLite's defaults are chosen for a twenty-year-old embedded device, not for
this. Every connection gets all four:

``foreign_keys = ON``
    **The most expensive default in the engine.** SQLite parses foreign keys
    without it and silently enforces nothing. Invariant I3 — every segment
    belongs to a chunk row — is a composite foreign key, so without this it is
    decoration. Off by default for backwards compatibility, and it has to be
    set per connection: it cannot live in the schema.

``journal_mode = WAL``
    Readers stop blocking the writer and the writer stops blocking readers,
    which is precisely this application's shape: one JobRunner writing, tool
    handlers reading throughout. In the default rollback journal a running job
    would freeze every ``transcript_status`` call, breaking invariant I4.
    WAL is a property of the database file, so it persists once set.

``busy_timeout = 10000``
    Claude Desktop starts **two instances of this server** (§18a), so two
    processes really do write to one file. SQLite's default is to fail
    instantly with SQLITE_BUSY rather than wait. Ten seconds of waiting turns
    a contended write into a slow one instead of a failed job.

``synchronous = NORMAL``
    Full fsync on every commit costs more than it buys here: under WAL, NORMAL
    risks losing the last transactions on **power loss**, not on a crash of
    this process. A lost chunk commit is re-transcribed on resume, which the
    pipeline already does by design.

## Writes BEGIN IMMEDIATE, reads do not

This is the subtlest thing in the file and it was found by a test rather than
by reasoning, so it is written down properly.

SQLite's default ``BEGIN`` is *deferred*: it takes no lock until the first
statement, and a read does not take a write lock. So two tasks can both SELECT
the oldest queued job, both decide they own it, and only then serialise on the
UPDATE — handing the same job out twice. That is precisely the failure
``SELECT ... FOR UPDATE SKIP LOCKED`` prevented on Postgres, and dropping the
lock without replacing it reintroduced it. ``test_concurrent_runners_never_
claim_the_same_job`` caught it: four claims, two distinct jobs.

``BEGIN IMMEDIATE`` takes the write lock up front, so a second writer waits out
its ``busy_timeout`` and then reads the row as already ``running``.

It is applied to **write** transactions only. Making every transaction
immediate would serialise the tool handlers behind a running job's chunk
commits and break invariant I4 — the thing WAL was chosen to avoid. So
``Database.begin()`` runs on an engine carrying the ``IMMEDIATE`` execution
option and ``Database.session()`` does not.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from transcriptor.config import Settings
from transcriptor.domain.errors import DatabaseUnreachable, MigrationsFailed

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# Numbered, e.g. 001_initial.sql.  Applied in filename order.
_MIGRATION_NAME = re.compile(r"^(\d+)_[a-z0-9_]+\.sql$")

_BUSY_TIMEOUT_MS = 10_000


def async_dsn(url: str) -> str:
    """Point the URL at the aiosqlite driver without touching the stored value."""
    if url.startswith("sqlite+"):
        return url
    return url.replace("sqlite://", "sqlite+aiosqlite://", 1)


def database_path(settings: Settings) -> Path:
    """Where the database file lives.

    Inside the data directory, which is inside the repository (N10), alongside
    the model weights and the scratch audio. One directory holds everything
    this project puts on disk, so "where is my data" and "how do I back it up"
    have the same one-line answer.
    """
    return settings.data_dir / "transcriptor.db"


def url_for(settings: Settings) -> str:
    """The SQLAlchemy URL, from the setting or from the default location."""
    if settings.database_url:
        return async_dsn(settings.database_url)
    return f"sqlite+aiosqlite:///{database_path(settings)}"


# --------------------------------------------------------------------- database


class Database:
    """Owns the engine and hands out sessions.  The single unit-of-work entry
    point for every store."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        # Same pool, different transaction mode. `execution_options` returns a
        # new engine sharing the underlying connection pool, so this costs no
        # extra connections — see the module docstring for why writes must
        # begin immediately and reads must not.
        self._write_engine = engine.execution_options(sqlite_txn="IMMEDIATE")
        self._sessionmaker = async_sessionmaker(
            engine,
            expire_on_commit=False,
            autoflush=False,
        )
        self._write_sessionmaker = async_sessionmaker(
            self._write_engine,
            expire_on_commit=False,
            autoflush=False,
        )

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def sessionmaker(self) -> async_sessionmaker[AsyncSession]:
        return self._sessionmaker

    @asynccontextmanager
    async def begin(self) -> AsyncIterator[AsyncSession]:
        """A write unit of work: commits on success, rolls back on any error.

        Nesting is not supported and not needed — a store method is one unit of
        work, and anything spanning two of them wants an explicit session
        passed in instead.

        Begins ``IMMEDIATE``, which is what stops two writers reading the same
        row and both deciding they own it.
        """
        async with self._write_sessionmaker() as session:
            async with session.begin():
                yield session

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A read-only session.  Nothing to commit, closed on exit.

        Deliberately *not* ``IMMEDIATE``: under WAL a reader never blocks the
        writer, which is what keeps tool handlers answering in under 300ms
        while a job is running (invariant I4).
        """
        async with self._sessionmaker() as session:
            yield session

    async def dispose(self) -> None:
        await self._engine.dispose()


async def connect(settings: Settings) -> Database:
    """Open the database, creating the file and its directory if needed.

    There is nothing to wait for any more. This used to retry for two minutes
    because the MCP client launches Transcriptor at login, likely before Docker
    had finished starting Postgres — a whole class of startup failure that
    disappears when the database is a file next to the code.
    """
    engine = create_engine_for(settings)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except (OSError, SQLAlchemyError) as exc:
        await engine.dispose()
        target = settings.database_url or str(database_path(settings))
        raise DatabaseUnreachable(
            f"database_unreachable: cannot open {target} — {exc}",
            dsn=target,
        ) from exc
    return Database(engine)


def create_engine_for(settings: Settings) -> AsyncEngine:
    """Build the async engine, with the pragmas applied to every connection.

    The pragma hook is registered on the *sync* engine underneath, because that
    is where SQLAlchemy emits connection events. Doing it per connection rather
    than once is not belt-and-braces: a pooled connection that was recycled
    would otherwise come back with foreign keys silently off.
    """
    path = database_path(settings)
    if not settings.database_url:
        path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(url_for(settings), pool_pre_ping=True)

    @event.listens_for(engine.sync_engine, "connect")
    def _pragmas(dbapi_connection, _record) -> None:  # noqa: ANN001
        # Stop the driver emitting its own BEGIN, so the `begin` handler below
        # can choose the mode. Without this the driver has already opened a
        # deferred transaction by the time anything else runs.
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            cursor.execute("PRAGMA synchronous = NORMAL")
        finally:
            cursor.close()

    @event.listens_for(engine.sync_engine, "begin")
    def _begin(conn) -> None:  # noqa: ANN001
        mode = conn.get_execution_options().get("sqlite_txn", "DEFERRED")
        conn.exec_driver_sql(f"BEGIN {mode}")

    return engine


# --------------------------------------------------------------- migrations


@dataclass(frozen=True, slots=True)
class Migration:
    version: str
    path: Path
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.blake2b(self.sql.encode(), digest_size=16).hexdigest()


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Every numbered .sql file, in filename order.

    A file that does not match the naming pattern is an error, not something to
    skip quietly — a migration silently ignored is the worst possible outcome.
    """
    found: list[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        if not _MIGRATION_NAME.match(path.name):
            raise MigrationsFailed(
                f"migration filename is not numbered as NNN_name.sql: {path.name}",
                file=path.name,
            )
        found.append(Migration(version=path.name, path=path, sql=path.read_text()))
    return found


_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    checksum   TEXT NOT NULL,
    applied_at TEXT NOT NULL
)
"""


def _driver_error(exc: BaseException) -> BaseException:
    """The database's own error, whether it arrived wrapped by SQLAlchemy or
    raised straight from sqlite3."""
    return getattr(exc, "orig", None) or exc


def split_statements(sql: str) -> list[str]:
    """Split a .sql file into individual statements.

    Not ``sql.split(";")``. The schema contains FTS triggers whose bodies are
    ``BEGIN ... END`` blocks full of semicolons, and splitting naively cuts
    them in half — producing errors that point at the wrong line of a statement
    that never existed.

    ``sqlite3.complete_statement`` is SQLite's own answer to "is this a whole
    statement yet", and it knows about trigger bodies. Comments accumulate onto
    the statement that follows them, which is what you want when the failure is
    reported back to a human.
    """
    statements: list[str] = []
    buffer: list[str] = []
    for line in sql.splitlines(keepends=True):
        buffer.append(line)
        candidate = "".join(buffer)
        if candidate.strip() and sqlite3.complete_statement(candidate):
            statements.append(candidate.strip())
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)
    return statements


async def _execute_script(conn: AsyncConnection, sql: str) -> str | None:
    """Run a migration file, one statement at a time, inside the caller's
    transaction.  Returns the statement that failed, or ``None``.

    Statement at a time rather than ``executescript``, for two reasons that
    both matter. ``executescript`` COMMITs whatever is open before it runs, so
    a file that failed half way would leave the first half applied — the whole
    point of running a migration in a transaction is that it does not. And it
    reports errors without saying which statement raised them, which is the
    difference between a fixable message and a puzzle.

    SQLite's DDL is transactional, so a failure here really does roll the whole
    file back.
    """
    for statement in split_statements(sql):
        try:
            await conn.exec_driver_sql(statement)
        except Exception:
            return statement
    return None


async def apply_migrations(conn: AsyncConnection, **_ignored) -> list[str]:
    """Apply pending migrations.  Returns the versions applied, newest last.

    **The whole thing runs in one ``IMMEDIATE`` transaction**, and that is not
    tidiness. Postgres held ``pg_advisory_lock`` across this; when it was
    dropped, the "which migrations are already applied" read and the apply that
    follows it landed in *different* transactions, so two runners both read an
    empty set and both tried to create the same tables. The second failed with
    ``table sources already exists``.

    Claude Desktop starts two instances of this server (§18a), so that is the
    normal case rather than an edge one. It surfaced on a CI runner within
    minutes of the claim that no lock was needed.

    Taking the write lock up front restores exactly what the advisory lock gave:
    one runner does the work, the other waits out its ``busy_timeout`` and then
    reads the migrations as already recorded.

    The cost is that the batch is now all-or-nothing rather than one
    transaction per file. On SQLite that is an improvement — DDL is
    transactional here, so a failure half way leaves nothing behind instead of
    a schema that is half of two versions.

    On failure the server refuses to serve rather than run on a half-built
    schema.

    ``**_ignored`` swallows the ``schema=`` argument the Postgres version took.
    SQLite has no schemas.
    """
    migrations = discover_migrations()

    # Take the write lock before reading anything, so the read and the writes
    # that depend on it cannot be separated by another writer.
    #
    # Set as an execution option rather than by executing "BEGIN IMMEDIATE":
    # SQLAlchemy opens the transaction itself on the first statement, and
    # issuing our own BEGIN on top of that is an error. The `begin` handler in
    # create_engine_for reads this option to decide the mode.
    await conn.execution_options(sqlite_txn="IMMEDIATE")
    try:
        await conn.execute(text(_SCHEMA_MIGRATIONS_DDL))

        rows = (
            await conn.execute(text("SELECT version, checksum FROM schema_migrations"))
        ).all()
        applied = {r.version: r.checksum for r in rows}

        # An already-applied migration whose content changed means the schema on
        # disk and the schema in the database have silently diverged.
        for m in migrations:
            if m.version in applied and applied[m.version] != m.checksum:
                raise MigrationsFailed(
                    f"{m.version} has changed since it was applied — "
                    "migrations are immutable once applied",
                    file=m.version,
                )

        newly: list[str] = []
        for m in migrations:
            if m.version in applied:
                continue
            failing = await _execute_script(conn, m.sql)
            if failing is not None:
                raise MigrationsFailed(
                    f"migrations_failed in {m.version}",
                    file=m.version,
                    statement=failing,
                )
            await conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, checksum, applied_at) "
                    "VALUES (:v, :c, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))"
                ),
                {"v": m.version, "c": m.checksum},
            )
            newly.append(m.version)
            log.info("migration applied", extra={"version": m.version})

        await conn.commit()
        return newly
    except MigrationsFailed:
        await conn.rollback()
        raise
    except (DBAPIError, sqlite3.Error) as exc:
        await conn.rollback()
        raise MigrationsFailed(
            f"migrations_failed: {_driver_error(exc)}",
        ) from exc


async def applied_versions(conn: AsyncConnection, **_ignored) -> list[str]:
    exists = (
        await conn.execute(
            text(
                "SELECT count(*) FROM sqlite_master "
                "WHERE type = 'table' AND name = 'schema_migrations'"
            )
        )
    ).scalar()
    if not exists:
        return []
    rows = (
        await conn.execute(
            text("SELECT version FROM schema_migrations ORDER BY version")
        )
    ).all()
    return [r.version for r in rows]
