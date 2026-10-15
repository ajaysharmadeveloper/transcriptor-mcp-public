"""The migration runner: idempotent, immutable, and honest about failure."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import text

from transcriptor.adapters.db import (
    apply_migrations,
    applied_versions,
    create_engine_for,
    discover_migrations,
    split_statements,
)
from transcriptor.config import Settings
from transcriptor.domain.errors import MigrationsFailed

pytestmark = pytest.mark.db


@asynccontextmanager
async def connection(tmp_path) -> AsyncIterator:
    """One connection to a database file of this test's own."""
    engine = create_engine_for(Settings(_env_file=None, data_dir=tmp_path))
    try:
        async with engine.connect() as conn:
            yield conn
    finally:
        await engine.dispose()


async def test_migrations_are_discovered_in_order() -> None:
    versions = [m.version for m in discover_migrations()]
    assert versions == sorted(versions)
    assert versions, "there should be at least one migration"


async def test_apply_is_idempotent(tmp_path) -> None:
    async with connection(tmp_path) as conn:
        first = await apply_migrations(conn)
        second = await apply_migrations(conn)
        assert first, "the first run applies everything"
        assert second == [], "the second run applies nothing"
        assert await applied_versions(conn) == first


async def test_the_schema_has_every_table(tmp_path) -> None:
    async with connection(tmp_path) as conn:
        await apply_migrations(conn)
        rows = (
            await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        ).all()
        tables = {r.name for r in rows}

    for expected in (
        "sources",
        "requests",
        "jobs",
        "job_events",
        "chunks",
        "renders",
        "segments",
        "schema_migrations",
    ):
        assert expected in tables, f"{expected} is missing from the schema"


async def test_the_search_indexes_are_built(tmp_path) -> None:
    """Both of them. One finds whole words, the other finds substrings, and a
    schema carrying only the first would degrade search silently."""
    async with connection(tmp_path) as conn:
        await apply_migrations(conn)
        rows = (
            await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'table'")
            )
        ).all()
        tables = {r.name for r in rows}
    assert "segments_fts" in tables
    assert "segments_trgm" in tables


async def test_the_triggers_that_maintain_them_exist(tmp_path) -> None:
    """An external-content FTS5 table does not update itself.

    Without these the index is correct exactly once — at the moment of the
    first insert — and then drifts, which shows up as search quietly missing
    recent transcripts rather than as an error.
    """
    async with connection(tmp_path) as conn:
        await apply_migrations(conn)
        rows = (
            await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            )
        ).all()
    assert {r.name for r in rows} == {"segments_ai", "segments_ad", "segments_au"}


async def test_foreign_keys_are_enforced_on_every_connection(tmp_path) -> None:
    """SQLite parses foreign keys and ignores them unless asked not to.

    Invariant I3 — every segment belongs to a chunk row — *is* a composite
    foreign key, so this pragma is the difference between an enforced
    invariant and a comment.
    """
    async with connection(tmp_path) as conn:
        enabled = (await conn.execute(text("PRAGMA foreign_keys"))).scalar()
    assert enabled == 1


async def test_the_database_is_in_wal_mode(tmp_path) -> None:
    """Readers must not block the writer: tool handlers answer throughout a
    running job, which is invariant I4."""
    async with connection(tmp_path) as conn:
        mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar()
    assert mode.lower() == "wal"


async def test_editing_an_applied_migration_is_rejected(tmp_path) -> None:
    """Divergence between the file on disk and the schema in the database is
    caught, rather than quietly running on a schema nobody can reproduce."""
    async with connection(tmp_path) as conn:
        await apply_migrations(conn)
        await conn.execute(text("UPDATE schema_migrations SET checksum = :c"), {"c": "x"})
        await conn.commit()
        with pytest.raises(MigrationsFailed):
            await apply_migrations(conn)


async def test_two_runners_do_not_both_apply_a_migration(tmp_path) -> None:
    """Claude Desktop starts two instances of this server, so this is the
    normal case rather than an edge one.

    Postgres needed an advisory lock for it. SQLite serialises writers itself:
    the loser waits out its busy_timeout and then finds the migrations already
    recorded.
    """
    async with connection(tmp_path) as a, connection(tmp_path) as b:
        results = await asyncio.gather(apply_migrations(a), apply_migrations(b))
        applied = [v for r in results for v in r]
        assert sorted(applied) == sorted(set(applied)), "a migration ran twice"
        assert len(applied) == len(discover_migrations())


async def test_migration_failure_names_the_file_and_statement(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken migration must say which file and which statement, and must not
    be recorded as applied."""
    from transcriptor.adapters import db as db_mod

    broken = db_mod.Migration(
        version="999_broken.sql",
        path=db_mod.MIGRATIONS_DIR / "999_broken.sql",
        sql="CREATE TABLE ok_one (id int);\nCREATE TABLE bad (id int PRIMARY KEY AUTOINCREMENT);",
    )
    monkeypatch.setattr(db_mod, "discover_migrations", lambda *a, **k: [broken])

    async with connection(tmp_path) as conn:
        with pytest.raises(MigrationsFailed) as excinfo:
            await apply_migrations(conn)
        assert excinfo.value.context["file"] == "999_broken.sql"
        assert "AUTOINCREMENT" in excinfo.value.context["statement"]

        # The whole file rolled back: neither table exists, nothing recorded.
        rows = (
            await conn.execute(
                text("SELECT name FROM sqlite_master WHERE name = 'ok_one'")
            )
        ).all()
        assert rows == [], "a failed migration left a table behind"
        assert await applied_versions(conn) == []


def test_trigger_bodies_survive_statement_splitting() -> None:
    """The reason splitting on ';' is not good enough.

    A trigger body is a BEGIN ... END block full of semicolons. Cut it up and
    the migration fails with an error pointing at a statement that was never
    written.
    """
    statements = split_statements(
        "CREATE TABLE t (a);\n"
        "CREATE TRIGGER tr AFTER INSERT ON t BEGIN\n"
        "  INSERT INTO t VALUES (1);\n"
        "  INSERT INTO t VALUES (2);\n"
        "END;\n"
    )
    assert len(statements) == 2, "the trigger was split into pieces"
    trigger = statements[1]
    assert trigger.count("INSERT INTO t VALUES") == 2, "the body lost a statement"
    assert trigger.rstrip().endswith("END;"), "the body was cut before its END"
