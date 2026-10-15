"""The ORM models and the .sql migration must not drift apart.

Keeping declarative models *and* hand-written migrations means two descriptions
of one schema.  That is a deliberate trade — the migration stays reviewable and
the partial unique index stays explicit — but it introduces the one risk the
raw-SQL version did not have: a model that no longer matches the table it maps
to.

So the drift is asserted rather than hoped for.  This is the test that earns the
two-sources-of-truth decision.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect

from transcriptor.adapters import orm

pytestmark = pytest.mark.db


async def _reflect(db, what):
    async with db.engine.connect() as conn:
        return await conn.run_sync(what)


async def test_every_model_has_a_migrated_table(db) -> None:
    names = await _reflect(db, lambda c: inspect(c).get_table_names())
    for table in orm.Base.metadata.tables:
        assert table in names, f"{table} is modelled but not migrated"


async def test_columns_match_the_migrated_schema(db) -> None:
    """Same column names, same nullability, in both directions."""
    for name, table in orm.Base.metadata.tables.items():
        actual = await _reflect(db, lambda c, n=name: inspect(c).get_columns(n))
        by_name = {c["name"]: c for c in actual}

        modelled = {c.name for c in table.columns}
        assert modelled == set(by_name), (
            f"{name}: model has {modelled - set(by_name)}, "
            f"database has {set(by_name) - modelled}"
        )

        for column in table.columns:
            # Generated columns report as nullable; skip those.
            if column.computed is not None:
                continue
            assert column.nullable == by_name[column.name]["nullable"], (
                f"{name}.{column.name}: model nullable={column.nullable}, "
                f"database nullable={by_name[column.name]['nullable']}"
            )


async def test_the_partial_unique_index_exists_as_modelled(db) -> None:
    """The index that makes invariant I1 structural, not aspirational."""
    indexes = await _reflect(db, lambda c: inspect(c).get_indexes("jobs"))
    active = next(i for i in indexes if i["name"] == "jobs_one_active_idx")
    # SQLite reflects this as 1 rather than True.
    assert bool(active["unique"]) is True
    assert active["column_names"] == ["result_key"]
    # Partial: it constrains only the active statuses. SQLite supports these,
    # so I1 keeps exactly the guarantee it had on Postgres.
    predicate = str(active.get("dialect_options", {}).get("sqlite_where", ""))
    for status in ("queued", "running", "paused"):
        assert status in predicate


async def test_segments_carry_the_composite_foreign_key(db) -> None:
    """Invariant I3 lives on this constraint."""
    fks = await _reflect(db, lambda c: inspect(c).get_foreign_keys("segments"))
    composite = [f for f in fks if f["referred_table"] == "chunks"]
    assert composite, "segments must reference chunks"
    assert set(composite[0]["constrained_columns"]) == {"job_id", "chunk_idx"}


async def test_the_search_index_does_not_strip_diacritics(db) -> None:
    """No stemming and no diacritic stripping.

    The first because there is no Hindi stemmer worth the name and 99 languages
    means no-stemming is the only consistent choice — the same reasoning that
    picked Postgres's 'simple' configuration. The second because stripping
    diacritics mangles Hindi, Arabic and Vietnamese rather than helping them,
    and FTS5's default *does* strip them. Getting this wrong would degrade
    search for most of the languages this supports, silently.
    """
    from sqlalchemy import text as sa_text

    async with db.engine.connect() as conn:
        sql = (
            await conn.execute(
                sa_text(
                    "SELECT sql FROM sqlite_master WHERE name = 'segments_fts'"
                )
            )
        ).scalar_one()
    assert "remove_diacritics 0" in sql


async def test_the_fuzzy_index_is_trigram(db) -> None:
    """Standing in for pg_trgm: this is what finds a half-remembered phrase."""
    from sqlalchemy import text as sa_text

    async with db.engine.connect() as conn:
        sql = (
            await conn.execute(
                sa_text("SELECT sql FROM sqlite_master WHERE name = 'segments_trgm'")
            )
        ).scalar_one()
    assert "trigram" in sql
