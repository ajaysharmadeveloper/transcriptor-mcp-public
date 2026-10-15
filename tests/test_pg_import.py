"""The one-off Postgres importer.

No Postgres is involved. The parts that can be wrong without a server are the
type conversions and the column lists, and those are exactly the parts that
fail *silently* — a UUID written in the wrong form still looks like a UUID and
joins to nothing.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid

import pytest

from transcriptor.adapters import orm
from transcriptor.adapters.pg_import import COLUMNS, TABLES, _convert


# ------------------------------------------------------------- conversions


def test_uuids_are_written_without_dashes() -> None:
    """The silent one.

    SQLAlchemy's Uuid type stores 32 hex characters on SQLite. Import them
    with dashes and every foreign key still looks like a UUID while joining to
    nothing at all — no error, just an empty transcript.
    """
    value = uuid.UUID("f5589ea5-ee09-43e3-8450-87fe6e8293d2")
    assert _convert(value) == "f5589ea5ee0943e3845087fe6e8293d2"
    assert "-" not in _convert(value)


def test_uuids_match_what_the_orm_writes() -> None:
    """Asserted against the ORM rather than against a hardcoded shape, so this
    keeps holding if the storage form ever changes."""
    from sqlalchemy.dialects import sqlite

    value = uuid.uuid4()
    bound = orm.Source.__table__.c.id.type.bind_processor(sqlite.dialect())
    expected = bound(value) if bound else value
    assert _convert(value) == expected


def test_aware_timestamps_keep_their_offset() -> None:
    value = dt.datetime(2026, 9, 9, 15, 53, 9, tzinfo=dt.UTC)
    assert _convert(value) == "2026-09-09T15:53:09+00:00"


def test_naive_timestamps_are_assumed_utc_rather_than_rejected() -> None:
    """A naive value stored as-is would make the reaper's heartbeat comparison
    raise later, a long way from the cause."""
    naive = dt.datetime(2026, 9, 9, 15, 53, 9)
    assert _convert(naive).endswith("+00:00")


def test_non_utc_timestamps_are_converted_not_relabelled() -> None:
    """The bug this guards against is stamping +00:00 onto a local time."""
    plus_two = dt.timezone(dt.timedelta(hours=2))
    value = dt.datetime(2026, 9, 9, 17, 53, 9, tzinfo=plus_two)
    assert _convert(value) == "2026-09-09T15:53:09+00:00"


def test_timestamps_round_trip_through_the_orm_type() -> None:
    """The real contract: what the importer writes, UtcDateTime must read."""
    value = dt.datetime(2026, 9, 9, 15, 53, 9, 255671, tzinfo=dt.UTC)
    written = _convert(value)
    read_back = orm.UtcDateTime().process_result_value(written, None)
    assert read_back == value


@pytest.mark.parametrize("value", [{"a": 1}, [1, 2, 3], []])
def test_json_values_are_serialised(value) -> None:
    assert json.loads(_convert(value)) == value


def test_strings_numbers_and_none_pass_through() -> None:
    for value in ("text", 42, 3.5, None, True):
        assert _convert(value) is value


# ----------------------------------------------------------- column lists


def test_every_imported_column_exists_in_the_target_schema() -> None:
    """The drift guard.

    Column lists are written out rather than `SELECT *`, so a column that
    exists in one schema and not the other has to be noticed here. Without
    this test the importer would fail at run time, against a database, for
    somebody performing a migration they only get to do once.
    """
    for table, columns in COLUMNS.items():
        modelled = {c.name for c in orm.Base.metadata.tables[table].columns}
        unknown = set(columns) - modelled
        assert not unknown, f"{table}: importing {unknown}, which the schema lacks"


def test_no_column_is_silently_left_behind() -> None:
    """The other direction: a column in the schema that the importer skips
    would import as NULL and nobody would notice until they went looking."""
    for table, columns in COLUMNS.items():
        modelled = {c.name for c in orm.Base.metadata.tables[table].columns}
        missing = modelled - set(columns)
        assert not missing, f"{table}: schema has {missing}, which is not imported"


def test_every_table_is_covered() -> None:
    assert set(TABLES) == set(COLUMNS)
    assert set(TABLES) == set(orm.Base.metadata.tables)


def test_tables_are_ordered_parents_first() -> None:
    """Foreign keys are enforced, so a child inserted before its parent is
    rejected outright. SQLAlchemy already knows the dependency order; this
    asserts the importer agrees with it."""
    correct = [t.name for t in orm.Base.metadata.sorted_tables]
    positions = {name: i for i, name in enumerate(TABLES)}
    for name in correct:
        for fk in orm.Base.metadata.tables[name].foreign_keys:
            parent = fk.column.table.name
            if parent != name:
                assert positions[parent] <= positions[name], (
                    f"{name} is imported before {parent}, which it references"
                )


def test_the_dropped_columns_are_not_imported() -> None:
    """They were audited as dead before the port and do not exist in the new
    schema. Importing them would fail against the target."""
    for dead in ("first_seen_at", "last_seen_at", "request_count"):
        assert dead not in COLUMNS["sources"]


def test_the_generated_search_column_is_not_imported() -> None:
    """`tsv` was a Postgres generated column. Its replacement is an FTS5 index
    that the triggers rebuild as rows arrive, so importing it would be both
    wrong and impossible."""
    assert "tsv" not in COLUMNS["segments"]
