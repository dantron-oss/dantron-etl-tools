"""Tests for the row loader under every sink: the same load, on both drivers, or neither is trusted.

A new file rather than more of ``tests/tools/test_write_table.py``. That suite's own docstring ends
"everything here runs against SQLite ... so the suite needs no server and no credentials", and that
promise is worth keeping: it is what lets a checkout of this library on its own prove the tool's
rulings with nothing installed. What is measured here is the other half -- the load that reaches a
real server -- and it belongs beside ``db.py``, which is where the connection plumbing every sink
shares already lives.

The defect this file exists for was found by probing the sink against a real Postgres: not one row
could be written. The load had been written against ``sqlite3``'s connection-level conveniences and
nothing else, so on Postgres it reached for ``connection.executemany``, which that driver keeps on
the cursor, and fell over with an ``AttributeError`` before a byte moved. Every Postgres test in the
sink's own suite was a refusal -- a bad identifier, a missing driver, a secret kept out of a message
-- and a refusal never gets as far as the load. That is how it shipped, and it is why the tests
below write rows and read them back rather than checking that a bad name is refused.

Four properties carry the file.

**The values come back the way they went in.** A null, a float, a date and a piece of text holding a
comma, a quote, a newline, a tab and a backslash. The hostile text is not decoration: Postgres's bulk
load reads a tab-delimited, backslash-escaped stream, so a load that builds that stream by hand and
forgets an escape silently splits one value into two columns, or turns the two characters ``\\N``
into a null. Round-tripping is the only check that catches it.

**A replace is all of the rows or none of them, on both drivers.** The sink's central promise, and
the property most likely to differ between a ``with connection:`` block on ``sqlite3`` and the same
block on ``psycopg``. Its failure mode is a build that deletes yesterday's figures, falls over part
way, and leaves the table empty -- so it is checked on a connection other than the one that wrote,
which is also how "the transaction was actually closed" gets proven.

**A refusal is this library's error, carrying the same answer on either driver.** A caller asking
"which rows offended, out of how many?" must not get a number from SQLite and a bare
``psycopg.errors.UniqueViolation`` from Postgres.

**The load does not cost a statement per row.** A publish in the consuming project is of the order of
a hundred thousand rows, and a row-by-row insert is not a slower way to do that, it is a different
build. Measured on the server rather than asserted about the source: a statement-level trigger fires
once per statement, so a bulk load fires it once and a row-by-row loop fires it once per row.

**Naming the offenders costs no attempt per row, and still names exactly the sequential ones.** The
same publish that made the load bulk makes the *diagnosis* of a failed load the next thing that does
not finish: a hundred and thirty thousand rows replayed one at a time, each one a savepoint, an
insert and a release, is a minute of round trips at a sub-millisecond latency and tens of minutes at
a realistic one -- spent while a transaction is held open, so a build that hit one bad row looks
wedged rather than failed. What the diagnosis costs is measured the same way the bulk load is, on
the server. What it *answers* must not move an inch: the offenders are the rows a sequential load
would have refused, second-and-later occurrences of a key rather than both rows holding it, the
table's existing keys counted in an append, and every failure kind in the frame found rather than
the first. And a constraint the rows cannot be judged against away from the server -- a check
expression, a key that has to exist in another table -- must still report its offenders by whatever
route, because "no rows offended" over a load that was just refused is the one answer worse than a
slow one.

The SQLite half runs anywhere. The Postgres half skips loudly when there is no server, and the
schema it creates is one nothing else names, dropped with everything in it on the way out.
"""

import logging
import os
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl.db import driver_for, open_connection
from dantron_etl.authoring import DataError, ExecutionError, MAX_REPORTED_ROWS, Ports, ToolError, run_tool
from dantron_etl.tools import write_table as write_table_module
from dantron_etl.tools.write_table import WriteTable

# The name of the variable a pipeline definition would carry, holding the connection for this run.
_VARIABLE = "ETL_TEST_LOAD_CONNECTION"

# The tests own this schema and nothing else. It is created, used and dropped by the fixture below.
# The server this runs against also holds real data in schemas of its own, and no test here names
# any schema but this one.
_SCRATCH_SCHEMA = "dantron_etl_db_scratch"

# Every character a tab-delimited, backslash-escaped bulk stream can be broken by, in one value.
# The trailing '\N' is the sequence that stream spells a null with: a load that does not escape it
# writes a null where a two-character string belongs, which no row count would ever show.
_HOSTILE_TEXT = 'a, comma "quoted", a\ttab, a\nnewline, a\\backslash and a \\N that is not a null'

# Enough rows that a statement per row is unmistakable, and few enough that no plausible bulk load
# would split them into chunks.
_BULK_HEIGHT = 500

# A frame with one bad row in it, at the size a publish meets and at four times that size. Two sizes
# rather than one because the question is not "is it fast" but "does what it costs follow the rows".
_DIAGNOSED_HEIGHT = 512
_QUADRUPLED_HEIGHT = 2048

# The mirror: a frame where all but the first row offends. Whatever names the offenders must not
# start costing with the number of bad rows once it has stopped costing with the number of rows.
_DENSE_HEIGHT = 200

# What naming the offenders may cost in statements against the table, at any of those sizes. One of
# them is the bulk load that already failed; the rest is headroom for an implementation that asks
# the table something on the way to its answer.
_ALLOWED_ATTEMPTS = 4


_SQLITE_DDL: tuple[str, ...] = (
    # 'seen' is declared a date, as on Postgres, so a Date column fits it: the table writers refuse
    # a column whose type does not fit the table's. SQLite still stores the date as ISO text.
    "create table target (key integer primary key, label text, amount real, seen date)",
    "create table strict (key integer primary key, amount real not null)",
    "create table guarded (key integer primary key, amount real check (amount >= 0))",
    "create table paired (key integer primary key, code text unique, amount real)",
)

_POSTGRES_DDL: tuple[str, ...] = (
    "create table {schema}.target (key integer primary key, label text, amount double precision, seen date)",
    "create table {schema}.strict (key integer primary key, amount double precision not null)",
    "create table {schema}.guarded (key integer primary key, amount double precision check (amount >= 0))",
    "create table {schema}.paired (key integer primary key, code text unique, amount double precision)",
)

# What both targets hold before a test writes. Dates are spelled per driver by the fixture: SQLite
# has no date type and holds the ISO text, Postgres holds a real date.
_BEFORE_TARGET: tuple[tuple[Any, ...], ...] = (
    (1, "first", 1.5, "2026-01-01"),
    (2, "second", 2.5, "2026-02-01"),
)
_BEFORE_STRICT: tuple[tuple[Any, ...], ...] = ((1, 1.5), (2, 2.5))


def _plain(value: Any) -> Any:
    """One spelling per value, so a row read from either driver compares against one expectation.

    The only difference between the two reads is the date: Postgres hands back a :class:`date`, and
    SQLite hands back the ISO text its adapter wrote. Both mean the same day, so both are compared
    as that day's ISO spelling.
    """
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _normalised(rows: Any) -> list[tuple[Any, ...]]:
    """Rows in key order, each value in its one comparable spelling."""
    return sorted((tuple(_plain(value) for value in row) for row in rows), key=lambda row: row[0])


class _Target:
    """One database holding the tables a test writes, and the name a definition would reach it by.

    ``read`` deliberately does not reuse whatever connection the tool used: it asks the server, on a
    connection of its own, what is there now. That is what makes "the rows landed" mean committed
    rather than merely sent, and what makes "the previous figures are still standing" mean the
    failed transaction was actually rolled back rather than left open.
    """

    def __init__(self, driver: str, variable: str, schema: str | None, read: Any) -> None:
        self.driver = driver
        self.variable = variable
        self.schema = schema
        self._read = read

    def read(self, table: str) -> list[tuple[Any, ...]]:
        return _normalised(self._read(table))


@pytest.fixture(scope="session")
def postgres_session(postgres_scratch: Any) -> Any:
    """A Postgres connection owning this file's scratch schema (see ``postgres_scratch`` in conftest)."""
    return postgres_scratch(_SCRATCH_SCHEMA)


def _seed_sqlite(database: Path) -> None:
    """The four tables, with ``target`` and ``strict`` holding their previous figures."""
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("pragma synchronous = off")  # built once per session as a template
        for statement in _SQLITE_DDL:
            connection.execute(statement)
        connection.executemany("insert into target values (?, ?, ?, ?)", _BEFORE_TARGET)
        connection.executemany("insert into strict values (?, ?)", _BEFORE_STRICT)
        connection.commit()


@pytest.fixture(params=["sqlite", "postgres"])
def target(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Target:
    """The same two tables, seeded the same way, on whichever driver this run is about.

    SQLite is here because the catalog's worked examples must run with no server at all. Postgres is
    here because it is the engine a publish actually meets, and a sink proven only against the
    driver whose conveniences it was written for is a sink nobody has tested.
    """
    if request.param == "sqlite":
        database = request.getfixturevalue("seeded_sqlite")(tmp_path / "store.db", _seed_sqlite)
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{database}")

        def read(table: str) -> list[tuple[Any, ...]]:
            with closing(sqlite3.connect(database, timeout=1)) as connection:
                return list(connection.execute(f"select * from {table}").fetchall())

        return _Target("sqlite", _VARIABLE, None, read)

    connection = request.getfixturevalue("postgres_session")
    dsn = request.getfixturevalue("postgres_dsn")
    for table in ("target", "strict", "guarded", "paired"):
        connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.{table}")
    for statement in _POSTGRES_DDL:
        connection.execute(statement.format(schema=_SCRATCH_SCHEMA))
    cursor = connection.cursor()
    cursor.executemany(
        f"insert into {_SCRATCH_SCHEMA}.target values (%s, %s, %s, %s)",
        [(key, label, amount, date.fromisoformat(seen)) for key, label, amount, seen in _BEFORE_TARGET],
    )
    cursor.executemany(f"insert into {_SCRATCH_SCHEMA}.strict values (%s, %s)", list(_BEFORE_STRICT))
    monkeypatch.setenv(_VARIABLE, dsn)

    def read(table: str) -> list[tuple[Any, ...]]:
        return list(connection.execute(f"select * from {_SCRATCH_SCHEMA}.{table}").fetchall())

    return _Target("postgres", _VARIABLE, _SCRATCH_SCHEMA, read)


def _write(target: _Target, table: str = "target", **overrides: Any) -> Mapping[str, Any]:
    """A replace of ``table`` on ``target``, with whatever a case changes."""
    config: dict[str, Any] = {"connection": target.variable, "table": table, "mode": "replace", **overrides}
    if target.schema is not None:
        config["schema"] = target.schema
    return config


def _every_shape() -> pl.DataFrame:
    """One row of ordinary values, one of the values that break a bulk stream, one of nothing much."""
    return pl.DataFrame(
        {
            "key": [1, 2, 3],
            "label": ["plain", _HOSTILE_TEXT, None],
            "amount": [10.5, None, -0.25],
            "seen": [date(2026, 1, 1), date(2026, 6, 15), None],
        }
    )


_EVERY_SHAPE_STORED: list[tuple[Any, ...]] = [
    (1, "plain", 10.5, "2026-01-01"),
    (2, _HOSTILE_TEXT, None, "2026-06-15"),
    (3, None, -0.25, None),
]


class TestRowsReachBothDrivers:
    """The regression: a sink that could not write a single row to one of the two engines it claims."""

    def test_every_value_lands_and_reads_back_the_same_on_either_driver(self, target: _Target) -> None:
        """A null, a float, a date and text full of the characters a bulk stream is delimited by.

        The whole defect in one assertion: on Postgres this wrote nothing at all, because the load
        reached for a convenience only ``sqlite3`` has and bound its values with a placeholder only
        ``sqlite3`` understands. It also catches the next mistake after that one -- a bulk stream
        assembled by hand -- because the second row's label carries a tab, a newline and a backslash,
        and an unescaped one of those either splits the value across columns or is rejected outright,
        while the trailing ``\\N`` arrives as a null instead of two characters.

        The read is on a connection of its own, so "landed" means committed.
        """
        run_tool(WriteTable(), Ports({"rows": _every_shape()}), _write(target))

        assert target.read("target") == _EVERY_SHAPE_STORED

    def test_an_append_adds_rows_without_removing_the_ones_already_there(self, target: _Target) -> None:
        """The other mode, on both drivers: the prior rows stay and the new ones sit beside them.

        Worth its own case rather than a variation of the replace, because the two modes differ in
        exactly the statement a rewritten load is most likely to get wrong: a load that emptied the
        table regardless of mode would pass every replace test in the suite and lose the history
        here.
        """
        frame = pl.DataFrame(
            {"key": [3], "label": ["third"], "amount": [3.5], "seen": [date(2026, 3, 1)]},
        )

        run_tool(WriteTable(), Ports({"rows": frame}), _write(target, mode="append"))

        assert target.read("target") == [
            (1, "first", 1.5, "2026-01-01"),
            (2, "second", 2.5, "2026-02-01"),
            (3, "third", 3.5, "2026-03-01"),
        ]

    @pytest.mark.parametrize(
        ("mode", "after"),
        [
            pytest.param("replace", [], id="a_replace_of_nothing_empties_the_table"),
            pytest.param("append", list(_BEFORE_TARGET), id="an_append_of_nothing_changes_nothing"),
        ],
    )
    def test_a_frame_with_no_rows_writes_nothing_and_does_not_fail(
        self,
        mode: str,
        after: list[tuple[Any, ...]],
        target: _Target,
    ) -> None:
        """Zero rows is the load's own edge, not a case the caller is asked to keep away from it.

        ``allow_empty`` is the gate on whether an empty frame is *accepted*; once it is, the load has
        to cope with a frame of no rows without the sink special-casing it first. On a bulk path this
        is where an empty stream, an empty parameter list or a statement built from nothing tends to
        raise, and a replace that raised here would already have emptied the table.
        """
        empty = pl.DataFrame(schema={"key": pl.Int64, "label": pl.String, "amount": pl.Float64, "seen": pl.Date})

        run_tool(WriteTable(), Ports({"rows": empty}), _write(target, mode=mode, allow_empty=True))

        assert target.read("target") == _normalised(after)


class TestAFailedWriteLeavesThePreviousFiguresStanding:
    """The sink's central promise, checked on the driver whose transaction handling is not sqlite3's."""

    @pytest.mark.parametrize(
        ("table", "frame", "before"),
        [
            pytest.param(
                "strict",
                pl.DataFrame({"key": [3, 4], "amount": [3.0, None]}),
                _BEFORE_STRICT,
                id="a_missing_value_where_the_table_demands_one",
            ),
            pytest.param(
                "target",
                pl.DataFrame(
                    {
                        "key": [3, 3],
                        "label": ["third", "again"],
                        "amount": [3.0, 4.0],
                        "seen": [date(2026, 3, 1), date(2026, 4, 1)],
                    }
                ),
                _BEFORE_TARGET,
                id="two_rows_claiming_the_same_key",
            ),
        ],
    )
    def test_a_replace_that_fails_part_way_leaves_the_prior_rows_where_they_were(
        self,
        table: str,
        frame: pl.DataFrame,
        before: tuple[tuple[Any, ...], ...],
        target: _Target,
    ) -> None:
        """The emptying and the loading are one transaction on both drivers, or a bad night costs a table.

        Two ways to fail part way, both after the delete and after at least one row is acceptable,
        which is exactly where a load that committed the delete on its own has already thrown
        yesterday's figures away. The refusal has to reach the caller as this library's error rather
        than as the driver's own exception type -- ``sqlite3.IntegrityError`` on one engine and
        ``psycopg.errors.NotNullViolation`` on the other are two different answers to one question --
        and it has to name the table, because a flow writes more than one.

        The contents are read on a connection of its own afterwards, so a load that raised without
        rolling back is caught: it would either show the emptied table or, having left the
        transaction open, show the old rows while holding a lock the next writer waits on.
        """
        with pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(target, table=table))

        assert table in str(caught.value), f"the refusal must name the table it was writing: {caught.value}"
        assert target.read(table) == _normalised(before), "a failed write must leave the previous figures standing"


class TestBothDriversAnswerARefusalTheSameWay:
    """A refusal carries rows and a total, and the two engines must not disagree about them."""

    def test_the_same_bad_load_reports_the_same_offending_rows_on_either_driver(
        self,
        tmp_path: Path,
        postgres_session: Any,
        postgres_dsn: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One frame, two engines, one answer to "which rows offended, out of how many?".

        ``DataError`` carries the offending rows, the total and a few examples, and that is what a
        reader fixing a bad night actually reads. The SQLite path already builds them by replaying
        the load a row at a time inside a transaction it discards. If the Postgres path instead
        reported nothing, or reported the whole frame, the same defect in the same data would tell
        two readers two different stories depending on where the table happened to live -- so what is
        pinned here is the agreement, not any particular tie-break rule, which the implementation
        stays free to choose as long as it chooses once.
        """
        frame = pl.DataFrame(
            {
                "key": [5, 6, 6],
                "label": ["once", "first six", "twice"],
                "amount": [5.0, 6.0, 7.0],
                "seen": [date(2026, 5, 1), date(2026, 6, 1), date(2026, 7, 1)],
            }
        )

        failures: dict[str, DataError] = {}
        for driver in ("sqlite", "postgres"):
            engine = _build_target(driver, tmp_path, postgres_session, postgres_dsn, monkeypatch)
            with pytest.raises(DataError) as caught:
                run_tool(WriteTable(), Ports({"rows": frame}), _write(engine))
            failures[driver] = caught.value
            assert engine.read("target") == _normalised(_BEFORE_TARGET)

        sqlite_failure, postgres_failure = failures["sqlite"], failures["postgres"]
        assert sqlite_failure.total_rows == postgres_failure.total_rows == frame.height
        assert 1 <= sqlite_failure.offending_rows <= frame.height, "the SQLite path must find the offender"
        assert postgres_failure.offending_rows == sqlite_failure.offending_rows, (
            f"the two drivers disagree about how many rows offend: "
            f"sqlite {sqlite_failure.offending_rows}, postgres {postgres_failure.offending_rows}"
        )
        assert postgres_failure.examples == sqlite_failure.examples, (
            f"the two drivers disagree about which rows offend: "
            f"sqlite {sqlite_failure.examples}, postgres {postgres_failure.examples}"
        )

    def test_a_row_replayed_after_the_offender_is_not_blamed_for_it(
        self,
        tmp_path: Path,
        postgres_session: Any,
        postgres_dsn: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The offender sits in the middle, so the replay has to carry on past it.

        A failed statement on Postgres poisons the whole transaction: every statement after it is
        refused until the block ends. So a replay that attempts rows one after another in a single
        transaction reports the first genuine offender and then every innocent row behind it, and a
        reader fixing a bad night is handed a list in which all but one entry is fiction. Isolating
        each attempt -- a savepoint per row -- is what keeps the list honest.

        The case above cannot see this, because its duplicate is the last row of three and nothing
        is replayed after the failure. This one puts a good row behind the bad one, which is the
        ordinary shape of a bad night and the only shape in which the difference shows.
        """
        frame = pl.DataFrame(
            {
                "key": [5, 6, 6, 7],
                "label": ["once", "first six", "twice", "after the offender"],
                "amount": [5.0, 6.0, 7.0, 8.0],
                "seen": [date(2026, 5, 1), date(2026, 6, 1), date(2026, 7, 1), date(2026, 8, 1)],
            }
        )

        for driver in ("sqlite", "postgres"):
            engine = _build_target(driver, tmp_path, postgres_session, postgres_dsn, monkeypatch)
            with pytest.raises(DataError) as caught:
                run_tool(WriteTable(), Ports({"rows": frame}), _write(engine))
            failure = caught.value
            assert failure.offending_rows == 1, (
                f"{driver}: one row duplicates a key and one row follows it, so exactly one row "
                f"offends -- {failure.offending_rows} were named, which means the row after the "
                f"offender was blamed for the state the offender left behind"
            )
            assert tuple(failure.examples) == (dict(zip(frame.columns, frame.row(2), strict=True)),), (
                f"{driver}: the row named is not the one that duplicated a key: {failure.examples}"
            )
            assert engine.read("target") == _normalised(_BEFORE_TARGET)


class TestTheLoadIsBulk:
    """A hundred thousand rows is the job; a statement per row is a different job that never finishes."""

    def test_the_whole_frame_lands_in_one_statement_rather_than_one_per_row(
        self,
        postgres_session: Any,
        postgres_dsn: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Counted by the server, not asserted about the source.

        A statement-level trigger fires once per statement however many rows that statement touched,
        so it tells a bulk load and a row-by-row loop apart without this test knowing a single word
        of the SQL the load emits: measured here, the driver's row-at-a-time path fires it once per
        row and a bulk stream fires it once. The rows are counted too, because a load that fired one
        trigger and wrote nothing would otherwise pass.
        """
        connection = postgres_session
        monkeypatch.setenv(_VARIABLE, postgres_dsn)
        connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.bulk")
        connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.statement_log")
        connection.execute(f"create table {_SCRATCH_SCHEMA}.bulk (key integer, label text)")
        connection.execute(f"create table {_SCRATCH_SCHEMA}.statement_log (noted integer)")
        connection.execute(
            f"create or replace function {_SCRATCH_SCHEMA}.note_statement() returns trigger "
            f"language plpgsql as $$ begin insert into {_SCRATCH_SCHEMA}.statement_log values (1); "
            f"return null; end $$"
        )
        connection.execute(
            f"create trigger note_statement after insert on {_SCRATCH_SCHEMA}.bulk "
            f"for each statement execute function {_SCRATCH_SCHEMA}.note_statement()"
        )
        frame = pl.DataFrame(
            {"key": list(range(_BULK_HEIGHT)), "label": [f"row {index}" for index in range(_BULK_HEIGHT)]}
        )

        run_tool(
            WriteTable(),
            Ports({"rows": frame}),
            {
                "connection": _VARIABLE,
                "schema": _SCRATCH_SCHEMA,
                "table": "bulk",
                "mode": "append",
                "allow_empty": False,
            },
        )

        written = connection.execute(f"select count(*) from {_SCRATCH_SCHEMA}.bulk").fetchone()[0]
        fired = connection.execute(f"select count(*) from {_SCRATCH_SCHEMA}.statement_log").fetchone()[0]
        assert written == _BULK_HEIGHT, "every row must be there before how it got there is worth measuring"
        assert fired <= 2, (
            f"{_BULK_HEIGHT} rows took {fired} statements: a load that costs a statement per row is "
            f"the wrong shape for the hundred thousand a publish writes"
        )


def _keyed_frame(keys: Sequence[int]) -> pl.DataFrame:
    """A frame shaped like ``target``, in which the keys are the only thing a case varies."""
    height = len(keys)
    return pl.DataFrame(
        {
            "key": list(keys),
            "label": [f"row {index}" for index in range(height)],
            "amount": [float(index) for index in range(height)],
            "seen": [date(2026, 1, 1)] * height,
        }
    )


def _rows_at(frame: pl.DataFrame, indices: Sequence[int]) -> tuple[dict[str, Any], ...]:
    """The frame's own rows at ``indices``, spelled the way a refusal spells its examples.

    A refusal keeps only the first few offenders, so a case that wants every one of them named has
    to stay inside that limit; the count beside it is what carries the rest.
    """
    return tuple(frame[list(indices)].head(MAX_REPORTED_ROWS).to_dicts())


def _one_offender(height: int) -> pl.DataFrame:
    """``height`` rows of distinct keys, bar one near the end that claims the first row's.

    Near the end, and with a good row behind it, on purpose: an offender in the first few rows would
    let something that stops at the first bad row look cheap while still being asked for the whole
    scope, and a frame that ends on its offender never has to carry on past one.
    """
    keys = list(range(1000, 1000 + height))
    keys[height - 2] = keys[0]
    return _keyed_frame(keys)


def _counted_table(connection: Any, dsn: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A table whose every insert statement is counted, failed ones and discarded ones included.

    The bulk test counts statements with an ``AFTER`` trigger writing a row, which is enough there
    because the load it measures succeeds and commits. Neither is true of this one: a statement that
    violates a constraint never reaches its ``AFTER`` trigger, and the whole pass that looks for the
    offenders is rolled back, which would take any row the trigger wrote with it. So the count is
    taken ``BEFORE`` the statement and kept in a sequence, which is the one thing on the server a
    rollback does not undo.
    """
    monkeypatch.setenv(_VARIABLE, dsn)
    connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.counted")
    connection.execute(f"drop sequence if exists {_SCRATCH_SCHEMA}.attempts")
    connection.execute(f"create table {_SCRATCH_SCHEMA}.counted (key integer primary key, label text)")
    connection.execute(f"create sequence {_SCRATCH_SCHEMA}.attempts")
    connection.execute(
        f"create or replace function {_SCRATCH_SCHEMA}.note_attempt() returns trigger language plpgsql "
        f"as $$ begin perform nextval('{_SCRATCH_SCHEMA}.attempts'); return null; end $$"
    )
    connection.execute(
        f"create trigger note_attempt before insert on {_SCRATCH_SCHEMA}.counted "
        f"for each statement execute function {_SCRATCH_SCHEMA}.note_attempt()"
    )


def _refusal_and_attempts(connection: Any, frame: pl.DataFrame) -> tuple[DataError, int]:
    """The refusal ``frame`` earns from the counted table, and what the server was asked for it."""
    connection.execute(f"truncate {_SCRATCH_SCHEMA}.counted")
    connection.execute(f"alter sequence {_SCRATCH_SCHEMA}.attempts restart with 1")
    config = {
        "connection": _VARIABLE,
        "schema": _SCRATCH_SCHEMA,
        "table": "counted",
        "mode": "append",
        "allow_empty": False,
    }
    with pytest.raises(DataError) as caught:
        run_tool(WriteTable(), Ports({"rows": frame}), config)
    last_value, called = connection.execute(f"select last_value, is_called from {_SCRATCH_SCHEMA}.attempts").fetchone()
    return caught.value, (last_value if called else 0)


class TestNamingTheOffendersCostsNoAttemptPerRow:
    """A publish that hits one bad row must fail, not hang for a minute holding a transaction open."""

    def test_one_bad_row_in_a_large_frame_costs_the_same_few_statements_however_many_rows_there_are(
        self,
        postgres_session: Any,
        postgres_dsn: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Measured on the server, at two sizes, because one size cannot tell a cost from a constant.

        Replaying the load a row at a time costs three round trips per row and attempts every row,
        the innocent ones included: measured here, a frame of five thousand with a single duplicate
        took over two seconds to fail against a server on the same host, which is a minute for the
        hundred and thirty thousand a publish writes and tens of minutes at a realistic latency. All
        of it is spent after the load has already failed, producing a diagnosis, while the
        transaction stays open -- so the build does not look failed, it looks stuck.

        The two sizes are what make this a statement about cost rather than about speed: quadrupling
        the rows must not quadruple what is asked of the server. The refusal is checked as well as
        the cost, because something that answered instantly and wrongly would otherwise pass.
        """
        _counted_table(postgres_session, postgres_dsn, monkeypatch)
        small = _one_offender(_DIAGNOSED_HEIGHT).select("key", "label")
        large = _one_offender(_QUADRUPLED_HEIGHT).select("key", "label")

        small_refusal, small_attempts = _refusal_and_attempts(postgres_session, small)
        large_refusal, large_attempts = _refusal_and_attempts(postgres_session, large)

        for height, refusal, frame in (
            (_DIAGNOSED_HEIGHT, small_refusal, small),
            (_QUADRUPLED_HEIGHT, large_refusal, large),
        ):
            assert refusal.total_rows == height
            assert refusal.offending_rows == 1, (
                f"{height} rows, one of which claims a key another already holds: {refusal.offending_rows} were named"
            )
            assert refusal.examples == _rows_at(frame, (height - 2,)), (
                f"the row named is not the one that claimed a key already taken: {refusal.examples}"
            )
        assert small_attempts <= _ALLOWED_ATTEMPTS, (
            f"{_DIAGNOSED_HEIGHT} rows cost {small_attempts} statements against the table; naming "
            f"the offenders must not cost an attempt per row"
        )
        assert large_attempts <= _ALLOWED_ATTEMPTS, (
            f"{_QUADRUPLED_HEIGHT} rows cost {large_attempts} statements against the table; naming "
            f"the offenders must not cost an attempt per row"
        )
        assert large_attempts <= small_attempts, (
            f"four times the rows cost more statements ({large_attempts}) than a quarter of them "
            f"({small_attempts}): what the diagnosis costs still follows the size of the frame"
        )

    def test_a_frame_where_nearly_every_row_offends_costs_no_more_than_one_where_a_single_row_does(
        self,
        postgres_session: Any,
        postgres_dsn: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The mirror of the case above, and the one a cheaper strategy is most likely to lose.

        A frame whose rows all claim the same key is the worst case for anything that pays per bad
        row rather than per row -- and the cost of the good case is often bought by making this one
        dearer than the row-by-row replay it replaced. So the same bound is held here, on a frame in
        which all but the first row offends, and the count is held exactly: every offender, not the
        first handful, because the reader's next question after "it failed" is "how much of it".
        """
        _counted_table(postgres_session, postgres_dsn, monkeypatch)
        frame = _keyed_frame([7] * _DENSE_HEIGHT).select("key", "label")

        refusal, attempts = _refusal_and_attempts(postgres_session, frame)

        assert refusal.offending_rows == _DENSE_HEIGHT - 1, (
            f"every row but the first claims a key the row before it already took, so "
            f"{_DENSE_HEIGHT - 1} rows offend; {refusal.offending_rows} were named"
        )
        assert refusal.examples == _rows_at(frame, (1, 2, 3, 4, 5)), (
            f"the rows named are not the ones that repeated the first row's key: {refusal.examples}"
        )
        assert attempts <= _ALLOWED_ATTEMPTS, (
            f"a frame of {_DENSE_HEIGHT} rows in which {_DENSE_HEIGHT - 1} offend cost {attempts} "
            f"statements against the table: the cost now follows the number of bad rows instead"
        )


class TestTheOffendersNamedAreTheRowsASequentialLoadWouldRefuse:
    """What "offends" means cannot move when how it is found does; both drivers, same answer."""

    @pytest.mark.parametrize(
        ("keys", "offending"),
        [
            pytest.param(
                [100, 101, 102, 103, 104, 101, 106, 107, 104, 109],
                (5, 8),
                id="a_key_repeated_later_in_the_frame_blames_the_later_row",
            ),
            pytest.param(
                [200, 201, 201, 201, 201, 202, 203, 203],
                (2, 3, 4, 7),
                id="a_run_of_rows_holding_one_key_blames_every_one_of_them_but_the_first",
            ),
        ],
    )
    def test_the_second_and_later_rows_claiming_a_key_offend_and_the_first_does_not(
        self,
        keys: list[int],
        offending: tuple[int, ...],
        target: _Target,
    ) -> None:
        """A duplicate is a disagreement between two rows, and only one of them is the offender.

        A sequential load settles that by order: the row that got there first is in the table, and
        the row behind it is refused. Anything that judges the frame against itself instead of
        against what precedes each row has two ways to get this wrong, and both are silent -- naming
        both rows doubles the reported scope of a bad night, and naming neither, because each key on
        its own looks fine somewhere, reports that nothing offended over a load that was refused.
        The run of four is here for the same reason at closer quarters: three of those rows offend,
        not one and not four.

        The table is read afterwards on a connection of its own: the pass that names the offenders
        exists to look, and a replace that had left any of it behind would show here.
        """
        frame = _keyed_frame(keys)

        with pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(target))

        refusal = caught.value
        assert refusal.total_rows == frame.height
        assert refusal.offending_rows == len(offending), (
            f"{target.driver}: the rows that offend are {list(offending)}; {refusal.offending_rows} were named"
        )
        assert refusal.examples == _rows_at(frame, offending), (
            f"{target.driver}: the rows named are not the ones a sequential load would refuse: {refusal.examples}"
        )
        assert target.read("target") == _normalised(_BEFORE_TARGET), (
            f"{target.driver}: naming the offenders must leave the table as it found it"
        )

    def test_an_append_counts_the_keys_the_table_already_holds_not_only_the_frame_s_own(
        self,
        target: _Target,
    ) -> None:
        """The frame alone cannot answer this, and a diagnosis that reads only the frame will say so.

        In an append, nothing is emptied, so a row whose key is already in the table is refused
        exactly as a row repeating a key inside the frame is -- and the frame, read on its own, shows
        nothing wrong with it. This is the case that separates judging the rows from judging the load:
        the first row here duplicates one already in the table and the last duplicates one four
        places in front of it, so an answer built from the frame alone finds one of the two.
        """
        frame = _keyed_frame([2, 7, 8, 8])

        with pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(target, mode="append"))

        refusal = caught.value
        assert refusal.offending_rows == 2, (
            f"{target.driver}: one row holds a key the table already holds and one repeats a key "
            f"inside the frame, so two rows offend; {refusal.offending_rows} were named"
        )
        assert refusal.examples == _rows_at(frame, (0, 3)), (
            f"{target.driver}: the rows named are not the two that would be refused: {refusal.examples}"
        )
        assert target.read("target") == _normalised(_BEFORE_TARGET)

    def test_a_missing_value_and_a_duplicate_key_in_one_frame_are_both_named(
        self,
        target: _Target,
    ) -> None:
        """Two kinds of wrong in one frame, and the answer is the union, not whichever was met first.

        A bad night rarely breaks one rule. Whatever finds the offenders has to keep looking after
        the first kind it recognises: an answer that stopped at the duplicate would send a reader to
        fix one row and rerun into the null, which is the same night lost twice.
        """
        frame = pl.DataFrame({"key": [10, 11, 11, 12, 13], "amount": [1.0, 2.0, 3.0, None, 4.0]})

        with pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(target, table="strict"))

        refusal = caught.value
        assert refusal.offending_rows == 2, (
            f"{target.driver}: one row repeats a key and one leaves out a value the table demands, "
            f"so two rows offend; {refusal.offending_rows} were named"
        )
        assert refusal.examples == _rows_at(frame, (2, 3)), (
            f"{target.driver}: the rows named are not the duplicate and the missing value: {refusal.examples}"
        )
        assert target.read("strict") == _normalised(_BEFORE_STRICT)


class TestARuleTheRowsCannotBeJudgedAgainstStillNamesItsOffenders:
    """The failure a cheaper diagnosis introduces: answering "none" because it could not tell."""

    def test_a_rule_the_table_carries_that_the_rows_cannot_be_checked_against_still_names_them(
        self,
        target: _Target,
    ) -> None:
        """The table refuses these rows for a reason held as an expression the frame cannot evaluate.

        A check is arbitrary SQL. Whatever names the offenders cheaply can answer "this key repeats"
        and "this value is missing" from the rows in hand, and cannot answer this one at all -- which
        is fine, so long as not being able to answer means falling back to something that can. What
        it must never mean is the empty answer: a refusal that says no rows offended, over a load the
        server has just refused, sends a reader looking for a fault in the tool rather than in the
        two rows that caused it, and is worse than the slow answer it replaced.
        """
        frame = pl.DataFrame({"key": [1, 2, 3, 4, 5], "amount": [1.0, 2.0, -3.0, 4.0, -5.0]})

        with pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(target, table="guarded", mode="append"))

        refusal = caught.value
        assert refusal.offending_rows == 2, (
            f"{target.driver}: two rows break a rule the table carries and the frame cannot be "
            f"judged against, so two rows offend; {refusal.offending_rows} were named"
        )
        assert refusal.examples == _rows_at(frame, (2, 4)), (
            f"{target.driver}: the rows named are not the two the table refused: {refusal.examples}"
        )
        assert target.read("guarded") == []

    def test_a_row_the_table_refused_carries_none_of_its_values_forward(
        self,
        target: _Target,
    ) -> None:
        """A row refused for one key does not thereby spend the others it was holding.

        Every other case here puts one rule on the frame's path, and with one rule this cannot be
        seen: a refused row has nothing else to spend. Two independent rules is where the question
        arises, and the answer follows from what a sequential load does -- a row the table refused
        was never written, so none of its values is in the table, so none of them is taken. A reading
        that judged each rule on its own, or that noted a row's values before deciding whether the
        row survived, blames the next row to carry one of them: a genuine offender, then an innocent
        row indicted for repeating a value that never landed.

        Here the second row repeats the first row's key and is refused for it. The code it carried
        goes nowhere, so the third row, which carries that same code and a key of its own, is clean.
        """
        frame = pl.DataFrame(
            {
                "key": [1, 1, 2],
                "code": ["first", "spent by a refused row", "spent by a refused row"],
                "amount": [1.0, 2.0, 3.0],
            }
        )

        with pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(target, table="paired", mode="append"))

        refusal = caught.value
        assert refusal.offending_rows == 1, (
            f"{target.driver}: only the row repeating a key offends -- the row behind it carries a "
            f"code the refused row never managed to write, so it is clean; "
            f"{refusal.offending_rows} rows were named"
        )
        assert refusal.examples == _rows_at(frame, (1,)), (
            f"{target.driver}: the row named is not the one that repeated the key: {refusal.examples}"
        )
        assert target.read("paired") == []

    def test_a_reading_that_breaks_falls_back_and_says_that_it_did(
        self,
        target: _Target,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Declining on any doubt is right; declining without a word is how a fault hides.

        The cheap route answers from the table's rules or not at all, and "not at all" has to mean
        the slow answer rather than the empty one -- so it catches everything. That breadth is what
        makes the answer trustworthy and also what would swallow a genuine fault in the reading: the
        refusal would still name the right rows, by the slow route, and the only symptom would be a
        build that used to fail in under a second taking a minute to say the same thing. Nobody
        debugs that, because nothing says it happened.

        So a decline it did not plan for is logged. The rules it is merely unable to judge -- a check
        expression, a key living in another table -- are the ordinary case and stay quiet; this is
        for the reading that broke.
        """
        real_driver_for = write_table_module.driver_for

        def reader_that_breaks(connection: Any) -> Any:
            loader = real_driver_for(connection)

            def broken(*_: Any, **__: Any) -> None:
                raise RuntimeError("the catalogue reader is broken")

            loader.table_rules = broken  # type: ignore[method-assign]
            return loader

        monkeypatch.setattr(write_table_module, "driver_for", reader_that_breaks)
        frame = pl.DataFrame({"key": [1, 1], "code": ["a", "b"], "amount": [1.0, 2.0]})

        with caplog.at_level(logging.WARNING), pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(target, table="paired", mode="append"))

        assert caught.value.offending_rows == 1, (
            f"{target.driver}: the slow route must still name the offender when the quick one breaks; "
            f"{caught.value.offending_rows} rows were named"
        )
        warned = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("paired" in message for message in warned), (
            f"{target.driver}: the reading broke and nothing said so, so the only sign left is a "
            f"diagnosis that got slower; warnings seen: {warned}"
        )

    def test_a_key_that_has_to_exist_in_another_table_still_names_the_rows_that_lack_one(
        self,
        postgres_session: Any,
        postgres_dsn: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The other rule the rows cannot settle by themselves: one whose answer is in another table.

        Postgres alone, and not because the other driver is exempt: SQLite enforces this only when
        the connection asks it to, and the connection here is opened by the tool. Kept as a case of
        its own rather than folded into the check above because the two fall outside a rows-only
        answer for different reasons -- an expression that cannot be evaluated against the frame, and
        a fact that is not in the frame at all -- and something that fell back for one could easily
        not fall back for the other.
        """
        connection = postgres_session
        monkeypatch.setenv(_VARIABLE, postgres_dsn)
        connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.dependent")
        connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.held")
        connection.execute(f"create table {_SCRATCH_SCHEMA}.held (key integer primary key)")
        connection.execute(
            f"create table {_SCRATCH_SCHEMA}.dependent (key integer primary key, "
            f"held_key integer references {_SCRATCH_SCHEMA}.held(key))"
        )
        connection.cursor().executemany(f"insert into {_SCRATCH_SCHEMA}.held values (%s)", [(1,), (2,)])
        frame = pl.DataFrame({"key": [10, 11, 12, 13], "held_key": [1, 99, 2, 98]})

        with pytest.raises(DataError) as caught:
            run_tool(
                WriteTable(),
                Ports({"rows": frame}),
                {
                    "connection": _VARIABLE,
                    "schema": _SCRATCH_SCHEMA,
                    "table": "dependent",
                    "mode": "append",
                    "allow_empty": False,
                },
            )

        refusal = caught.value
        assert refusal.offending_rows == 2, (
            f"two rows name a key no other table holds, so two rows offend; {refusal.offending_rows} were named"
        )
        assert refusal.examples == _rows_at(frame, (1, 3)), (
            f"the rows named are not the two whose key is not there: {refusal.examples}"
        )
        assert connection.execute(f"select count(*) from {_SCRATCH_SCHEMA}.dependent").fetchone()[0] == 0


class TestAMergeThatFailedDoesNotSpoilTheNextOne:
    """The scratch a merge builds is not always undone by the transaction that held it."""

    def test_a_merge_can_be_run_again_after_one_that_raised(
        self,
        target: _Target,
    ) -> None:
        """A merge stages its rows in a temporary table, and on one driver that table outlives a
        rollback.

        Measured, which is the only reason this test exists: sqlite runs ``CREATE TEMPORARY TABLE``
        outside the unit of work, so a merge that raises between building the scratch and dropping
        it leaves the scratch standing, and the next attempt collides with a name rather than
        failing on its rows. Postgres does not -- ``ON COMMIT DROP`` fires whether the transaction
        commits or rolls back -- so that half of this passes without doing anything, and is here so
        the two drivers are held to one promise rather than to whichever one each finds easy.

        A publish cannot reach this: the tables are written in the order they are configured, and
        the first thing to fail in a flow is almost never the merge at the end of it. So the promise
        is asked of the loader, where it is made.
        """
        with open_connection(os.environ[target.variable]) as connection:
            loader = driver_for(connection)
            columns = ["key", "code", "amount"]

            with pytest.raises(Exception):  # noqa: B017 - the driver's own error, which differs
                with loader.transaction(connection):
                    # A row of the wrong width: the one thing neither driver quietly accepts, and it
                    # lands after the scratch exists and before anything drops it.
                    loader.merge_rows(connection, target.schema, "paired", columns, ["key"], [(1, "a")], keep_published=True)

            with loader.transaction(connection):
                loader.merge_rows(connection, target.schema, "paired", columns, ["key"], [(1, "a", 1.0)], keep_published=True)

        assert target.read("paired") == _normalised([(1, "a", 1.0)]), (
            f"{target.driver}: the merge after a failed one did not land -- the scratch the first "
            f"one left behind is still standing, and the second collided with it"
        )


class TestAConnectionFromNeitherDriverIsRefusedByName:
    """An unknown connection is refused in this library's words, the way an unknown scheme already is."""

    def test_a_connection_this_library_does_not_know_is_refused_rather_than_reached_into(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``db.py`` refuses an unrecognised scheme by naming the ones it has; the load must match.

        A load that dispatches by asking "is this the one driver I recognise, and otherwise assume
        the other" has two ways to be wrong here and this pins both. Reaching for the second driver's
        API hands the caller an ``AttributeError`` raised three frames inside a library they did not
        call. Falling back to a generic path instead is worse: the connection below answers every
        method a load might ask for, so that write would appear to succeed, through a driver whose
        transaction and escaping rules nothing in this suite has ever checked.
        """
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{tmp_path / 'unused.db'}")
        monkeypatch.setattr(write_table_module, "open_connection", _foreign_connection)
        config = {"connection": _VARIABLE, "table": "target", "mode": "replace"}

        with pytest.raises(ToolError) as caught:
            run_tool(WriteTable(), Ports({"rows": _every_shape()}), config)

        message = str(caught.value)
        assert not isinstance(caught.value, ExecutionError), (
            f"an unknown connection is a set-up this library can name, not an accident to wrap: {message}"
        )
        assert "AttributeError" not in message, f"the driver's own accident must not be the answer: {message}"
        assert "sqlite" in message.lower(), f"the refusal must name the drivers it does know: {message}"
        assert any(word in message.lower() for word in ("postgres", "psycopg")), (
            f"the refusal must name the drivers it does know: {message}"
        )


class _ForeignCursor:
    """A cursor that answers everything the sink asks of one, and records what it was asked."""

    description = (("key",), ("label",), ("amount",), ("seen",))

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: str, parameters: Any = None) -> "_ForeignCursor":
        self.statements.append(statement)
        return self

    def executemany(self, statement: str, parameters: Any = None) -> "_ForeignCursor":
        self.statements.append(statement)
        return self

    def fetchall(self) -> list[Any]:
        return []

    def close(self) -> None:
        return None


class _ForeignConnection:
    """A plausible third driver: a full DB-API connection that is neither of the two this library knows.

    Deliberately obliging. It is not missing a method the load might reach for -- it answers all of
    them -- so a load that gets as far as using it does not crash, it writes through an API whose
    transaction and escaping rules nobody here has checked. Refusing it has to be a decision the load
    makes about which driver it was handed, not an accident of a missing attribute.
    """

    def __init__(self) -> None:
        self.cursors: list[_ForeignCursor] = []

    def cursor(self) -> _ForeignCursor:
        cursor = _ForeignCursor()
        self.cursors.append(cursor)
        return cursor

    def execute(self, statement: str, parameters: Any = None) -> _ForeignCursor:
        return self.cursor().execute(statement, parameters)

    def __enter__(self) -> "_ForeignConnection":
        return self

    def __exit__(self, *exception: Any) -> bool:
        return False

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


@contextmanager
def _foreign_connection(uri: str) -> Iterator[_ForeignConnection]:
    """Stands in for ``open_connection`` so a connection from neither driver reaches the load."""
    yield _ForeignConnection()


def _build_target(
    driver: str, tmp_path: Path, postgres_session: Any, dsn: str, monkeypatch: pytest.MonkeyPatch
) -> _Target:
    """The ``target`` fixture's body, for the one test that needs both drivers inside a single case."""
    if driver == "sqlite":
        database = tmp_path / "parity.db"
        database.unlink(missing_ok=True)
        with closing(sqlite3.connect(database)) as connection:
            for statement in _SQLITE_DDL:
                connection.execute(statement)
            connection.executemany("insert into target values (?, ?, ?, ?)", _BEFORE_TARGET)
            connection.commit()
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{database}")

        def read(table: str) -> list[tuple[Any, ...]]:
            with closing(sqlite3.connect(database, timeout=1)) as connection:
                return list(connection.execute(f"select * from {table}").fetchall())

        return _Target("sqlite", _VARIABLE, None, read)

    connection = postgres_session
    connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.target")
    connection.execute(_POSTGRES_DDL[0].format(schema=_SCRATCH_SCHEMA))
    connection.cursor().executemany(
        f"insert into {_SCRATCH_SCHEMA}.target values (%s, %s, %s, %s)",
        [(key, label, amount, date.fromisoformat(seen)) for key, label, amount, seen in _BEFORE_TARGET],
    )
    monkeypatch.setenv(_VARIABLE, dsn)

    def read_postgres(table: str) -> list[tuple[Any, ...]]:
        return list(connection.execute(f"select * from {_SCRATCH_SCHEMA}.{table}").fetchall())

    return _Target("postgres", _VARIABLE, _SCRATCH_SCHEMA, read_postgres)
