"""Tests for the PublishTables sink: every related table commits together, or none of them does.

``write_table`` writes one table per step, and one step is one transaction. A set of related tables
-- rows, a series over periods, an audit, an append-only change log, a dimension keyed per party --
written as one step each is that many separate transactions, so a build that falls over half way
through leaves the destination holding three tables from tonight and two from last night. Nothing in
the flow contract passes scratch space between steps, so there is nowhere outside a single step for
that transaction to live. That is the whole reason this tool exists, and

**all of the tables or none of them** is the property every other test here is arranged around.
:class:`TestAFailedPublishLeavesEveryTableWhereItWas` is the one to read first: a publish that fails
on its third table must leave the first two holding exactly the rows and exactly the values they
held before, the append-only log no longer than it was, and the dimension untouched -- on both
engines, because ``sqlite3`` and ``psycopg`` disagree about what a unit of work even is, and a sink
proven against one of them is a sink proven against the driver it was written for.

The rest of what is pinned, and why each is worth a test of its own:

**One stamp across every table.** The instant belongs to the publish, not to the data and not to any
one table. Tables stamped a few milliseconds apart cannot be read afterwards as one set of figures:
a reader asking "which build produced this row" gets a different answer per table and no way to
group them. So the tables are checked against each other rather than against a clock, and a second
publish is checked for carrying a different instant, which is what separates a real stamp from a
constant.

**A merge keeps a published value the incoming row is silent about.** The distinction the plain
upsert gets wrong. A refresh that fills a gap should fill it, and a refresh that overwrites a value
should overwrite it -- but a null the build has nothing to say about must leave the published value
standing, because the alternative is that one unreadable upstream source publishes a dimension of
nulls over every row. Not truncating a merge target protects only the keys the build does not hold,
and a build that holds them all is protected by nothing but the ``COALESCE``.

**Append never truncates.** A change log is the one table a rebuild must not rewrite; a publish that
emptied it would pass every replace test in this file and lose history no rerun can bring back.

**A configuration that cannot publish is refused before anything is written.** An unknown mode, a
merge with no key, a key naming a column nobody carries, two entries fighting over one port or one
table, a table that is not there, columns that do not line up. Each is checked for the refusal *and*
for the destination still holding the previous publish, because a refusal that arrives after four
tables have been emptied is not a refusal.

**A column that cannot land in its table's column is refused rather than written.** SQLite has no
opinion about this at all -- it stores the string ``'heavy'`` in an ``integer`` column and reports
success -- so nothing but the tool checking the frame against the destination's own types stands
between a mistyped column and a published table full of values of the wrong kind. Measured, not
assumed: see the note on :class:`TestAColumnThatCannotLand`.

**Empty frames follow ``write_table``'s ruling, per table.** A replace of nothing empties a table,
which is the loudest silence a nightly build leaves behind, so an empty frame stops the publish --
the *whole* publish, since it is one unit -- unless that entry says in writing that it may be empty.

**The ports come from the configuration.** A publish of five tables reads five frames and a publish
of one reads one, which no static attribute can say; the derivation follows ``write_excel``'s,
down to passing over a malformed configuration rather than refusing it there.

The SQLite half runs anywhere. The Postgres half skips loudly when there is no server. The schema it
creates is one nothing else names, created and dropped by the fixture, and no test here names any
other schema or reads from one.
"""

import sqlite3
import time
from collections.abc import Callable, Mapping
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl.tool import ConfigError, Ports, ToolError, input_ports_for, run_tool
from dantron_etl.tools.publish_tables import PublishTables

# The name a pipeline definition would carry: the name of the variable, never the connection itself.
_VARIABLE = "ETL_TEST_PUBLISH_CONNECTION"


# The tests own this schema and nothing else. It is created, used and dropped by the fixture below.
# The server this runs against also holds real data in schemas of its own, and no test here names
# any schema but this one.
_SCRATCH_SCHEMA = "dantron_etl_publish_scratch"

# The column every table carries the publish instant in. Named once: it is a configuration field, a
# column in five tables and a column on the receipt, and all four of those have to agree.
_STAMP_COLUMN = "built_at"

# What the last publish left behind. Every failure test asserts the destination still holds exactly
# this, which is what "the previous figures are still standing" means.
_EARLIER = datetime(2026, 1, 1, 0, 0, 0)

_SQLITE_DDL: tuple[str, ...] = (
    "create table entry_rows (entry_id integer primary key, label text, weight integer, "
    "amount real, built_at timestamp)",
    "create table entry_series (period text primary key, amount real, built_at timestamp)",
    "create table entry_audit (note_id integer, note text not null, built_at timestamp)",
    "create table entry_changes (entry_id integer, movement real, built_at timestamp)",
    "create table entry_parties (party text primary key, label text, amount real, built_at timestamp)",
    "create table entry_series_mirror (period text primary key, amount real, built_at timestamp)",
    "create table built_at (value integer, built_at timestamp)",
)

_POSTGRES_DDL: tuple[str, ...] = (
    "create table {schema}.entry_rows (entry_id integer primary key, label text, weight integer, "
    "amount double precision, built_at timestamp)",
    "create table {schema}.entry_series (period text primary key, amount double precision, built_at timestamp)",
    "create table {schema}.entry_audit (note_id integer, note text not null, built_at timestamp)",
    "create table {schema}.entry_changes (entry_id integer, movement double precision, built_at timestamp)",
    "create table {schema}.entry_parties (party text primary key, label text, amount double precision, "
    "built_at timestamp)",
    "create table {schema}.entry_series_mirror (period text primary key, amount double precision, built_at timestamp)",
    "create table {schema}.built_at (value integer, built_at timestamp)",
)

# The five tables one publish names.
_PUBLISHED: tuple[str, ...] = ("entry_rows", "entry_series", "entry_audit", "entry_changes", "entry_parties")

# Every table the destination holds. The sixth carries the same columns as ``entry_series`` and is
# named by no publish here: it is what lets the two "one port twice / one table twice" refusals be
# about the repetition and nothing else. A second entry pointed at a table whose columns happen not
# to match would be refused for the column disagreement instead, and the test would pass over an
# implementation that never checks for a repeat at all.
_TABLES: tuple[str, ...] = (*_PUBLISHED, "entry_series_mirror", "built_at")

# The previous publish, as each table holds it. Read back and compared against after every refusal.
_EARLIER_CONTENTS: Mapping[str, tuple[tuple[Any, ...], ...]] = {
    "entry_rows": ((1, "first", 10, 1.5, _EARLIER), (2, "second", 20, 2.5, _EARLIER)),
    "entry_series": (("2026-01", 100.0, _EARLIER),),
    "entry_audit": ((1, "checked", _EARLIER),),
    "entry_changes": ((1, 0.5, _EARLIER),),
    "entry_parties": (("a", "Alpha", 10.0, _EARLIER), ("b", "Beta", 20.0, _EARLIER)),
    "entry_series_mirror": (("2025-12", 50.0, _EARLIER),),
    # Named after the stamp column and carrying a column of that name. Left empty: what it is
    # for is the one publish that must never reach it.
    "built_at": (),
}


def _plain(value: Any) -> Any:
    """One spelling per value, so a row read from either engine compares against one expectation.

    Two engines, one meaning. Postgres hands back a ``datetime`` for the stamp column; SQLite hands
    back the ISO text its adapter wrote. Both are read as that same text. An offset is stripped
    because whether the publish instant is spelled in UTC-aware or in naive form is the
    implementation's choice and not a property under test -- what is under test is that every table
    carries the *same* one. Stripping it from an ordinary text value would be wrong, and no text
    column here is ever written with one in it.
    """
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value.isoformat(sep=" ")
    if isinstance(value, str):
        return value.replace("+00:00", "").removesuffix("Z")
    return value


def _normalised(rows: Any) -> list[tuple[Any, ...]]:
    """Rows in a settled order, each value in its one comparable spelling."""
    return sorted((tuple(_plain(value) for value in row) for row in rows), key=repr)


class _Target:
    """One database holding the five tables a publish writes, and the name a definition reaches it by.

    ``read`` never reuses the connection the tool used. It asks the server on a connection of its
    own, which is what makes "the rows landed" mean committed rather than merely sent, and what makes
    "the previous figures are still standing" mean the failed transaction was rolled back rather than
    left open holding a lock.
    """

    def __init__(self, driver: str, variable: str, schema: str | None, read: Callable[[str], Any]) -> None:
        self.driver = driver
        self.variable = variable
        self.schema = schema
        self._read = read

    def read(self, table: str) -> list[tuple[Any, ...]]:
        return _normalised(self._read(table))

    def everything(self) -> dict[str, list[tuple[Any, ...]]]:
        """Every table this publish writes, as it stands now -- the whole destination in one value."""
        return {table: self.read(table) for table in _TABLES}


def _as_it_was() -> dict[str, list[tuple[Any, ...]]]:
    """The whole destination as the previous publish left it."""
    return {table: _normalised(rows) for table, rows in _EARLIER_CONTENTS.items()}


@pytest.fixture(scope="session")
def postgres_session(postgres_scratch: Any) -> Any:
    """A Postgres connection owning this file's scratch schema (see ``postgres_scratch`` in conftest)."""
    return postgres_scratch(_SCRATCH_SCHEMA)


def _seed_sqlite(database: Path) -> None:
    """Every table, holding what the previous publish left in it."""
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("pragma synchronous = off")  # built once per session as a template
        for statement in _SQLITE_DDL:
            connection.execute(statement)
        for table, rows in _EARLIER_CONTENTS.items():
            if not rows:  # a table the previous publish left empty has nothing to seed
                continue
            marks = ", ".join("?" for _ in rows[0])
            connection.executemany(f"insert into {table} values ({marks})", [tuple(row) for row in rows])
        connection.commit()


@pytest.fixture(params=["sqlite", "postgres"])
def target(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Target:
    """The same five tables, holding the same previous publish, on whichever engine this run is about.

    SQLite is here because the catalog's worked examples have to run with no server at all. Postgres
    is here because it is what a publish actually meets, and because the two engines disagree about
    every mechanism this tool depends on: what a transaction is, whether a failed statement poisons
    the rest of one, and how a staging table is made and disposed of.
    """
    if request.param == "sqlite":
        database = request.getfixturevalue("seeded_sqlite")(tmp_path / "publish.db", _seed_sqlite)
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{database}")

        def read(table: str) -> list[tuple[Any, ...]]:
            with closing(sqlite3.connect(database, timeout=1)) as connection:
                return list(connection.execute(f"select * from {table}").fetchall())

        return _Target("sqlite", _VARIABLE, None, read)

    connection = request.getfixturevalue("postgres_session")
    dsn = request.getfixturevalue("postgres_dsn")
    for table in _TABLES:
        connection.execute(f"drop table if exists {_SCRATCH_SCHEMA}.{table}")
    for statement in _POSTGRES_DDL:
        connection.execute(statement.format(schema=_SCRATCH_SCHEMA))
    cursor = connection.cursor()
    for table, rows in _EARLIER_CONTENTS.items():
        if not rows:  # a table the previous publish left empty has nothing to seed
            continue
        marks = ", ".join("%s" for _ in rows[0])
        cursor.executemany(f"insert into {_SCRATCH_SCHEMA}.{table} values ({marks})", [tuple(row) for row in rows])
    monkeypatch.setenv(_VARIABLE, dsn)

    def read(table: str) -> list[tuple[Any, ...]]:
        return list(connection.execute(f"select * from {_SCRATCH_SCHEMA}.{table}").fetchall())

    return _Target("postgres", _VARIABLE, _SCRATCH_SCHEMA, read)


def _entries() -> list[dict[str, Any]]:
    """The five tables of one publish: three replaced, one appended to, one merged on its key."""
    return [
        {"port": "rows", "table": "entry_rows", "mode": "replace"},
        {"port": "series", "table": "entry_series", "mode": "replace"},
        {"port": "audit", "table": "entry_audit", "mode": "replace"},
        {"port": "changes", "table": "entry_changes", "mode": "append"},
        {"port": "parties", "table": "entry_parties", "mode": "merge", "keys": ["party"]},
    ]


def _publish(target: _Target, **overrides: Any) -> dict[str, Any]:
    """One publish's configuration on ``target``, with whatever a case changes."""
    config: dict[str, Any] = {
        "connection": target.variable,
        "stamp_column": _STAMP_COLUMN,
        "tables": _entries(),
        **overrides,
    }
    if target.schema is not None:
        config["schema"] = target.schema
    return config


def _frames(**overrides: pl.DataFrame) -> Ports:
    """Tonight's figures on the five ports, none of them carrying the stamp column.

    The stamp is deliberately absent from every frame. It is a fact about the publish rather than
    about the data, so the flow neither computes it nor gets to disagree with it between tables.
    """
    frames: dict[str, pl.DataFrame] = {
        "rows": pl.DataFrame(
            {
                "entry_id": [3, 4],
                "label": ["third", "fourth"],
                "weight": [30, 40],
                "amount": [3.5, 4.5],
            }
        ),
        "series": pl.DataFrame({"period": ["2026-02", "2026-03"], "amount": [200.0, 300.0]}),
        "audit": pl.DataFrame({"note_id": [1, 2], "note": ["read", "again"]}),
        "changes": pl.DataFrame({"entry_id": [3], "movement": [1.25]}),
        "parties": pl.DataFrame({"party": ["a", "c"], "label": [None, "Gamma"], "amount": [11.0, 30.0]}),
    }
    frames.update(overrides)
    return Ports(frames)


def _a_bad_audit() -> pl.DataFrame:
    """Tonight's audit with a missing value in its second row, where the table demands one.

    The failure is deliberately not in the first row: a publish that fell over before writing
    anything at all would prove nothing about what it does to the tables it had already written.
    """
    return pl.DataFrame({"note_id": [1, 2], "note": ["read", None]})


def _stamp_of(receipt: pl.DataFrame) -> Any:
    """The one instant the receipt says this publish carried, in its comparable spelling."""
    return _plain(receipt[_STAMP_COLUMN][0])


def _written_stamps(target: _Target) -> set[Any]:
    """Every distinct instant now standing in the destination, bar the one the last publish left."""
    return {
        _plain(row[-1]) for table in _PUBLISHED for row in target.read(table) if _plain(row[-1]) != _plain(_EARLIER)
    }


class TestAPublishWritesEveryTableAndHandsBackOneReceipt:
    """The ordinary night: five tables, three modes, one instant, one row of accounting for it."""

    def test_every_table_is_written_by_its_own_mode_and_the_receipt_counts_the_rows(self, target: _Target) -> None:
        """One step writes all five, each by the mode its entry names, and says what it wrote.

        The three modes are checked here against each other rather than one at a time, because the
        way this goes wrong is not a mode that fails outright -- that would be caught anywhere -- but
        a mode leaking into its neighbour: a load that empties every table regardless would pass a
        replace-only suite and silently destroy the change log, and a load that empties none of them
        would double tonight's figures into the rows and the series.

        The receipt is one row. It carries the stamp and the number of rows written per table, and
        nothing else: five pass-through ports would be five ports nobody wires, where a count per
        table is what a reader checking tonight's build against last night's actually reads.
        """
        result = run_tool(PublishTables(), _frames(), _publish(target))

        receipt = result["receipt"]
        assert receipt.height == 1, f"the receipt is one row, not {receipt.height}"
        assert set(receipt.columns) == {_STAMP_COLUMN, *_PUBLISHED}, (
            f"the receipt must carry the stamp and one count per table; it carries: {receipt.columns}"
        )
        assert {table: receipt[table][0] for table in _PUBLISHED} == {
            "entry_rows": 2,
            "entry_series": 2,
            "entry_audit": 2,
            "entry_changes": 1,
            "entry_parties": 2,
        }

        stamp = _stamp_of(receipt)
        assert target.everything() == {
            "entry_rows": _normalised([(3, "third", 30, 3.5, stamp), (4, "fourth", 40, 4.5, stamp)]),
            "entry_series": _normalised([("2026-02", 200.0, stamp), ("2026-03", 300.0, stamp)]),
            "entry_audit": _normalised([(1, "read", stamp), (2, "again", stamp)]),
            "entry_changes": _normalised([(1, 0.5, _EARLIER), (3, 1.25, stamp)]),
            "entry_parties": _normalised(
                [("a", "Alpha", 11.0, stamp), ("b", "Beta", 20.0, _EARLIER), ("c", "Gamma", 30.0, stamp)]
            ),
            "entry_series_mirror": _normalised(_EARLIER_CONTENTS["entry_series_mirror"]),
            # Named after the stamp column; no publish here reaches it, and none may.
            _STAMP_COLUMN: [],
        }


class TestAFailedPublishLeavesEveryTableWhereItWas:
    """The property the tool exists for, and the one class in this file to read first.

    A publish is one transaction over every table it names. The failure this excludes is the one the
    consuming project's own comment describes: a build that empties and reloads five related tables
    one at a time, falls over on the third, and leaves the destination holding two tables of
    tonight's figures beside three of last night's -- published, agreeing with nothing, and with no
    record of which is which.
    """

    def test_a_publish_that_fails_on_one_table_changes_none_of_them(self, target: _Target) -> None:
        """The third table's load is refused, and all five stand exactly as the last publish left them.

        Every mode is represented among the four tables that must not move, and each has its own way
        of going wrong if the transaction is not one: the two replaced before the failure would show
        tonight's rows or nothing at all, the appended log would have grown by a row belonging to a
        publish that never happened, and the merged dimension would carry a mixture of tonight's
        values and last night's with no stamp able to tell them apart.

        Read afterwards on a connection of its own, so a run that raised without rolling back is
        caught: it would show either the changed tables or, having left its transaction open, the old
        rows behind a lock the next reader waits on.
        """
        before = _as_it_was()

        with pytest.raises(ToolError) as caught:
            run_tool(PublishTables(), _frames(audit=_a_bad_audit()), _publish(target))

        assert "entry_audit" in str(caught.value), (
            f"the refusal must name the table whose load was refused: {caught.value}"
        )
        assert target.everything() == before, (
            "a publish that failed part way left the destination changed; all of the tables commit "
            "together or none of them does"
        )

    def test_a_publish_that_failed_can_be_rerun_and_then_lands_whole(self, target: _Target) -> None:
        """After a failure there is nothing half-written left to get in the way of the next attempt.

        The consequence of the promise that a reader relies on at two in the morning: fix the frame,
        run it again, and the publish lands whole. A tool that left a staging table behind, or a
        transaction open, or one table already emptied, would fail the second run for a reason that
        has nothing to do with the data -- which is the shape in which "all or nothing" usually turns
        out to be only nearly true.
        """
        with pytest.raises(ToolError):
            run_tool(PublishTables(), _frames(audit=_a_bad_audit()), _publish(target))

        result = run_tool(PublishTables(), _frames(), _publish(target))

        stamp = _stamp_of(result["receipt"])
        assert target.read("entry_audit") == _normalised([(1, "read", stamp), (2, "again", stamp)])
        assert target.read("entry_changes") == _normalised([(1, 0.5, _EARLIER), (3, 1.25, stamp)]), (
            "the attempt that failed appended to the change log after all"
        )


class TestTheStampIsOneInstantAcrossThePublish:
    """One publish, one instant, written identically onto every table it wrote."""

    def test_one_publish_writes_one_instant_everywhere_and_the_next_publish_writes_another(
        self, target: _Target
    ) -> None:
        """Every table agrees with every other and with the receipt, and the next publish moves on.

        Two mistakes are caught. Taking the instant per table -- ``datetime.now()`` reached for
        inside the loop -- gives five values milliseconds apart, so the tables of one build can no
        longer be selected as one build. Taking it from somewhere fixed -- a constant, or a value off
        one of the frames -- gives every build the same answer, so "which build produced this row"
        has no answer at all. The distinct count catches the first; the second publish catches the
        second.

        The sleep is there so that the second assertion is about the tool rather than about the
        resolution of a clock. The first needs no such help: a publish spends far longer than a clock
        tick between its first table and its last.
        """
        first = run_tool(PublishTables(), _frames(), _publish(target))

        assert _written_stamps(target) == {_stamp_of(first["receipt"])}, (
            "one publish did not write one instant onto every table it wrote"
        )

        time.sleep(0.01)
        second = run_tool(PublishTables(), _frames(), _publish(target))

        assert _stamp_of(second["receipt"]) != _stamp_of(first["receipt"]), (
            "two publishes carried the same stamp, so the stamp is not this publish's own instant"
        )
        assert {row[-1] for row in target.read("entry_rows")} == {_stamp_of(second["receipt"])}, (
            "a replaced table still carries the first publish's stamp"
        )


class TestMergeKeepsWhatTheIncomingRowIsSilentAbout:
    """Three behaviours, and the first is the one a plain upsert gets wrong."""

    def test_a_null_leaves_the_published_value_standing_while_a_value_overwrites_it(self, target: _Target) -> None:
        """A gap is kept, a value is overwritten, a new key is inserted, an absent key is untouched.

        Party 'a' arrives with no label and a new amount: the published label must survive and the
        amount must move. Party 'c' is new. Party 'b' is not in tonight's frame at all and must be
        exactly as it was, stamp included -- the stamp is how a reader tells which rows this publish
        actually touched.

        ``= EXCLUDED`` passes two of those four and fails the first, and fails it in the way that
        matters most: on a night when one upstream source could not be read, the whole dimension
        arrives with nulls in that column and a plain upsert publishes them over every row. Not
        truncating protects only the keys the build does not hold, and the build holds them all.
        """
        result = run_tool(PublishTables(), _frames(), _publish(target))

        stamp = _stamp_of(result["receipt"])
        assert target.read("entry_parties") == _normalised(
            [("a", "Alpha", 11.0, stamp), ("b", "Beta", 20.0, _EARLIER), ("c", "Gamma", 30.0, stamp)]
        )

    def test_a_second_publish_replaces_the_series_and_adds_to_the_log(self, target: _Target) -> None:
        """Running the same publish twice: the replaced tables do not double, the log does grow.

        The mirror of the first test in this file, and what catches a mode leaking into its
        neighbour in the other direction. A change log is append-only precisely because it is the one
        record a rebuild cannot recreate, so a load that truncated it would lose history no rerun can
        bring back -- and it would look perfectly healthy in any single-run test.
        """
        run_tool(PublishTables(), _frames(), _publish(target))
        second = run_tool(PublishTables(), _frames(), _publish(target))

        stamp = _stamp_of(second["receipt"])
        assert target.read("entry_series") == _normalised([("2026-02", 200.0, stamp), ("2026-03", 300.0, stamp)]), (
            "a replaced table doubled its rows over two publishes"
        )
        changes = target.read("entry_changes")
        assert len(changes) == 3, f"the append-only log is not three rows after two publishes: {changes}"
        assert _normalised([(1, 0.5, _EARLIER)])[0] in changes, "the change log lost the row it already held"


def _an_unknown_mode(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    config["tables"][0]["mode"] = "upsert"
    return config, inputs


def _a_merge_with_no_key(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    del config["tables"][4]["keys"]
    return config, inputs


def _a_key_the_frame_does_not_carry(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    config["tables"][4]["keys"] = ["counterparty"]
    return config, inputs


def _two_entries_on_one_port(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    """Both entries carry columns their table accepts, so the repeated port is the only fault."""
    config["tables"].append({"port": "series", "table": "entry_series_mirror", "mode": "replace"})
    return config, inputs


def _two_entries_on_one_table(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    """A second port writing the same table, and the two replaces would silently overwrite each other."""
    config["tables"].append({"port": "mirror", "table": "entry_series", "mode": "replace"})
    return config, _frames(mirror=pl.DataFrame({"period": ["2026-04"], "amount": [400.0]}))


def _a_table_that_is_not_there(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    config["tables"][1]["table"] = "entry_summary"
    return config, inputs


def _a_stamp_named_after_a_table_whose_columns_line_up(
    config: dict[str, Any], inputs: Ports
) -> tuple[dict[str, Any], Ports]:
    """The collision in the one shape the column-agreement rule cannot see.

    A table named after the stamp column, carrying a column of that name: the frame's columns plus
    the stamp are exactly the table's, so nothing disagrees and the publish runs. Then the receipt
    seeds the stamp under its column name and writes a row count under the table's, which is the
    same name -- and the count wins. The publish reports success and hands back an instant that is
    a row count. Measured before this was refused; it is why the case above is not enough.
    """
    return (
        {**config, "tables": [{"port": "only", "table": _STAMP_COLUMN, "mode": "replace"}]},
        Ports({"only": pl.DataFrame({"value": [1]})}),
    )


def _a_stamp_named_after_a_table(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    """The stamp column given the name of one of the tables being published.

    The receipt carries one column per table and one for the stamp, so two of them answering to one
    name would be a receipt with a column missing -- missing quietly, reporting a row count as an
    instant or the other way about, over a publish that otherwise succeeded.

    Measured: it is already refused, though not by any rule about stamps. The stamp is added to the
    frame's side when the columns are compared against the table's, so a stamp named after a table
    makes that comparison fail and the publish stops before anything is written. The message it
    gives is about columns rather than about the name, which reads oddly -- but the property worth
    holding is that the publish does not proceed, and nothing quietly loses a column. Pinned here so
    that a later change to how the columns are compared cannot open the hole by accident.
    """
    return {**config, "stamp_column": config["tables"][0]["table"]}, inputs


def _columns_that_do_not_line_up(config: dict[str, Any], inputs: Ports) -> tuple[dict[str, Any], Ports]:
    """The table has a column, ``amount``, that the frame lacks."""
    return config, _frames(series=pl.DataFrame({"period": ["2026-02"]}))


def _a_frame_carrying_a_column_the_table_lacks(
    config: dict[str, Any], inputs: Ports
) -> tuple[dict[str, Any], Ports]:
    """The mirror: the frame has a column, ``currency``, that the table lacks."""
    return config, _frames(
        series=pl.DataFrame({"period": ["2026-02"], "amount": [200.0], "currency": ["EUR"]})
    )


class TestAConfigurationThatCannotPublishIsRefused:
    """Ten ways a publish is wrong before it starts, each refused with nothing written."""

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(_an_unknown_mode, id="an_unknown_mode"),
            pytest.param(_a_merge_with_no_key, id="a_merge_with_no_key"),
            pytest.param(_a_key_the_frame_does_not_carry, id="a_key_naming_a_column_nobody_carries"),
            pytest.param(_two_entries_on_one_port, id="two_entries_claiming_one_port"),
            pytest.param(_two_entries_on_one_table, id="two_entries_claiming_one_table"),
            pytest.param(_a_table_that_is_not_there, id="a_table_that_is_not_there"),
            pytest.param(_columns_that_do_not_line_up, id="a_frame_whose_columns_are_not_the_tables"),
            pytest.param(_a_frame_carrying_a_column_the_table_lacks, id="a_frame_carrying_a_column_the_table_lacks"),
            pytest.param(_a_stamp_named_after_a_table, id="a_stamp_named_after_a_table"),
            pytest.param(
                _a_stamp_named_after_a_table_whose_columns_line_up, id="a_stamp_named_after_a_table_that_carries_it"
            ),
        ],
    )
    def test_it_is_refused_and_the_destination_is_untouched(
        self,
        mutate: Callable[[dict[str, Any], Ports], tuple[dict[str, Any], Ports]],
        target: _Target,
    ) -> None:
        """Each is a mistake in the configuration, so each is this library's ConfigError, not a driver's.

        The second assertion is the one worth having. A refusal that arrives after three tables have
        been emptied is not a refusal, it is a failed publish with a helpful message, and the point
        of checking a configuration is that it costs nothing and happens before the destination is
        touched. Two of these -- the missing table, and the columns that do not line up -- can only
        be answered by asking the server, which is exactly where the temptation to find out by trying
        it comes from.
        """
        before = _as_it_was()
        config, inputs = mutate(_publish(target), _frames())

        with pytest.raises(ConfigError):
            run_tool(PublishTables(), inputs, config)

        assert target.everything() == before, "a configuration was refused after the destination had changed"


class TestAColumnThatCannotLand:
    """A frame column that will not go into its table's column, refused rather than written.

    Measured rather than assumed. On SQLite, inserting ``'heavy'`` into a column declared ``integer``
    stores the five characters and reports success, and ``1.5`` into the same column stores ``1.5``;
    they come back as a ``str`` and a ``float`` out of an ``integer`` column. Nothing in that engine
    stands between a mistyped column and a published table, so the tool has to.
    """

    @pytest.mark.parametrize(
        ("weight", "reason"),
        [
            pytest.param(["heavy", "light"], "text where the table holds a whole number", id="text_for_an_integer"),
            pytest.param([1.5, 2.5], "a fraction where the table holds a whole number", id="a_fraction_for_an_integer"),
        ],
    )
    def test_it_is_refused_by_name_and_nothing_is_published(
        self, weight: list[Any], reason: str, target: _Target
    ) -> None:
        """The refusal names the column, and the previous publish is still standing afterwards.

        Naming the column is what makes this worth checking rather than leaving to the engine. One of
        the two engines says nothing at all, and the other says it at load time, in its own words,
        about a stream it was reading -- by which point the publish has already emptied three tables
        and what a reader is told is about a copy stream rather than about ``weight``.
        """
        before = _as_it_was()
        rows = pl.DataFrame({"entry_id": [3, 4], "label": ["third", "fourth"], "weight": weight, "amount": [3.5, 4.5]})

        with pytest.raises(ToolError) as caught:
            run_tool(PublishTables(), _frames(rows=rows), _publish(target))

        assert "weight" in str(caught.value), f"the refusal must name the column ({reason}): {caught.value}"
        assert target.everything() == before, "a column that could not land was published anyway"


class TestAnEmptyFrameStopsThePublish:
    """``write_table``'s ruling, carried per table: a replace of nothing empties a table."""

    def test_an_empty_frame_stops_every_table_until_its_entry_allows_it(self, target: _Target) -> None:
        """One empty frame stops the whole publish, and an entry that expects one says so in writing.

        Per table rather than per publish, because the branches differ: a series that is legitimately
        empty in a quiet period is one entry's business, and turning the check off for all five to
        accommodate it would let a genuinely broken upstream empty the other four in silence.

        And the refusal stops *everything*, not merely its own table. That follows from the publish
        being one unit: a set of figures missing one of its tables is not a publish, so there is no
        version of "carry on with the other four" that leaves something readable behind.
        """
        before = _as_it_was()
        empty = pl.DataFrame(schema={"period": pl.String, "amount": pl.Float64})

        with pytest.raises(ToolError) as caught:
            run_tool(PublishTables(), _frames(series=empty), _publish(target))

        assert "entry_series" in str(caught.value), f"the refusal must name the empty table: {caught.value}"
        assert target.everything() == before, "an empty frame emptied the tables around it"

        allowed = _publish(target)
        allowed["tables"][1]["allow_empty"] = True
        result = run_tool(PublishTables(), _frames(series=empty), allowed)

        assert result["receipt"]["entry_series"][0] == 0
        assert target.read("entry_series") == [], "the entry that allowed an empty frame was not replaced by it"
        assert len(target.read("entry_rows")) == 2, "the rest of the publish did not go ahead"


class TestThePortsComeFromTheTablesNamed:
    """A publish of five tables reads five frames, and a publish of one reads one."""

    def test_the_ports_are_the_entries_in_order_and_a_malformed_configuration_is_passed_over(self) -> None:
        """Derived from ``tables``, in the order written, following ``write_excel``'s rule exactly.

        No static attribute can say this: two steps of this tool publish different numbers of tables
        and therefore read different numbers of frames. The malformed cases are passed over rather
        than refused here, because this is asked of the configuration as a person wrote it, before
        any default is filled in and before the template has checked anything -- a refusal here would
        arrive in words about ports when what is wrong is a field, and the template check a moment
        later says so properly.
        """
        tool = PublishTables()
        one = {"connection": "C", "tables": [{"port": "only", "table": "entry_changes", "mode": "append"}]}

        assert input_ports_for(tool, {"connection": "C", "tables": _entries()}) == (
            "rows",
            "series",
            "audit",
            "changes",
            "parties",
        )
        assert input_ports_for(tool, one) == ("only",)
        for malformed in ({}, {"tables": "rows"}, {"tables": [{"table": "entry_rows"}]}, {"tables": []}):
            assert input_ports_for(tool, malformed) == (), f"a malformed configuration was not passed over: {malformed}"

    def test_a_publish_of_one_table_reads_one_frame_and_writes_only_that_table(self, target: _Target) -> None:
        """However many entries the configuration names, that is how many frames the step reads.

        Run rather than only asked, because the derivation is only half of it: the boundary refuses
        an undeclared port, and a tool that derived one set of ports and then reached for another
        would pass the question above and fail here.
        """
        config = _publish(target, tables=[{"port": "changes", "table": "entry_changes", "mode": "append"}])
        inputs = Ports({"changes": pl.DataFrame({"entry_id": [3], "movement": [1.25]})})

        result = run_tool(PublishTables(), inputs, config)

        assert result["receipt"].columns == [_STAMP_COLUMN, "entry_changes"]
        assert len(target.read("entry_changes")) == 2
        assert target.read("entry_rows") == _normalised(_EARLIER_CONTENTS["entry_rows"]), (
            "a publish named one table and wrote another"
        )
