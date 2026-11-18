"""Tests for the WriteTable tool: all of the rows or none of them, and the old figures still there.

WriteTable is the catalog's first sink. One input port, ``rows``; the rulings pinned below are what
separate a bad night from a lost table.

**Truncate-and-load inside one transaction.** A replace empties the table and loads the frame as one
unit, so a write that fails part way leaves the previous figures standing rather than an empty
table. That is the single most important test in this file: the failure mode it excludes is a
build that deletes yesterday's numbers, falls over on row four hundred, and reports an error over a
table that is now empty. Every refusal below therefore checks the table's contents afterwards, not
only the exception.

**The output port is a pass-through of the input, not a receipt.** The contract requires at least one
output port and refuses a zero-column one, so a sink has to produce something, and there are two
candidates. A receipt -- one row naming the table and the count -- loses the data, so a flow that
wants to write the same set twice or assert on it afterwards has to read it back; it introduces a
second vocabulary of column names that every consuming step must learn; and it cannot be documented
honestly, because the interesting fields of a receipt (when it ran, how long it took, what the table
holds now) either drift between runs or make a worked example unrepeatable, and ``check_tool``
compares a documented output by value. A pass-through keeps the sink composable, keeps "is this step
terminal?" a property of the flow graph rather than of the tool, and loses nothing: the row count
written is the height of the frame that came in, precisely because the write is all-or-nothing.

**The mode is written down, with no default.** ``replace`` and ``append`` differ in whether running
the flow twice is safe, and a default would silently pick one of them: an author who meant to
accumulate would find the history deleted on the first run, and an author who meant to rebuild would
find the figures doubled on the second. Neither is safe to assume, so the word is required.

**The tool never creates a table.** A table created from whatever the frame happened to infer this
run has types decided by one day's data, no key, no not-null and no index -- the very constraints
that make truncate-and-load safe -- and a typo in ``table`` would quietly create a second table,
report success, and leave the real one going stale. DDL belongs in a migration somebody reviewed, so
there is no ``create`` field to turn on.

**A frame with no rows stops the run unless the flow says otherwise.** A replace of nothing empties
the table, which is the loudest way an upstream step that matched nothing can be silent -- the same
reason AssertRows fails on an empty frame -- so ``allow_empty`` is how a branch that is legitimately
sometimes empty says so in writing.

**The rows must supply exactly the table's columns.** An extra column would be dropped, losing data
the author thought they wrote; a missing one would be filled with a null or a default, leaving a
column of nothing where a figure belongs. Order need not match, because the load names its columns.

Everything here runs against SQLite, in a temporary file or in memory, so the suite needs no server
and no credentials.
"""

import os
import sqlite3
import sys
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    YAML_UNSAFE_VALUES,
    ConfigError,
    DataError,
    Ports,
    Tool,
    ToolError,
    check_tool,
    run_tool,
)
from dantron_etl.tools.write_table import WriteTable

_VARIABLE = "ETL_TEST_TARGET_CONNECTION"

# A recognisable stand-in for a credential. Every refusal is checked for its absence.
_SECRET = "p4ssw0rd-not-for-a-log"

_BEFORE_TARGET = [(1, "first"), (2, "second")]
_BEFORE_STRICT = [(1, 1.0), (2, 2.0)]


def _seed(database: Path) -> None:
    """A target with a key, a target with a not-null column, and a bystander that must not move."""
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("pragma synchronous = off")  # built once per session as a template
        connection.execute("create table target (key integer primary key, label text)")
        connection.execute("create table strict (key integer primary key, amount real not null)")
        connection.execute("create table bystander (key integer)")
        connection.executemany("insert into target values (?, ?)", _BEFORE_TARGET)
        connection.executemany("insert into strict values (?, ?)", _BEFORE_STRICT)
        connection.execute("insert into bystander values (7)")
        connection.commit()


def _contents(database: Path, table: str) -> list[tuple[Any, ...]]:
    """What a table holds, read on a fresh connection with a short timeout.

    The timeout matters: a tool that left an uncommitted transaction open would block a later
    writer, and a test that waited the default five seconds for that would read as a hang rather
    than as the defect it is.
    """
    with closing(sqlite3.connect(database, timeout=1)) as connection:
        return sorted(connection.execute(f"select * from {table}").fetchall())


def _tables(database: Path) -> set[str]:
    """Every table the database holds -- so 'nothing was created' is checkable."""
    with closing(sqlite3.connect(database, timeout=1)) as connection:
        return {row[0] for row in connection.execute("select name from sqlite_master where type = 'table'")}


@pytest.fixture
def database(tmp_path: Path, seeded_sqlite: Any) -> Path:
    """A temporary SQLite file holding the tables the tests write: a copy of one seeded per session."""
    return seeded_sqlite(tmp_path / "store.db", _seed)


@pytest.fixture
def connection(database: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The NAME of an environment variable holding a URI for ``database``; restored on teardown."""
    monkeypatch.setenv(_VARIABLE, f"sqlite:///{database}")
    return _VARIABLE


def _rows() -> pl.DataFrame:
    """The frame a flow hands a sink: two rows, in the target's own columns."""
    return pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})


def _write(connection_field: str, **overrides: Any) -> Mapping[str, Any]:
    """A replace of ``target``, with whatever a case changes."""
    return {"connection": connection_field, "table": "target", "mode": "replace", **overrides}


# ---------------------------------------------------------------------------
# What the table holds afterwards, and what leaves on the port
# ---------------------------------------------------------------------------


class TestWriteTableWritesAndHandsTheRowsOn:
    """The two modes, and the pass-through that makes a sink composable."""

    def test_a_replace_leaves_the_table_holding_exactly_these_rows_and_hands_them_on(
        self,
        connection: str,
        database: Path,
    ) -> None:
        """The old rows are gone, the new rows are there, and the frame leaves unchanged.

        The pass-through is compared with row order checked, because a sink that reorders the rows
        it was handed would make every downstream step's idea of the data differ from the sink's.
        """
        frame = _rows()

        result = run_tool(WriteTable(), Ports({"rows": frame}), _write(connection))

        assert _contents(database, "target") == [(3, "third"), (4, "fourth")]
        assert result.names == ("rows",)
        assert_frame_equal(result["rows"], _rows(), check_row_order=True)
        assert_frame_equal(frame, _rows(), check_row_order=True)  # the input is read-only
        assert _contents(database, "bystander") == [(7,)], "nothing else in the database moves"

    def test_an_append_adds_to_what_is_there_without_removing_it(self, connection: str, database: Path) -> None:
        """The other mode, and the reason it has to be asked for: the prior rows stay."""
        result = run_tool(WriteTable(), Ports({"rows": _rows()}), _write(connection, mode="append"))

        assert _contents(database, "target") == [(1, "first"), (2, "second"), (3, "third"), (4, "fourth")]
        assert_frame_equal(result["rows"], _rows(), check_row_order=True)

    def test_columns_in_another_order_are_written_by_name(self, connection: str, database: Path) -> None:
        """A frame's column order is nobody's contract, so the load names its columns.

        If the values were bound positionally, this write would put the label in the key column --
        which in this table would be caught by a type, and in a table of two text columns would not
        be caught at all.
        """
        frame = pl.DataFrame({"label": ["third", "fourth"], "key": [3, 4]})

        run_tool(WriteTable(), Ports({"rows": frame}), _write(connection))

        assert _contents(database, "target") == [(3, "third"), (4, "fourth")]


# ---------------------------------------------------------------------------
# The one that matters: a failed write leaves the previous figures standing
# ---------------------------------------------------------------------------


class TestWriteTableIsAllOfTheRowsOrNoneOfThem:
    """A replace that fails part way must not leave the table empty."""

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
                pl.DataFrame({"key": [3, 3], "label": ["third", "again"]}),
                _BEFORE_TARGET,
                id="two_rows_claiming_the_same_key",
            ),
        ],
    )
    def test_a_write_that_fails_part_way_leaves_the_prior_contents_intact(
        self,
        table: str,
        frame: pl.DataFrame,
        before: list[tuple[Any, ...]],
        connection: str,
        database: Path,
    ) -> None:
        """The emptying and the loading are one transaction, or a failed build costs a table.

        Both cases fail on the *second* row, after the delete and after one successful insert, which
        is exactly where an implementation that committed the delete on its own would already have
        thrown yesterday's figures away. The refusal has to name the table, has to belong to this
        library's error family, and must not be a configuration fault -- there is nothing wrong with
        the file, and sending the reader there wastes the night.

        The last assertion is the other half of a transaction being closed: a tool that raised
        without rolling back would leave a write lock behind and block the next writer.
        """
        with pytest.raises(ToolError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(connection, table=table))

        assert not isinstance(caught.value, ConfigError), (
            f"the file is fine; the rows are not, so this must not send the reader to the configuration: {caught.value}"
        )
        assert table in str(caught.value), f"the refusal must name the table it was writing: {caught.value}"
        assert _contents(database, table) == before, "a failed write must leave the previous figures standing"

        with closing(sqlite3.connect(database, timeout=1)) as probe:
            probe.execute("insert into bystander values (8)")
            probe.commit()


# ---------------------------------------------------------------------------
# Refusals, each checked against the table it must not have touched
# ---------------------------------------------------------------------------


class TestWriteTableRefusals:
    """Everything this tool refuses, and the fact that the table is untouched when it does."""

    @pytest.mark.parametrize(
        ("frame", "offending"),
        [
            pytest.param(
                pl.DataFrame({"key": [3], "label": ["third"], "extra": [1]}),
                "extra",
                id="a_column_the_table_has_not_got_would_be_dropped_silently",
            ),
            pytest.param(
                pl.DataFrame({"key": [3]}),
                "label",
                id="a_column_the_rows_do_not_carry_would_be_filled_with_nothing",
            ),
            pytest.param(
                pl.DataFrame({"key": [3], "Label": ["third"]}),
                "Label",
                id="a_name_differing_only_in_case_is_not_a_match",
            ),
        ],
    )
    def test_the_rows_must_supply_exactly_the_table_s_columns(
        self,
        frame: pl.DataFrame,
        offending: str,
        connection: str,
        database: Path,
    ) -> None:
        """A disagreement is refused before anything is emptied, and both lists are shown.

        Showing what the rows carry beside what the table has is what turns the refusal into the
        answer rather than the complaint -- the two mistakes made here are a misspelling and a
        column somebody added upstream, and both are obvious the moment the lists sit together.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(WriteTable(), Ports({"rows": frame}), _write(connection))

        message = str(caught.value)
        assert offending in message, f"the refusal must name the column that disagrees: {message}"
        assert "key" in message, f"the refusal must show the columns on both sides: {message}"
        assert _contents(database, "target") == _BEFORE_TARGET, "nothing may be emptied before this is checked"

    @pytest.mark.parametrize("mode", ["replace", "append"])
    def test_an_empty_frame_stops_the_run_and_leaves_the_table_alone(
        self,
        mode: str,
        connection: str,
        database: Path,
    ) -> None:
        """A branch that matched nothing is the loudest silence a nightly build has.

        A data fault, not a configuration one: the file is right and the rows are missing, so the
        reader's next move is to look at what the flow produced. Refused in both modes, because an
        upstream step that produced nothing is worth stopping for whichever way this sink was going
        to write it.
        """
        empty = pl.DataFrame(schema={"key": pl.Int64, "label": pl.String})

        with pytest.raises(DataError) as caught:
            run_tool(WriteTable(), Ports({"rows": empty}), _write(connection, mode=mode))

        assert "allow_empty" in str(caught.value), f"the refusal must say how to accept it: {caught.value}"
        assert _contents(database, "target") == _BEFORE_TARGET

    @pytest.mark.parametrize(
        ("mode", "after"),
        [
            pytest.param("replace", [], id="a_replace_of_nothing_empties_the_table"),
            pytest.param("append", _BEFORE_TARGET, id="an_append_of_nothing_changes_nothing"),
        ],
    )
    def test_an_empty_frame_is_written_only_when_the_flow_says_it_may_be(
        self,
        mode: str,
        after: list[tuple[Any, ...]],
        connection: str,
        database: Path,
    ) -> None:
        """``allow_empty`` is the flow saying in writing that this branch may be empty."""
        empty = pl.DataFrame(schema={"key": pl.Int64, "label": pl.String})

        result = run_tool(WriteTable(), Ports({"rows": empty}), _write(connection, mode=mode, allow_empty=True))

        assert _contents(database, "target") == after
        assert result["rows"].columns == ["key", "label"], "an output port carries its schema with no rows"

    @pytest.mark.parametrize(
        ("overrides", "named"),
        [
            pytest.param({"table": "no_such_table"}, "no_such_table", id="an_unqualified_name"),
            pytest.param(
                {"schema": "elsewhere", "table": "no_such_table"},
                "elsewhere.no_such_table",
                id="a_name_qualified_by_its_schema",
            ),
        ],
    )
    def test_a_table_that_is_not_there_is_refused_and_no_table_is_created(
        self,
        overrides: Mapping[str, Any],
        named: str,
        connection: str,
        database: Path,
    ) -> None:
        """A sink may not run DDL, so a target that is not there is an author's mistake, by name.

        A tool that created what it could not find would turn a typo into a second table that
        collects the figures while the real one goes stale, and would decide the types, the keys and
        the constraints of a permanent table from one day's data.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(WriteTable(), Ports({"rows": _rows()}), _write(connection, **overrides))

        assert named in str(caught.value), f"the refusal must name the table it looked for: {caught.value}"
        assert _tables(database) == {"target", "strict", "bystander"}, "no table may be created"

    @pytest.mark.parametrize(
        ("overrides", "offending"),
        [
            pytest.param({"table": "target; drop table bystander"}, "target; drop table bystander", id="a_statement"),
            pytest.param({"table": 'target"'}, 'target"', id="a_quote_in_the_table"),
            pytest.param({"table": "two words"}, "two words", id="a_space_in_the_table"),
            pytest.param({"table": "main.target"}, "main.target", id="a_dot_belongs_in_the_schema_field"),
            pytest.param({"schema": "main; drop table bystander"}, "main; drop table bystander", id="in_the_schema"),
        ],
    )
    def test_a_name_that_could_carry_sql_is_refused_before_anything_is_written(
        self,
        overrides: Mapping[str, Any],
        offending: str,
        connection: str,
        database: Path,
    ) -> None:
        """A name is spliced into a statement and cannot be bound, so only a plain name is allowed.

        This is where the absence of a ``query`` field earns its keep: the whole grammar a sink
        accepts is one name per field, so refusing anything else is a two-line rule rather than a
        parser nobody in this library wants to own.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(WriteTable(), Ports({"rows": _rows()}), _write(connection, **overrides))

        assert offending in str(caught.value), f"the refusal must quote the name it refused: {caught.value}"
        assert _contents(database, "target") == _BEFORE_TARGET
        assert "bystander" in _tables(database), "nothing may be executed on the strength of a bad name"

    @pytest.mark.parametrize(
        ("value", "named"),
        [
            pytest.param(
                f"postgresql://writer:{_SECRET}@host:5432/store",
                ("environment variable",),
                id="the_connection_itself_in_the_field",
            ),
            pytest.param(
                "ETL_TEST_VARIABLE_THAT_IS_NOT_SET",
                ("ETL_TEST_VARIABLE_THAT_IS_NOT_SET",),
                id="a_variable_the_environment_has_not_got",
            ),
        ],
    )
    def test_the_connection_field_names_a_variable_and_no_refusal_repeats_its_value(
        self,
        value: str,
        named: tuple[str, ...],
        connection: str,
        database: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same ruling as the source's, and it matters more here: this end holds a write.

        A definition file names a variable and the environment holds the secret, so a definition is
        safe to commit and safe to hand to another client as a template. A refusal that quoted the
        value would move the credential out of the file and into the log.
        """
        monkeypatch.delenv("ETL_TEST_VARIABLE_THAT_IS_NOT_SET", raising=False)

        with pytest.raises(ConfigError) as caught:
            run_tool(WriteTable(), Ports({"rows": _rows()}), _write(value))

        message = str(caught.value)
        assert _SECRET not in message, f"a refusal must not repeat a credential: {message}"
        for word in named:
            assert word in message, f"the refusal should mention {word!r}: {message}"
        assert _contents(database, "target") == _BEFORE_TARGET

    def test_a_driver_the_environment_has_not_got_is_named_in_the_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """This library declares no database driver, so a scheme needing one says what to install."""
        for package in ("psycopg", "adbc_driver_postgresql"):
            monkeypatch.setitem(sys.modules, package, None)
        monkeypatch.setenv(_VARIABLE, f"postgresql://writer:{_SECRET}@host:5432/store")

        with pytest.raises(ConfigError) as caught:
            run_tool(WriteTable(), Ports({"rows": _rows()}), _write(_VARIABLE))

        message = str(caught.value)
        assert "install" in message.lower(), f"the refusal must say what to do: {message}"
        assert any(package in message for package in ("psycopg", "adbc_driver_postgresql")), (
            f"the refusal must name a package that would carry this scheme: {message}"
        )
        assert _SECRET not in message, f"a refusal must not repeat a credential: {message}"


# ---------------------------------------------------------------------------
# The declaration and the catalog
# ---------------------------------------------------------------------------


class TestWriteTableInTheCatalog:
    """Declared, described, documented with examples that actually write, and resolvable by name."""

    def test_the_mode_has_to_be_written_down_and_no_field_offers_to_create_a_table(
        self,
        connection: str,
        database: Path,
    ) -> None:
        """Five fields, one of them required with no default, and deliberately no sixth.

        ``mode`` carries no default because the two values differ in whether re-running the flow is
        safe, and a default would choose one of them silently. Both spellings have to survive a
        definition file authored in YAML, which is why they are words rather than a boolean.
        """
        template = WriteTable().config_template()

        assert set(template) == {"connection", "schema", "table", "mode", "allow_empty"}
        assert "create" not in template, "a sink that ran DDL would decide a permanent schema from one run"
        assert template["mode"].required is True
        assert template["mode"].default is None, "neither mode is safe to assume"
        assert tuple(template["mode"].enum or ()) == ("replace", "append")
        assert not set(template["mode"].enum or ()) & YAML_UNSAFE_VALUES
        assert template["allow_empty"].default is False
        assert template["table"].required is True
        assert template["connection"].required is True

        with pytest.raises(ConfigError) as caught:
            run_tool(
                WriteTable(),
                Ports({"rows": _rows()}),
                {"connection": connection, "table": "target"},
            )

        assert "mode" in str(caught.value)
        assert _contents(database, "target") == _BEFORE_TARGET

    def test_write_table_is_a_sink_that_still_hands_its_rows_on(self) -> None:
        """One input, one output of the same name, and a category that says what it is.

        The output port is the ruling several other sinks will copy: a pass-through, not a receipt.
        The contract refuses a tool with no output port and refuses an output with no columns, so a
        sink must produce something -- and the only thing it can produce that neither loses the data
        nor invents a second vocabulary is the rows it just wrote.
        """
        tool = WriteTable()

        assert isinstance(tool, Tool)
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",), "a pass-through, so a sink can sit mid-flow"
        assert tool.category == "output"
        assert tool.category in CATEGORIES

    def test_write_table_conforms_and_its_documented_examples_run_twice_in_memory(
        self,
        environment_restored: None,
    ) -> None:
        """Conformance runs every worked example, and a sink's examples really do write.

        No exemption for the two tools that need a database: the examples run against an in-memory
        SQLite database the tool's own documentation builds, so reading the catalog costs no server,
        no credential and no file on disk. Run twice on purpose -- a documented sink that could only
        be checked once would be a documented sink nobody could re-check, which is what a receipt
        carrying a count of what the table now holds would have made it.
        """
        tool = WriteTable()

        check_tool(tool)
        check_tool(tool)

        assert tool.name == "write_table"
        examples = tool.help().examples
        assert len(examples) >= 2, "one example per mode: the difference is the thing to document"
        assert {str(example.configuration["mode"]) for example in examples} == {"replace", "append"}
        assert all(example.check_row_order for example in examples), (
            "a pass-through preserves order, and that is the one property a reader most wants checked"
        )
        for example in examples:
            assert example.setup is not None, (
                "an example that needs a database declares it, rather than having help() build one: "
                "help() is a description, and the catalog page generator calls it for every tool"
            )
            with example.setup():
                uri = os.environ[str(example.configuration["connection"])]
                assert "mode=memory" in uri, f"a documented example must need no file and no server: {uri}"

        assert "write_table" in REGISTRY
        assert isinstance(REGISTRY.get("write_table"), WriteTable)
