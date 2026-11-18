"""Tests for the ReadTable tool: a named table onto one port, and a credential nobody committed.

ReadTable is the catalog's first source. It declares no input ports and one output port, ``rows``,
and everything interesting about it is a decision about *where the connection comes from* and *what
a step is allowed to name*. Six rulings are pinned below, each because the alternative fails
quietly.

**The configuration names an environment variable; the environment holds the secret.** The
``connection`` field carries the NAME of a variable, never a URI, so a pipeline definition is safe
to commit and safe to hand to another client as a template. Two mistakes follow from that and both
are pinned: a field holding the connection *itself* is refused with a message that deliberately
does not repeat it, and a variable that is not set is refused by name. No refusal anywhere in this
tool may repeat the value it read -- these messages are read in a log, and a log is not a vault.

**The environment, and never a file.** In an orchestrator or in CI there is no ``.env`` to parse;
the secret arrives from a secret store. A tool that insisted on a file would fail there, and a tool
that parsed one would bake a single project's layout into a library that ships elsewhere. So a
variable present in a ``.env`` file and absent from the environment is *absent*, and the test below
writes such a file to prove it.

**A named table, and no ``query`` field.** A ``query`` field would be arbitrary SQL arriving from a
configuration file: nothing could lift the column references out of it to pre-flight them, its
dialect would tie a portable definition to one database, and it would duplicate Filter and
SelectColumns in a second language that none of this library's checks can read -- the same reason
nothing here ever calls ``eval`` on a configuration. So a step names a schema and a table, each held
to a plain-identifier grammar, and that is what closes the injection surface: an identifier cannot
be bound as a parameter, so the only defence is refusing anything that is not one, which is cheap
when the grammar is one name and impossible when it is a whole query.

**No driver is declared.** The library's dependencies stay polars plus the standard library, so
sqlite works with nothing installed and any other scheme imports a driver the consumer supplied --
lazily, by scheme, with a refusal that names the package to install when they have not.

**A table that is not there is a refusal an author can act on**, naming the table, in this library's
error family -- not the driver's own exception arriving from a library the author has never heard
of. The same goes for a sqlite path that is not there: a mistyped path must not be created and then
read as an empty database, which would report success having read nothing.

**Rows arrive in no order.** A database promises none without an ``ORDER BY``, and the standing rule
in this library is that a step needing meaningful order puts a Sort in front of it. That ruling is
pinned where it can actually be checked -- no worked example of this tool may claim row order --
and every assertion below sorts before it compares.

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

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    PortError,
    Ports,
    Tool,
    check_tool,
    run_tool,
)
from dantron_etl.tools.read_table import ReadTable

# The name of the variable the tests export. A name, because that is all a configuration ever holds.
_VARIABLE = "ETL_TEST_SOURCE_CONNECTION"

# A recognisable stand-in for a credential. Every refusal is checked for its absence.
_SECRET = "p4ssw0rd-not-for-a-log"

_ENTRIES = "create table entries (key integer, label text, amount real, flag integer)"
_BLANK = "create table blank (a integer, b text)"


def _seed(database: Path) -> None:
    """One database with a table carrying every shape a read has to preserve, and an empty one."""
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("pragma synchronous = off")  # built once per session as a template
        connection.execute(_ENTRIES)
        connection.execute(_BLANK)
        connection.executemany(
            "insert into entries values (?, ?, ?, ?)",
            [(1, "first", 10.5, 1), (2, None, -3.0, 0), (3, "third", 0.0, None)],
        )
        connection.commit()


@pytest.fixture
def database(tmp_path: Path, seeded_sqlite: Any) -> Path:
    """A temporary SQLite file holding the tables the tests read: a copy of one seeded per session."""
    return seeded_sqlite(tmp_path / "store.db", _seed)


@pytest.fixture
def connection(database: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The NAME of an environment variable holding a URI for ``database``; restored on teardown."""
    monkeypatch.setenv(_VARIABLE, f"sqlite:///{database}")
    return _VARIABLE


@pytest.fixture
def secretive_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeded_sqlite: Any) -> str:
    """A connection whose URI carries something no refusal may ever repeat."""
    path = seeded_sqlite(tmp_path / f"{_SECRET}.db", _seed)
    monkeypatch.setenv(_VARIABLE, f"sqlite:///{path}")
    return _VARIABLE


def _read(connection_field: str, **overrides: Any) -> Mapping[str, Any]:
    """The configuration for reading ``entries``, with whatever a case changes."""
    return {"connection": connection_field, "table": "entries", **overrides}


# ---------------------------------------------------------------------------
# What arrives on the port
# ---------------------------------------------------------------------------


class TestReadTableReadsATable:
    """The rows the table holds, with the types the database gave them, on one declared port."""

    def test_a_table_arrives_on_the_rows_port_with_the_types_the_database_gave(self, connection: str) -> None:
        """Every row, every column, and dtypes that are the database's rather than this tool's.

        The ``flag`` column is the point of the dtype assertion: it holds 1, 0 and a missing value,
        which a tool inventing its own types would hand back as a boolean. It comes back as a whole
        number, because that is what the database stores. Sorted before comparing, because a
        database promises no order and this tool adds none.
        """
        result = run_tool(ReadTable(), Ports({}), _read(connection))

        assert result.names == ("rows",), "one declared output port"
        frame = result["rows"]
        assert frame.schema == {
            "key": pl.Int64,
            "label": pl.String,
            "amount": pl.Float64,
            "flag": pl.Int64,
        }
        assert frame.sort("key").rows() == [
            (1, "first", 10.5, 1),
            (2, None, -3.0, 0),
            (3, "third", 0.0, None),
        ]

    def test_a_table_with_no_rows_still_arrives_carrying_its_columns(self, connection: str) -> None:
        """An empty read is a typed, named, zero-row frame -- not a frame with nothing on it.

        The contract refuses an output port with no columns at all, and a downstream step compiles
        its own expressions against this port's schema, so the columns have to survive a read that
        found nothing. The dtypes of an empty read are deliberately not pinned: they are whatever
        the reader could infer with no row in front of it, which is the database's business rather
        than this tool's.
        """
        result = run_tool(ReadTable(), Ports({}), _read(connection, table="blank"))

        frame = result["rows"]
        assert frame.height == 0
        assert frame.columns == ["a", "b"]
        assert frame.width == 2, "an output port must carry its schema even when it carries no rows"


# ---------------------------------------------------------------------------
# The connection: a name in the file, the value in the environment
# ---------------------------------------------------------------------------


class TestReadTableResolvesItsConnectionThroughTheEnvironment:
    """A definition file names a variable. Nothing else, and nothing echoed back."""

    @pytest.mark.parametrize(
        ("value", "named"),
        [
            pytest.param(
                f"postgresql://reader:{_SECRET}@host:5432/store",
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
    def test_the_connection_field_names_a_variable_and_the_refusal_says_so_without_echoing_a_value(
        self,
        value: str,
        named: tuple[str, ...],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both halves of the ruling, and the reason it exists.

        A field holding the connection itself is the mistake that puts a credential into a
        committed file; it is refused, and the refusal must not quote the value, because doing so
        would move the credential from the file into the log. A variable that is not set is a
        configuration fault the author can act on, so its refusal names the variable it looked for.
        """
        monkeypatch.setenv(_VARIABLE, "sqlite:///unused.db")
        monkeypatch.delenv("ETL_TEST_VARIABLE_THAT_IS_NOT_SET", raising=False)

        with pytest.raises(ConfigError) as caught:
            run_tool(ReadTable(), Ports({}), _read(value))

        message = str(caught.value)
        assert _SECRET not in message, f"a refusal must not repeat a credential: {message}"
        for word in named:
            assert word in message, f"the refusal should mention {word!r}: {message}"

    def test_a_variable_that_exists_only_in_a_dotenv_file_is_still_absent(
        self,
        database: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The tool reads the environment and must not parse a file to fill it.

        In an orchestrator or in CI there is no such file -- the secret arrives from a secret store
        -- so a tool that read one would work here and fail there, and a tool that knew where to
        look for one would carry this project's layout into every other repository the library
        ships to. Loading a file into the environment is one line the consumer writes.
        """
        variable = "ETL_TEST_DOTENV_ONLY_CONNECTION"
        monkeypatch.delenv(variable, raising=False)
        (tmp_path / ".env").write_text(f"{variable}=sqlite:///{database}\n")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ConfigError) as caught:
            run_tool(ReadTable(), Ports({}), _read(variable))

        assert variable in str(caught.value)

    @pytest.mark.parametrize(
        ("value", "named"),
        [
            pytest.param(f"mysql://reader:{_SECRET}@host:3306/store", ("mysql", "sqlite"), id="a_scheme_it_has_not"),
            pytest.param(f"/var/lib/{_SECRET}.db", ("sqlite",), id="no_scheme_at_all"),
        ],
    )
    def test_a_uri_whose_scheme_the_library_does_not_know_is_refused_by_scheme(
        self,
        value: str,
        named: tuple[str, ...],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The refusal names the scheme and the schemes it does know, and repeats nothing else.

        Naming the schemes it knows is what turns the refusal into the answer, exactly as listing
        the columns present does for a misspelled column elsewhere in this library.
        """
        monkeypatch.setenv(_VARIABLE, value)

        with pytest.raises(ConfigError) as caught:
            run_tool(ReadTable(), Ports({}), _read(_VARIABLE))

        message = str(caught.value)
        assert _SECRET not in message, f"a refusal must not repeat a credential: {message}"
        for word in named:
            assert word in message, f"the refusal should mention {word!r}: {message}"

    def test_a_driver_the_environment_has_not_got_is_named_in_the_refusal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A scheme needing a driver the consumer never installed must say which package to install.

        This library declares no database driver -- that is the point of something shipping into
        other repositories -- so the import is lazy and by scheme. Left unhandled it surfaces as an
        ImportError from inside a phase name, which tells the reader nothing they can act on.
        """
        for package in ("psycopg", "adbc_driver_postgresql"):
            monkeypatch.setitem(sys.modules, package, None)
        monkeypatch.setenv(_VARIABLE, f"postgresql://reader:{_SECRET}@host:5432/store")

        with pytest.raises(ConfigError) as caught:
            run_tool(ReadTable(), Ports({}), _read(_VARIABLE))

        message = str(caught.value)
        assert "install" in message.lower(), f"the refusal must say what to do: {message}"
        assert any(package in message for package in ("psycopg", "adbc_driver_postgresql")), (
            f"the refusal must name a package that would carry this scheme: {message}"
        )
        assert _SECRET not in message, f"a refusal must not repeat a credential: {message}"


# ---------------------------------------------------------------------------
# What a step may name, and what happens when it names something that is not there
# ---------------------------------------------------------------------------


def _bad_name_cases() -> list[Any]:
    """Configuration overrides whose table or schema is not a plain name."""
    return [
        pytest.param({"table": "entries; drop table entries"}, "entries; drop table entries", id="a_second_statement"),
        pytest.param({"table": 'entries"'}, 'entries"', id="a_quote_in_the_table"),
        pytest.param({"table": "entries'"}, "entries'", id="an_apostrophe_in_the_table"),
        pytest.param({"table": "two words"}, "two words", id="a_space_in_the_table"),
        pytest.param({"table": "other.entries"}, "other.entries", id="a_dot_belongs_in_the_schema_field"),
        pytest.param({"table": "1entries"}, "1entries", id="a_leading_digit"),
        pytest.param({"schema": "other; drop table entries"}, "other; drop table entries", id="a_statement_in_schema"),
    ]


class TestReadTableRefusesWhatItCannotName:
    """The grammar of a name is the whole injection fence, so it is checked before anything opens."""

    @pytest.mark.parametrize(("overrides", "offending"), _bad_name_cases())
    def test_a_name_that_could_carry_sql_is_refused_before_a_connection_is_opened(
        self,
        overrides: Mapping[str, Any],
        offending: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A name is spliced into a statement and cannot be bound, so anything but a name is refused.

        The connection deliberately points at a path that does not exist: if the tool opened it
        first, the refusal would be about the path, and the assertion that the offending name is
        quoted is what proves the order. The path is checked afterwards too, because a tool that
        reached the database at all would have left a file behind.
        """
        missing = tmp_path / "never-opened.db"
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{missing}")

        with pytest.raises(ConfigError) as caught:
            run_tool(ReadTable(), Ports({}), _read(_VARIABLE, **overrides))

        assert offending in str(caught.value), f"the refusal must quote the name it refused: {caught.value}"
        assert not missing.exists(), "nothing should have been opened, let alone created"

    @pytest.mark.parametrize(
        ("overrides", "named"),
        [
            pytest.param({"table": "no_such_table"}, "no_such_table", id="an_unqualified_name"),
            pytest.param(
                {"schema": "other", "table": "no_such_table"},
                "other.no_such_table",
                id="a_name_qualified_by_its_schema",
            ),
        ],
    )
    def test_a_table_that_is_not_there_is_a_refusal_naming_it_rather_than_the_driver_s_own_error(
        self,
        overrides: Mapping[str, Any],
        named: str,
        secretive_connection: str,
    ) -> None:
        """The commonest authoring mistake, answered in this library's words and its error family.

        A ConfigError rather than an execution fault, because the author can act on it: the name is
        wrong, or the target does not have that table yet. The qualified case also pins how the two
        fields compose into one name -- ``schema.table`` -- which is the only place a test can see
        it, since the identifiers are never handed to the database separately.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(ReadTable(), Ports({}), _read(secretive_connection, **overrides))

        message = str(caught.value)
        assert named in message, f"the refusal must name the table it looked for: {message}"
        assert _SECRET not in message, f"a refusal must not repeat the connection string: {message}"

    def test_a_sqlite_path_that_is_not_there_is_refused_rather_than_created(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A mistyped path must not become an empty database that reads as a successful run.

        This is the same defect the library refuses everywhere else it can: a name nothing carries
        quietly creating something new, leaving the thing the author meant untouched and the run
        reporting success having read nothing at all.
        """
        missing = tmp_path / "not-a-database.db"
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{missing}")

        with pytest.raises(ConfigError) as caught:
            run_tool(ReadTable(), Ports({}), _read(_VARIABLE))

        assert not missing.exists(), f"the path must not be created: {caught.value}"


# ---------------------------------------------------------------------------
# The declaration and the catalog
# ---------------------------------------------------------------------------


class TestReadTableInTheCatalog:
    """Declared, described, documented with an example that runs, and resolvable by name."""

    def test_the_template_names_a_table_and_offers_no_query(self) -> None:
        """Three fields, and deliberately not a fourth.

        A ``query`` field is arbitrary SQL from a configuration file: unreadable to every check this
        library has, un-pre-flightable by a loader, and tied to one dialect. Narrowing a read is a
        Filter or a SelectColumns after it, where the columns named are visible to the same
        machinery that checks every other step.
        """
        template = ReadTable().config_template()

        assert set(template) == {"connection", "schema", "table"}
        assert "query" not in template, "arbitrary SQL from a file is the hole this tool exists without"
        assert template["connection"].required is True
        assert template["connection"].non_empty is True
        assert template["table"].required is True
        assert template["table"].non_empty is True
        assert template["schema"].required is False
        assert template["schema"].default is None, "no schema means an unqualified name"

    def test_read_table_declares_no_input_ports_and_refuses_one_it_was_wired(self, connection: str) -> None:
        """A source has nothing upstream, and a flow that wired it one has made a wiring mistake."""
        tool = ReadTable()

        assert isinstance(tool, Tool), "validate_inputs is a required member even with no inputs"
        assert tuple(tool.input_ports) == ()
        assert tuple(tool.output_ports) == ("rows",)

        with pytest.raises(PortError) as caught:
            run_tool(tool, Ports({"rows": pl.DataFrame({"key": [1]})}), _read(connection))

        assert "rows" in str(caught.value)

    def test_read_table_conforms_and_documents_an_example_that_runs_in_memory(
        self,
        environment_restored: None,
    ) -> None:
        """Conformance runs every worked example, and this tool's examples need a real database.

        No exemption: the examples run against an in-memory SQLite database the tool's own
        documentation builds, so reading the catalog costs no server, no credential and no file on
        disk. And no example may claim row order, because the tool cannot keep that promise -- a
        database returns rows in no guaranteed order, and a step that needs one puts a Sort in front
        of it.
        """
        tool = ReadTable()

        check_tool(tool)

        assert tool.name == "read_table"
        assert tool.category == "input"
        assert tool.category in CATEGORIES

        examples = tool.help().examples
        assert examples, "a source with no runnable example is a source nobody can copy"
        assert all(not example.check_row_order for example in examples), (
            "a read promises no row order, so no example of it may claim one"
        )
        for example in examples:
            assert example.setup is not None, (
                "an example that needs a database declares it, rather than having help() build one: "
                "help() is a description, and the catalog page generator calls it for every tool"
            )
            with example.setup():
                uri = os.environ[str(example.configuration["connection"])]
                assert "mode=memory" in uri, f"a documented example must need no file and no server: {uri}"

        assert "read_table" in REGISTRY
        assert isinstance(REGISTRY.get("read_table"), ReadTable)
