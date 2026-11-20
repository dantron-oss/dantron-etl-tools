"""The database plumbing: quoting column names, keeping a credential out of every message, and
naming the real reason a table could not be read.

Run against SQLite, and against a stand-in driver for what only Postgres would show (the COPY
statement, a connection error that quotes the password). No server is needed.
"""

import sqlite3
import sys
import traceback
import types
from contextlib import closing
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl import db
from dantron_etl.authoring import ConfigError, Ports, run_tool
from dantron_etl.registry import REGISTRY

_VARIABLE = "DB_SAFETY_TEST_URI"

# A connection string whose password the driver below echoes, as one real driver does for a password
# holding a space. Kept in a constant, so a traceback's quoted source line does not hold it.
_LEAKY_URI = "postgresql://alice:s3cret PW@db.example:5433/ledger"


def _table(path: Path) -> Path:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute('create table awkward ("order" integer, "my col" text, "MixedCase" text)')
        connection.commit()
    return path


class TestAColumnNameThatNeedsQuoting:
    @pytest.mark.parametrize("mode", ["append", "replace"])
    def test_is_written_by_write_table(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
        database = _table(tmp_path / "store.db")
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{database}")
        frame = pl.DataFrame({"order": [1, 2], "my col": ["a", "b"], "MixedCase": ["x", "y"]})
        run_tool(REGISTRY.get("write_table"), Ports({"rows": frame}), {"connection": _VARIABLE, "table": "awkward", "mode": mode})
        with closing(sqlite3.connect(database)) as connection:
            assert connection.execute('select "order", "my col", "MixedCase" from awkward order by 1').fetchall() == [
                (1, "a", "x"),
                (2, "b", "y"),
            ]

    def test_the_copy_statement_quotes_every_column(self) -> None:
        seen: list[str] = []

        class _Stream:
            def __enter__(self) -> "_Stream":
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def write_row(self, row: tuple[Any, ...]) -> None:
                return None

        class _Cursor:
            def __enter__(self) -> "_Cursor":
                return self

            def __exit__(self, *_: Any) -> None:
                return None

            def copy(self, statement: str) -> _Stream:
                seen.append(statement)
                return _Stream()

        class _Connection:
            def cursor(self) -> _Cursor:
                return _Cursor()

        fake_psycopg = types.SimpleNamespace(errors=types.SimpleNamespace(IntegrityError=Exception), Error=Exception)
        loader = db._PostgresLoader(fake_psycopg)  # type: ignore[attr-defined]
        loader.load_rows(_Connection(), "s.t", ["order", 'say "hi"'], [(1, "a")])
        assert seen == ['COPY s.t ("order", "say ""hi""") FROM STDIN']

    def test_the_insert_statement_quotes_every_column(self) -> None:
        statement = db.driver_for(sqlite3.connect(":memory:"))._insert_statement("t", ["order", "a b"])
        assert statement == 'INSERT INTO t ("order", "a b") VALUES (?, ?)'


class TestAConnectionErrorNeverShowsTheCredential:
    @pytest.fixture
    def leaky_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def connect(uri: str) -> None:
            raise RuntimeError(f'unexpected spaces found in "s3cret PW" while reading {uri}')

        monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(connect=connect))

    def test_the_message_names_where_but_not_the_password_or_user(self, leaky_driver: None) -> None:
        with pytest.raises(ConfigError) as raised:
            with db.open_connection(_LEAKY_URI):
                pass
        message = str(raised.value)
        assert "'ledger'" in message and "'db.example'" in message and "5433" in message and "RuntimeError" in message
        assert "s3cret" not in message and "alice" not in message

    def test_nothing_chained_to_the_refusal_carries_it_either(self, leaky_driver: None) -> None:
        with pytest.raises(ConfigError) as raised:
            with db.open_connection(_LEAKY_URI):
                pass
        assert raised.value.__cause__ is None and raised.value.__context__ is None
        shown = "".join(traceback.format_exception(raised.value))
        assert "s3cret" not in shown

    def test_a_uri_that_cannot_even_be_parsed_still_refuses_safely(self, leaky_driver: None) -> None:
        with pytest.raises(ConfigError) as raised:
            with db.open_connection("postgresql://alice:pw@host:notaport/db"):
                pass
        assert "pw" not in str(raised.value).replace("postgresql", "")


class TestAFailedReadSaysWhyIfTheTableIsThere:
    def test_a_missing_table_is_not_found(self, tmp_path: Path) -> None:
        database = _table(tmp_path / "store.db")
        with closing(sqlite3.connect(database)) as connection:
            with pytest.raises(ConfigError) as raised:
                db.probe_columns(connection, "nowhere")
        assert "table 'nowhere' was not found" in str(raised.value)

    def test_a_file_that_is_not_a_database_says_so(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.db"
        broken.write_bytes(b"this is not a sqlite file at all" * 10)
        with closing(sqlite3.connect(broken)) as connection:
            with pytest.raises(ConfigError) as raised:
                db.probe_columns(connection, "orders")
        message = str(raised.value)
        assert "could not read table 'orders'" in message and "not a database" in message
        assert "was not found" not in message

    def test_a_table_that_is_there_gives_its_columns(self, tmp_path: Path) -> None:
        with closing(sqlite3.connect(_table(tmp_path / "store.db"))) as connection:
            assert db.probe_columns(connection, "awkward") == ("order", "my col", "MixedCase")


class TestSmallerConnectionRules:
    @pytest.mark.parametrize("name", ["orders\n", "orders;", "1orders", ""])
    def test_an_identifier_with_anything_after_it_is_refused(self, name: str) -> None:
        with pytest.raises(ConfigError):
            db.validate_identifier(name, "table")

    def test_an_environment_name_with_a_newline_is_refused(self) -> None:
        with pytest.raises(ConfigError):
            db.resolve_connection_uri("DATABASE_URL\n", "connection")

    def test_a_file_uri_with_no_mode_does_not_create_a_missing_database(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.db"
        with pytest.raises(ConfigError) as raised:
            with db.open_connection(f"file:{missing}"):
                pass
        assert "not created" in str(raised.value)
        assert not missing.exists()

    def test_a_file_uri_to_an_existing_database_still_opens(self, tmp_path: Path) -> None:
        database = _table(tmp_path / "store.db")
        with db.open_connection(f"file:{database}") as connection:
            assert db.probe_columns(connection, "awkward")

    def test_an_in_memory_file_uri_is_opened_as_written(self) -> None:
        with db.open_connection("file:db_safety_memory?mode=memory&cache=shared") as connection:
            connection.execute("create table t (a integer)")

    def test_an_unknown_scheme_lists_every_scheme_accepted(self) -> None:
        with pytest.raises(ConfigError) as raised:
            with db.open_connection("mysql://host/db"):
                pass
        for scheme in ("sqlite", "file", "postgresql", "postgres"):
            assert scheme in str(raised.value)


# A password nobody would type by accident, in pieces, so a test can ask that no piece of it shows.
_PIECES = ("Kw7qZ", "Rv3xT", "Lp9mJ")


def _no_piece_of_the_password(text: str) -> None:
    for piece in _PIECES:
        assert piece not in text, f"part of the password shows: {text}"


class TestWhereAConnectionPoints:
    """describe_connection: the database an environment variable names, in plain words, never the password."""

    @pytest.fixture(autouse=True)
    def _unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(_VARIABLE, raising=False)

    def _described(self, monkeypatch: pytest.MonkeyPatch, value: str) -> str:
        monkeypatch.setenv(_VARIABLE, value)
        return db.describe_connection(_VARIABLE)

    @pytest.mark.parametrize(
        ("password", "written"),
        [
            pytest.param(f"{_PIECES[0]}{_PIECES[1]}", f"{_PIECES[0]}{_PIECES[1]}", id="plain"),
            pytest.param(f"{_PIECES[0]}:{_PIECES[1]}", f"{_PIECES[0]}:{_PIECES[1]}", id="holding_a_colon"),
            pytest.param(f"{_PIECES[0]}@{_PIECES[1]}", f"{_PIECES[0]}%40{_PIECES[1]}", id="an_at_sign_percent_encoded"),
            pytest.param(f"{_PIECES[0]}/{_PIECES[1]}", f"{_PIECES[0]}%2F{_PIECES[1]}", id="a_slash_percent_encoded"),
            pytest.param(f"{_PIECES[0]}?{_PIECES[1]}", f"{_PIECES[0]}?{_PIECES[1]}", id="holding_a_question_mark"),
        ],
    )
    def test_an_address_names_host_port_and_database_and_never_the_password(
        self, monkeypatch: pytest.MonkeyPatch, password: str, written: str
    ) -> None:
        described = self._described(monkeypatch, f"postgresql://etl_user:{written}@db.example.internal:5433/orders")

        assert "'orders'" in described and "db.example.internal" in described and "5433" in described, described
        assert _VARIABLE in described, "names the variable it read"
        _no_piece_of_the_password(described)
        assert written not in described and password not in described

    def test_a_port_not_stated_is_not_shown_and_the_short_scheme_reads_the_same(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        described = self._described(monkeypatch, f"postgres://etl_user:{_PIECES[0]}@db.example.internal/orders")

        assert "'orders'" in described and "db.example.internal" in described
        assert "port" not in described
        _no_piece_of_the_password(described)

    def test_a_password_given_after_the_address_is_not_shown_either(self, monkeypatch: pytest.MonkeyPatch) -> None:
        value = f"postgresql://db.example.internal:5433/orders?sslmode=require&password={_PIECES[0]}"

        described = self._described(monkeypatch, value)

        assert "'orders'" in described and "db.example.internal" in described
        _no_piece_of_the_password(described)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(f"postgresql://etl_user:{_PIECES[0]}@{_PIECES[1]}@db.example.internal/orders", id="an_at_sign_not_encoded"),
            pytest.param(f"postgresql://etl_user:{_PIECES[0]}/{_PIECES[1]}@db.example.internal/orders", id="a_slash_not_encoded"),
            pytest.param(f"postgresql://etl_user:{_PIECES[0]}@{_PIECES[1]}?a=b@host/orders", id="an_at_sign_then_a_query"),
            pytest.param(f"postgresql://etl_user:{_PIECES[0]}@host:{_PIECES[1]}/orders", id="a_port_that_is_not_a_number"),
            pytest.param(f"postgresql://etl_user:{_PIECES[0]}%ZZ@host/orders", id="a_broken_percent_code"),
            pytest.param(f"postgresql://etl_user:{_PIECES[0]}@host/orders?{_PIECES[1]}", id="a_query_with_no_equals"),
        ],
    )
    def test_an_address_that_cannot_be_read_safely_says_so_and_shows_none_of_it(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """Read the way the Postgres driver reads it, each of these puts part of the password in the host or the name."""
        described = self._described(monkeypatch, value)

        assert "cannot be read" in described, described
        assert _VARIABLE in described
        _no_piece_of_the_password(described)
        assert "etl_user" not in described and "orders" not in described

    def test_key_value_form_names_host_port_and_database_and_says_a_run_cannot_open_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        value = f"host=db.example.internal port=5433 dbname=orders user=etl_user password='{_PIECES[0]} \\' {_PIECES[1]}'"

        described = self._described(monkeypatch, value)

        assert "'orders'" in described and "db.example.internal" in described and "5433" in described, described
        assert "postgresql://" in described, "says how to write it so a run can open it"
        _no_piece_of_the_password(described)

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(f"host=db.example.internal dbname=orders password='{_PIECES[0]} {_PIECES[1]}", id="a_quote_never_closed"),
            pytest.param(f"host=db.example.internal dbname orders password={_PIECES[0]}", id="a_key_with_no_equals"),
        ],
    )
    def test_key_value_form_that_cannot_be_read_says_so_and_shows_none_of_it(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        described = self._described(monkeypatch, value)

        assert "cannot be read" in described, described
        _no_piece_of_the_password(described)

    def test_a_sqlite_address_names_the_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        described = self._described(monkeypatch, f"sqlite:///{tmp_path / 'store.db'}")

        assert str(tmp_path / "store.db") in described and "SQLite" in described

    def test_an_address_of_another_kind_is_not_echoed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        described = self._described(monkeypatch, f"mysql://etl_user:{_PIECES[0]}@db.example.internal/orders")

        assert "postgresql://" in described, "says which addresses a run opens"
        _no_piece_of_the_password(described)
        assert "db.example.internal" not in described

    def test_a_variable_that_is_not_set_says_so_instead_of_failing(self) -> None:
        described = db.describe_connection(_VARIABLE)

        assert _VARIABLE in described and "not set" in described

    @pytest.mark.parametrize(
        "name",
        [
            pytest.param(f"postgresql://etl_user:{_PIECES[0]}@db.example.internal/orders", id="the_address_itself"),
            pytest.param(None, id="nothing"),
        ],
    )
    def test_a_setting_that_is_not_a_variable_name_is_not_echoed(self, name: Any) -> None:
        described = db.describe_connection(name)

        assert "environment variable" in described, described
        _no_piece_of_the_password(described)
        assert "db.example.internal" not in described

    @pytest.mark.parametrize("value", [f"postgresql://etl:{_PIECES[0]}@{_PIECES[0]}.example/orders", f"postgresql://etl:{_PIECES[0]}@host/{_PIECES[0]}"])
    def test_a_password_also_written_as_the_host_or_database_shows_neither(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """A test server often uses one word for user, password and database. Naming the database would name the password."""
        described = self._described(monkeypatch, value)

        assert "password is also written" in described, described
        _no_piece_of_the_password(described)
