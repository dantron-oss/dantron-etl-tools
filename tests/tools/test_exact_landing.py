"""Exact decimals and floats written into a table's numeric, double precision and real columns.

A database rounds a value with more places than a ``numeric(p,s)`` column holds, without a word:
1.235 goes into ``numeric(5,2)`` as 1.24. So ``write_table`` and ``publish_tables`` check every value
before anything is written. A Decimal column goes into ``numeric(p,s)`` only when every value fits s
places and p digits. A Float64 column goes in on the same terms, read to 15 significant digits, as
``round`` reads it, and what is written is that reading. A Decimal column never goes into a float
column. A Float64 column into a float column is written exactly as before.

Each rule runs on SQLite here and on Postgres wherever ``DANTRON_TEST_POSTGRES`` names a server.
SQLite has no exact type: the declared ``numeric(5,2)`` is checked the same way, and the digits are
sent as text, which SQLite then stores by its own rule.
"""

import sqlite3
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl.authoring import DataError, Ports, run_tool
from dantron_etl.db import declared_type_of
from dantron_etl.tools._table_load import numeric_misfits
from dantron_etl.tools.publish_tables import PublishTables
from dantron_etl.tools.write_table import WriteTable

_VARIABLE = "ETL_TEST_EXACT_TARGET"
_SCHEMA = "exact_landing_scratch"

_DDL = (
    "create table priced (key integer primary key, amount numeric(5,2), loose numeric, measure double precision)",
    "create table rated (key integer primary key, rate real)",
)
_BEFORE = [(9, Decimal("1.00"), Decimal("1"), 1.0)]


def _exact(values: list[str | None], places: int) -> pl.Series:
    """Exact decimals, written as text so nothing passes through a float."""
    return pl.Series([None if v is None else Decimal(v) for v in values], dtype=pl.Decimal(38, places))


class _Target:
    def __init__(self, engine: str, schema: str | None, read: Any) -> None:
        self.engine = engine
        self.schema = schema
        self._read = read

    def read(self, table: str) -> list[tuple[Any, ...]]:
        return sorted(self._read(table), key=lambda row: row[0])

    def config(self, table: str = "priced", **overrides: Any) -> dict[str, Any]:
        config: dict[str, Any] = {"connection": _VARIABLE, "table": table, "mode": "replace", **overrides}
        if self.schema is not None:
            config["schema"] = self.schema
        return config


def _seed(connection: Any, prefix: str) -> None:
    for statement in _DDL:
        connection.execute(statement.replace("create table ", f"create table {prefix}"))
    connection.execute(f"insert into {prefix}priced values (9, 1.00, 1, 1.0)")


@pytest.fixture(params=["sqlite", "postgres"])
def target(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Target:
    """The same tables on SQLite, and on Postgres when there is a server to reach."""
    if request.param == "sqlite":
        database = tmp_path / "exact.db"
        with closing(sqlite3.connect(database)) as connection:
            _seed(connection, "")
            connection.commit()
        monkeypatch.setenv(_VARIABLE, f"sqlite:///{database}")

        def read_sqlite(table: str) -> list[tuple[Any, ...]]:
            with closing(sqlite3.connect(database, timeout=1)) as connection:
                return list(connection.execute(f"select * from {table}").fetchall())

        return _Target("sqlite", None, read_sqlite)

    session = request.getfixturevalue("postgres_scratch")(_SCHEMA)
    monkeypatch.setenv(_VARIABLE, request.getfixturevalue("postgres_dsn"))
    for table in ("priced", "rated"):
        session.execute(f"drop table if exists {_SCHEMA}.{table}")
    _seed(session, f"{_SCHEMA}.")

    def read_postgres(table: str) -> list[tuple[Any, ...]]:
        return list(session.execute(f"select * from {_SCHEMA}.{table}").fetchall())

    return _Target("postgres", _SCHEMA, read_postgres)


def _priced(amount: pl.Series, *, measure: list[float | None] | None = None) -> pl.DataFrame:
    height = amount.len()
    return pl.DataFrame(
        {
            "key": list(range(1, height + 1)),
            "amount": amount,
            "loose": _exact([None] * height, 0),
            "measure": pl.Series(measure if measure is not None else [None] * height, dtype=pl.Float64),
        }
    )


def _write(target: _Target, frame: pl.DataFrame, **overrides: Any) -> None:
    run_tool(WriteTable(), Ports({"rows": frame}), target.config(**overrides))


def _as_figures(rows: list[tuple[Any, ...]], column: int) -> list[Decimal | None]:
    """One column of what the table holds, as exact figures whichever engine stored it."""
    return [None if row[column] is None else Decimal(str(row[column])) for row in rows]


# ---------------------------------------------------------------------------
# An exact column into numeric(p,s)
# ---------------------------------------------------------------------------


class TestAnExactColumnGoesIntoNumericOnlyWhenEveryValueFits:
    def test_values_that_fit_the_places_and_the_digits_land(self, target: _Target) -> None:
        frame = _priced(_exact(["1.2300", "-999.9900", "999.9900", None], 4))

        _write(target, frame)

        stored = target.read("priced")
        assert [row[0] for row in stored] == [1, 2, 3, 4]
        assert _as_figures(stored, 1) == [Decimal("1.23"), Decimal("-999.99"), Decimal("999.99"), None]
        if target.engine == "postgres":
            assert [str(row[1]) for row in stored[:3]] == ["1.23", "-999.99", "999.99"]

    @pytest.mark.parametrize(
        ("values", "places", "shown"),
        [
            (["1.23", "1.235"], 3, "1.235"),
            (["1.23", "1000.00"], 2, "1000.00"),
            (["1.23", "-1000.00"], 2, "-1000.00"),
            (["1.23", "0.001"], 3, "0.001"),
        ],
        ids=["one_place_too_many", "one_digit_too_many", "negative_too_wide", "a_tiny_fraction"],
    )
    def test_a_value_the_column_would_round_is_refused_and_nothing_is_written(
        self, target: _Target, values: list[str], places: int, shown: str
    ) -> None:
        with pytest.raises(DataError) as caught:
            _write(target, _priced(_exact(values, places)))

        message = str(caught.value)
        for word in ("'amount'", "numeric(5,2)", shown, "round([amount], 2)", "1 of 2", "nothing is written"):
            assert word in message, f"refusal should name {word!r}: {message}"
        assert caught.value.examples[0]["key"] == 2
        assert [row[0] for row in target.read("priced")] == [9], "the table keeps what it held"

    def test_an_exact_column_into_a_numeric_with_no_places_lands_digit_for_digit(self, target: _Target) -> None:
        frame = _priced(_exact([None, None], 2)).with_columns(loose=_exact(["0.0725", "-12.5000"], 4))

        _write(target, frame)

        stored = target.read("priced")
        assert _as_figures(stored, 2) == [Decimal("0.0725"), Decimal("-12.5")]
        if target.engine == "postgres":
            assert [str(row[2]) for row in stored] == ["0.0725", "-12.5000"]

    def test_a_whole_number_column_must_fit_the_digits_too(self, target: _Target) -> None:
        _write(target, _priced(pl.Series([999, -999, None])))
        assert _as_figures(target.read("priced"), 1) == [Decimal(999), Decimal(-999), None]

        with pytest.raises(DataError) as caught:
            _write(target, _priced(pl.Series([999, 1000])))
        assert "1000" in str(caught.value) and "numeric(5,2)" in str(caught.value)


# ---------------------------------------------------------------------------
# A float column into numeric(p,s)
# ---------------------------------------------------------------------------


class TestAFloatColumnGoesIntoNumericReadTo15Digits:
    def test_floats_that_fit_once_read_to_15_digits_land_as_that_reading(self, target: _Target) -> None:
        frame = _priced(pl.Series([10.25, 0.1 + 0.2, -999.99, None], dtype=pl.Float64))

        _write(target, frame)

        assert _as_figures(target.read("priced"), 1) == [Decimal("10.25"), Decimal("0.3"), Decimal("-999.99"), None]

    @pytest.mark.parametrize("value", [10.255, 1000.0, float("nan"), float("inf")], ids=str)
    def test_a_float_the_column_would_round_or_cannot_hold_is_refused(self, target: _Target, value: float) -> None:
        with pytest.raises(DataError) as caught:
            _write(target, _priced(pl.Series([1.5, value], dtype=pl.Float64)))

        message = str(caught.value)
        for word in ("'amount'", "numeric(5,2)", "15 significant digits", "round([amount], 2)"):
            assert word in message, f"refusal should name {word!r}: {message}"
        assert [row[0] for row in target.read("priced")] == [9]

    def test_a_float32_column_is_read_to_6_digits(self, target: _Target) -> None:
        _write(target, _priced(pl.Series([0.1, 2.5], dtype=pl.Float32)))

        assert _as_figures(target.read("priced"), 1) == [Decimal("0.1"), Decimal("2.5")]


# ---------------------------------------------------------------------------
# Float columns of the table
# ---------------------------------------------------------------------------


class TestFloatColumnsOfTheTable:
    @pytest.mark.parametrize("height", [1, 0], ids=["with_rows", "with_no_rows"])
    def test_an_exact_column_into_double_precision_is_refused(self, target: _Target, height: int) -> None:
        frame = _priced(_exact([None], 2)).with_columns(measure=_exact(["1.50"], 2)).head(height)

        with pytest.raises(DataError) as caught:
            _write(target, frame, allow_empty=True)

        message = str(caught.value)
        for word in ("'measure'", "Decimal", "Float64", "cast_columns", "nothing is written"):
            assert word in message, f"refusal should name {word!r}: {message}"
        assert [row[0] for row in target.read("priced")] == [9]

    def test_an_exact_column_into_real_is_refused(self, target: _Target) -> None:
        frame = pl.DataFrame({"key": [1], "rate": _exact(["0.0725"], 4)})

        with pytest.raises(DataError, match="'rate'"):
            _write(target, frame, table="rated")

    def test_a_float_column_into_double_precision_is_written_exactly_as_before(self, target: _Target) -> None:
        """The path a project publishing Float64 to double precision columns takes, untouched."""
        floats = [0.1 + 0.2, 1 / 3, 123456789.123456789, -2.5e-12, None]
        frame = _priced(pl.Series([None] * 5, dtype=pl.Float64), measure=floats)

        _write(target, frame)

        assert [row[3] for row in target.read("priced")] == floats

    def test_a_float_column_into_a_numeric_with_no_places_is_written_as_before(self, target: _Target) -> None:
        frame = _priced(pl.Series([None], dtype=pl.Float64)).with_columns(loose=pl.Series([0.1 + 0.2]))

        _write(target, frame)

        stored = target.read("priced")[0][2]
        assert float(stored) == 0.1 + 0.2


# ---------------------------------------------------------------------------
# publish_tables runs the same checks, for every table, before anything is written
# ---------------------------------------------------------------------------


class TestPublishTablesChecksTheSameWay:
    def _publish(self, target: _Target, priced: pl.DataFrame, rated: pl.DataFrame, **entry: Any) -> None:
        config: dict[str, Any] = {
            "connection": _VARIABLE,
            "tables": [
                {"port": "priced", "table": "priced", "mode": "replace", **entry},
                {"port": "rated", "table": "rated", "mode": "replace"},
            ],
        }
        if target.schema is not None:
            config["schema"] = target.schema
        run_tool(PublishTables(), Ports({"priced": priced, "rated": rated}), config)

    def test_one_value_that_does_not_fit_stops_every_table(self, target: _Target) -> None:
        rated = pl.DataFrame({"key": [1], "rate": [0.5]})

        with pytest.raises(DataError, match="numeric\\(5,2\\)"):
            self._publish(target, _priced(_exact(["1.235"], 3)), rated)

        assert [row[0] for row in target.read("priced")] == [9]
        assert target.read("rated") == []

    @pytest.mark.parametrize("mode", ["replace", "merge"])
    def test_values_that_fit_are_published(self, target: _Target, mode: str) -> None:
        entry = {"mode": mode, "keys": ["key"], "when_incoming_is_null": "overwrite"} if mode == "merge" else {}
        rated = pl.DataFrame({"key": [1], "rate": [0.5]})

        self._publish(target, _priced(_exact(["1.2300", "-0.5000"], 4)), rated, **entry)

        stored = target.read("priced")
        assert _as_figures([row for row in stored if row[0] != 9], 1) == [Decimal("1.23"), Decimal("-0.5")]
        assert target.read("rated") == [(1, 0.5)]


# ---------------------------------------------------------------------------
# The fit itself, with no database
# ---------------------------------------------------------------------------


class TestWhatFitsANumericColumn:
    def test_an_exact_value_fits_when_it_has_the_places_and_the_digits(self) -> None:
        column = _exact(["1.2300", "1.2350", "999.99", "1000.00", "-999.99", "-1000.00", "0", None], 4)

        assert numeric_misfits(column, 5, 2).to_list() == [False, True, False, True, False, True, False, False]

    def test_a_negative_scale_holds_only_whole_hundreds(self) -> None:
        column = pl.Series([1200, 1250, 9999900, 10000000, -300, None])

        assert numeric_misfits(column, 5, -2).to_list() == [False, True, False, True, False, False]

    def test_more_places_than_an_exact_column_holds_still_counts_the_digits_before_the_point(self) -> None:
        column = _exact(["1.25", "9" * 20 + ".00", "1" + "0" * 20 + ".00"], 2)

        assert numeric_misfits(column, 60, 40).to_list() == [False, False, True]

    def test_a_float_is_read_to_15_significant_digits_and_not_a_number_never_fits(self) -> None:
        column = pl.Series([0.1 + 0.2, 1.005, 2.675, float("nan"), float("-inf"), None])

        assert numeric_misfits(column, 12, 2).to_list() == [False, True, True, True, True, False]

    @pytest.mark.parametrize(
        ("spelling", "declared"),
        [
            ("numeric(12,2)", "Decimal"),
            ("numeric", "Decimal"),
            ("decimal(10,4)", "Decimal"),
            ("double precision", "Float64"),
            ("real", "Float64"),
            ("float8", "Float64"),
        ],
    )
    def test_a_numeric_is_declared_exact_and_a_float_column_float(self, spelling: str, declared: str) -> None:
        assert declared_type_of(spelling) == declared


class TestARoundTripOnPostgres:
    def test_an_exact_column_written_and_read_back_keeps_its_places(
        self, postgres_scratch: Any, postgres_dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dantron_etl.tools.read_table import ReadTable

        session = postgres_scratch(_SCHEMA)
        session.execute(f"drop table if exists {_SCHEMA}.round_trip")
        session.execute(f"create table {_SCHEMA}.round_trip (key integer primary key, amount numeric(12,2))")
        monkeypatch.setenv(_VARIABLE, postgres_dsn)
        frame = pl.DataFrame({"key": [1, 2, 3], "amount": _exact(["1.5000", "-0.0100", None], 4)})

        run_tool(
            WriteTable(),
            Ports({"rows": frame}),
            {"connection": _VARIABLE, "schema": _SCHEMA, "table": "round_trip", "mode": "replace"},
        )
        back = run_tool(ReadTable(), Ports({}), {"connection": _VARIABLE, "schema": _SCHEMA, "table": "round_trip"})

        rows = back["rows"].sort("key")
        assert rows.schema["amount"] == pl.Decimal(38, 2)
        assert [None if v is None else str(v) for v in rows["amount"]] == ["1.50", "-0.01", None]
