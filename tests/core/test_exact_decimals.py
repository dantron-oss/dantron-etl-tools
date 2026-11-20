"""The exact decimal type: its name and its places.

A flow names the type ``Decimal`` and states ``places`` beside it, from 0 to 38. Total digits are
always 38. These tests pin the type name and the places check, and that every tool that names column
types still refuses ``Decimal`` until its own support is built.
"""

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl
import pytest

import dantron_etl
from dantron_etl.authoring import ConfigError, Ports, run_tool
from dantron_etl.column_types import (
    DECLARED_DTYPE,
    DECLARED_TYPES,
    EXACT_DIGITS,
    EXACT_TYPE,
    check_places,
    declared_dtype,
    exact_dtype,
)
from dantron_etl.tools.cast_columns import CastColumns
from dantron_etl.tools.read_csv import ReadCsv
from dantron_etl.tools.unpivot import Unpivot
from dantron_etl.tools.write_excel import WriteExcel


def exact_series(values: list[str | None], places: int, *, precision: int = 38, name: str = "x") -> pl.Series:
    """A series of exact decimals, written as text so nothing passes through a float."""
    return pl.Series(name, [None if v is None else Decimal(v) for v in values], dtype=pl.Decimal(precision, places))


# ---------------------------------------------------------------------------
# The name and its places
# ---------------------------------------------------------------------------


def test_decimal_is_a_type_a_flow_may_name() -> None:
    assert EXACT_TYPE == "Decimal"
    assert EXACT_DIGITS == 38
    assert "Decimal" in DECLARED_TYPES
    assert dantron_etl.COLUMN_TYPES is DECLARED_TYPES
    # Its dtype depends on its places, so it has no single entry in the fixed table.
    assert "Decimal" not in DECLARED_DTYPE
    assert set(DECLARED_DTYPE) == set(DECLARED_TYPES) - {"Decimal"}


@pytest.mark.parametrize("places", [0, 2, 16, 38])
def test_the_exact_dtype_has_38_digits_and_the_places_asked_for(places: int) -> None:
    assert exact_dtype(places) == pl.Decimal(38, places)
    assert declared_dtype("Decimal", places) == pl.Decimal(38, places)


@pytest.mark.parametrize("places", [-1, 39, 2.0, True, None, "2"])
def test_the_exact_dtype_refuses_places_it_cannot_hold(places: Any) -> None:
    with pytest.raises(ValueError, match="places"):
        exact_dtype(places)


def test_every_other_type_name_gives_its_fixed_dtype() -> None:
    for name in DECLARED_TYPES:
        if name != "Decimal":
            assert declared_dtype(name) == DECLARED_DTYPE[name]
    with pytest.raises(ValueError, match="places"):
        declared_dtype("Decimal")
    with pytest.raises(ValueError, match="places"):
        declared_dtype("Float64", 2)


@pytest.mark.parametrize("places", [0, 1, 2, 37, 38])
def test_places_from_0_to_38_are_accepted_with_decimal(places: int) -> None:
    assert check_places("Decimal", places, "columns[0].places") == places


@pytest.mark.parametrize("declared", ["Float64", "Int64", "String", "Date"])
def test_no_places_is_accepted_with_any_other_type(declared: str) -> None:
    assert check_places(declared, None, "columns[0].places") is None


def test_places_are_required_with_decimal() -> None:
    with pytest.raises(ConfigError) as caught:
        check_places("Decimal", None, "columns[0].places")
    message = str(caught.value)
    assert "'columns[0].places'" in message
    assert "0 to 38" in message


@pytest.mark.parametrize("places", [0, 2])
@pytest.mark.parametrize("declared", ["Float64", "Int64", "String"])
def test_places_are_refused_with_any_other_type(declared: str, places: int) -> None:
    """Zero places is refused too: it is still a setting the type cannot use."""
    with pytest.raises(ConfigError) as caught:
        check_places(declared, places, "values_places")
    message = str(caught.value)
    assert "'values_places'" in message
    assert declared in message
    assert "only Decimal" in message


@pytest.mark.parametrize("places", [-1, 39, 100])
def test_places_outside_0_to_38_are_refused(places: int) -> None:
    with pytest.raises(ConfigError) as caught:
        check_places("Decimal", places, "columns[1].places")
    message = str(caught.value)
    assert "'columns[1].places'" in message
    assert str(places) in message
    assert "0 to 38" in message


@pytest.mark.parametrize("places", [2.5, 2.0, "2", True, False])
def test_places_that_are_not_a_whole_number_are_refused(places: Any) -> None:
    """2.0 and true are refused as well: YAML wrote a number with a point, or a yes."""
    with pytest.raises(ConfigError) as caught:
        check_places("Decimal", places, "columns[0].places")
    assert "whole number" in str(caught.value)


# ---------------------------------------------------------------------------
# No tool takes the type yet
# ---------------------------------------------------------------------------


def _refused(tool: Any, config: Mapping[str, Any], inputs: Mapping[str, pl.DataFrame] | None = None) -> str:
    with pytest.raises(ConfigError) as caught:
        run_tool(tool, Ports(dict(inputs or {})), config)
    return str(caught.value)


def _says_not_yet(message: str, tool: str, field: str) -> None:
    assert repr(field) in message
    assert "Decimal" in message
    assert f"{tool} does not take exact decimals yet" in message


def test_cast_columns_refuses_decimal_for_now() -> None:
    rows = pl.DataFrame({"amount": ["1.50"], "price": ["2.25"]})
    alone = {"columns": [{"column": "amount", "type": "Decimal"}]}
    _says_not_yet(_refused(CastColumns(), alone, {"rows": rows}), "cast_columns", "columns[0].type")
    second = {"columns": [{"column": "price", "type": "Float64"}, {"column": "amount", "type": "Decimal"}]}
    _says_not_yet(_refused(CastColumns(), second, {"rows": rows}), "cast_columns", "columns[1].type")


def test_read_csv_refuses_decimal_for_now(tmp_path: Path) -> None:
    path = tmp_path / "orders.csv"
    path.write_text("amount\n1.50\n", encoding="utf-8")
    message = _refused(ReadCsv(), {"path": str(path), "column_types": [{"column": "amount", "type": "Decimal"}]})
    _says_not_yet(message, "read_csv", "column_types[0].type")


def test_unpivot_refuses_decimal_for_now() -> None:
    rows = pl.DataFrame({"order": ["A"], "jan": [1.5], "feb": [2.5]})
    config = {"by": ["order"], "names_to": "month", "values_to": "amount", "values_type": "Decimal"}
    _says_not_yet(_refused(Unpivot(), config, {"rows": rows}), "unpivot", "values_type")


@pytest.mark.parametrize(
    ("setting", "rule"),
    [
        ("formats", {"type": "Decimal", "format": "0.00"}),
        ("widths", {"type": "Decimal", "width": 12}),
        ("wrap", {"type": "Decimal"}),
    ],
)
def test_write_excel_refuses_a_decimal_rule_for_now(tmp_path: Path, setting: str, rule: dict[str, Any]) -> None:
    sheet = {"port": "orders", "sheet": "Orders", setting: [rule]}
    config = {"path": str(tmp_path / "pack.xlsx"), "sheets": [sheet]}
    message = _refused(WriteExcel(), config, {"orders": pl.DataFrame({"amount": [1.5]})})
    _says_not_yet(message, "write_excel", f"sheets[0].{setting}[0].type")
    assert not (tmp_path / "pack.xlsx").exists()
