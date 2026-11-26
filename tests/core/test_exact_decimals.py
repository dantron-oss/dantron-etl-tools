"""The exact decimal type: its name, its places, and the helpers that convert to and from it.

A flow names the type ``Decimal`` and states ``places`` beside it, from 0 to 38. Total digits are
always 38. These tests pin the type name and the places check, that every tool that names column
types still refuses ``Decimal`` until its own support is built, and the helpers that convert a
column to an exact one and back. A conversion never rounds: each value converts exactly, or its row
is reported as not fitting.
"""

import random
from collections.abc import Mapping
from datetime import date
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
    ExactConversion,
    check_places,
    declared_dtype,
    exact_dtype,
    exact_from_exact,
    exact_from_float,
    exact_from_text,
    exact_from_whole,
    exact_misfit_refusal,
    exact_to_float,
    to_exact,
)
from dantron_etl.tools.cast_columns import CastColumns
from dantron_etl.tools.read_csv import ReadCsv
from dantron_etl.tools.unpivot import Unpivot


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


# ---------------------------------------------------------------------------
# Converting to an exact column: every value exactly, or the rows that do not fit
# ---------------------------------------------------------------------------


def _converted(conversion: ExactConversion, places: int) -> tuple[list[Any], list[bool]]:
    """The values and the misfit marks, after checking the shape every conversion promises."""
    assert conversion.values.dtype == pl.Decimal(38, places)
    assert conversion.misfits.dtype == pl.Boolean
    assert conversion.values.len() == conversion.misfits.len()
    # A row that did not fit is empty; a row that fit holds a value unless the source was empty.
    assert not (conversion.misfits & conversion.values.is_not_null()).any()
    return conversion.values.to_list(), conversion.misfits.to_list()


def test_text_of_plain_digits_converts_exactly() -> None:
    text = pl.Series("amount", ["12.50", "-12.5", "0", "-0.00", "007.10", "1.2300", "5", None])
    values, misfits = _converted(exact_from_text(text, 2), 2)
    assert values == [
        Decimal("12.50"),
        Decimal("-12.50"),
        Decimal("0.00"),
        Decimal("0.00"),
        Decimal("7.10"),
        Decimal("1.23"),
        Decimal("5.00"),
        None,
    ]
    assert misfits == [False] * 8
    assert exact_from_text(text, 2).values.name == "amount"


def test_text_that_would_need_rounding_does_not_fit() -> None:
    """Trailing zeros are not places: 1.2300 fits 2 places, 1.2350 does not."""
    text = pl.Series(["1.005", "1.235", "-0.001", "1.230", "1.2350"])
    values, misfits = _converted(exact_from_text(text, 2), 2)
    assert misfits == [True, True, True, False, True]
    assert values == [None, None, None, Decimal("1.23"), None]
    assert _converted(exact_from_text(text, 3), 3)[1] == [False, False, False, False, False]


@pytest.mark.parametrize(
    "written",
    ["1e3", "1E3", ".5", "+1.25", " 1.50", "1.50 ", "1,000.00", "12.", "", "abc", "--1", "1-", "١٢", "NaN", "inf"],
)
def test_text_not_written_as_plain_digits_does_not_fit(written: str) -> None:
    values, misfits = _converted(exact_from_text(pl.Series([written, "1.50"]), 2), 2)
    assert misfits == [True, False]
    assert values == [None, Decimal("1.50")]


@pytest.mark.parametrize(
    ("places", "fits", "too_wide"),
    [
        (2, "9" * 36 + ".99", "1" + "0" * 36 + ".00"),
        (2, "-" + "9" * 36 + ".99", "-1" + "0" * 36),
        (2, "000" + "9" * 36 + ".99", "1" + "0" * 36 + ".5"),
        (0, "9" * 38, "1" + "0" * 38),
        (38, "0." + "9" * 38, "1.0"),
        (38, "-0." + "0" * 37 + "1", "-1"),
    ],
)
def test_text_at_the_38_digit_edge(places: int, fits: str, too_wide: str) -> None:
    values, misfits = _converted(exact_from_text(pl.Series([fits, too_wide]), places), places)
    assert misfits == [False, True]
    assert values == [Decimal(fits), None]


def test_whole_numbers_convert_exactly() -> None:
    whole = pl.Series("count", [7, -3, 0, None], dtype=pl.Int64)
    values, misfits = _converted(exact_from_whole(whole, 2), 2)
    assert values == [Decimal("7.00"), Decimal("-3.00"), Decimal("0.00"), None]
    assert misfits == [False] * 4
    for dtype in (pl.Int8, pl.Int32, pl.UInt8, pl.UInt64):
        assert _converted(exact_from_whole(pl.Series([7], dtype=dtype), 0), 0) == ([Decimal("7")], [False])


@pytest.mark.parametrize(
    ("places", "values", "dtype", "expected_misfits"),
    [
        # 38 - 19 = 19 whole digits: every Int64 fits, at both ends.
        (19, [2**63 - 1, -(2**63)], pl.Int64, [False, False]),
        # 18 whole digits: the largest that fits, and the smallest that does not, on both sides.
        (20, [10**18 - 1, 10**18, -(10**18 - 1), -(10**18)], pl.Int64, [False, True, False, True]),
        # 38 places leave no whole digit: only zero fits.
        (38, [0, 1, -1], pl.Int64, [False, True, True]),
        (18, [2**64 - 1], pl.UInt64, [False]),
        (19, [2**64 - 1, 10**19 - 1], pl.UInt64, [True, False]),
    ],
)
def test_whole_numbers_at_the_38_digit_edge(
    places: int, values: list[int], dtype: pl.DataType, expected_misfits: list[bool]
) -> None:
    converted, misfits = _converted(exact_from_whole(pl.Series(values, dtype=dtype), places), places)
    assert misfits == expected_misfits
    assert converted == [None if misfit else Decimal(value) for value, misfit in zip(values, expected_misfits)]


def test_exact_to_more_places_is_exact() -> None:
    held = exact_series(["1.50", "-2.25", "0.00", None], 2)
    values, misfits = _converted(exact_from_exact(held, 4), 4)
    assert values == [Decimal("1.5000"), Decimal("-2.2500"), Decimal("0.0000"), None]
    assert misfits == [False] * 4
    # The same places at fewer digits in all, as a source may give, becomes 38 digits.
    narrow = exact_series(["1.50"], 2, precision=18)
    assert _converted(exact_from_exact(narrow, 2), 2) == ([Decimal("1.50")], [False])


def test_exact_to_fewer_places_converts_only_values_that_fit() -> None:
    """1.2350 does not become 1.24, nor 1.2250 become 1.22, as the engine's own conversion would."""
    held = exact_series(["1.2300", "-1.2000", "0.0000", "1.2350", "1.2250", "-0.0001", None], 4)
    values, misfits = _converted(exact_from_exact(held, 2), 2)
    assert values == [Decimal("1.23"), Decimal("-1.20"), Decimal("0.00"), None, None, None, None]
    assert misfits == [False, False, False, True, True, True, False]


@pytest.mark.parametrize(
    ("held", "places", "expected_misfits"),
    [
        (["9" * 36 + ".99", "-" + "9" * 36 + ".99"], 2, [False, False]),
        (
            ["9" * 36 + ".99", "-" + "9" * 36 + ".99", "9" * 35 + ".99", "-" + "9" * 35 + ".99"],
            3,
            [True, True, False, False],
        ),
        (["0.99", "-0.99", "1.00", "-1.00"], 38, [False, False, True, True]),
    ],
)
def test_exact_at_the_38_digit_edge(held: list[str], places: int, expected_misfits: list[bool]) -> None:
    values, misfits = _converted(exact_from_exact(exact_series(held, 2), places), places)
    assert misfits == expected_misfits
    assert values == [None if misfit else Decimal(text) for text, misfit in zip(held, expected_misfits)]


def test_a_float_is_read_to_15_significant_digits_then_must_fit() -> None:
    floats = pl.Series("price", [0.1 + 0.2, 1.5, -2.25, 0.0, -0.0, 0.1 + 0.7, None])
    values, misfits = _converted(exact_from_float(floats, 2), 2)
    assert values == [
        Decimal("0.30"),
        Decimal("1.50"),
        Decimal("-2.25"),
        Decimal("0.00"),
        Decimal("0.00"),
        Decimal("0.80"),
        None,
    ]
    assert misfits == [False] * 7
    assert str(values[4]) == "0.00", "negative zero becomes plain zero"


def test_a_float_that_would_need_rounding_does_not_fit() -> None:
    """1.005 is read as 1.005, so it fits three places and not two, whatever its binary value."""
    floats = pl.Series([1.005, 2.675, 0.125, -0.125])
    assert _converted(exact_from_float(floats, 2), 2)[1] == [True, True, True, True]
    assert _converted(exact_from_float(floats, 3), 3) == (
        [Decimal("1.005"), Decimal("2.675"), Decimal("0.125"), Decimal("-0.125")],
        [False] * 4,
    )


@pytest.mark.parametrize("special", [float("nan"), float("inf"), float("-inf")])
def test_not_a_number_and_infinity_do_not_fit(special: float) -> None:
    assert _converted(exact_from_float(pl.Series([special, 1.0]), 2), 2) == ([None, Decimal("1.00")], [True, False])


@pytest.mark.parametrize(
    ("value", "places", "expected"),
    [
        # The 15-digit reading, both sides of where it stops.
        (1 / 3, 15, "0.333333333333333"),
        (1 / 3, 14, None),
        (0.9999999999999999, 0, "1"),
        (0.999999999999999, 0, None),
        (-0.9999999999999999, 0, "-1"),
        # The shortest text's 16th digit decides: 5 goes away from zero, even where the binary
        # value lies just below the half (4.91649168992446483...); 4 goes toward zero.
        (4.916491689924465, 14, "4.91649168992447"),
        (-4.916491689924465, 14, "-4.91649168992447"),
        (7.755858880277834, 14, "7.75585888027783"),
        (123456789012345680000.0, 0, "123456789012346000000"),
        # The 38-digit edge, and the smallest place.
        (1e35, 2, "1" + "0" * 35),
        (-1e35, 2, "-1" + "0" * 35),
        (1e36, 2, None),
        (999999999999999.9, 23, None),
        (1e-7, 7, "0.0000001"),
        (1e-7, 6, None),
        (5e-324, 38, None),
        (1.5e-10, 38, "0.00000000015"),
    ],
)
def test_floats_at_the_edges(value: float, places: int, expected: str | None) -> None:
    values, misfits = _converted(exact_from_float(pl.Series([value]), places), places)
    assert misfits == [expected is None]
    assert values == [None if expected is None else Decimal(expected)]


def test_a_32_bit_float_is_read_to_6_significant_digits() -> None:
    narrow = pl.Series([0.1, 1.005, 1234.5], dtype=pl.Float32)
    assert _converted(exact_from_float(narrow, 2), 2) == (
        [Decimal("0.10"), None, Decimal("1234.50")],
        [False, True, False],
    )


def test_to_exact_picks_the_conversion_by_the_column_type() -> None:
    assert to_exact(pl.Series(["1.5"]), 2).values.to_list() == [Decimal("1.50")]
    assert to_exact(pl.Series([3]), 2).values.to_list() == [Decimal("3.00")]
    assert to_exact(exact_series(["1.2300"], 4), 2).values.to_list() == [Decimal("1.23")]
    assert to_exact(pl.Series([0.1 + 0.2]), 2).values.to_list() == [Decimal("0.30")]
    typeless = to_exact(pl.Series("x", [None, None]), 2)
    assert _converted(typeless, 2) == ([None, None], [False, False])
    empty = to_exact(pl.Series("x", [], dtype=pl.String), 2)
    assert _converted(empty, 2) == ([], [])
    for other in (pl.Series([True]), pl.Series([date(2026, 1, 1)])):
        with pytest.raises(TypeError, match="exact"):
            to_exact(other, 2)
    with pytest.raises(TypeError):
        exact_from_text(pl.Series([1.5]), 2)
    with pytest.raises(ValueError, match="places"):
        exact_from_text(pl.Series(["1.5"]), 39)


def test_the_refusal_for_values_that_do_not_fit_names_them_and_says_what_to_do() -> None:
    message = exact_misfit_refusal("amount", pl.Series([1.005, 2.675, 1.005, 9.9999]), 2)
    assert "'amount'" in message
    assert "Decimal with 2 places" in message
    assert "1.005, 2.675, 9.9999" in message
    assert "round([amount], 2)" in message
    assert "on_invalid: set_null" in message
    text = exact_misfit_refusal("amount", pl.Series(["1e3", "1.005"]), 2)
    assert "'1e3', '1.005'" in text
    assert "plain digits" in text


# ---------------------------------------------------------------------------
# Converting from an exact column to Float64
# ---------------------------------------------------------------------------


def test_exact_to_float_lands_on_the_nearest_double() -> None:
    held = exact_series(["99999999999999.99", "457200000000000000000000.00", "-0.10", "0.00", None], 2)
    floats = exact_to_float(held)
    assert floats.dtype == pl.Float64
    assert floats.name == "x"
    assert floats.to_list() == [99999999999999.98, 4.572e23, -0.1, 0.0, None]
    assert str(floats[3]) == "0.0"
    long = exact_series(["1234567890123456789012345678.9012345678"], 10)
    assert exact_to_float(long).to_list() == [float("1234567890123456789012345678.9012345678")]


def _figures(places: int, count: int, seed: int) -> list[Decimal]:
    """Figures of every size whose digits, counting the places held, make a whole number up to 2**53."""
    rng = random.Random(seed)
    figures = []
    for _ in range(count):
        unscaled = min(int(10 ** rng.uniform(0, 15.96)), 2**53)
        if rng.random() < 0.2:
            unscaled = 2**53 - rng.randint(0, 999)
        figures.append(Decimal(unscaled if rng.random() < 0.5 else -unscaled).scaleb(-places))
    return figures


@pytest.mark.parametrize("places", [0, 1, 2, 3, 4, 6, 10, 16, 22])
def test_todays_conversion_to_float_already_lands_on_the_nearest_double_for_these_figures(places: int) -> None:
    """cast_columns to Float64 keeps giving what it gives today, and so does the helper.

    A project converts its exact prices this way before any arithmetic. Today's conversion is the
    nearest double whenever the figure's digits, counting every place the column holds, form a
    whole number no larger than 2**53, and the column holds 22 places or fewer. That covers every
    figure of 15 digits or fewer at those places, such as any price under 10 trillion at 2 places.
    """
    figures = _figures(places, 3000, seed=places)
    held = pl.DataFrame({"price": pl.Series(figures, dtype=pl.Decimal(38, places))})
    today = run_tool(CastColumns(), Ports({"rows": held}), {"columns": [{"column": "price", "type": "Float64"}]})
    nearest = [float(figure) for figure in figures]
    assert today["rows"]["price"].to_list() == nearest
    assert exact_to_float(held["price"]).to_list() == nearest


@pytest.mark.parametrize(
    ("written", "places", "today"),
    [
        ("457200000000000000000000.00", 2, 4.5720000000000004e23),
        ("36533858054.140500", 6, 36533858054.140495),
        ("0.0000000000000000000000125", 25, 1.2499999999999999e-23),
    ],
)
def test_where_todays_conversion_to_float_misses_the_nearest_double(written: str, places: int, today: float) -> None:
    """Past 2**53 in the digits held, or past 22 places, today's conversion can miss by one step."""
    held = pl.DataFrame({"price": exact_series([written], places)})
    result = run_tool(CastColumns(), Ports({"rows": held}), {"columns": [{"column": "price", "type": "Float64"}]})
    assert result["rows"]["price"].to_list() == [today]
    assert exact_to_float(held["price"]).to_list() == [float(written)] != [today]
