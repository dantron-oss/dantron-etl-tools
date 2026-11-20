"""A null with no type, in any argument a row fills, gives a null answer and never an engine error.

A typeless null is the literal `null`, or a column whose every value is missing and which was read
with no type. Every function in the table is called with one in each argument a row fills, in turn,
against a schema, as every tool compiles. Arguments written in the file (a pattern, a count of
places, a format) refuse a null on purpose and are not called here.
"""

from datetime import date
from typing import Any

import polars as pl
import pytest

from dantron_etl.expr import compile_expr
from dantron_etl.expr.arguments import Value
from dantron_etl.expr.functions import DEFINITIONS, FUNCTIONS

FRAME = pl.DataFrame(
    {
        "t": ["03/02/2001"],
        "n": [3],
        "x": [1.5],
        "d": [date(2026, 1, 31)],
        "nothing": pl.Series([None], dtype=pl.Null),
    }
)

# One working call per function. A new function has to be added here, or the first test fails.
CALLS: dict[str, list[str]] = {
    "coalesce": ["[n]", "1"],
    "round": ["[x]", "2"],
    "year": ["[d]"],
    "month": ["[d]"],
    "day": ["[d]"],
    "quarter": ["[d]"],
    "day_of_week": ["[d]"],
    "add_months": ["[d]", "[n]"],
    "add_days": ["[d]", "[n]"],
    "months_between": ["[d]", "[d]", "'inclusive'"],
    "days_between": ["[d]", "[d]", "'elapsed'"],
    "make_date": ["[n]", "[n]", "[n]"],
    "month_end": ["[d]"],
    "month_start": ["[d]"],
    "parse_date": ["[t]", "'%d/%m/%Y'"],
    "abs": ["[x]"],
    "floor": ["[x]"],
    "ceil": ["[x]"],
    "sign": ["[x]"],
    "mod": ["[n]", "2"],
    "least": ["[n]", "[x]"],
    "greatest": ["[n]", "[x]"],
    "nullif": ["[n]", "3"],
    "lower": ["[t]"],
    "upper": ["[t]"],
    "trim": ["[t]"],
    "len": ["[t]"],
    "left": ["[t]", "2"],
    "right": ["[t]", "2"],
    "substr": ["[t]", "2", "3"],
    "split_part": ["[t]", "'/'", "2"],
    "pad_left": ["[t]", "12", "'0'"],
    "pad_right": ["[t]", "12", "'0'"],
    "starts_with": ["[t]", "'0'"],
    "ends_with": ["[t]", "'1'"],
    "contains": ["[t]", "'2'"],
    "regex_extract": ["[t]", "'([0-9]+)'", "1"],
    "regex_replace": ["[t]", "'/'", "'-'"],
    "regex_match": ["[t]", "'^0'"],
    "concat": ["[t]", "'x'"],
    "format_number": ["[x]", "2", "','", "'.'"],
    "uuid5": ["'6ba7b810-9dad-11d1-80b4-00c04fd430c8'", "[t]"],
}

# A null is skipped by these, so the answer is the other value; or (nullif's second) compared with nothing.
NOT_NULL_ANSWERS: dict[tuple[str, int], Any] = {
    ("coalesce", 0): 1,
    ("coalesce", 1): 3,
    ("least", 0): 1.5,
    ("least", 1): 3,
    ("greatest", 0): 1.5,
    ("greatest", 1): 3,
    ("nullif", 1): 3,
}


def _cases() -> list[Any]:
    cases = []
    for name, arguments in CALLS.items():
        function = FUNCTIONS[name]
        for position in range(len(arguments)):
            param = function.params[0] if function.variadic else function.params[position]
            if not isinstance(param, Value):
                continue  # written in the file; a null there is refused on purpose
            for null in ("null", "[nothing]"):
                written = list(arguments)
                written[position] = null
                cases.append(
                    pytest.param(name, position, f"{name}({', '.join(written)})", id=f"{name}-{position}-{null}")
                )
    return cases


def test_every_function_is_called_here() -> None:
    assert set(CALLS) == {function.name for function in DEFINITIONS}


@pytest.mark.parametrize(("name", "position", "text"), _cases())
def test_a_typeless_null_gives_a_null_answer(name: str, position: int, text: str) -> None:
    answer = FRAME.select(compile_expr(text, FRAME.schema).alias("out"))["out"].to_list()
    assert answer == [NOT_NULL_ANSWERS.get((name, position))]


@pytest.mark.parametrize(
    ("text", "dtype"),
    [
        # The table promises a whole number, so the null answer is one.
        ("year(null)", pl.Int32),
        ("month([nothing])", pl.Int32),
        ("day(null)", pl.Int32),
        ("quarter([nothing])", pl.Int32),
        ("day_of_week(null)", pl.Int32),
        # The answer takes its argument's type, and a typeless null has none, as round(null, 2) has none.
        ("abs(null)", pl.Null),
        ("abs([nothing])", pl.Null),
        ("add_months(null, 1)", pl.Null),
        ("add_days([nothing], 1)", pl.Null),
    ],
)
def test_the_null_answer_has_the_promised_type_or_none_at_all(text: str, dtype: pl.DataType) -> None:
    assert FRAME.select(compile_expr(text, FRAME.schema).alias("out"))["out"].dtype == dtype


def test_a_null_date_is_not_stopped_for_a_count_too_large_to_move_it_by() -> None:
    """A null moved by any count is null: there is no date to move, so there is nothing to refuse."""
    from dantron_etl.authoring import REGISTRY, Ports, run_tool

    rows = pl.DataFrame({"d": [None], "n": [200_000]}, schema={"d": pl.Date, "n": pl.Int64})
    config = {"columns": [{"name": "out", "value": "add_days([d], [n])"}]}
    assert run_tool(REGISTRY.get("formula"), Ports({"rows": rows}), config)["rows"]["out"].to_list() == [None]
