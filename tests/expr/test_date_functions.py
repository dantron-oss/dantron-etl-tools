"""months_between, days_between, make_date and month_end give exactly what the tools give.

Each function answers the same question as a tool: ``months_between`` and ``days_between`` as
``date_diff``, ``make_date`` as ``assemble_date`` with ``on_invalid: set_null``, and ``month_end``
as ``assemble_date`` with ``day: '31'`` and ``on_invalid: clamp``. Both sides call one shared
function, and these tests run the tool and the function over the same rows to prove they agree:
month ends, leap years, reversed pairs, the same day at both ends, and nulls.
"""

import random
from datetime import date, datetime, timedelta

import polars as pl
import pytest

from dantron_etl.authoring import Ports, run_tool
from dantron_etl.expr import ExpressionError, compile_expr
from dantron_etl.registry import REGISTRY


def _pairs() -> pl.DataFrame:
    """Hand-picked pairs, then a few hundred random ones from a fixed seed."""
    picked = [
        (date(2026, 1, 31), date(2026, 2, 28)),
        (date(2026, 1, 1), date(2026, 12, 31)),
        (date(2024, 1, 31), date(2024, 2, 29)),
        (date(2024, 2, 29), date(2025, 2, 28)),
        (date(2024, 2, 29), date(2028, 2, 29)),
        (date(2026, 3, 31), date(2026, 4, 30)),
        (date(2026, 5, 15), date(2026, 5, 15)),
        (date(2026, 1, 1), date(2025, 12, 31)),
        (date(2026, 2, 28), date(2026, 1, 31)),
        (date(2000, 2, 29), date(1900, 2, 28)),
        (None, date(2026, 3, 1)),
        (date(2026, 3, 1), None),
        (None, None),
    ]
    rng = random.Random(20261119)
    start = date(1999, 1, 1)
    for _ in range(400):
        one = start + timedelta(days=rng.randrange(0, 12_000))
        other = start + timedelta(days=rng.randrange(0, 12_000))
        picked.append((one, other))
    return pl.DataFrame(
        {"start": [a for a, _ in picked], "end": [b for _, b in picked]},
        schema={"start": pl.Date, "end": pl.Date},
    )


def _tool(name: str, frame: pl.DataFrame, config: dict) -> pl.Series:
    return run_tool(REGISTRY.get(name), Ports({"rows": frame}), config)["rows"][config["output"]]


def _function(frame: pl.DataFrame, text: str) -> pl.Series:
    return frame.select(compile_expr(text, frame.schema).alias("result"))["result"]


class TestCountingBetweenDatesAgreesWithDateDiff:
    @pytest.mark.parametrize("counting", ["inclusive", "elapsed"])
    @pytest.mark.parametrize(("function", "unit"), [("months_between", "month"), ("days_between", "day")])
    def test_on_every_pair(self, function: str, unit: str, counting: str) -> None:
        frame = _pairs()
        by_tool = _tool(
            "date_diff", frame, {"start": "[start]", "end": "[end]", "unit": unit, "counting": counting, "output": "n"}
        )
        by_function = _function(frame, f"{function}([start], [end], '{counting}')")
        assert by_function.dtype == by_tool.dtype == pl.Int64
        assert by_function.to_list() == by_tool.to_list()

    def test_known_answers(self) -> None:
        frame = _pairs().head(13)
        assert _function(frame, "months_between([start], [end], 'inclusive')").to_list() == [
            1, 12, 1, 12, 49, 1, 1, -1, -1, -1201, None, None, None,
        ]
        assert _function(frame, "days_between([start], [end], 'elapsed')").to_list()[:8] == [28, 364, 29, 365, 1461, 30, 0, -1]

    @pytest.mark.parametrize("counting", ["inclusive", "elapsed"])
    def test_a_timestamp_without_a_zone_is_read_as_its_date_by_both(self, counting: str) -> None:
        frame = pl.DataFrame(
            {"start": [datetime(2026, 1, 31, 15, 0), datetime(2026, 1, 31, 23, 59)], "end": [datetime(2026, 1, 31, 9, 0), datetime(2026, 3, 1)]}
        )
        by_tool = _tool(
            "date_diff", frame, {"start": "[start]", "end": "[end]", "unit": "month", "counting": counting, "output": "n"}
        )
        assert _function(frame, f"months_between([start], [end], '{counting}')").to_list() == by_tool.to_list()

    def test_a_counting_word_that_is_not_one_of_the_two_is_refused(self) -> None:
        with pytest.raises(ExpressionError) as raised:
            compile_expr("months_between([start], [end], 'both')")
        assert "'inclusive' or 'elapsed'" in str(raised.value)

    def test_a_counting_from_a_column_is_refused(self) -> None:
        with pytest.raises(ExpressionError):
            compile_expr("days_between([start], [end], [how])")


def _parts() -> pl.DataFrame:
    years = [2024, 2025, 2000, 1900, 1, 9999, 0, 10000, None]
    months = [1, 2, 4, 12, 0, 13, None]
    days = [1, 28, 29, 30, 31, 0, 32, None]
    rows = [(y, m, d) for y in years for m in months for d in days]
    return pl.DataFrame(
        {"y": [r[0] for r in rows], "m": [r[1] for r in rows], "d": [r[2] for r in rows]},
        schema={"y": pl.Int64, "m": pl.Int64, "d": pl.Int64},
    )


class TestMakeDateAgreesWithAssembleDate:
    def test_on_every_combination_of_good_and_impossible_parts(self) -> None:
        frame = _parts()
        by_tool = _tool(
            "assemble_date",
            frame,
            {"year": "[y]", "month": "[m]", "day": "[d]", "on_invalid": "set_null", "output": "made"},
        )
        by_function = _function(frame, "make_date([y], [m], [d])")
        assert by_function.dtype == by_tool.dtype == pl.Date
        assert by_function.to_list() == by_tool.to_list()

    def test_an_impossible_date_is_null_and_a_real_one_is_itself(self) -> None:
        frame = pl.DataFrame({"y": [2024, 2025, 2026], "m": [2, 2, 4], "d": [29, 29, 31]})
        assert _function(frame, "make_date([y], [m], [d])").to_list() == [date(2024, 2, 29), None, None]

    def test_a_decimal_part_is_refused(self) -> None:
        frame = pl.DataFrame({"y": [2026.0], "m": [1], "d": [1]})
        with pytest.raises(ExpressionError) as raised:
            compile_expr("make_date([y], [m], [d])", frame.schema)
        assert "whole numbers" in str(raised.value)


class TestMonthEndAgreesWithAssembleDatesClamp:
    def test_on_every_day_of_four_years(self) -> None:
        days = pl.date_range(date(2023, 1, 1), date(2026, 12, 31), eager=True)
        frame = pl.DataFrame({"day": days}).with_columns(
            pl.col("day").dt.year().cast(pl.Int64).alias("y"), pl.col("day").dt.month().cast(pl.Int64).alias("m")
        )
        frame = pl.concat([frame, pl.DataFrame({"day": [None], "y": [None], "m": [None]}, schema=frame.schema)])
        by_tool = _tool(
            "assemble_date",
            frame,
            {"year": "[y]", "month": "[m]", "day": "31", "on_invalid": "clamp", "output": "end"},
        )
        by_function = _function(frame, "month_end([day])")
        assert by_function.to_list() == by_tool.to_list()
        assert by_function.to_list()[:1] == [date(2023, 1, 31)]
        assert date(2024, 2, 29) in by_function.to_list()
        assert by_function.to_list()[-1] is None


class TestAZoneAwareTimestampIsRefused:
    @pytest.mark.parametrize(
        "text",
        ["months_between([ts], [d], 'inclusive')", "days_between([d], [ts], 'elapsed')", "month_end([ts])"],
    )
    def test_by_name(self, text: str) -> None:
        frame = pl.DataFrame({"ts": [datetime(2026, 1, 31, 23, 30)], "d": [date(2026, 1, 1)]}).with_columns(
            pl.col("ts").dt.replace_time_zone("America/Toronto")
        )
        with pytest.raises(ExpressionError) as raised:
            compile_expr(text, frame.schema)
        assert "[ts]" in str(raised.value) and "America/Toronto" in str(raised.value) and "Date" in str(raised.value)
