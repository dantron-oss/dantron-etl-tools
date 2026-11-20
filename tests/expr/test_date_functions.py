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

from dantron_etl.authoring import DataError, Ports, run_tool
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


def _values(frame: pl.DataFrame, text: str) -> list:
    return _function(frame, text).to_list()


def _formula(frame: pl.DataFrame, value: str) -> list:
    config = {"columns": [{"name": "out", "value": value}]}
    return run_tool(REGISTRY.get("formula"), Ports({"rows": frame}), config)["rows"]["out"].to_list()


class TestThePartsOfADate:
    """month, day, quarter and day_of_week read a part of a date, as year does."""

    frame = pl.DataFrame(
        {
            "d": [date(2024, 2, 29), date(2026, 3, 31), date(2026, 4, 1), date(2026, 11, 16), date(2026, 11, 22), None],
        }
    )

    def test_each_part_of_each_date(self) -> None:
        assert _values(self.frame, "month([d])") == [2, 3, 4, 11, 11, None]
        assert _values(self.frame, "day([d])") == [29, 31, 1, 16, 22, None]
        assert _values(self.frame, "quarter([d])") == [1, 1, 2, 4, 4, None]
        # ISO: Monday is 1 and Sunday is 7. 16 November 2026 is a Monday, the 22nd a Sunday.
        assert _values(self.frame, "day_of_week([d])") == [4, 2, 3, 1, 7, None]

    @pytest.mark.parametrize("function", ["year", "month", "day", "quarter", "day_of_week"])
    def test_each_part_is_a_32_bit_whole_number_that_can_be_added_to(self, function: str) -> None:
        """Int32 like year(), so month([d]) * 100 is 1100 and does not wrap round in a small type."""
        result = _function(self.frame, f"{function}([d]) * 1000")
        assert result.dtype == pl.Int32
        assert (
            result.to_list()[3]
            == {"year": 2026, "month": 11, "day": 16, "quarter": 4, "day_of_week": 1}[function] * 1000
        )

    def test_a_timestamp_is_read_on_its_own_clock_as_year_reads_it(self) -> None:
        """23:30 on New Year's Eve in Toronto is already 1 January in UTC; every part reads Toronto's."""
        frame = pl.DataFrame({"ts": [datetime(2025, 12, 31, 23, 30), datetime(2026, 1, 1, 0, 30)]})
        zoned = frame.with_columns(pl.col("ts").dt.replace_time_zone("America/Toronto"))
        for one in (frame, zoned):
            assert _values(one, "year([ts])") == [2025, 2026]
            assert _values(one, "month([ts])") == [12, 1]
            assert _values(one, "day([ts])") == [31, 1]
            assert _values(one, "quarter([ts])") == [4, 1]
            assert _values(one, "day_of_week([ts])") == [3, 4]

    @pytest.mark.parametrize("function", ["month", "day", "quarter", "day_of_week"])
    @pytest.mark.parametrize("argument", ["[label]", "[qty]", "[flag]", "'2026-01-01'"])
    def test_anything_that_is_not_a_date_is_refused_as_year_refuses_it(self, function: str, argument: str) -> None:
        frame = pl.DataFrame({"label": ["2026-01-01"], "qty": [3], "flag": [True]})
        with pytest.raises(ExpressionError) as raised:
            compile_expr(f"{function}({argument})", frame.schema)
        assert f"{function}()" in str(raised.value)


class TestAddDays:
    """add_days moves a date by whole days; a negative count goes back. The sibling of add_months."""

    def test_forward_back_and_across_month_and_year_ends(self) -> None:
        frame = pl.DataFrame(
            {
                "d": [
                    date(2024, 2, 28),
                    date(2023, 2, 28),
                    date(2026, 1, 1),
                    date(2026, 5, 15),
                    None,
                    date(2026, 5, 15),
                ],
                "n": [1, 1, -1, 0, 3, None],
            }
        )
        assert _values(frame, "add_days([d], [n])") == [
            date(2024, 2, 29),
            date(2023, 3, 1),
            date(2025, 12, 31),
            date(2026, 5, 15),
            None,
            None,
        ]
        assert _function(frame, "add_days([d], [n])").dtype == pl.Date

    def test_it_undoes_itself_and_agrees_with_days_between(self) -> None:
        frame = _pairs().drop_nulls()
        moved = frame.select(
            compile_expr("add_days([start], days_between([start], [end], 'elapsed'))", frame.schema).alias("x")
        )["x"]
        assert moved.to_list() == frame["end"].to_list()

    def test_a_timestamp_keeps_its_time_of_day_across_a_change_of_clocks(self) -> None:
        """A day is a calendar day, as add_months moves by calendar months: noon stays noon."""
        frame = pl.DataFrame({"ts": [datetime(2026, 3, 7, 12, 0)]})
        zoned = frame.with_columns(pl.col("ts").dt.replace_time_zone("America/Toronto"))
        assert _values(frame, "add_days([ts], 1)") == [datetime(2026, 3, 8, 12, 0)]
        (moved,) = _values(zoned, "add_days([ts], 1)")
        assert (moved.day, moved.hour) == (8, 12)

    @pytest.mark.parametrize("count", ["1.5", "[amount]", "[qty] / 2", "'1'", "[d]"])
    def test_a_count_that_is_not_a_whole_number_is_refused(self, count: str) -> None:
        frame = pl.DataFrame({"d": [date(2026, 1, 1)], "amount": [1.0], "qty": [2]})
        with pytest.raises(ExpressionError) as raised:
            compile_expr(f"add_days([d], {count})", frame.schema)
        assert "add_days()" in str(raised.value)

    def test_the_first_argument_has_to_be_a_date(self) -> None:
        frame = pl.DataFrame({"label": ["2026-01-01"]})
        with pytest.raises(ExpressionError) as raised:
            compile_expr("add_days([label], 1)", frame.schema)
        assert "add_days()" in str(raised.value)

    def test_a_count_too_large_to_move_by_exactly_stops_the_step(self) -> None:
        """The engine counts a move in nanoseconds, and past 106,751 days it wraps to a wrong date."""
        frame = pl.DataFrame({"d": [date(2000, 1, 1)] * 3, "n": [106_751, -106_751, 106_752]})
        with pytest.raises(DataError) as raised:
            _formula(frame, "add_days([d], [n])")
        assert "106,751" in str(raised.value) and "106752" in str(raised.value)
        assert raised.value.offending_rows == 1
        assert _formula(frame.head(2), "add_days([d], [n])") == [date(2292, 4, 10), date(1707, 9, 23)]


class TestMonthStart:
    """month_start is the first day of a date's month: month_end's sibling."""

    def test_on_every_day_of_four_years(self) -> None:
        days = pl.date_range(date(2023, 1, 1), date(2026, 12, 31), eager=True)
        frame = pl.DataFrame({"day": [*days.to_list(), None]})
        assert _values(frame, "month_start([day])") == _values(frame, "make_date(year([day]), month([day]), 1)")
        assert _values(frame, "month_start([day])")[59:61] == [date(2023, 3, 1), date(2023, 3, 1)]
        assert _values(frame, "month_start([day])")[-1] is None

    def test_a_timestamp_gives_a_date(self) -> None:
        frame = pl.DataFrame({"ts": [datetime(2026, 3, 31, 23, 30)]})
        result = _function(frame, "month_start([ts])")
        assert result.dtype == pl.Date and result.to_list() == [date(2026, 3, 1)]


class TestParseDate:
    """parse_date reads text as a date in a stated format. Text it cannot read stops the step."""

    frame = pl.DataFrame({"code": ["a", "b", "c"], "shipped": ["03/02/2001", None, "29/02/2024"]})

    def test_text_in_the_format_is_read_and_a_null_stays_null(self) -> None:
        result = _function(self.frame, "parse_date([shipped], '%d/%m/%Y')")
        assert result.dtype == pl.Date
        assert result.to_list() == [date(2001, 2, 3), None, date(2024, 2, 29)]

    def test_compactly_stored_text_is_read_the_same_way(self) -> None:
        frame = self.frame.with_columns(pl.col("shipped").cast(pl.Categorical))
        assert _values(frame, "parse_date([shipped], '%d/%m/%Y')") == [date(2001, 2, 3), None, date(2024, 2, 29)]

    @pytest.mark.parametrize(
        "unreadable", ["2001-02-03", "31/02/2001", "03/02/2001 junk", "", "01/01/0000", "01/01/10000"]
    )
    def test_text_that_does_not_make_a_date_in_the_format_stops_the_step(self, unreadable: str) -> None:
        frame = pl.DataFrame({"code": ["a", "b", "c"], "shipped": ["03/02/2001", unreadable, None]})
        with pytest.raises(DataError) as raised:
            _formula(frame, "parse_date([shipped], '%d/%m/%Y')")
        said = str(raised.value)
        assert "columns[0].value" in said and "[shipped]" in said and "'%d/%m/%Y'" in said
        assert f"'{unreadable}'" in said, said
        assert "1 of 3 row(s)" in said and "'code': 'b'" in said

    def test_the_text_reached_is_the_text_checked(self) -> None:
        """case_when checks the rung only on the rows that reach it, as it checks a division."""
        frame = pl.DataFrame({"shipped": ["03/02/2001", "n/a", None]})
        config = {
            "cases": [
                {"when": "[shipped] = 'n/a'", "then": "null"},
                {"when": "true", "then": "parse_date([shipped], '%d/%m/%Y')"},
            ],
            "output": "on",
        }
        out = run_tool(REGISTRY.get("case_when"), Ports({"rows": frame}), config)["rows"]["on"].to_list()
        assert out == [date(2001, 2, 3), None, None]
        with pytest.raises(DataError):
            run_tool(REGISTRY.get("case_when"), Ports({"rows": frame}), {**config, "cases": config["cases"][1:]})

    def test_an_empty_frame_is_not_stopped(self) -> None:
        assert _formula(self.frame.head(0), "parse_date([shipped], '%d/%m/%Y')") == []

    @pytest.mark.parametrize(
        ("format_", "said"),
        [
            ("[fmt]", "quoted"),
            ("'%Q'", "engine"),
            ("''", "year, month and day"),
            ("'%Y-%m'", "year, month and day"),
            ("'%d/%m'", "year, month and day"),
        ],
    )
    def test_a_format_that_cannot_read_a_whole_date_is_refused_when_the_file_loads(
        self, format_: str, said: str
    ) -> None:
        with pytest.raises(ExpressionError) as raised:
            compile_expr(f"parse_date([shipped], {format_})")
        assert "parse_date()" in str(raised.value) and said in str(raised.value), str(raised.value)

    @pytest.mark.parametrize(
        ("format_", "text", "expected"),
        [
            ("%Y-%m-%d", "2001-02-03", date(2001, 2, 3)),
            ("%d %b %Y", "03 Feb 2001", date(2001, 2, 3)),
            ("%Y%m%d", "20010203", date(2001, 2, 3)),
            ("%d/%m/%Y %H:%M", "03/02/2001 23:59", date(2001, 2, 3)),
        ],
    )
    def test_other_formats_the_engine_reads(self, format_: str, text: str, expected: date) -> None:
        frame = pl.DataFrame({"t": [text]})
        assert _values(frame, f"parse_date([t], '{format_}')") == [expected]

    def test_the_text_must_be_text(self) -> None:
        frame = pl.DataFrame({"n": [20010203]})
        with pytest.raises(ExpressionError) as raised:
            compile_expr("parse_date([n], '%Y%m%d')", frame.schema)
        assert "parse_date()" in str(raised.value)
