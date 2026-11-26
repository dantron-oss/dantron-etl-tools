"""round() and format_number(): halves away from zero, on the number as a person reads it.

The ruling: ``round`` rounds a half away from zero, as a spreadsheet's ROUND does, and it decides
what is a half on the decimal a person reads, not on the binary fraction the column happens to
hold. ``1.005`` is held as ``1.00499999999999989...``; read as a person reads it, it is a half, so
``round(1.005, 2)`` is ``1.01``.

"The decimal a person reads" is made exact this way: a decimal is read to 15 significant digits,
the most a double always carries faithfully from decimal and back, and what a spreadsheet shows.
A value that differs from a half only past its fifteenth digit is a half. When the digit being
rounded on lies past the fifteenth, nothing within the fifteen digits can be rounded, and the
value comes back as it is.

``format_number(x, n, ...)`` writes ``round(x, n)`` to n places, so the two always agree.
"""

import random
from decimal import ROUND_HALF_UP, Decimal

import polars as pl
import pytest

from dantron_etl.authoring import Ports, run_tool
from dantron_etl.expr import compile_expr
from dantron_etl.tools.group_by import GroupBy


def rounded(values: list[float | None], digits: int, dtype: pl.DataType = pl.Float64) -> list[float | None]:
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    return frame.select(compile_expr(f"round([x], {digits})", frame.schema).alias("out"))["out"].to_list()


def written(values: list[float | None], digits: int, dtype: pl.DataType = pl.Float64) -> list[str | None]:
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    text = f"format_number([x], {digits}, ',', '.')"
    return frame.select(compile_expr(text, frame.schema).alias("out"))["out"].to_list()


class TestAHalfRoundsAwayFromZero:
    @pytest.mark.parametrize(
        ("value", "digits", "expected"),
        [
            (2.5, 0, 3.0),
            (-2.5, 0, -3.0),
            (0.5, 0, 1.0),
            (-0.5, 0, -1.0),
            (1.5, 0, 2.0),
            (0.125, 2, 0.13),
            (-0.125, 2, -0.13),
            (10.125, 2, 10.13),
            (100.5, 0, 101.0),
        ],
    )
    def test_an_exact_half_goes_away_from_zero_not_to_the_even_digit(
        self, value: float, digits: int, expected: float
    ) -> None:
        assert rounded([value], digits) == [expected]

    @pytest.mark.parametrize(
        ("value", "digits", "expected"),
        [
            # Each of these is held just below its half, and a spreadsheet still rounds it up.
            (1.005, 2, 1.01),
            (-1.005, 2, -1.01),
            (1.015, 2, 1.02),
            (0.285, 2, 0.29),
            (0.145, 2, 0.15),
            (2.675, 2, 2.68),
            (-15.215, 2, -15.22),
            (999999.995, 2, 1000000.0),
            (502.305, 2, 502.31),
            # A half with fifteen significant digits, the last of them the 5.
            (1234567890123.45, 1, 1234567890123.5),
        ],
    )
    def test_a_half_written_as_a_decimal_is_a_half_whatever_the_double_holds(
        self, value: float, digits: int, expected: float
    ) -> None:
        assert rounded([value], digits) == [expected]

    @pytest.mark.parametrize(
        ("value", "digits", "expected"),
        [
            # Arithmetic leaves noise past the fifteenth digit; the half is still a half.
            (1.15 * 3, 1, 3.5),  # 3.4499999999999997
            (476.72499999999997, 2, 476.73),
            (234.32500000000002, 2, 234.33),
            (-53.794999999999995, 2, -53.8),
        ],
    )
    def test_noise_past_the_fifteenth_digit_does_not_decide_a_half(
        self, value: float, digits: int, expected: float
    ) -> None:
        assert rounded([value], digits) == [expected]

    @pytest.mark.parametrize(
        ("value", "digits", "expected"),
        [
            (1.0049999, 2, 1.0),
            (1.0050001, 2, 1.01),
            (0.29, 2, 0.29),  # 0.29 * 100 is 28.999999999999996 as a double
            (0.57, 2, 0.57),
            (9.995, 2, 10.0),
            (-9.995, 2, -10.0),
            (0.0049999999999999, 2, 0.0),
            (123.456, 1, 123.5),
            (123.449, 1, 123.4),
        ],
    )
    def test_a_value_that_is_not_a_half_rounds_to_the_nearest(self, value: float, digits: int, expected: float) -> None:
        assert rounded([value], digits) == [expected]


class TestWhatPassesThrough:
    def test_a_digit_past_the_fifteenth_is_not_rounded_on(self) -> None:
        """The value comes back as it is: nothing within its fifteen digits lies past n places."""
        values = [41097410227123.7, 1234567890123456789.0, 0.30000000000000004, 12345678901234.567]
        assert rounded(values, 2) == [41097410227123.7, 1234567890123456789.0, 0.3, 12345678901234.567]
        assert rounded([0.30000000000000004], 16) == [0.30000000000000004]

    def test_null_nan_infinity_and_zero_keep_what_they_are(self) -> None:
        out = rounded([None, float("nan"), float("inf"), float("-inf"), 0.0], 2)
        assert out[0] is None and out[1] != out[1] and out[2:] == [float("inf"), float("-inf"), 0.0]

    def test_a_negative_that_rounds_to_zero_keeps_its_sign_as_it_always_did(self) -> None:
        out = rounded([-0.004, -0.0, 0.004], 2)
        assert [str(value) for value in out] == ["-0.0", "-0.0", "0.0"]

    def test_a_whole_number_column_is_unchanged_and_keeps_its_type(self) -> None:
        frame = pl.DataFrame({"x": pl.Series([125, -125, None], dtype=pl.Int64)})
        result = frame.select(compile_expr("round([x], 0)", frame.schema).alias("out"))["out"]
        assert result.dtype == pl.Int64 and result.to_list() == [125, -125, None]

    def test_a_decimal_typed_column_rounds_its_exact_value_away_from_zero_to_exactly_those_places(self) -> None:
        values = [Decimal("1.005"), Decimal("-2.500"), Decimal("1.004"), None]
        frame = pl.DataFrame({"x": pl.Series(values, dtype=pl.Decimal(10, 3))})
        for schema in (frame.schema, None):
            result = frame.select(compile_expr("round([x], 2)", schema).alias("out"))["out"]
            assert result.dtype == pl.Decimal(38, 2)
            assert [None if value is None else format(value, "f") for value in result.to_list()] == [
                "1.01",
                "-2.50",
                "1.00",
                None,
            ]

    def test_a_32_bit_decimal_is_read_to_the_digits_it_carries_and_keeps_its_type(self) -> None:
        frame = pl.DataFrame({"x": pl.Series([1.005, 2.5, 0.125], dtype=pl.Float32)})
        result = frame.select(compile_expr("round([x], 2)", frame.schema).alias("out"))["out"]
        assert result.dtype == pl.Float32
        assert result.to_list() == pytest.approx([1.01, 2.5, 0.13])


class TestEveryWayAnExpressionIsCompiled:
    def test_without_a_schema_the_answers_and_the_types_are_the_same(self) -> None:
        """compile_expr without columns cannot see the type, and must still give the same answer."""
        frame = pl.DataFrame({"f": [1.005, -2.5, 0.29, None], "i": [3, -4, 5, None]})
        blind = frame.select(compile_expr("round([f], 2)").alias("f"), compile_expr("round([i], 0)").alias("i"))
        seen = frame.select(
            compile_expr("round([f], 2)", frame.schema).alias("f"), compile_expr("round([i], 0)", frame.schema).alias("i")
        )
        assert blind.equals(seen) and blind.schema == seen.schema
        assert blind["f"].to_list() == [1.01, -2.5, 0.29, None]

    def test_a_literal_rounds_the_same_way(self) -> None:
        out = pl.DataFrame({"a": [1]}).select(compile_expr("round(1.005, 2)").alias("o"), compile_expr("round(-2.5, 0)").alias("p"))
        assert out.row(0) == (1.01, -3.0)

    def test_inside_a_group_by_value_each_row_is_rounded_before_the_sum(self) -> None:
        rows = pl.DataFrame({"g": ["a", "a", "b"], "x": [0.125, 0.125, 1.005]})
        config = {"by": ["g"], "aggregations": [{"name": "total", "function": "sum", "value": "round([x], 2)"}]}
        result = run_tool(GroupBy(), Ports({"rows": rows}), config)["rows"]
        assert result.to_dicts() == [{"g": "a", "total": 0.26}, {"g": "b", "total": 1.01}]


class TestAgainstExactDecimalRounding:
    def test_every_value_written_with_up_to_fifteen_digits_rounds_as_its_decimal_does(self) -> None:
        """Thousands of decimals, a third of them exact halves, against Python's exact decimal rounding."""
        generator = random.Random(20261119)
        texts: list[str] = []
        for _ in range(4000):
            digits = "".join(generator.choice("0123456789") for _ in range(generator.randint(1, 15))).lstrip("0") or "5"
            point = generator.randint(-3, len(digits) + 2)
            if point <= 0:
                text = "0." + "0" * -point + digits
            elif point >= len(digits):
                text = digits + "0" * (point - len(digits))
            else:
                text = digits[:point] + "." + digits[point:]
            texts.append(("-" if generator.random() < 0.5 else "") + text)
        for _ in range(2000):
            places = generator.randint(0, 4)
            half = (Decimal(2 * generator.randint(0, 10 ** generator.randint(1, 12)) + 1) / 2).scaleb(-places)
            texts.append(("-" if generator.random() < 0.5 else "") + str(half))
        for places in range(0, 7):
            got = rounded([float(text) for text in texts], places)
            for text, answer in zip(texts, got, strict=True):
                expected = float(Decimal(text).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))
                assert answer == expected or (answer == 0 and expected == 0), (text, places, answer, expected)


class TestFormatNumberWritesWhatRoundGives:
    @pytest.mark.parametrize(
        ("value", "digits", "expected"),
        [
            (-15.215, 2, "-15.22"),
            (1.005, 2, "1.01"),
            (2.675, 2, "2.68"),
            (2.5, 0, "3"),
            (-2.5, 0, "-3"),
            (0.125, 2, "0.13"),
            (999999.995, 2, "1,000,000.00"),
            (476.72499999999997, 2, "476.73"),
            (12345678901234.567, 2, "12,345,678,901,234.57"),
        ],
    )
    def test_a_half_is_written_the_way_round_rounds_it(self, value: float, digits: int, expected: str) -> None:
        assert written([value], digits) == [expected]

    def test_writing_a_value_and_writing_its_rounded_value_give_the_same_text(self) -> None:
        generator = random.Random(7)
        values = [round(generator.uniform(-1e7, 1e7), generator.randint(0, 5)) for _ in range(3000)]
        values += [generator.randint(-(10**8), 10**8) / 200 for _ in range(3000)]
        frame = pl.DataFrame({"x": values})
        for digits in (0, 1, 2, 3):
            direct = frame.select(compile_expr(f"format_number([x], {digits}, ',', '.')", frame.schema))
            via = frame.select(compile_expr(f"format_number(round([x], {digits}), {digits}, ',', '.')", frame.schema))
            assert direct.to_series().to_list() == via.to_series().to_list()

    def test_a_whole_number_column_keeps_every_digit(self) -> None:
        assert written([9007199254740993], 2, pl.Int64) == ["9,007,199,254,740,993.00"]
