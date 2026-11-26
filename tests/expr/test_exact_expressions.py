"""Exact decimals in expressions: typed numbers, arithmetic, comparisons, functions and overflow.

A column of type Decimal holds exact values. Arithmetic on it is exact, and only round() rounds. A
number typed in the expression is exact as written. An exact value meeting a float column stops the
step. A flow with no exact column computes exactly what it did before; the characterisation
snapshot holds that side, and a few cases here say it in words.
"""

from decimal import Decimal
import polars as pl
import pytest

from dantron_etl import load_pipeline
from dantron_etl.expr import compile_expr, exact_constant


def exact(values: list[str | None], places: int, *, precision: int = 38) -> pl.Series:
    """An exact series written as text, so nothing passes through a float."""
    return pl.Series([None if value is None else Decimal(value) for value in values], dtype=pl.Decimal(precision, places))


def _column(frame: pl.DataFrame, text: str) -> pl.Series:
    return frame.select(compile_expr(text, frame.schema).alias("out"))["out"]


# ---------------------------------------------------------------------------
# Unit 9: a typed number keeps the text it was written with
# ---------------------------------------------------------------------------


class TestATypedNumberKeepsItsWrittenText:
    @pytest.mark.parametrize(
        ("text", "value"),
        [
            ("0.0725", Decimal("0.0725")),
            ("1.10", Decimal("1.10")),
            ("7", Decimal("7")),
            ("0.0", Decimal("0.0")),
            ("1e-9", Decimal("1E-9")),
            ("2.50E+2", Decimal("250")),
            ("0.1234567890123456789", Decimal("0.1234567890123456789")),
        ],
    )
    def test_the_value_is_the_number_as_written_with_its_places(self, text: str, value: Decimal) -> None:
        found = exact_constant(text)
        assert found == value and found is not None
        assert found.as_tuple().exponent == value.as_tuple().exponent

    @pytest.mark.parametrize(
        ("text", "value"),
        [("-0.5", Decimal("-0.5")), ("-1.10", Decimal("-1.10")), ("- -2.25", Decimal("2.25")), ("-(0.30)", Decimal("-0.30"))],
    )
    def test_a_minus_sign_in_front_is_part_of_the_number(self, text: str, value: Decimal) -> None:
        found = exact_constant(text)
        assert found == value and found is not None
        assert found.as_tuple().exponent == value.as_tuple().exponent

    @pytest.mark.parametrize("text", ["[amount]", "'1.10'", "null", "true", "date'2026-01-01'", "abs(1.5)", "[a] + 1"])
    def test_anything_that_is_not_a_typed_number_has_no_exact_value(self, text: str) -> None:
        assert exact_constant(text) is None

    def test_a_number_from_the_rules_file_keeps_the_value_yaml_gave_it(self) -> None:
        """YAML reads 1.10 as the float 1.1, so a rule keeps its value but can lose a trailing zero."""
        document = {
            "name": "rules",
            "steps": [
                {"id": "entries", "tool": "read_csv", "config": {"path": "entries.csv"}},
                {
                    "id": "priced",
                    "tool": "formula",
                    "config": {"columns": [{"name": "rate", "value": "(${rules.rate})"}]},
                    "inputs": {"rows": "entries.rows"},
                },
                {
                    "id": "written",
                    "tool": "write_csv",
                    "config": {"path": "out.csv", "overwrite": True},
                    "inputs": {"rows": "priced.rows"},
                },
            ],
        }
        for rule, expected in ((0.0725, "0.0725"), (1.10, "1.1"), (-2.5, "-2.5"), (7, "7"), (1e-9, "1E-9")):
            written = load_pipeline(document, rules={"rate": rule}).step("priced").config["columns"][0]["value"]
            found = exact_constant(written)
            assert found is not None and str(found) == expected, (rule, written)

    def test_with_no_exact_column_near_a_typed_number_is_the_float_it_always_was(self) -> None:
        frame = pl.DataFrame({"f": [2.0], "n": [3]})
        for text, value in (("0.0725", 0.0725), ("[f] * 1.10", 2.2), ("[n] * 0.5", 1.5), ("-0.5 + [n]", 2.5)):
            out = _column(frame, text)
            assert out.dtype == pl.Float64 and out.to_list() == [value], text
        assert _column(frame, "[n] + 7").dtype == pl.Int64

    def test_an_exact_value_is_a_number_to_every_type_check_never_a_date(self) -> None:
        from dantron_etl.expr.families import Family, dtype_of_value, family_of_value, holding_phrase

        assert family_of_value(Decimal("0.0725")) is Family.NUMBER
        assert dtype_of_value(Decimal("0.0725")) == pl.Decimal(38, 4)
        assert dtype_of_value(Decimal("1E+5")) == pl.Decimal(38, 0)
        assert holding_phrase(Decimal("1.5")) == "is a decimal"


# ---------------------------------------------------------------------------
# Unit 10: +, -, the minus sign and abs on exact values
# ---------------------------------------------------------------------------

PRICES = pl.DataFrame(
    {
        "key": ["p", "q", "r"],
        "d2": exact(["812.40", "100.00", None], 2),
        "d4": exact(["0.0725", "-7.0000", "1.0000"], 4),
        "n": [7, 3, None],
        "f": [0.0725, 1.5, 2.0],
    }
)


def _formula(frame: pl.DataFrame, value: str) -> pl.Series:
    from dantron_etl.authoring import REGISTRY, Ports, run_tool

    config = {"columns": [{"name": "out", "value": value}]}
    return run_tool(REGISTRY.get("formula"), Ports({"rows": frame}), config)["rows"]["out"]


def _formula_refusal(frame: pl.DataFrame, value: str) -> str:
    from dantron_etl.authoring import DataError

    with pytest.raises(DataError) as raised:
        _formula(frame, value)
    return str(raised.value)


def _exactly(out: pl.Series, places: int, values: list[str | None]) -> None:
    """The column is exact with 38 digits and these places, and holds these values, digit for digit."""
    assert out.dtype == pl.Decimal(38, places), out.dtype
    assert [None if value is None else format(value, "f") for value in out.to_list()] == values


class TestAddingAndSubtractingExactValues:
    def test_two_exact_columns_give_the_larger_places(self) -> None:
        _exactly(_formula(PRICES, "[d2] + [d4]"), 4, ["812.4725", "93.0000", None])
        _exactly(_formula(PRICES, "[d4] - [d2]"), 4, ["-812.3275", "-107.0000", None])

    def test_a_typed_number_beside_an_exact_column_is_exact_as_written(self) -> None:
        _exactly(_formula(PRICES, "[d2] + 0.0725"), 4, ["812.4725", "100.0725", None])
        _exactly(_formula(PRICES, "0.0725 + [d2]"), 4, ["812.4725", "100.0725", None])
        _exactly(_formula(PRICES, "[d2] - 1.10"), 2, ["811.30", "98.90", None])
        _exactly(_formula(PRICES, "[d2] + -0.5"), 2, ["811.90", "99.50", None])
        _exactly(_formula(PRICES, "-0.5 + [d2]"), 2, ["811.90", "99.50", None])

    def test_a_whole_number_beside_an_exact_column_is_exact(self) -> None:
        _exactly(_formula(PRICES, "[d2] - 7"), 2, ["805.40", "93.00", None])
        _exactly(_formula(PRICES, "[d2] + [n]"), 2, ["819.40", "103.00", None])
        _exactly(_formula(PRICES, "[n] - [d4]"), 4, ["6.9275", "10.0000", None])

    def test_with_no_exact_column_a_typed_number_still_gives_a_float(self) -> None:
        assert _formula(PRICES, "[f] + 0.0725").dtype == pl.Float64
        assert _formula(PRICES, "[n] + 0.5").to_list() == [7.5, 3.5, None]

    def test_null_beside_an_exact_column_gives_an_exact_null(self) -> None:
        _exactly(_formula(PRICES, "[d2] + null"), 2, [None, None, None])
        _exactly(_formula(PRICES, "null - [d4]"), 4, [None, None, None])

    def test_the_minus_sign_and_abs_keep_the_places(self) -> None:
        _exactly(_formula(PRICES, "-[d4]"), 4, ["-0.0725", "7.0000", "-1.0000"])
        _exactly(_formula(PRICES, "abs([d4])"), 4, ["0.0725", "7.0000", "1.0000"])
        _exactly(_formula(PRICES, "abs(-[d2] - 0.005)"), 3, ["812.405", "100.005", None])

    @pytest.mark.parametrize("text", ["[d2] + 1e-40", "[d2] - 123456789012345678901234567890123456789.5"])
    def test_a_typed_number_an_exact_decimal_cannot_hold_is_refused(self, text: str) -> None:
        from dantron_etl.expr import ExpressionError

        with pytest.raises(ExpressionError) as raised:
            compile_expr(text, PRICES.schema)
        said = str(raised.value)
        assert "38 digits" in said and "[d2]" in said


BIGGEST = "9" * 36 + ".00"  # 38 digits in all, the most there is at 2 places


class TestAnAnswerTooLargeToHold:
    FRAME = pl.DataFrame(
        {
            "key": ["small", "large"],
            "a": exact(["1.00", BIGGEST], 2),
            "b": exact(["2.00", "1.00"], 2),
            "w": exact(["1", "9" * 37], 0),
        }
    )

    @pytest.mark.parametrize("text", ["[a] + [b]", "-[a] - [b]", "[b] + [a]", "[a] + 1", "[a] + 1.01", "[w] + 0.05"])
    def test_stops_the_step_naming_the_rows_and_what_to_do(self, text: str) -> None:
        said = _formula_refusal(self.FRAME, text)
        assert "columns[0].value" in said and text in said and "38 digits" in said
        assert "1 of 2 row(s)" in said and "'key': 'large'" in said and "round" in said

    def test_just_under_the_limit_is_exact(self) -> None:
        frame = self.FRAME.with_columns(pl.Series("b", [Decimal("2.00"), Decimal("-1.00")], dtype=pl.Decimal(38, 2)))
        _exactly(_formula(frame, "[a] + [b]"), 2, ["3.00", "9" * 35 + "8.00"])
        _exactly(_formula(frame, "-[a] - [b]"), 2, ["-3.00", "-" + "9" * 35 + "8.00"])
        _exactly(_formula(self.FRAME, "[w] - 1"), 0, ["0", "9" * 36 + "8"])
        _exactly(_formula(self.FRAME, "[w] + 0.5"), 1, ["1.5", "9" * 37 + ".5"])
        _exactly(_formula(self.FRAME, "[a] + 0.01"), 2, ["1.01", "9" * 36 + ".01"])

    def test_on_a_row_a_case_never_reaches_it_neither_stops_nor_breaks(self) -> None:
        from dantron_etl.authoring import REGISTRY, Ports, run_tool

        config = {"cases": [{"when": "[a] < 100", "then": "[a] + [b]"}], "output": "out"}
        out = run_tool(REGISTRY.get("case_when"), Ports({"rows": self.FRAME}), config)["rows"]["out"]
        _exactly(out, 2, ["3.00", None])


# ---------------------------------------------------------------------------
# Unit 11: multiplying exact values adds their places, with no rounding
# ---------------------------------------------------------------------------


class TestMultiplyingExactValues:
    FRAME = pl.DataFrame(
        {
            "key": ["p", "q", "r", "s"],
            "a": exact(["812.40", "0.07", "-19.99", None], 2),
            "b": exact(["0.0725", "0.0700", "3.0000", "1.0000"], 4),
            "c": exact(["0.07", "0.07", "0.15", "1.00"], 2),
            "n": [3, -2, 0, None],
        }
    )

    def test_the_product_has_both_places_and_is_never_rounded(self) -> None:
        _exactly(_formula(self.FRAME, "[a] * [b]"), 6, ["58.899000", "0.004900", "-59.970000", None])
        _exactly(_formula(self.FRAME, "[c] * [a]"), 4, ["56.8680", "0.0049", "-2.9985", None])

    def test_a_typed_number_or_a_whole_number_adds_its_own_places(self) -> None:
        _exactly(_formula(self.FRAME, "[a] * 0.0725"), 6, ["58.899000", "0.005075", "-1.449275", None])
        _exactly(_formula(self.FRAME, "0.5 * [a]"), 3, ["406.200", "0.035", "-9.995", None])
        _exactly(_formula(self.FRAME, "[a] * 3"), 2, ["2437.20", "0.21", "-59.97", None])
        _exactly(_formula(self.FRAME, "[n] * [a]"), 2, ["2437.20", "-0.14", "0.00", None])
        _exactly(_formula(self.FRAME, "[a] * -1.0"), 3, ["-812.400", "-0.070", "19.990", None])

    def test_with_no_exact_column_a_product_is_the_float_it_was(self) -> None:
        frame = pl.DataFrame({"f": [812.40], "n": [3]})
        assert _formula(frame, "[f] * 0.0725").to_list() == [812.40 * 0.0725]
        assert _formula(frame, "[n] * 0.5").dtype == pl.Float64
        assert _formula(frame, "[n] * 2").dtype == pl.Int64

    def test_places_past_38_are_refused_when_the_step_starts(self) -> None:
        from dantron_etl.authoring import ConfigError

        frame = pl.DataFrame({"x": exact(["1.5"], 20), "y": exact(["2.5"], 19), "z": exact(["0.5"], 18)})
        with pytest.raises(ConfigError) as raised:
            _formula(frame.head(0), "[x] * [y]")
        said = str(raised.value)
        assert "39 places" in said and "[x]" in said and "[y]" in said and "round" in said
        _exactly(_formula(frame, "[x] * [z]"), 38, ["0." + "75" + "0" * 36])
        with pytest.raises(ConfigError):
            _formula(frame, "[x] * 0.0000000000000000001")

    WHOLE = pl.DataFrame(
        {
            "key": ["under", "over", "cross under", "cross over"],
            "a": exact(["9999999999999999999", "10000000000000000000", "1000000000000000000", "1000000000000000000"], 0),
            "b": exact(["10000000000000000001", "10000000000000000000", "99999999999999999999", "100000000000000000000"], 0),
        }
    )

    @pytest.mark.parametrize("text", ["[a] * [b]", "[b] * [a]", "-[a] * [b]"])
    def test_a_product_past_38_digits_stops_the_step_and_one_just_under_is_exact(self, text: str) -> None:
        said = _formula_refusal(self.WHOLE, text)
        assert "38 digits" in said and "2 of 4 row(s)" in said and "'key': 'over'" in said and "cross over" in said
        under = self.WHOLE.filter(pl.col("key").str.contains("under"))
        sign = "-" if text.startswith("-") else ""
        _exactly(_formula(under, text), 0, [sign + "9" * 38, sign + "99999999999999999999" + "0" * 18])

    def test_a_large_value_times_a_small_one_is_exact_when_the_product_fits(self) -> None:
        frame = pl.DataFrame({"big": exact(["1" + "0" * 31], 0), "tiny": exact(["0.0000003"], 7)})
        _exactly(_formula(frame, "[big] * [tiny]"), 7, ["3" + "0" * 24 + ".0000000"])
        _exactly(_formula(frame, "[tiny] * [big]"), 7, ["3" + "0" * 24 + ".0000000"])

    def test_on_a_row_a_case_never_reaches_a_product_too_large_neither_stops_nor_breaks(self) -> None:
        from dantron_etl.authoring import REGISTRY, Ports, run_tool

        config = {"cases": [{"when": "[key] = 'under'", "then": "[a] * [b]"}], "output": "out"}
        out = run_tool(REGISTRY.get("case_when"), Ports({"rows": self.WHOLE}), config)["rows"]["out"]
        _exactly(out, 0, ["9" * 38, None, None, None])


# ---------------------------------------------------------------------------
# Unit 12: an exact value meeting a float column stops the step
# ---------------------------------------------------------------------------

MIXED = pl.DataFrame(
    {
        "key": ["p", "q", "r", "s"],
        "d": exact(["812.40", "-7.50", None, "100.00"], 2),
        "d4": exact(["0.0725", "-7.5000", "1.2500", None], 4),
        "f": [0.0725, 1.5, 2.0, None],
        "f32": pl.Series([0.5, 1.5, 2.0, None], dtype=pl.Float32),
        "n": [3, -2, None, 100],
    }
)


class TestAnExactValueMeetingAFloatColumn:
    @pytest.mark.parametrize(
        ("text", "named"),
        [
            ("[d] + [f]", "[f]"),
            ("[f] - [d]", "[f]"),
            ("[d] * [f]", "[f]"),
            ("[d] + [f32]", "[f32]"),
            ("[d] = [f]", "[f]"),
            ("[f] < [d]", "[f]"),
            ("[d] >= [n] / 2", "[n] / 2"),
            ("[d] + round([f], 2)", "round([f], 2)"),
            ("coalesce([d], [f])", "[f]"),
            ("coalesce([f], 0, [d])", "[f]"),
            ("least([d], [f])", "[f]"),
            ("greatest([f], [d4])", "[f]"),
            ("nullif([d], [f])", "[f]"),
            ("nullif([f], [d])", "[f]"),
            ("mod([d], [f])", "[f]"),
            ("mod([f], [d4])", "[f]"),
        ],
    )
    def test_stops_the_step_naming_the_float_and_what_to_do(self, text: str, named: str) -> None:
        from dantron_etl.expr import ExpressionError

        with pytest.raises(ExpressionError) as raised:
            compile_expr(text, MIXED.schema)
        said = str(raised.value)
        assert f"{named} " in said and "a float" in said and "cast_columns" in said, said

    def test_a_float_column_beside_a_typed_number_or_a_whole_number_is_as_it_was(self) -> None:
        assert _formula(MIXED, "[f] + 0.5").dtype == pl.Float64
        assert _formula(MIXED, "coalesce([f], 0)").to_list() == [0.0725, 1.5, 2.0, 0.0]
        assert _formula(MIXED, "least([f], [n])").dtype == pl.Float64

    def test_exact_beside_a_whole_number_is_allowed_and_exact(self) -> None:
        assert _formula(MIXED, "[d] = [n] * 0 + 100").to_list() == [False, False, None, True]
        _exactly(_formula(MIXED, "coalesce([d], [n])"), 2, ["812.40", "-7.50", None, "100.00"])
        _exactly(_formula(MIXED, "least([d], [n])"), 2, ["3.00", "-7.50", None, "100.00"])
        _exactly(_formula(MIXED, "greatest([n], [d], 0.125)"), 3, ["812.400", "0.125", "0.125", "100.000"])


class TestFunctionsThatCombineExactValues:
    def test_coalesce_gives_the_larger_places_and_a_typed_zero_is_exact(self) -> None:
        _exactly(_formula(MIXED, "coalesce([d], [d4])"), 4, ["812.4000", "-7.5000", "1.2500", "100.0000"])
        _exactly(_formula(MIXED, "coalesce([d], 0.0)"), 2, ["812.40", "-7.50", "0.00", "100.00"])
        _exactly(_formula(MIXED, "coalesce(null, [d4])"), 4, ["0.0725", "-7.5000", "1.2500", None])

    def test_least_and_greatest_compare_by_value_and_skip_nulls(self) -> None:
        _exactly(_formula(MIXED, "least([d], [d4])"), 4, ["0.0725", "-7.5000", "1.2500", "100.0000"])
        _exactly(_formula(MIXED, "greatest([d4], [d])"), 4, ["812.4000", "-7.5000", "1.2500", "100.0000"])

    def test_nullif_compares_by_value_and_keeps_the_first_values_type(self) -> None:
        _exactly(_formula(MIXED, "nullif([d], 100)"), 2, ["812.40", "-7.50", None, None])
        _exactly(_formula(MIXED, "nullif([d], -7.5000)"), 2, ["812.40", None, None, "100.00"])
        _exactly(_formula(MIXED, "nullif([d], [d4])"), 2, ["812.40", None, None, "100.00"])
        out = _formula(MIXED, "nullif([n], [d])")
        assert out.dtype == pl.Int64 and out.to_list() == [3, -2, None, None]

    def test_mod_is_exact_with_the_first_values_sign_as_in_sql(self) -> None:
        _exactly(_formula(MIXED, "mod([d], 3)"), 2, ["2.40", "-1.50", None, "1.00"])
        _exactly(_formula(MIXED, "mod([d], -0.7)"), 2, ["0.40", "-0.50", None, "0.60"])
        _exactly(_formula(MIXED, "mod([d], [d4])"), 4, ["0.0375", "0.0000", None, None])
        _exactly(_formula(MIXED, "mod(7, [d4])"), 4, ["0.0400", "7.0000", "0.7500", None])

    def test_mod_by_a_zero_exact_value_stops_the_step(self) -> None:
        frame = pl.DataFrame({"key": ["p", "q"], "a": exact(["7.00", "8.00"], 2), "b": exact(["3.00", "0.00"], 2)})
        said = _formula_refusal(frame, "mod([a], [b])")
        assert "divides by zero" in said and "'key': 'q'" in said
        _exactly(_formula(frame.head(1), "mod([a], [b])"), 2, ["1.00"])

    def test_a_value_too_large_for_the_places_it_meets_stops_the_step(self) -> None:
        frame = pl.DataFrame(
            {"key": ["fits", "large"], "w": exact([None, "9" * 37], 0), "d": exact(["1.25", "1.25"], 2)}
        )
        said = _formula_refusal(frame, "coalesce([w], [d])")
        assert "38 digits" in said and "1 of 2 row(s)" in said and "'key': 'large'" in said
        _exactly(_formula(frame.head(1), "coalesce([w], [d])"), 2, ["1.25"])
        assert "'key': 'large'" in _formula_refusal(frame, "greatest([w], [d])")

