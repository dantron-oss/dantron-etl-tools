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

