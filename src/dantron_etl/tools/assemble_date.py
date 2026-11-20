"""AssembleDate: build one date column from separate year, month and day parts.

The day defaults to 1. `on_invalid` decides only what happens to a day the month is too short for; a
bad month, year or non-digit text is always refused. A null part gives a null date.

Why: docs/design/tools.md#assemble_date
"""

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

import polars as pl

from ..authoring import (
    MAX_YEAR,
    MIN_YEAR,
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    check_output_column,
    compile_for,
    compiled_dtype,
    date_or_null,
    refuse_rows_without_an_answer,
    require_not_blank,
    row_status,
    value_caption,
)
from ..authoring import days_in_month as month_length

_ON_INVALID_ENUM: Final[tuple[str, ...]] = ("clamp", "set_null", "error")
_DIGITS_ONLY: Final[str] = r"^[0-9]+$"


class AssembleDate:
    """Build a date column from separate year, month and day parts."""

    name = "assemble_date"
    description = "Build a date from separate year, month and day columns."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "year": ConfigField(
                type="expression",
                description="The year: a column such as '[year]', a fixed value, or an expression.",
                required=True,
            ),
            "month": ConfigField(
                type="expression",
                description="The month as a number from 1 to 12. A month name is not read.",
                required=True,
            ),
            "day": ConfigField(
                type="expression",
                description="The day of the month. Left out, it is 1.",
                default="1",
            ),
            "on_invalid": ConfigField(
                type="string",
                description="What to do with a day the month is too short for, such as 31 June: "
                "'clamp' to the month's last day, 'set_null', or 'error' (stop the run).",
                default="error",
                enum=_ON_INVALID_ENUM,
            ),
            "output": ConfigField(
                type="string",
                description="The name of the new date column, such as 'period_start'.",
                required=True,
                non_empty=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use assemble_date to build a real date from separate year, month and day values, "
                "for example a period given as year 2026 and month 3. The date goes in a new column "
                "at the end of `rows`. The day is 1 unless you give one. For the last day of each "
                "month, write day '31' with on_invalid 'clamp': each row gets its own month's last "
                "day, the 28th, 29th, 30th or 31st."
            ),
            examples=(_plain_example(), _clamp_example()),
            notes=(
                "The month must be a number. Month names are spelled many ways, so turn them into "
                "numbers first with a rule list or lookup someone can see and change.",
                "Each part must be a whole number, or text of digits only. '+5' or '5.0' is refused.",
                "'on_invalid' only covers a day the month is too short for. A month outside 1 to 12, "
                "a year no date can hold, or text that is not digits is refused under every setting.",
                "A null in any part gives a null date under every setting, including 'error'. A part "
                "nobody filled in is not the same as an impossible date.",
                "Under 'set_null', a row whose parts make no date gets a null date. A note counts "
                "those rows, and each is marked with the parts it held. A row with a missing part is "
                "not marked.",
                "An output name already in use is refused.",
            ),
            field_help={
                "day": "For example '[day]', or '31' with on_invalid 'clamp' for the end of the month.",
                "on_invalid": "The value is 'set_null', not 'null': an unquoted null in YAML means "
                "no value at all.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        if config["on_invalid"] is None:
            raise ConfigError(
                "config field 'on_invalid' was given with no value; write 'clamp', 'set_null' or "
                "'error', or leave the field out entirely to take its default"
            )
        if config["day"] is None:
            raise ConfigError(
                "config field 'day' was given with no value; leave it out to default to the first "
                "of the month, or write an expression for it"
            )
        require_not_blank(config["output"], "output")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a part that is not a whole number or text, and an output name in use."""
        frame = inputs["rows"]
        for part in ("year", "month", "day"):
            column, dtype = compiled_dtype(frame, config[part], part)
            if not (dtype.is_integer() or dtype == pl.String):
                raise ConfigError(
                    f"config field {part!r} must give a whole number or digits-only text, but "
                    f"column {column!r} holds {dtype}, in expression: `{config[part]}`"
                )
        check_output_column(frame, config["output"], "output", replace=None)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        policy = config["on_invalid"]

        for part in ("year", "month", "day"):
            refuse_rows_without_an_answer(frame, config[part], part)
        year_expr = compile_for(frame, config["year"], "year")
        month_expr = compile_for(frame, config["month"], "month")
        day_expr = compile_for(frame, config["day"], "day")

        staged = (
            frame.lazy()
            .with_columns(
                year_expr.alias("__year_raw"),
                month_expr.alias("__month_raw"),
                day_expr.alias("__day_raw"),
            )
            .collect()
        )

        year_value, year_bad = _read_part(staged["__year_raw"])
        month_value, month_bad = _read_part(staged["__month_raw"])
        day_value, day_bad = _read_part(staged["__day_raw"])

        staged = staged.with_columns(
            year_value.alias("__year"),
            month_value.alias("__month"),
            day_value.alias("__day"),
            year_bad.alias("__year_bad"),
            month_bad.alias("__month_bad"),
            day_bad.alias("__day_bad"),
        )

        year_col = pl.col("__year")
        month_col = pl.col("__month")
        day_col = pl.col("__day")
        text_bad = pl.col("__year_bad") | pl.col("__month_bad") | pl.col("__day_bad")

        invalid_month = month_col.is_not_null() & ((month_col < 1) | (month_col > 12))
        invalid_year = year_col.is_not_null() & ((year_col < MIN_YEAR) | (year_col > MAX_YEAR))
        hard_invalid = (text_bad | invalid_month | invalid_year).fill_null(False)

        days_in_month = month_length(year_col, month_col)

        day_out_of_range = (
            day_col.is_not_null()
            & ~pl.col("__day_bad")
            & ~invalid_month
            & ~invalid_year
            & ((day_col < 1) | (day_col > days_in_month))
        ).fill_null(False)

        if policy == "clamp":
            offending = hard_invalid
            resolved_day = pl.max_horizontal(pl.min_horizontal(day_col, days_in_month), pl.lit(1))
        else:
            offending = (hard_invalid | day_out_of_range).fill_null(False)
            resolved_day = day_col

        missing = (
            (year_col.is_null() & ~pl.col("__year_bad"))
            | (month_col.is_null() & ~pl.col("__month_bad"))
            | (day_col.is_null() & ~pl.col("__day_bad"))
        ).fill_null(False) & ~offending

        buildable = (~offending & ~missing).fill_null(False)

        staged = staged.with_columns(
            offending.alias("__offending"),
            buildable.alias("__buildable"),
            resolved_day.alias("__resolved_day"),
        )

        if policy != "set_null" and staged["__offending"].any():
            refusing = frame.filter(staged["__offending"])
            raise DataError(
                f"assemble_date: on_invalid={policy!r} cannot resolve every date part",
                offending=refusing,
                total=frame.height,
            )

        # The same build make_date() uses. A row with a missing part gets null.
        date_expr = pl.when(pl.col("__buildable")).then(date_or_null(year_col, month_col, pl.col("__resolved_day")))

        result = staged.with_columns(date_expr.alias(config["output"])).select([*frame.columns, config["output"]])
        if policy == "set_null" and staged["__offending"].any():
            reasons, note = _emptied_report(staged, config["output"])
            return Ports(
                {"rows": result}, notes=[note], status={"rows": row_status([(config["output"], reasons)], frame.height)}
            )
        return Ports({"rows": result})


def _emptied_report(staged: pl.DataFrame, output: str) -> tuple[pl.Series, str]:
    """One reason per row whose parts made no date under 'set_null', and the note counting them."""
    parts = pl.format(
        "year {}, month {}, day {}",
        value_caption("__year_raw"),
        value_caption("__month_raw"),
        value_caption("__day_raw"),
    )
    described = staged.select(
        pl.when(pl.col("__offending")).then(pl.format("{} do not make a date, so it is empty", parts)).alias("reason"),
        parts.alias("parts"),
    )
    emptied = int(staged["__offending"].sum())
    example = described.filter(staged["__offending"])["parts"][0]
    note = (
        f"{emptied:,} of {staged.height:,} rows got an empty {output!r} because their parts do not "
        f"make a date (for example {example})"
    )
    return described["reason"], note


def _read_part(series: pl.Series) -> tuple[pl.Expr, pl.Expr]:
    """The part as a whole number (or null), and whether it was text that was not digits alone.

    A digits check, not a cast: polars' lenient cast reads '+5' as 5, which this tool must refuse.
    """
    column = pl.col(series.name)
    if series.dtype == pl.String:
        digit_ok = column.str.contains(_DIGITS_ONLY)
        malformed = column.is_not_null() & ~digit_ok.fill_null(False)
        value = pl.when(digit_ok).then(column.cast(pl.Int64, strict=False)).otherwise(None)
        return value, malformed
    return column.cast(pl.Int64), pl.lit(False)


def _plain_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "year": [2026, 2025], "month": [1, 6]})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "year": [2026, 2025],
                    "month": [1, 6],
                    "period_start": [date(2026, 1, 1), date(2025, 6, 1)],
                },
                schema_overrides={"period_start": pl.Date},
            )
        }
    )
    return WorkedExample(
        scenario="build the first day of each month from a year and a month",
        configuration={"year": "[year]", "month": "[month]", "output": "period_start"},
        explanation="No 'day' is given, so each row gets day 1.",
        inputs=inputs,
        outputs=outputs,
    )


def _clamp_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "year": [2026, 2024], "month": [2, 2]})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "year": [2026, 2024],
                    "month": [2, 2],
                    "period_end": [date(2026, 2, 28), date(2024, 2, 29)],
                },
                schema_overrides={"period_end": pl.Date},
            )
        }
    )
    return WorkedExample(
        scenario="build the last day of each month",
        configuration={
            "year": "[year]",
            "month": "[month]",
            "day": "31",
            "on_invalid": "clamp",
            "output": "period_end",
        },
        explanation="'day: 31' with 'on_invalid: clamp' gives each row its own month's last day: the 28th in an "
        "ordinary February, the 29th in a leap year.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(AssembleDate())
