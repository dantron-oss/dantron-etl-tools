"""AssembleDate: separate year, month and day parts become one real date column.

The source of a date is often not a date. It is text an earlier step has already pulled apart --
a year here, a month there, and frequently no day at all, because a period that names only a month
never stated one. This tool is the step that turns those parts into a `pl.Date` column, so that
everything downstream that compares, sorts or counts between dates works from a date the parts
actually named rather than a date near it.

The day defaults to the first of the month -- the only day every month has, and so the one default
that does not quietly depend on the month's length. The other end of a whole month is instead
written down: `day: '31'` with `on_invalid: clamp`, which resolves per row to whatever day that
row's month actually ends on. `on_invalid` governs a day the calendar is too short for, and nothing
else about a part that names an impossible date: a month outside 1-12, a year no calendar date
could hold, and a part written as anything other than digits are all refused the same way under
`clamp` as under `error`, because none of them is a day the calendar was short of. A missing part is
a different question again -- a null in any part gives a null date under every policy, `error`
included, because absent data and impossible data deserve two different answers.
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
    require_not_blank,
)
from ..authoring import days_in_month as month_length

_ON_INVALID_ENUM: Final[tuple[str, ...]] = ("clamp", "set_null", "error")
_DIGITS_ONLY: Final[str] = r"^[0-9]+$"


class AssembleDate:
    """Build a real date column from separate year, month and day parts."""

    name = "assemble_date"
    description = "Assemble year, month and day parts into one real date column, appended to the input."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "year": ConfigField(
                type="expression",
                description="The year part: a column, a fixed value or a computed expression.",
                required=True,
            ),
            "month": ConfigField(
                type="expression",
                description="The month part, numbered 1-12. A month name is not read here.",
                required=True,
            ),
            "day": ConfigField(
                type="expression",
                description="The day-of-month part. Defaults to '1', the only day every month has.",
                default="1",
            ),
            "on_invalid": ConfigField(
                type="string",
                description="What to do with a day the calendar is too short for: 'clamp' it into "
                "the month, carry 'set_null' (a null date), or 'error' (the default).",
                default="error",
                enum=_ON_INVALID_ENUM,
            ),
            "output": ConfigField(
                type="string",
                description="The name of the date column to add; it may not replace a column that is already there.",
                required=True,
                non_empty=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Turns separate year, month and day parts into one real `pl.Date` column, appended "
                "after every input column. The day defaults to the first of the month; the last day "
                "of a month is instead written down as `day: '31'` with `on_invalid: clamp`, which "
                "resolves per row to whatever day that row's own month ends on -- the 28th, 29th, "
                "30th or 31st."
            ),
            examples=(_plain_example(), _clamp_example()),
            notes=(
                "A month is a number, never a name: the source's own spellings of a month name are "
                "a policy decision for a rule list or lookup a reviewer can see and change, not "
                "something this tool guesses at.",
                "A null part gives a null date under every policy, including 'error': a part nobody "
                "supplied is a different question from a part that names an impossible date, and "
                "conflating them would fail a whole run over rows nobody filled in.",
            ),
            field_help={
                "on_invalid": "Spelled 'set_null', never 'null': an unquoted null in a definition file "
                "is the null value itself, not a name for this policy.",
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
        """Refuse a part that is not a whole number or digit text, off the schema alone."""
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

        # The same build make_date() does, from the one shared function. Every row left is buildable
        # or missing a part, and date_or_null gives null for the second.
        date_expr = pl.when(pl.col("__buildable")).then(date_or_null(year_col, month_col, pl.col("__resolved_day")))

        result = staged.with_columns(date_expr.alias(config["output"])).select([*frame.columns, config["output"]])
        return Ports({"rows": result})


def _read_part(series: pl.Series) -> tuple[pl.Expr, pl.Expr]:
    """(the part as a nullable whole number, whether it was non-null text that was not digits alone).

    A digits check rather than a cast, because the engine's own non-strict cast reads a signed
    spelling like ``'+5'`` as 5, which is exactly the reading this tool must refuse.
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
        scenario="a period stated only by month defaults to its first day",
        configuration={"year": "[year]", "month": "[month]", "output": "period_start"},
        explanation="No 'day' is configured, so each row takes the default '1' -- the only day every month has.",
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
        scenario="the last day of the month, written once and resolved per row",
        configuration={
            "year": "[year]",
            "month": "[month]",
            "day": "31",
            "on_invalid": "clamp",
            "output": "period_end",
        },
        explanation="'day: 31' names the last day of the month; 'on_invalid: clamp' resolves it to "
        "the 28th in an ordinary February and the 29th in a leap one, from the same fixed value.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(AssembleDate())
