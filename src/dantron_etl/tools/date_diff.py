"""DateDiff: count the whole months or days between two dates.

`unit` and `counting` are required: `inclusive` counts the unit each end falls in, `elapsed` only
completed units. A backwards range counts negative; a missing date gives null.

Why: docs/design/tools.md#date_diff
"""

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    check_output_column,
    compile_for,
    compiled_dtype,
    count_between,
    refuse_rows_without_an_answer,
    require_not_blank,
    ZONE_FIELD,
    as_calendar_date,
    check_zone_name,
    require_stated_zone,
)

_UNIT_ENUM: Final[tuple[str, ...]] = ("month", "day")
_COUNTING_ENUM: Final[tuple[str, ...]] = ("inclusive", "elapsed")


class DateDiff:
    """Count the whole months or days between two dates, as a signed whole number."""

    name = "date_diff"
    description = "Count the months or days between two dates, in a new column."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "start": ConfigField(
                type="expression",
                description="The start date: a column such as '[start_date]', or an expression giving a date.",
                required=True,
            ),
            "end": ConfigField(
                type="expression",
                description="The end date, written the same way as 'start'.",
                required=True,
            ),
            "unit": ConfigField(
                type="string",
                description="What to count: 'month' or 'day'.",
                required=True,
                enum=_UNIT_ENUM,
            ),
            "counting": ConfigField(
                type="string",
                description="'inclusive' counts both end units (1 January to 31 January is 31 days); "
                "'elapsed' counts completed units only (30 days).",
                required=True,
                enum=_COUNTING_ENUM,
            ),
            "output": ConfigField(
                type="string",
                description="The name of the new count column, such as 'term_months'.",
                required=True,
                non_empty=True,
            ),
            "in_zone": ZONE_FIELD,
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use date_diff to count the months or days between two dates, for example the length "
                "of a contract term, often to divide an amount by it. The count goes in a new whole-"
                "number column at the end of `rows`. A month counts once the end's day of the month "
                "reaches the start's: 31 January to 28 February is one month, and 1 January to 31 "
                "December is twelve, counted inclusively."
            ),
            examples=(_plain_example(), _surprising_example()),
            notes=(
                "'unit' and 'counting' have no default, because the two ways of counting differ by "
                "one and a default would decide every divisor silently.",
                "'inclusive' counts the unit each end falls in, so it is never zero: the same day at "
                "both ends is one day, and one month.",
                "'elapsed' counts only completed units, exactly one less in either direction, like "
                "SQL's DATEDIFF or subtracting dates in Excel.",
                "A backwards range (end before start) gives a negative count of the same size. It is "
                "not repaired or refused.",
                "A missing date on either side gives a null count, never zero.",
                "A timestamp is read as the date it falls on. When it has a time zone, 'in_zone' must "
                "say which, because 23:30 on 31 January in Toronto is 1 February in UTC. There is no "
                "default; 'in_zone' is refused when neither side has a zone.",
                "A side that is not a date is refused, and so is an output name already in use.",
                "months_between and days_between in an expression give the same counts.",
            ),
            field_help={
                "counting": "Use 'inclusive' for a period stated first day to last day, such as a "
                "contract from 1 January to 31 December.",
                "output": "For example 'term_months'. It must be a new column name.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        if config["counting"] is None:
            raise ConfigError("config field 'counting' was given with no value; write 'inclusive' or 'elapsed'")
        require_not_blank(config["output"], "output")
        if config["in_zone"] is not None:
            check_zone_name(config["in_zone"], "in_zone")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a side that is not a date, a missing or unneeded zone, and an output name in use."""
        frame = inputs["rows"]
        for side in ("start", "end"):
            column, dtype = compiled_dtype(frame, config[side], side)
            if dtype.base_type() not in (pl.Date, pl.Datetime):
                raise ConfigError(
                    f"config field {side!r} must give a date, but column {column!r} holds {dtype}, "
                    f"in expression: `{config[side]}`"
                )
        _require_zone_where_needed(frame, config)
        check_output_column(frame, config["output"], "output", replace=None)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]

        for side in ("start", "end"):
            refuse_rows_without_an_answer(frame, config[side], side)
        start_expr, end_expr = (
            as_calendar_date(compile_for(frame, config[side], side), compiled_dtype(frame, config[side], side)[1], config["in_zone"])
            for side in ("start", "end")
        )

        staged = frame.lazy().with_columns(start_expr.alias("__start"), end_expr.alias("__end")).collect()

        # The same function months_between() and days_between() use.
        signed = count_between(pl.col("__start"), pl.col("__end"), config["unit"], config["counting"])

        result = staged.with_columns(signed.cast(pl.Int64).alias(config["output"])).select(
            [*frame.columns, config["output"]]
        )
        return Ports({"rows": result})


def _require_zone_where_needed(frame: pl.DataFrame, config: Mapping[str, Any]) -> None:
    """Require 'in_zone' when a side is a zoned timestamp, and refuse it when neither side is."""
    zoned = False
    for side in ("start", "end"):
        column, dtype = compiled_dtype(frame, config[side], side)
        require_stated_zone(dtype, config["in_zone"], "in_zone", reading=f"config field {side!r} (column {column!r})")
        zoned = zoned or (isinstance(dtype, pl.Datetime) and dtype.time_zone is not None)
    if config["in_zone"] is not None and not zoned:
        raise ConfigError(
            "config field 'in_zone' is set, but neither 'start' nor 'end' is a timestamp that carries a "
            "time zone; there is no zone to read them in, so remove it"
        )


def _plain_example() -> WorkedExample:
    inputs = Ports(
        {"rows": pl.DataFrame({"key": [1], "from_date": [date(2026, 1, 1)], "to_date": [date(2026, 12, 31)]})}
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1],
                    "from_date": [date(2026, 1, 1)],
                    "to_date": [date(2026, 12, 31)],
                    "counted": pl.Series([12], dtype=pl.Int64),
                }
            )
        }
    )
    return WorkedExample(
        scenario="count the months in a one-year term",
        configuration={
            "start": "[from_date]",
            "end": "[to_date]",
            "unit": "month",
            "counting": "inclusive",
            "output": "counted",
        },
        explanation="1 January to 31 December, counted inclusively, is twelve months: both ends count, and the "
        "end's day (31) has reached the start's (1).",
        inputs=inputs,
        outputs=outputs,
    )


def _surprising_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "from_date": [date(2026, 1, 1), None],
                    "to_date": [date(2025, 12, 31), date(2026, 3, 1)],
                },
                schema_overrides={"from_date": pl.Date, "to_date": pl.Date},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "from_date": [date(2026, 1, 1), None],
                    "to_date": [date(2025, 12, 31), date(2026, 3, 1)],
                    "counted": pl.Series([-1, None], dtype=pl.Int64),
                },
                schema_overrides={"from_date": pl.Date, "to_date": pl.Date},
            )
        }
    )
    return WorkedExample(
        scenario="a backwards range counts negative, and a missing date counts null",
        configuration={
            "start": "[from_date]",
            "end": "[to_date]",
            "unit": "month",
            "counting": "inclusive",
            "output": "counted",
        },
        explanation="The first row runs backwards by one month, so it counts -1; it is not repaired. The second "
        "row has no start date, so its count is null, not zero.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(DateDiff())
