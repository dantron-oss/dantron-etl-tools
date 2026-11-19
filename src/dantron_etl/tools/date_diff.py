"""DateDiff: how many whole units lie between two dates.

This count is a divisor. A figure billed for a period is divided by the number of units that
period spans to get a figure per unit, so a count that is one too many understates every unit and a
count that is one too few overstates it -- neither announces itself, because the figure still looks
like a figure.

A month is counted once the end's day of the month has reached the start's: January 31 to February
28 is one month, and January 1 to December 31 is a full twelve. `counting` is required and says
which ends contribute a unit: `inclusive` counts the unit each end falls in and can never come out
as zero; `elapsed` counts only completed units, one less in magnitude in both directions, like SQL's
DATEDIFF and Excel's subtraction. A backwards range counts
negative and keeps its size, because which end was mistyped is a judgement for a repair step and not
this tool's to make. A missing date on either side gives a null count, never a zero, under every
combination of unit and counting.
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
    refuse_division_by_zero,
    require_not_blank,
    ZONE_FIELD,
    as_calendar_date,
    check_zone_name,
    require_stated_zone,
)

_UNIT_ENUM: Final[tuple[str, ...]] = ("month", "day")
_COUNTING_ENUM: Final[tuple[str, ...]] = ("inclusive", "elapsed")


class DateDiff:
    """Count the whole months or days between two dates, signed and never zero by accident."""

    name = "date_diff"
    description = "Count the whole months or days between two dates, appended as a signed integer."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "start": ConfigField(
                type="expression",
                description="The range's start: a column, a fixed value or a computed expression giving a date.",
                required=True,
            ),
            "end": ConfigField(
                type="expression",
                description="The range's end, in the same shape as 'start'.",
                required=True,
            ),
            "unit": ConfigField(
                type="string",
                description="What is counted: 'month' or 'day'. No default -- a divisor's unit is "
                "never inferred from silence.",
                required=True,
                enum=_UNIT_ENUM,
            ),
            "counting": ConfigField(
                type="string",
                description="Which ends contribute a unit: 'inclusive' (both ends count, the reading "
                "a stated period takes) or 'elapsed' (completed units only). No default.",
                required=True,
                enum=_COUNTING_ENUM,
            ),
            "output": ConfigField(
                type="string",
                description="The name of the count column to add; it may not replace a column that is already there.",
                required=True,
                non_empty=True,
            ),
            "in_zone": ZONE_FIELD,
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Counts the whole months or days between 'start' and 'end', appended as a signed "
                "integer after every input column. A month counts once the end's day of the month "
                "has reached the start's day -- so January 31 to February 28 is one month, not two, "
                "and January 1 to December 31 is a full twelve. A day counts each calendar date at "
                "both ends the same way, so under 'inclusive' the same day at both ends is one day "
                "and one month alike."
            ),
            examples=(_plain_example(), _surprising_example()),
            notes=(
                "'counting' has no default, because the two readings differ by one and a default "
                "would decide every divisor silently. 'inclusive' counts the unit each end falls "
                "in and can never come out as zero; 'elapsed' counts only completed units and is "
                "exactly one less, in either direction, like SQL's DATEDIFF.",
                "A backwards range counts negative and keeps its size, because which end was "
                "mistyped is a judgement for a repair step and not this tool's to make.",
                "A timestamp is read as the date it falls on. When it carries a time zone, 'in_zone' "
                "must say in which zone, because 23:30 on 31 January in Toronto is 1 February in "
                "UTC; 'in_zone: UTC' reads the UTC date. There is no default.",
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        if config["counting"] is None:
            raise ConfigError("config field 'counting' was given with no value; write 'inclusive' or 'elapsed'")
        require_not_blank(config["output"], "output")
        if config["in_zone"] is not None:
            check_zone_name(config["in_zone"], "in_zone")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a side that is not a date, off the schema alone."""
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
            refuse_division_by_zero(frame, config[side], side)
        start_expr, end_expr = (
            as_calendar_date(compile_for(frame, config[side], side), compiled_dtype(frame, config[side], side)[1], config["in_zone"])
            for side in ("start", "end")
        )

        staged = frame.lazy().with_columns(start_expr.alias("__start"), end_expr.alias("__end")).collect()

        # The same count months_between() and days_between() give, from the one shared function.
        signed = count_between(pl.col("__start"), pl.col("__end"), config["unit"], config["counting"])

        result = staged.with_columns(signed.cast(pl.Int64).alias(config["output"])).select(
            [*frame.columns, config["output"]]
        )
        return Ports({"rows": result})


def _require_zone_where_needed(frame: pl.DataFrame, config: Mapping[str, Any]) -> None:
    """'in_zone' is needed when a side is a timestamp in a zone, and refused when neither side is."""
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
        scenario="a term stated first day to last day is a full twelve months",
        configuration={
            "start": "[from_date]",
            "end": "[to_date]",
            "unit": "month",
            "counting": "inclusive",
            "output": "counted",
        },
        explanation="Both ends contribute a month under 'inclusive', and the end's day (31) has reached "
        "the start's (1), so nothing is subtracted.",
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
        scenario="a mistyped end year counts negative; a missing start counts null",
        configuration={
            "start": "[from_date]",
            "end": "[to_date]",
            "unit": "month",
            "counting": "inclusive",
            "output": "counted",
        },
        explanation="The first row runs backwards by one month and is reported as -1, not repaired "
        "or refused; the second row has no start date at all, and counts null rather than zero.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(DateDiff())
