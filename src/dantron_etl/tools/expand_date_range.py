"""ExpandDateRange: turn each row into one row per month from `start` to `end`, both included.

The only tool that multiplies rows. A row with a missing date, or an end before its start, goes to
`unexpanded` with the input's columns, never dropped and never given a null period.

Why: docs/design/tools.md#expand_date_range
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
    compile_for,
    compiled_dtype,
    refuse_division_by_zero,
    require_not_blank,
    ZONE_FIELD,
    as_calendar_date,
    check_zone_name,
    require_stated_zone,
)

# Only month for now. Why: docs/design/tools.md#expand_date_range
_UNIT_ENUM: Final[tuple[str, ...]] = ("month",)


class ExpandDateRange:
    """Expand each row of `rows` into one row per month between `start` and `end`, inclusive."""

    name = "expand_date_range"
    description = "Turn each row with a start and end date into one row per month between them."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows", "unexpanded")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "start": ConfigField(
                type="expression", description="The first date of the range, such as '[term_start]'.", required=True
            ),
            "end": ConfigField(
                type="expression", description="The last date of the range, such as '[term_end]'.", required=True
            ),
            "unit": ConfigField(
                type="string",
                description="The period each row is split into. Only 'month' is offered.",
                default="month",
                enum=_UNIT_ENUM,
            ),
            "period_column": ConfigField(
                type="string",
                description="The name of the new column holding each period's first day.",
                default="period_start",
            ),
            "in_zone": ZONE_FIELD,
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use expand_date_range to turn one row per contract or term into one row per month it "
                "covers, for example to spread a yearly amount over its months. `rows` has one row "
                "per month from 'start' to 'end', both included, with the other columns repeated and "
                "the month's first day in a new column at the end. A row that cannot be expanded goes "
                "to `unexpanded`, unchanged."
            ),
            examples=(_clean_term_example(), _unexpanded_example()),
            notes=(
                "Both ends count: January to December is twelve rows, however far into its month "
                "each date falls. The same day at both ends is one row.",
                "A row with a missing date, or an end before its start, goes to `unexpanded`. It is "
                "never dropped and never given a null period.",
                "Backwards is judged on the dates, not the months, so two dates in the same month the "
                "wrong way round also go to `unexpanded`. That agrees with date_diff's negative count.",
                "A timestamp is read as its calendar date first, as in date_diff, so two times on the "
                "same day are the same day whichever comes first.",
                "`unexpanded` has exactly the input's columns, with no period or reason column, so it "
                "can be unioned back with other input rows.",
                "Amounts are repeated on every month row, not split. Divide them in a later step, for "
                "example by a date_diff count.",
                "A 'period_column' name already on the input is refused.",
            ),
            field_help={
                "start": "A date or a timestamp. A timestamp with a time zone also needs 'in_zone'.",
                "end": "A date or a timestamp. A timestamp with a time zone also needs 'in_zone'.",
                "in_zone": "For example 'UTC'. Needed when 'start' or 'end' is a timestamp with a time "
                "zone, because the day depends on the zone; refused otherwise.",
                "period_column": "For example 'month'. It must be a new column name.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        require_not_blank(config["period_column"], "period_column")
        if config["in_zone"] is not None:
            check_zone_name(config["in_zone"], "in_zone")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse ends that are not dates, a missing or unneeded zone, and a period column name in use."""
        frame = inputs["rows"]
        period_column = config["period_column"]
        if period_column in frame.columns:
            raise ConfigError(
                f"config field 'period_column' names column {period_column!r}, which the input "
                f"already carries; adding a period column over one already there would overwrite "
                f"data nobody asked it to. Choose a name nothing on 'rows' uses."
            )

        for field in ("start", "end"):
            column, dtype = compiled_dtype(frame, config[field], field)
            if dtype.base_type() not in (pl.Date, pl.Datetime):
                raise ConfigError(
                    f"config field {field!r} produces column {column!r}, which holds {dtype}, not "
                    f"a date, in expression: `{config[field]}`"
                )
        _require_zone_where_needed(frame, config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        # Timestamps become calendar dates first, so two times on one day are not a backwards range.
        for side in ("start", "end"):
            refuse_division_by_zero(frame, config[side], side)
        start, end = (
            as_calendar_date(compile_for(frame, config[side], side), compiled_dtype(frame, config[side], side)[1], config["in_zone"])
            for side in ("start", "end")
        )

        valid = start.is_not_null() & end.is_not_null() & (end >= start)

        period_start = pl.date_ranges(
            start.dt.month_start(),
            end.dt.month_start(),
            interval="1mo",
            eager=False,
        ).alias(config["period_column"])

        rows = frame.filter(valid).with_columns(period_start).explode(config["period_column"])
        unexpanded = frame.filter(~valid)

        return Ports({"rows": rows, "unexpanded": unexpanded})


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


def _clean_term_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1],
                    "amount": [120.0],
                    "term_start": [date(2026, 1, 1)],
                    "term_end": [date(2026, 3, 31)],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 1, 1],
                    "amount": [120.0, 120.0, 120.0],
                    "term_start": [date(2026, 1, 1)] * 3,
                    "term_end": [date(2026, 3, 31)] * 3,
                    "period_start": [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)],
                }
            ),
            "unexpanded": pl.DataFrame(
                {
                    "key": pl.Series([], dtype=pl.Int64),
                    "amount": pl.Series([], dtype=pl.Float64),
                    "term_start": pl.Series([], dtype=pl.Date),
                    "term_end": pl.Series([], dtype=pl.Date),
                }
            ),
        }
    )
    return WorkedExample(
        scenario="turn a three-month term into one row per month",
        configuration={"start": "[term_start]", "end": "[term_end]"},
        explanation="January to March, both included, gives three rows. The other columns are repeated on each, "
        "and the period column is added at the end.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _unexpanded_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "amount": [50.0, 10.0],
                    "term_start": [date(2026, 1, 1), date(2026, 5, 1)],
                    "term_end": [date(2026, 2, 28), date(2026, 1, 31)],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 1],
                    "amount": [50.0, 50.0],
                    "term_start": [date(2026, 1, 1)] * 2,
                    "term_end": [date(2026, 2, 28)] * 2,
                    "period_start": [date(2026, 1, 1), date(2026, 2, 1)],
                }
            ),
            "unexpanded": pl.DataFrame(
                {
                    "key": [2],
                    "amount": [10.0],
                    "term_start": [date(2026, 5, 1)],
                    "term_end": [date(2026, 1, 31)],
                }
            ),
        }
    )
    return WorkedExample(
        scenario="set aside a term whose end is before its start",
        configuration={"start": "[term_start]", "end": "[term_end]"},
        explanation="Key 1 expands as usual. Key 2's end is before its start, so it gives no rows on 'rows' and "
        "goes to 'unexpanded' with its own columns.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(ExpandDateRange())
