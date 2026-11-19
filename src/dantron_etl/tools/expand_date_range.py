"""ExpandDateRange: one row in, one row per period out -- the one tool that multiplies row count.

**Both ends are inclusive, at period granularity.** January to December is twelve rows, not eleven
and not thirteen: the period containing the start counts however far into it the start falls, and
the period containing the end counts the same way, because an end spelled as the last day of a
month is the overwhelmingly common spelling and truncating it would drop a whole period's money from
every such row. The same day at both ends is one row, never zero -- a single day still sits inside
the period it belongs to.

**A backwards range produces no rows and leaves on ``unexpanded``,** judged on the dates themselves
and not on the periods they fall in. A range whose end sits before its start is real data -- a
mistyped year, a transposed day and month -- not an impossibility, and the fix belongs to a repair
step where it can be reviewed, not to the arithmetic here. Two ends inside the same period, the
wrong way round, are backwards too: period arithmetic alone would give that row one perfectly
plausible period, which is exactly the shape a transposed day and month makes, so the comparison has
to run on the dates before anything is truncated to a period.

**A row that cannot be expanded must not vanish, and must not become a null period either.** The
obvious implementation explodes an empty period list into one row whose period is null, which is
worse than dropping the row: it looks like a schedule row, it carries the row's whole figure, and it
sits in a report as a period that does not exist. So such a row is set aside on ``unexpanded``
instead, carrying the input's schema exactly -- no period column, no reason column, because the
reason is one predicate away downstream and a reason column would stop the port being
union-compatible with the input it came from.
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

# One member, on purpose: every other unit brings its own boundary question -- what a partial
# week or a partial year is worth -- that nothing has asked for yet, and a whitelist with an
# untested arm in it is worse than a short one. Spelled to avoid every word YAML reads as a boolean.
_UNIT_ENUM: Final[tuple[str, ...]] = ("month",)


class ExpandDateRange:
    """Expand each row of `rows` into one row per period between `start` and `end`, inclusive."""

    name = "expand_date_range"
    description = "Expand each row into one row per period between two dates; set the rest aside."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows", "unexpanded")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "start": ConfigField(type="expression", description="The range's first date, inclusive.", required=True),
            "end": ConfigField(type="expression", description="The range's last date, inclusive.", required=True),
            "unit": ConfigField(
                type="string",
                description="The period a range is broken into; only 'month' ships.",
                default="month",
                enum=_UNIT_ENUM,
            ),
            "period_column": ConfigField(
                type="string",
                description="The name of the added column carrying each period's first day.",
                default="period_start",
            ),
            "in_zone": ZONE_FIELD,
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every row between 'start' and 'end', inclusive of both, becomes one row per month "
                "on 'rows', with every other column carried unchanged and the period column added "
                "last. A row this cannot expand -- a missing end, or an end before its start -- is "
                "never dropped and never given a null period; it leaves on 'unexpanded' instead, "
                "carrying the input's schema exactly."
            ),
            examples=(_clean_term_example(), _unexpanded_example()),
            notes=(
                "A backwards range is judged on the dates, not on the periods they fall in: two "
                "ends inside the same period the wrong way round still leave on 'unexpanded', which "
                "is what makes this agree with DateDiff's negative count on the same row. A "
                "timestamp is read as its calendar date first, in both tools, so two times on the "
                "same day are one day whichever comes first.",
                "'unexpanded' carries no period column and no reason column; the reason is one "
                "predicate away downstream, and adding one would stop the port being "
                "union-compatible with the rows that arrived.",
            ),
            field_help={
                "start": "Must produce a date or a timestamp; a timestamp is read as its calendar date, "
                "in the zone 'in_zone' states when it carries one.",
                "end": "Must produce a date or a timestamp; a timestamp is read as its calendar date, "
                "in the zone 'in_zone' states when it carries one.",
                "in_zone": "Required when 'start' or 'end' is a timestamp in a time zone, because which "
                "day it falls on depends on the zone. 'UTC' reads the UTC date. Refused otherwise.",
                "period_column": "Refused if the input already carries a column of that name.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        require_not_blank(config["period_column"], "period_column")
        if config["in_zone"] is not None:
            check_zone_name(config["in_zone"], "in_zone")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """The two questions the schema alone can answer: are the ends dates, and is the name free.

        Both are refused here rather than in ``execute``, so a range built from a number or a piece
        of text, or a period column that would overwrite one already there, is reported as a
        configuration fault before a single row moves -- at a frame of a million rows or of none.
        """
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
        # A timestamp is read as its calendar date before anything else, as date_diff reads it.
        # So 15:00 and 09:00 on one day are the same day, not a backwards range.
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
        scenario="a clean term expands into one row per month it covers",
        configuration={"start": "[term_start]", "end": "[term_end]"},
        explanation="January through March, inclusive of both ends, is three rows; every other "
        "column is repeated onto each and the period column is added last.",
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
        scenario="a backwards term is set aside instead of vanishing or becoming a null period",
        configuration={"start": "[term_start]", "end": "[term_end]"},
        explanation="Key 1's term expands normally; key 2's end falls before its start, so it "
        "contributes no rows to 'rows' and leaves on 'unexpanded' carrying the input's own columns.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(ExpandDateRange())
