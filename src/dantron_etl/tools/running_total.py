"""RunningTotal: add running sums of number columns, in the order the rows arrive, per group if asked.

It mirrors SQL's SUM(col) OVER (PARTITION BY group ORDER BY <arrival> ROWS UNBOUNDED PRECEDING).
It never sorts. A null adds nothing. A total that does not fit its type stops the step.

Why: docs/design/tools.md#exact-totals
"""

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    require_columns,
    require_not_blank,
)
from ..column_types import EXACT_DIGITS, exact_dtype
from .group_by import describe_rows, exact_results, exact_running_parts

_INT64_MIN: Final[int] = -(2**63)
_INT64_MAX: Final[int] = 2**63 - 1


class RunningTotal:
    """Add one running-total column to `rows` per entry in 'columns', restarting for each group."""

    name = "running_total"
    description = "Add running totals of number columns, in row order, starting again for each group."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="The running totals to add, one entry per new column.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One running total: the number column it adds up, and the new column's name.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The number column to add up, such as 'amount'.",
                            required=True,
                            non_empty=True,
                        ),
                        "output": ConfigField(
                            type="string",
                            description="The name of the new column, such as 'balance'.",
                            required=True,
                            non_empty=True,
                        ),
                    },
                ),
            ),
            "group_by": ConfigField(
                type="array",
                description="The columns whose values start the total again. Left out, one total runs over every row.",
                required=False,
                items=ConfigField(
                    type="string",
                    description="One column, such as 'account'.",
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use running_total for a running balance: each row gets the total of a number column "
                "so far, like Alteryx's Running Total. Name 'group_by' columns to start the total "
                "again for each account or customer. The rows keep their order, and each new column "
                "goes on the end, in the order listed. The total runs in the order the rows arrive, "
                "so put a sort in front for a meaningful order, such as by date."
            ),
            examples=(_balance_example(), _nulls_example()),
            notes=(
                "This step does not sort. Without a sort in front, the totals follow whatever order "
                "the rows happened to be in.",
                "It works as SQL's SUM(col) OVER (PARTITION BY group ORDER BY <arrival> ROWS UNBOUNDED PRECEDING).",
                "A group's rows need not be next to each other. Each group's total runs over its own "
                "rows, in the order they arrive.",
                "Rows whose 'group_by' value is null form one group of their own, as in SQL's PARTITION BY.",
                "A null adds nothing: null, 5, null, 3 gives null, 5, 5, 8. A total is null only "
                "before its group's first value, so a group of nothing but nulls stays null, not zero.",
                "A whole-number column gives Int64 totals. A total past Int64's range stops the run, "
                "naming the group and the column, instead of wrapping round to a wrong number.",
                "A float column gives Float64 totals.",
                "An exact Decimal column gives exact totals at its places, with 38 digits in all. A "
                "total that needs more than 38 digits stops the run, naming the group and the column.",
                "Every running value must fit, not only the last: a total that goes past the limit and "
                "comes back still stops the run.",
                "A column that is not a number is refused. So is an 'output' name the input already "
                "has, or one used twice. There is no setting to overwrite a column.",
            ),
            field_help={
                "columns": "For example [{column: amount, output: balance}]. Each entry adds one new "
                "column at the end.",
                "group_by": "For example ['account']. Leave it out for one total over every row.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a blank name, and an output name used twice."""
        seen: set[str] = set()
        for index, entry in enumerate(config["columns"]):
            require_not_blank(entry["column"], f"columns[{index}].column")
            require_not_blank(entry["output"], f"columns[{index}].output")
            if entry["output"] in seen:
                raise ConfigError(
                    f"config field 'columns[{index}].output' is {entry['output']!r}, a name an earlier "
                    f"entry already uses; give each running total its own name"
                )
            seen.add(entry["output"])

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a missing column, a column that is not a number, and an output name already in use."""
        frame = inputs["rows"]
        require_columns(frame, _group_columns(config), "group_by", "rows")
        for index, entry in enumerate(config["columns"]):
            column, output = entry["column"], entry["output"]
            require_columns(frame, [column], f"columns[{index}].column", "rows")
            dtype = frame.schema[column]
            if not dtype.is_numeric():
                raise ConfigError(
                    f"config field 'columns[{index}].column' names column {column!r}, which holds "
                    f"{dtype}, not a number; a running total adds up whole numbers, floats or exact "
                    f"decimals. Convert it with cast_columns in a step before this one"
                )
            if output in frame.columns:
                raise ConfigError(
                    f"config field 'columns[{index}].output' is {output!r}, which the input already "
                    f"has as a column; write the running total to a name nothing uses"
                )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        within = _group_columns(config)
        totals = [_running_total(frame, index, entry, within) for index, entry in enumerate(config["columns"])]
        return Ports({"rows": frame.with_columns(totals)})


def _group_columns(config: Mapping[str, Any]) -> list[str]:
    """The group columns, a column named twice counted once, in the order first written."""
    return list(dict.fromkeys(config.get("group_by") or []))


def _running(expression: pl.Expr, within: list[str]) -> pl.Expr:
    """``expression``'s running total in row order, per group; a null adds nothing."""
    total = expression.cum_sum().forward_fill()
    return total.over(within) if within else total


def _running_total(frame: pl.DataFrame, index: int, entry: Mapping[str, Any], within: list[str]) -> pl.Series:
    """One entry's running total, or a refusal where a running value does not fit its type."""
    column, output = entry["column"], entry["output"]
    dtype = frame.schema[column]
    if dtype.is_float():
        return frame.select(_running(pl.col(column).cast(pl.Float64), within).alias(output)).to_series()
    if isinstance(dtype, pl.Decimal):
        places = dtype.scale or 0
        parts = frame.select(exact_running_parts(pl.col(column), within).alias(output)).to_series()
        values, too_large = exact_results(parts, "sum", places)
        if too_large.any():
            _refuse(
                frame.filter(too_large),
                frame.height,
                index,
                column,
                within,
                f"needs more than {EXACT_DIGITS} digits, the most an exact column holds: at {places} places "
                f"that is {EXACT_DIGITS - places} before the point, so no total is given. Convert the column "
                f"to Float64 with cast_columns for figures this large",
            )
        return values.cast(exact_dtype(places))
    total = frame.select(_running(pl.col(column).cast(pl.Int128), within).alias(output)).to_series()
    outside = total.is_not_null() & ~total.is_between(_INT64_MIN, _INT64_MAX)
    if outside.any():
        _refuse(
            frame.with_columns(total).filter(outside),
            frame.height,
            index,
            column,
            within,
            f"does not fit in Int64, which holds {_INT64_MIN:,} to {_INT64_MAX:,}, and would have wrapped "
            f"round to a wrong number, so no total is given. Convert the column to Decimal with places 0 "
            f"with cast_columns for whole numbers this large; the totals shown are the true ones",
        )
    return total.cast(pl.Int64)


def _refuse(offending: pl.DataFrame, total: int, index: int, column: str, within: list[str], why: str) -> None:
    """Stop the step: the running total of ``column`` does not fit, first in the group of ``offending``'s first row."""
    where = f"in the group {describe_rows(offending, within)}" if within else "over all the rows"
    raise DataError(
        f"config field 'columns[{index}]' adds up column {column!r}, and its running total {where} {why}",
        offending=offending,
        total=total,
    )


def _exact(values: list[str]) -> pl.Series:
    """Exact amounts at 2 places, written as text so nothing passes through a float."""
    return pl.Series([Decimal(value) for value in values], dtype=exact_dtype(2))


def _balance_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "account": ["A", "A", "B", "A", "B"],
                    "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 3)],
                    "amount": _exact(["100.00", "-30.00", "20.00", "50.00", "5.00"]),
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "account": ["A", "A", "B", "A", "B"],
                    "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 3)],
                    "amount": _exact(["100.00", "-30.00", "20.00", "50.00", "5.00"]),
                    "balance": _exact(["100.00", "70.00", "20.00", "120.00", "25.00"]),
                }
            )
        }
    )
    return WorkedExample(
        scenario="a running balance for each account",
        configuration={"group_by": ["account"], "columns": [{"column": "amount", "output": "balance"}]},
        explanation="Each account's balance runs over its own rows, in the order they arrive. A's rows are "
        "the 1st, 2nd and 4th, so its balance goes 100.00, 70.00, 120.00; B's goes 20.00, 25.00. The rows do not move, and an exact "
        "amount gives an exact balance.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _nulls_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"amount": pl.Series([None, 5.5, None, 3.0], dtype=pl.Float64)})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "amount": pl.Series([None, 5.5, None, 3.0], dtype=pl.Float64),
                    "total": pl.Series([None, 5.5, 5.5, 8.5], dtype=pl.Float64),
                }
            )
        }
    )
    return WorkedExample(
        scenario="one running total over every row, with nulls",
        configuration={"columns": [{"column": "amount", "output": "total"}]},
        explanation="With no 'group_by', one total runs over every row. A null adds nothing: the first "
        "row has no value yet, so its total is null, and the third row keeps 5.5.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(RunningTotal())
