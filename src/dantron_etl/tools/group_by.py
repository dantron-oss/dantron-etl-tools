"""GroupBy: one row per group, with the summaries the configuration names.

The summaries (sum, count, max ...) are chosen from a fixed list, `AGGREGATE_FUNCTIONS`, and each
reads an expression. A null 'by' value is a group of its own, as in SQL's GROUP BY. A sum of nothing
but nulls is null. Groups keep the order they first appear in.

Why: docs/design/tools.md#group_by
"""

from collections.abc import Mapping
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
    compile_for,
    compiled_dtype,
    references,
    refuse_division_by_zero,
    require_columns,
    require_not_blank,
)

# The aggregates a flow may name. cross_tab offers the same list. Why no mean:
# docs/design/tools.md#group_by
AGGREGATE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"sum", "count", "count_non_null", "count_distinct", "min", "max", "first", "first_non_null", "join_text"}
)

# The one aggregate that reads no column.
_NO_VALUE_FUNCTION: Final[str] = "count"

# The one aggregate that reads a second field, 'separator'.
_JOINING_FUNCTION: Final[str] = "join_text"

# The default separator for join_text, shared with cross_tab. Defaults inside a list element are
# never filled in by the config template, so this is the only place it is written.
DEFAULT_SEPARATOR: Final[str] = ", "

# The whole-number types polars sums into themselves, where a large total wraps round silently.
# Their sums are worked out as Int64, checked against these limits, and cast back. Shared with cross_tab.
NARROW_SUM_LIMITS: Final[dict[pl.DataType, tuple[int, int]]] = {
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.UInt32: (0, 2**32 - 1),
}


def sum_without_wrapping(expression: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """The sum of ``expression``, worked out as Int64 when its type would wrap past its limit."""
    return expression.cast(pl.Int64).sum() if dtype in NARROW_SUM_LIMITS else expression.sum()


def totals_that_do_not_fit(frame: pl.DataFrame, column: str, dtype: pl.DataType) -> pl.DataFrame:
    """The rows of ``frame`` whose ``column`` lies outside what ``dtype`` can hold."""
    low, high = NARROW_SUM_LIMITS[dtype]
    return frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_between(low, high))


class GroupBy:
    """One row per distinct combination of `by`, with the aggregations named in the configuration."""

    name = "group_by"
    description = "Make one row per group, with totals, counts or other summaries for each group."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="The columns to group by. Rows with the same values here become one row.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One column to group by. Naming a column twice counts it once.",
                ),
            ),
            "aggregations": ConfigField(
                type="array",
                description="The columns to work out for each group, one entry per output column.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One output column: its name, the function, and what the function reads.",
                    fields={
                        "name": ConfigField(type="string", description="The output column's name.", required=True),
                        "function": ConfigField(
                            type="string",
                            description="What to work out, such as 'sum', 'count' or 'max'.",
                            required=True,
                            enum=tuple(sorted(AGGREGATE_FUNCTIONS)),
                        ),
                        "value": ConfigField(
                            type="expression",
                            description="The column or expression the function reads, such as "
                            "'[amount]'. Leave it out for 'count', which counts rows.",
                            required=False,
                        ),
                        "separator": ConfigField(
                            type="string",
                            description="For 'join_text' only: what goes between the joined values. "
                            "The default is a comma and a space.",
                            required=False,
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use group_by for subtotals: one row per customer, per month, or per any mix of "
                "columns, like a pivot table's row labels or SQL's GROUP BY. The output `rows` has "
                "the 'by' columns, then one column per aggregation. Every other column is dropped. "
                "Groups come out in the order each one first appears in the input, not sorted."
            ),
            examples=(_total_and_count_example(), _carried_column_example(), _joined_text_example()),
            notes=(
                "Rows whose 'by' value is null form one group of their own, as in SQL's GROUP BY. "
                "They are never dropped.",
                "To keep a column that is the same on every row of a group, take its 'min' or 'max'.",
                "'sum' of a group where every value is null is null, not zero. A zero would look "
                "like a measured amount.",
                "'sum' skips nulls in a group that has at least one value: 10 and null sum to 10.",
                "'first' and 'join_text' depend on the order the rows arrive in, so the same data "
                "in another order can give another answer. Sort first if that matters.",
                "Use 'first' for a column that is the same on every row of a group. If you are not "
                "sure it is, add a 'count_distinct' of the same column to check.",
                "'first' takes the first row's value, even when it is null and a later row has a value.",
                "'first_non_null' skips nulls and takes the first value there is. Only a group of "
                "nulls gives null.",
                "'count' counts the rows of a group and reads no column. 'count_non_null' counts the "
                "values of one column that are not null.",
                "Some other libraries call the non-null count 'count'. Check which one you mean "
                "when copying a step from them.",
                "'count_distinct' counts the different values and skips nulls, as SQL's "
                "COUNT(DISTINCT) does: a, a, null, b counts 2, and a group of nulls counts 0.",
                "To count a blank as a value too, count coalesce([v], '(blank)') instead.",
                "'join_text' joins a group's text into one cell, in the order the rows arrived, "
                "with 'separator' between (a comma and a space if none is given).",
                "'join_text' skips null values. A group where every value is null gives empty text, "
                "not null.",
                "That is the opposite of 'sum' on purpose: joining no text gives empty text, while "
                "adding no numbers gives a total nobody measured.",
                "'join_text' only joins text. Turn numbers or dates into text in an earlier step.",
                "A 'separator' on any function other than 'join_text' is refused, not ignored.",
                "A 'value' that reads no column, such as '1', is refused. It would be worked out "
                "once per group, so 'sum' of 1 would not count rows. Use 'count' for that.",
                "'sum' keeps the type of the column it adds up. If an Int32 or UInt32 total is too "
                "big for that type, the run stops and names the aggregation, instead of giving a "
                "wrong number. Cast the column to Int64 first if totals can be that large.",
            ),
            field_help={
                "by": "For example ['customer_id', 'month']. Each different combination of these "
                "values becomes one output row.",
                "aggregations": "For example {name: total, function: sum, value: '[amount]'}. Only "
                "these columns and the 'by' columns are kept.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse what the configuration alone shows is wrong: clashing names, a misplaced separator, a missing or extra value."""
        by_columns = set(config["by"])
        seen_names: set[str] = set()
        for index, aggregation in enumerate(config["aggregations"]):
            name = aggregation["name"]
            function = aggregation["function"]
            value = aggregation.get("value")

            require_not_blank(name, f"aggregations[{index}].name")
            if name in by_columns:
                raise ConfigError(
                    f"aggregation output name {name!r} collides with group-by column {name!r}; "
                    f"give the aggregation a different name"
                )
            if name in seen_names:
                raise ConfigError(f"two aggregations share the output name {name!r}; each must be unique")
            seen_names.add(name)

            if aggregation.get("separator") is not None and function != _JOINING_FUNCTION:
                raise ConfigError(
                    f"config field 'aggregations[{index}].separator' sits beside function "
                    f"{function!r}, which has nothing to separate; only 'join_text' reads a "
                    f"separator, so either make the function 'join_text' or drop the separator"
                )

            if function == _NO_VALUE_FUNCTION:
                if value is not None:
                    raise ConfigError(
                        f"aggregation {name!r} uses 'count', which counts rows and takes no 'value'; "
                        f"use 'count_non_null' to count the values of a column instead"
                    )
            elif value is None:
                raise ConfigError(f"aggregation {name!r} declares function {function!r} but no 'value' to aggregate")
            elif not references(value):
                # polars works out a fixed value once per group, so sum of 1 would give 1, not a row count.
                raise ConfigError(
                    f"config field 'aggregations[{index}].value' holds `{value}`, which reads no column; "
                    f"{function!r} of a fixed value is worked out once per group, not once per row. To "
                    f"count rows use 'count'. To add up a fixed amount per row, write it as a column in "
                    f"a formula step before this one and aggregate that column"
                )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Check the named columns against the frame's schema.

        Refuses a missing 'by' column, a 'value' naming a missing column, a 'sum' of something that is not a
        number, and a 'join_text' of something that is not text. An empty frame is refused the same way.
        """
        frame = inputs["rows"]
        require_columns(frame, _by_columns(config), "by", "rows")

        for index, aggregation in enumerate(config["aggregations"]):
            value = aggregation.get("value")
            if value is None:
                continue
            field = f"aggregations[{index}].value"
            if aggregation["function"] == "sum":
                column, dtype = compiled_dtype(frame, value, field)
                if not dtype.is_numeric():
                    raise ConfigError(
                        f"config field {field!r} sums column {column!r}, which holds {dtype}, not a "
                        f"number, in expression: `{value}`"
                    )
            elif aggregation["function"] == _JOINING_FUNCTION:
                column, dtype = compiled_dtype(frame, value, field)
                if dtype != pl.String:
                    raise ConfigError(
                        f"config field {field!r} asks 'join_text' to join column {column!r}, which "
                        f"holds {dtype}, not text, in expression: `{value}`; format it into text in "
                        f"the step before this one, where the formatting is written down"
                    )
            else:
                compile_for(frame, value, field)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        by_columns = _by_columns(config)
        for index, aggregation in enumerate(config["aggregations"]):
            if aggregation.get("value") is not None:
                refuse_division_by_zero(frame, aggregation["value"], f"aggregations[{index}].value")
        aggregates = [
            _compile_aggregation(frame, index, aggregation) for index, aggregation in enumerate(config["aggregations"])
        ]
        grouped = frame.group_by(by_columns, maintain_order=True).agg(aggregates)
        return Ports({"rows": _narrow_totals_back(frame, grouped, config)})


def _by_columns(config: Mapping[str, Any]) -> list[str]:
    """The group key, a column named twice counted once, in the order it was first written."""
    return list(dict.fromkeys(config["by"]))


def _narrow_totals_back(frame: pl.DataFrame, grouped: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """Give each sum of a 32-bit column its own type back, or stop if a total does not fit in it."""
    for index, aggregation in enumerate(config["aggregations"]):
        if aggregation["function"] != "sum":
            continue
        field = f"aggregations[{index}].value"
        _, dtype = compiled_dtype(frame, aggregation["value"], field)
        if dtype not in NARROW_SUM_LIMITS:
            continue
        name = aggregation["name"]
        low, high = NARROW_SUM_LIMITS[dtype]
        offending = totals_that_do_not_fit(grouped, name, dtype)
        if offending.height:
            raise DataError(
                f"aggregation {name!r} ({field}) adds up {dtype} values, and the total of some "
                f"groups does not fit in {dtype}, which holds {low:,} to {high:,}; it would have "
                f"wrapped round to a wrong number. Cast the column to Int64 with cast_columns before "
                f"this step. The totals shown are the true ones",
                offending=offending,
                total=grouped.height,
            )
        grouped = grouped.with_columns(pl.col(name).cast(dtype))
    return grouped


def _compile_aggregation(frame: pl.DataFrame, index: int, aggregation: Mapping[str, Any]) -> pl.Expr:
    """One aggregation's expression, named for the column it produces."""
    name = aggregation["name"]
    function = aggregation["function"]
    value = aggregation.get("value")

    if function == "count":
        return pl.len().cast(pl.Int64).alias(name)

    expression = compile_for(frame, value, f"aggregations[{index}].value")

    if function == "sum":
        _, dtype = compiled_dtype(frame, value, f"aggregations[{index}].value")
        total = sum_without_wrapping(expression, dtype)
        return pl.when(expression.is_not_null().any()).then(total).otherwise(None).alias(name)
    if function == "count_non_null":
        return expression.count().cast(pl.Int64).alias(name)
    if function == "count_distinct":
        return expression.drop_nulls().n_unique().cast(pl.Int64).alias(name)
    if function == "min":
        return expression.min().alias(name)
    if function == "first":
        return expression.first().alias(name)
    if function == "first_non_null":
        return expression.drop_nulls().first().alias(name)
    if function == _JOINING_FUNCTION:
        separator = aggregation.get("separator")
        separator = DEFAULT_SEPARATOR if separator is None else separator
        return expression.str.join(separator).alias(name)
    if function == "max":
        return expression.max().alias(name)
    # Reached only if AGGREGATE_FUNCTIONS grew without a branch here. Never fall through to a default.
    raise AssertionError(f"unreachable aggregate function {function!r}")  # pragma: no cover


def _total_and_count_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A"],
                    "amount": [10.0, 20.0, 5.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "total": [15.0, 20.0],
                    "lines": [2, 1],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a total and a row count for each code",
        configuration={
            "by": ["code"],
            "aggregations": [
                {"name": "total", "function": "sum", "value": "[amount]"},
                {"name": "lines", "function": "count"},
            ],
        },
        explanation="The rows are grouped by 'code', in the order A and B first appear. 'total' adds "
        "up 'amount' in each group and 'lines' counts the group's rows.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _carried_column_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", None],
                    "region": ["x", "y", "x", "z"],
                    "amount": [10.0, None, 5.0, 3.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", None],
                    "total": [15.0, None, 3.0],
                    "region_of_group": ["x", "y", "z"],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a blank code is its own group, and 'min' keeps a column that does not vary",
        configuration={
            "by": ["code"],
            "aggregations": [
                {"name": "total", "function": "sum", "value": "[amount]"},
                {"name": "region_of_group", "function": "min", "value": "[region]"},
            ],
        },
        explanation="The row with no 'code' becomes a group of its own; it is not dropped. Group B's "
        "total is null because its only amount is missing. 'region' is the same on every row of a "
        "group, so taking its 'min' keeps it on the output.",
        inputs=inputs,
        outputs=outputs,
    )


def _joined_text_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", "B", "C"],
                    "note": ["late", "checked", None, "held", None],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "C"],
                    "notes": ["late", "checked; held", ""],
                    "lines": [2, 2, 1],
                }
            )
        }
    )
    return WorkedExample(
        scenario="join each group's notes into one cell",
        configuration={
            "by": ["code"],
            "aggregations": [
                {"name": "notes", "function": "join_text", "value": "[note]", "separator": "; "},
                {"name": "lines", "function": "count"},
            ],
        },
        explanation="Each group's notes are joined in the order the rows arrived, with '; ' between. "
        "A's missing note is skipped, so 'notes' reads 'late' while 'lines' still counts two rows. "
        "C has a row but no note, so its notes are empty text, not null.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(GroupBy())
