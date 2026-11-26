"""CrossTab: turn a long frame wide: one row per 'by' key, one column per value of a key column.

The new columns are sorted as text, and the key column must already be text. A cell with no rows is
null, never zero. The aggregates are group_by's, imported, not copied.

Why: docs/design/tools.md#cross_tab
"""

from collections.abc import Mapping, Sequence
from typing import Any

from ..column_types import EXACT_DIGITS

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
)
from .group_by import (
    AGGREGATE_FUNCTIONS,
    DEFAULT_SEPARATOR,
    NARROW_SUM_LIMITS,
    SHARED_AGGREGATES,
    STATISTICS,
    TRUTH_AGGREGATES,
    describe_rows,
    exact_sum_parts,
    exact_totals,
    fraction_problem,
    shared_aggregate,
    sum_without_wrapping,
    totals_that_do_not_fit,
    unsuitable,
    widened,
)


class CrossTab:
    """One row per combination of 'by', one column per value of `names_from`, each cell an aggregate."""

    name = "cross_tab"
    description = "Turn a long table into a grid, with one column per value, such as one column per month."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="The columns that stay as rows, such as ['customer'].",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One column that stays as a row label. Naming it twice counts it once.",
                ),
            ),
            "names_from": ConfigField(
                type="string",
                description="The column whose values become the new column headings, such as 'month'. "
                "It must hold text.",
                required=True,
                non_empty=True,
            ),
            "values_from": ConfigField(
                type="string",
                description="The column that fills the cells, such as 'amount'. Needed even for "
                "'count', which ignores it.",
                required=True,
                non_empty=True,
            ),
            "function": ConfigField(
                type="string",
                description="How the rows in one cell become one value, such as 'sum' or 'count'.",
                required=True,
                enum=tuple(sorted(AGGREGATE_FUNCTIONS)),
            ),
            "fraction": ConfigField(
                type="number",
                description="For 'quantile' only, and required there: the fraction from 0 to 1, such as "
                "0.5 for the median or 0.9 for the 90th percentile.",
                required=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use cross_tab to turn a long table into a grid, like a pivot table: one row per "
                "customer and one column per month, say. `rows` has the 'by' columns first, in the "
                "order each combination first appears, then one column per value of 'names_from', "
                "sorted as text. Each cell holds the 'function' of 'values_from' over its rows. Every "
                "other column is dropped, so this is usually the last step before a report."
            ),
            examples=(_totals_by_period_example(), _counts_by_period_example(), _joined_notes_example()),
            notes=(
                "A cell with no rows under it is null, and there is no setting to make it zero. A "
                "zero would look like a measured amount.",
                "'sum' of a cell whose values are all null is null too, as in group_by.",
                "'names_from' must hold text, because its values become column names. Format a number "
                "or date as text in an earlier step. The new columns are sorted as text.",
                "A row whose 'names_from' value is null or blank stops the run, naming the column and "
                "counting the rows. Label or drop those rows in an earlier step.",
                "A null and a blank are refused together, as the same missing fact written two ways. "
                "A value of only spaces counts as blank.",
                "A 'names_from' value that equals a 'by' column's name is refused, because one of the "
                "two columns would be lost. A clash with any other input column is fine, since it is "
                "dropped.",
                "A column cannot be both in 'by' and the 'names_from' or 'values_from' column.",
                "The functions are group_by's, with group_by's rules: 'first' takes the first row's "
                "value even when null, 'first_non_null' the first value there is, and "
                "'count_distinct' skips nulls, as SQL's COUNT(DISTINCT) does.",
                "'join_text' fills a cell as it fills a group: values in arrival order, joined with a "
                "comma and a space, nulls skipped. A cell whose values are all null is empty text.",
                "So with 'join_text', a cell with rows but no values is empty text, while a cell with "
                "no rows is null. group_by never has the second case.",
                "A cell of an exact Decimal column keeps its places, with 38 digits in all. A 'sum' "
                "that needs more than 38 digits stops the run, naming the row and the cell.",
                "'sum' keeps the type of 'values_from'. If an Int32 or UInt32 total is too big for that "
                "type, the run stops instead of giving a wrong number. Cast to Int64 first if totals "
                "can be that large.",
                "'quantile' reads 'fraction', from 0 to 1. Leave 'fraction' out for every other "
                "function; it is refused there, not ignored.",
                "'mean', 'median', 'std', 'quantile', 'last', 'last_non_null', 'any_true' and "
                "'all_true' fill a cell as they fill a group in group_by, from the same code.",
            ),
            field_help={
                "by": "For example ['customer_id']. Each different combination becomes one row.",
                "names_from": "For example 'month', holding text such as '2026-01'.",
                "values_from": "For example 'amount'.",
                "function": "For example 'sum'. The list is the same as group_by's.",
                "fraction": "For example 0.9, beside function: quantile, for the 90th percentile.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a 'by' column that is also the names_from or values_from column, and a bad fraction.

        A fraction is bad when 'quantile' has none, another function has one, or it lies outside 0 to 1.
        """
        by_columns = set(_by_columns(config))
        for name in ("names_from", "values_from"):
            column = config[name]
            if column in by_columns:
                raise ConfigError(
                    f"config field {name!r} names column {column!r}, which is also in 'by'; the "
                    f"'by' columns come through as themselves, so a column cannot be a 'by' "
                    f"column and the {name!r} column at once"
                )
        problem = fraction_problem(config["function"], config.get("fraction"), "fraction")
        if problem is not None:
            raise ConfigError(problem)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Check the columns and types against the schema, so an empty frame is refused the same way.

        Waiting for a value of 'names_from' would let an empty frame through with nothing produced.
        """
        frame = inputs["rows"]
        require_columns(frame, _by_columns(config), "by", "rows")
        require_columns(frame, [config["names_from"]], "names_from", "rows")
        require_columns(frame, [config["values_from"]], "values_from", "rows")

        key = config["names_from"]
        key_dtype = frame.schema[key]
        if key_dtype != pl.String:
            raise ConfigError(
                f"config field 'names_from' names column {key!r}, which holds {key_dtype}, not "
                f"text; the values of that column become column names, so format it into text in the step "
                f"before this one"
            )

        measure = config["values_from"]
        measure_dtype = frame.schema[measure]
        if config["function"] == "sum" and not measure_dtype.is_numeric():
            raise ConfigError(
                f"config field 'values_from' names column {measure!r}, which holds {measure_dtype}, not "
                f"a number, and 'function' asks for a sum"
            )
        problem = unsuitable(config["function"], measure_dtype)
        if config["function"] in STATISTICS | TRUTH_AGGREGATES and problem is not None:
            raise ConfigError(
                f"config field 'values_from' names column {measure!r}, which holds {measure_dtype}, "
                f"{problem}, and 'function' asks for {config['function']!r}"
            )
        if config["function"] == "join_text" and measure_dtype != pl.String:
            raise ConfigError(
                f"config field 'values_from' names column {measure!r}, which holds {measure_dtype}, not "
                f"text, and 'function' asks for 'join_text'; format it into text in the step "
                f"before this one, where the formatting is written down"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        by_columns = _by_columns(config)
        key = config["names_from"]

        _refuse_unusable_keys(frame, key)
        names = sorted(frame.get_column(key).unique().to_list())
        _refuse_collisions(names, key, by_columns)

        measure_dtype = frame.schema[config["values_from"]]
        cells = [_cell(config, name, measure_dtype) for name in names]
        wide = frame.group_by(by_columns, maintain_order=True).agg(cells)
        if config["function"] == "sum" and measure_dtype in NARROW_SUM_LIMITS:
            wide = _narrow_cells_back(wide, names, config["values_from"], measure_dtype)
        if config["function"] == "sum" and isinstance(measure_dtype, pl.Decimal):
            wide = _exact_cells_back(wide, names, config["values_from"], measure_dtype.scale, by_columns)
        return Ports({"rows": wide})


def _by_columns(config: Mapping[str, Any]) -> list[str]:
    """The 'by' columns, a column named twice counted once, in the order it was first written."""
    return list(dict.fromkeys(config["by"]))


def _refuse_unusable_keys(frame: pl.DataFrame, key: str) -> None:
    """Refuse any row whose names_from value is null or blank, since it cannot become a column name.

    Null and blank are one rule: different extracts write the same missing period both ways.
    """
    unusable = pl.col(key).is_null() | (pl.col(key).str.strip_chars() == "")
    offending = frame.filter(unusable)
    if offending.height:
        raise DataError(
            f"column {key!r} has no usable value on some rows, and a value that is missing or blank "
            f"cannot become a column name; drop those rows or label them before this step",
            offending=offending,
            total=frame.height,
        )


def _refuse_collisions(names: Sequence[str], key: str, by_columns: Sequence[str]) -> None:
    """Refuse a value of the key column that would take a 'by' column's name."""
    taken = set(by_columns)
    for name in names:
        if name in taken:
            raise ConfigError(
                f"column {key!r} holds the value {name!r}, which would become an output column of "
                f"that name and collide with 'by' column {name!r}; rename the 'by' column, or "
                f"change the value, because one of the two would be lost"
            )


def _narrow_cells_back(wide: pl.DataFrame, names: Sequence[str], measure: str, dtype: pl.DataType) -> pl.DataFrame:
    """Give each summed cell column the measure's 32-bit type back, or stop if a total does not fit."""
    low, high = NARROW_SUM_LIMITS[dtype]
    for name in names:
        offending = totals_that_do_not_fit(wide, name, dtype)
        if offending.height:
            raise DataError(
                f"cell column {name!r} adds up column {measure!r}, which holds {dtype}, and some "
                f"totals do not fit in {dtype}, which holds {low:,} to {high:,}; they would have "
                f"wrapped round to a wrong number. Cast {measure!r} to Int64 with cast_columns before "
                f"this step. The totals shown are the true ones",
                offending=offending,
                total=wide.height,
            )
    return wide.with_columns(pl.col(name).cast(dtype) for name in names)


def _exact_cells_back(
    wide: pl.DataFrame, names: Sequence[str], measure: str, places: int, by_columns: list[str]
) -> pl.DataFrame:
    """Put each exact cell total back together from its parts, or stop if one needs more than 38 digits."""
    for name in names:
        totals, too_large = exact_totals(wide[name], places)
        if too_large.any():
            offending = wide.filter(too_large).select(by_columns)
            raise DataError(
                f"cell column {name!r} adds up exact column {measure!r}, and some totals need more than "
                f"{EXACT_DIGITS} digits, the most an exact column holds: at {places} places that is "
                f"{EXACT_DIGITS - places} before the point. The first is the row {describe_rows(offending, by_columns)}. "
                f"No total is given rather than a wrong one. Convert {measure!r} to Float64 with cast_columns "
                f"for figures this large, or round it to fewer places first",
                offending=offending,
                total=wide.height,
            )
        wide = wide.with_columns(totals)
    return wide


def _cell(config: Mapping[str, Any], name: str, measure_dtype: pl.DataType) -> pl.Expr:
    """One value column: the aggregate of `values_from` over this cell's rows, or null if it has none."""
    here = pl.col(config["names_from"]) == name
    measure = widened(pl.col(config["values_from"]), measure_dtype).filter(here)
    aggregate = config["function"]

    # Do not remove this guard: polars gives 0 for a cell with no rows, and for 'sum' of only
    # nulls. Both must be null. Why: docs/design/tools.md#cross_tab
    measured = measure.is_not_null().any() if aggregate == "sum" else here.any()
    answer = _aggregate(measure, aggregate, measure_dtype, config.get("fraction"))
    return pl.when(measured).then(answer).otherwise(None).alias(name)


def _aggregate(measure: pl.Expr, aggregate: str, measure_dtype: pl.DataType, fraction: float | None) -> pl.Expr:
    """The chosen aggregate over one cell's values; `_cell` makes an empty cell null."""
    if aggregate == "count":
        return measure.len().cast(pl.Int64)
    if aggregate == "sum":
        if isinstance(measure_dtype, pl.Decimal):
            return exact_sum_parts(measure)
        return sum_without_wrapping(measure, measure_dtype)
    if aggregate == "count_non_null":
        return measure.count().cast(pl.Int64)
    if aggregate == "count_distinct":
        return measure.drop_nulls().n_unique().cast(pl.Int64)
    if aggregate == "min":
        return measure.min()
    if aggregate == "first":
        return measure.first()
    if aggregate == "first_non_null":
        return measure.drop_nulls().first()
    if aggregate == "join_text":
        # group_by's default separator; this tool has no separator setting.
        return measure.str.join(DEFAULT_SEPARATOR)
    if aggregate == "max":
        return measure.max()
    if aggregate in SHARED_AGGREGATES:
        return shared_aggregate(measure, aggregate, fraction)
    # Reached only if AGGREGATE_FUNCTIONS grew without a branch here. Never fall through to a default.
    raise AssertionError(f"unreachable aggregate {aggregate!r}")  # pragma: no cover


def _totals_by_period_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "A", "B"],
                    "period": ["p2", "p1", "p1"],
                    "amount": [5.0, 10.0, 7.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": [10.0, 7.0],
                    "p2": [5.0, None],
                }
            )
        }
    )
    return WorkedExample(
        scenario="total amount per code and period",
        configuration={"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        explanation="The periods arrive as p2 then p1, but the columns are sorted: p1, then p2. Code B has no row "
        "in p2, so that cell is null, not zero.",
        inputs=inputs,
        outputs=outputs,
    )


def _counts_by_period_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "A", "A", "B"],
                    "period": ["p1", "p1", "p2", "p2"],
                    "amount": [5.0, 10.0, 7.0, 3.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": [2, None],
                    "p2": [1, 1],
                },
                schema={"code": pl.String, "p1": pl.Int64, "p2": pl.Int64},
            )
        }
    )
    return WorkedExample(
        scenario="count the rows in each cell",
        configuration={"by": ["code"], "names_from": "period", "values_from": "amount", "function": "count"},
        explanation="Two rows have code A in p1, so that cell is 2. Code B has no row in p1, so its cell is null, "
        "not 0. 'count' ignores 'values_from', but it is still written.",
        inputs=inputs,
        outputs=outputs,
    )


def _joined_notes_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "A", "B"],
                    "period": ["p1", "p2", "p1"],
                    "note": ["late", None, "checked"],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": ["late", "checked"],
                    "p2": ["", None],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a grid of notes, where two empty-looking cells mean different things",
        configuration={"by": ["code"], "names_from": "period", "values_from": "note", "function": "join_text"},
        explanation="Code A has a row in p2 with no note, so its cell is empty text: there was a row, with "
        "nothing to join. Code B has no row in p2 at all, so its cell is null.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(CrossTab())
