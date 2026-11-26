"""GroupBy: one row per group, with the summaries the configuration names.

The summaries (sum, count, max ...) are chosen from a fixed list, `AGGREGATE_FUNCTIONS`, and each
reads an expression. A null 'by' value is a group of its own, as in SQL's GROUP BY. A sum of nothing
but nulls is null. Groups keep the order they first appear in.

Why: docs/design/tools.md#group_by
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..column_types import EXACT_DIGITS, exact_dtype, exact_from_text
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
    refuse_rows_without_an_answer,
    require_columns,
    require_not_blank,
)

# The aggregates a flow may name. cross_tab offers the same list. Why: docs/design/tools.md#group_by
AGGREGATE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "sum",
        "count",
        "count_non_null",
        "count_distinct",
        "min",
        "max",
        "first",
        "first_non_null",
        "last",
        "last_non_null",
        "join_text",
        "mean",
        "median",
        "std",
        "quantile",
        "any_true",
        "all_true",
    }
)

# The aggregates that work out a figure from numbers. Shared with cross_tab.
STATISTICS: Final[frozenset[str]] = frozenset({"mean", "median", "std", "quantile"})

# The aggregates over true/false values. Shared with cross_tab.
TRUTH_AGGREGATES: Final[frozenset[str]] = frozenset({"any_true", "all_true"})

# The one aggregate that reads 'fraction'. Shared with cross_tab.
FRACTION_FUNCTION: Final[str] = "quantile"

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


# The statistics whose answer cannot be exact, and why, for the refusal of an exact column.
_INEXACT_STATISTICS: Final[dict[str, str]] = {
    "std": "A standard deviation takes a square root, so its answer cannot be exact. Convert the column "
    "to Float64 first with cast_columns, so the change is written in the pipeline",
    "quantile": "'quantile' interpolates between two values by a fraction, so its answer cannot be exact. "
    "Convert the column to Float64 first with cast_columns, so the change is written in the pipeline; "
    "for the exact middle value, use 'median'",
}


def unsuitable(function: str, dtype: pl.DataType) -> str | None:
    """Why ``function`` cannot summarise a column of ``dtype``, worded to follow "which holds <dtype>, "; None if it can.

    Shared with cross_tab, so the two refuse the same columns in the same words.
    """
    if function in STATISTICS:
        if isinstance(dtype, pl.Decimal) and function in _INEXACT_STATISTICS:
            return f"an exact decimal. {_INEXACT_STATISTICS[function]}"
        if not dtype.is_numeric():
            return "not a number"
    if function in TRUTH_AGGREGATES and dtype != pl.Boolean:
        return "not true/false. Compare it in an expression first, such as [amount] > 0"
    return None


def fraction_problem(function: str, fraction: Any, field: str) -> str | None:
    """What is wrong with ``fraction`` beside ``function``, as a refusal naming ``field``; None if nothing.

    Shared with cross_tab. The fraction is required by 'quantile', refused beside anything else, and
    must lie from 0 to 1.
    """
    if function != FRACTION_FUNCTION:
        if fraction is None:
            return None
        return (
            f"config field {field!r} sits beside function {function!r}, which reads no fraction; only "
            f"'quantile' reads one, so either make the function 'quantile' or drop the fraction"
        )
    if fraction is None:
        return f"'quantile' needs config field {field!r}, the fraction from 0 to 1, such as 0.9 for the 90th percentile"
    if not 0 <= fraction <= 1:
        return f"config field {field!r} is {fraction}, but a quantile's fraction lies from 0 to 1, such as 0.9"
    return None


def shared_aggregate(expression: pl.Expr, function: str, fraction: float | None) -> pl.Expr:
    """One of :data:`SHARED_AGGREGATES` over ``expression``, worked out once for both group_by and cross_tab.

    Each skips nulls, and a group with no value gives null. 'any_true' and 'all_true' are guarded for
    that, because the engine answers false and true for a group of nulls.
    """
    if function == "mean":
        return expression.mean()
    if function == "median":
        return expression.median()
    if function == "std":
        return expression.std(ddof=1)
    if function == FRACTION_FUNCTION:
        assert fraction is not None, "validate_config requires a fraction beside 'quantile'"
        return expression.quantile(fraction, interpolation="linear")
    if function == "last":
        return expression.last()
    if function == "last_non_null":
        return expression.drop_nulls().last()
    if function == "any_true":
        return pl.when(expression.is_not_null().any()).then(expression.any()).otherwise(None)
    if function == "all_true":
        return pl.when(expression.is_not_null().any()).then(expression.all()).otherwise(None)
    raise AssertionError(f"unreachable aggregate function {function!r}")  # pragma: no cover


# The aggregates both tools work out with :func:`shared_aggregate`, so the two cannot answer differently.
SHARED_AGGREGATES: Final[frozenset[str]] = STATISTICS | TRUTH_AGGREGATES | {"last", "last_non_null"}


def widened(expression: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """``expression`` with an exact column of ``dtype`` widened to 38 digits, keeping its places.

    Done before aggregating, because on some engine versions a sum keeps the source's narrower
    digits even when the total needs more. Any other type is left as it is. Shared with cross_tab.
    """
    if isinstance(dtype, pl.Decimal) and dtype.precision != exact_dtype(dtype.scale).precision:
        return expression.cast(exact_dtype(dtype.scale))
    return expression


# Why: docs/design/tools.md#exact-totals
#: Where an exact value's digits are split in two, so each part's total fits a 128-bit whole number.
_PART: Final[int] = 10**19

#: The places a mean or median of an exact column gives, at least: the division rule's 16.
DIVISION_PLACES: Final[int] = 16

#: The aggregates of an exact column the library works out itself, rather than the engine.
EXACT_AGGREGATES: Final[frozenset[str]] = frozenset({"sum", "mean", "median"})


def exact_places(function: str, places: int) -> int:
    """The places of ``function`` over an exact column of ``places``. Shared with cross_tab.

    A sum keeps the column's places. A mean or median divides, so it gives 16 places, or the
    column's own if it holds more.
    """
    return places if function == "sum" else max(DIVISION_PLACES, places)


def exact_parts(expression: pl.Expr, function: str) -> pl.Expr:
    """What ``exact_results`` needs to work out ``function`` of an exact column, gathered per group.

    Shared with cross_tab. Null for a group with no value. For 'sum' and 'mean', the total in two
    whole-number parts that cannot overflow: each value is read as a whole number of its smallest
    place (12.34 at 2 places is 1234), split at 10**19 into a high and a low part, and the parts are
    added up separately as 128-bit whole numbers, each under 10**19 per value. 'mean' adds the count.
    For 'median', the two middle values, which are the same value when the count is odd. The engine's
    own sum is not used: past 38 digits it gives a wrong total silently, and on polars 1.38 a crash
    when it is read.
    """
    if function == "median":
        ordered = expression.drop_nulls().sort()
        count = expression.count().cast(pl.Int64)
        parts = pl.struct(
            ordered.slice((count - 1) // 2, 1).first().alias("low"),
            ordered.slice(count // 2, 1).first().alias("high"),
        )
    else:
        high, low = _halves(expression)
        fields = [high.sum().alias("high"), low.sum().alias("low")]
        if function == "mean":
            fields.append(expression.count().cast(pl.Int128).alias("count"))
        parts = pl.struct(*fields)
    return pl.when(expression.is_not_null().any()).then(parts).otherwise(None)


def exact_running_parts(expression: pl.Expr, within: list[str]) -> pl.Expr:
    """The 'sum' parts of ``exact_parts``, as running totals in row order, one per row.

    Shared with running_total. The totals start again for each group of the ``within`` columns,
    or run over every row when it is empty. A null adds nothing, and the parts stay null until
    the group's first value, so ``exact_results`` gives null there.
    """
    high, low = (half.cum_sum().forward_fill() for half in _halves(expression))
    if within:
        high, low = high.over(within), low.over(within)
    return pl.when(high.is_not_null()).then(pl.struct(high.alias("high"), low.alias("low")))


def exact_results(parts: pl.Series, function: str, places: int) -> tuple[pl.Series, pl.Series]:
    """Work out ``function`` from the parts ``exact_parts`` gathered, as ``Decimal(38, exact_places(...))``.

    Shared with cross_tab. Returns the results, null where the parts were null or the answer does not
    fit, and a Boolean series, true exactly where the answer (or, for a mean, the total it divides)
    needs more than 38 digits. A mean or median rounds its last place half away from zero.
    """
    column = pl.col("parts")
    part, extra = pl.lit(_PART, dtype=pl.Int128), exact_places(function, places) - places
    if function == "median":
        # Half of low + high, without adding the two: each is halved, and the halves' remainders
        # (0 or 1 each) added back.
        two = pl.lit(2, dtype=pl.Int128)
        low, high = _whole(column.struct.field("low")), _whole(column.struct.field("high"))
        spare = low % two + high % two
        result, fits = _divided(low // two + high // two + spare // two, spare % two, two, extra)
    else:
        high, low = column.struct.field("high"), column.struct.field("low")
        # Carry the low total's excess into the high part, so the low part is back under 10**19. The
        # total is then high * 10**19 + low with 0 <= low < 10**19, and it fits while it lies
        # strictly between -10**38 and 10**38.
        high, low = high + low // part, low % part
        fits = (high < part) & ((high > -part) | ((high == -part) & (low > 0)))
        total = pl.when(fits).then(high * part + low)
        if function == "sum":
            result = total
        else:
            count = column.struct.field("count")
            result, quotient_fits = _divided(total // count, total % count, count, extra)
            fits = fits & quotient_fits
    worked = parts.to_frame("parts").select(
        _written_at(result, places + extra).alias("text"),
        (column.is_not_null() & ~fits.fill_null(False)).alias("too_large"),
    )
    values = exact_from_text(worked["text"], places + extra).values.alias(parts.name)
    return values, worked["too_large"].alias(parts.name)


def _halves(expression: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """An exact value as a whole number of its smallest place, split at 10**19 into a high and a low part."""
    part, whole = pl.lit(_PART, dtype=pl.Int128), _whole(expression)
    return whole // part, whole % part


def _whole(expression: pl.Expr) -> pl.Expr:
    """An exact value as a 128-bit whole number of its smallest place: 12.34 at 2 places is 1234."""
    return expression.cast(pl.String).str.replace(".", "", literal=True).cast(pl.Int128)


def _divided(quotient: pl.Expr, remainder: pl.Expr, divisor: pl.Expr, extra: int) -> tuple[pl.Expr, pl.Expr]:
    """``quotient + remainder / divisor`` with ``extra`` more places, the last rounded half away from zero.

    All three are 128-bit whole numbers with ``0 <= remainder < divisor``, as floor division gives.
    Returns the answer as a whole number of its new smallest place, and whether it fits 38 digits.
    """
    scale = pl.lit(10**extra, dtype=pl.Int128)
    room = (quotient.abs() < pl.lit(10 ** (EXACT_DIGITS - extra), dtype=pl.Int128)).fill_null(False)
    spread = remainder * scale
    # floor(value * 10**extra), and what is left over, as a fraction of the divisor.
    floor = pl.when(room).then(quotient * scale + spread // divisor)
    left = spread % divisor
    # Half away from zero: a positive value goes up at a half, a negative one only past a half.
    up = ((floor >= 0) & (left * 2 >= divisor)) | ((floor < 0) & (left * 2 > divisor))
    result = floor + up.cast(pl.Int128)
    return result, room & (result.abs() < pl.lit(10**EXACT_DIGITS, dtype=pl.Int128))


def _written_at(whole: pl.Expr, places: int) -> pl.Expr:
    """A whole number of the smallest place, written as plain digits with ``places`` after the point."""
    digits = whole.abs().cast(pl.String).str.zfill(places + 1)
    point = digits.str.len_chars() - places
    return pl.concat_str(
        pl.when(whole < 0).then(pl.lit("-")).otherwise(pl.lit("")),
        digits.str.slice(0, point),
        pl.lit("." if places else ""),
        digits.str.slice(point),
    )


def describe_rows(frame: pl.DataFrame, columns: list[str]) -> str:
    """The first row of ``frame`` as column='value' pairs, such as "code='A', month='2026-01'"."""
    first = frame.select(columns).row(0, named=True)
    return ", ".join(f"{column}={value!r}" for column, value in first.items())


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
                        "fraction": ConfigField(
                            type="number",
                            description="For 'quantile' only, and required there: the fraction from 0 "
                            "to 1, such as 0.5 for the median or 0.9 for the 90th percentile.",
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
            examples=(
                _total_and_count_example(),
                _carried_column_example(),
                _joined_text_example(),
                _statistics_example(),
            ),
            notes=(
                "Rows whose 'by' value is null form one group of their own, as in SQL's GROUP BY. "
                "They are never dropped.",
                "To keep a column that is the same on every row of a group, take its 'min' or 'max'.",
                "'sum' of a group where every value is null is null, not zero. A zero would look "
                "like a measured amount.",
                "'sum' skips nulls in a group that has at least one value: 10 and null sum to 10.",
                "'first', 'last' and 'join_text' depend on the order the rows arrive in, so the same "
                "data in another order can give another answer. Sort first if that matters.",
                "Use 'first' for a column that is the same on every row of a group. If you are not "
                "sure it is, add a 'count_distinct' of the same column to check.",
                "'first' takes the first row's value, even when it is null and a later row has a value.",
                "'first_non_null' skips nulls and takes the first value there is. Only a group of "
                "nulls gives null.",
                "'last' and 'last_non_null' are the same from the other end: 'last' takes the last "
                "row's value, even when it is null and an earlier row has a value.",
                "'mean', 'median', 'std' and 'quantile' skip nulls, as SQL's AVG does. A group with no "
                "value gives null, and 'std' of a single value is null.",
                "'std' is the sample standard deviation, as SQL's STDDEV and Excel's STDEV.S.",
                "'quantile' reads 'fraction', from 0 to 1, and interpolates between the two nearest "
                "values, as SQL's PERCENTILE_CONT and Excel's PERCENTILE.INC: 0.25 of 1, 2, 3, 4 is 1.75.",
                "'mean' and 'median' of an exact Decimal column are exact: the total divided by the "
                "count, or the middle value (the two middle values added and divided by 2). They give "
                "16 places after the point, or the column's own if it holds more, the last rounded half "
                "away from zero: the mean of 1, 0 and 0 is 0.3333333333333333.",
                "'std' and 'quantile' refuse an exact Decimal column, because their answer cannot be "
                "exact. Convert it to Float64 first, or use 'median' for the exact middle value.",
                "'any_true' and 'all_true' read a true/false column, skip nulls, and give null for a "
                "group with no value, as SQL's BOOL_OR and BOOL_AND.",
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
                "'sum', 'min', 'max', 'first' and 'last' of an exact Decimal column keep its places, "
                "with 38 digits in all.",
                "A 'sum' of an exact Decimal column whose total needs more than 38 digits stops the "
                "run, naming the group, instead of giving a wrong number. Convert to Float64 for "
                "figures that large.",
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
        """Refuse what the configuration alone shows is wrong.

        That is clashing names, a misplaced separator or fraction, a missing fraction or one outside 0
        to 1, and a missing or extra value.
        """
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

            problem = fraction_problem(function, aggregation.get("fraction"), f"aggregations[{index}].fraction")
            if problem is not None:
                raise ConfigError(problem)

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

        Refuses a missing 'by' column, a 'value' naming a missing column, a 'sum' or a statistic of something
        that is not a number, a statistic of an exact decimal, 'any_true' or 'all_true' of something that is
        not true/false, and a 'join_text' of something that is not text. An empty frame is refused the same way.
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
            elif aggregation["function"] in STATISTICS | TRUTH_AGGREGATES:
                column, dtype = compiled_dtype(frame, value, field)
                problem = unsuitable(aggregation["function"], dtype)
                if problem is not None:
                    raise ConfigError(
                        f"config field {field!r} asks {aggregation['function']!r} of column {column!r}, "
                        f"which holds {dtype}, {problem}, in expression: `{value}`"
                    )
            else:
                compile_for(frame, value, field)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        by_columns = _by_columns(config)
        for index, aggregation in enumerate(config["aggregations"]):
            if aggregation.get("value") is not None:
                refuse_rows_without_an_answer(frame, aggregation["value"], f"aggregations[{index}].value")
        aggregates = [
            _compile_aggregation(frame, index, aggregation) for index, aggregation in enumerate(config["aggregations"])
        ]
        grouped = frame.group_by(by_columns, maintain_order=True).agg(aggregates)
        grouped = _exact_totals_back(frame, grouped, config, by_columns)
        return Ports({"rows": _narrow_totals_back(frame, grouped, config)})


def _by_columns(config: Mapping[str, Any]) -> list[str]:
    """The group key, a column named twice counted once, in the order it was first written."""
    return list(dict.fromkeys(config["by"]))


def _exact_totals_back(
    frame: pl.DataFrame, grouped: pl.DataFrame, config: Mapping[str, Any], by_columns: list[str]
) -> pl.DataFrame:
    """Work out each exact sum, mean and median from its parts, or stop where one needs more than 38 digits."""
    for index, aggregation in enumerate(config["aggregations"]):
        function = aggregation["function"]
        if function not in EXACT_AGGREGATES:
            continue
        field = f"aggregations[{index}].value"
        column, dtype = compiled_dtype(frame, aggregation["value"], field)
        if not isinstance(dtype, pl.Decimal):
            continue
        name = aggregation["name"]
        results, too_large = exact_results(grouped[name], function, dtype.scale)
        if too_large.any():
            offending = grouped.filter(too_large).select(by_columns)
            raise DataError(
                f"aggregation {name!r} ({field}) asks {function!r} of exact column {column!r}, and for some "
                f"groups {too_large_because(function, dtype.scale)}. The first is the group "
                f"{describe_rows(offending, by_columns)}. No answer is given rather than a wrong one. Convert "
                f"the column to Float64 with cast_columns for figures this large",
                offending=offending,
                total=grouped.height,
            )
        grouped = grouped.with_columns(results)
    return grouped


def too_large_because(function: str, places: int) -> str:
    """Why an exact answer of ``function`` did not fit, worded to follow "for some groups". Shared with cross_tab."""
    room = f"{EXACT_DIGITS} digits, the most an exact column holds"
    if function == "sum":
        return f"the total needs more than {room}: at {places} places that is {EXACT_DIGITS - places} before the point"
    result = exact_places(function, places)
    answer = f"the {function} needs more than {room}: it keeps {result} places, which leaves {EXACT_DIGITS - result} before the point"
    return answer + (", or the total it divides needs more than 38 digits" if function == "mean" else "")


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

    _, dtype = compiled_dtype(frame, value, f"aggregations[{index}].value")
    expression = widened(compile_for(frame, value, f"aggregations[{index}].value"), dtype)

    if isinstance(dtype, pl.Decimal) and function in EXACT_AGGREGATES:
        return exact_parts(expression, function).alias(name)
    if function == "sum":
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
    if function in SHARED_AGGREGATES:
        return shared_aggregate(expression, function, aggregation.get("fraction")).alias(name)
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


def _statistics_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "product": ["A", "A", "A", "A", "B", "B"],
                    "days": [2.0, 4.0, None, 9.0, 3.0, None],
                    "late": [False, False, None, True, False, None],
                },
                schema={"product": pl.String, "days": pl.Float64, "late": pl.Boolean},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "product": ["A", "B"],
                    "mean_days": [5.0, 3.0],
                    "median_days": [4.0, 3.0],
                    "days_90th": [8.0, 3.0],
                    "any_late": [True, False],
                },
            )
        }
    )
    return WorkedExample(
        scenario="the mean, median and 90th percentile of each product's delivery days",
        configuration={
            "by": ["product"],
            "aggregations": [
                {"name": "mean_days", "function": "mean", "value": "[days]"},
                {"name": "median_days", "function": "median", "value": "[days]"},
                {"name": "days_90th", "function": "quantile", "value": "[days]", "fraction": 0.9},
                {"name": "any_late", "function": "any_true", "value": "[late]"},
            ],
        },
        explanation="The blank days are skipped, so A's mean is (2 + 4 + 9) / 3 = 5. The 90th percentile "
        "lies between A's two largest values, 4 and 9, so it is 4 + 0.8 * 5 = 8. 'any_late' is true "
        "where any row is late.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(GroupBy())
