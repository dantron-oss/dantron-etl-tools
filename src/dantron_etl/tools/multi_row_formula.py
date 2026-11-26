"""MultiRowFormula: set one column with an expression that can read other rows, like Alteryx's Multi-Row Formula.

[Row-1:amount] is 'amount' one row up and [Row+2:amount] two rows down, in the order the rows
arrived, within each group when 'group_by' names one. A row that is not there reads as null.
[Row-N:<output>] reads what this step already wrote for that earlier row, so such a formula is
worked out a few row positions at a time; any other formula is worked out for all rows at once.

Why: docs/design/tools.md#multi_row_formula
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple

import polars as pl

from ..authoring import (
    REGISTRY,
    ROW_REACH,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    RowReference,
    ToolHelp,
    WorkedExample,
    compile_for,
    compiled_dtype,
    references,
    refuse_rows_without_an_answer,
    require_columns,
    require_not_blank,
    row_references_for,
    rows_without_an_answer,
)
from ..column_types import (
    DECLARED_TYPES,
    EXACT_TYPE,
    check_places,
    cut_fractions,
    declared_dtype,
    exact_to_float,
    to_exact,
)


class _Plan(NamedTuple):
    """What a step reads and writes, worked out from the schema alone."""

    #: Row references to other columns: plain reads of the input, shifted.
    reads: tuple[RowReference, ...]
    #: Row references to earlier rows of the output column: the values this step wrote.
    writes_back: tuple[RowReference, ...]
    #: The expression, compiled against the input plus one column per row reference.
    expression: pl.Expr
    #: The type the output is converted to, or None to keep the expression's own type.
    target: pl.DataType | None


class MultiRowFormula:
    """Set one column of `rows` with an expression that can read the rows above and below."""

    name = "multi_row_formula"
    description = "Work out a column from other rows, such as the change from the row above or a running balance."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "output": ConfigField(
                type="string",
                description="The column to set, such as 'change'. A column already there is updated in place; "
                "a new one is added at the end.",
                required=True,
                non_empty=True,
            ),
            "expression": ConfigField(
                type="expression",
                description="The expression, such as '[amount] - [Row-1:amount]'. [Row-N:column] reads N rows "
                "up and [Row+N:column] N rows down; [column] is the current row.",
                required=True,
            ),
            "group_by": ConfigField(
                type="array",
                description="Columns whose values make a group, such as ['account']. A row reference never "
                "reads into another group. Left out, all rows are one group.",
                non_empty=True,
                items=ConfigField(type="string", description="One column of `rows`.", non_empty=True),
            ),
            "type": ConfigField(
                type="string",
                description="The type of the output, such as 'Float64'. Needed when the expression reads "
                "earlier rows of a new output column; otherwise the expression's own type is kept.",
                enum=DECLARED_TYPES,
            ),
            "places": ConfigField(
                type="integer",
                description="Only with type 'Decimal', and required there: the digits after the point, from "
                "0 to 38, such as 2.",
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use multi_row_formula when a row's value depends on the rows around it: the change from "
                "the row above, the next row's date, filling blanks down, or a running balance. It works "
                "like formula, with one more kind of reference: [Row-1:amount] is 'amount' one row up and "
                "[Row+1:amount] one row down. Rows are read in the order they arrived; put a sort in front "
                "for a meaningful order. With 'group_by', each group is read on its own. `rows` comes out "
                "with every row in its order and the output column set."
            ),
            examples=(
                _change_example(),
                _next_date_example(),
                _fill_down_example(),
                _running_balance_example(),
            ),
            notes=(
                f"[Row-N:column] reads N rows up and [Row+N:column] N rows down, N from 1 to {ROW_REACH}. "
                "Spaces inside the brackets are allowed: [Row - 1 : unit price]. Row-0 and a reach past "
                f"{ROW_REACH} are refused.",
                "A row that is not there reads as null: above the first row, below the last, or in "
                "another group. Write coalesce([Row-1:amount], 0) to read zero there.",
                "With 'group_by', a reference never crosses into another group, and the rows of a group "
                "need not be next to each other: each group is read in arrival order. Rows with no value "
                "in a group column form a group of their own, as in SQL's PARTITION BY.",
                "[Row-N:<output>] reads the value this step already wrote for that earlier row, so "
                "coalesce([amount], [Row-1:amount]) written to 'amount' fills every run of blanks with "
                "the last value above it. [Row+N:<output>] is refused: that value is not written yet.",
                "A plain [<output>] on a column already there reads the current row's value as it arrived.",
                "A formula that reads earlier rows of its own output is worked out a few rows at a time, "
                "so it is slower than any other formula, which is worked out for all rows at once. A "
                "fill-down stays quick. A running balance over one very long group (tens of thousands of "
                "rows) takes seconds to tens of seconds; for a running balance, running_total is the "
                "fast choice.",
                "'type' is needed when the expression reads earlier rows of a new output column, because "
                "its type cannot be worked out otherwise. When it reads earlier rows of a column already "
                "there, the output keeps that column's type.",
                "With 'type', or a column whose type is kept, the result converts as cast_columns converts "
                "it: nothing is rounded, and a value that does not fit stops the step.",
                "Everything formula refuses is refused here: a column that is not there, a type mismatch, "
                "an exact decimal meeting a float. A division by zero on a row stops the step, as does an "
                "exact answer past 38 digits. With a self-reference, only the settled values decide, and "
                "the refusal names every such row.",
                "The output column is updated in place when it is there and added at the end when not. "
                "No other column and no row moves. No rows in gives no rows out, with the output column.",
                "It mirrors SQL's LAG and LEAD over a window partitioned by 'group_by' in arrival order.",
            ),
            field_help={
                "output": "For example 'balance'. Name a column already there to update it in place.",
                "expression": "For example '[amount] - [Row-1:amount]', or "
                "'coalesce([Row-1:balance], 0) + [amount]' for a running balance written to 'balance'.",
                "group_by": "For example ['account'], so each account's first row reads null above it.",
                "type": "For example 'Float64', or 'Decimal' with places: 2 for exact money.",
                "places": "For example 2. Only with type 'Decimal'.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a blank output, bad places, a group column named twice, a bad reach, and a later row of the output."""
        output = config["output"]
        require_not_blank(output, "output")
        declared, places = config.get("type"), config.get("places")
        if declared is not None:
            check_places(declared, places, "places")
        elif places is not None:
            raise ConfigError(
                f"config field 'places' is {places!r} with no type; places goes with type: Decimal, so add "
                f"that or remove places"
            )
        group_by = config.get("group_by") or []
        for index, column in enumerate(group_by):
            if column in group_by[:index]:
                raise ConfigError(f"config field 'group_by' names column {column!r} twice; name each column once")
        for reference in row_references_for(config["expression"], "expression"):
            if reference.column == output and reference.offset > 0:
                raise ConfigError(
                    f"config field 'expression' reads [{reference.written}], a later row of {output!r}, the "
                    f"column this step writes; that value is not written yet when this row is worked out. "
                    f"Read an earlier row with [Row-1:{output}], or write to a new column"
                )

    def column_references(self, config: Mapping[str, Any]) -> Mapping[str, frozenset[str]]:
        """The input columns the expression reads: the plain ones, and the column of each row reference.

        An earlier row of the output reads what this step writes, so it asks nothing of the input.
        """
        text, output = config["expression"], config["output"]
        found = row_references_for(text, "expression")
        written = {reference.written for reference in found}
        plain = {name for name in references(text) if name not in written}
        return {"expression": frozenset(plain | {ref.column for ref in found if ref.column != output})}

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        _plan(inputs["rows"], config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        plan = _plan(frame, config)
        group_by = list(config.get("group_by") or [])
        work = frame.with_columns([_shifted(reference, group_by) for reference in plan.reads])
        if plan.writes_back:
            values = _worked_out_in_turn(work, plan, config, group_by, frame.columns)
        else:
            refuse_rows_without_an_answer(work, config["expression"], "expression", shown=frame.columns)
            values = work.select(plan.expression).to_series()
            if plan.target is not None:
                values = _convert(values, plan.target, frame, config)
        return Ports({"rows": frame.with_columns(values.alias(config["output"]))})


def _plan(frame: pl.DataFrame, config: Mapping[str, Any]) -> _Plan:
    """Check the settings against the frame's schema and work out what the step reads and writes."""
    text, output = config["expression"], config["output"]
    require_columns(frame, config.get("group_by") or [], "group_by", "rows")
    found = row_references_for(text, "expression")
    reads = tuple(reference for reference in found if reference.column != output)
    writes_back = tuple(reference for reference in found if reference.column == output)
    for reference in reads:
        if reference.column not in frame.columns:
            raise ConfigError(
                f"config field 'expression' reads [{reference.written}], but column {reference.column!r} is not "
                f"on input port 'rows'; the columns on 'rows' are: {', '.join(frame.columns)}"
            )

    target = None if config.get("type") is None else declared_dtype(config["type"], config.get("places"))
    if writes_back and target is None:
        if output not in frame.columns:
            raise ConfigError(
                f"config field 'type' is needed: the expression reads [{writes_back[0].written}], an earlier row "
                f"of {output!r}, which is a new column, so its type cannot be worked out. Say the type, such "
                f"as type: Float64"
            )
        target = frame.schema[output]

    schema = dict(frame.schema)
    schema.update({reference.written: frame.schema[reference.column] for reference in reads})
    schema.update({reference.written: target for reference in writes_back})
    shape = pl.DataFrame(schema=schema)
    expression = compile_for(shape, text, "expression")
    _, produced = compiled_dtype(shape, text, "expression")
    if target is not None:
        _check_convertible(produced, target, config)
    return _Plan(reads, writes_back, expression, target)


def _check_convertible(produced: pl.DataType, target: pl.DataType, config: Mapping[str, Any]) -> None:
    """Refuse a result type that cannot become the output's type, as cast_columns refuses it."""
    if produced == target:
        return
    wanted = _type_name(target, config)
    if isinstance(target, pl.Decimal) and not (produced in (pl.String, pl.Null) or produced.is_numeric()):
        raise ConfigError(
            f"config field 'expression' gives {produced}, which is not a number, and the output is {wanted}; only "
            f"text, whole numbers, decimals and floats become exact decimals"
        )
    if target == pl.Date and isinstance(produced, pl.Datetime) and produced.time_zone is not None:
        raise ConfigError(
            f"config field 'expression' gives a timestamp in time zone {produced.time_zone!r}, and the output is "
            f"a Date; which day it falls on depends on the zone, so read it as a date in the expression first"
        )


def _type_name(target: pl.DataType, config: Mapping[str, Any]) -> str:
    """The output's type as the author would write it."""
    if config.get("type") is not None:
        return str(config["type"]) if config["type"] != EXACT_TYPE else f"Decimal with {config['places']} places"
    return str(target)


def _shifted(reference: RowReference, group_by: Sequence[str]) -> pl.Expr:
    """The referenced column moved so each row holds the value N rows away, within its group."""
    moved = pl.col(reference.column).shift(-reference.offset)
    return (moved.over(group_by) if group_by else moved).alias(reference.written)


def _convert(values: pl.Series, target: pl.DataType, rows: pl.DataFrame, config: Mapping[str, Any]) -> pl.Series:
    """`values` as the output's type, by cast_columns' rules: nothing rounded, a misfit stops the step.

    `rows` are the input's rows the values belong to, in arrival order, for the refusal.
    """
    total = rows.height
    if values.dtype == target:
        return values
    wanted = _type_name(target, config)
    output = config["output"]
    shown_rows = rows.with_columns(values.alias(output))
    if isinstance(target, pl.Decimal):
        places = target.scale or 0
        converted = to_exact(values, places)
        if converted.misfits.any():
            raise DataError(
                f"config field 'expression' gives values for {output!r} that do not fit {wanted} without "
                f"rounding, such as {_shown(values.filter(converted.misfits))}; round them in the expression, "
                f"such as round(..., {places})",
                offending=shown_rows.filter(converted.misfits),
                total=total,
            )
        return converted.values.alias(values.name)
    if target == pl.Float64 and isinstance(values.dtype, pl.Decimal):
        return exact_to_float(values)
    cut = cut_fractions(values, target)
    if cut.any():
        raise DataError(
            f"config field 'expression' gives values for {output!r} with a fraction, such as "
            f"{_shown(values.filter(cut))}, so they do not read as {wanted} without cutting the fraction off; "
            f"round them in the expression, such as round(..., 0), or choose a type that holds them",
            offending=shown_rows.filter(cut),
            total=total,
        )
    try:
        return values.cast(target)
    except pl.exceptions.PolarsError as error:
        read, _ = _converted_or_null(values, target)
        unread = values.is_not_null() & read.is_null()
        raise DataError(
            f"config field 'expression' gives values for {output!r} that do not read as {wanted}: {error}",
            offending=shown_rows.filter(unread),
            total=total,
        ) from error


def _shown(values: pl.Series) -> str:
    """Up to three different values, for a refusal."""
    return ", ".join(str(value) for value in values.unique(maintain_order=True).head(3).to_list())


def _converted_or_null(values: pl.Series, target: pl.DataType) -> tuple[pl.Series, pl.Series]:
    """`values` as the output's type, with null where a value does not convert, and where that was.

    The same rules as :func:`_convert`, which is asked for the refusal once a row is known to fail.
    """
    none = pl.repeat(False, values.len(), dtype=pl.Boolean, eager=True)
    if values.dtype == target:
        return values, none
    if isinstance(target, pl.Decimal):
        converted = to_exact(values, target.scale or 0)
        return converted.values.alias(values.name), converted.misfits
    if target == pl.Float64 and isinstance(values.dtype, pl.Decimal):
        return exact_to_float(values), none
    cut = cut_fractions(values, target)
    try:
        read = values.cast(target, strict=False)
    except pl.exceptions.PolarsError:
        # A pair the engine refuses outright, such as text to true/false: no value converts.
        read = pl.repeat(None, values.len(), dtype=target, eager=True)
    misfits = cut | (values.is_not_null() & read.is_null())
    return pl.select(pl.when(misfits).then(None).otherwise(read)).to_series().alias(values.name), misfits


# How many places in a group are worked out together at first, and the fewest after that.
_FIRST_WINDOW: Final[int] = 1024
_SMALLEST_WINDOW: Final[int] = 256


def _worked_out_in_turn(
    work: pl.DataFrame, plan: _Plan, config: Mapping[str, Any], group_by: Sequence[str], columns: Sequence[str]
) -> pl.Series:
    """Work out a formula that reads earlier rows of its own output, giving what a row-by-row walk gives.

    Rows are put in order of their place in their group (every group's first row, then every
    group's second, and so on) and taken a window of places at a time. Within a window, the formula
    is worked out for all its rows at once, reading what the last pass wrote, and repeated until a
    pass changes nothing. That is the row-by-row answer: a row reads only rows above it, so when a
    pass changes nothing, every row equals the formula worked out on the final rows above it. Each
    pass makes at least one more place final, so the repeating always ends.

    A row with no answer, or a value that does not convert, stops the step only once every row is
    settled, and the refusal names every such row. `work` is the input with the reads of other
    columns already added; `columns` are the input's own, which a refusal shows.

    Why: docs/design/tools.md#multi_row_formula
    """
    text, target, total = config["expression"], plan.target, work.height
    assert target is not None

    # Where each row sits in its group, and an order that puts every group's first rows first.
    index = pl.int_range(pl.len(), dtype=pl.Int64)
    key_names = [f"key{number}" for number in range(len(group_by))]
    keys = work.select(index.alias("row"), *[pl.col(column).alias(name) for column, name in zip(group_by, key_names)])
    keys = keys.with_columns((index.over(key_names) if key_names else index).alias("position"))
    order = keys.sort(["position", "row"]).get_column("row")
    keys = keys.with_columns(order.arg_sort().alias("placed"))
    positions = keys.get_column("position").gather(order)
    count = int(positions.max() or 0) + 1 if total else 0
    starts = positions.search_sorted(pl.Series(range(count + 1), dtype=pl.Int64), side="left").to_list()

    # For each self-reference, where in that order the row N up in the same group is (null for none).
    reach: list[tuple[str, pl.Series]] = []
    for reference in plan.writes_back:
        earlier = pl.col("placed").shift(-reference.offset)
        found = keys.select(earlier.over(key_names) if key_names else earlier).to_series().gather(order)
        reach.append((reference.written, found))
    ordered = work.drop([name for name, _ in reach], strict=False)[order]

    shape = pl.DataFrame(schema={**ordered.schema, **{name: target for name, _ in reach}})
    stops = rows_without_an_answer(shape, text, "expression")
    selected = [plan.expression.alias("value"), (stops if stops is not None else pl.lit(False)).alias("stop")]

    done = pl.Series("value", [], dtype=target)
    failed = False
    first, window = 0, _FIRST_WINDOW
    while first < count:
        last = min(first + window, count)
        start, size = starts[first], starts[last] - starts[first]
        rows = ordered.slice(start, size)
        guess = pl.repeat(None, size, dtype=target, eager=True).alias("value")
        turns = 0
        while True:
            known = pl.concat([done, guess], rechunk=False)
            got = rows.hstack([known.gather(found.slice(start, size)).alias(name) for name, found in reach])
            got = got.select(selected)
            values, misfits = _converted_or_null(got.get_column("value"), target)
            turns += 1
            settled = values.eq_missing(guess).all()
            guess = values
            if settled:
                break
        # Only the settled pass decides: a row that failed only while the rows above it were
        # still unsettled has an answer.
        failed = failed or bool((got.get_column("stop") | misfits).any())
        done = pl.concat([done, guess])
        first = last
        # Widen the window while it settles in few passes (a fill-down), narrow it while it needs
        # many (a running balance needs one pass per place).
        window = window * 2 if turns * 4 <= window else max(window // 2, _SMALLEST_WINDOW)

    placed = keys.get_column("placed")
    if failed:
        read = [done.gather(found.gather(placed)).alias(name) for name, found in reach]
        _refuse(work.drop([name for name, _ in reach], strict=False).hstack(read), plan, config, columns)
    return done.gather(placed)


def _refuse(rows: pl.DataFrame, plan: _Plan, config: Mapping[str, Any], columns: Sequence[str]) -> None:
    """Stop the step on every row whose own calculation has no answer, or whose value does not convert.

    `rows` are the rows in arrival order, each with the settled values its self-references read, so
    the rows named are those formula would name. The refusal shows the input's own `columns`.
    """
    refuse_rows_without_an_answer(rows, config["expression"], "expression", shown=columns)
    assert plan.target is not None
    _convert(rows.select(plan.expression).to_series(), plan.target, rows.select(columns), config)
    raise AssertionError("a row was found failing, and neither check stopped on it")  # pragma: no cover


def _change_example() -> WorkedExample:
    rows = {"account": ["A", "B", "A", "B"], "amount": [100.0, 40.0, 130.0, 25.0]}
    return WorkedExample(
        scenario="work out the change from the previous row of the same account",
        configuration={"output": "change", "expression": "[amount] - [Row-1:amount]", "group_by": ["account"]},
        explanation="Each row takes away the amount of the row above it in the same account. The first row of "
        "each account has no row above, so its change is null.",
        inputs=Ports({"rows": pl.DataFrame(rows)}),
        outputs=Ports({"rows": pl.DataFrame({**rows, "change": [None, None, 30.0, -15.0]})}),
        check_row_order=True,
    )


def _next_date_example() -> WorkedExample:
    from datetime import date

    rows = {"shipment": [1, 2, 3], "date": [date(2026, 1, 5), date(2026, 1, 9), date(2026, 2, 2)]}
    return WorkedExample(
        scenario="put the next row's date beside each row",
        configuration={"output": "next_date", "expression": "[Row+1:date]"},
        explanation="Each row reads the date one row down. The last row has no row below, so its next date is null.",
        inputs=Ports({"rows": pl.DataFrame(rows)}),
        outputs=Ports({"rows": pl.DataFrame({**rows, "next_date": [date(2026, 1, 9), date(2026, 2, 2), None]})}),
        check_row_order=True,
    )


def _fill_down_example() -> WorkedExample:
    account = ["A", "A", "B", "A", "B", "B"]
    return WorkedExample(
        scenario="fill blanks down with the last value above, within each account",
        configuration={"output": "rate", "expression": "coalesce([rate], [Row-1:rate])", "group_by": ["account"]},
        explanation="A blank rate takes the rate this step already wrote for the row above in the same account, so "
        "a run of blanks of any length is filled. Account B's rows never read account A's.",
        inputs=Ports({"rows": pl.DataFrame({"account": account, "rate": [1.5, None, 2.0, None, None, 2.5]})}),
        outputs=Ports({"rows": pl.DataFrame({"account": account, "rate": [1.5, 1.5, 2.0, 1.5, 2.0, 2.5]})}),
        check_row_order=True,
    )


def _running_balance_example() -> WorkedExample:
    rows = {"account": ["A", "B", "A", "A"], "amount": [100.0, 50.0, -30.0, 10.0]}
    return WorkedExample(
        scenario="keep a running balance for each account",
        configuration={
            "output": "balance",
            "type": "Float64",
            "expression": "coalesce([Row-1:balance], 0) + [amount]",
            "group_by": ["account"],
        },
        explanation="Each row adds its amount to the balance this step wrote for the row above in the same account; "
        "the first row of an account starts from 0. 'type' is needed because 'balance' is a new column that "
        "reads its own earlier rows.",
        inputs=Ports({"rows": pl.DataFrame(rows)}),
        outputs=Ports({"rows": pl.DataFrame({**rows, "balance": [100.0, 50.0, 70.0, 80.0]})}),
        check_row_order=True,
    )


REGISTRY.register(MultiRowFormula())
