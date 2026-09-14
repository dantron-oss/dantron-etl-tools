"""CastColumns: one input, one output, and a column's type stated rather than inferred.

A type comes from wherever the data came from, and sometimes that is not the type the data *is*. A
CSV has no types at all. A database column defined years ago holds digits as text. Two branches of
one flow, about to be stacked, disagree about a column because one of them happened to see only
whole numbers. In each case the fix is the same: somebody who knows says what the column is.

**This is a step, not a setting on another tool.** It could have been a field on Union, and on
ReadTable, and on ReadExcel, and then the same words would be written down four times and drift. As
a step it serves all of them, and -- the part that matters more -- a type change becomes a node on
the canvas with a `why` beside it, rather than a line of configuration inside a step that appears to
be doing something else. A column that changes type is worth seeing.

**A value that will not convert stops the run, unless the flow says otherwise.** Turning it into a
null by default would produce a frame that looks right and is not: the column has the declared type,
the row is still there, and the value is simply gone. So ``on_invalid`` defaults to ``error``, and
the refusal names the column, the type asked for, and the offending rows.

``on_invalid: empty`` is the other honest answer, and it is the one a large messy source usually
wants: three unreadable values out of forty-six thousand should not stop a nightly build. It leaves
them null and **reports how many, out of how many, and what share** -- which raises the step to a
warning, so the run carries on and the fact travels with it, into the log and onto the canvas. The
count is taken *before* the values are replaced, because afterwards the evidence is gone: a null
that failed to convert and a null that was always there look identical.

**Casting a column to the type it already has is refused**, like every other declaration here that
does nothing. The usual cause is a configuration edited halfway, and saying so beats agreeing.

**Nothing moves.** Same rows, same order, same column positions -- only the type changes, which is
what makes this safe to put anywhere in a flow.
"""

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

import polars as pl

from ..column_types import DECLARED_DTYPE, DECLARED_TYPES
from ..tool import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    require_columns,
)

# 'empty' rather than 'null': `null` unquoted in a definition file IS the null value, so an author
# reaching for the obvious word would hand over None and the boundary would read it as unset. Spelled
# the same way AssembleDate spells the same choice.
_ON_INVALID_ENUM: Final[tuple[str, ...]] = ("error", "empty")


class CastColumns:
    """Read named columns of `rows` as the types the flow says they are."""

    name = "cast_columns"
    description = "Read named columns of `rows` as the types the flow states, leaving every row where it is."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="The columns to read as a stated type, one entry each.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One column's type: which column, and what it is.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The column to read as a stated type.",
                            required=True,
                            non_empty=True,
                        ),
                        "type": ConfigField(
                            type="string",
                            description="The type the column is, spelled the way the query engine spells it.",
                            required=True,
                            enum=DECLARED_TYPES,
                        ),
                    },
                ),
            ),
            "on_invalid": ConfigField(
                type="string",
                description="What a value that will not convert does: 'error' stops the run "
                "(default), 'empty' leaves it null and reports how many, out of how many.",
                default="error",
                enum=_ON_INVALID_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Each named column is read as the type stated for it. Every row stays, in its own "
                "order, and every column keeps its position -- only the type changes. A value that "
                "will not convert stops the run rather than becoming a null."
            ),
            examples=(_settles_a_text_number(), _two_columns_at_once()),
            notes=(
                "Put this in front of a Union whose two sides disagree about a column: the union "
                "refuses rather than choosing which side to rewrite, and this is where the flow "
                "says which one was right.",
                "A value that will not convert stops the run by default. 'on_invalid: empty' "
                "leaves it null instead and reports how many out of how many, raising the step to "
                "a warning -- the run carries on, and the size of the problem carries on with it.",
                "Casting a column to the type it already has is refused: it is a declaration that "
                "does nothing, and the usual cause is a configuration edited halfway.",
                "Not every pair converts. Text does not become a truth value here, because the "
                "spellings a source might use for one -- true, T, Y, 1 -- are a decision a flow "
                "should make in the open with CaseWhen rather than have a cast guess at.",
            ),
            field_help={
                "columns": "Order does not matter; each entry names a different column, and a "
                "column named twice is refused rather than resolved by whichever came last.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        named: set[str] = set()
        for index, entry in enumerate(config["columns"]):
            column = entry["column"]
            if column in named:
                raise ConfigError(
                    f"config field 'columns' states a type for {column!r} twice, the second at entry "
                    f"{index}; a column is given one type or none"
                )
            named.add(column)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        frame = inputs["rows"]
        require_columns(frame, [entry["column"] for entry in config["columns"]], "columns", "rows")

        inert = [
            entry["column"]
            for entry in config["columns"]
            if frame.schema[entry["column"]] == DECLARED_DTYPE[entry["type"]]
        ]
        if inert:
            raise ConfigError(
                f"config field 'columns' reads {inert[0]!r} as the type it already holds; a cast "
                f"that changes nothing is a configuration edited halfway"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        lenient = config["on_invalid"] == "empty"
        notes: list[str] = []

        for entry in config["columns"]:
            column, declared = entry["column"], entry["type"]
            if lenient:
                # Counted before the values are replaced, because afterwards the evidence is gone:
                # a null that failed to convert and a null that was always there look identical, and
                # the whole point of this mode is to say how many of the first there were.
                read = frame.select(pl.col(column).cast(DECLARED_DTYPE[declared], strict=False)).to_series()
                lost = int((frame[column].is_not_null() & read.is_null()).sum())
                if lost:
                    share = lost / frame.height * 100 if frame.height else 0.0
                    notes.append(
                        f"{lost:,} of {frame.height:,} values in {column!r} ({share:.2g}%) did not "
                        f"read as {declared} and are now empty"
                    )
                frame = frame.with_columns(read.alias(column))
                continue
            try:
                frame = frame.with_columns(pl.col(column).cast(DECLARED_DTYPE[declared]))
            except Exception as error:
                # Named as a data refusal rather than a configuration one: the configuration is a
                # legal thing to ask for, and what stopped it is the values. That is the difference
                # between "go and look at the flow" and "go and look at the rows".
                #
                # The examples are gathered with a non-strict cast, which answers null for the values
                # that would not convert -- and is itself wrapped, because a pair the engine refuses
                # outright rather than value by value would otherwise throw here, turning a refusal
                # into a crash inside the reporting of it.
                try:
                    offending = inputs["rows"].filter(
                        pl.col(column).is_not_null()
                        & pl.col(column).cast(DECLARED_DTYPE[declared], strict=False).is_null()
                    )
                except Exception:  # noqa: BLE001 - the report is best-effort; the refusal is not
                    offending = inputs["rows"].head(0)
                raise DataError(
                    f"column {column!r} does not read as {declared}: {error}",
                    offending=offending,
                    total=inputs["rows"].height,
                ) from error
        return Ports({"rows": frame}, notes=notes)


def _settles_a_text_number() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"code": ["A1", "B2"], "amount": ["10.5", "-3"]})})
    outputs = Ports({"rows": pl.DataFrame({"code": ["A1", "B2"], "amount": [10.5, -3.0]})})
    return WorkedExample(
        scenario="read a column of digits as the numbers they are",
        configuration={"columns": [{"column": "amount", "type": "Float64"}]},
        explanation="`amount` arrived as text, which is what a source with no types gives. It stays "
        "the second column and every row stays where it was; only the type changes.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _two_columns_at_once() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"count": ["2", "5"], "opened_on": ["2026-01-01", "2026-02-01"]})})
    outputs = Ports(
        {"rows": pl.DataFrame({"count": [2, 5], "opened_on": [date(2026, 1, 1), date(2026, 2, 1)]})}
    )
    return WorkedExample(
        scenario="settle two columns in one step",
        configuration={
            "columns": [
                {"column": "count", "type": "Int64"},
                {"column": "opened_on", "type": "Date"},
            ]
        },
        explanation="Each entry names one column and the type it is. The order entries are written "
        "in does not matter, since no column is named twice. A date is read in the order the engine "
        "reads it -- a source writing its digits in another order wants AssembleDate, where the "
        "order is stated.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(CastColumns())
