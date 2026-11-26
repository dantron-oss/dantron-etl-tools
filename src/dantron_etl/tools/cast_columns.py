"""CastColumns: change named columns to the types the flow states.

A value that will not convert stops the run, or with `on_invalid: set_null` becomes null and is
counted and marked. Rows and column positions never move. A conversion to `Decimal` never rounds:
a value that does not fit its places is a value that will not convert.

Why: docs/design/tools.md#cast_columns
"""

from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any, Final

import polars as pl

from ..column_types import (
    DECLARED_TYPES,
    EXACT_TYPE,
    check_places,
    cut_fractions,
    declared_dtype,
    exact_misfit_refusal,
    exact_to_float,
    fraction_refusal,
    to_exact,
)
from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    as_calendar_date,
    check_zone_name,
    require_columns,
    require_stated_zone,
    row_status,
    unreadable_report,
)

# 'set_null', not 'null': YAML reads an unquoted null as no value at all.
_ON_INVALID_ENUM: Final[tuple[str, ...]] = ("error", "set_null")



class CastColumns:
    """Change named columns of `rows` to the types the flow states."""

    name = "cast_columns"
    description = "Change the type of columns, such as text to number or text to date."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="The columns to change, one entry each, with the type each should be.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One column and its new type.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The column to change, such as 'amount'.",
                            required=True,
                            non_empty=True,
                        ),
                        "type": ConfigField(
                            type="string",
                            description="The new type, such as 'Float64' for decimals, 'Decimal' for "
                            "exact decimals, 'Int64' for whole numbers, 'Date' or 'String'.",
                            required=True,
                            enum=DECLARED_TYPES,
                        ),
                        "places": ConfigField(
                            type="integer",
                            description="Only with type 'Decimal', and required there: the digits "
                            "after the point, from 0 to 38, such as 2.",
                        ),
                        "in_zone": ConfigField(
                            type="string",
                            description="Only when turning a timestamp with a time zone into a 'Date': "
                            "the zone whose calendar decides the day, such as 'UTC'.",
                        ),
                    },
                ),
            ),
            "on_invalid": ConfigField(
                type="string",
                description="What to do with a value that will not convert: 'error' stops the run; "
                "'set_null' makes it null and reports how many.",
                default="error",
                enum=_ON_INVALID_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use cast_columns to say what type a column really is: numbers that arrived as text "
                "from a CSV, dates stored as text, or a column two tables disagree about before a "
                "union. `rows` comes out with the same rows and columns in the same order; only the "
                "types change. A value that will not convert stops the run unless 'on_invalid' is "
                "'set_null'."
            ),
            examples=(_settles_a_text_number(), _two_columns_at_once(), _exact_prices()),
            notes=(
                "Put this in front of a union whose two sides disagree about a column's type. union "
                "refuses rather than choosing which side to change.",
                "A value that will not convert stops the run by default, naming the column, the type "
                "and the rows.",
                "With 'on_invalid: set_null', such a value becomes null, and the step reports how "
                "many out of how many as a warning; the run carries on.",
                "With 'set_null', each emptied row is also marked with the column and the value it "
                "held. A blank is not marked, because it did not fail.",
                "Changing a column to the type it already has is refused. It usually means a "
                "configuration edited halfway.",
                "A column named twice is refused.",
                "Text does not become true/false here, because sources spell it many ways (true, T, "
                "Y, 1). Use case_when to decide. That pair is refused when the step runs, not when "
                "the pipeline loads.",
                "A decimal becomes a whole number only when it already is one: 12.0 becomes 12, but "
                "1.99 stops the run, or is emptied and counted under 'set_null'. Round it first.",
                "'Decimal' is an exact decimal with 'places' digits after the point. A value converts "
                "only when it fits without rounding: '1.005' does not fit 2 places, so it stops the run, "
                "or is emptied and counted under 'set_null'. Round it first.",
                "Text becomes a Decimal only when written as plain digits, with an optional point and "
                "minus sign, such as -12.50. Exponents, spaces and commas do not convert.",
                "A Float64 becomes a Decimal read to 15 significant digits, as round reads it: 0.1 + 0.2 "
                "becomes 0.30. Not-a-number and infinity do not convert.",
                "A Decimal becomes a Float64 through its digits, so each value lands on the nearest "
                "double, however many digits it has.",
                "A timestamp with a time zone becomes a Date only with 'in_zone', because the day "
                "depends on the zone: 23:30 on 31 January in Toronto is 1 February in UTC. There is "
                "no default.",
                "A date cast to a number gives the days since 1 January 1970.",
                "A timestamp cast to Int64 gives a count in the column's own time unit: microseconds "
                "for a timestamp this library made, but a source may give nanoseconds or "
                "milliseconds. To count days or months between dates, use date_diff.",
            ),
            field_help={
                "columns": "For example [{column: amount, type: Float64}, {column: price, type: Decimal, "
                "places: 2}, {column: opened_on, type: Date}]. The order of the entries does not matter.",
                "on_invalid": "Use 'set_null' for a large, messy source where a few bad values should "
                "not stop a nightly run.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        named: set[str] = set()
        for index, entry in enumerate(config["columns"]):
            column = entry["column"]
            check_places(entry["type"], entry.get("places"), f"columns[{index}].places")
            zone = entry.get("in_zone")
            if zone is not None:
                check_zone_name(zone, f"columns[{index}].in_zone")
                if entry["type"] != "Date":
                    raise ConfigError(
                        f"config field 'columns[{index}].in_zone' is set for {column!r}, which is read as "
                        f"{entry['type']}; a zone only decides the day a timestamp is read as a Date, so remove it"
                    )
            if column in named:
                raise ConfigError(
                    f"config field 'columns' states a type for {column!r} twice, the second at entry "
                    f"{index}; a column is given one type or none"
                )
            named.add(column)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        frame = inputs["rows"]
        require_columns(frame, [entry["column"] for entry in config["columns"]], "columns", "rows")

        inert = [entry["column"] for entry in config["columns"] if frame.schema[entry["column"]] == _target(entry)]
        if inert:
            raise ConfigError(
                f"config field 'columns' reads {inert[0]!r} as the type it already holds; a cast "
                f"that changes nothing is a configuration edited halfway"
            )

        for index, entry in enumerate(config["columns"]):
            column, zone, dtype = entry["column"], entry.get("in_zone"), frame.schema[entry["column"]]
            field = f"columns[{index}].in_zone"
            if entry["type"] == EXACT_TYPE and not _converts_to_exact(dtype):
                raise ConfigError(
                    f"config field 'columns[{index}].type' reads {column!r} as Decimal, but it holds {dtype}, "
                    f"which is not a number; only text, whole numbers, decimals and floats become exact decimals"
                )
            if entry["type"] == "Date":
                require_stated_zone(dtype, zone, field.rpartition(".")[2], reading=f"column {column!r}")
            if zone is not None and not (isinstance(dtype, pl.Datetime) and dtype.time_zone is not None):
                raise ConfigError(
                    f"config field {field!r} is set for {column!r}, which holds {dtype} and carries no time "
                    f"zone; there is no zone to read it in, so remove it"
                )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        lenient = config["on_invalid"] == "set_null"
        notes: list[str] = []
        # One entry per column that emptied a value, in the order the columns were cast.
        marks: list[tuple[str, pl.Series]] = []

        for entry in config["columns"]:
            column, declared, target = entry["column"], entry["type"], _target(entry)
            if declared == "Date" and entry.get("in_zone") is not None:
                # A timestamp in a zone, read as the date it falls on in the zone the author stated.
                frame = frame.with_columns(as_calendar_date(pl.col(column), frame.schema[column], entry["in_zone"]))
                continue
            if declared == EXACT_TYPE:
                frame = _to_exact(inputs["rows"], frame, column, entry["places"], lenient, marks, notes)
                continue
            if declared == "Float64" and isinstance(frame.schema[column], pl.Decimal):
                # Through the digits, so every value lands on the nearest double. Nothing can fail here.
                frame = frame.with_columns(exact_to_float(frame[column]))
                continue
            if lenient:
                # Counted before the values are replaced; afterwards a failed value and a null look
                # alike. The count and the row marks come from the same mask.
                cut = cut_fractions(frame[column], target)
                read = frame.select(pl.col(column).cast(target, strict=False)).to_series()
                read = pl.select(pl.when(cut).then(None).otherwise(read)).to_series().alias(column)
                failed = frame[column].is_not_null() & read.is_null()
                if failed.any():
                    reasons, note = unreadable_report(frame, column, failed, declared)
                    marks.append((column, reasons))
                    notes.append(note)
                frame = frame.with_columns(read.alias(column))
                continue
            cut = cut_fractions(frame[column], target)
            if cut.any():
                raise DataError(
                    fraction_refusal(column, frame[column].filter(cut), declared),
                    offending=inputs["rows"].filter(cut),
                    total=inputs["rows"].height,
                )
            try:
                frame = frame.with_columns(pl.col(column).cast(target))
            except Exception as error:
                # A data refusal: the values stopped it, not the configuration. The sample rows are
                # best effort, so a type pair polars refuses outright does not crash the report.
                try:
                    offending = inputs["rows"].filter(
                        pl.col(column).is_not_null()
                        & pl.col(column).cast(target, strict=False).is_null()
                    )
                except Exception:  # noqa: BLE001 - the report is best-effort; the refusal is not
                    offending = inputs["rows"].head(0)
                raise DataError(
                    f"column {column!r} does not read as {declared}: {error}",
                    offending=offending,
                    total=inputs["rows"].height,
                ) from error
        status = {"rows": row_status(marks, frame.height)} if marks else None
        return Ports({"rows": frame}, notes=notes, status=status)


def _target(entry: Mapping[str, Any]) -> pl.DataType:
    """The dtype one entry asks for: its type, with its places for a Decimal."""
    return declared_dtype(entry["type"], entry.get("places"))


def _converts_to_exact(dtype: pl.DataType) -> bool:
    """Whether a column of ``dtype`` can become an exact decimal: text, a number, or nothing at all."""
    return dtype in (pl.String, pl.Null) or dtype.is_numeric()


def _to_exact(
    source: pl.DataFrame,
    frame: pl.DataFrame,
    column: str,
    places: int,
    lenient: bool,
    marks: list[tuple[str, pl.Series]],
    notes: list[str],
) -> pl.DataFrame:
    """``frame`` with ``column`` converted to an exact decimal, or a refusal naming what does not fit.

    Under 'set_null' a value that does not fit is emptied, and the note and the row marks are added
    to ``notes`` and ``marks``.
    """
    converted = to_exact(frame[column], places)
    if converted.misfits.any():
        if not lenient:
            raise DataError(
                exact_misfit_refusal(column, frame[column].filter(converted.misfits), places),
                offending=source.filter(converted.misfits),
                total=source.height,
            )
        reasons, note = unreadable_report(frame, column, converted.misfits, f"Decimal with {places} places")
        marks.append((column, reasons))
        notes.append(note)
    return frame.with_columns(converted.values)


def _settles_a_text_number() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"code": ["A1", "B2"], "amount": ["10.5", "-3"]})})
    outputs = Ports({"rows": pl.DataFrame({"code": ["A1", "B2"], "amount": [10.5, -3.0]})})
    return WorkedExample(
        scenario="turn a column of digits held as text into numbers",
        configuration={"columns": [{"column": "amount", "type": "Float64"}]},
        explanation="'amount' arrived as text, as it does from a CSV. It becomes Float64 and stays the second "
        "column; nothing else changes.",
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
        scenario="change two columns in one step",
        configuration={
            "columns": [
                {"column": "count", "type": "Int64"},
                {"column": "opened_on", "type": "Date"},
            ]
        },
        explanation="Each entry names a column and its type. A date is read as year-month-day; for digits in "
        "another order, use assemble_date.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _exact_prices() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"code": ["A1", "B2", "C3"], "price": ["812.40", "19.9", None]})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {"code": ["A1", "B2", "C3"], "price": [Decimal("812.40"), Decimal("19.90"), None]},
                schema={"code": pl.String, "price": pl.Decimal(38, 2)},
            )
        }
    )
    return WorkedExample(
        scenario="turn prices held as text into exact decimals with two places",
        configuration={"columns": [{"column": "price", "type": "Decimal", "places": 2}]},
        explanation="Each price is held exactly, with two places: 19.9 becomes 19.90. A price written with "
        "more places, such as 1.005, would stop the run rather than be rounded.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(CastColumns())
