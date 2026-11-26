"""Unpivot: turn a wide frame long; every column outside `by` becomes a name and a value.

The reverse of cross_tab. Turned columns must share a type unless `values_type` states one; exact
decimals of different places stack at the larger places. Rows come out column by column. A null cell
is a row with a null value unless `exclude_nulls`.

Why: docs/design/tools.md#unpivot
"""

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

import polars as pl

from ..column_types import (
    DECLARED_TYPES,
    EXACT_DIGITS,
    EXACT_TYPE,
    check_places,
    cut_fractions,
    declared_dtype,
    exact_dtype,
    exact_misfit_refusal,
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
    require_columns,
    require_not_blank,
    row_status,
    unreadable_report,
)

# cast_columns' words for the same choice. 'set_null', not 'null': YAML reads an unquoted null as no value.
_ON_INVALID_ENUM: Final[tuple[str, ...]] = ("error", "set_null")


class Unpivot:
    """Turn `rows` long: each column outside `by` becomes rows naming it and holding its cells."""

    name = "unpivot"
    description = "Turn columns into rows, such as one column per month into one row per month."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="The columns that stay as they are, such as ['customer']. Every other "
                "column is turned into rows.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One column that stays as it is. Naming it twice counts it once.",
                ),
            ),
            "names_to": ConfigField(
                type="string",
                description="The name of the new column holding each old column's heading, such as 'month'.",
                required=True,
                non_empty=True,
            ),
            "values_to": ConfigField(
                type="string",
                description="The name of the new column holding each old cell's value, such as 'amount'.",
                required=True,
                non_empty=True,
            ),
            "values_type": ConfigField(
                type="string",
                description="Optional: the type to read every turned column as, such as 'Float64'. "
                "Without it, those columns must already share a type.",
                enum=DECLARED_TYPES,
            ),
            "values_places": ConfigField(
                type="integer",
                description="Only with values_type 'Decimal', and required there: the digits after "
                "the point, from 0 to 38, such as 2.",
            ),
            # No default: the rule refusing 'on_invalid' without 'values_type' reads whether it was written.
            "on_invalid": ConfigField(
                type="string",
                description="With 'values_type': 'error' stops the run on a value that will not "
                "convert (the default); 'set_null' makes it null and reports how many.",
                enum=_ON_INVALID_ENUM,
            ),
            "exclude_nulls": ConfigField(
                type="boolean",
                description="Set true to leave out cells that are null, instead of making a row with "
                "no value.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use unpivot to turn a wide table into a long one, for example one column per month "
                "into one row per month, so later steps can filter, join or group by month. This is "
                "the reverse of cross_tab. Each input row gives one output row per turned column: the "
                "'by' columns, in the order you list them, then 'names_to' holding the old heading as "
                "text, then 'values_to' holding the cell."
            ),
            examples=(
                _periods_become_rows(),
                _some_columns_stay_put(),
                _a_stated_type_melts_a_mixed_block(),
                _null_cells_are_left_out(),
                _exact_columns_stack_at_the_larger_places(),
            ),
            notes=(
                "There is no setting listing the columns to turn. Every column not in 'by' is turned; "
                "name a column in 'by' to keep it as it is.",
                "A 'by' that names every column is refused, because nothing would be turned.",
                "The turned columns must share a type, or the step is refused, naming both columns and "
                "types. A column that is all null takes the others' type. This is union's rule.",
                "Rows come out column by column: all the rows of the first turned column, then all of "
                "the second, in the table's order. Nothing is sorted. Sort afterwards if needed.",
                "By default a null cell becomes a row with a null value, not no row: the row existed "
                "and the cell was empty.",
                "'exclude_nulls: true' leaves null cells out, as SQL's UNPIVOT does by default "
                "(EXCLUDE NULLS). Empty text, spaces, zero and not-a-number are values and are kept.",
                "With 'exclude_nulls', a null in a 'by' column drops nothing, and the remaining rows "
                "keep the order the full unpivot gives. The null rows are never built, which saves "
                "memory on a sparse sheet.",
                "With 'values_type' and 'exclude_nulls', a cell is judged as it arrived, so text that "
                "fails to convert under 'set_null' is kept as a row with a null value and a warning mark.",
                "'values_type' reads every turned column as one type, so columns that differ, such as "
                "a block of monthly figures with a text column beside it, can still be turned. Reading "
                "figures as text would lose digits.",
                "A decimal with a fraction does not convert to a whole-number 'values_type': 1.99 is a "
                "value that will not convert. Round it first.",
                "'on_invalid: error' (the default) stops the run on a value that will not convert, "
                "naming the column and the values. 'set_null' makes it null, reports how many out of "
                "how many as a warning, and marks each row; only columns that lost a value are reported.",
                "'on_invalid' without 'values_type' is refused, because there is nothing to convert.",
                "'values_type: Decimal' with 'values_places' reads every turned column as an exact "
                "decimal, by cast_columns' rule: a value that does not fit the places without rounding "
                "will not convert.",
                "Without 'values_type', exact Decimal columns of different places are stacked at the "
                "larger places, which adds zeros and loses nothing. An exact column beside a Float64 "
                "column is refused, naming the float column: convert one of them with cast_columns.",
                "Unlike cast_columns, reading a column as the type it already has is fine here, "
                "because the turned columns are all the ones you did not name.",
                "'names_to' and 'values_to' must be two different new names, not used by 'by' or any "
                "column that is turned.",
            ),
            field_help={
                "by": "For example ['customer_id']. Every other column is turned into rows.",
                "names_to": "For example 'month'. It always holds text.",
                "values_to": "For example 'amount'.",
                "values_type": "For example 'Float64', when a text column sits among the figures.",
                "values_places": "For example 2, beside values_type: Decimal, for figures to the cent.",
                "on_invalid": "Use 'set_null' with 'values_type' when some cells are expected not to "
                "convert, such as notes beside figures.",
                "exclude_nulls": "Set it on a sparse sheet to skip the empty cells.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse blank or clashing new names, places that do not suit the type, and on_invalid without values_type."""
        if config["values_type"] is None and config["values_places"] is not None:
            raise ConfigError(
                f"config field 'values_places' is {config['values_places']!r} with no 'values_type' beside it; "
                f"places belong to values_type: Decimal, so state that type or drop the places"
            )
        if config["values_type"] is not None:
            check_places(config["values_type"], config["values_places"], "values_places")
        by_columns = set(_by_columns(config))
        for field in ("names_to", "values_to"):
            column = config[field]
            require_not_blank(column, field)
            if column in by_columns:
                raise ConfigError(
                    f"config field {field!r} names new column {column!r}, which is also in "
                    f"'by'; the 'by' columns come through as themselves, so one of the two "
                    f"would be lost"
                )

        if config["names_to"] == config["values_to"]:
            raise ConfigError(
                f"config fields 'names_to' and 'values_to' both name column {config['names_to']!r}; they are "
                f"two columns -- the melted column's name and its cell -- and cannot share one name"
            )

        if config["on_invalid"] is not None and config["values_type"] is None:
            raise ConfigError(
                f"config field 'on_invalid' is given as {config['on_invalid']!r} with no "
                f"'values_type' beside it; it says what becomes of a value that will not convert, "
                f"and with no stated type nothing converts -- state a 'values_type', or drop "
                f"'on_invalid'"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Check the columns and types from the schema, so an empty frame is refused the same way."""
        frame = inputs["rows"]
        by_columns = _by_columns(config)
        require_columns(frame, by_columns, "by", "rows")

        melted = _melted_columns(frame, by_columns)
        if not melted:
            raise ConfigError(
                f"config field 'by' names every column on input port 'rows', so nothing is left "
                f"to melt and the step would reshape nothing; the columns present are "
                f"{', '.join(frame.columns)}"
            )

        # A clash with a 'by' column was refused in validate_config; here, a clash with a turned column.
        for field in ("names_to", "values_to"):
            column = config[field]
            if column in melted:
                raise ConfigError(
                    f"config field {field!r} names new column {column!r}, which input port 'rows' "
                    f"already carries and is about to melt; write to a name nothing uses"
                )

        # Called for its refusal: the types must agree, judged from the schema. A stated
        # 'values_type' settles the type itself, so there is nothing to refuse.
        if config["values_type"] is None:
            _settled_dtype(frame, melted)
        elif config["values_type"] == EXACT_TYPE:
            for column in melted:
                dtype = frame.schema[column]
                if not (dtype in (pl.String, pl.Null) or dtype.is_numeric()):
                    raise ConfigError(
                        f"config field 'values_type' reads every turned column as Decimal, but {column!r} "
                        f"holds {dtype}, which is not a number; name it in 'by' so it is not turned, or "
                        f"convert it in the step before this one"
                    )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        by_columns = _by_columns(config)
        melted = _melted_columns(frame, by_columns)

        if config["values_type"] is None:
            # Cast so an all-null column takes the others' type. Nothing can fail to read here.
            settled = _settled_dtype(frame, melted)
            values = {column: _as_settled(frame, column, settled) for column in melted}
            notes: list[str] = []
            blocks: list[pl.DataFrame] | None = None
        else:
            values, notes, blocks = _stated_values(
                frame, melted, config["values_type"], config["values_places"], config["on_invalid"] == "set_null"
            )

        # One frame per turned column, stacked in the frame's column order (column-major).
        # With 'exclude_nulls', null cells are dropped per column before its rows are built, which
        # is what saves memory: do not move the filter after the stack
        # (TestExcludeNullsDoesNotBuildTheRowsItLeavesOut). A cell is judged as it arrived, before
        # conversion. Each status block loses the same rows as its column, so marks stay aligned.
        column_blocks = blocks if blocks is not None else [None] * len(melted)
        parts: list[pl.DataFrame] = []
        kept_blocks: list[pl.DataFrame] = []
        for column, block in zip(melted, column_blocks, strict=True):
            part = frame.select(*(pl.col(name) for name in by_columns))
            value = values[column]
            if config["exclude_nulls"]:
                keep = frame[column].is_not_null()
                part = part.filter(keep)
                value = value.filter(keep)
                if block is not None:
                    block = block.filter(keep)
            parts.append(
                part.with_columns(
                    pl.lit(column, dtype=pl.String).alias(config["names_to"]),
                    value.alias(config["values_to"]),
                )
            )
            if block is not None:
                kept_blocks.append(block)
        return Ports(
            {"rows": pl.concat(parts, how="vertical")},
            notes=notes,
            status={"rows": pl.concat(kept_blocks, how="vertical")} if blocks is not None else None,
        )


def _by_columns(config: Mapping[str, Any]) -> list[str]:
    """The 'by' columns, a column named twice counted once, in the order it was written."""
    return list(dict.fromkeys(config["by"]))


def _melted_columns(frame: pl.DataFrame, by_columns: Sequence[str]) -> list[str]:
    """Every column outside 'by', in the frame's order, read from the schema, not the rows."""
    kept = set(by_columns)
    return [name for name in frame.columns if name not in kept]


def _settled_dtype(frame: pl.DataFrame, melted: Sequence[str]) -> pl.DataType:
    """The one type the turned columns share, or a refusal naming two that differ.

    A Null-typed column is skipped and takes the others' type. If every column is Null, the answer is Null,
    so a run where everything upstream matched nothing keeps its shape. Exact columns settle at 38 digits
    and the largest places among them; an exact column beside a float is refused, naming the float.
    """
    typed = [(name, frame.schema[name]) for name in melted if frame.schema[name] != pl.Null]
    if not typed:
        return pl.Null

    exact = [(name, dtype) for name, dtype in typed if isinstance(dtype, pl.Decimal)]
    if exact:
        floats = [(name, dtype) for name, dtype in typed if dtype.is_float()]
        if floats:
            raise ConfigError(
                f"the columns about to be melted mix exact decimals and floats: {exact[0][0]} is "
                f"{exact[0][1]} and {floats[0][0]!r} is {floats[0][1]}; stacked together the exact figures "
                f"would become floats. Convert one side with cast_columns in the step before this one, "
                f"such as {floats[0][0]!r} to Decimal, or state values_type"
            )
        if len(exact) == len(typed):
            return exact_dtype(max(dtype.scale for _, dtype in exact))  # type: ignore[attr-defined]

    first_name, settled = typed[0]
    for name, dtype in typed[1:]:
        if dtype != settled:
            raise ConfigError(
                f"the columns about to be melted give different types, so one of them would be "
                f"rewritten into the other on the way through: {first_name} is {settled} and "
                f"{name} is {dtype}; cast one of them in the step before this one, or name it in "
                f"'by' so it is not melted at all"
            )
    return settled


def _as_settled(frame: pl.DataFrame, column: str, settled: pl.DataType) -> pl.Series:
    """One turned column in the type the turned columns share.

    An exact column is widened to the shared places through ``to_exact``, which adds zeros and never
    rounds; a value with too many digits before the point to take them is refused by name.
    """
    if not (isinstance(settled, pl.Decimal) and isinstance(frame.schema[column], pl.Decimal)):
        return frame[column].cast(settled)
    places = settled.scale  # type: ignore[attr-defined]
    converted = to_exact(frame[column], places)
    if converted.misfits.any():
        shown = ", ".join(str(value) for value in frame[column].filter(converted.misfits).head(3).to_list())
        raise DataError(
            f"melted column {column!r} holds values too large to stack at {places} places, the most among "
            f"the turned columns, such as {shown}; an exact column holds 38 digits in all, so at most "
            f"{EXACT_DIGITS - places} before the point. Round the column with more places to fewer in a "
            f"formula step, or convert the columns to Float64 with cast_columns",
            offending=frame.filter(converted.misfits),
            total=frame.height,
        )
    return converted.values


def _stated_values(
    frame: pl.DataFrame,
    melted: Sequence[str],
    declared: str,
    places: int | None,
    lenient: bool,
) -> tuple[dict[str, pl.Series], list[str], list[pl.DataFrame] | None]:
    """Each turned column read as the stated type, a note per column that lost a value, and the status blocks.

    cast_columns' rule and wording. Only columns that lost a value get a note. The status comes back as one
    block per turned column, unstacked, because exclude_nulls drops rows per column and each block must
    lose the same rows. None when nothing was lost.
    """
    target = declared_dtype(declared, places)
    values: dict[str, pl.Series] = {}
    notes: list[str] = []
    blocks: list[pl.DataFrame] = []
    degraded = False

    for column in melted:
        if declared == EXACT_TYPE:
            assert places is not None, "validate_config requires places beside Decimal"
            converted = to_exact(frame[column], places)
            if converted.misfits.any() and not lenient:
                raise DataError(
                    exact_misfit_refusal(column, frame[column].filter(converted.misfits), places, melted=True),
                    offending=frame.filter(converted.misfits),
                    total=frame.height,
                )
            if converted.misfits.any():
                reasons, note = unreadable_report(frame, column, converted.misfits, f"Decimal with {places} places")
                blocks.append(row_status([(column, reasons)], frame.height))
                notes.append(note)
                degraded = True
            else:
                blocks.append(row_status([], frame.height))
            values[column] = converted.values
            continue
        if lenient:
            # Counted before the values are replaced, as cast_columns does. Known gap shared with
            # cast_columns: a type pair polars refuses for a whole column (Date to Float64) raises
            # an ExecutionError, not a DataError. Fix both together.
            cut = cut_fractions(frame[column], target)
            read = frame[column].cast(target, strict=False)
            read = pl.select(pl.when(cut).then(None).otherwise(read)).to_series().alias(column)
            failed = frame[column].is_not_null() & read.is_null()
            if failed.any():
                reasons, note = unreadable_report(frame, column, failed, declared)
                blocks.append(row_status([(column, reasons)], frame.height))
                notes.append(note)
                degraded = True
            else:
                blocks.append(row_status([], frame.height))
            values[column] = read
            continue
        cut = cut_fractions(frame[column], target)
        if cut.any():
            raise DataError(
                fraction_refusal(column, frame[column].filter(cut), declared, melted=True),
                offending=frame.filter(cut),
                total=frame.height,
            )
        try:
            values[column] = frame[column].cast(target)
        except Exception as error:
            # A data refusal naming the column. The sample rows are best effort, so a type pair
            # polars refuses outright does not crash the report.
            try:
                offending = frame.filter(
                    pl.col(column).is_not_null() & pl.col(column).cast(target, strict=False).is_null()
                )
            except Exception:  # noqa: BLE001 - the report is best-effort; the refusal is not
                offending = frame.head(0)
            raise DataError(
                f"melted column {column!r} does not read as {declared}: {error}",
                offending=offending,
                total=frame.height,
            ) from error
    return values, notes, (blocks if degraded else None)


def _periods_become_rows() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {"code": ["A", "B"], "p1": [10.0, 7.0], "p2": [5.0, None]},
                schema={"code": pl.String, "p1": pl.Float64, "p2": pl.Float64},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", "B"],
                    "period": ["p1", "p1", "p2", "p2"],
                    "amount": [10.0, 7.0, 5.0, None],
                },
                schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
            )
        }
    )
    return WorkedExample(
        scenario="turn one column per period into one row per period",
        configuration={"by": ["code"], "names_to": "period", "values_to": "amount"},
        explanation="All of p1 comes first, then all of p2. B's p2 cell was empty, so it becomes a row with a "
        "null amount, not a missing row. p1 and p2 are not named anywhere: every column outside 'by' "
        "is turned.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _some_columns_stay_put() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "site": ["north", "south"],
                    "opening": [10.0, 7.0],
                    "closing": [12.0, 6.0],
                },
                schema={"code": pl.String, "site": pl.String, "opening": pl.Float64, "closing": pl.Float64},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", "B"],
                    "site": ["north", "south", "north", "south"],
                    "measure": ["opening", "opening", "closing", "closing"],
                    "amount": [10.0, 7.0, 12.0, 6.0],
                },
                schema={"code": pl.String, "site": pl.String, "measure": pl.String, "amount": pl.Float64},
            )
        }
    )
    return WorkedExample(
        scenario="keep a descriptive column as it is by naming it in 'by'",
        configuration={"by": ["code", "site"], "names_to": "measure", "values_to": "amount"},
        explanation="'site' is text. Named in 'by', it stays as it is; otherwise it would be turned with the "
        "figures and refused for having a different type.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _a_stated_type_melts_a_mixed_block() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": [10.0, 7.0],
                    "p2": [5.0, 2.0],
                    "comment": ["checked", None],
                },
                schema={"code": pl.String, "p1": pl.Float64, "p2": pl.Float64, "comment": pl.String},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"] * 3,
                    "period": ["p1", "p1", "p2", "p2", "comment", "comment"],
                    "amount": [10.0, 7.0, 5.0, 2.0, None, None],
                },
                schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
            )
        }
    )
    return WorkedExample(
        scenario="turn a block of figures that has a text column beside it",
        configuration={
            "by": ["code"],
            "names_to": "period",
            "values_to": "amount",
            "values_type": "Float64",
            "on_invalid": "set_null",
        },
        explanation="'comment' is text and its position changes from load to load, so it cannot be named in 'by'. "
        "'values_type: Float64' reads every turned column as a figure, and 'on_invalid: set_null' "
        "makes the text null and reports how many. A later step drops the rows whose heading is not a "
        "period.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _null_cells_are_left_out() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "record_id": [1, 2, 3],
                    "c1": ["Heading", None, ""],
                    "c2": [None, "Total", "7.5"],
                },
                schema={"record_id": pl.Int64, "c1": pl.String, "c2": pl.String},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "record_id": [1, 3, 2, 3],
                    "column": ["c1", "c1", "c2", "c2"],
                    "value": ["Heading", "", "Total", "7.5"],
                },
                schema={"record_id": pl.Int64, "column": pl.String, "value": pl.String},
            )
        }
    )
    return WorkedExample(
        scenario="turn a sparse sheet into rows, leaving out the empty cells",
        configuration={"by": ["record_id"], "names_to": "column", "values_to": "value", "exclude_nulls": True},
        explanation="The two null cells are left out: record 2's c1 and record 1's c2. Record 3's c1 holds empty "
        "text, which is a value, so it is kept. The rows keep the full unpivot's order: all of c1, "
        "then all of c2.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _exact_columns_stack_at_the_larger_places() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {"code": ["A", "B"], "price": [Decimal("812.40"), Decimal("19.99")], "rate": [Decimal("0.0725"), None]},
                schema={"code": pl.String, "price": pl.Decimal(38, 2), "rate": pl.Decimal(38, 4)},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", "B"],
                    "measure": ["price", "price", "rate", "rate"],
                    "value": [Decimal("812.4000"), Decimal("19.9900"), Decimal("0.0725"), None],
                },
                schema={"code": pl.String, "measure": pl.String, "value": pl.Decimal(38, 4)},
            )
        }
    )
    return WorkedExample(
        scenario="turn exact decimals of different places into one column",
        configuration={"by": ["code"], "names_to": "measure", "values_to": "value"},
        explanation="'price' holds 2 places and 'rate' 4, so the values are stacked at 4 places: 812.40 becomes "
        "812.4000, the same figure. A Float64 column among them would be refused instead.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Unpivot())
