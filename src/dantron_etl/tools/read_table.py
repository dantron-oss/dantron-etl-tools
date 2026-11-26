"""ReadTable: read every row of one named database table onto the `rows` port.

The step names a schema and a table, never a query. `connection` names an environment variable that
holds the address. Column types come from the values; an empty table is typed from its definition.
A Postgres numeric arrives exact, at its declared places when it states them.

Why: docs/design/tools.md#read_table
"""

import sqlite3
from collections.abc import Mapping, Sequence, Set
from contextlib import AbstractContextManager
from typing import Any

import polars as pl

from ..column_types import DECLARED_DTYPE, EXACT_DIGITS, exact_dtype, exact_from_text
from ..db import (
    driver_for,
    open_connection,
    probe_columns,
    qualified_name,
    quoted_identifier,
    resolve_connection_uri,
    validate_identifier,
)
from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    example_database,
)

_EXAMPLE_CONNECTION = "SOURCE_DATABASE_URL"
_EXAMPLE_URI = "file:read_table_example?mode=memory&cache=shared"


class ReadTable:
    """Read every row of one named table onto the `rows` port."""

    name = "read_table"
    description = "Read every row of one database table."
    category = "input"
    input_ports = ()
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "connection": ConfigField(
                type="string",
                description="The name of the environment variable that holds the database address, "
                "such as 'SOURCE_DATABASE_URL'. Never the address itself.",
                required=True,
                non_empty=True,
            ),
            "schema": ConfigField(
                type="string",
                description="The schema the table is in, such as 'sales'. Left out, the table name is "
                "used on its own.",
                required=False,
                default=None,
            ),
            "table": ConfigField(
                type="string",
                description="The table to read, such as 'orders'. A plain name, never a query.",
                required=True,
                non_empty=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use read_table to start a flow from a database table. Every row of the table goes to "
                "`rows`, each column typed by the values it holds. There is no query setting: to keep "
                "fewer rows or columns, put filter or select_columns after it."
            ),
            examples=(_reads_a_table_example(),),
            notes=(
                "'connection' names an environment variable that holds the address, so the address "
                "and its password never sit in the pipeline file.",
                "Each column's type comes from all its values, not a sample. A column that is null in "
                "every row has type Null.",
                "An empty table is typed from its definition instead: whole numbers, decimals and "
                "text, and on Postgres also true/false, dates, timestamps without a zone and times.",
                "In an empty table, a column whose values could arrive as more than one kind (a "
                "numeric with no places stated, a zone-aware timestamp, any SQLite column of another "
                "type) stays Null.",
                "A Postgres numeric(p,s) column arrives as Decimal with s places, whatever its values, "
                "and so does an empty table's. 38 digits are kept in all.",
                "A Postgres numeric with no places stated arrives at the most places any of its values "
                "has, so its type can change from one night to the next. Declare the places at the "
                "source, such as numeric(18,2), and the type stays put.",
                "A numeric value a Decimal cannot hold (more than 38 digits, or NaN) stops the step, "
                "naming the column.",
                "SQLite has no exact type: its numbers arrive as Int64 or Float64.",
                "Rows arrive in no particular order. Put a sort after it if order matters.",
                "The schema and table names must be plain names. Anything else is refused before a "
                "connection opens.",
            ),
            field_help={
                "connection": "For example 'SOURCE_DATABASE_URL'. Set that variable in the environment "
                "the flow runs in.",
                "schema": "For example 'sales'. The table is then read as sales.orders.",
                "table": "For example 'orders'.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a schema or table that is not a plain name, before any connection opens.

        A name is put into the SQL text and cannot be a parameter, so this check is the defence against a
        name carrying a second statement.
        """
        if config["schema"] is not None:
            validate_identifier(config["schema"], "schema")
        validate_identifier(config["table"], "table")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        uri = resolve_connection_uri(config["connection"], "connection")
        qualified = qualified_name(config["schema"], config["table"])
        with open_connection(uri) as connection:
            columns = probe_columns(connection, qualified)
            numeric = _numeric_columns(connection, config["schema"], config["table"])
            for name, (digits, places) in numeric.items():
                _refuse_places_past_the_limit(name, digits, places, qualified)
            # Every row decides the type, not a sample. Why: docs/design/tools.md#read_table
            query = select_statement(qualified, columns, set(numeric))
            frame = pl.read_database(query, connection, infer_schema_length=None)
            frame = exact_from_numeric_text(frame, numeric, qualified)
            if frame.height == 0:
                frame = _typed_from_the_table(frame, connection, config["schema"], config["table"])
        return Ports({"rows": exact_at_full_width(frame)})


def _numeric_columns(connection: Any, schema: str | None, table: str) -> dict[str, tuple[int | None, int | None]]:
    """Each exact numeric column of the table and what it declares; none for a driver this library does not load."""
    try:
        loader = driver_for(connection)
    except ConfigError:
        return {}
    return loader.exact_columns(connection, schema, table)


def _typed_from_the_table(frame: pl.DataFrame, connection: Any, schema: str | None, table: str) -> pl.DataFrame:
    """An empty read, each column typed as a row of it would have arrived, from the table's definition.

    A column whose type depends on the value stays Null. A connection opened by ADBC is left alone: that
    driver already types an empty read.
    """
    try:
        loader = driver_for(connection)
    except ConfigError:
        return frame
    types = loader.value_types(connection, schema, table)
    return frame.with_columns(
        pl.col(name).cast(DECLARED_DTYPE[types[name]])
        for name, dtype in frame.schema.items()
        if dtype == pl.Null and types.get(name) is not None
    )


def select_statement(qualified: str, columns: Sequence[str], numeric: Set[str]) -> str:
    """The query that reads every row: each numeric column as its text, every other column as it is.

    The text of a numeric keeps every digit, which a driver's own number might not. A table with no
    numeric is read with a plain ``SELECT *``.
    """
    if not numeric:
        return f"SELECT * FROM {qualified}"
    listed = (
        f"{quoted_identifier(name)}::text AS {quoted_identifier(name)}" if name in numeric else quoted_identifier(name)
        for name in columns
    )
    return f"SELECT {', '.join(listed)} FROM {qualified}"


def exact_from_numeric_text(
    frame: pl.DataFrame, declared: Mapping[str, tuple[int | None, int | None]], qualified: str
) -> pl.DataFrame:
    """``frame`` with each numeric column, read as text, made exact.

    ``declared`` gives each numeric column's stated digits and places, both None when none were
    stated. A column with places stated arrives with those places, whatever its values, even with
    none. A column without arrives at the most places any of its values has, and stays Null when it
    has no value. A value an exact column cannot hold stops the read, naming the column.
    """
    for name, (digits, places) in declared.items():
        _refuse_places_past_the_limit(name, digits, places, qualified)
    converted: list[pl.Series] = []
    for name, (digits, places) in declared.items():
        text = frame[name].cast(pl.String)
        if places is None:
            if text.null_count() == text.len():
                converted.append(pl.Series(name, [None] * text.len(), dtype=pl.Null))
                continue
            longest = text.str.extract(r"\.([0-9]+)$", 1).str.len_chars().max()
            held = min(int(longest or 0), EXACT_DIGITS)
        else:
            held = max(places, 0)
        conversion = exact_from_text(text, held)
        if conversion.misfits.any():
            shown = ", ".join(text.filter(conversion.misfits).unique(maintain_order=True).head(3).to_list())
            raise DataError(
                _numeric_refusal(name, digits, places, held, shown, qualified),
                offending=frame.filter(conversion.misfits),
                total=frame.height,
            )
        converted.append(conversion.values)
    return frame.with_columns(converted) if converted else frame


def exact_at_full_width(frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` with each exact column held to 38 digits in all, keeping its places.

    A driver may hand over a ``numeric(12,2)`` as 12 digits. Every exact column in a pipeline has 38,
    so two columns of the same places always match. Widening the total loses nothing.
    """
    narrow = {
        name: exact_dtype(dtype.scale or 0)
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Decimal) and dtype.precision != EXACT_DIGITS
    }
    return frame.cast(narrow) if narrow else frame


def _refuse_places_past_the_limit(name: str, digits: int | None, places: int | None, qualified: str) -> None:
    if places is not None and places > EXACT_DIGITS:
        raise ConfigError(
            f"column {name!r} of table {qualified!r} is declared numeric({digits},{places}): {places} places, "
            f"more than an exact column holds. An exact column holds {EXACT_DIGITS} digits in all, so at most "
            f"{EXACT_DIGITS} places. Declare fewer places at the source, or read the table through a view that "
            f"rounds the column"
        )


def _numeric_refusal(
    name: str, digits: int | None, places: int | None, held: int, shown: str, qualified: str
) -> str:
    if places is None:
        how = (
            f"It is declared numeric with no places stated, so it arrives at the most places any of its values "
            f"has, {held} here"
        )
        remedy = "Declare its places at the source, such as numeric(18,2)"
    else:
        how = f"It is declared numeric({digits},{places}), so it arrives with {held} places"
        remedy = f"Declare it with at most {EXACT_DIGITS} digits at the source, such as numeric({EXACT_DIGITS},{held})"
    return (
        f"column {name!r} of table {qualified!r} holds values an exact column cannot hold, such as {shown}. {how}. "
        f"An exact column holds {EXACT_DIGITS} digits in all, so at {held} places it holds at most "
        f"{EXACT_DIGITS - held} digits before the point, and it holds no NaN or Infinity. {remedy}, or read the "
        f"table through a view that converts the column to double precision"
    )


def _seed_example(connection: sqlite3.Connection) -> None:
    """The tables the example reads."""
    connection.execute("create table sample (key integer, label text)")
    connection.executemany("insert into sample values (?, ?)", [(1, "first"), (2, "second")])


def _example_database() -> AbstractContextManager[None]:
    """The in-memory database the example runs against, named by its connection variable for the run."""
    return example_database(_EXAMPLE_URI, _EXAMPLE_CONNECTION, _seed_example)


def _reads_a_table_example() -> WorkedExample:
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "label": ["first", "second"]})})
    return WorkedExample(
        scenario="read a table from a database",
        configuration={"connection": _EXAMPLE_CONNECTION, "table": "sample"},
        explanation="'connection' names the environment variable that holds the database address. Each column "
        "comes in with the type of its values.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_example_database,
    )


REGISTRY.register(ReadTable())
