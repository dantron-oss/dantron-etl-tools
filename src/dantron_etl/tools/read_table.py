"""ReadTable: read every row of one named database table onto the `rows` port.

The step names a schema and a table, never a query. `connection` names an environment variable that
holds the address. Column types come from the values; an empty table is typed from its definition.

Why: docs/design/tools.md#read_table
"""

import sqlite3
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

import polars as pl

from ..column_types import DECLARED_DTYPE
from ..db import (
    driver_for,
    open_connection,
    probe_columns,
    qualified_name,
    resolve_connection_uri,
    validate_identifier,
)
from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
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
    description = "Read a database table into a table."
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
                "numeric, a zone-aware timestamp, any SQLite column of another type) stays Null.",
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
            probe_columns(connection, qualified)
            # Every row decides the type, not a sample. Why: docs/design/tools.md#read_table
            frame = pl.read_database(f"SELECT * FROM {qualified}", connection, infer_schema_length=None)
            if frame.height == 0:
                frame = _typed_from_the_table(frame, connection, config["schema"], config["table"])
        return Ports({"rows": frame})


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
