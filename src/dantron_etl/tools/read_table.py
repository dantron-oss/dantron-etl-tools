"""ReadTable: a named table in a database, arriving on one port with the database's own types.

The catalog's first source, and the whole of what it may name is a schema and a table -- never a
query. A ``query`` field would be arbitrary SQL arriving from a configuration file: nothing could
lift its column references out to pre-flight them, its dialect would tie a portable definition to
one database, and it would duplicate Filter and SelectColumns in a second language none of this
library's checks can read. So a step names two identifiers instead, each held to a plain-name
grammar and checked before a connection opens, and narrowing a read is a Filter or a SelectColumns
put after it, where the columns named are visible to the same machinery that checks every other
step.

``connection`` names an environment variable rather than holding a connection string, so a pipeline
definition naming this tool is safe to commit and safe to hand to another client as a template --
see :mod:`~.db` for the whole of that ruling and the refusals it earns.

Rows arrive in no particular order: a database promises none without an ``ORDER BY``, and the
standing rule in this library is that a step needing meaningful order puts a Sort in front of it
rather than leaning on a source to keep a promise it cannot make.
"""

import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import polars as pl

from ..db import open_connection, probe_columns, qualified_name, resolve_connection_uri, validate_identifier
from ..authoring import (
    REGISTRY,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    temporary_environment,
)

_EXAMPLE_CONNECTION = "SOURCE_DATABASE_URL"
_EXAMPLE_URI = "file:read_table_example?mode=memory&cache=shared"


class ReadTable:
    """Read every row of one named table onto the ``rows`` port."""

    name = "read_table"
    description = "Read a named table from a database onto the `rows` port."
    category = "input"
    input_ports = ()
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "connection": ConfigField(
                type="string",
                description="The name of an environment variable holding the connection URI; never the URI itself.",
                required=True,
                non_empty=True,
            ),
            "schema": ConfigField(
                type="string",
                description="The schema the table sits in; left out, the table is named unqualified.",
                required=False,
                default=None,
            ),
            "table": ConfigField(
                type="string",
                description="The table to read; a plain identifier, never a query.",
                required=True,
                non_empty=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every row of the named table arrives on `rows`, with the columns and the dtypes "
                "the database gave them -- this tool invents neither. No `query` field exists: "
                "narrowing what comes back is a Filter or a SelectColumns put after this one."
            ),
            examples=(_reads_a_table_example(),),
            notes=(
                "`connection` names an environment variable holding the connection URI; the URI "
                "itself never sits in a configuration file.",
                "Rows arrive in no particular order. A step that needs one puts a Sort in front "
                "of it, which is why no worked example here claims to preserve order.",
            ),
            field_help={
                "schema": "Left out, `table` is read unqualified; given, the two compose as "
                "`schema.table`, the only form either is ever handed to the database.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a schema or a table that is not a plain identifier, before any connection opens.

        An identifier is spliced into a statement and cannot be bound as a parameter, so this
        grammar check is the whole of the defence against a name carrying a second statement --
        which is why it runs here, ahead of every step that would otherwise touch a database.
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
            # Every row decides the dtype, not the first hundred. A column that is null for its
            # first hundred rows and holds a value afterwards is ordinary in a real table -- an
            # identifier only recent rows carry, a field added by a later migration -- and sampling
            # infers `Null` from the head and then fails on the first real value, or, worse, infers
            # a narrower type than the column needs and mistypes it without saying so. This costs a
            # pass over the result and buys a source that cannot invent a schema the table does not
            # have, which is the promise this tool's docstring makes.
            frame = pl.read_database(f"SELECT * FROM {qualified}", connection, infer_schema_length=None)
        return Ports({"rows": frame})


@contextmanager
def _example_database() -> Iterator[None]:
    """An in-memory database the example reads, entered around the example and torn down after.

    Built here rather than inside ``help()``, which must describe the tool without changing the
    process doing the describing: the catalog page generator calls ``help()`` on every tool, and a
    description that exported a variable or opened a connection as a side effect would be exactly
    the disturbance ``check_tool`` now refuses. The anchor connection is what keeps the shared,
    in-memory database alive for the whole of the block; closing it on the way out, in every case,
    is what lets the same example run again the next time the catalog is checked.
    """
    anchor = sqlite3.connect(_EXAMPLE_URI, uri=True)
    try:
        anchor.execute("create table sample (key integer, label text)")
        anchor.executemany("insert into sample values (?, ?)", [(1, "first"), (2, "second")])
        anchor.commit()
        with temporary_environment(**{_EXAMPLE_CONNECTION: _EXAMPLE_URI}):
            yield
    finally:
        anchor.close()


def _reads_a_table_example() -> WorkedExample:
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "label": ["first", "second"]})})
    return WorkedExample(
        scenario="read a two-row table from an in-memory database",
        configuration={"connection": _EXAMPLE_CONNECTION, "table": "sample"},
        explanation="'connection' names the environment variable the URI lives in, never the URI "
        "itself; the rows arrive with the columns and the dtypes the database gave them.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_example_database,
    )


REGISTRY.register(ReadTable())
