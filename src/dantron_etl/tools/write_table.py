"""WriteTable: truncate-and-load or append into a named table, all of the rows or none of them.

The catalog's first sink. Two rulings carry the whole design.

**Truncate-and-load is one transaction.** A ``replace`` empties the table and loads the frame as a
single unit, so a write that fails part way -- a ``NOT NULL`` column fed a missing value, two rows
claiming the same key -- leaves the previous figures standing rather than an empty table. That is
the one property worth protecting above all others here: the failure mode it excludes is a build
that deletes yesterday's numbers, falls over on row four hundred, and reports an error over a table
that is now empty.

**This tool never creates a table.** A table created from whatever the frame happened to infer this
run has types decided by one day's data, no key, no not-null and no index -- the very constraints
that make truncate-and-load safe -- and a typo in ``table`` would quietly create a second table,
report success, and leave the real one going stale. DDL belongs in a migration somebody reviewed, so
there is no field here to turn that on.

``mode`` carries no default: ``replace`` and ``append`` differ in whether running the flow twice is
safe, and defaulting to either would silently choose it for an author who meant the other one. The
rows must supply exactly the table's columns, order-insensitively -- an extra column would be
dropped, a missing one would be filled with a null where a figure belongs, and a case-only
difference is refused rather than guessed. An empty frame stops the run unless ``allow_empty`` says
otherwise, because a replace of nothing is the loudest and most expensive silence a nightly build
can leave behind. The output port is a pass-through of the rows just written, not a receipt: the
receipt loses the data a flow might want to write twice or assert on, and every one of its
interesting fields would drift between runs in a way a documented example cannot survive.
"""

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import polars as pl

from ..db import open_connection, probe_columns, qualified_name, resolve_connection_uri, validate_identifier
from ..tool import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    temporary_environment,
)

_MODE_ENUM = ("replace", "append")

_EXAMPLE_CONNECTION = "TARGET_DATABASE_URL"
_EXAMPLE_URI = "file:write_table_example?mode=memory&cache=shared"


class WriteTable:
    """Load `rows` into a named table by `mode`, and hand the same rows on unchanged."""

    name = "write_table"
    description = "Truncate-and-load or append `rows` into a named table."
    category = "output"
    input_ports = ("rows",)
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
                description="The table to write; a plain identifier, and one this tool never creates.",
                required=True,
                non_empty=True,
            ),
            "mode": ConfigField(
                type="string",
                description="'replace' empties the table before loading; 'append' loads without "
                "emptying it. Required, with no default: the two differ in whether running the "
                "flow twice is safe.",
                required=True,
                enum=_MODE_ENUM,
            ),
            "allow_empty": ConfigField(
                type="boolean",
                description="Whether a frame with no rows is written instead of stopping the run.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "'rows' is loaded into the named table: 'replace' empties it first, inside the "
                "same transaction as the load, so a write that fails part way leaves the previous "
                "contents standing rather than an empty table; 'append' loads without emptying it. "
                "The table must already exist and its columns must exactly match the frame's, "
                "order-insensitively -- this tool runs no DDL of its own."
            ),
            examples=(_replace_example(), _append_example()),
            notes=(
                "'mode' has no default: 'replace' and 'append' differ in whether re-running the "
                "flow is safe, and either default would choose one of them silently.",
                "An empty frame stops the run unless 'allow_empty' says otherwise: a replace of "
                "nothing empties the table, the loudest silence an upstream step can leave behind.",
                "The output port hands the same rows on unchanged, so this sink is composable "
                "rather than terminal by construction.",
            ),
            field_help={
                "mode": "Required; 'replace' or 'append', with no default, so a run always says which one it means.",
                "allow_empty": "Defaults to false: an empty frame stops the run until a flow says "
                "in writing that this branch may be empty.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a schema or a table that is not a plain identifier, before any connection opens."""
        if config["schema"] is not None:
            validate_identifier(config["schema"], "schema")
        validate_identifier(config["table"], "table")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a table that is not there, a column disagreement, or an unwelcome empty frame.

        All three are checked here, ahead of ``execute``, so that nothing is emptied or loaded on
        the strength of a configuration that was already wrong: probing the table's columns is the
        same query that proves the table exists, and the frame's own columns are compared against
        it before a single row moves.
        """
        frame = inputs["rows"]
        uri = resolve_connection_uri(config["connection"], "connection")
        qualified = qualified_name(config["schema"], config["table"])
        with open_connection(uri) as connection:
            table_columns = probe_columns(connection, qualified)

        _require_matching_columns(frame, table_columns, qualified)

        if frame.height == 0 and not config["allow_empty"]:
            raise DataError(
                f"write_table refuses an empty frame for table {qualified!r}; set allow_empty to accept it",
                offending=frame,
                total=0,
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        uri = resolve_connection_uri(config["connection"], "connection")
        qualified = qualified_name(config["schema"], config["table"])
        with open_connection(uri) as connection:
            _write_rows(connection, qualified, config["mode"], frame)
        return Ports({"rows": frame})


def _require_matching_columns(frame: pl.DataFrame, table_columns: Sequence[str], qualified: str) -> None:
    """Refuse rows whose columns are not exactly the table's, in either direction, case-sensitively."""
    frame_columns = set(frame.columns)
    table_column_set = set(table_columns)
    if frame_columns == table_column_set:
        return

    extra = sorted(frame_columns - table_column_set)
    missing = sorted(table_column_set - frame_columns)
    disagreement = "; ".join(
        part
        for part in (
            f"extra: {', '.join(extra)}" if extra else "",
            f"missing: {', '.join(missing)}" if missing else "",
        )
        if part
    )
    raise ConfigError(
        f"the rows do not match the columns of table {qualified!r} ({disagreement}); the rows "
        f"carry: {', '.join(frame.columns)}; the table has: {', '.join(table_columns)}"
    )


def _write_rows(connection: Any, qualified: str, mode: str, frame: pl.DataFrame) -> None:
    """Truncate (if replacing) and load ``frame`` in one transaction, naming ``qualified`` on failure."""
    columns = list(frame.columns)
    rows = frame.rows()
    insert_sql = f"INSERT INTO {qualified} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"

    try:
        with connection:
            if mode == "replace":
                connection.execute(f"DELETE FROM {qualified}")
            if rows:
                connection.executemany(insert_sql, rows)
    except sqlite3.IntegrityError as error:
        offending = _offending_rows(connection, qualified, mode, insert_sql, frame, rows)
        raise DataError(
            f"writing table {qualified!r} was refused",
            offending=offending,
            total=frame.height,
        ) from error
    except sqlite3.Error as error:
        raise DataError(
            f"writing table {qualified!r} failed: {error}",
            offending=frame.head(0),
            total=frame.height,
        ) from error


def _offending_rows(
    connection: Any,
    qualified: str,
    mode: str,
    insert_sql: str,
    frame: pl.DataFrame,
    rows: Sequence[tuple[Any, ...]],
) -> pl.DataFrame:
    """Which rows offend, found by re-running the load row by row in a transaction that is discarded.

    A bulk failure is unambiguous about *that* something offended but not *which* row did, and a
    reader fixing a bad night needs the row. Re-running one row at a time costs nothing on the
    happy path -- this only runs after the bulk attempt has already failed and rolled back -- and
    every row is attempted regardless of an earlier failure, so more than one offender is reported
    at once rather than one per run. The whole of it is rolled back at the end: this pass exists to
    look, not to write.
    """
    if mode == "replace":
        connection.execute(f"DELETE FROM {qualified}")
    offending_indices = []
    for index, row in enumerate(rows):
        try:
            connection.execute(insert_sql, row)
        except sqlite3.IntegrityError:
            offending_indices.append(index)
    connection.rollback()
    return frame[offending_indices] if offending_indices else frame.head(0)


@contextmanager
def _example_database() -> Iterator[None]:
    """An in-memory database the examples write into, entered around each example and torn down after.

    Built here rather than inside ``help()``, for the same reason as every source's: the catalog
    page generator calls ``help()`` on every tool, and a description that opened a connection or
    exported a variable as a side effect of describing itself is exactly the disturbance
    ``check_tool`` now refuses.
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


def _replace_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    outputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    return WorkedExample(
        scenario="replace empties the table before loading the new rows",
        configuration={"connection": _EXAMPLE_CONNECTION, "table": "sample", "mode": "replace"},
        explanation="The two rows already in 'sample' are gone afterwards, replaced by exactly the "
        "rows on 'rows'; the same rows leave on the output port unchanged.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


def _append_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    outputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    return WorkedExample(
        scenario="append adds to what the table already holds",
        configuration={"connection": _EXAMPLE_CONNECTION, "table": "sample", "mode": "append"},
        explanation="The two rows already in 'sample' stay; the rows on 'rows' are added beside "
        "them, and those same rows leave on the output port unchanged.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


REGISTRY.register(WriteTable())
