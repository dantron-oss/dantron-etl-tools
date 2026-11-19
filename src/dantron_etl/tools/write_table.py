"""WriteTable: load one frame into one existing table, replacing or appending, all rows or none.

`replace` empties and loads in one transaction, so a failed load leaves the old rows. The tool never
creates a table. When the database refuses rows, the step names the rows that offended.

Why: docs/design/tools.md#write_table
"""

import logging
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import polars as pl

from ..db import (
    RowLoader,
    TableRules,
    driver_for,
    open_connection,
    qualified_name,
    resolve_connection_uri,
    validate_identifier,
)
from ..authoring import (
    REGISTRY,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    example_database,
)
from ._table_load import require_landable_columns, require_matching_table, require_rows

_MODE_ENUM = ("replace", "append")

_EXAMPLE_CONNECTION = "TARGET_DATABASE_URL"
_EXAMPLE_URI = "file:write_table_example?mode=memory&cache=shared"


class WriteTable:
    """Load `rows` into a named table by `mode`, and pass the same rows on."""

    name = "write_table"
    description = "Save rows to one database table, replacing what is there or adding to it."
    category = "output"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "connection": ConfigField(
                type="string",
                description="The name of the environment variable that holds the database address, "
                "such as 'TARGET_DATABASE_URL'. Never the address itself.",
                required=True,
                non_empty=True,
            ),
            "schema": ConfigField(
                type="string",
                description="The schema the table is in, such as 'reporting'. Left out, the table name "
                "is used on its own.",
                required=False,
                default=None,
            ),
            "table": ConfigField(
                type="string",
                description="The table to write, such as 'monthly_sales'. It must already exist.",
                required=True,
                non_empty=True,
            ),
            "mode": ConfigField(
                type="string",
                description="'replace' empties the table and loads the rows; 'append' adds the rows to "
                "what is there.",
                required=True,
                enum=_MODE_ENUM,
            ),
            "allow_empty": ConfigField(
                type="boolean",
                description="Set true to allow writing a table with no rows. Left out, an empty table "
                "stops the run.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use write_table to save rows into one existing database table. 'replace' empties "
                "it and loads the rows in one transaction, so a load that fails part way leaves the "
                "old rows in place, never an empty table. 'append' adds the rows. The same rows pass "
                "on through `rows`. To load several related tables together, use publish_tables."
            ),
            examples=(_replace_example(), _append_example()),
            notes=(
                "'mode' has no default: 'replace' can be run twice safely and 'append' cannot, so the "
                "flow must say which it means.",
                "The table must already exist. This tool never creates or changes a table; that "
                "belongs in a reviewed database migration.",
                "The table's columns and the rows' columns must match exactly, in any order. An extra "
                "or missing column is refused, and so is a name that differs only in case.",
                "A table with no rows stops the run unless 'allow_empty' is true, because a replace "
                "with nothing empties the table.",
                "A column whose values will not fit the table's column type is refused before anything "
                "is written, naming the column: text in a whole-number column, or a fraction. "
                "publish_tables runs the same checks.",
                "If the database refuses rows during the load (a missing required value, a repeated "
                "key), the step stops, names the table and shows the rows that were refused.",
                "The same rows pass on through `rows`, so the flow can continue after the write.",
            ),
            field_help={
                "connection": "For example 'TARGET_DATABASE_URL'. Set that variable in the environment "
                "the flow runs in.",
                "mode": "Use 'replace' for a table rebuilt on every run, 'append' for a log that grows.",
                "allow_empty": "Set it when an empty result is normal for this table.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a schema or table that is not a plain name, before any connection opens."""
        if config["schema"] is not None:
            validate_identifier(config["schema"], "schema")
        validate_identifier(config["table"], "table")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a missing table, a column mismatch, an unallowed empty frame, and values that will not fit.

        All are checked before anything is emptied or loaded; publish_tables runs the same checks.
        """
        frame = inputs["rows"]
        uri = resolve_connection_uri(config["connection"], "connection")
        qualified = qualified_name(config["schema"], config["table"])
        with open_connection(uri) as connection:
            require_matching_table(connection, frame, qualified)
            require_rows("write_table", frame, qualified, config["allow_empty"])
            declared = driver_for(connection).declared_column_types(connection, config["schema"], config["table"])
        require_landable_columns(frame, declared, qualified)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        uri = resolve_connection_uri(config["connection"], "connection")
        qualified = qualified_name(config["schema"], config["table"])
        with open_connection(uri) as connection:
            _write_rows(connection, config["schema"], config["table"], qualified, config["mode"], frame)
        return Ports({"rows": frame})


@dataclass(frozen=True)
class _Load:
    """One attempted write: the loader, connection, names, mode, frame and its rows as tuples.

    Gathered once so schema, table and qualified name cannot be passed in the wrong order. `rows` holds
    the whole frame as tuples, needed by the load and by a replay.
    """

    loader: RowLoader
    connection: Any
    schema: str | None
    table: str
    qualified: str
    mode: str
    frame: pl.DataFrame
    columns: Sequence[str]
    rows: Sequence[tuple[Any, ...]]


def _write_rows(
    connection: Any,
    schema: str | None,
    table: str,
    qualified: str,
    mode: str,
    frame: pl.DataFrame,
) -> None:
    """Empty (when replacing) and load `frame` in one transaction, naming the table on failure.

    The driver's loader supplies the transaction, the bulk load and the errors meaning "the rows were
    wrong", so an unknown driver is refused by name and every engine reports the same way.
    """
    loader = driver_for(connection)
    load = _Load(
        loader=loader,
        connection=connection,
        schema=schema,
        table=table,
        qualified=qualified,
        mode=mode,
        frame=frame,
        columns=list(frame.columns),
        rows=frame.rows(),
    )

    try:
        with loader.transaction(connection):
            if mode == "replace":
                connection.execute(f"DELETE FROM {qualified}")
            loader.load_rows(connection, qualified, load.columns, load.rows)
    except loader.integrity_errors as error:
        offending = _offending_rows(load)
        raise DataError(
            f"writing table {qualified!r} was refused",
            offending=offending,
            total=frame.height,
        ) from error
    except loader.driver_errors as error:
        raise DataError(
            f"writing table {qualified!r} failed: {error}",
            offending=None,
            total=frame.height,
        ) from error


def _offending_rows(load: _Load) -> pl.DataFrame:
    """Which rows offend: judged in memory from the table's rules, or else by replaying the load.

    Both must give the same answer. The judged route is tried first and may decline; declining costs time,
    an empty answer would cost a reader the night. Why: docs/design/tools.md#write_table
    """
    catalogued = _catalogued_offenders(load)
    if catalogued is not None:
        return load.frame[catalogued]
    return _replayed_offenders(load)


def _catalogued_offenders(load: _Load) -> list[int] | None:
    """The rows the table's rules refuse, judged in memory, or None if they cannot be judged.

    Offending means a missing required value, or the second and later rows claiming a key (never the
    first). When appending, keys the table already holds count too.
    """
    try:
        rules = load.loader.table_rules(load.connection, load.schema, load.table)
        if rules.beyond_the_frame or not _rules_cover_the_frame(rules, load.frame):
            return None
        if _carries_a_null_the_engine_would_settle(rules, load.frame):
            return None
        offenders = _refused_rows(load, rules)
    except Exception as failure:  # noqa: BLE001 - any doubt about the rules means the slow answer, never the empty one
        # Logged, unlike the declines above: a reading that broke, which would otherwise show only
        # as a slow diagnosis.
        logging.getLogger(__name__).warning(
            "the rows for table %r could not be judged against its rules, so the ones that offended "
            "were found by attempting them one at a time instead, which is correct but far slower: %s",
            load.qualified,
            failure,
        )
        _roll_back_quietly(load.connection)
        return None
    return offenders or None


def _roll_back_quietly(connection: Any) -> None:
    """Roll back what a declined reading left open, so the replay starts clean. Never raises."""
    try:
        connection.rollback()
    except Exception:  # noqa: BLE001 - the answer being salvaged is the one worth reporting, not this
        pass


def _rules_cover_the_frame(rules: TableRules, frame: pl.DataFrame) -> bool:
    """Whether every rule read names a column the frame carries, and there is at least one to apply."""
    carried = set(frame.columns)
    named = set(rules.not_null) | set(rules.unique)
    return bool(named) and named <= carried


def _carries_a_null_the_engine_would_settle(rules: TableRules, frame: pl.DataFrame) -> bool:
    """Whether a key column the engine fills itself (SQLite's rowid alias) holds a null.

    SQLite would number that row itself, and a later row could collide with a value never in the frame,
    so such a frame takes the slow route.
    """
    return any(frame[column].null_count() for column in rules.settled_by_the_engine if column in frame.columns)


def _refused_rows(load: _Load, rules: TableRules) -> list[int]:
    """The row numbers a load done row by row would have refused, in frame order.

    Missing values are found per column. Repeated keys are walked in order, keeping only accepted rows'
    keys, because a refused row never reaches the table and its key stays free for a later row.
    """
    frame = load.frame
    offenders: set[int] = set()
    for column in rules.not_null:
        offenders.update(_indices_where(frame, pl.col(column).is_null()))
    if not rules.unique:
        return sorted(offenders)

    accepted = {column: _held_values(load, column) for column in rules.unique}
    keys = {column: frame[column].to_list() for column in rules.unique}
    for index in range(frame.height):
        if index in offenders:
            continue
        carried = [(column, keys[column][index]) for column in rules.unique]
        if any(value is not None and value in accepted[column] for column, value in carried):
            offenders.add(index)
            continue
        for column, value in carried:
            if value is not None:
                accepted[column].add(value)
    return sorted(offenders)


def _held_values(load: _Load, column: str) -> set[Any]:
    """Which of the frame's keys the table already holds in `column`; none when replacing.

    One statement, asking only for the frame's keys. Read through the frame's type, strictly, so a value
    that will not convert sends the check to the slow route.
    """
    if load.mode != "append":
        return set()
    frame = load.frame
    wanted = frame[column].drop_nulls().unique().to_list()
    if not wanted:
        return set()
    held = load.loader.existing_values(load.connection, load.qualified, column, wanted)
    return set(pl.Series(column, list(held), dtype=frame.schema[column]).to_list())


def _indices_where(frame: pl.DataFrame, mask: pl.Expr) -> list[int]:
    """The row numbers of ``frame`` where ``mask`` holds."""
    if frame.height == 0:
        return []
    return frame.select(pl.arg_where(mask)).to_series().to_list()


def _replayed_offenders(load: _Load) -> pl.DataFrame:
    """Which rows offend, found by loading row by row in a transaction that is then discarded.

    The only route for rules the frame cannot be judged against (a check expression, a foreign key). Every
    row is attempted, so all offenders are reported at once.
    """
    loader, connection = load.loader, load.connection
    offending_indices = []
    with loader.discarded_transaction(connection):
        if load.mode == "replace":
            connection.execute(f"DELETE FROM {load.qualified}")
        for index, row in enumerate(load.rows):
            try:
                with loader.attempted_row(connection):
                    loader.insert_row(connection, load.qualified, load.columns, row)
            except loader.integrity_errors:
                offending_indices.append(index)
    return load.frame[offending_indices] if offending_indices else load.frame.head(0)


def _seed_example(connection: sqlite3.Connection) -> None:
    """The tables the examples write into."""
    connection.execute("create table sample (key integer, label text)")
    connection.executemany("insert into sample values (?, ?)", [(1, "first"), (2, "second")])


def _example_database() -> AbstractContextManager[None]:
    """The in-memory database the example runs against, named by its connection variable for the run."""
    return example_database(_EXAMPLE_URI, _EXAMPLE_CONNECTION, _seed_example)


def _replace_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    outputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    return WorkedExample(
        scenario="replace a table's rows with new ones",
        configuration={"connection": _EXAMPLE_CONNECTION, "table": "sample", "mode": "replace"},
        explanation="The two rows already in 'sample' are removed and the two new rows loaded. The same rows pass "
        "on through 'rows'.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


def _append_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    outputs = Ports({"rows": pl.DataFrame({"key": [3, 4], "label": ["third", "fourth"]})})
    return WorkedExample(
        scenario="add rows to a table",
        configuration={"connection": _EXAMPLE_CONNECTION, "table": "sample", "mode": "append"},
        explanation="The two rows already in 'sample' stay, and the new rows are added. The same rows pass on "
        "through 'rows'.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


REGISTRY.register(WriteTable())
