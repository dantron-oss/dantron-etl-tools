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
                "A column whose values will not fit the type the table declares for it is refused "
                "before anything is written, naming the column: text in a whole-number column, or "
                "a fraction. These are the same checks publish_tables runs.",
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
        """Refuse a table that is not there, a column disagreement, an unwelcome empty frame, or a
        column whose values will not fit the column the table declares.

        All of them are checked here, ahead of ``execute``, so nothing is emptied or loaded on the
        strength of a configuration that was already wrong. They are the same checks publish_tables
        runs, from the same module.
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
    """One attempted write, in the terms every question asked about it afterwards is asked in.

    These nine values travelled as nine parameters through five functions, four of which did
    nothing with most of them but pass them on. Threading them by hand is also the shape in which
    ``schema``, ``table`` and ``qualified`` -- three strings in a row, no two of them
    interchangeable and no type able to tell them apart -- get handed over in the wrong order. They
    are settled once, at the top of the write, and everything downstream asks this instead.

    ``rows`` is materialised here rather than where it is needed because the load itself needs it
    and a failure may want it a second time to replay. That holds the whole frame as Python tuples,
    which is fine at the hundred thousand rows a publish writes and is where to look first if that
    ever stops being true.
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
    """Truncate (if replacing) and load ``frame`` in one transaction, naming ``qualified`` on failure.

    Nothing here is written in one driver's dialect. The loader is asked for first, before a single
    statement runs, so a connection from a driver this library does not know is refused by name
    rather than reached into -- and the unit of work, the bulk load and the exception types that
    mean "the rows were wrong" all come from that loader. A caller asking which rows offended, out
    of how many, gets the same answer whichever engine the table happened to live on.
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
    """Which rows offend: read the table's rules and judge the frame, or replay the load row by row.

    Both routes answer the same question and must answer it identically, because a reader fixing a
    bad night reads the answer and never the route. The difference is only what it costs: replaying
    a hundred and thirty thousand rows one at a time is three round trips per row, spent after the
    load has already failed and while its transaction is still open, so a build that hit one bad row
    looks wedged rather than failed.

    The catalogued route is tried first and is allowed to decline. It declines when the table
    carries a rule the rows cannot be judged against, when it cannot read the rules at all, and --
    the important one -- when it finds nothing wrong with a load the server has just refused, which
    means it was missing something whatever it thought it understood. Declining costs the slow
    answer; giving the empty one would cost a reader the night.
    """
    catalogued = _catalogued_offenders(load)
    if catalogued is not None:
        return load.frame[catalogued]
    return _replayed_offenders(load)


def _catalogued_offenders(load: _Load) -> list[int] | None:
    """The rows the table's own rules refuse, judged in memory, or ``None`` if they cannot be.

    The rules come out of the catalog in a handful of statements, none of them against the table, and
    the frame is judged against them without attempting a single row. What counts as offending is
    unchanged: a row missing a value the table demands, and the second and later rows claiming a key
    -- never the first, which a sequential load would have got in. In an append the keys the table
    already holds count too, and they are asked for in one statement, because a frame read against
    itself alone cannot see them.
    """
    try:
        rules = load.loader.table_rules(load.connection, load.schema, load.table)
        if rules.beyond_the_frame or not _rules_cover_the_frame(rules, load.frame):
            return None
        if _carries_a_null_the_engine_would_settle(rules, load.frame):
            return None
        offenders = _refused_rows(load, rules)
    except Exception as failure:  # noqa: BLE001 - any doubt about the rules means the slow answer, never the empty one
        # Said out loud, unlike the declines above it. Those are the rules this route cannot judge --
        # an expression, a key in another table -- and they are the ordinary case, expected on tables
        # that carry them and silent for that reason. This one is a reading that broke, and catching
        # it as widely as trustworthiness demands is exactly what would hide it: the refusal still
        # names the right rows by the slow route, so the only symptom is a diagnosis that used to
        # take a second taking a minute, which reads as a slow server rather than as a fault here.
        logging.getLogger(__name__).warning(
            # Says the rows could not be judged, not that the rules could not be read: the reading
            # is only one of the things under this handler. A value the server returned that will
            # not convert to the frame's own type lands here too, and pointing a reader at the
            # catalog for a comparison that failed sends them to the wrong file.
            "the rows for table %r could not be judged against its rules, so the ones that offended "
            "were found by attempting them one at a time instead, which is correct but far slower: %s",
            load.qualified,
            failure,
        )
        _roll_back_quietly(load.connection)
        return None
    return offenders or None


def _roll_back_quietly(connection: Any) -> None:
    """Undo whatever a declined reading left open, so the route that replaces it starts from a usable connection.

    Nothing was written -- this route only reads -- but on one of the two drivers a statement that
    failed leaves the transaction refusing every statement after it, and the replay is about to open
    one. Its own failure would surface in place of the refusal the caller is owed -- which is also
    why this cannot be allowed to raise on its own account.
    """
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
    """Whether the frame leaves a key for the engine to decide, which the rows cannot then be judged on.

    SQLite fills a null in a rowid-alias key with a number of its own, and a later row holding that
    number collides with a value that was never in the frame. Reading the rows would find nothing
    wrong with either of them. It is only ever a null that does this, so a frame without one in those
    columns -- which is nearly all of them -- takes the quick route as before.
    """
    return any(frame[column].null_count() for column in rules.settled_by_the_engine if column in frame.columns)


def _refused_rows(load: _Load, rules: TableRules) -> list[int]:
    """The row numbers a sequential load would have refused, in frame order.

    A missing value is a property of the row alone, so those are found in one pass per column. A
    repeated key is not: it is a disagreement between a row and everything that got in ahead of it,
    and which rows those are depends on the answer being computed. A row already refused never
    reaches the table, so its key is still free for a later row to take -- which is why the keys are
    walked in order, carrying forward only what was actually accepted, rather than judged against the
    frame as written. Judging the frame as written is right whenever one key rule is in play and
    wrong as soon as two are: a row refused for its key would still be blamed for the second key it
    was carrying, and an innocent row behind it blamed for repeating it.

    The walk costs no statement and no round trip, which is the whole point of this route; it runs
    only after a load has already failed.
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
    """Which of the frame's keys ``column`` already holds -- none, when the table is being emptied.

    Asked in one statement for all of them, and only for the keys the frame actually carries, so a
    table of millions is not read to judge a frame of thousands. What comes back is spelled the
    server's way and has to be compared against the frame's own values, so it is read through the
    frame's type for that column: a strict conversion, because a value that will not convert means
    the comparison about to be made is not one to trust, and the slow route is right there.
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
    """Which rows offend, found by re-running the load row by row in a transaction that is discarded.

    The fallback, and the only route that can answer for a rule the frame cannot be judged against:
    a check expression, a key whose answer lives in another table. It costs an attempt per row, which
    is why it is no longer the first thing tried, and it is kept because the alternative for those
    rules is reporting that nothing offended over a load that was refused.

    Every row is attempted regardless of an earlier failure, so more than one offender is reported at
    once rather than one per run, and each attempt sits inside the loader's own undo, because on one
    of the two drivers a failed statement poisons every statement after it in the same transaction.
    The whole of it is rolled back at the end: this pass exists to look, not to write.
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
