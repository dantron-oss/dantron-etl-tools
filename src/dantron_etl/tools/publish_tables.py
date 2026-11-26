"""PublishTables: write several related tables in one transaction: all of them or none.

Each entry reads one port and replaces, appends to or merges into one existing table. The stamp time
is read once and written on every table. Every check runs before anything is emptied.

Why: docs/design/tools.md#publish_tables
"""

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import polars as pl

from ..db import (
    RowLoader,
    driver_for,
    open_connection,
    qualified_name,
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
from ._table_load import (
    landing_frame,
    require_landable_columns,
    require_matching_table,
    require_one_row_per_key,
    require_rows,
)

#: Spelled so a definition file hands each of them over as the word itself.
_MODE_ENUM: Final[tuple[str, ...]] = ("replace", "append", "merge")

# What a merge does with a null in an incoming row. No default: the two publish different figures.
_WHEN_NULL_ENUM: Final[tuple[str, ...]] = ("keep_published", "overwrite")

_EXAMPLE_CONNECTION: Final[str] = "TARGET_DATABASE_URL"
_EXAMPLE_URI: Final[str] = "file:publish_tables_example?mode=memory&cache=shared"


@dataclass(frozen=True, slots=True)
class _Entry:
    """One table of a publish, with its qualified name worked out once."""

    port: str
    table: str
    qualified: str
    mode: str
    key: tuple[str, ...]
    allow_empty: bool
    #: For a merge, what a null in an incoming row does: "keep_published" or "overwrite". None otherwise.
    when_incoming_is_null: str | None


@dataclass(frozen=True, slots=True)
class _Stamp:
    """The time this publish carries, and the column every table carries it in."""

    column: str
    at: datetime


class PublishTables:
    """Write every named frame to its own table, in one transaction, and give back one receipt row."""

    name = "publish_tables"
    description = "Save several tables to a database together: all of them, or none if any fails."
    category = "output"
    # Empty: the ports come from 'tables', through input_ports_for.
    input_ports = ()
    output_ports = ("receipt",)

    def input_ports_for(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        """One input port per table entry, in the order written.

        This sees the configuration before it is checked, so it skips anything malformed. The config check
        that follows refuses it by name.
        """
        entries = config.get("tables")
        if not isinstance(entries, list):
            return ()
        return tuple(
            entry["port"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("port"), str) and entry["port"]
        )

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
                description="The schema every table is in, such as 'reporting'. Left out, the table "
                "names are used on their own.",
                required=False,
                default=None,
            ),
            "stamp_column": ConfigField(
                type="string",
                description="Optional: a column every table has, which this step fills with the time "
                "of this publish, the same value on every table.",
                required=False,
                default=None,
            ),
            "tables": ConfigField(
                type="array",
                description="One entry per table: the input port to read, the table to write, and how.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One table: the port its rows come from, the table name, and the mode.",
                    fields={
                        "port": ConfigField(
                            type="string",
                            description="The input port whose rows go to this table, such as 'totals'.",
                            required=True,
                            non_empty=True,
                        ),
                        "table": ConfigField(
                            type="string",
                            description="The table to write, such as 'monthly_totals'. It must already exist.",
                            required=True,
                            non_empty=True,
                        ),
                        "mode": ConfigField(
                            type="string",
                            description="'replace' empties the table and loads; 'append' adds rows; "
                            "'merge' updates rows matching 'keys' and inserts the rest.",
                            required=True,
                            enum=_MODE_ENUM,
                        ),
                        "keys": ConfigField(
                            type="array",
                            description="For 'merge' only: the columns that identify a row, such as "
                            "['customer_id'].",
                            non_empty=True,
                            items=ConfigField(
                                type="string",
                                description="One key column.",
                                non_empty=True,
                            ),
                        ),
                        "when_incoming_is_null": ConfigField(
                            type="string",
                            description="For 'merge' only: when a new row has a null, 'keep_published' "
                            "keeps the value already in the table; 'overwrite' writes the null.",
                            enum=_WHEN_NULL_ENUM,
                        ),
                        "allow_empty": ConfigField(
                            type="boolean",
                            description="Set true to allow this table's rows to be empty. Left out, an "
                            "empty port stops the whole publish.",
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use publish_tables to update several related tables together, such as a summary and "
                "its detail, so they always agree. Each entry in 'tables' reads one input port and "
                "writes one existing table. All the tables are written in one transaction: if any "
                "fails, every table stays as the last publish left it. `receipt` is one row with the "
                "publish time (when 'stamp_column' is set) and the number of rows written to each table."
            ),
            examples=(_together_example(), _merge_example()),
            notes=(
                "All the tables are saved together or none are. That is why this tool exists instead "
                "of several write_table steps, which are one transaction each.",
                "The stamp time is read once and written identically onto every table, so the tables "
                "of one publish can be picked out together later. It is in UTC, without a zone.",
                "A merge entry must set 'when_incoming_is_null'. 'keep_published' writes "
                "coalesce(incoming, published): a value overwrites, a gap is filled, and a null leaves "
                "the published value standing.",
                "'overwrite' writes the null, as SQL's MERGE does. There is no default, because the "
                "two publish different figures.",
                "A merge needs one row per key. A null in a key column, or the same key twice, is "
                "refused before anything is written, naming the key and the rows.",
                "'mode' has no default: the three differ in whether running the flow twice is safe, "
                "and an append-only log is the one table a rebuild must never rewrite.",
                "An empty port stops the whole publish, not only its own table, unless that entry has "
                "'allow_empty: true'.",
                "The tables must already exist; this tool never creates one. Each frame's columns must "
                "match its table's exactly, apart from 'stamp_column', which the frames never carry.",
                "A column whose values will not fit the table's column type is refused before anything "
                "is written. write_table runs the same checks.",
                "A number goes into a numeric(p,s) column only when every value fits s places and p "
                "digits; the database would round the rest without a word. A Float64 is read to 15 "
                "significant digits first, as round reads it, and that reading is what is written: "
                "10.25 fits numeric(12,2), 10.255 is refused. Round first with a formula step.",
                "A Decimal column is never written to a double precision, real or float column. Convert "
                "it with cast_columns to Float64 first. A Float64 into a float column is written as it is.",
                "SQLite has no exact type: a Decimal is sent as its digits, and SQLite stores them by "
                "its own rule. A numeric(p,s) column is checked the same way on both.",
                "A port or table named by two entries is refused, and so is a 'stamp_column' named "
                "like one of the tables.",
                "A merge loads its rows into a temporary table named _etl_merge_staging first. A "
                "permanent table of that name is hidden while a merge runs.",
            ),
            field_help={
                "connection": "For example 'TARGET_DATABASE_URL'. Set that variable in the environment "
                "the flow runs in.",
                "stamp_column": "For example 'published_at'. Every table here must have it, and no "
                "frame may.",
                "tables": "For example {port: totals, table: monthly_totals, mode: replace}. Each port "
                "named here becomes an input of this step.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse what the configuration alone shows, before any connection opens.

        A merge without keys or without `when_incoming_is_null`, that setting on another mode, a repeated port
        or table, and a stamp named like a table. Each would otherwise fail silently.
        """
        if config["schema"] is not None:
            validate_identifier(config["schema"], "schema")

        entries = _entries(config)
        for entry in entries:
            validate_identifier(entry.table, "tables[].table")
            if entry.mode == "merge" and not entry.key:
                raise ConfigError(
                    f"the entry publishing table {entry.qualified!r} is a merge with no 'keys'; a "
                    f"merge matches a published row on the key, so there is nothing it could do "
                    f"without one"
                )
            if entry.mode == "merge" and entry.when_incoming_is_null is None:
                raise ConfigError(
                    f"the entry publishing table {entry.qualified!r} is a merge that does not say what a "
                    f"null in an incoming row does; write when_incoming_is_null: keep_published to leave "
                    f"the published value standing, or when_incoming_is_null: overwrite to write the null"
                )
            if entry.mode != "merge" and entry.when_incoming_is_null is not None:
                raise ConfigError(
                    f"the entry publishing table {entry.qualified!r} sets 'when_incoming_is_null', which "
                    f"only a merge reads; this entry is a {entry.mode!r}, so remove it"
                )

        _refuse_repeats(entry.port for entry in entries)
        _refuse_repeats((entry.qualified for entry in entries), what="table")
        _refuse_a_stamp_named_after_a_table(config["stamp_column"], entries)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse every problem with the tables and frames before anything is written.

        The first pass checks each table against the configuration (exists, columns, merge key), so the author
        hears which entry is wrong. The second checks the rows (empty frames, values that will not fit, keys).
        """
        entries = _entries(config)
        stamp = config["stamp_column"]
        uri = resolve_connection_uri(config["connection"], "connection")
        with open_connection(uri) as connection:
            for entry in entries:
                _require_a_table_these_rows_can_be_published_to(connection, entry, inputs[entry.port], stamp)

            loader = driver_for(connection)
            for entry in entries:
                frame = inputs[entry.port]
                require_rows(
                    "publish_tables",
                    frame,
                    entry.qualified,
                    entry.allow_empty,
                    consequence=", and the publish is one unit, so nothing else is written either",
                )
                require_landable_columns(
                    frame,
                    loader.declared_column_types(connection, config["schema"], entry.table),
                    entry.qualified,
                    loader.numeric_declarations(connection, config["schema"], entry.table),
                )
                if entry.mode == "merge":
                    require_one_row_per_key(frame, entry.key, entry.qualified)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Write every table inside one transaction, and return one receipt row.

        The stamp is read once, before the transaction. A refusal from any table is raised inside the unit of
        work, so the loader's undo leaves the destination as the previous publish left it, on both drivers.
        """
        entries = _entries(config)
        stamp = _stamp_for(config)
        uri = resolve_connection_uri(config["connection"], "connection")
        with open_connection(uri) as connection:
            loader = driver_for(connection)
            with loader.transaction(connection):
                for entry in entries:
                    _publish_one(loader, connection, config["schema"], entry, inputs[entry.port], stamp)
        return Ports({"receipt": _receipt(entries, inputs, stamp)})


def _stamp_for(config: Mapping[str, Any]) -> _Stamp | None:
    """This publish's stamp, or None when no stamp column is named. Called once, from execute.

    UTC without a zone: the column is a plain timestamp on both engines, and figures read later elsewhere
    must still say when they were built.
    """
    column = config["stamp_column"]
    if not column:
        return None
    return _Stamp(column=column, at=datetime.now(UTC).replace(tzinfo=None))


def _entries(config: Mapping[str, Any]) -> tuple[_Entry, ...]:
    """The table entries as `_Entry` values. Defaults inside list elements are not filled, so they are applied here."""
    schema = config["schema"]
    return tuple(
        _Entry(
            port=entry["port"],
            table=entry["table"],
            qualified=qualified_name(schema, entry["table"]),
            mode=entry["mode"],
            key=tuple(entry.get("keys") or ()),
            allow_empty=bool(entry.get("allow_empty", False)),
            when_incoming_is_null=entry.get("when_incoming_is_null"),
        )
        for entry in config["tables"]
    )


def _refuse_a_stamp_named_after_a_table(stamp: str | None, entries: Sequence[_Entry]) -> None:
    """Refuse a stamp column named like a table being published.

    The receipt has one column per table plus the stamp, so a shared name would put a row count where the
    time belongs.
    """
    if stamp is None:
        return
    for entry in entries:
        if entry.table == stamp:
            raise ConfigError(
                f"the stamp column {stamp!r} is also the name of a table this publish writes; the "
                f"receipt carries one column per table and one for the stamp, so the two would "
                f"land in one column and the instant would be lost"
            )


def _refuse_repeats(names: Iterable[str], what: str = "port") -> None:
    """Refuse a name two entries claim, naming it."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ConfigError(
                f"two entries claim the same {what} {name!r}; a publish writes each {what} once, "
                f"and the second of two would silently replace the first"
            )
        seen.add(name)


def _require_a_table_these_rows_can_be_published_to(
    connection: Any,
    entry: _Entry,
    frame: pl.DataFrame,
    stamp: str | None,
) -> None:
    """Check the table exists, its columns match, and the merge key is among them.

    The key is checked against the table's own column names, because a merge puts those names into its
    ON CONFLICT clause.
    """
    table_columns = require_matching_table(connection, frame, entry.qualified, stamp)
    missing = [column for column in entry.key if column not in table_columns]
    if missing:
        raise ConfigError(
            f"the merge key for table {entry.qualified!r} names column(s) nobody carries: "
            f"{', '.join(missing)}; the table has: {', '.join(table_columns)}"
        )


def _publish_one(
    loader: RowLoader,
    connection: Any,
    schema: str | None,
    entry: _Entry,
    frame: pl.DataFrame,
    stamp: _Stamp | None,
) -> None:
    """Load one table inside the publish's transaction, refused by name if the engine refuses it.

    The stamp is added to the row tuples, not to the frame, which belongs to the caller.
    """
    columns = [*frame.columns, stamp.column] if stamp else list(frame.columns)
    landing = landing_frame(frame, loader.numeric_declarations(connection, schema, entry.table))
    rows = [(*row, stamp.at) for row in landing.rows()] if stamp else landing.rows()

    try:
        if entry.mode == "replace":
            connection.execute(f"DELETE FROM {entry.qualified}")
            loader.load_rows(connection, entry.qualified, columns, rows)
        elif entry.mode == "append":
            loader.load_rows(connection, entry.qualified, columns, rows)
        else:
            keep_published = entry.when_incoming_is_null == "keep_published"
            loader.merge_rows(connection, schema, entry.table, columns, entry.key, rows, keep_published=keep_published)
    except loader.driver_errors as error:
        raise DataError(
            f"publishing table {entry.qualified!r} was refused: {error}; the publish is one unit, "
            f"so none of its tables was written. Which rows offended is not worked out here",
            offending=None,
            total=frame.height,
        ) from error


def _receipt(entries: Sequence[_Entry], inputs: Ports, stamp: _Stamp | None) -> pl.DataFrame:
    """One row: the publish time, and the number of rows written to each table."""
    counts: dict[str, list[Any]] = {stamp.column: [stamp.at]} if stamp else {}
    for entry in entries:
        counts[entry.table] = [inputs[entry.port].height]
    return pl.DataFrame(counts)


def _seed_example(connection: sqlite3.Connection) -> None:
    """The tables the examples publish into."""
    connection.execute("create table totals (period text primary key, amount real)")
    connection.execute("create table movements (period text, change real)")
    connection.execute("create table parties (party text primary key, label text, amount real)")
    connection.executemany("insert into totals values (?, ?)", [("2026-01", 100.0)])
    connection.executemany("insert into movements values (?, ?)", [("2026-01", 1.0)])
    connection.executemany("insert into parties values (?, ?, ?)", [("a", "Alpha", 10.0)])


def _example_database() -> AbstractContextManager[None]:
    """The in-memory database the example runs against, named by its connection variable for the run."""
    return example_database(_EXAMPLE_URI, _EXAMPLE_CONNECTION, _seed_example)


def _together_example() -> WorkedExample:
    inputs = Ports(
        {
            "totals": pl.DataFrame({"period": ["2026-02", "2026-03"], "amount": [200.0, 300.0]}),
            "log": pl.DataFrame({"period": ["2026-02"], "change": [100.0]}),
        }
    )
    outputs = Ports({"receipt": pl.DataFrame({"totals": [2], "movements": [1]})})
    return WorkedExample(
        scenario="replace one table and add to another, together",
        configuration={
            "connection": _EXAMPLE_CONNECTION,
            "tables": [
                {"port": "totals", "table": "totals", "mode": "replace"},
                {"port": "log", "table": "movements", "mode": "append"},
            ],
        },
        explanation="'totals' is emptied and reloaded with the two rows on its port. 'movements' keeps its row "
        "and gains the one on 'log'. Both happen in one transaction, so a failure on either leaves "
        "both as they were. The receipt counts the rows written to each table.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


def _merge_example() -> WorkedExample:
    inputs = Ports({"parties": pl.DataFrame({"party": ["a", "c"], "label": [None, "Gamma"], "amount": [11.0, 30.0]})})
    outputs = Ports({"receipt": pl.DataFrame({"parties": [2]})})
    return WorkedExample(
        scenario="merge new rows into a table, keeping published values where the new row is blank",
        configuration={
            "connection": _EXAMPLE_CONNECTION,
            "tables": [
                {
                    "port": "parties",
                    "table": "parties",
                    "mode": "merge",
                    "keys": ["party"],
                    "when_incoming_is_null": "keep_published",
                }
            ],
        },
        explanation="Party 'a' is published as ('Alpha', 10.0). Tonight's row has a new amount and no label, so "
        "it becomes ('Alpha', 11.0): 'keep_published' keeps the label. With 'overwrite' the label "
        "would become null. Party 'c' is new and is inserted. The receipt counts the two incoming "
        "rows.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


REGISTRY.register(PublishTables())
