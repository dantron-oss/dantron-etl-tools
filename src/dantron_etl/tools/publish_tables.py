"""PublishTables: a set of related tables, committed together or not at all.

``write_table`` writes one table, and one step is one transaction. A set of tables that are only
meaningful beside each other -- rows, a series over periods, an audit, an append-only change log, a
dimension keyed per party -- written as one step each is that many transactions, so a build that
falls over half way through leaves the destination holding three tables from tonight and two from
last night. Published, agreeing with nothing, and with no record of which is which. Nothing in the
flow contract passes scratch space between steps, so there is nowhere outside a single step for that
transaction to live. That is the whole reason this tool exists and the property every ruling below
is arranged around: **all of the tables, or none of them**.

**One instant, taken once.** The stamp belongs to the publish rather than to the data or to any one
table, so it is read once and written identically onto every table. Reached for inside the loop it
would give each table a value milliseconds from its neighbour's, and the tables of one build could
no longer be selected as one build; taken from a constant or off a frame it would give every build
the same answer, and "which build produced this row" would have none. The frames never carry it.

**A merge keeps what the incoming row is silent about.** ``coalesce(excluded.c, target.c)``, not
``= excluded.c``. A refresh that carries a value overwrites it and a refresh that fills a gap fills
it, but a null the build has nothing to say about leaves the published value standing -- because the
alternative is that one unreadable upstream source publishes a dimension of nulls over every row
that already held a figure. A merge stages its rows first, in a temporary table named
``_etl_merge_staging``: the bulk path on one of the two drivers is ``COPY``, which carries no
conflict resolution of its own, so the rows are loaded in bulk and upserted out of the scratch in
one statement. One fixed name, created and dropped inside the publish -- a destination that holds a
permanent table of that name has it shadowed for as long as a merge is running.

**Append never truncates.** A change log is the one table a rebuild cannot recreate, so the mode
that adds to it is a mode of its own and ``mode`` carries no default: the three differ in whether
running the flow twice is safe, and any default would choose one of them for an author who meant
another.

**This tool never creates a table.** No field turns that on. A table created from whatever a frame
happened to infer tonight has types decided by one day's data, no key and no not-null -- the very
constraints that make truncate-and-load safe -- and a typo in ``table`` would quietly create a
second table and leave the real one going stale. DDL belongs in a migration somebody reviewed.

**A column that cannot land in its table's column is refused rather than written.** One of the two
engines has no opinion about this at all: sqlite stores the text ``'heavy'`` in a column declared
``integer`` and reports success. So the frames are read against the destination's own declared types
before anything is emptied, and the refusal names the column.

Everything answerable from the configuration alone is refused in :meth:`PublishTables.validate_config`
and everything needing the frames and the server in :meth:`PublishTables.validate_inputs`, both of
which run before ``execute``. That split is not tidiness: a refusal that arrives after four tables
have been emptied is not a refusal, it is a failed publish with a helpful message.
"""

import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import polars as pl

from ..db import (
    RowLoader,
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
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    temporary_environment,
)

#: Spelled so a definition file hands each of them over as the word itself.
_MODE_ENUM: Final[tuple[str, ...]] = ("replace", "append", "merge")

_EXAMPLE_CONNECTION: Final[str] = "TARGET_DATABASE_URL"
_EXAMPLE_URI: Final[str] = "file:publish_tables_example?mode=memory&cache=shared"


@dataclass(frozen=True, slots=True)
class _Entry:
    """One table of a publish, as every question asked about it afterwards is asked in.

    ``qualified`` is settled here, once, rather than recomposed at each of the four places that
    needs it -- ``schema``, ``table`` and the name they compose into are three strings no type can
    tell apart, and recomposing is the shape in which two of them get handed over the wrong way
    round.
    """

    port: str
    table: str
    qualified: str
    mode: str
    key: tuple[str, ...]
    allow_empty: bool


@dataclass(frozen=True, slots=True)
class _Stamp:
    """The instant this publish carries, and the column every one of its tables carries it in.

    One value rather than two optionals travelling beside each other through the write and the
    receipt. A column with no instant stamps every table null and an instant with no column has
    nowhere to land; neither is a state a publish is ever in, so neither is a state this can hold.
    """

    column: str
    at: datetime


class PublishTables:
    """Write every named frame to its own table, in one transaction, and hand back one receipt."""

    name = "publish_tables"
    description = "Write one frame per named port to its own table, all of them in one transaction."
    category = "output"
    # Empty on purpose: the ports come from 'tables', through input_ports_for below. A tool answers
    # that question one way or the other, never both.
    input_ports = ()
    output_ports = ("receipt",)

    def input_ports_for(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        """One port per table entry, in the order the entries were written.

        The configuration handed over here is the one a person wrote. No blank in it has been filled
        in yet, and it has not been checked against the config template. So anything malformed is
        passed over rather than refused here. The config check runs a moment later and says what is
        wrong, in words about the field that is wrong.
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
                description="The name of an environment variable holding the connection URI; never the URI itself.",
                required=True,
                non_empty=True,
            ),
            "schema": ConfigField(
                type="string",
                description="The schema every table sits in; left out, the tables are named unqualified.",
                required=False,
                default=None,
            ),
            "stamp": ConfigField(
                type="string",
                description="A column every one of these tables carries, written with this "
                "publish's own instant -- one value, onto every table. Left out, nothing is "
                "stamped and the frames alone say what lands.",
                required=False,
                default=None,
            ),
            "tables": ConfigField(
                type="array",
                description="One entry per table: which input port it reads, which table it is "
                "written to, and how. These entries are also what say which ports this step reads.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One table: the port the rows arrive on, and how they are written.",
                    fields={
                        "port": ConfigField(
                            type="string",
                            description="The input port carrying the rows for this table.",
                            required=True,
                            non_empty=True,
                        ),
                        "table": ConfigField(
                            type="string",
                            description="The table to write; a plain identifier, and one this tool never creates.",
                            required=True,
                            non_empty=True,
                        ),
                        "mode": ConfigField(
                            type="string",
                            description="'replace' empties the table before loading; 'append' adds "
                            "without emptying it; 'merge' upserts on 'key', keeping any published "
                            "value the incoming row carries a null for. Required, with no default.",
                            required=True,
                            enum=_MODE_ENUM,
                        ),
                        "key": ConfigField(
                            type="array",
                            description="The columns a merge matches a published row on; required "
                            "by 'merge' and read by no other mode.",
                            non_empty=True,
                            items=ConfigField(
                                type="string",
                                description="One column of the merge key.",
                                non_empty=True,
                            ),
                        ),
                        "allow_empty": ConfigField(
                            type="boolean",
                            description="Whether a frame with no rows is written for this table "
                            "instead of stopping the publish. Left out, it is false.",
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Each entry in 'tables' reads one input port and writes one table: 'replace' "
                "empties it first, 'append' adds to what it already holds, 'merge' upserts on "
                "'key'. Every table is written inside one transaction, so a publish that fails on "
                "any of them leaves all of them exactly as the last publish left them. The tables "
                "must already exist -- this tool runs no DDL of its own -- and each frame's "
                "columns must be exactly its table's, apart from the 'stamp' column, which the "
                "frames never carry and this tool writes. The one output port is a receipt: the "
                "instant this publish carried, and the number of rows written to each table."
            ),
            examples=(_together_example(), _merge_example()),
            notes=(
                "The tables commit together or not at all. That is the whole reason this tool "
                "exists rather than a chain of write_table steps, which is one transaction each.",
                "The stamp is one instant, read once and written identically onto every table, so "
                "the tables of one build can be selected afterwards as one build.",
                "A merge writes coalesce(incoming, published): a value overwrites and a gap is "
                "filled, but a null the build has nothing to say about leaves the published value "
                "standing rather than blanking it.",
                "'mode' has no default: the three differ in whether re-running the flow is safe, "
                "and an append-only log is the one table a rebuild must never rewrite.",
                "An empty frame stops the whole publish -- not merely its own table -- unless that "
                "entry says in writing that it may be empty.",
            ),
            field_help={
                "connection": "The name of an environment variable, so a definition file that is "
                "committed and handed on never holds a credential.",
                "stamp": "Optional; a column every table named here carries, one no frame may, and "
                "one no table here may be named after -- the receipt gives it a column of its own "
                "beside one per table.",
                "tables": "Also what says which input ports this step reads: one port per entry, "
                "in the order the entries are written.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse everything answerable without the frames, before any connection opens.

        A repeated port and a repeated table are both silent failures rather than loud ones, which
        is why each is refused here rather than left to be noticed. A port carries one frame, so two
        entries naming it publish the same rows twice under two names; two entries naming one table
        run two loads against it in the same transaction, and whichever runs second is the only one
        that survives. A stamp sharing a name with one of the tables is the third of the same kind:
        the receipt carries one column per table and one for the stamp, so the two would land in one
        column and the publish would report a row count where a reader expects the instant.
        """
        if config["schema"] is not None:
            validate_identifier(config["schema"], "schema")

        entries = _entries(config)
        for entry in entries:
            validate_identifier(entry.table, "tables[].table")
            if entry.mode == "merge" and not entry.key:
                raise ConfigError(
                    f"the entry publishing table {entry.qualified!r} is a merge with no 'key'; a "
                    f"merge matches a published row on the key, so there is nothing it could do "
                    f"without one"
                )

        _refuse_repeats(entry.port for entry in entries)
        _refuse_repeats((entry.qualified for entry in entries), what="table")
        _refuse_a_stamp_named_after_a_table(config["stamp"], entries)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a table that is not there, columns that do not line up, a key nobody carries, a
        column that cannot land, and an unwelcome empty frame -- all before anything is written.

        Two passes over the entries, and the order between them is the ruling. Everything in the
        first pass is a mistake in the configuration and is answered for every table before any of
        the rows are looked at, so an author is told which entry is wrong rather than which frame
        happened to be read first. The second pass is about the rows themselves.

        Both passes run here, ahead of ``execute``, on one connection: probing a table's columns is
        the same query that proves the table exists, and reading what it declares each column as is
        three more. That is a handful of statements against the catalog, spent before the
        destination is touched at all, against a publish that would otherwise empty four tables on
        the strength of a configuration already known to be wrong.
        """
        entries = _entries(config)
        stamp = config["stamp"]
        uri = resolve_connection_uri(config["connection"], "connection")
        with open_connection(uri) as connection:
            for entry in entries:
                _require_a_table_these_rows_can_be_published_to(connection, entry, inputs[entry.port], stamp)

            loader = driver_for(connection)
            for entry in entries:
                frame = inputs[entry.port]
                if frame.height == 0 and not entry.allow_empty:
                    raise DataError(
                        f"publish_tables refuses an empty frame for table {entry.qualified!r}, and "
                        f"the publish is one unit, so nothing else is written either; set "
                        f"allow_empty on that entry to accept it",
                        offending=frame,
                        total=0,
                    )
                _require_landable_columns(
                    frame,
                    loader.declared_column_types(connection, config["schema"], entry.table),
                    entry.qualified,
                )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Write every table inside one transaction, and hand back one row of accounting for it.

        The instant is read once, here, before the transaction opens, and the same value goes onto
        every table and onto the receipt.

        A refusal from any table is raised from inside the unit of work, so the loader's own undo
        runs on the way out and the destination is left exactly as the previous publish left it --
        on both drivers, which disagree about what a unit of work even is. The refusal names the
        table whose load was refused and carries the engine's own words about it; which rows
        offended is ``write_table``'s question, asked of one table at a time on a connection that is
        not holding a publish open.
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
    """This publish's stamp, read here and nowhere else, or ``None`` when no column is named for it.

    The one call is from ``execute``, before the transaction opens, and the value it returns is what
    every table and the receipt are written with. Read per table instead, each would carry a value
    milliseconds from its neighbour's and the tables of one build could no longer be selected as one
    build.

    The instant is naive rather than aware, because the column it lands in is a plain timestamp on
    both engines and a value carrying an offset is written through two different conversions to get
    there. UTC rather than local, because a set of figures read a year later in another place still
    has to say when it was built.
    """
    column = config["stamp"]
    if not column:
        return None
    return _Stamp(column=column, at=datetime.now(UTC).replace(tzinfo=None))


def _entries(config: Mapping[str, Any]) -> tuple[_Entry, ...]:
    """The configuration's table entries, read once into the shape the rest of this module asks.

    ``key`` and ``allow_empty`` are read with ``.get`` because a default declared on a member of an
    element is never filled in: an element nobody wrote has nothing to fill, so the value they take
    when they are absent is stated here rather than in the template, where the next reader would
    trust it.
    """
    schema = config["schema"]
    return tuple(
        _Entry(
            port=entry["port"],
            table=entry["table"],
            qualified=qualified_name(schema, entry["table"]),
            mode=entry["mode"],
            key=tuple(entry.get("key") or ()),
            allow_empty=bool(entry.get("allow_empty", False)),
        )
        for entry in config["tables"]
    )


def _refuse_a_stamp_named_after_a_table(stamp: str | None, entries: Sequence[_Entry]) -> None:
    """Refuse a stamp column answering to the name of one of the tables being published.

    The receipt is one row carrying the instant under the stamp's name and a count under each
    table's, so a name shared between them is a receipt one column short: the count is written over
    the instant and the publish reports a row count where a reader looks for when it was built.

    Most of the time the column comparison would refuse this first, for its own reasons -- the stamp
    is added to the frame's side there, so a stamp named after a table usually makes some table's
    columns disagree. Not always: a table named after the stamp column *and* carrying a column of
    that name lines up perfectly, publishes correctly, and hands back the receipt with the instant
    missing. That is the case this rule is here for, and it is why the rule is about the name rather
    than a by-product of a rule about columns.
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
    """Prove the table is there, that its columns line up, and that its merge key is one of them.

    Probing the columns is the same statement that proves the table exists, so one round trip
    answers both; the key is checked against the table rather than against the frame because a merge
    splices those names into its ``ON CONFLICT``, and a name that came back from the server's own
    catalog is a name the statement can safely carry.
    """
    table_columns = probe_columns(connection, entry.qualified)
    _require_matching_columns(frame, table_columns, stamp, entry.qualified)
    missing = [column for column in entry.key if column not in table_columns]
    if missing:
        raise ConfigError(
            f"the merge key for table {entry.qualified!r} names column(s) nobody carries: "
            f"{', '.join(missing)}; the table has: {', '.join(table_columns)}"
        )


def _require_matching_columns(
    frame: pl.DataFrame,
    table_columns: Sequence[str],
    stamp: str | None,
    qualified: str,
) -> None:
    """Refuse rows whose columns, plus the stamp, are not exactly the table's, case-sensitively.

    The stamp is added to the frame's side rather than removed from the table's, which is the same
    comparison read the way it is meant: the frames never carry it, every one of these tables does,
    and a table that does not carry it is refused here rather than at load time.
    """
    carried = set(frame.columns) | ({stamp} if stamp else set())
    expected = set(table_columns)
    if carried == expected:
        return

    extra = sorted(carried - expected)
    missing = sorted(expected - carried)
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


def _require_landable_columns(
    frame: pl.DataFrame,
    declared: Mapping[str, str | None],
    qualified: str,
) -> None:
    """Refuse a column whose values will not go into the column the table declared, naming it.

    Worth doing here rather than leaving to the engine because neither engine does it usefully. One
    says nothing at all -- sqlite stores ``'heavy'`` in an ``integer`` column and reports success --
    and the other says it at load time, in its own words, about a copy stream: by which point the
    publish has already emptied the tables ahead of this one, and what a reader is told is about a
    stream rather than about a column they can go and look at.

    A type this library does not recognise is passed over rather than refused. A domain, an enum, a
    JSON column: each is a column something else has an opinion about, and refusing on a reading
    this side did not understand would stop a publish that was perfectly correct.
    """
    for column, dtype in frame.schema.items():
        wanted = declared.get(column)
        if wanted is None or _fits(dtype, wanted):
            continue
        raise DataError(
            f"column {column!r} carries {dtype} where table {qualified!r} declares its column as "
            f"{wanted}; nothing is published",
            offending=frame,
            total=frame.height,
        )


#: The base dtype each remaining declared type is, for the types that admit no widening at all.
_DECLARED_BASE: Final[Mapping[str, Any]] = {
    "String": pl.String,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
    "Time": pl.Time,
}


def _fits(dtype: pl.DataType, declared: str) -> bool:
    """Whether a frame column of ``dtype`` can land in a column declared ``declared``.

    A column of nothing but nulls fits anywhere: it carries no value to be wrong about. A whole
    number lands in a decimal column, because widening loses nothing; a decimal does not land in a
    whole-number column, because narrowing does, and quietly. Everything else has to be the type it
    says it is.
    """
    if dtype == pl.Null:
        return True
    if declared == "Int64":
        return dtype.is_integer()
    if declared == "Float64":
        return dtype.is_numeric()
    base = _DECLARED_BASE.get(declared)
    # A declared type this module has no entry for is passed over, exactly as an unrecognised SQL
    # type is. The two vocabularies live in different modules and the reading side is a safety net:
    # one growing a name the other has not learned should cost the check, never the publish.
    return base is None or dtype.base_type() == base


def _publish_one(
    loader: RowLoader,
    connection: Any,
    schema: str | None,
    entry: _Entry,
    frame: pl.DataFrame,
    stamp: _Stamp | None,
) -> None:
    """One table's load, inside the publish's transaction, refused by name if the engine refuses it.

    The stamp is added to the rows here rather than to the frame, because the frame is the caller's
    and a tool may not write to its own input -- and because adding a column to a frame to take it
    straight back apart into tuples is a copy of the whole thing for nothing.
    """
    columns = [*frame.columns, stamp.column] if stamp else list(frame.columns)
    rows = [(*row, stamp.at) for row in frame.rows()] if stamp else frame.rows()

    try:
        if entry.mode == "replace":
            connection.execute(f"DELETE FROM {entry.qualified}")
            loader.load_rows(connection, entry.qualified, columns, rows)
        elif entry.mode == "append":
            loader.load_rows(connection, entry.qualified, columns, rows)
        else:
            loader.merge_rows(connection, schema, entry.table, columns, entry.key, rows)
    except loader.driver_errors as error:
        raise DataError(
            f"publishing table {entry.qualified!r} was refused: {error}; the publish is one unit, "
            f"so none of its tables was written. Which rows offended is not worked out here -- "
            f"naming them means reading the table on a connection that is not holding a publish "
            f"open, which is write_table's job over a single table -- so the count that follows is "
            f"no rows examined rather than no rows at fault",
            offending=frame.head(0),
            total=frame.height,
        ) from error


def _receipt(entries: Sequence[_Entry], inputs: Ports, stamp: _Stamp | None) -> pl.DataFrame:
    """One row: the instant this publish carried, and the rows written to each table.

    One row rather than one port per table. Five pass-through ports would be five ports nobody
    wires, where a count per table is what a reader checking tonight's build against last night's
    actually reads.
    """
    counts: dict[str, list[Any]] = {stamp.column: [stamp.at]} if stamp else {}
    for entry in entries:
        counts[entry.table] = [inputs[entry.port].height]
    return pl.DataFrame(counts)


@contextmanager
def _example_database() -> Iterator[None]:
    """An in-memory database the examples publish into, entered around each example and torn down after.

    Built here rather than inside ``help()``, for the same reason as every sink's: the catalog page
    generator calls ``help()`` on every tool, and a description that opened a connection or exported
    a variable as a side effect of describing itself is exactly the disturbance ``check_tool``
    refuses.
    """
    anchor = sqlite3.connect(_EXAMPLE_URI, uri=True)
    try:
        anchor.execute("create table totals (period text primary key, amount real)")
        anchor.execute("create table movements (period text, change real)")
        anchor.execute("create table parties (party text primary key, label text, amount real)")
        anchor.executemany("insert into totals values (?, ?)", [("2026-01", 100.0)])
        anchor.executemany("insert into movements values (?, ?)", [("2026-01", 1.0)])
        anchor.executemany("insert into parties values (?, ?, ?)", [("a", "Alpha", 10.0)])
        anchor.commit()
        with temporary_environment(**{_EXAMPLE_CONNECTION: _EXAMPLE_URI}):
            yield
    finally:
        anchor.close()


def _together_example() -> WorkedExample:
    inputs = Ports(
        {
            "totals": pl.DataFrame({"period": ["2026-02", "2026-03"], "amount": [200.0, 300.0]}),
            "log": pl.DataFrame({"period": ["2026-02"], "change": [100.0]}),
        }
    )
    outputs = Ports({"receipt": pl.DataFrame({"totals": [2], "movements": [1]})})
    return WorkedExample(
        scenario="two related tables are replaced and appended to in one transaction",
        configuration={
            "connection": _EXAMPLE_CONNECTION,
            "tables": [
                {"port": "totals", "table": "totals", "mode": "replace"},
                {"port": "log", "table": "movements", "mode": "append"},
            ],
        },
        explanation="'totals' is emptied and reloaded with the two rows on its port; 'movements' "
        "keeps the row it already held and gains the one on 'log'. Both happen inside one "
        "transaction, so a failure on either would leave both exactly as they were. The receipt is "
        "one row carrying the number of rows written to each table.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


def _merge_example() -> WorkedExample:
    inputs = Ports({"parties": pl.DataFrame({"party": ["a", "c"], "label": [None, "Gamma"], "amount": [11.0, 30.0]})})
    outputs = Ports({"receipt": pl.DataFrame({"parties": [2]})})
    return WorkedExample(
        scenario="a merge fills a gap, overwrites a value, and leaves what it is silent about standing",
        configuration={
            "connection": _EXAMPLE_CONNECTION,
            "tables": [{"port": "parties", "table": "parties", "mode": "merge", "key": ["party"]}],
        },
        explanation="Party 'a' is already published as ('Alpha', 10.0). Tonight's row carries a new "
        "amount and no label, so afterwards it reads ('Alpha', 11.0): the amount moves and the "
        "published label survives, because a null says the build had nothing to report rather than "
        "that the label is now empty. Party 'c' is new and is inserted. The receipt counts the two "
        "incoming rows.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
        setup=_example_database,
    )


REGISTRY.register(PublishTables())
