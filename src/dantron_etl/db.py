"""Connection plumbing shared by every source and sink: an environment name in, an open connection out.

A pipeline definition is a file that gets committed and handed to another client as a template, so
it must never hold a credential. What it holds instead is the *name* of an environment variable, and
resolving that name to a real connection -- reading the environment, never a file; picking the
driver by scheme; refusing a name that could carry SQL before a connection ever opens -- is the same
four steps for a source reading a table and a sink writing one. Written once here so a fifth tool
reaching for a database inherits the same refusals rather than reinventing a slightly different set
of them.

**The environment, and never a file.** ``os.environ`` is the only place a value is read from. In an
orchestrator or in CI there is no ``.env`` to parse, the secret arrives from a secret store, and a
tool that parsed one anyway would bake a single project's layout into a library that ships to every
other consumer of it.

**A name is the whole injection fence.** ``schema`` and ``table`` are spliced into a statement and
cannot be bound as parameters, so the only defence available is refusing anything that is not a
plain identifier -- cheap to check, and the check runs before a connection opens, so a name that
could carry a second statement never gets the chance to run one.

**No driver is declared.** SQLite needs nothing beyond the standard library; anything else imports
its driver lazily, by scheme, and a missing one is refused by naming the package to install rather
than surfacing as a bare ``ImportError`` a caller has never heard of.
"""

import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

from .errors import ConfigError

__all__ = [
    "RowLoader",
    "TableRules",
    "declared_type_of",
    "driver_for",
    "open_connection",
    "probe_columns",
    "qualified_name",
    "quoted_identifier",
    "require_matching_columns",
    "resolve_connection_uri",
    "validate_identifier",
]

# Matched with fullmatch: '$' alone would also accept a name with a newline at its end.
_ENV_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_SQLITE_PREFIX: Final[str] = "sqlite:///"
_KNOWN_SCHEMES: Final[tuple[str, ...]] = ("sqlite", "file", "postgresql", "postgres")
_POSTGRES_SCHEMES: Final[frozenset[str]] = frozenset({"postgresql", "postgres"})

# The word that makes a SQLite table definition unreadable to anything short of a SQL parser. The
# pragmas expose keys, nullability and unique indexes; they say nothing at all about a CHECK, which
# lives only in the definition's own text.
_CHECK_WORD: Final[re.Pattern[str]] = re.compile(r"\bcheck\b", re.IGNORECASE)

# How many values one SQLite statement may carry. The real ceiling is higher on every version this
# runs on; staying well under it costs one extra round trip per half-thousand distinct values.
_SQLITE_PARAMETER_CHUNK: Final[int] = 500

# Where the two pragmas that carry a rule keep it. ``table_info`` hands back one row per column,
# ``(cid, name, type, notnull, dflt_value, pk)``, and ``index_info`` one row per indexed column,
# ``(seqno, cid, name)``. The last of ``table_info``'s is the one worth naming: ``pk`` is not a flag
# but the column's position in the primary key, counting from one and zero for a column outside it,
# so it answers "is this column in the key" and "does the key span columns" at once.
_TABLE_INFO_NAME: Final[int] = 1
_TABLE_INFO_DECLARED_TYPE: Final[int] = 2
_TABLE_INFO_NOT_NULL: Final[int] = 3
_TABLE_INFO_KEY_POSITION: Final[int] = 5
_INDEX_INFO_NAME: Final[int] = 2

# The scratch table a merge loads into before it upserts. One fixed name rather than one per merge,
# dropped again as soon as the upsert has read it, so two merges in the same unit of work do not
# collide. Spelled unlike anything a schema would hold, because on sqlite a temporary table shadows
# a permanent one of the same name and the merge would then read its own scratch as the target.
_MERGE_STAGING: Final[str] = "_etl_merge_staging"

# Every SQL type name this library is willing to claim it understands, as the declared type it
# means. Both drivers are read into this one vocabulary -- sqlite hands back the word from the
# definition, postgres hands back ``format_type``'s spelling -- so a caller comparing a frame
# against a table gets the same answer whichever engine the table lives on. A name that is not here
# is answered with ``None``, which means "no opinion" rather than "wrong": a domain, an enum, a
# JSON column and a type this list has never heard of are all passed over, because any doubt buys
# the slow answer and never the wrong one.
_SQL_TYPE_NAMES: Final[dict[str, str]] = {
    "int": "Int64",
    "int2": "Int64",
    "int4": "Int64",
    "int8": "Int64",
    "integer": "Int64",
    "smallint": "Int64",
    "bigint": "Int64",
    "tinyint": "Int64",
    "mediumint": "Int64",
    "float": "Float64",
    "float4": "Float64",
    "float8": "Float64",
    "real": "Float64",
    "double": "Float64",
    "double precision": "Float64",
    "numeric": "Float64",
    "decimal": "Float64",
    "bpchar": "String",
    "char": "String",
    "character": "String",
    "character varying": "String",
    "clob": "String",
    "nvarchar": "String",
    "string": "String",
    "text": "String",
    "varchar": "String",
    "bool": "Boolean",
    "boolean": "Boolean",
    "date": "Date",
    "datetime": "Datetime",
    "timestamp": "Datetime",
    "timestamptz": "Datetime",
    "time": "Time",
    "timetz": "Time",
}


def _register_adapters() -> None:
    """Bind ``date``/``datetime`` to ISO text ourselves, ahead of the deprecation.

    sqlite3's own default adapters for these two types are deprecated in 3.12 and gone in 3.14, so
    a column of ``pl.Date`` bound straight through would warn today and raise on a later Python.
    Registering our own keeps every write on the safe side of that regardless of interpreter.
    """
    sqlite3.register_adapter(date, lambda value: value.isoformat())
    sqlite3.register_adapter(datetime, lambda value: value.isoformat(sep=" "))


_register_adapters()


def validate_identifier(value: str, field: str) -> None:
    """Refuse anything that is not a plain name, quoting the offender.

    Called on ``schema`` and ``table`` before a connection is ever opened: an identifier cannot be
    bound as a parameter, so this grammar check is the only defence against a name carrying a second
    statement, and it has to run first for that defence to mean anything.
    """
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ConfigError(
            f"config field {field!r} must be a plain identifier (letters, digits and underscores, "
            f"not starting with a digit); refusing {value!r}"
        )


def qualified_name(schema: str | None, table: str) -> str:
    """The one name ``schema`` and ``table`` compose into, the only form either is ever handed to a database."""
    return f"{schema}.{table}" if schema else table


def resolve_connection_uri(name: str, field: str) -> str:
    """The connection string named by the environment variable ``name``, read from ``os.environ`` alone.

    Two refusals, and neither repeats a value that might be a credential. A ``name`` that is not
    itself a plausible environment variable -- because it is the connection string that was meant to
    sit in the environment, not the field -- is refused without echoing it. A variable that is a
    plausible name but is not set is refused by that name, which is safe to repeat because it is
    never the secret itself.
    """
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise ConfigError(
            f"config field {field!r} must name an environment variable holding the connection, not "
            f"the connection string itself"
        )
    value = os.environ.get(name)
    if value is None:
        raise ConfigError(f"environment variable {name!r} is not set")
    return value


@contextmanager
def open_connection(uri: str) -> Iterator[Any]:
    """A DB-API connection for ``uri``, closed on the way out whether the block succeeds or raises."""
    connection = _connect(uri)
    try:
        yield connection
    finally:
        connection.close()


def _connect(uri: str) -> Any:
    try:
        scheme = urlparse(uri).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme == "sqlite":
        return _connect_sqlite_path(uri)
    if scheme == "file":
        return _connect_sqlite_file_uri(uri)
    if scheme in _POSTGRES_SCHEMES:
        return _connect_postgres(uri)
    known = ", ".join(_KNOWN_SCHEMES)
    if scheme:
        raise ConfigError(f"connection scheme {scheme!r} is not recognised; recognised schemes are: {known}")
    raise ConfigError(f"connection has no recognisable scheme; recognised schemes are: {known}")


def _connect_sqlite_path(uri: str) -> sqlite3.Connection:
    """A file-backed sqlite connection, refusing a path that is not there rather than creating one.

    ``sqlite3.connect`` would happily create a missing path, and a read or a write against the file
    it just created would report success having touched nothing the author meant. So existence is
    checked here, before ``connect`` is ever called.
    """
    if not uri.startswith(_SQLITE_PREFIX):
        raise ConfigError(f"a sqlite connection must be written as {_SQLITE_PREFIX}<path>")
    path = Path(uri[len(_SQLITE_PREFIX) :])
    if not path.exists():
        raise ConfigError(f"sqlite database '{path}' does not exist; a missing path is refused, not created")
    return sqlite3.connect(path)


def _connect_sqlite_file_uri(uri: str) -> sqlite3.Connection:
    """A sqlite ``file:`` URI. One that states no ``mode`` opens an existing file only.

    sqlite would otherwise create a missing file, and a read from it would report a missing table
    while leaving an empty database behind. A URI that states a mode (``mode=memory``,
    ``mode=rwc``) is opened as written.
    """
    if "mode=" not in uri:
        uri = f"{uri}{'&' if '?' in uri else '?'}mode=rw"
    failure: str | None = None
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        failure = f"{type(error).__name__}: {error}"
    raise ConfigError(f"could not open the sqlite database the file: URI names ({failure}); a missing file is not created")


def _could_not_connect(uri: str, failure: str) -> str:
    """A refusal to connect that names where, and never the connection string, the user or the password.

    A driver's own message can quote the connection string, password included, so only the kind of
    failure is kept from it.
    """
    try:
        parsed = urlparse(uri)
        host, database = parsed.hostname, parsed.path.lstrip("/")
        port = parsed.port
    except ValueError:
        host = database = port = None
    where = ", ".join(
        part
        for part in (
            f"database {database!r}" if database else "",
            f"host {host!r}" if host else "",
            f"port {port}" if port else "",
        )
        if part
    )
    return (
        f"could not connect to the postgresql database{f' ({where})' if where else ''}: the driver raised "
        f"{failure}. Its own message is not shown, because it can contain the connection string or the "
        f"password; check the value of the environment variable the step names"
    )


def _connect_postgres(uri: str) -> Any:
    """A postgres connection through whichever driver the consumer installed, imported lazily by scheme.

    The library declares no database driver, so this is the one place that reaches for one, and it
    reaches for it only when a postgres connection is actually asked for. Neither package present is
    refused with the name of what to install, rather than the bare ``ImportError`` that would
    otherwise surface from a library the caller has never heard of.
    """
    try:
        import psycopg
    except ImportError:
        psycopg = None  # type: ignore[assignment]
    connect = psycopg.connect if psycopg is not None else None

    if connect is None:
        try:
            import adbc_driver_postgresql.dbapi as adbc_postgres
        except ImportError:
            adbc_postgres = None  # type: ignore[assignment]
        if adbc_postgres is not None:
            connect = adbc_postgres.connect

    if connect is not None:
        # The refusal is raised after the except block, so the driver's error is not even chained to
        # it: a traceback, a trace or a log cannot show its message.
        failure: str | None = None
        try:
            return connect(uri)
        except Exception as error:  # noqa: BLE001 - any driver failure is reported the same safe way
            failure = type(error).__name__
        raise ConfigError(_could_not_connect(uri, failure))

    raise ConfigError(
        "connection scheme 'postgresql' needs a driver this environment does not have installed; "
        "install 'psycopg' or 'adbc_driver_postgresql' to use a postgresql connection"
    )


def probe_columns(connection: Any, qualified: str) -> tuple[str, ...]:
    """The column names ``qualified`` carries, or a refusal naming it if it is not there.

    A single zero-row query answers both questions at once: it fails exactly when the table does
    not exist, in this library's own words rather than the driver's, and it succeeds with a cursor
    description that names every column without reading a single row of data.
    """
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT * FROM {qualified} LIMIT 0")
        return tuple(entry[0] for entry in cursor.description)
    except Exception as error:  # noqa: BLE001 - every failure is named below, in this library's words
        if _is_missing_table(error):
            raise ConfigError(f"table {qualified!r} was not found") from error
        first_line = str(error).strip().splitlines()[0] if str(error).strip() else ""
        raise ConfigError(
            f"could not read table {qualified!r}: {type(error).__name__}{f': {first_line}' if first_line else ''}"
        ) from error
    finally:
        cursor.close()


def _is_missing_table(error: Exception) -> bool:
    """Whether a driver's error says the table (or its schema) does not exist, and nothing else."""
    if type(error).__name__ in ("UndefinedTable", "InvalidSchemaName"):
        return True
    text = str(error).lower()
    return "no such table" in text or ("does not exist" in text and ("relation" in text or "schema" in text))


def require_matching_columns(
    frame_columns: Sequence[str],
    table_columns: Sequence[str],
    qualified: str,
    *,
    stamp: str | None = None,
) -> None:
    """Refuse rows whose columns, plus ``stamp`` if one is given, are not exactly the table's.

    Compared in both directions and case-sensitively. A column the rows carry and the table lacks is
    refused, and so is a column the table has and the rows lack. The stamp is a column the writer
    adds itself, so it counts on the rows' side.
    """
    carried = set(frame_columns) | ({stamp} if stamp else set())
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
        f"carry: {', '.join(frame_columns)}; the table has: {', '.join(table_columns)}"
    )


def quoted_identifier(name: str) -> str:
    """One column name, spelled so a statement reads it as a name and nothing else.

    The names quoted here come back from the database's own catalog rather than from a definition,
    so the grammar check that guards ``schema`` and ``table`` has nothing to bite on. Quoting is
    what keeps a column whose name is a reserved word, or holds a space, readable -- and it is the
    same spelling on both drivers.
    """
    return '"' + name.replace('"', '""') + '"'


def declared_type_of(sql_type: str) -> str | None:
    """The declared type ``sql_type`` means, or ``None`` when this library has no opinion about it.

    One reading for both engines. The width, the precision and the time zone are dropped before the
    lookup -- ``varchar(20)``, ``numeric(12, 2)`` and ``timestamp without time zone`` each say
    nothing about *which* of the declared types a value has to be, and carrying them would mean a
    table declared one place differently from another was answered differently. An array is not
    reduced to its element type: a column of them holds something no declared type here describes,
    so it is passed over with everything else unrecognised.
    """
    name = sql_type.strip().lower()
    if name.endswith("[]"):
        return None
    name = name.split("(")[0].strip()
    for suffix in (" without time zone", " with time zone"):
        name = name.removesuffix(suffix)
    return _SQL_TYPE_NAMES.get(name.strip())


def _merge_statement(qualified: str, staging: str, columns: Sequence[str], key: Sequence[str]) -> str:
    """Upsert every staged row into ``qualified``, keeping what the incoming row is silent about.

    ``coalesce(excluded.c, {target}.c)`` rather than ``= excluded.c``, and the difference is the
    whole substance of a merge. A refresh that carries a value overwrites; a refresh that carries a
    null has nothing to say about that column and the published value stands. The plain assignment
    behaves identically on three of the four cases and, on the night one upstream source cannot be
    read, publishes a column of nulls over every row that already held a figure.

    ``WHERE true`` is not decoration. sqlite refuses ``INSERT ... SELECT ... ON CONFLICT`` outright
    without a ``WHERE`` on the select -- it reads the ``ON`` as the start of a join -- and postgres
    accepts the same clause, so one statement serves both.
    """
    quoted = [quoted_identifier(column) for column in columns]
    named = ", ".join(quoted)
    conflict = ", ".join(quoted_identifier(column) for column in key)
    updated = [
        f"{quoted_identifier(column)} = coalesce(excluded.{quoted_identifier(column)}, "
        f"{qualified}.{quoted_identifier(column)})"
        for column in columns
        if column not in set(key)
    ]
    resolution = f"UPDATE SET {', '.join(updated)}" if updated else "NOTHING"
    return (
        f"INSERT INTO {qualified} ({named}) SELECT {named} FROM {staging} WHERE true "
        f"ON CONFLICT ({conflict}) DO {resolution}"
    )


@dataclass(frozen=True)
class TableRules:
    """What a table demands of a row, as far as the catalog can be read and the rows can be judged.

    ``beyond_the_frame`` is the field that matters. It is true whenever the table carries a rule the
    rows in hand cannot be judged against -- a CHECK expression, a key whose answer lives in another
    table, a unique constraint spanning columns, an exclusion constraint, a unique index no
    constraint stands behind. It defaults to true, so a reader that half-understands a table says so
    rather than reporting the rules it did recognise as though they were all of them.

    ``settled_by_the_engine`` names the columns in which a null is not a value to be judged, because
    the engine decides what goes there. SQLite fills a null in a rowid-alias key with a number of its
    own choosing, which a later row can then collide with, and accepts a null in any other key more
    than once; neither is visible in the rows as written. A frame carrying a null in one of those
    columns is one this route cannot answer for.
    """

    not_null: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()
    settled_by_the_engine: tuple[str, ...] = ()
    beyond_the_frame: bool = True


class RowLoader:
    """One driver's answers to the four questions a load asks that the DB-API does not settle.

    The DB-API says how a statement is executed and nothing about how a batch of rows is written,
    how a unit of work is opened, how a single row is attempted and undone, or which exception
    types mean "the rows were wrong" rather than "the run went wrong". Every one of those differs
    between the two drivers this library knows, and a sink that answered them by reaching for the
    conveniences of the driver it happened to be written against is a sink that works on exactly
    that driver. So they are answered here, once, beside the connection plumbing, and a sink asks
    for the loader belonging to the connection it was handed rather than assuming one.
    """

    placeholder: str
    integrity_errors: tuple[type[BaseException], ...]
    driver_errors: tuple[type[BaseException], ...]

    def load_rows(
        self, connection: Any, qualified: str, columns: Sequence[str], rows: Sequence[tuple[Any, ...]]
    ) -> None:
        """Write every row of ``rows`` into ``qualified`` without costing a statement per row."""
        raise NotImplementedError

    def insert_row(self, connection: Any, qualified: str, columns: Sequence[str], row: tuple[Any, ...]) -> None:
        """Write one row, so a failed bulk load can be replayed to find which rows offended."""
        connection.execute(self._insert_statement(qualified, columns), row)

    def _insert_statement(self, qualified: str, columns: Sequence[str]) -> str:
        # Every column quoted, as a merge already quotes them: a column named like a reserved word
        # ('order') or holding a space is otherwise a syntax error that reads as bad rows.
        marks = ", ".join(self.placeholder for _ in columns)
        return f"INSERT INTO {qualified} ({', '.join(quoted_identifier(column) for column in columns)}) VALUES ({marks})"

    def table_rules(self, connection: Any, schema: str | None, table: str) -> TableRules:
        """What ``table`` demands of a row, read from the catalog rather than learned by attempting one.

        The one method here that answers rather than refusing to. Every other is something a load
        cannot proceed without, so a loader that has not written it should say so loudly; this one
        is an optimisation, and its default -- ``beyond_the_frame``, the admission that the rows
        cannot be judged away from the server -- costs a slower diagnosis and never a wrong one.
        """
        return TableRules()

    def declared_column_types(self, connection: Any, schema: str | None, table: str) -> dict[str, str | None]:
        """What ``table`` declares each of its columns as, in this library's own vocabulary.

        An optimisation in the same sense as :meth:`table_rules`: the empty answer costs a caller
        the type check it wanted and never a wrong refusal, so a loader that has not written this
        one is allowed to decline the whole question rather than guess at it.
        """
        return {}

    def merge_rows(
        self,
        connection: Any,
        schema: str | None,
        table: str,
        columns: Sequence[str],
        key: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
    ) -> None:
        """Upsert ``rows`` into ``table`` on ``key``, through ``_etl_merge_staging``, which it disposes of.

        Staged rather than done a row at a time because the bulk path on one of the two drivers is
        ``COPY``, which cannot carry a conflict resolution of its own. Loading into scratch and
        upserting from it in one statement keeps the bulk load and gives both engines the same
        answer, which a per-row upsert on one and a ``COPY`` on the other would not.
        """
        raise NotImplementedError

    def existing_values(self, connection: Any, qualified: str, column: str, values: Sequence[Any]) -> set[Any]:
        """Which of ``values`` ``column`` already holds, asked in one statement rather than one per value."""
        raise NotImplementedError

    def transaction(self, connection: Any) -> AbstractContextManager[None]:
        """A unit of work: committed if the block returns, rolled back if it raises."""
        raise NotImplementedError

    def discarded_transaction(self, connection: Any) -> AbstractContextManager[None]:
        """A unit of work always rolled back, for a pass that exists to look rather than to write."""
        raise NotImplementedError

    def attempted_row(self, connection: Any) -> AbstractContextManager[None]:
        """One row's attempt, undone on failure without taking the surrounding pass down with it."""
        raise NotImplementedError


class _SqliteLoader(RowLoader):
    placeholder = "?"
    integrity_errors = (sqlite3.IntegrityError,)
    driver_errors = (sqlite3.Error,)

    def load_rows(
        self, connection: Any, qualified: str, columns: Sequence[str], rows: Sequence[tuple[Any, ...]]
    ) -> None:
        if rows:
            connection.executemany(self._insert_statement(qualified, columns), rows)

    def table_rules(self, connection: Any, schema: str | None, table: str) -> TableRules:
        """The pragmas, plus the definition's own text, because a CHECK is not in the pragmas.

        ``table_info`` gives nullability and the key; ``index_list``/``index_info`` give the unique
        indexes, and say whether each one stands behind a constraint or was created on its own;
        ``foreign_key_list`` gives the keys whose answer is in another table. A CHECK appears in
        none of them -- it exists only as text in the definition -- so a table whose definition says
        the word is declared unjudgeable outright rather than parsed. That over-reaches on a column
        merely named for it, which costs a slower diagnosis and never a wrong one.
        """
        prefix = f"{quoted_identifier(schema)}." if schema else ""
        name = quoted_identifier(table)
        info = connection.execute(f"pragma {prefix}table_info({name})").fetchall()
        key = [entry[_TABLE_INFO_NAME] for entry in info if entry[_TABLE_INFO_KEY_POSITION]]
        # A key over one column is a rule a row can be read against; a key over several is not, and
        # neither is any one of its columns, so it contributes nothing rather than its first column.
        unique = list(key) if len(key) == 1 else []
        beyond = len(key) > 1
        for entry in connection.execute(f"pragma {prefix}index_list({name})").fetchall():
            index_name, is_unique, origin, partial = entry[1], entry[2], entry[3], entry[4]
            if not is_unique:
                continue
            if origin == "c" or partial:
                beyond = True
                continue
            indexed = [
                row[_INDEX_INFO_NAME]
                for row in connection.execute(f"pragma {prefix}index_info({quoted_identifier(index_name)})").fetchall()
            ]
            if len(indexed) == 1 and indexed[0] is not None:
                unique.append(indexed[0])
            else:
                beyond = True
        if connection.execute(f"pragma {prefix}foreign_key_list({name})").fetchall():
            beyond = True
        definition = connection.execute(
            f"select sql from {prefix}sqlite_master where type = 'table' and name = ?", (table,)
        ).fetchone()
        if definition is None or definition[0] is None or _CHECK_WORD.search(definition[0]):
            beyond = True
        return TableRules(
            not_null=tuple(entry[_TABLE_INFO_NAME] for entry in info if entry[_TABLE_INFO_NOT_NULL]),
            unique=tuple(dict.fromkeys(unique)),
            settled_by_the_engine=tuple(key),
            beyond_the_frame=beyond,
        )

    def declared_column_types(self, connection: Any, schema: str | None, table: str) -> dict[str, str | None]:
        """``table_info``'s third column, which is the word the definition used and nothing more.

        ``cursor.description`` cannot answer this on sqlite. Its type codes are absent outside the
        detect-types modes, and the storage class a value came back as is a property of the value
        rather than of the column: this engine stores the text ``'heavy'`` in a column declared
        ``integer`` and reports success. The declaration is the only thing here that says what the
        column was meant to hold.
        """
        prefix = f"{quoted_identifier(schema)}." if schema else ""
        info = connection.execute(f"pragma {prefix}table_info({quoted_identifier(table)})").fetchall()
        return {entry[_TABLE_INFO_NAME]: declared_type_of(entry[_TABLE_INFO_DECLARED_TYPE] or "") for entry in info}

    def merge_rows(
        self,
        connection: Any,
        schema: str | None,
        table: str,
        columns: Sequence[str],
        key: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
    ) -> None:
        """Scratch built from a zero-row select, and dropped by name on both sides of the upsert.

        This engine has neither ``LIKE`` nor ``ON COMMIT DROP``, so the shape of the scratch table
        is taken from a select that returns nothing and its disposal is explicit. Dropped *before*
        it is created as well as after it is read, because a temporary table here can outlive the
        unit of work that made it: a publish that failed part way would otherwise leave one behind,
        and the next attempt would collide with it and fail for a reason that has nothing to do with
        the rows.
        """
        qualified = qualified_name(schema, table)
        connection.execute(f"DROP TABLE IF EXISTS temp.{_MERGE_STAGING}")
        connection.execute(f"CREATE TEMPORARY TABLE {_MERGE_STAGING} AS SELECT * FROM {qualified} WHERE 0")
        if rows:
            connection.executemany(self._insert_statement(_MERGE_STAGING, columns), rows)
            connection.execute(_merge_statement(qualified, _MERGE_STAGING, columns, key))
        connection.execute(f"DROP TABLE IF EXISTS temp.{_MERGE_STAGING}")

    def existing_values(self, connection: Any, qualified: str, column: str, values: Sequence[Any]) -> set[Any]:
        quoted = quoted_identifier(column)
        found: set[Any] = set()
        for start in range(0, len(values), _SQLITE_PARAMETER_CHUNK):
            chunk = list(values[start : start + _SQLITE_PARAMETER_CHUNK])
            marks = ", ".join("?" for _ in chunk)
            statement = f"SELECT {quoted} FROM {qualified} WHERE {quoted} IN ({marks})"
            found.update(row[0] for row in connection.execute(statement, chunk).fetchall())
        return found

    @contextmanager
    def transaction(self, connection: Any) -> Iterator[None]:
        with connection:
            yield

    @contextmanager
    def discarded_transaction(self, connection: Any) -> Iterator[None]:
        try:
            yield
        finally:
            connection.rollback()

    @contextmanager
    def attempted_row(self, connection: Any) -> Iterator[None]:
        """Nothing to undo: a failed statement on this driver leaves the rest of the pass usable."""
        yield


class _PostgresLoader(RowLoader):
    """The psycopg loader, whose every answer differs from sqlite3's in a way that has already bitten.

    ``COPY ... FROM STDIN`` rather than a row at a time, because a publish is of the order of a
    hundred thousand rows and a statement per row is not a slower way to do that, it is a different
    job. The stream is written through the driver's own row writer rather than a hand-built line,
    because that stream is tab-delimited and backslash-escaped: a value carrying a tab, a newline
    or the two characters ``\\N`` silently lands in the wrong column, or as a null, if the escaping
    is done by hand.

    ``with connection:`` is *not* the unit of work here -- on this driver it closes the connection,
    where sqlite3's only commits. And a failed statement poisons the whole transaction, so replaying
    rows one at a time needs a savepoint per row rather than a bare attempt.
    """

    placeholder = "%s"

    def __init__(self, psycopg: Any) -> None:
        self.integrity_errors = (psycopg.errors.IntegrityError,)
        self.driver_errors = (psycopg.Error,)

    def load_rows(
        self, connection: Any, qualified: str, columns: Sequence[str], rows: Sequence[tuple[Any, ...]]
    ) -> None:
        if not rows:
            return
        statement = f"COPY {qualified} ({', '.join(quoted_identifier(column) for column in columns)}) FROM STDIN"
        with connection.cursor() as cursor, cursor.copy(statement) as stream:
            for row in rows:
                stream.write_row(row)

    def table_rules(self, connection: Any, schema: str | None, table: str) -> TableRules:
        """``pg_attribute`` for nullability, ``pg_constraint`` for the rules, ``pg_index`` for the rest.

        Three catalog reads and not one statement against the table itself. The constraint query
        resolves ``conkey`` to column names in the server, so a single-column key arrives as the name
        the frame knows it by; everything else -- a CHECK, a foreign key, an exclusion, a key
        spanning columns -- arrives as a reason to say the rows cannot be judged here. The last query
        is for the unique index no constraint stands behind, which enforces uniqueness just as firmly
        and appears in neither of the first two.
        """
        qualified = qualified_name(schema, table)
        cursor = connection.cursor()
        try:
            cursor.execute(
                "select attname from pg_attribute where attrelid = %s::regclass and attnum > 0 "
                "and not attisdropped and attnotnull",
                (qualified,),
            )
            not_null = tuple(row[0] for row in cursor.fetchall())
            cursor.execute(
                "select c.contype, (select array_agg(a.attname order by k.ord) "
                "from unnest(c.conkey) with ordinality as k(attnum, ord) "
                "join pg_attribute a on a.attrelid = c.conrelid and a.attnum = k.attnum) "
                "from pg_constraint c where c.conrelid = %s::regclass",
                (qualified,),
            )
            constraints = cursor.fetchall()
            cursor.execute(
                "select count(*) from pg_index i where i.indrelid = %s::regclass and i.indisunique "
                "and not exists (select 1 from pg_constraint c where c.conindid = i.indexrelid)",
                (qualified,),
            )
            unbacked = cursor.fetchone()[0]
        finally:
            cursor.close()
        return _rules_from_constraints(not_null, constraints, unbacked > 0)

    def declared_column_types(self, connection: Any, schema: str | None, table: str) -> dict[str, str | None]:
        """``format_type`` over ``pg_attribute``, which is the only reading that survives a domain.

        ``cursor.description`` carries type codes here, unlike on sqlite, but they are OIDs a caller
        would have to resolve against the catalog anyway -- and a domain resolves to its own OID
        rather than to the type underneath it. Asking the catalog directly is the same round trip
        and gives the name the definition was written with.
        """
        qualified = qualified_name(schema, table)
        cursor = connection.cursor()
        try:
            cursor.execute(
                "select attname, format_type(atttypid, atttypmod) from pg_attribute "
                "where attrelid = %s::regclass and attnum > 0 and not attisdropped",
                (qualified,),
            )
            return {name: declared_type_of(spelling) for name, spelling in cursor.fetchall()}
        finally:
            cursor.close()

    def merge_rows(
        self,
        connection: Any,
        schema: str | None,
        table: str,
        columns: Sequence[str],
        key: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
    ) -> None:
        """Scratch shaped by ``LIKE``, loaded by ``COPY``, and upserted from in one statement.

        ``LIKE`` copies the target's ``NOT NULL`` declarations, so a row the target would refuse is
        refused on the way into the scratch table rather than surviving as far as the upsert.
        ``ON COMMIT DROP`` disposes of the table whichever way the unit of work ends; the explicit
        drop afterwards is what lets a second merge in the same unit of work reuse the name.
        """
        qualified = qualified_name(schema, table)
        connection.execute(f"CREATE TEMPORARY TABLE {_MERGE_STAGING} (LIKE {qualified}) ON COMMIT DROP")
        if rows:
            self.load_rows(connection, _MERGE_STAGING, columns, rows)
            connection.execute(_merge_statement(qualified, _MERGE_STAGING, columns, key))
        connection.execute(f"DROP TABLE {_MERGE_STAGING}")

    def existing_values(self, connection: Any, qualified: str, column: str, values: Sequence[Any]) -> set[Any]:
        quoted = quoted_identifier(column)
        cursor = connection.cursor()
        try:
            cursor.execute(f"SELECT {quoted} FROM {qualified} WHERE {quoted} = ANY(%s)", (list(values),))
            return {row[0] for row in cursor.fetchall()}
        finally:
            cursor.close()

    @contextmanager
    def transaction(self, connection: Any) -> Iterator[None]:
        with connection.transaction():
            yield

    @contextmanager
    def discarded_transaction(self, connection: Any) -> Iterator[None]:
        with connection.transaction(force_rollback=True):
            yield

    @contextmanager
    def attempted_row(self, connection: Any) -> Iterator[None]:
        with connection.transaction():
            yield


def _rules_from_constraints(
    not_null: Sequence[str],
    constraints: Sequence[tuple[str, Sequence[str] | None]],
    unbacked_unique_index: bool,
) -> TableRules:
    """Turn what ``pg_constraint`` said into rules the rows can be judged against, or an admission.

    A primary key or a unique constraint over one column is a rule a frame can be read against.
    Everything else the catalog reports -- with the sole exception of the not-null constraint newer
    servers record here, which ``attnotnull`` has already answered -- is a rule it cannot be, and
    one of those is enough to make the whole answer unsafe to give from the rows alone.
    """
    unique: list[str] = []
    beyond = unbacked_unique_index
    for contype, columns in constraints:
        if contype in ("p", "u"):
            if columns is not None and len(columns) == 1:
                unique.append(columns[0])
            else:
                beyond = True
        elif contype != "n":
            beyond = True
    return TableRules(not_null=tuple(not_null), unique=tuple(dict.fromkeys(unique)), beyond_the_frame=beyond)


def driver_for(connection: Any) -> RowLoader:
    """The loader belonging to ``connection``, or a refusal naming the drivers this library knows.

    Dispatch is by asking what the connection is, never by recognising one driver and assuming the
    other. A connection from a third driver answers most of the methods a load reaches for, so
    assuming would not fail loudly: it would write through transaction and escaping rules nothing
    here has ever checked. It is refused by name instead, the way an unrecognised scheme already is.
    """
    if isinstance(connection, sqlite3.Connection):
        return _SqliteLoader()

    try:
        import psycopg
    except ImportError:
        psycopg = None  # type: ignore[assignment]
    if psycopg is not None and isinstance(connection, psycopg.Connection):
        return _PostgresLoader(psycopg)

    kind = type(connection)
    described = f"{kind.__module__}.{kind.__qualname__}"
    raise ConfigError(
        f"rows cannot be loaded through a connection of type {described!r}. This library loads "
        f"through sqlite3 for a 'sqlite' connection and psycopg for a 'postgresql' one. A "
        f"postgresql connection from any other driver lands here -- open_connection falls back to "
        f"ADBC when psycopg is missing -- so install 'psycopg' to write to postgresql."
    )
