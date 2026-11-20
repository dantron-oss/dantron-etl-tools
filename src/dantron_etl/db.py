"""Database connections and the table helpers the table tools share.

A step names an environment variable; :func:`resolve_connection_uri` reads the connection string
from it and :func:`open_connection` opens it. The driver is picked by the URI's scheme: SQLite
needs only the standard library, Postgres uses ``psycopg`` (or ADBC), imported only when needed.
A missing driver is refused with the package to install.

:func:`driver_for` gives the :class:`RowLoader` for a connection, which writes rows, merges, and
reads what a table declares. Table and schema names are checked with :func:`validate_identifier`
before any connection opens.

Why: docs/design/database.md
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
from urllib.parse import unquote, urlparse

from .errors import ConfigError

__all__ = [
    "RowLoader",
    "TableRules",
    "declared_type_of",
    "describe_connection",
    "driver_for",
    "open_connection",
    "probe_columns",
    "qualified_name",
    "quoted_identifier",
    "require_matching_columns",
    "resolve_connection_uri",
    "validate_identifier",
]

# Matched with fullmatch: a pattern ending in '$' would also accept a trailing newline.
_ENV_NAME: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_SQLITE_PREFIX: Final[str] = "sqlite:///"
_KNOWN_SCHEMES: Final[tuple[str, ...]] = ("sqlite", "file", "postgresql", "postgres")
_POSTGRES_SCHEMES: Final[frozenset[str]] = frozenset({"postgresql", "postgres"})

# A SQLite CHECK rule appears only in the table definition's text, never in the pragmas.
_CHECK_WORD: Final[re.Pattern[str]] = re.compile(r"\bcheck\b", re.IGNORECASE)

# How many values one SQLite statement may carry. SQLite's own limit is higher.
_SQLITE_PARAMETER_CHUNK: Final[int] = 500

# Positions in the rows the pragmas return. ``table_info`` gives (cid, name, type, notnull,
# dflt_value, pk) per column; ``pk`` is the column's position in the primary key, 0 when outside it.
# ``index_info`` gives (seqno, cid, name) per indexed column.
_TABLE_INFO_NAME: Final[int] = 1
_TABLE_INFO_DECLARED_TYPE: Final[int] = 2
_TABLE_INFO_NOT_NULL: Final[int] = 3
_TABLE_INFO_KEY_POSITION: Final[int] = 5
_INDEX_INFO_NAME: Final[int] = 2

# The temporary table a merge loads into before it upserts.
# Why: docs/design/database.md#merging-through-a-staging-table
_MERGE_STAGING: Final[str] = "_etl_merge_staging"

# The SQL type names the library understands, and the column type each means. Any other name
# gets ``None`` ("no opinion").
# Why: docs/design/database.md#one-type-vocabulary-for-both-engines
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
    """Store Python dates and datetimes in SQLite as ISO text.

    Why: docs/design/database.md#dates-in-sqlite
    """
    sqlite3.register_adapter(date, lambda value: value.isoformat())
    sqlite3.register_adapter(datetime, lambda value: value.isoformat(sep=" "))


_register_adapters()


def validate_identifier(value: str, field: str) -> None:
    """Refuse ``value`` unless it is a plain name: letters, digits and underscores, not starting with a digit.

    ``field`` is the setting it came from, used in the message. Raises :class:`ConfigError`. Call it on
    a schema or table name before opening a connection.

    Why: docs/design/database.md#a-plain-name-is-the-only-guard
    """
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ConfigError(
            f"config field {field!r} must be a plain identifier (letters, digits and underscores, "
            f"not starting with a digit); refusing {value!r}"
        )


def qualified_name(schema: str | None, table: str) -> str:
    """Return ``schema.table``, or ``table`` alone when there is no schema."""
    return f"{schema}.{table}" if schema else table


def resolve_connection_uri(name: str, field: str) -> str:
    """Return the connection string held in the environment variable ``name``.

    ``field`` is the setting that named the variable, used in the message. Raises :class:`ConfigError`
    if ``name`` does not look like a variable name (it may be the connection string itself, so it is
    not repeated) or if the variable is not set. Reads ``os.environ`` only, never a ``.env`` file.

    Why: docs/design/database.md#credentials-come-from-the-environment
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
    """Open a DB-API connection for ``uri`` as a context manager, and close it on the way out.

    Use it as ``with open_connection(uri) as connection:``. The scheme picks the driver:
    ``sqlite:///<path>``, ``file:`` (SQLite), or ``postgresql://`` / ``postgres://``. Raises
    :class:`ConfigError` for an unknown scheme, a SQLite file that does not exist, a missing Postgres
    driver, or a failed connection. The message never shows the password.
    """
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
    """Open a ``sqlite:///<path>`` database, refusing a path that does not exist.

    Why: docs/design/database.md#a-missing-sqlite-file-is-refused
    """
    if not uri.startswith(_SQLITE_PREFIX):
        raise ConfigError(f"a sqlite connection must be written as {_SQLITE_PREFIX}<path>")
    path = Path(uri[len(_SQLITE_PREFIX) :])
    if not path.exists():
        raise ConfigError(f"sqlite database '{path}' does not exist; a missing path is refused, not created")
    return sqlite3.connect(path)


def _connect_sqlite_file_uri(uri: str) -> sqlite3.Connection:
    """Open a SQLite ``file:`` URI. With no ``mode`` stated, the file must already exist.

    A URI that states a mode (``mode=memory``, ``mode=rwc``) is opened as written.
    """
    if "mode=" not in uri:
        uri = f"{uri}{'&' if '?' in uri else '?'}mode=rw"
    failure: str | None = None
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        failure = f"{type(error).__name__}: {error}"
    raise ConfigError(f"could not open the sqlite database the file: URI names ({failure}); a missing file is not created")


@dataclass(frozen=True)
class _Where:
    """The parts of a Postgres address that say which database it is. Never the user or the password."""

    host: str | None
    port: str | None
    database: str | None


class _Unreadable(ValueError):
    """An address that cannot be read without risking showing part of its password."""


class _PasswordInPlace(_Unreadable):
    """An address whose password is also written in its host, port or database name."""


_POSTGRES_PREFIXES: Final[tuple[str, ...]] = ("postgresql://", "postgres://")
_PORT: Final[re.Pattern[str]] = re.compile(r"[0-9]+(,[0-9]*)*")


def _decoded(text: str) -> str:
    """``text`` with its %xx codes read, refusing a broken one as the Postgres driver does."""
    if re.search(r"%(?![0-9A-Fa-f]{2})", text):
        raise _Unreadable(text)
    return unquote(text, errors="strict")


def _uri_where(uri: str) -> tuple[_Where, str | None, str | None]:
    """Read a ``postgresql://`` address the way the Postgres driver reads it: the place, and the password.

    The password comes back as written and as decoded, so a caller can check it shows nowhere. An
    address is refused when an @ is left after the one before the host: written that way, the driver
    reads part of the password as the host or the database name.
    """
    rest = uri.split("://", 1)[1]
    password: str | None = None
    stop = next((at for at, char in enumerate(rest) if char in "@/"), len(rest))
    if stop < len(rest) and rest[stop] == "@":
        userinfo, rest = rest[:stop], rest[stop + 1 :]
        password = userinfo.partition(":")[2] or None
    if "@" in rest:
        raise _Unreadable(uri)
    end = next((at for at, char in enumerate(rest) if char in "/?"), len(rest))
    hosts, rest = rest[:end], rest[end:]
    path, _, query = rest.partition("?")
    found: dict[str, str] = {}
    for item in query.split("&") if query else ():
        key, equals, value = item.partition("=")
        if not equals:
            raise _Unreadable(uri)
        found[_decoded(key)] = value
    names, ports = [], []
    for one in hosts.split(",") if hosts else ():
        if one.startswith("["):  # an IPv6 address, such as [::1]:5432
            name, _, port = one.partition("]")
            name, port = f"{name}]", port.removeprefix(":")
        else:
            name, _, port = one.partition(":")
        names.append(_decoded(name))
        ports.append(port)
    host = _decoded(found["host"]) if "host" in found else ",".join(names) or None
    port = _decoded(found["port"]) if "port" in found else ",".join(ports).strip(",") or None
    database = _decoded(found["dbname"]) if "dbname" in found else (_decoded(path[1:]) if path[1:] else None)
    if "password" in found:
        password = found["password"]
    return _Where(host, port, database), password, _decoded(password) if password else None


def _key_value_where(text: str) -> tuple[_Where, str | None, str | None]:
    """Read ``host=... dbname=...`` pairs the way the Postgres driver reads them: the place, and the password.

    A value is a word, or text in single quotes; a backslash keeps the next character as it is.
    """
    found: dict[str, str] = {}
    at, size = 0, len(text)
    while True:
        while at < size and text[at].isspace():
            at += 1
        if at == size:
            break
        start = at
        while at < size and text[at] != "=" and not text[at].isspace():
            at += 1
        key = text[start:at]
        while at < size and text[at].isspace():
            at += 1
        if not key or at == size or text[at] != "=":
            raise _Unreadable(text)
        at += 1
        while at < size and text[at].isspace():
            at += 1
        value: list[str] = []
        quoted = at < size and text[at] == "'"
        at += 1 if quoted else 0
        while at < size:
            char = text[at]
            if char == "\\" and at + 1 < size:
                value.append(text[at + 1])
                at += 2
                continue
            if (quoted and char == "'") or (not quoted and char.isspace()):
                break
            value.append(char)
            at += 1
        if quoted:
            if at == size:
                raise _Unreadable(text)
            at += 1
        found[key] = "".join(value)
    where = _Where(found.get("host") or found.get("hostaddr"), found.get("port"), found.get("dbname"))
    password = found.get("password")
    return where, password, password


def _postgres_where(value: str) -> _Where:
    """Where a Postgres address points, or _Unreadable when that cannot be said without risking the password.

    The last check is blunt on purpose: if the password, as written or decoded, appears in anything
    that would be shown, nothing is shown.
    """
    try:
        if value.lower().startswith(_POSTGRES_PREFIXES):
            where, written, decoded = _uri_where(value)
        else:
            where, written, decoded = _key_value_where(value)
    except (ValueError, IndexError, UnicodeDecodeError) as error:
        raise _Unreadable(value) from error
    if where.port is not None and not _PORT.fullmatch(where.port):
        raise _Unreadable(value)
    shown = " ".join(part for part in (where.host, where.port, where.database) if part)
    if any(secret and secret in shown for secret in (written, decoded)):
        raise _PasswordInPlace(value)
    return where


def _place(where: _Where) -> str:
    """``database 'x' on host y, port z``, leaving out what the address does not state."""
    parts = [f"database {where.database!r}" if where.database else "the default database"]
    parts.append(f"on host {where.host}" if where.host else "on the default host")
    text = " ".join(parts)
    return f"{text}, port {where.port}" if where.port else text


def describe_connection(name: Any) -> str:
    """Which database the environment variable ``name`` points at now, in words a person can check.

    For a Postgres address: the database, the host, and the port when one is stated. For SQLite: the
    file. Never the user or the password, and it never raises: a variable that is not set, or an
    address that cannot be read safely, is said in words. The confirmation before a real run uses it.
    """
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        return (
            "a database the step's connection setting cannot name, because the setting is not the name "
            "of an environment variable; a run refuses it"
        )
    value = os.environ.get(name)
    source = f"environment variable {name}"
    if value is None:
        return f"the database named by {source}, which is not set, so which database it is cannot be shown"
    lowered = value.strip().lower()
    if lowered.startswith(_SQLITE_PREFIX):
        return f"the SQLite file {value.strip()[len(_SQLITE_PREFIX) :]} (from {source})"
    if lowered.startswith("file:"):
        location = value.strip()[len("file:") :].partition("?")[0]
        if "mode=memory" in lowered:
            return f"an in-memory SQLite database (from {source})"
        return f"the SQLite file {location} (from {source})"
    is_address = lowered.startswith(_POSTGRES_PREFIXES)
    if "://" in lowered and not is_address:
        return (
            f"the database named by {source}, whose value is not an address a run opens (it opens "
            f"sqlite:///, file: and postgresql:// addresses), so which database it is cannot be shown"
        )
    try:
        where = _postgres_where(value.strip())
    except _PasswordInPlace:
        return (
            f"the database named by {source}, whose password is also written in its host, port or "
            f"database name, so which database it is cannot be shown without showing the password"
        )
    except _Unreadable:
        return (
            f"the database named by {source}, whose value cannot be read safely, so which database it "
            f"is cannot be shown. In a postgresql:// address, write an @ or / in the password as %40 "
            f"or %2F"
        )
    if is_address:
        return f"{_place(where)} (from {source})"
    return (
        f"{_place(where)} (from {source}), written as key=value pairs, which a run cannot open; "
        f"write it as postgresql://host:port/database"
    )


def _could_not_connect(uri: str, failure: str) -> str:
    """Build the message for a failed Postgres connection.

    It names the database, host and port, and the kind of error. It never shows the driver's own
    message, which can contain the password.
    """
    try:
        place = _postgres_where(uri)
        host, database, port = place.host, place.database, place.port
    except _Unreadable:
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
    """Open a Postgres connection with ``psycopg``, or with ADBC if psycopg is not installed.

    Raises :class:`ConfigError` naming the package to install when neither is present.
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
        # Raised outside the except block so the driver's message, which may hold the password,
        # is not chained into the traceback.
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
    """Return the column names of table ``qualified``, without reading any rows.

    Raises :class:`ConfigError` saying the table was not found, or naming the driver's error.
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
    """Say whether a driver's error means the table or its schema does not exist."""
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
    """Refuse rows whose columns are not exactly the table's columns.

    ``frame_columns`` are the rows' columns, ``table_columns`` the table's. ``stamp`` is a column the
    writer adds itself, so it counts on the rows' side. The check runs both ways and is case-sensitive:
    a column only the rows have, or only the table has, raises :class:`ConfigError` listing both.
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
    """Return a column name in double quotes, so SQL reads it as a name.

    Use it for names read from the database itself, which may be reserved words or hold spaces.
    """
    return '"' + name.replace('"', '""') + '"'


def declared_type_of(sql_type: str) -> str | None:
    """Return the column type name (``"Int64"``, ``"String"`` ...) a SQL type means, or ``None``.

    ``sql_type`` is the type as the database spells it, such as ``varchar(20)`` or ``timestamp with
    time zone``. Width, precision and time zone are ignored. ``None`` means "no opinion": an array, a
    domain, an enum, JSON or any name not in the table.

    Why: docs/design/database.md#one-type-vocabulary-for-both-engines
    """
    name = sql_type.strip().lower()
    if name.endswith("[]"):
        return None
    name = name.split("(")[0].strip()
    for suffix in (" without time zone", " with time zone"):
        name = name.removesuffix(suffix)
    return _SQL_TYPE_NAMES.get(name.strip())


# The type of value psycopg returns for each Postgres type, where that never varies.
_POSTGRES_VALUE_TYPES: Final[dict[str, str]] = {
    "smallint": "Int64",
    "integer": "Int64",
    "bigint": "Int64",
    "real": "Float64",
    "double precision": "Float64",
    "text": "String",
    "character varying": "String",
    "character": "String",
    "boolean": "Boolean",
    "date": "Date",
    "timestamp without time zone": "Datetime",
    "time without time zone": "Time",
}


def _bare_type_name(spelling: str) -> str:
    """Drop a type's width or precision: ``timestamp(3) without time zone`` becomes ``timestamp without time zone``."""
    return re.sub(r"\(\s*\d+(\s*,\s*\d+)?\s*\)", "", spelling).strip().lower()


def _sqlite_value_type(spelling: str) -> str | None:
    """Return the kind of value a SQLite column stores, by SQLite's own rule on its declared type.

    A type naming INT stores whole numbers; CHAR, CLOB or TEXT stores text; REAL, FLOA or DOUB stores
    figures. Any other column stores whatever it was given, so the answer is ``None``.
    """
    name = spelling.upper()
    if "INT" in name:
        return "Int64"
    if any(word in name for word in ("CHAR", "CLOB", "TEXT")):
        return "String"
    if any(word in name for word in ("REAL", "FLOA", "DOUB")):
        return "Float64"
    return None


def _merge_statement(
    qualified: str, staging: str, columns: Sequence[str], key: Sequence[str], *, keep_published: bool
) -> str:
    """Build the upsert from the staging table into ``qualified``.

    With ``keep_published``, a null in an incoming row leaves the published value as it is. Without
    it, the null is written.

    Why: docs/design/database.md#merging-through-a-staging-table
    """
    quoted = [quoted_identifier(column) for column in columns]
    named = ", ".join(quoted)
    conflict = ", ".join(quoted_identifier(column) for column in key)
    def assigned(column: str) -> str:
        incoming = f"excluded.{quoted_identifier(column)}"
        if keep_published:
            return f"{quoted_identifier(column)} = coalesce({incoming}, {qualified}.{quoted_identifier(column)})"
        return f"{quoted_identifier(column)} = {incoming}"

    updated = [assigned(column) for column in columns if column not in set(key)]
    resolution = f"UPDATE SET {', '.join(updated)}" if updated else "NOTHING"
    return (
        f"INSERT INTO {qualified} ({named}) SELECT {named} FROM {staging} WHERE true "
        f"ON CONFLICT ({conflict}) DO {resolution}"
    )


@dataclass(frozen=True)
class TableRules:
    """The rules a table sets on its rows, as far as they can be read from the database's catalog.

    - ``not_null``: columns that refuse a null.
    - ``unique``: columns with a one-column unique rule or primary key.
    - ``settled_by_the_engine``: key columns where SQLite fills or accepts a null on its own, so a
      null there cannot be judged from the rows.
    - ``beyond_the_frame``: true when the table has any rule the rows cannot be checked against
      alone (a CHECK, a foreign key, a key over several columns ...). It defaults to true.

    Why: docs/design/database.md#reading-a-tables-rules
    """

    not_null: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()
    settled_by_the_engine: tuple[str, ...] = ()
    beyond_the_frame: bool = True


class RowLoader:
    """How one database driver writes rows, runs a transaction and reads a table's rules.

    Get one with :func:`driver_for`, never by building it yourself. The two drivers differ in each of
    these, and the standard Python database interface does not settle them.

    Why: docs/design/database.md#one-loader-per-driver
    """

    placeholder: str
    integrity_errors: tuple[type[BaseException], ...]
    driver_errors: tuple[type[BaseException], ...]

    def load_rows(
        self, connection: Any, qualified: str, columns: Sequence[str], rows: Sequence[tuple[Any, ...]]
    ) -> None:
        """Write all of ``rows`` into table ``qualified``, in bulk."""
        raise NotImplementedError

    def insert_row(self, connection: Any, qualified: str, columns: Sequence[str], row: tuple[Any, ...]) -> None:
        """Write one row. Used to replay a failed bulk load row by row and find the bad rows."""
        connection.execute(self._insert_statement(qualified, columns), row)

    def _insert_statement(self, qualified: str, columns: Sequence[str]) -> str:
        # Quote every column, so a name like 'order' or one holding a space is not a syntax error.
        marks = ", ".join(self.placeholder for _ in columns)
        return f"INSERT INTO {qualified} ({', '.join(quoted_identifier(column) for column in columns)}) VALUES ({marks})"

    def table_rules(self, connection: Any, schema: str | None, table: str) -> TableRules:
        """Return the :class:`TableRules` for ``table``, read from the database's catalog.

        The default answers "cannot judge from the rows". That makes a failure slower to explain, never
        wrong.
        """
        return TableRules()

    def declared_column_types(self, connection: Any, schema: str | None, table: str) -> dict[str, str | None]:
        """Return each column's declared type, in the library's type names (see :func:`declared_type_of`).

        The default is an empty answer, which skips the type check rather than guess.
        """
        return {}

    def value_types(self, connection: Any, schema: str | None, table: str) -> dict[str, str | None]:
        """Return the type each column's values come back as when read, from the table's definition alone.

        Used for a read with no rows to look at. ``None`` where the answer depends on the value. The default
        is an empty answer.
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
        *,
        keep_published: bool,
    ) -> None:
        """Upsert ``rows`` into ``table`` on the ``key`` columns.

        ``keep_published`` says what a null in an incoming row does: true leaves the published value as it
        is, false writes the null. The rows go through a temporary table named ``_etl_merge_staging``,
        which is removed afterwards.

        Why: docs/design/database.md#merging-through-a-staging-table
        """
        raise NotImplementedError

    def existing_values(self, connection: Any, qualified: str, column: str, values: Sequence[Any]) -> set[Any]:
        """Return which of ``values`` column ``column`` of ``qualified`` already holds."""
        raise NotImplementedError

    def transaction(self, connection: Any) -> AbstractContextManager[None]:
        """A transaction as a context manager: committed if the block returns, rolled back if it raises."""
        raise NotImplementedError

    def discarded_transaction(self, connection: Any) -> AbstractContextManager[None]:
        """A transaction that is always rolled back, for a pass that only looks."""
        raise NotImplementedError

    def attempted_row(self, connection: Any) -> AbstractContextManager[None]:
        """Try one row inside a pass. A failure undoes that row only."""
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
        """Read the rules from SQLite's pragmas, and from the table definition's text for CHECK.

        A definition that contains the word "check" anywhere is treated as unjudgeable.

        Why: docs/design/database.md#sqlite-check-rules
        """
        prefix = f"{quoted_identifier(schema)}." if schema else ""
        name = quoted_identifier(table)
        info = connection.execute(f"pragma {prefix}table_info({name})").fetchall()
        key = [entry[_TABLE_INFO_NAME] for entry in info if entry[_TABLE_INFO_KEY_POSITION]]
        # Only a one-column key can be checked against the rows. A key over several columns cannot.
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
        """Read each column's declared type from ``pragma table_info``.

        Why: docs/design/database.md#sqlite-declared-types
        """
        prefix = f"{quoted_identifier(schema)}." if schema else ""
        info = connection.execute(f"pragma {prefix}table_info({quoted_identifier(table)})").fetchall()
        return {entry[_TABLE_INFO_NAME]: declared_type_of(entry[_TABLE_INFO_DECLARED_TYPE] or "") for entry in info}

    def value_types(self, connection: Any, schema: str | None, table: str) -> dict[str, str | None]:
        """Answer by SQLite's own rule for which kind of value a declared type stores."""
        prefix = f"{quoted_identifier(schema)}." if schema else ""
        info = connection.execute(f"pragma {prefix}table_info({quoted_identifier(table)})").fetchall()
        return {entry[_TABLE_INFO_NAME]: _sqlite_value_type(entry[_TABLE_INFO_DECLARED_TYPE] or "") for entry in info}

    def merge_rows(
        self,
        connection: Any,
        schema: str | None,
        table: str,
        columns: Sequence[str],
        key: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
        *,
        keep_published: bool,
    ) -> None:
        """Stage the rows in a temporary table, upsert from it, then drop it.

        The staging table is dropped before it is created as well as after.

        Why: docs/design/database.md#merging-in-sqlite
        """
        qualified = qualified_name(schema, table)
        connection.execute(f"DROP TABLE IF EXISTS temp.{_MERGE_STAGING}")
        connection.execute(f"CREATE TEMPORARY TABLE {_MERGE_STAGING} AS SELECT * FROM {qualified} WHERE 0")
        if rows:
            connection.executemany(self._insert_statement(_MERGE_STAGING, columns), rows)
            connection.execute(_merge_statement(qualified, _MERGE_STAGING, columns, key, keep_published=keep_published))
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
        """Nothing to undo: on SQLite a failed statement leaves the rest of the pass usable."""
        yield


class _PostgresLoader(RowLoader):
    """The loader for ``psycopg`` connections.

    Rows are written with ``COPY ... FROM STDIN`` through the driver's own row writer. A transaction
    uses ``connection.transaction()``, and each replayed row gets its own savepoint.

    Why: docs/design/database.md#postgres-loading
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
        """Read the rules from ``pg_attribute``, ``pg_constraint`` and ``pg_index``.

        The last query finds unique indexes that no constraint stands behind.
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
        """Read each column's declared type with ``format_type`` over ``pg_attribute``.

        Why: docs/design/database.md#postgres-declared-types
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

    def value_types(self, connection: Any, schema: str | None, table: str) -> dict[str, str | None]:
        """Answer only for types psycopg always returns as one kind of Python value.

        A numeric, a zone-aware timestamp, a uuid or a JSON column has no answer.
        """
        qualified = qualified_name(schema, table)
        cursor = connection.cursor()
        try:
            cursor.execute(
                "select attname, format_type(atttypid, atttypmod) from pg_attribute "
                "where attrelid = %s::regclass and attnum > 0 and not attisdropped",
                (qualified,),
            )
            return {name: _POSTGRES_VALUE_TYPES.get(_bare_type_name(spelling)) for name, spelling in cursor.fetchall()}
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
        *,
        keep_published: bool,
    ) -> None:
        """Stage the rows in a temporary table made with ``LIKE``, load it by ``COPY``, upsert, then drop it.

        Why: docs/design/database.md#merging-in-postgres
        """
        qualified = qualified_name(schema, table)
        connection.execute(f"CREATE TEMPORARY TABLE {_MERGE_STAGING} (LIKE {qualified}) ON COMMIT DROP")
        if rows:
            self.load_rows(connection, _MERGE_STAGING, columns, rows)
            connection.execute(_merge_statement(qualified, _MERGE_STAGING, columns, key, keep_published=keep_published))
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
    """Turn what ``pg_constraint`` reports into :class:`TableRules`.

    A primary key or unique constraint over one column becomes a ``unique`` column. Any other
    constraint, except not-null, sets ``beyond_the_frame``.
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
    """Return the :class:`RowLoader` for ``connection``.

    Accepts a ``sqlite3`` connection or a ``psycopg`` one. Any other connection, including an ADBC
    one, raises :class:`ConfigError`.

    Why: docs/design/database.md#an-unknown-driver-is-refused
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
