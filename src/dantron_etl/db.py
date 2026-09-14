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
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

from .tool import ConfigError

__all__ = [
    "open_connection",
    "probe_columns",
    "qualified_name",
    "resolve_connection_uri",
    "validate_identifier",
]

_ENV_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SQLITE_PREFIX: Final[str] = "sqlite:///"
_KNOWN_SCHEMES: Final[tuple[str, ...]] = ("sqlite", "postgresql")
_POSTGRES_SCHEMES: Final[frozenset[str]] = frozenset({"postgresql", "postgres"})


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
    if not _IDENTIFIER.match(value):
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
    if not _ENV_NAME.match(name):
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
    scheme = urlparse(uri).scheme.lower()
    if scheme == "sqlite":
        return _connect_sqlite_path(uri)
    if scheme == "file":
        return sqlite3.connect(uri, uri=True)
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
        raise ConfigError(f"sqlite database {str(path)!r} does not exist; a missing path is refused, not created")
    return sqlite3.connect(path)


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
    if psycopg is not None:
        return psycopg.connect(uri)

    try:
        import adbc_driver_postgresql.dbapi as adbc_postgres
    except ImportError:
        adbc_postgres = None  # type: ignore[assignment]
    if adbc_postgres is not None:
        return adbc_postgres.connect(uri)

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
    except Exception as error:  # noqa: BLE001 - any failure at this point means the table is not there
        raise ConfigError(f"table {qualified!r} was not found") from error
    columns = tuple(entry[0] for entry in cursor.description)
    cursor.close()
    return columns
