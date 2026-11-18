"""Fixtures shared by more than one test file.

- ``seeded_sqlite``: a SQLite file built once per session, copied for each test.
- ``postgres_dsn`` and ``postgres_scratch``: the test Postgres server, or a skip when there is none.
- ``environment_restored``: puts ``os.environ`` back the way it was after a test.
- ``fixture_flow_sandbox``: a folder to run ``pipelines/fixture.yaml`` in, so its sinks never
  write into the checkout.

Markers (registered in ``pyproject.toml``):

- ``postgres`` is added here to every test that reaches the Postgres server. Run
  ``pytest -m "not postgres"`` to leave them out.
- ``slow`` marks a test that takes seconds on its own. Run ``pytest -m "not slow"`` to leave them out.
"""

import os
import shutil
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from dantron_etl.paths import project_root

PROJECT_ROOT = project_root(__file__)

#: Where a Postgres server for the tests is. Unset, every Postgres test skips.
POSTGRES_VARIABLE = "DANTRON_TEST_POSTGRES"


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def seeded_sqlite(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Path, Callable[[Path], None]], Path]:
    """``seeded_sqlite(path, seed)`` puts a fresh copy of a seeded database at ``path``.

    ``seed(file)`` builds the tables in an empty file. It runs once per session for each seed
    function. Every later call copies that file. A commit costs about a second on some disks, and a
    copy costs almost nothing. Each test still gets its own file, so no test sees another's writes.
    """
    templates: dict[Callable[[Path], None], Path] = {}
    folder = tmp_path_factory.mktemp("sqlite_templates")

    def copy_of(destination: Path, seed: Callable[[Path], None]) -> Path:
        template = templates.get(seed)
        if template is None:
            template = folder / f"template_{len(templates)}.db"
            seed(template)
            templates[seed] = template
        destination.unlink(missing_ok=True)
        shutil.copyfile(template, destination)
        return destination

    return copy_of


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


@cache
def _postgres_reachable() -> bool:
    """Whether the server named by ``POSTGRES_VARIABLE`` accepts a connection. Asked once."""
    dsn = os.environ.get(POSTGRES_VARIABLE)
    if not dsn:
        return False
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=3):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    """The test server's DSN. The test skips when there is no server to reach."""
    if not _postgres_reachable():
        pytest.skip(f"no Postgres to test against: set {POSTGRES_VARIABLE}")
    return os.environ[POSTGRES_VARIABLE]


@pytest.fixture(scope="session")
def postgres_scratch(postgres_dsn: str) -> Iterator[Callable[[str], Any]]:
    """``postgres_scratch(schema)`` gives an autocommit connection that owns ``schema``.

    The schema is dropped and made fresh the first time it is asked for, and dropped again at the
    end of the session. The server may hold real data elsewhere, so a test only ever names its own
    scratch schema.
    """
    import psycopg

    opened: dict[str, Any] = {}

    def connection_for(schema: str) -> Any:
        if schema not in opened:
            connection = psycopg.connect(postgres_dsn, connect_timeout=5, autocommit=True)
            connection.execute(f"drop schema if exists {schema} cascade")
            connection.execute(f"create schema {schema}")
            opened[schema] = connection
        return opened[schema]

    try:
        yield connection_for
    finally:
        for schema, connection in opened.items():
            connection.execute(f"drop schema if exists {schema} cascade")
            connection.close()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test that reaches the Postgres server, so ``-m "not postgres"`` can leave it out.

    A test reaches it by asking for ``postgres_dsn`` (directly or through another fixture), or
    through a two-engine fixture declared with ``params=["sqlite", "postgres"]`` that asks for the
    server only when its parameter is ``"postgres"``. A test parametrized directly with the word
    ``"postgres"`` (a SQL dialect name, say) needs no server and is not marked.
    """
    for item in items:
        if "postgres_dsn" in getattr(item, "fixturenames", ()) or _runs_a_two_engine_fixture_on_postgres(item):
            item.add_marker(pytest.mark.postgres)


def _runs_a_two_engine_fixture_on_postgres(item: pytest.Item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    definitions = item._fixtureinfo.name2fixturedefs  # type: ignore[attr-defined]
    for name, value in callspec.params.items():
        if not (isinstance(value, str) and value == "postgres"):
            continue
        declared = definitions.get(name, ())
        if declared and declared[-1].params is not None and "postgres" in declared[-1].params:
            return True
    return False


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


@pytest.fixture
def environment_restored() -> Iterator[None]:
    """Put ``os.environ`` back as it was, whatever the test exported.

    A worked example that needs a database exports a variable whose name the test cannot know in
    advance, so the whole environment is saved and restored.
    """
    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


# ---------------------------------------------------------------------------
# The fixture flow
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixture_flow_sandbox(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A folder to run ``pipelines/fixture.yaml`` from, so its sinks write here and not in the checkout.

    The flow's paths are relative to the working folder: it reads ``pipelines/fixture/entries.csv``
    and writes two files under ``output_files/``. This folder holds a copy of the inputs and an empty
    ``output_files/``. Run the flow with ``contextlib.chdir(fixture_flow_sandbox)``.
    """
    sandbox = tmp_path_factory.mktemp("fixture_flow")
    shutil.copytree(PROJECT_ROOT / "pipelines" / "fixture", sandbox / "pipelines" / "fixture")
    (sandbox / "output_files").mkdir()
    return sandbox
