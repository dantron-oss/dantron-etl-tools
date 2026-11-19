"""The world a worked example runs in while the conformance check runs it.

An example that reads or writes a file gets a fresh scratch directory, named in its configuration by
the ``{scratch}`` token. An example that needs an environment variable sets it for the run with
``temporary_environment``, and one that needs a database uses ``example_database``. The check then
confirms the example left the environment and the working directory as it found them.

Why: docs/design/core.md#the-scratch-directory
"""

import contextvars
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from .errors import ConformanceError

__all__ = ["SCRATCH_TOKEN", "example_database", "scratch_path", "temporary_environment"]


@contextmanager
def temporary_environment(**values: str) -> Iterator[None]:
    """Set each of ``values`` in the environment for the ``with`` block, then put back what was there.

    A variable that had a value gets that value back; one that was absent is removed again. This
    happens whether the block finishes or raises. Use it in a worked example's ``setup``.
    """
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def example_database(uri: str, variable: str, seed: Callable[[sqlite3.Connection], None]) -> Iterator[None]:
    """Open a shared in-memory SQLite database for one worked example, for the ``with`` block.

    ``uri`` names the database, such as ``file:name?mode=memory&cache=shared``. ``seed`` builds its
    tables on an open connection, which is then committed. For the block, the environment variable
    ``variable`` holds ``uri``, so the example's configuration can name it. The database is closed
    on the way out, so the example can run again.

    Why: docs/design/core.md#a-worked-example-database
    """
    anchor = sqlite3.connect(uri, uri=True)
    try:
        seed(anchor)
        anchor.commit()
        with temporary_environment(**{variable: uri}):
            yield
    finally:
        anchor.close()


def _process_state() -> tuple[dict[str, str], str]:
    """Return the environment and the working directory, to compare later."""
    return dict(os.environ), os.getcwd()


def _refuse_disturbance(refusal: str, before: tuple[dict[str, str], str]) -> None:
    """Raise ``refusal`` as a :class:`ConformanceError` if the environment or working directory
    differs from ``before``, naming what changed. Nothing else (files, connections) is compared.
    """
    environment_before, directory_before = before
    environment_after, directory_after = _process_state()
    if directory_after != directory_before:
        raise ConformanceError(
            f"{refusal}: the working directory changed, from '{directory_before}' to '{directory_after}'"
        )
    changed = sorted(
        name
        for name in set(environment_before) | set(environment_after)
        if environment_before.get(name) != environment_after.get(name)
    )
    if changed:
        raise ConformanceError(f"{refusal}: environment variable(s) changed: {', '.join(changed)}")


# The scratch directory. ``{scratch}`` is the only token; ``${scratch}`` is an unbound rule
# reference and is left alone so the placeholder check still refuses it.

SCRATCH_TOKEN: Final[str] = "{scratch}"

# Why: docs/design/core.md#why-scratch-is-left-alone
_SCRATCH_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\$)\{scratch\}")

# The directory the current check opened, if any. A ContextVar so a setup sees only its own check's.
_current_scratch_directory: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_current_scratch_directory", default=None
)


def scratch_path(value: str) -> Path:
    """Return ``value`` as a path, with ``{scratch}`` replaced by the current check's directory.

    Call it from a worked example's ``setup``. Raises :class:`ConformanceError` when no worked
    example is being checked, because there is no directory yet.
    """
    directory = _current_scratch_directory.get()
    if directory is None:
        raise ConformanceError(
            "scratch_path() has nothing to resolve against: it may only be called while check_tool "
            "is checking a worked example whose configuration holds the {scratch} token"
        )
    return Path(_SCRATCH_PATTERN.sub(lambda _: str(directory), value))  # a function, for Windows paths


def _uses_scratch(value: Any) -> bool:
    """Say whether ``{scratch}`` (not ``${scratch}``) appears anywhere in ``value``, at any depth."""
    if isinstance(value, str):
        return _SCRATCH_PATTERN.search(value) is not None
    if isinstance(value, list):
        return any(_uses_scratch(item) for item in value)
    if isinstance(value, dict):
        return any(_uses_scratch(item) for item in value.values())
    return False


def _resolved_scratch(value: Any, directory: Path) -> Any:
    """Return a copy of ``value`` with every ``{scratch}`` replaced by ``directory``.

    Lists and dicts are rebuilt, so the stored example still shows ``{scratch}`` on the catalog page.
    """
    if isinstance(value, str):
        # A function, not a string: re.sub reads backslashes in a string as escapes (C:\Users).
        return _SCRATCH_PATTERN.sub(lambda _: str(directory), value)
    if isinstance(value, list):
        return [_resolved_scratch(item, directory) for item in value]
    if isinstance(value, dict):
        return {key: _resolved_scratch(item, directory) for key, item in value.items()}
    return value
