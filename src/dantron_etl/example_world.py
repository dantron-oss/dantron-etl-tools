"""The world a worked example runs in while the conformance check runs it.

An example that reads or writes a file gets a fresh scratch directory, named in its configuration by
the ``{scratch}`` token. An example that needs an environment variable sets it for the run with
``temporary_environment``. The check then confirms the example left the environment and the working
directory as it found them.

**A worked example's world** is a directory the checker opens for the run, not a path frozen onto
the page. A tool that touches the outside world -- a reader that needs a file already there, a
writer that needs somewhere real to put one -- used to have that path frozen into its example's own
configuration before any check ever entered, because :attr:`WorkedExample.setup` builds its world
too late for the example's own fields to see it. The generated catalog showed a reader
``path: /tmp/etl_worked_example_figures.xlsx``, for exactly the third of it that touches disk. The
``{scratch}`` token is what an example may hold instead: :func:`check_tool` opens a fresh directory
for the run, resolves the token onto a private copy of the configuration, and puts the directory
away again before returning -- the stored configuration a rendered page reads is never touched, only
the copy the run itself sees. What this does not do is watch the filesystem at large: a file written
anywhere other than inside that directory is exactly as invisible as it always was, the same
admission the process-disturbance check below already makes about the environment and the working
directory.
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
    """Set each of ``values`` in the environment for the block, then put back what was there.

    A source's or a sink's declared setup needs exactly this once it stops building its own world
    inside ``help()``: somewhere to resolve a connection from for the duration of one checked
    example, without touching whatever a variable of the same name already meant to the process
    that is doing the checking. "Put back" is not "unset" -- a variable that already carried a real
    value is restored to it, never removed, because the process asking for the check keeps running
    after it. A variable that was absent is left absent again. Both directions hold whether the
    block finishes cleanly or raises, which is the pairing a tool would otherwise have to hand-write
    once per source and once per sink, getting one of the two halves right and the other half not.
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
    """A shared in-memory SQLite database for one worked example, named by ``variable`` for the block.

    ``uri`` names a shared in-memory database, such as ``file:name?mode=memory&cache=shared``.
    ``seed`` builds its tables on an open connection, which is then committed. The environment
    variable ``variable`` holds ``uri`` for the block, so the example's configuration can name it.

    Built in a setup rather than inside ``help()``, which must describe the tool without changing the
    process doing the describing: the catalog page generator calls ``help()`` on every tool, and
    ``check_tool`` refuses a description that opens a connection or exports a variable.

    The seeding connection stays open for the whole block, because a shared in-memory database
    lives only while a connection to it is open. It is closed on the way out in every case, so the
    same example can run again the next time the catalog is checked.
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
    """A snapshot of the two things a description or a declared world must not permanently move."""
    return dict(os.environ), os.getcwd()


def _refuse_disturbance(refusal: str, before: tuple[dict[str, str], str]) -> None:
    """Compare the process against a snapshot taken earlier, and raise ``refusal`` naming what moved.

    Only the environment and the working directory are compared, because those are the two
    disturbances that are both realistic here and cheap to observe from outside a tool's own code.
    A file left on disk, a connection left open, an in-memory database still referenced from a
    module global, a rewritten ``sys.path`` -- none of it is visible from here, and this function
    does not claim otherwise. It catches the mistake this module was built to stop, not every
    mistake a tool's own code could make.
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


# ---------------------------------------------------------------------------
# The scratch directory -- see the module docstring for what this does and does not buy.
#
# ``{scratch}`` is one token with one fixed meaning, not the start of a substitution language: a
# label like ``{total}`` is not this module's business and reaches a tool exactly as written. And it
# is deliberately not ``${scratch}`` -- that spelling means a pipeline parameter nobody bound, and it
# must still be refused by the existing placeholder rule even though it contains the token's own
# name, which is why the pattern below never fires on a ``{scratch}`` preceded by ``$``.
# ---------------------------------------------------------------------------

SCRATCH_TOKEN: Final[str] = "{scratch}"

# The lookbehind is the sharp bit: substituting before ``_check_no_placeholder`` ever ran would
# rescue ``${scratch}``, turning a configuration nobody bound into one that runs -- against a
# directory this module is about to remove.
_SCRATCH_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\$)\{scratch\}")

# The directory the checker opened for whichever example is running right now, if any. A
# ``ContextVar`` rather than a plain module global, so a resolver reached from inside a tool's own
# declared setup sees exactly the directory the current check opened and never one left over from an
# earlier check in the same process.
_current_scratch_directory: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_current_scratch_directory", default=None
)


def scratch_path(value: str) -> Path:
    """``value`` with every ``{scratch}`` token resolved against the directory the checker opened.

    Public so a declared ``setup`` can resolve the same token its example's own configuration holds,
    rather than a second, private way of spelling "the directory this check is using". Raises
    :class:`ConformanceError` when no example is being checked: a setup that cannot be entered
    outside :func:`check_tool` is correct, not a limitation, because there is no directory yet for it
    to resolve against.
    """
    directory = _current_scratch_directory.get()
    if directory is None:
        raise ConformanceError(
            "scratch_path() has nothing to resolve against: it may only be called while check_tool "
            "is checking a worked example whose configuration holds the {scratch} token"
        )
    return Path(_SCRATCH_PATTERN.sub(lambda _: str(directory), value))  # a function: see _resolved_scratch


def _uses_scratch(value: Any) -> bool:
    """Whether ``{scratch}`` (never ``${scratch}``) appears anywhere inside ``value``.

    Recurses into a list or a dict at any depth, mirroring the walk :func:`_check_no_placeholder`
    already does over a configuration for the same reason: the shape this catalog's configurations
    actually have is an ordered list of small objects, and stopping at the top level would miss the
    common case rather than the exotic one.
    """
    if isinstance(value, str):
        return _SCRATCH_PATTERN.search(value) is not None
    if isinstance(value, list):
        return any(_uses_scratch(item) for item in value)
    if isinstance(value, dict):
        return any(_uses_scratch(item) for item in value.values())
    return False


def _resolved_scratch(value: Any, directory: Path) -> Any:
    """``value`` with every ``{scratch}`` token replaced by ``directory``, rebuilt rather than rewritten.

    A list or a dict is rebuilt into a new list or dict rather than mutated in place, because the
    stored configuration -- the one a rendered catalog page reads back -- must still show the token
    afterwards. Writing into the original container would resolve it for whichever check happened to
    run first and leave the page showing a directory that existed for a few milliseconds on somebody
    else's machine.
    """
    if isinstance(value, str):
        # A function, not the string itself: re.sub reads backslashes in a replacement string as
        # escapes, and every Windows path is full of them (C:\Users\... is a bad \U escape).
        return _SCRATCH_PATTERN.sub(lambda _: str(directory), value)
    if isinstance(value, list):
        return [_resolved_scratch(item, directory) for item in value]
    if isinstance(value, dict):
        return {key: _resolved_scratch(item, directory) for key, item in value.items()}
    return value
