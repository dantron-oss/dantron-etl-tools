"""Where the project is, answered from the caller's location or refused by name.

Every script that ships beside this library writes a file somewhere and reads a file from somewhere,
and all of those places hang off one directory: the root of whichever project the caller belongs to.
This module is the only place that question is answered, because it has exactly one dangerous wrong
answer and answering it in four scripts means writing that wrong answer four times.

**The wrong answer, measured.** A library that resolves the root by walking up from its own file::

    ROOT = Path(__file__).resolve().parents[2]

is right while the package sits in the ``src/`` of the repository it belongs to, and silently wrong
the moment the same line ships inside an installed distribution::

    /client/.venv/lib/python3.13/site-packages/<pkg>/paths.py  ->  parents[2]
    /client/.venv/lib/python3.13

Nothing raises. That directory exists, is readable, and is not the project, so every path joined onto
it afterwards is a real path to the wrong place. The project this library was extracted from paid for
that failure twice in one day -- a workbook built from another checkout's pipeline file, and a run
that read a sixteen-step flow where a six-hundred-and-seventy-two-step one was meant -- and both
looked entirely ordinary going past. So what is designed for here is *refusing*, not resolving.

Three designs were on the table, and the reason the other two lost is worth keeping, because the
first one is shorter than the one that won and somebody will want it back:

*Infer from the library's own* ``__file__``. The trap above, sited where it does the most damage: one
wrong answer, computed once at import, reused by every consumer, never raised. It is excluded by
construction here rather than by discipline -- :func:`project_root` takes the caller's location
because this library has no honest opinion about where the caller's project is, and there is no
module-level constant for it to be baked into.

*Environment variable, else the current working directory.* Right for a command-line tool that a
person runs from inside a project; wrong for a library. The working directory of a long-running
process -- a daemon, a notebook kernel, a scheduler -- is an accident of how it was launched, and it
can change underneath a running process. It also fails in the direction that hurts: it answers
plausibly rather than refusing.

*Environment variable, else search upward from a start the caller names, else refuse.* What this
module does. Right in a repository, right in a linked worktree, right when the caller is a loose
script and when it is an installed module, and -- the reason it wins -- **refusable**: when no marker
is found there is nothing plausible to return, so the outcome is a named exception instead of a path.
The cost is that every caller passes ``__file__``. That cost is bought deliberately: one explicit
argument at each call site, against a class of defect that has already cost two rebuilds.

The environment variable is kept, under this library's own name, because a root override is how a
process is told which checkout it belongs to when several are on disk at once. It is checked *before*
the search, not after it: the case it exists for is the one where the search would otherwise succeed
with the wrong answer. It is also validated like any other candidate -- a variable pointing at a
directory that is not a project is a typo in a deployment, and trusting it because somebody set it on
purpose is how a stale checkout goes unnoticed for a day.

Markers are matched as files *or* directories, which is not a detail: a linked worktree's ``.git`` is
a file containing a ``gitdir:`` line, so an implementation testing ``is_dir()`` refuses every checkout
that is a worktree.

A folder holding ``etl-project.yaml`` is a project even with no ``pyproject.toml`` and no ``.git``.

    from <this package>.paths import project_root

    ROOT = project_root(__file__)
"""

import os
from pathlib import Path

#: The variable a container, a CI job or a shell sets to name the project explicitly. It wins over
#: anything the search would find, and it is spelled out in every refusal below because it is the one
#: move available to a reader who is looking at one.
ROOT_ENV_VAR = "ETL_PROJECT_ROOT"

#: What makes a directory a project. Either of the first two alone is wrong somewhere real: a source
#: checkout that has not been packaged has no ``pyproject.toml``, and an exported or vendored tree has
#: no ``.git``. A folder holding the project settings file is a project too.
ROOT_MARKERS = ("pyproject.toml", ".git", "etl-project.yaml")


class ProjectRootNotFound(RuntimeError):  # noqa: N818 -- the sentence a reader wants beats the suffix
    """No directory above the start carries a project marker, and no override said where to look.

    Raised rather than returning a plausible directory, because a plausible directory is the failure
    this module exists to prevent. Descends from ``RuntimeError`` so a caller that wraps everything
    still catches it, and is named so a caller that wants to handle only this can.
    """


def _is_a_project(directory: Path) -> bool:
    """A directory carrying any marker, as a file or a directory -- see the note on worktrees above."""
    return any((directory / marker).exists() for marker in ROOT_MARKERS)


def project_root(start: str | os.PathLike[str]) -> Path:
    """The root of the project the caller belongs to, as an absolute path.

    ``start`` is where the caller is, normally its own ``__file__``. It has no default: a default
    would have to be either this module's location or the working directory, and those are the two
    designs the module docstring rejects. A file start is read as the directory holding it.

    The override named by :data:`ROOT_ENV_VAR` is consulted first and must itself be a project, so a
    variable pointing somewhere that is not one fails here rather than three reads later.

    Raises:
        ProjectRootNotFound: if the override does not name a project, or if no marker in
            :data:`ROOT_MARKERS` is found at the start or above it. The message names where the search
            began, what it was looking for, and the variable that settles it.
    """
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        named = Path(override).expanduser()
        if not named.is_dir() or not _is_a_project(named):
            raise ProjectRootNotFound(
                f"{ROOT_ENV_VAR}={override} does not name a project: no {' or '.join(ROOT_MARKERS)} "
                f"in that directory (it may not exist at all). Point {ROOT_ENV_VAR} at a project "
                f"root, or unset it to search upward from the caller instead."
            )
        return named.resolve()

    begin = Path(start).expanduser().resolve()
    if begin.is_file():
        begin = begin.parent
    for directory in (begin, *begin.parents):
        if _is_a_project(directory):
            return directory

    raise ProjectRootNotFound(
        f"no project root at or above {begin}: nothing there carries "
        f"{' or '.join(ROOT_MARKERS)}. If this code is installed rather than sitting in a checkout, "
        f"there is nothing above it to find -- set {ROOT_ENV_VAR} to the project root instead."
    )
