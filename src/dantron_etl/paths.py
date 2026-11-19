"""Find the root folder of the caller's project, or refuse by name.

Call ``project_root(__file__)`` with your own file. It returns the nearest folder at or above that
file holding ``pyproject.toml``, ``.git`` or ``etl-project.yaml``. Set ``ETL_PROJECT_ROOT`` to name
the root outright. When nothing is found it raises :class:`ProjectRootNotFound` rather than guess.

    from <this package>.paths import project_root

    ROOT = project_root(__file__)

Why: docs/design/paths.md
"""

import os
from pathlib import Path

#: The environment variable that names the project root outright. It wins over the search.
ROOT_ENV_VAR = "ETL_PROJECT_ROOT"

#: A folder holding any of these, as a file or a folder, is a project root.
#: Why: docs/design/paths.md#what-marks-a-project
ROOT_MARKERS = ("pyproject.toml", ".git", "etl-project.yaml")


class ProjectRootNotFound(RuntimeError):  # noqa: N818 -- the sentence a reader wants beats the suffix
    """Raised when no project root can be found, or ``ETL_PROJECT_ROOT`` names a folder that is not one.

    It is a ``RuntimeError``, so a caller that catches everything still catches it.
    """


def _is_a_project(directory: Path) -> bool:
    """Say whether the folder holds any of the markers, as a file or a folder."""
    return any((directory / marker).exists() for marker in ROOT_MARKERS)


def project_root(start: str | os.PathLike[str]) -> Path:
    """Return the absolute path of the project root the caller belongs to.

    Args:
        start: where the caller is, normally its own ``__file__``. A file means the folder holding
            it. There is no default (why: docs/design/paths.md#the-caller-says-where-it-is).

    If :data:`ROOT_ENV_VAR` is set, its folder is returned, provided it is a project. Otherwise the
    search goes from ``start`` upward and returns the first folder holding a marker.

    Raises:
        ProjectRootNotFound: the variable names a folder that is not a project, or no folder at or
            above ``start`` holds a marker. The message says where the search began.
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
