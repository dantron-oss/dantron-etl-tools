"""Writing a starter project: what ``new <folder>`` makes.

The files are real text files in the ``scaffold/`` folder beside this module. A name part starting with
``dot-`` is written with a leading dot instead (``dot-gitignore`` becomes ``.gitignore``), so the
templates themselves are never hidden or read by git as settings.

Text files may hold ``@@command@@`` and ``@@package@@``, filled in with this command's name and
this package's name. The two launchers also hold ``@@fallback@@``: what to run when the command is
not on the PATH.
"""

from __future__ import annotations

import shlex
import sys
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

from .project_settings import ProjectError

__all__ = ["write_project"]

PACKAGE = __package__.split(".")[0]
COMMAND = PACKAGE.replace("_", "-")
TEMPLATES = "scaffold"
DOT_PREFIX = "dot-"
WINDOWS_LAUNCHER = "Open ETL viewer.cmd"
MAC_LAUNCHER = "Open ETL viewer.command"
NOT_FOUND = (
    "The @@command@@ command was not found. Install the package that provides it, "
    "then double-click this file again."
)


def write_project(folder: Path | str) -> list[Path]:
    """Write a starter project into ``folder`` and return the files written.

    The folder may be missing or empty. A folder with anything in it is refused and left alone.
    """
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        raise ProjectError(f"{folder} is a file. Choose a name for a new folder.")
    if folder.is_dir() and any(folder.iterdir()):
        raise ProjectError(f"{folder} already has files in it. Choose a new folder name, or an empty folder.")
    written: list[Path] = []
    for relative, source in _templates(resources.files(__package__) / TEMPLATES, ()):
        target = folder.joinpath(*relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = _filled(source.read_text(encoding="utf-8"), target.name)
        if target.suffix == ".cmd":
            text = text.replace("\n", "\r\n")
        target.write_bytes(text.encode("utf-8"))
        if target.suffix == ".command":
            target.chmod(target.stat().st_mode | 0o111)
        written.append(target)
    return written


def _templates(folder: Traversable, parts: tuple[str, ...]) -> list[tuple[tuple[str, ...], Traversable]]:
    """Every template file under ``folder``, with the path parts it is written to."""
    found: list[tuple[tuple[str, ...], Traversable]] = []
    for item in sorted(folder.iterdir(), key=lambda item: item.name):
        if item.name == "__pycache__":
            continue
        name = "." + item.name.removeprefix(DOT_PREFIX) if item.name.startswith(DOT_PREFIX) else item.name
        if item.is_dir():
            found += _templates(item, (*parts, name))
        else:
            found.append(((*parts, name), item))
    return found


def _filled(text: str, name: str) -> str:
    """The template text with its markers filled in."""
    if name == WINDOWS_LAUNCHER:
        text = text.replace("@@fallback@@", _windows_fallback())
    elif name == MAC_LAUNCHER:
        text = text.replace("@@fallback@@", _mac_fallback())
    return text.replace("@@command@@", COMMAND).replace("@@package@@", PACKAGE)


def _windows_fallback() -> str:
    """The Python that made this project, when it was made on Windows; else a plain message."""
    if sys.platform == "win32" and sys.executable:
        return f'"{sys.executable.replace("%", "%%")}" -m {PACKAGE} view'
    return f"echo {NOT_FOUND}"


def _mac_fallback() -> str:
    """The Python that made this project, when it was made on a Mac; else a plain message."""
    if sys.platform == "darwin" and sys.executable:
        return f"    {shlex.quote(sys.executable)} -m {PACKAGE} view"
    return f'    echo "{NOT_FOUND}"'
