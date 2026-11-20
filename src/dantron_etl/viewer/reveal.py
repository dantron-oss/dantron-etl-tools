"""Showing a file in this machine's file manager, for the viewer's "Open location" button.

`how_to_open` decides the command from plain facts about the machine, so every case can be tested
without a desktop. `carry_out` runs it: a list of arguments, never a shell, with a short wait.

Why: docs/design/viewer.md#looking-at-a-file-a-step-names
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath

__all__ = [
    "HELPERS",
    "Launch",
    "Machine",
    "NoLaunch",
    "Opened",
    "Ran",
    "Target",
    "carry_out",
    "how_to_open",
    "open_location",
    "spawn",
    "this_machine",
]

#: The programs the table below may use. Each is looked up on PATH once per request.
HELPERS = ("explorer", "explorer.exe", "wslpath", "open", "code", "gdbus", "xdg-open")

#: How long to wait for a program before taking it as started. A file manager may never return.
WAIT_SECONDS = 5.0

#: Files VS Code shows as a "cannot display" tab rather than as text.
_NOT_TEXT = frozenset({".xlsx", ".xlsm", ".xls", ".xlsb", ".ods", ".parquet", ".zip", ".pdf"})
_WORKBOOKS = frozenset({".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"})

_NOTHING = (
    "This computer has no file manager the viewer can open, so nothing was opened. "
    "Copy the path, or use Preview to see what the file holds."
)


@dataclass(frozen=True)
class Machine:
    """The facts the choice of command rests on.

    ``platform`` is ``sys.platform``. ``programs`` maps each helper found on PATH to its full path.
    ``kernel`` is the Linux kernel's release name, which says whether this is WSL.
    """

    platform: str
    environ: Mapping[str, str]
    programs: Mapping[str, str]
    kernel: str = ""


@dataclass(frozen=True)
class Target:
    """The file to show: its full path, whether it is there, and whether its folder is."""

    path: str
    exists: bool
    folder_exists: bool


@dataclass(frozen=True)
class Launch:
    """A command to run, and what to tell the reader once it worked.

    ``any_exit``: Explorer answers 1 even when it worked, so its exit code means nothing.
    ``windows_path``: a command that prints a path in Windows form; its answer is added to the end
    of the last argument. ``otherwise``: what to try when this command fails.
    """

    argv: tuple[str, ...]
    done: str
    any_exit: bool = False
    windows_path: tuple[str, ...] | None = None
    otherwise: Launch | None = None


@dataclass(frozen=True)
class NoLaunch:
    """No command fits. ``reason`` says why, in words for the reader."""

    reason: str


@dataclass(frozen=True)
class Ran:
    """What a program did: its exit code, or None when it was still running after the wait."""

    code: int | None
    output: str


@dataclass(frozen=True)
class Opened:
    """What the reader is told: whether something opened, and what."""

    opened: bool
    message: str
    command: tuple[str, ...] | None = None


Spawn = Callable[[Sequence[str], float], Ran]


# ---------------------------------------------------------------------------
# Which command, for which machine
# ---------------------------------------------------------------------------


def _pure(platform: str, path: str) -> PurePath:
    return PureWindowsPath(path) if platform == "win32" else PurePosixPath(path)


def _is_wsl(machine: Machine) -> bool:
    """WSL: a WSL kernel, a WSL session, and the two Windows helpers reachable.

    A container on a WSL machine has the kernel but not the session, and no Explorer to open.
    """
    kernel = machine.kernel.lower()
    session = bool(machine.environ.get("WSL_DISTRO_NAME") or machine.environ.get("WSL_INTEROP"))
    return (
        machine.platform.startswith("linux")
        and ("microsoft" in kernel or "wsl" in kernel)
        and session
        and "explorer.exe" in machine.programs
        and "wslpath" in machine.programs
    )


def _has_vs_code(machine: Machine) -> bool:
    """A terminal inside VS Code: its ``code`` command and the hook its terminal sets."""
    return "code" in machine.programs and bool(machine.environ.get("VSCODE_IPC_HOOK_CLI"))


def _has_desktop(machine: Machine) -> bool:
    return machine.platform.startswith(("linux", "freebsd", "openbsd", "netbsd")) and bool(
        machine.environ.get("DISPLAY") or machine.environ.get("WAYLAND_DISPLAY")
    )


def _vs_code(target: Target, folder: str, programs: Mapping[str, str]) -> Launch | NoLaunch:
    if not target.exists:
        return NoLaunch(
            "The file is not there yet. VS Code can open a folder only by switching this window to it, "
            f"so nothing was opened. The file will be written in {folder}."
        )
    where = "If the file is inside the folder VS Code has open, the file panel on the left shows where it is."
    suffix = PurePosixPath(target.path).suffix.lower()
    if suffix in _NOT_TEXT:
        kind = "a workbook" if suffix in _WORKBOOKS else "this kind of file"
        done = (
            f"Opened the file in VS Code. VS Code cannot show {kind} as text, so its tab says so. "
            f"{where} Use Preview to see what it holds."
        )
    else:
        done = f"Opened the file in VS Code. {where}"
    return Launch((programs["code"], "-r", target.path), done)


def _desktop(target: Target, folder: str, programs: Mapping[str, str]) -> Launch | NoLaunch:
    opener = programs.get("xdg-open")
    fallback = (
        Launch((opener, folder), f"Opened the file's folder. Look for {PurePosixPath(target.path).name} there.")
        if opener
        else None
    )
    if not target.exists:
        if opener is None:
            return NoLaunch(_NOTHING)
        return Launch((opener, folder), "The file is not there yet, so its folder was opened instead.")
    if "gdbus" in programs:
        uri = PurePosixPath(target.path).as_uri()
        ask = (
            programs["gdbus"], "call", "--session",
            "--dest", "org.freedesktop.FileManager1",
            "--object-path", "/org/freedesktop/FileManager1",
            "--method", "org.freedesktop.FileManager1.ShowItems",
            f"['{uri}']", "",
        )  # fmt: skip
        return Launch(ask, "Opened the file manager with the file selected.", otherwise=fallback)
    return fallback or NoLaunch(_NOTHING)


def how_to_open(target: Target, machine: Machine) -> Launch | NoLaunch:
    """The command that shows ``target`` in this machine's file manager, or why there is none.

    In this order: Windows Explorer, macOS Finder, Explorer from WSL, VS Code from its own terminal,
    a Linux desktop's file manager. A file not there yet has its folder opened instead.
    """
    path = _pure(machine.platform, target.path)
    if not path.is_absolute():
        return NoLaunch(f"The path {target.path} is not a full path, so nothing was opened.")
    folder = str(path.parent)
    if not target.exists and not target.folder_exists:
        return NoLaunch(f"The file is not there yet, and neither is its folder, {folder}. Nothing was opened.")
    programs = machine.programs
    missing = "The file is not there yet, so {app} was opened at its folder."

    if machine.platform == "win32":
        if "explorer" not in programs:
            return NoLaunch(_NOTHING)
        if target.exists:
            return Launch((programs["explorer"], f"/select,{target.path}"), "Opened Explorer with the file selected.", True)
        return Launch((programs["explorer"], folder), missing.format(app="Explorer"), True)

    if machine.platform == "darwin":
        if "open" not in programs:
            return NoLaunch(_NOTHING)
        if target.exists:
            return Launch((programs["open"], "-R", target.path), "Opened Finder with the file selected.")
        return Launch((programs["open"], folder), missing.format(app="Finder"))

    if _is_wsl(machine):
        convert = (programs["wslpath"], "-w")
        if target.exists:
            return Launch(
                (programs["explorer.exe"], "/select,"),
                "Opened Windows Explorer with the file selected.",
                any_exit=True,
                windows_path=(*convert, target.path),
            )
        return Launch(
            (programs["explorer.exe"], ""),
            missing.format(app="Windows Explorer"),
            any_exit=True,
            windows_path=(*convert, folder),
        )

    if _has_vs_code(machine):
        return _vs_code(target, folder, programs)

    if _has_desktop(machine):
        return _desktop(target, folder, programs)

    return NoLaunch(_NOTHING)


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def this_machine() -> Machine:
    """The facts about the machine the viewer runs on."""
    kernel = ""
    if sys.platform.startswith("linux"):
        try:
            kernel = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf8").strip()
        except OSError:
            kernel = ""
    found = {name: shutil.which(name) for name in HELPERS}
    return Machine(sys.platform, dict(os.environ), {name: path for name, path in found.items() if path}, kernel)


def spawn(argv: Sequence[str], timeout: float) -> Ran:
    """Run ``argv`` with no shell. Wait up to ``timeout`` seconds; a program still running is left to run."""
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(  # noqa: S603 -- an argument list from the table above, no shell
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=os.name == "posix",
        )
        try:
            code = process.wait(timeout)
        except subprocess.TimeoutExpired:
            return Ran(None, "")
        output.seek(0)
        return Ran(code, output.read(4000).decode("utf8", errors="replace").strip())


def _run(run: Spawn, argv: Sequence[str]) -> Ran | str:
    """The program's answer, or a sentence saying why it could not be started."""
    try:
        return run(argv, WAIT_SECONDS)
    except (OSError, subprocess.SubprocessError) as error:
        return f"The viewer could not start {Path(argv[0]).name}: {error}"


def carry_out(launch: Launch, run: Spawn = spawn) -> Opened:
    """Run ``launch``, then its fallback if it fails, and say what happened."""
    argv = list(launch.argv)
    if launch.windows_path is not None:
        converted = _run(run, launch.windows_path)
        if isinstance(converted, str):
            return Opened(False, converted, launch.windows_path)
        if converted.code != 0 or not converted.output.strip():
            told = f" It said: {converted.output}" if converted.output else ""
            return Opened(False, f"The viewer could not work out the file's Windows path.{told}", launch.windows_path)
        argv[-1] += converted.output.strip()
    ran = _run(run, argv)
    if isinstance(ran, Ran) and (ran.code is None or ran.code == 0 or launch.any_exit):
        return Opened(True, launch.done, tuple(argv))
    if launch.otherwise is not None:
        return carry_out(launch.otherwise, run)
    if isinstance(ran, str):
        return Opened(False, ran, tuple(argv))
    told = f" It said: {ran.output}" if ran.output else ""
    return Opened(False, f"{Path(argv[0]).name} did not open the file (it stopped with code {ran.code}).{told}", tuple(argv))


def open_location(target: Target, machine: Machine | None = None, run: Spawn | None = None) -> Opened:
    """Show ``target`` in this machine's file manager, or say why that cannot be done."""
    found = how_to_open(target, machine if machine is not None else this_machine())
    if isinstance(found, NoLaunch):
        return Opened(False, found.reason)
    return carry_out(found, run if run is not None else spawn)
