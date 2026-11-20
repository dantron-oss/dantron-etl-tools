"""Which command shows a file in this machine's file manager: viewer/reveal.py.

`how_to_open` is pure: it takes the facts about a machine and returns the command or the reason
there is none. Every row of the table is tested here through it, with made-up machines. The thin
part that runs a command is handed a fake, so no file manager is ever started by a test.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from dantron_etl.viewer.reveal import Launch, Machine, NoLaunch, Ran, Target, carry_out, how_to_open, open_location, spawn

FILE = "/work/project/output/totals.csv"
FOLDER = "/work/project/output"


def machine(platform: str = "linux", env: dict[str, str] | None = None, *programs: str, kernel: str = "") -> Machine:
    return Machine(platform, env or {}, {name: f"/bin/{name}" for name in programs}, kernel)


def there(path: str = FILE) -> Target:
    return Target(path, exists=True, folder_exists=True)


def not_yet(path: str = FILE) -> Target:
    return Target(path, exists=False, folder_exists=True)


WSL_KERNEL = "5.15.167.4-microsoft-standard-WSL2"
WSL_ENV = {"WSL_DISTRO_NAME": "Ubuntu", "WSL_INTEROP": "/run/WSL/1_interop"}
CODE_ENV = {"VSCODE_IPC_HOOK_CLI": "/tmp/vscode-ipc-1.sock"}
DESKTOP_ENV = {"DISPLAY": ":0"}


# ---------------------------------------------------------------------------
# The table, one row at a time
# ---------------------------------------------------------------------------


class TestWindows:
    def test_explorer_opens_with_the_file_selected_and_any_exit_code_counts(self) -> None:
        found = how_to_open(there(r"C:\work\out files\totals.csv"), machine("win32", {}, "explorer"))
        assert isinstance(found, Launch)
        # Two arguments, so a path with a space is quoted on its own and Explorer still reads it.
        assert found.argv == ("/bin/explorer", "/select,", r"C:\work\out files\totals.csv")
        assert found.any_exit

    def test_a_file_not_there_yet_opens_its_folder(self) -> None:
        found = how_to_open(not_yet(r"C:\work\out\totals.csv"), machine("win32", {}, "explorer"))
        assert isinstance(found, Launch)
        assert found.argv == ("/bin/explorer", r"C:\work\out")
        assert "not there yet" in found.done


class TestMac:
    def test_finder_opens_with_the_file_selected(self) -> None:
        found = how_to_open(there(), machine("darwin", {}, "open"))
        assert isinstance(found, Launch)
        assert found.argv == ("/bin/open", "-R", FILE)
        assert not found.any_exit

    def test_a_file_not_there_yet_opens_its_folder(self) -> None:
        found = how_to_open(not_yet(), machine("darwin", {}, "open"))
        assert isinstance(found, Launch)
        assert found.argv == ("/bin/open", FOLDER)


class TestWsl:
    def test_windows_explorer_selects_the_file_by_its_windows_path(self) -> None:
        found = how_to_open(there(), machine("linux", WSL_ENV, "explorer.exe", "wslpath", kernel=WSL_KERNEL))
        assert isinstance(found, Launch)
        assert found.windows_path == ("/bin/wslpath", "-w", FILE)
        assert found.argv == ("/bin/explorer.exe", "/select,")
        assert found.any_exit
        fake = Fake(Ran(0, "C:\\work\\out files\\totals.csv\n"), Ran(1, ""))
        assert carry_out(found, fake).opened
        assert fake.calls[1] == ("/bin/explorer.exe", "/select,", "C:\\work\\out files\\totals.csv")

    def test_a_file_not_there_yet_opens_its_folder_in_explorer(self) -> None:
        found = how_to_open(not_yet(), machine("linux", WSL_ENV, "explorer.exe", "wslpath", kernel=WSL_KERNEL))
        assert isinstance(found, Launch)
        assert found.windows_path == ("/bin/wslpath", "-w", FOLDER)
        assert found.argv == ("/bin/explorer.exe",)

    def test_wsl_comes_before_vs_code(self) -> None:
        facts = machine("linux", {**WSL_ENV, **CODE_ENV}, "explorer.exe", "wslpath", "code", kernel=WSL_KERNEL)
        found = how_to_open(there(), facts)
        assert isinstance(found, Launch)
        assert found.argv[0] == "/bin/explorer.exe"

    @pytest.mark.parametrize(
        ("env", "programs"),
        [
            ({}, ("explorer.exe", "wslpath")),  # a container on a WSL kernel: no WSL session
            (WSL_ENV, ("wslpath",)),  # no Windows programs reachable
            (WSL_ENV, ("explorer.exe",)),  # no way to convert the path
        ],
        ids=["no-wsl-session", "no-explorer", "no-wslpath"],
    )
    def test_a_wsl_kernel_alone_is_not_wsl(self, env: dict[str, str], programs: tuple[str, ...]) -> None:
        found = how_to_open(there(), machine("linux", env, *programs, kernel=WSL_KERNEL))
        assert not (isinstance(found, Launch) and "explorer.exe" in found.argv[0])

    def test_wsl_settings_on_a_plain_linux_kernel_are_not_wsl(self) -> None:
        found = how_to_open(there(), machine("linux", WSL_ENV, "explorer.exe", "wslpath", kernel="6.8.0-generic"))
        assert isinstance(found, NoLaunch)


class TestVsCode:
    def test_the_remote_cli_opens_the_file_in_the_window_already_open(self) -> None:
        found = how_to_open(there(), machine("linux", CODE_ENV, "code"))
        assert isinstance(found, Launch)
        assert found.argv == ("/bin/code", "-r", FILE)

    def test_it_wins_inside_a_container_on_a_wsl_kernel(self) -> None:
        facts = machine("linux", {**CODE_ENV, **DESKTOP_ENV}, "code", "xdg-open", kernel=WSL_KERNEL)
        found = how_to_open(there(), facts)
        assert isinstance(found, Launch)
        assert found.argv[0] == "/bin/code"

    def test_a_workbook_is_opened_and_the_reader_is_told_it_cannot_show_as_text(self) -> None:
        found = how_to_open(there("/work/project/output/pack.xlsx"), machine("linux", CODE_ENV, "code"))
        assert isinstance(found, Launch)
        assert found.argv == ("/bin/code", "-r", "/work/project/output/pack.xlsx")
        assert "cannot show" in found.done
        assert "cannot show" not in how_to_open(there(), machine("linux", CODE_ENV, "code")).done  # type: ignore[union-attr]

    def test_a_file_not_there_yet_opens_nothing_and_names_the_folder(self) -> None:
        found = how_to_open(not_yet(), machine("linux", CODE_ENV, "code"))
        assert isinstance(found, NoLaunch)
        assert FOLDER in found.reason

    @pytest.mark.parametrize(
        ("env", "programs"),
        [({}, ("code",)), (CODE_ENV, ()), ({"VSCODE_IPC_HOOK_CLI": ""}, ("code",))],
        ids=["no-hook", "no-code", "empty-hook"],
    )
    def test_it_needs_both_the_cli_and_the_terminal_hook(self, env: dict[str, str], programs: tuple[str, ...]) -> None:
        assert isinstance(how_to_open(there(), machine("linux", env, *programs)), NoLaunch)


class TestLinuxDesktop:
    def test_the_file_manager_is_asked_to_show_the_item_with_the_folder_as_fallback(self) -> None:
        found = how_to_open(there("/work/a b,c/it's.csv"), machine("linux", DESKTOP_ENV, "gdbus", "xdg-open"))
        assert isinstance(found, Launch)
        assert found.argv[:2] == ("/bin/gdbus", "call")
        assert "org.freedesktop.FileManager1.ShowItems" in found.argv
        assert found.argv[found.argv.index("--timeout") + 1] == "3"  # a bus that hangs falls back
        assert "['file:///work/a%20b%2Cc/it%27s.csv']" in found.argv
        assert found.otherwise is not None
        assert found.otherwise.argv == ("/bin/xdg-open", "/work/a b,c")

    def test_without_dbus_the_folder_is_opened(self) -> None:
        found = how_to_open(there(), machine("linux", {"WAYLAND_DISPLAY": "wayland-0"}, "xdg-open"))
        assert isinstance(found, Launch)
        assert found.argv == ("/bin/xdg-open", FOLDER)
        assert found.otherwise is None

    def test_a_file_not_there_yet_opens_its_folder(self) -> None:
        found = how_to_open(not_yet(), machine("linux", DESKTOP_ENV, "gdbus", "xdg-open"))
        assert isinstance(found, Launch)
        assert found.argv == ("/bin/xdg-open", FOLDER)
        assert "not there yet" in found.done

    def test_no_display_means_no_desktop(self) -> None:
        assert isinstance(how_to_open(there(), machine("linux", {}, "gdbus", "xdg-open")), NoLaunch)


class TestNothingToOpenWith:
    @pytest.mark.parametrize(
        "facts",
        [machine("linux"), machine("darwin"), machine("win32"), machine("linux", {}, "xdg-open", "gdbus")],
        ids=["bare-linux", "mac-without-open", "windows-without-explorer", "linux-without-a-display"],
    )
    def test_it_says_so_plainly(self, facts: Machine) -> None:
        found = how_to_open(there(), facts)
        assert isinstance(found, NoLaunch)
        assert "nothing was opened" in found.reason

    def test_a_desktop_with_dbus_alone_is_not_used_without_a_fallback(self) -> None:
        assert isinstance(how_to_open(there(), machine("linux", DESKTOP_ENV, "gdbus")), NoLaunch)
        assert isinstance(how_to_open(not_yet(), machine("linux", DESKTOP_ENV, "gdbus")), NoLaunch)

    def test_a_folder_that_is_not_there_either_opens_nothing(self) -> None:
        found = how_to_open(Target(FILE, exists=False, folder_exists=False), machine("darwin", {}, "open"))
        assert isinstance(found, NoLaunch)
        assert FOLDER in found.reason

    def test_a_relative_path_is_never_handed_to_a_program(self) -> None:
        found = how_to_open(there("output/totals.csv"), machine("darwin", {}, "open"))
        assert isinstance(found, NoLaunch)


class TestNoOptionsFromFileNames:
    @pytest.mark.parametrize(
        "facts",
        [
            machine("darwin", {}, "open"),
            machine("linux", CODE_ENV, "code"),
            machine("linux", DESKTOP_ENV, "xdg-open"),
            machine("linux", WSL_ENV, "explorer.exe", "wslpath", kernel=WSL_KERNEL),
        ],
        ids=["mac", "vs-code", "desktop", "wsl"],
    )
    def test_a_file_named_like_an_option_is_passed_as_its_full_path(self, facts: Machine) -> None:
        found = how_to_open(there("/work/project/-n.csv"), facts)
        assert isinstance(found, Launch)
        every = [*found.argv[1:], *(found.windows_path or ())[1:]]
        assert "-n.csv" not in every
        assert not any(arg.startswith("-n") for arg in every)


# ---------------------------------------------------------------------------
# Carrying a launch out, with a fake in place of the real programs
# ---------------------------------------------------------------------------


class Fake:
    """Records each command and answers from a list, in order."""

    def __init__(self, *answers: Ran | Exception) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> Ran:
        assert isinstance(argv, list | tuple) and all(isinstance(arg, str) for arg in argv)
        assert timeout <= 10
        self.calls.append(tuple(argv))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class TestCarryOut:
    def test_a_zero_exit_is_done(self) -> None:
        fake = Fake(Ran(0, ""))
        result = carry_out(Launch(("/bin/open", "-R", FILE), "Opened Finder."), fake)
        assert result.opened and result.message == "Opened Finder."
        assert fake.calls == [("/bin/open", "-R", FILE)]

    def test_a_non_zero_exit_is_a_failure_with_what_the_program_said(self) -> None:
        result = carry_out(Launch(("/bin/open", "-R", FILE), "Opened Finder."), Fake(Ran(1, "no such app")))
        assert not result.opened
        assert "no such app" in result.message

    def test_explorer_counts_as_done_whatever_it_answers(self) -> None:
        result = carry_out(Launch(("/bin/explorer", "/select,x"), "Opened Explorer.", any_exit=True), Fake(Ran(1, "")))
        assert result.opened

    def test_a_program_still_running_after_the_wait_counts_as_started(self) -> None:
        assert carry_out(Launch(("/bin/xdg-open", FOLDER), "Opened."), Fake(Ran(None, ""))).opened

    def test_the_windows_path_is_added_as_the_last_argument(self) -> None:
        fake = Fake(Ran(0, "C:\\work\\totals.csv\r\n"), Ran(1, ""))
        launch = Launch(("/bin/explorer.exe", "/select,"), "Opened.", any_exit=True, windows_path=("/bin/wslpath", "-w", FILE))
        assert carry_out(launch, fake).opened
        assert fake.calls == [("/bin/wslpath", "-w", FILE), ("/bin/explorer.exe", "/select,", "C:\\work\\totals.csv")]

    def test_a_path_that_will_not_convert_launches_nothing(self) -> None:
        fake = Fake(Ran(1, "wslpath: bad path"))
        launch = Launch(("/bin/explorer.exe", "/select,"), "Opened.", any_exit=True, windows_path=("/bin/wslpath", "-w", FILE))
        result = carry_out(launch, fake)
        assert not result.opened
        assert len(fake.calls) == 1

    def test_the_fallback_runs_when_the_first_fails(self) -> None:
        fake = Fake(Ran(1, "no file manager"), Ran(0, ""))
        launch = Launch(("/bin/gdbus", "call"), "Shown.", otherwise=Launch(("/bin/xdg-open", FOLDER), "Folder."))
        result = carry_out(launch, fake)
        assert result.opened and result.message == "Folder."
        assert len(fake.calls) == 2

    def test_a_program_that_cannot_start_is_reported_not_raised(self) -> None:
        result = carry_out(Launch(("/bin/open", FILE), "Opened."), Fake(FileNotFoundError("/bin/open")))
        assert not result.opened
        assert "could not" in result.message

    def test_open_location_says_why_when_there_is_no_command(self) -> None:
        fake = Fake()
        result = open_location(there(), machine("linux"), fake)
        assert not result.opened
        assert fake.calls == []


class TestSpawn:
    """The one function that starts a program, tried with Python itself instead of a file manager."""

    def test_it_returns_the_exit_code_and_what_the_program_printed(self) -> None:
        ran = spawn([sys.executable, "-c", "import sys; print('said'); sys.exit(3)"], 10)
        assert ran == Ran(3, "said")

    def test_a_program_still_running_after_the_wait_is_left_running(self) -> None:
        started = time.monotonic()
        ran = spawn([sys.executable, "-c", "import time; time.sleep(30)"], 0.5)
        assert ran.code is None
        assert time.monotonic() - started < 10

    def test_text_that_looks_like_a_shell_command_stays_one_argument(self, tmp_path: Path) -> None:
        marker = tmp_path / "made"
        ran = spawn([sys.executable, "-c", "import sys; print(len(sys.argv))", f"x; touch {marker}"], 10)
        assert ran == Ran(0, "2")
        assert not marker.exists()
