"""The ``dantron-etl`` command, and ``dantron-etl new``, which writes a starter project.

The command is the front door for someone who is not a developer: one name, a handful of words
after it. These tests drive it the way that person would, through ``main(argv)``, from inside a
folder, reading what it prints. A starter project made by ``new`` is the fixture for most of
them, so the tests also prove that what ``new`` writes loads, checks and runs.

The two helpers ``view`` leans on are tested directly, so nothing here opens a browser or starts
the viewer's loop: ``choose_port`` against real sockets, ``browser_opener`` against a described
machine rather than this one.
"""

import io
import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

import dantron_etl
from dantron_etl import REGISTRY
from dantron_etl.cli import browser_opener, choose_port, main
from dantron_etl.viewer.project import load_project
from dantron_etl.viewer.scaffold import write_project

COMMAND = "dantron-etl"
SUBCOMMANDS = ("view", "run", "check", "tools", "help", "functions", "catalog", "migrate", "new")
WORKTREE = Path(__file__).resolve().parents[2]
WSL_RELEASE = "5.15.167.4-microsoft-standard-WSL2"
LINUX_RELEASE = "6.8.0-45-generic"


def call(argv: list[str]) -> int:
    """``main(argv)``'s exit code, whether it returns one or argparse raises ``SystemExit``."""
    try:
        return main(argv)
    except SystemExit as stop:
        if stop.code is None:
            return 0
        return stop.code if isinstance(stop.code, int) else 1


def printed(capsys: pytest.CaptureFixture[str]) -> str:
    """Everything printed so far, standard output and standard error together."""
    captured = capsys.readouterr()
    return captured.out + captured.err


def fold(text: str) -> str:
    """``text`` with every run of whitespace made one space, so wrapping does not matter."""
    return " ".join(text.split())


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Path:
    """A starter project made by ``new demo``, with the current folder inside it."""
    monkeypatch.chdir(tmp_path)
    assert call(["new", "demo"]) == 0
    capsys.readouterr()
    root = (tmp_path / "demo").resolve()
    monkeypatch.chdir(root)
    return root


def example_document(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "pipelines" / "example.yaml").read_text(encoding="utf-8"))


def writer_targets(root: Path) -> list[Path]:
    """Where the example's ``write_csv`` steps write, as absolute paths under the project."""
    steps = example_document(root)["steps"]
    targets = [root / step["config"]["path"] for step in steps if step["tool"] == "write_csv"]
    assert targets, "the example pipeline must end in a write_csv step"
    return targets


def writer_ids(root: Path) -> list[str]:
    return [step["id"] for step in example_document(root)["steps"] if step["tool"] == "write_csv"]


def files_under(folder: Path) -> dict[str, bytes]:
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob("*")) if path.is_file()}


# ---------------------------------------------------------------------------
# The command itself: version, help, the catalog words
# ---------------------------------------------------------------------------


class TestCommandBasics:
    def test_version_prints_the_package_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert call(["--version"]) == 0
        assert dantron_etl.__version__ in printed(capsys)

    def test_help_names_the_command_and_every_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert call(["--help"]) == 0
        text = printed(capsys)
        assert text.lstrip().startswith(f"usage: {COMMAND}")
        missing = [word for word in SUBCOMMANDS if not re.search(rf"\b{word}\b", text)]
        assert not missing, f"--help leaves out {missing}"

    @pytest.mark.parametrize("subcommand", [word for word in SUBCOMMANDS if word != "migrate"])
    def test_each_subcommand_has_help_under_the_command_name(
        self, subcommand: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert call([subcommand, "--help"]) == 0
        assert f"{COMMAND} {subcommand}" in printed(capsys)

    def test_tools_lists_every_registered_tool(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert call(["tools"]) == 0
        words = set(printed(capsys).replace(",", " ").split())
        missing = sorted(tool.name for tool in REGISTRY.tools() if tool.name not in words)
        assert not missing, f"tools leaves out {missing}"

    def test_help_prints_a_tool_and_refuses_a_misspelt_one_naming_the_close_match(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert call(["help", "join"]) == 0
        assert fold(REGISTRY.get("join").description) in fold(printed(capsys))
        assert call(["help", "joim"]) == 1
        text = printed(capsys)
        assert "joim" in text and "join" in text.replace("joim", "")

    def test_functions_lists_the_expression_functions(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert call(["functions"]) == 0
        assert re.search(r"\bcoalesce\b", printed(capsys))

    def test_catalog_writes_the_page_and_the_data_into_the_folder_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        out = tmp_path / "catalog_out"
        assert call(["catalog", "--out", str(out)]) == 0
        assert (out / "tool_catalog.html").is_file()
        data = json.loads((out / "tool_catalog.json").read_text(encoding="utf-8"))
        assert {tool["name"] for tool in data["tools"]} == {tool.name for tool in REGISTRY.tools()}

    def test_migrate_hands_its_arguments_to_the_migration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        old = tmp_path / "old.yaml"
        text = (
            "name: Old\nsteps:\n"
            "  - {id: a, tool: read_csv, config: {path: a.csv}}\n"
            "  - {id: b, tool: read_csv, config: {path: b.csv}}\n"
            "  - id: j\n    tool: join\n    config: {left_on: [k], right_on: [k]}\n"
            "    inputs: {left: a.rows, right: b.rows}\n"
        )
        old.write_text(text, encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        # A dry run that finds changes exits 1 in the migration; the command passes that on.
        assert call(["migrate", str(old)]) == 1
        output = printed(capsys)
        assert "left_on" in output and "left_keys" in output
        assert old.read_text(encoding="utf-8") == text
        # Mirror: --write goes through too, and the file changes.
        assert call(["migrate", "--write", str(old)]) == 0
        assert "left_keys" in old.read_text(encoding="utf-8")

    def test_python_dash_m_runs_the_command(self) -> None:
        environment = {**os.environ, "PYTHONPATH": str(WORKTREE / "src")}
        result = subprocess.run(
            [sys.executable, "-m", "dantron_etl", "--version"],
            capture_output=True,
            text=True,
            cwd=WORKTREE,
            env=environment,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert dantron_etl.__version__ in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# dantron-etl new
# ---------------------------------------------------------------------------


class TestNewProject:
    def test_new_writes_the_starter_project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        assert call(["new", "demo"]) == 0
        root = tmp_path / "demo"
        for relative in (
            "etl-project.yaml",
            "pipelines/example.yaml",
            "pipelines/example/orders.csv",
            "Open ETL viewer.cmd",
            "Open ETL viewer.command",
            ".vscode/tasks.json",
        ):
            assert (root / relative).is_file(), f"new did not write {relative}"
        assert (root / "output").is_dir()
        for launcher in ("Open ETL viewer.cmd", "Open ETL viewer.command"):
            assert re.search(r"\bview\b", (root / launcher).read_text(encoding="utf-8")), launcher
        if os.name == "posix":
            assert (root / "Open ETL viewer.command").stat().st_mode & stat.S_IXUSR
        tasks = json.loads((root / ".vscode" / "tasks.json").read_text(encoding="utf-8"))
        found = [task for task in tasks["tasks"] if task.get("label") == "Open ETL viewer"]
        assert len(found) == 1
        task = found[0]
        command_line = " ".join([str(task.get("command", ""))] + [str(arg) for arg in task.get("args", [])])
        assert re.search(r"\bview\b", command_line)

    def test_write_project_returns_the_files_it_wrote_and_nothing_outside_the_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "made"
        written = write_project(folder)
        assert written, "write_project returned no files"
        assert all(Path(path).is_file() for path in written)
        assert all(Path(path).resolve().is_relative_to(folder.resolve()) for path in written)
        assert {Path(path).resolve() for path in written} == {
            path.resolve() for path in folder.rglob("*") if path.is_file()
        }
        assert sorted(path.name for path in tmp_path.iterdir()) == ["made"]

    def test_new_refuses_a_folder_that_is_not_empty_and_leaves_it_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        folder = tmp_path / "taken"
        folder.mkdir()
        (folder / "notes.txt").write_text("mine", encoding="utf-8")
        before = files_under(folder)
        monkeypatch.chdir(tmp_path)
        assert call(["new", "taken"]) == 1
        assert "taken" in printed(capsys)
        assert files_under(folder) == before
        assert sorted(path.name for path in folder.iterdir()) == ["notes.txt"]

    def test_new_accepts_an_existing_empty_folder(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        folder = tmp_path / "empty"
        folder.mkdir()
        monkeypatch.chdir(tmp_path)
        assert call(["new", "empty"]) == 0
        assert (folder / "etl-project.yaml").is_file()
        assert (folder / "pipelines" / "example.yaml").is_file()

    def test_the_new_project_loads_and_its_example_checks_ok_with_its_step_count(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        loaded = load_project(project)
        assert "example" in loaded.pipelines
        entry = loaded.pipelines["example"]
        assert entry.problem is None
        assert entry.steps == len(example_document(project)["steps"])
        assert call(["check", "pipelines/example.yaml"]) == 0
        output = printed(capsys)
        assert "ok" in output.lower()
        assert str(len(example_document(project)["steps"])) in output


# ---------------------------------------------------------------------------
# dantron-etl run
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_is_a_preview_by_default_and_writes_nothing(
        self, project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        targets = writer_targets(project)
        assert call(["run", "example"]) == 0
        output = printed(capsys)
        assert all(not target.exists() for target in targets)
        assert sorted(path.name for path in (project / "output").iterdir()) in ([], [".gitkeep"])
        assert all(step_id in output for step_id in writer_ids(project)), "the held-back writer is not named"

    def test_run_write_with_yes_writes_the_output(self, project: Path) -> None:
        targets = writer_targets(project)
        assert call(["run", "example", "--write", "--yes"]) == 0
        assert all(target.is_file() for target in targets)

    @pytest.mark.parametrize(
        ("typed", "code", "written"),
        [("exampel\n", 1, False), ("example\n", 0, True)],
        ids=["wrong-name-writes-nothing", "right-name-writes"],
    )
    def test_run_write_asks_for_the_pipeline_name_first(
        self, project: Path, monkeypatch: pytest.MonkeyPatch, typed: str, code: int, written: bool
    ) -> None:
        targets = writer_targets(project)
        monkeypatch.setattr(sys, "stdin", io.StringIO(typed))
        assert call(["run", "example", "--write"]) == code
        assert all(target.is_file() is written for target in targets)


# ---------------------------------------------------------------------------
# dantron-etl check, on files the loader refuses
# ---------------------------------------------------------------------------


class TestCheckRefusals:
    def test_check_reports_a_misspelt_tool_and_exits_1(self, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = project / "pipelines" / "misspelt.yaml"
        bad.write_text(
            "name: Misspelt\nsteps:\n"
            "  - {id: read, tool: read_csv, config: {path: pipelines/example/orders.csv}}\n"
            "  - id: out\n    tool: write_csvv\n    config: {path: output/x.csv, overwrite: true}\n"
            "    inputs: {rows: read.rows}\n",
            encoding="utf-8",
        )
        assert call(["check", "pipelines/misspelt.yaml"]) == 1
        output = printed(capsys)
        assert "write_csvv" in output
        assert "ok" not in output.lower().split()

    def test_check_lists_every_dead_end_and_exits_1(self, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
        dead = project / "pipelines" / "dead.yaml"
        dead.write_text(
            "name: Dead ends\nsteps:\n"
            "  - {id: read, tool: read_csv, config: {path: pipelines/example/orders.csv}}\n"
            "  - id: tidy_up\n    tool: sort\n    config: {by: [{column: order_id}]}\n    inputs: {rows: read.rows}\n"
            "  - id: first_few\n    tool: limit_rows\n    config: {count: 3}\n    inputs: {rows: read.rows}\n"
            "  - id: out\n    tool: write_csv\n    config: {path: output/x.csv, overwrite: true}\n"
            "    inputs: {rows: read.rows}\n",
            encoding="utf-8",
        )
        assert call(["check", "pipelines/dead.yaml"]) == 1
        output = printed(capsys)
        assert "tidy_up" in output and "first_few" in output


# ---------------------------------------------------------------------------
# view's helpers: which port, and how to open a browser
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def can_bind(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


@pytest.fixture
def whoami_server() -> Iterator[Any]:
    """``start(root)`` runs a tiny server answering ``/api/whoami`` for ``root``; returns its port."""
    servers: list[ThreadingHTTPServer] = []

    def start(root: Path) -> int:
        body = json.dumps({"app": f"{COMMAND} viewer", "project": str(root)}).encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - the name http.server calls
                if self.path.split("?")[0] == "/api/whoami":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def log_message(self, *args: Any) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server.server_address[1]

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


class TestChoosePort:
    def test_a_free_preferred_port_is_used(self, tmp_path: Path) -> None:
        port = free_port()
        assert choose_port(tmp_path, preferred=port) == (port, False)

    def test_a_viewer_for_the_same_project_is_reused(self, tmp_path: Path, whoami_server: Any) -> None:
        port = whoami_server(tmp_path)
        assert choose_port(tmp_path, preferred=port) == (port, True)

    def test_a_port_held_by_another_project_or_anything_else_moves_to_a_free_one(
        self, tmp_path: Path, whoami_server: Any
    ) -> None:
        other = tmp_path / "other"
        other.mkdir()
        port = whoami_server(other)
        chosen, running = choose_port(tmp_path, preferred=port)
        assert running is False
        assert chosen != port
        assert can_bind(chosen)

        # The same for a plain listener that answers no viewer question at all.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
            holder.bind(("127.0.0.1", 0))
            holder.listen()
            port = holder.getsockname()[1]
            chosen, running = choose_port(tmp_path, preferred=port)
            assert running is False
            assert chosen != port
            assert can_bind(chosen)


class TestBrowserOpener:
    @pytest.mark.parametrize(
        ("environ", "release"),
        [({}, WSL_RELEASE), ({"WSL_DISTRO_NAME": "Ubuntu"}, LINUX_RELEASE)],
        ids=["release-names-microsoft", "distro-variable-set"],
    )
    def test_wsl_opens_through_windows_even_without_a_display(self, environ: dict[str, str], release: str) -> None:
        opener = browser_opener(environ, "linux", release)
        assert isinstance(opener, list) and opener
        assert opener[0] in ("wslview", "explorer.exe")

    def test_windows_has_an_opener(self) -> None:
        opener = browser_opener({}, "win32", "10")
        assert isinstance(opener, list) and opener

    @pytest.mark.parametrize(
        ("environ", "platform", "release", "expected"),
        [
            ({}, "darwin", "23.1.0", ["open"]),
            ({"DISPLAY": ":0"}, "linux", LINUX_RELEASE, ["xdg-open"]),
            ({"WAYLAND_DISPLAY": "wayland-0"}, "linux", LINUX_RELEASE, ["xdg-open"]),
            ({}, "linux", LINUX_RELEASE, None),
        ],
        ids=["macos", "linux-x11", "linux-wayland", "linux-container-without-display"],
    )
    def test_opener_by_platform_and_display(
        self, environ: dict[str, str], platform: str, release: str, expected: list[str] | None
    ) -> None:
        assert browser_opener(environ, platform, release) == expected
