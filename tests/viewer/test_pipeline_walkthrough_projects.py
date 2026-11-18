"""Running a project by name: `make run p=<name>`, which is scripts/pipeline_walkthrough_page.py <name>.

A name stands for three things: the flow in pipelines/<name>.yaml, its inputs in pipelines/<name>/,
and its settings in pipelines/<name>/settings.yaml. These tests cover finding and checking those,
saving a run so the viewer can serve it, and starting or reusing the viewer.
"""

from __future__ import annotations

import json
import socket
import urllib.request
from pathlib import Path

import pytest

from dantron_etl.paths import project_root
from dantron_etl.pipeline import PipelineError
from dantron_etl.preview import MANIFEST, PAGE_NAME, PANEL_MANIFEST, discover, serve
from dantron_etl.runner import StepFailedError

PROJECT_ROOT = project_root(__file__)

import scripts.pipeline_walkthrough_page as runner  # noqa: E402 -- the module the tests change in place
from scripts.pipeline_walkthrough_page import (  # noqa: E402
    ProjectError,
    load_project,
    projects,
    write_run,
)

_FIXTURE = PROJECT_ROOT / "pipelines" / "fixture.yaml"


@pytest.fixture(autouse=True)
def _runs_in_the_sandbox(fixture_flow_sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every run here starts in a scratch folder, so the fixture flow's sinks never write into the checkout."""
    monkeypatch.chdir(fixture_flow_sandbox)


def _project(pipelines: Path, name: str, flow: str | None = None, settings: str | None = None) -> None:
    """A project under ``pipelines``: a flow file, and a settings file when one is given."""
    pipelines.mkdir(parents=True, exist_ok=True)
    (pipelines / f"{name}.yaml").write_text(flow if flow is not None else _FIXTURE.read_text(encoding="utf8"), encoding="utf8")
    if settings is not None:
        (pipelines / name).mkdir(exist_ok=True)
        (pipelines / name / "settings.yaml").write_text(settings, encoding="utf8")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class TestFindingAProject:
    def test_the_sample_project_reads_its_rules_and_its_focus(self) -> None:
        project = load_project("fixture")
        assert project.pipeline == _FIXTURE
        assert project.rules == {"revenue_type": "REVENUE"}
        assert project.focus.label == "party" and project.focus.identity == ("party",)

    @pytest.mark.parametrize("name", ["../fixture", "a/b", "a\\b", "", ".hidden", "two words", "-x"])
    def test_a_name_that_could_point_outside_pipelines_is_refused(self, name: str) -> None:
        with pytest.raises(ProjectError, match="is not a project name"):
            load_project(name)

    def test_a_missing_pipeline_names_the_file_and_lists_the_projects_there(self, tmp_path: Path) -> None:
        _project(tmp_path, "sales", settings="focus: party\n")
        _project(tmp_path, "costs", settings="focus: party\n")
        with pytest.raises(ProjectError) as caught:
            load_project("slaes", tmp_path)
        message = str(caught.value)
        assert str(tmp_path / "slaes.yaml") in message
        assert "costs, sales" in message
        assert projects(tmp_path) == ["costs", "sales"]

    def test_a_missing_settings_file_names_the_file_and_shows_what_to_put_in_it(self, tmp_path: Path) -> None:
        _project(tmp_path, "sales")
        with pytest.raises(ProjectError) as caught:
            load_project("sales", tmp_path)
        message = str(caught.value)
        assert str(tmp_path / "sales" / "settings.yaml") in message
        assert "rules:" in message and "focus:" in message


class TestTheSettingsFile:
    @pytest.mark.parametrize(
        ("settings", "named"),
        [
            pytest.param("rule:\n  revenue_type: REVENUE\nfocus: party\n", "'rule'", id="a_misspelt_key"),
            pytest.param("rules: [REVENUE]\nfocus: party\n", "'rules'", id="rules_that_are_a_list"),
            pytest.param("rules:\n  revenue_type: REVENUE\n", "which column", id="no_focus"),
            pytest.param("focus: []\n", "which column", id="an_empty_focus"),
            pytest.param("focus: [party, 3]\n", "which column", id="a_focus_that_is_not_a_column_name"),
            pytest.param("", "which column", id="an_empty_file"),
            pytest.param("- just a list\n", "must hold 'rules' and 'focus'", id="not_a_mapping"),
            pytest.param("focus: [party\n", "not readable YAML", id="broken_yaml"),
        ],
    )
    def test_a_settings_file_that_cannot_work_is_refused_by_name(self, tmp_path: Path, settings: str, named: str) -> None:
        _project(tmp_path, "sales", settings=settings)
        with pytest.raises(ProjectError) as caught:
            load_project("sales", tmp_path)
        assert named in str(caught.value)
        assert str(tmp_path / "sales" / "settings.yaml") in str(caught.value), "the refusal names the file"

    def test_rules_may_be_left_out_and_focus_may_list_several_columns(self, tmp_path: Path) -> None:
        _project(tmp_path, "sales", settings="focus: [account, customer]\n")
        project = load_project("sales", tmp_path)
        assert project.rules == {}
        assert project.focus.identity == ("account", "customer") and project.focus.label == "account"


class TestSavingARun:
    def test_a_run_is_saved_where_the_viewer_finds_it_with_a_panel_for_every_step(self, tmp_path: Path) -> None:
        directory, trace = write_run(load_project("fixture"), tmp_path)

        assert directory == tmp_path / "fixture"
        assert [run.name for run in discover(tmp_path)] == ["fixture"]
        ports = json.loads((directory / MANIFEST).read_text(encoding="utf8"))
        assert set(ports) == {f"{ref.step}.{ref.port}" for ref in trace.frames}
        panels = json.loads((directory / PANEL_MANIFEST).read_text(encoding="utf8"))
        assert set(panels) == {event.step_id for event in trace.events}
        assert "<title>Monthly totals by party" in (directory / PAGE_NAME).read_text(encoding="utf-8")

    def test_a_new_run_replaces_the_old_one_and_leaves_other_projects_alone(self, tmp_path: Path) -> None:
        stale = tmp_path / "fixture" / "a_step_since_removed__rows.parquet"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"old")
        neighbour = tmp_path / "another" / MANIFEST
        neighbour.parent.mkdir()
        neighbour.write_text("{}", encoding="utf8")

        write_run(load_project("fixture"), tmp_path)

        assert not stale.exists(), "a file from the previous run would offer data this run never produced"
        assert neighbour.read_text(encoding="utf8") == "{}"

    def test_a_flow_refused_at_load_leaves_the_previous_run_as_it_was(self, tmp_path: Path) -> None:
        """Nothing ran, so there is nothing to replace the last good run with."""
        pipelines, runs = tmp_path / "pipelines", tmp_path / "runs"
        _project(pipelines, "fixture", settings="focus: party\n")  # the rule the flow needs is missing
        previous = runs / "fixture" / MANIFEST
        previous.parent.mkdir(parents=True)
        previous.write_text("{}", encoding="utf8")

        with pytest.raises(PipelineError, match="revenue_type"):
            write_run(load_project("fixture", pipelines), runs)
        assert previous.read_text(encoding="utf8") == "{}"

    def test_a_step_that_fails_still_saves_what_ran_before_it(self, tmp_path: Path) -> None:
        source = tmp_path / "entries.csv"
        source.write_text("party,amount\nNorthwind,ten\n", encoding="utf8")
        flow = (
            "name: fails at the second step\n"
            "steps:\n"
            "  - id: entries\n"
            "    tool: read_csv\n"
            f"    config: {{path: '{source}'}}\n"
            "  - id: typed\n"
            "    tool: cast_columns\n"
            "    config: {columns: [{column: amount, type: Float64}]}\n"
            "    inputs: {rows: entries.rows}\n"
            "  - id: published\n"
            "    tool: write_csv\n"
            f"    config: {{path: '{tmp_path / 'out.csv'}'}}\n"
            "    inputs: {rows: typed.rows}\n"
        )
        pipelines, runs = tmp_path / "pipelines", tmp_path / "runs"
        _project(pipelines, "broken", flow=flow, settings="focus: party\n")

        with pytest.raises(StepFailedError) as caught:
            write_run(load_project("broken", pipelines), runs)

        assert caught.value.step_id == "typed"
        ports = json.loads((runs / "broken" / MANIFEST).read_text(encoding="utf8"))
        assert "entries.rows" in ports, "the rows that went into the failing step are there to look at"
        assert not any(key.startswith("published.") for key in ports)

        page = (runs / "broken" / PAGE_NAME).read_text(encoding="utf-8")
        assert 'data-goto="published" class="notrun"' in page, "the step after the failure is shown as not run"
        assert 'data-goto="entries" class="notrun"' not in page, "and a step that ran is not"
        panels = json.loads((runs / "broken" / PANEL_MANIFEST).read_text(encoding="utf8"))
        assert set(panels) == {"entries", "typed", "published"}, "every step can still be opened"
        failing = (runs / "broken" / "panels" / panels["typed"]).read_text(encoding="utf-8")
        assert 'data-side="in"' in failing and "rows &larr;" in failing, "the failing step shows what went in"


class TestTheViewer:
    def test_a_running_viewer_is_recognised_only_for_a_run_it_serves(self, tmp_path: Path) -> None:
        write_run(load_project("fixture"), tmp_path)
        server = serve(tmp_path, host="127.0.0.1", port=0)
        try:
            port = server.server_address[1]
            assert runner._viewer_serves("fixture", port)
            assert not runner._viewer_serves("another", port)
        finally:
            server.shutdown()
            server.server_close()
        assert not runner._viewer_serves("fixture", port), "nothing is listening any more"

    def test_a_running_viewer_is_reused_rather_than_a_second_one_started(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_run(load_project("fixture"), tmp_path)
        server = serve(tmp_path, host="127.0.0.1", port=0)
        opened: list[str] = []
        monkeypatch.setattr(runner, "open_in_browser", opened.append)
        monkeypatch.setattr(runner, "serve", lambda *_, **__: pytest.fail("a second viewer was started"))
        try:
            port = server.server_address[1]
            runner.view("fixture", tmp_path, port)
        finally:
            server.shutdown()
            server.server_close()
        assert opened == [f"http://localhost:{port}/fixture/"]

    def test_with_no_viewer_running_one_is_started_serves_the_run_and_stops_on_ctrl_c(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_run(load_project("fixture"), tmp_path)
        port = _free_port()
        opened: list[str] = []
        answered: list[int] = []

        def pressed_ctrl_c(_: float) -> None:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/fixture/api/ports", timeout=5) as answer:
                answered.append(answer.status)
            raise KeyboardInterrupt

        monkeypatch.setattr(runner, "open_in_browser", opened.append)
        monkeypatch.setattr(runner.time, "sleep", pressed_ctrl_c)
        runner.view("fixture", tmp_path, port)

        assert opened == [f"http://localhost:{port}/fixture/"]
        assert answered == [200], "the viewer was serving the run while it was open"
        with pytest.raises(OSError):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/fixture/api/ports", timeout=2)

    def test_from_wsl_the_address_goes_to_the_windows_browser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[list[str]] = []
        monkeypatch.setattr(runner.shutil, "which", lambda name: "/mnt/c/Windows/explorer.exe" if name == "explorer.exe" else None)
        monkeypatch.setattr(runner.subprocess, "run", lambda command, **_: ran.append(command))
        monkeypatch.setattr(runner.webbrowser, "open", lambda _: pytest.fail("the Linux browser was asked instead"))
        runner.open_in_browser("http://localhost:8899/fixture/")
        assert ran == [["/mnt/c/Windows/explorer.exe", "http://localhost:8899/fixture/"]]

    def test_elsewhere_the_system_browser_is_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        asked: list[str] = []
        monkeypatch.setattr(runner.shutil, "which", lambda _: None)
        monkeypatch.setattr(runner.subprocess, "run", lambda *_, **__: pytest.fail("explorer.exe is not there"))
        monkeypatch.setattr(runner.webbrowser, "open", lambda url: asked.append(url) or True)
        runner.open_in_browser("http://localhost:8899/fixture/")
        assert asked == ["http://localhost:8899/fixture/"]


class TestTheCommand:
    def test_a_good_run_exits_zero_and_saves_the_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path)
        assert runner.main(["fixture", "--no-view"]) == 0
        assert (tmp_path / "fixture" / PAGE_NAME).is_file()
        assert "6 steps ran" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("name", "said"),
        [("nosuch", "there is no pipeline 'nosuch'"), ("../fixture", "is not a project name")],
    )
    def test_a_refusal_is_one_plain_message_and_a_failing_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str, said: str
    ) -> None:
        monkeypatch.setattr(runner, "RUNS_ROOT", tmp_path)
        assert runner.main([name, "--no-view"]) == 1
        error = capsys.readouterr().err
        assert error.startswith(f"cannot run '{name}': ") and said in error
        assert "Traceback" not in error
        assert not any(tmp_path.iterdir()), "nothing was saved"

