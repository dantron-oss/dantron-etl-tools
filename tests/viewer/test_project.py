"""Finding a project and its pipelines: viewer/project.py.

A project is a folder. It holds pipeline files, and maybe an `etl-project.yaml` that says where the
pipelines are, where runs are saved, and what rules and focus each pipeline uses. These tests build
small projects in a scratch folder and check what the viewer reads from them.

Reading a project only reads YAML. Project code (a rules loader) runs only when a pipeline is
checked, loaded or run. Several tests below pin that line from both sides.
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from dantron_etl.errors import ToolError
from dantron_etl.focus import FocusSpec, RelatedKey
from dantron_etl.paths import ProjectRootNotFound, project_root
from dantron_etl.spec import Pipeline, PortRef
from dantron_etl.viewer.project import (
    SETTINGS_FILE,
    PipelineEntry,
    Project,
    ProjectError,
    find_project,
    load_entry,
    load_project,
    pipeline_rules,
)

REPO_ROOT = project_root(__file__)
FIXTURE_FLOW = REPO_ROOT / "pipelines" / "fixture.yaml"
FIXTURE_INPUTS = REPO_ROOT / "pipelines" / "fixture"

#: A flow that reads one rule. Its output path is filled in per test so it lands in the scratch folder.
_RULED_FLOW = """\
name: Paid orders
steps:
  - id: orders
    tool: read_csv
    config: {{path: data/orders.csv}}
  - id: paid
    tool: filter
    config: {{predicate: "[status] = '${{rules.status}}'"}}
    inputs: {{rows: orders.rows}}
  - id: saved
    tool: write_csv
    config: {{path: "{out}", overwrite: true}}
    inputs: {{rows: paid.true}}
"""

#: A flow that reads no rules at all.
_PLAIN_FLOW = """\
name: {title}
steps:
  - id: orders
    tool: read_csv
    config: {{path: data/orders.csv}}
  - id: saved
    tool: write_csv
    config: {{path: output/orders.csv, overwrite: true}}
    inputs: {{rows: orders.rows}}
"""


@pytest.fixture(autouse=True)
def _no_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root override would beat the upward search; these tests are about the search."""
    monkeypatch.delenv("ETL_PROJECT_ROOT", raising=False)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf8")
    return path


def _project_folder(tmp_path: Path, settings: str | None = None) -> Path:
    """A project folder, with a settings file when one is given and an empty one otherwise."""
    root = tmp_path / "project"
    root.mkdir()
    _write(root / SETTINGS_FILE, settings if settings is not None else "")
    return root


def _ruled_flow(root: Path) -> str:
    (root / "output").mkdir(exist_ok=True)
    return _RULED_FLOW.format(out=(root / "output" / "paid.csv").as_posix())


def _refusal(root: Path, name: str) -> str | None:
    """The refusal for pipeline ``name``, wherever the project reports it.

    A settings problem may stop the whole project, mark one entry, or surface when the pipeline's
    rules are resolved. Any of the three is a refusal; silence is not.
    """
    try:
        project = load_project(root)
    except ProjectError as error:
        return str(error)
    entry = project.pipelines[name]
    if entry.problem:
        return entry.problem
    try:
        pipeline_rules(project, entry)
    except ProjectError as error:
        return str(error)
    return None


class TestTheSettingsFileMarksAProject:
    def test_a_folder_holding_only_the_settings_file_is_found_from_a_subfolder(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path)
        deeper = root / "pipelines" / "inputs"
        deeper.mkdir(parents=True)

        project = find_project(deeper)

        assert isinstance(project, Project)
        assert project.root == root.resolve()
        assert project.settings_path == (root / SETTINGS_FILE).resolve()

    def test_a_folder_with_no_marker_still_refuses(self, tmp_path: Path) -> None:
        nowhere = tmp_path / "nowhere" / "deeper"
        nowhere.mkdir(parents=True)
        _write(nowhere / "etl-project.yml", "")  # close, but not the settings file

        with pytest.raises((ProjectRootNotFound, ProjectError)):
            find_project(nowhere)


class TestDiscoveryWithoutASettingsFile:
    def test_defaults_names_titles_step_counts_and_rules_needed(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        (root / ".git").mkdir(parents=True)
        pipes = root / "pipelines"
        _write(pipes / "zeta.yaml", _PLAIN_FLOW.format(title="The last one"))
        # Three references, two names, out of order: rules_needed is sorted and distinct.
        repeated = "'${rules.status}' or [code] = '${rules.code}' or [state] = '${rules.status}'"
        _write(pipes / "alpha.yml", _ruled_flow(root).replace("'${rules.status}'", repeated))
        _write(pipes / "untitled.yaml", "steps:\n  - {id: a, tool: read_csv, config: {path: a.csv}}\n")

        project = load_project(root)

        assert project.settings_path is None
        assert project.pipeline_folders == (root.resolve() / "pipelines",)
        assert project.runs_folder == root.resolve() / "output_files" / "runs"
        assert list(project.pipelines) == ["alpha", "untitled", "zeta"]
        alpha, untitled, zeta = (project.pipelines[n] for n in ("alpha", "untitled", "zeta"))
        assert isinstance(alpha, PipelineEntry)
        assert alpha.path == (pipes / "alpha.yml").resolve() and alpha.path.is_absolute()
        assert (alpha.title, alpha.steps, alpha.rules_needed) == ("Paid orders", 3, ("code", "status"))
        assert (zeta.title, zeta.steps, zeta.rules_needed) == ("The last one", 2, ())
        assert untitled.title == "untitled", "a file with no name: falls back to its file name"
        assert all(entry.problem is None for entry in project.pipelines.values())
        assert all(entry.focus is None and entry.rules_source is None for entry in project.pipelines.values())
        assert project.problems == ()

    def test_files_that_are_not_pipelines_are_skipped_and_noted(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path)
        pipes = root / "pipelines"
        _write(pipes / "good.yaml", _PLAIN_FLOW.format(title="Good"))
        _write(pipes / "lookup.yaml", "codes: [a, b, c]\n")
        _write(pipes / "a list.yaml", "- 1\n- 2\n")
        _write(pipes / "my flow.yaml", _PLAIN_FLOW.format(title="Badly named"))
        _write(pipes / "nested" / "inner.yaml", _PLAIN_FLOW.format(title="Inner"))

        project = load_project(root)

        assert list(project.pipelines) == ["good"]
        noted = "\n".join(project.problems)
        for skipped in ("lookup", "my flow", "a list"):
            assert skipped in noted, f"{skipped!r} was skipped without a note: {project.problems}"
        assert "inner" not in noted, "a file in a subfolder is not a pipeline candidate at all"

    def test_unreadable_yaml_is_listed_with_a_problem(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path)
        _write(root / "pipelines" / "broken.yaml", "steps: [\n  - {id: a\n")
        _write(root / "pipelines" / "fine.yaml", _PLAIN_FLOW.format(title="Fine"))

        project = load_project(root)

        broken = project.pipelines["broken"]
        assert broken.problem and broken.steps == 0
        assert project.pipelines["fine"].problem is None


class TestTheSettingsFile:
    def test_two_pipeline_folders_with_a_duplicate_name_first_folder_wins(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipeline_folders: [flows, more]\nruns_folder: saved/runs\n")
        _write(root / "flows" / "shared.yaml", _PLAIN_FLOW.format(title="First"))
        _write(root / "more" / "shared.yaml", _PLAIN_FLOW.format(title="Second"))
        _write(root / "more" / "extra.yaml", _PLAIN_FLOW.format(title="Extra"))

        project = load_project(root)

        assert project.pipeline_folders == (root.resolve() / "flows", root.resolve() / "more")
        assert project.runs_folder == root.resolve() / "saved" / "runs"
        assert list(project.pipelines) == ["extra", "shared"]
        assert project.pipelines["shared"].title == "First"
        assert any("shared" in note and "more" in note for note in project.problems), project.problems

    def test_an_unknown_top_level_key_is_refused_by_name(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipeline_folder: [pipelines]\n")

        with pytest.raises(ProjectError) as caught:
            load_project(root)

        message = str(caught.value)
        assert issubclass(ProjectError, ToolError)
        assert "pipeline_folder" in message and SETTINGS_FILE in message
        assert "pipeline_folders" in message, "the refusal should list the keys that are allowed"

    def test_an_unknown_key_inside_a_pipeline_entry_is_refused_by_name(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipelines:\n  orders:\n    focus_column: party\n")
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))

        message = _refusal(root, "orders")

        assert message is not None and "focus_column" in message

    def test_two_rules_sources_for_one_pipeline_are_refused(self, tmp_path: Path) -> None:
        root = _project_folder(
            tmp_path, "pipelines:\n  orders:\n    rules: {status: PAID}\n    rules_file: config/rules.yaml\n"
        )
        _write(root / "config" / "rules.yaml", "status: PAID\n")
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))

        message = _refusal(root, "orders")

        assert message is not None and "orders" in message and "rules_file" in message

    def test_a_pipeline_named_in_settings_with_no_file_is_noted(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipelines:\n  ghost:\n    rules: {status: PAID}\n")
        _write(root / "pipelines" / "real.yaml", _PLAIN_FLOW.format(title="Real"))

        project = load_project(root)

        assert list(project.pipelines) == ["real"]
        assert any("ghost" in note for note in project.problems), project.problems


class TestRules:
    def test_inline_rules_load_the_pipeline(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipelines:\n  orders:\n    rules: {status: PAID}\n")
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))
        project = load_project(root)
        entry = project.pipelines["orders"]

        assert pipeline_rules(project, entry) == {"status": "PAID"}
        assert entry.rules_source is not None
        loaded = load_entry(project, entry)
        assert isinstance(loaded, Pipeline)
        assert [step.id for step in loaded.steps] == ["orders", "paid", "saved"]
        assert loaded.step("paid").config["predicate"] == "[status] = 'PAID'"

    def test_a_relative_output_path_loads_from_the_project_folder_wherever_the_caller_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _project_folder(tmp_path)
        (root / "output").mkdir()
        _write(root / "pipelines" / "orders.yaml", _PLAIN_FLOW.format(title="Orders"))
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        project = load_project(root)

        loaded = load_entry(project, project.pipelines["orders"])

        assert loaded.step("saved").config["path"] == "output/orders.csv"
        assert Path.cwd() == elsewhere.resolve(), "loading left the working folder changed"

    @pytest.mark.parametrize(
        "rules_text",
        ("status: PAID\n", "status: {value: PAID, why: Only paid orders count.}\n"),
        ids=("name-value", "value-and-why"),
    )
    def test_a_rules_file_in_either_form(self, rules_text: str, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipelines:\n  orders:\n    rules_file: config/rules.yaml\n")
        _write(root / "config" / "rules.yaml", rules_text)
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))
        project = load_project(root)

        assert pipeline_rules(project, project.pipelines["orders"]) == {"status": "PAID"}

    def test_a_rules_file_mixing_both_forms_is_refused(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipelines:\n  orders:\n    rules_file: config/rules.yaml\n")
        _write(root / "config" / "rules.yaml", "status: {value: PAID, why: Paid only.}\nregion: NORTH\n")
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))

        assert _refusal(root, "orders") is not None

    def test_a_rules_loader_runs_project_code_only_when_rules_are_asked_for(self, tmp_path: Path) -> None:
        module = "project_rules_probe_ran"
        root = _project_folder(tmp_path, f"pipelines:\n  orders:\n    rules_loader: '{module}:rules'\n")
        marker = root / "loader-ran.txt"
        _write(
            root / f"{module}.py",
            f"""\
            from pathlib import Path

            def rules():
                Path({str(marker)!r}).write_text("ran")
                return {{"status": "PAID"}}
            """,
        )
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))
        path_before = list(sys.path)
        try:
            project = load_project(root)
            assert not marker.exists(), "load_project ran project code; only reading YAML is allowed there"
            assert module not in sys.modules

            rules = pipeline_rules(project, project.pipelines["orders"])

            assert rules == {"status": "PAID"}
            assert marker.exists(), "the loader was not called"
            assert sys.path == path_before, "the project root was left on sys.path after importing"
        finally:
            sys.modules.pop(module, None)
            sys.path[:] = path_before

    def test_a_rules_loader_that_raises_does_not_stop_the_listing(self, tmp_path: Path) -> None:
        module = "project_rules_probe_raises"
        root = _project_folder(tmp_path, f"pipelines:\n  orders:\n    rules_loader: '{module}:rules'\n")
        _write(root / f"{module}.py", "raise RuntimeError('project code ran at import')\n")
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))
        try:
            project = load_project(root)
            assert project.pipelines["orders"].problem is None
        finally:
            sys.modules.pop(module, None)

    def test_a_rules_loader_returning_something_other_than_a_mapping_is_refused(self, tmp_path: Path) -> None:
        module = "project_rules_probe_list"
        root = _project_folder(tmp_path, f"pipelines:\n  orders:\n    rules_loader: '{module}:rules'\n")
        _write(root / f"{module}.py", "def rules():\n    return [('status', 'PAID')]\n")
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))
        try:
            project = load_project(root)
            with pytest.raises(ProjectError):
                pipeline_rules(project, project.pipelines["orders"])
        finally:
            sys.modules.pop(module, None)

    def test_missing_rules_name_the_pipeline_the_rules_and_the_settings_file(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path)
        flow = _ruled_flow(root).replace("'${rules.status}'", "'${rules.status}' or [kind] = '${rules.kind}'")
        _write(root / "pipelines" / "orders.yaml", flow)
        _write(root / "pipelines" / "plain.yaml", _PLAIN_FLOW.format(title="Plain"))
        project = load_project(root)

        with pytest.raises(ProjectError) as caught:
            pipeline_rules(project, project.pipelines["orders"])

        message = str(caught.value)
        for needed in ("orders", "status", "kind", SETTINGS_FILE):
            assert needed in message, f"{needed!r} missing from {message!r}"
        assert pipeline_rules(project, project.pipelines["plain"]) == {}, "a flow with no rules needs none"


class TestFocus:
    @pytest.mark.parametrize(
        ("focus_text", "identity"),
        (("party", ("party",)), ("[account, customer]", ("account", "customer"))),
        ids=("one-column", "list"),
    )
    def test_a_column_or_a_list_of_columns(self, focus_text: str, identity: tuple[str, ...], tmp_path: Path) -> None:
        root = _project_folder(tmp_path, f"pipelines:\n  orders:\n    focus: {focus_text}\n")
        _write(root / "pipelines" / "orders.yaml", _PLAIN_FLOW.format(title="Orders"))

        focus = load_project(root).pipelines["orders"].focus

        assert isinstance(focus, FocusSpec)
        assert focus.identity == identity
        assert focus.label == identity[0]

    def test_a_mapping_with_via_turns_step_port_text_into_port_refs(self, tmp_path: Path) -> None:
        settings = """\
        pipelines:
          orders:
            focus:
              columns: [account, customer]
              via:
                "lines.rows": {from: "orders.rows", key: order_id}
        """
        root = _project_folder(tmp_path, textwrap.dedent(settings))
        _write(root / "pipelines" / "orders.yaml", _PLAIN_FLOW.format(title="Orders"))

        focus = load_project(root).pipelines["orders"].focus

        assert isinstance(focus, FocusSpec)
        assert focus.identity == ("account", "customer")
        assert dict(focus.via) == {PortRef("lines", "rows"): RelatedKey(PortRef("orders", "rows"), "order_id")}

    def test_a_bad_focus_is_refused(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipelines:\n  orders:\n    focus: 42\n")
        _write(root / "pipelines" / "orders.yaml", _PLAIN_FLOW.format(title="Orders"))

        message = _refusal(root, "orders")

        assert message is not None and "focus" in message


class TestSaveRows:
    """save_rows: whether a run keeps the rows at every port. Top level for every pipeline, an entry for one."""

    def _two_pipelines(self, tmp_path: Path, settings: str) -> Project:
        root = _project_folder(tmp_path, settings)
        for name in ("first", "second"):
            _write(root / "pipelines" / f"{name}.yaml", _PLAIN_FLOW.format(title=name))
        return load_project(root)

    @pytest.mark.parametrize(
        ("settings", "first", "second"),
        [
            pytest.param("", True, True, id="left_out_rows_are_saved"),
            pytest.param("save_rows: false\n", False, False, id="false_at_the_top_covers_every_pipeline"),
            pytest.param("save_rows: true\n", True, True, id="true_at_the_top"),
            pytest.param(
                "save_rows: false\npipelines:\n  first:\n    save_rows: true\n",
                True,
                False,
                id="an_entry_overrides_a_false_top",
            ),
            pytest.param(
                "pipelines:\n  first:\n    save_rows: false\n", False, True, id="an_entry_overrides_the_default"
            ),
            pytest.param(
                "save_rows: true\npipelines:\n  second:\n    save_rows: false\n",
                True,
                False,
                id="an_entry_overrides_a_true_top",
            ),
        ],
    )
    def test_the_top_level_sets_every_pipeline_and_an_entry_overrides_it(
        self, tmp_path: Path, settings: str, first: bool, second: bool
    ) -> None:
        project = self._two_pipelines(tmp_path, settings)

        assert (project.pipelines["first"].save_rows, project.pipelines["second"].save_rows) == (first, second)
        assert project.pipelines["first"].problem is None

    @pytest.mark.parametrize("value", ["'false'", "0", "1", "null", "[false]", "keep"])
    def test_anything_but_true_or_false_at_the_top_is_refused_naming_the_setting(
        self, tmp_path: Path, value: str
    ) -> None:
        with pytest.raises(ProjectError) as caught:
            self._two_pipelines(tmp_path, f"save_rows: {value}\n")

        message = str(caught.value)
        assert "save_rows" in message and "true" in message and "false" in message, message

    @pytest.mark.parametrize("value", ["'true'", "0", "null"])
    def test_anything_but_true_or_false_in_an_entry_marks_that_pipeline_only(self, tmp_path: Path, value: str) -> None:
        project = self._two_pipelines(tmp_path, f"pipelines:\n  first:\n    save_rows: {value}\n")

        problem = project.pipelines["first"].problem
        assert problem is not None and "save_rows" in problem and "first" in problem, problem
        assert project.pipelines["second"].problem is None


class TestTheOlderPerPipelineSettingsFile:
    def test_its_rules_and_focus_are_still_read_and_the_flow_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "project"
        (root / "output_files").mkdir(parents=True)
        (root / "pyproject.toml").write_text("")
        (root / "pipelines").mkdir()
        shutil.copy(FIXTURE_FLOW, root / "pipelines" / "fixture.yaml")
        shutil.copytree(FIXTURE_INPUTS, root / "pipelines" / "fixture")
        _write(root / "pipelines" / "fixture" / "settings.yaml", "rules: {revenue_type: REVENUE}\nfocus: party\n")
        monkeypatch.chdir(root)

        project = load_project(root)
        entry = project.pipelines["fixture"]

        assert list(project.pipelines) == ["fixture"], "the inputs folder and its settings are not pipelines"
        assert entry.problem is None and entry.rules_source is not None
        assert entry.focus is not None and entry.focus.identity == ("party",)
        assert pipeline_rules(project, entry) == {"revenue_type": "REVENUE"}
        loaded = load_entry(project, entry)
        assert len(loaded.steps) == 6 == entry.steps

    def test_settings_in_both_places_mark_only_that_pipeline(self, tmp_path: Path) -> None:
        root = _project_folder(tmp_path, "pipelines:\n  orders:\n    rules: {status: PAID}\n")
        _write(root / "pipelines" / "orders.yaml", _ruled_flow(root))
        _write(root / "pipelines" / "orders" / "settings.yaml", "rules: {status: OPEN}\n")
        _write(root / "pipelines" / "other.yaml", _PLAIN_FLOW.format(title="Other"))

        project = load_project(root)

        problem = project.pipelines["orders"].problem
        assert problem is not None and "settings.yaml" in problem and SETTINGS_FILE in problem
        assert project.pipelines["other"].problem is None


# A real project with large pipelines, named by a local setting so the repository names none.
_LARGE_PROJECT = os.environ.get("ETL_LARGE_PROJECT", "")


@pytest.mark.skipif(
    not _LARGE_PROJECT or not (Path(_LARGE_PROJECT) / "pipelines").is_dir(),
    reason="set ETL_LARGE_PROJECT to a project folder with large pipelines to run this",
)
def test_a_real_project_lists_its_large_pipelines_read_only() -> None:
    """Reads the YAML only. Never loads or runs these flows."""
    project = load_project(Path(_LARGE_PROJECT))

    assert project.pipelines, f"no pipelines listed in {_LARGE_PROJECT}"
    assert all(entry.steps > 0 for entry in project.pipelines.values()), sorted(project.pipelines)
    assert max(entry.steps for entry in project.pipelines.values()) > 100
