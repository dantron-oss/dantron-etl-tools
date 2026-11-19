"""Preview plans, what a real run writes, and saving runs without losing the last good one."""

import contextlib
import json
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl import load_pipeline
from dantron_etl.preview import MANIFEST, PAGE_NAME, persist_ports
from dantron_etl.viewer.project import ProjectError, load_entry, load_project
from dantron_etl.viewer.runs import (
    FAILED_SUFFIX,
    RUN_FILE,
    Plan,
    execute,
    last_runs,
    latest,
    preview_plan,
    save_run,
    writes,
    writes_digest,
)

NAME = "orders"
DATA = "pipelines/orders/orders.csv"
TARGET = "kept.csv"


def step(step_id: str, tool: str, config: dict[str, Any], inputs: dict[str, str] | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": step_id, "tool": tool, "config": config}
    if inputs:
        entry["inputs"] = inputs
    return entry


def flow(*steps: dict[str, Any]) -> Any:
    return load_pipeline({"name": "a flow", "steps": list(steps)})


READ = step("read", "read_csv", {"path": "in.csv"})
CHECK = step("check", "assert_rows", {"predicate": "[amount] > 0", "label": "positive"}, {"rows": "read.rows"})


# ---------------------------------------------------------------------------
# A project on disk: read_csv -> filter -> write_csv, focused on party
# ---------------------------------------------------------------------------


def pipeline_text(read_config: str = f"path: {DATA}", predicate: str = "[amount] > 0") -> str:
    return (
        "name: Orders kept\n"
        "steps:\n"
        "  - id: read\n"
        "    tool: read_csv\n"
        f"    config: {{{read_config}}}\n"
        "  - id: keep\n"
        "    tool: filter\n"
        f"    config: {{predicate: '{predicate}'}}\n"
        "    inputs: {rows: read.rows}\n"
        "  - id: save\n"
        "    tool: write_csv\n"
        f"    config: {{path: {TARGET}, overwrite: true}}\n"
        "    inputs: {rows: keep.true}\n"
    )


def make_project(root: Path, parties: int = 7) -> None:
    (root / "etl-project.yaml").write_text(f"pipelines:\n  {NAME}:\n    focus: party\n", encoding="utf8")
    (root / "pipelines" / NAME).mkdir(parents=True)
    (root / "pipelines" / f"{NAME}.yaml").write_text(pipeline_text(), encoding="utf8")
    rows = ["party,amount"]
    for n in range(parties):
        rows += [f"{101 + n},{10 + n}", f"{101 + n},{-1 - n}", f"{101 + n},{5}"]
    (root / DATA).write_text("\n".join(rows) + "\n", encoding="utf8")


def run(root: Path, kind: str = "preview", **kwargs: Any) -> Any:
    project = load_project(root)
    return execute(project, project.pipelines[NAME], kind, **kwargs)


def runs_folder(root: Path) -> Path:
    return load_project(root).runs_folder


def snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*")) if p.is_file()}


def every_path(root: Path) -> set[Path]:
    return set(root.rglob("*"))


def outside(paths: set[Path], folder: Path) -> set[Path]:
    return {p for p in paths if p != folder and folder not in p.parents and p not in folder.parents}


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    make_project(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# preview_plan
# ---------------------------------------------------------------------------


class TestPreviewPlan:
    def test_writer_and_everything_downstream_of_it_is_held_back(self) -> None:
        spec = flow(
            READ,
            step("save", "write_csv", {"path": "a.csv"}, {"rows": "read.rows"}),
            step("keep", "filter", {"predicate": "[amount] > 0"}, {"rows": "save.rows"}),
            step("save2", "write_csv", {"path": "b.csv"}, {"rows": "keep.true"}),
        )
        plan = preview_plan(spec)
        assert isinstance(plan, Plan)
        assert plan.runs == ("read",)
        assert plan.held_back == ("save", "keep", "save2")

    def test_an_assert_step_runs_in_a_preview(self) -> None:
        spec = flow(READ, CHECK, step("save", "write_csv", {"path": "a.csv"}, {"rows": "check.rows"}))
        plan = preview_plan(spec)
        assert plan.runs == ("read", "check")
        assert plan.held_back == ("save",)

    def test_a_flow_with_no_writer_holds_nothing_back(self) -> None:
        plan = preview_plan(flow(READ, CHECK))
        assert plan.runs == ("read", "check")
        assert plan.held_back == ()

    def test_a_step_feeding_a_writer_still_runs_and_order_follows_the_flow(self) -> None:
        spec = flow(
            READ,
            step("keep", "filter", {"predicate": "[amount] > 0"}, {"rows": "read.rows"}),
            step("save", "write_csv", {"path": "a.csv"}, {"rows": "keep.true"}),
            step("check", "assert_rows", {"predicate": "[amount] < 1", "label": "small"}, {"rows": "keep.false"}),
        )
        plan = preview_plan(spec)
        assert plan.runs == tuple(s for s in spec.order() if s != "save")
        assert "keep" in plan.runs
        assert plan.held_back == ("save",)


# ---------------------------------------------------------------------------
# writes and writes_digest
# ---------------------------------------------------------------------------


class TestWrites:
    def test_a_relative_csv_path_is_resolved_against_the_root_and_an_absolute_one_kept(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "elsewhere" / "abs.csv"
        root = tmp_path / "root"
        # The loader checks an output folder exists, from the working folder, so both are made.
        (root / "out").mkdir(parents=True)
        elsewhere.parent.mkdir()
        with contextlib.chdir(root):
            spec = flow(
            READ,
                step("save", "write_csv", {"path": "out/a.csv"}, {"rows": "read.rows"}),
                step("save_abs", "write_csv", {"path": str(elsewhere)}, {"rows": "read.rows"}),
            )
        found = writes(spec, root)
        assert [(w.step, w.tool, w.kind) for w in found] == [
            ("save", "write_csv", "file"),
            ("save_abs", "write_csv", "file"),
        ]
        assert Path(found[0].target) == tmp_path / "root" / "out" / "a.csv"
        assert Path(found[1].target) == elsewhere

    def test_a_table_writer_names_schema_and_table(self, tmp_path: Path) -> None:
        spec = flow(
            READ,
            step("load", "write_table", {"connection": "TARGET_DB", "schema": "finance", "table": "sample", "mode": "replace"}, {"rows": "read.rows"}),
            step("load_plain", "write_table", {"connection": "TARGET_DB", "table": "plain", "mode": "append"}, {"rows": "read.rows"}),
        )
        found = writes(spec, tmp_path)
        assert [(w.step, w.kind, w.target) for w in found] == [
            ("load", "table", "finance.sample"),
            ("load_plain", "table", "plain"),
        ]

    def test_publish_tables_reports_one_write_per_table(self, tmp_path: Path) -> None:
        tables = [
            {"port": "totals", "table": "totals", "mode": "replace"},
            {"port": "log", "table": "movements", "mode": "append"},
        ]
        spec = flow(
            READ,
            step("pub", "publish_tables", {"connection": "TARGET_DB", "schema": "finance", "tables": tables}, {"totals": "read.rows", "log": "read.rows"}),
        )
        found = writes(spec, tmp_path)
        assert [(w.step, w.tool, w.kind, w.target) for w in found] == [
            ("pub", "publish_tables", "table", "finance.totals"),
            ("pub", "publish_tables", "table", "finance.movements"),
        ]

    def test_digest_is_stable_and_changes_when_a_target_changes(self, tmp_path: Path) -> None:
        def digest(path: str) -> str:
            return writes_digest(writes(flow(READ, step("save", "write_csv", {"path": path}, {"rows": "read.rows"})), tmp_path))

        first = digest("a.csv")
        assert re.fullmatch(r"[0-9a-f]{64}", first)
        assert digest("a.csv") == first
        assert digest("b.csv") != first


# ---------------------------------------------------------------------------
# execute: preview and real
# ---------------------------------------------------------------------------


class TestExecutePreview:
    def test_a_preview_writes_only_into_the_runs_folder(self, project_root: Path) -> None:
        before = every_path(project_root)
        record = run(project_root)
        folder = runs_folder(project_root)

        assert not (project_root / TARGET).exists()
        assert outside(every_path(project_root), folder) == outside(before, folder)
        assert (record.kind, record.outcome, record.held_back) == ("preview", "finished", ("save",))
        assert (record.steps_total, record.steps_ran) == (3, 2)

    def test_the_saved_preview_holds_its_record_ports_and_page(self, project_root: Path) -> None:
        run(project_root)
        saved = runs_folder(project_root) / NAME

        written = json.loads((saved / RUN_FILE).read_text(encoding="utf8"))
        assert (written["kind"], written["outcome"], written["held_back"]) == ("preview", "finished", ["save"])
        ports = json.loads((saved / MANIFEST).read_text(encoding="utf8"))
        assert "read.rows" in ports and "keep.true" in ports
        assert not any(key.startswith("save.") for key in ports)
        assert (saved / PAGE_NAME).is_file()

    def test_a_new_run_replaces_the_old_one_and_leaves_other_pipelines_alone(self, project_root: Path) -> None:
        run(project_root)
        folder = runs_folder(project_root)
        stale = folder / NAME / "a_step_since_removed__rows.parquet"
        stale.write_bytes(b"old")
        neighbour = folder / "another" / MANIFEST
        neighbour.parent.mkdir()
        neighbour.write_text("{}", encoding="utf8")

        run(project_root)

        assert not stale.exists(), "a file from the previous run would offer rows this run never produced"
        assert (folder / NAME / MANIFEST).is_file()
        assert neighbour.read_text(encoding="utf8") == "{}"

    def test_entities_lists_every_value_as_text_with_no_cap(self, project_root: Path) -> None:
        run(project_root)
        entities = json.loads((runs_folder(project_root) / NAME / "entities.json").read_text(encoding="utf8"))
        assert entities["label"] == "party"
        assert entities["values"] == [str(101 + n) for n in range(7)]


class TestExecuteReal:
    def test_a_real_run_without_the_current_digest_refuses_and_writes_nothing(self, project_root: Path) -> None:
        before = snapshot(project_root)
        with pytest.raises(ProjectError):
            run(project_root, "real")
        with pytest.raises(ProjectError):
            run(project_root, "real", confirm_digest="0" * 64)
        assert snapshot(project_root) == before
        assert not (project_root / TARGET).exists()

    def test_a_real_run_with_the_confirmed_digest_writes_its_target(self, project_root: Path) -> None:
        project = load_project(project_root)
        entry = project.pipelines[NAME]
        digest = writes_digest(writes(load_entry(project, entry), project.root))

        record = execute(project, entry, "real", confirm_digest=digest)

        assert (record.kind, record.outcome, record.held_back) == ("real", "finished", ())
        assert (project_root / TARGET).is_file()
        written = json.loads((project.runs_folder / NAME / RUN_FILE).read_text(encoding="utf8"))
        assert written["kind"] == "real"


# ---------------------------------------------------------------------------
# Failures and refusals never touch the last good run
# ---------------------------------------------------------------------------


class TestFailedRuns:
    def test_a_failure_at_the_first_step_keeps_the_good_run_and_names_the_missing_file(self, project_root: Path) -> None:
        run(project_root)
        folder = runs_folder(project_root)
        good_before = snapshot(folder / NAME)

        (project_root / DATA).unlink()
        record = run(project_root)

        assert snapshot(folder / NAME) == good_before
        assert (record.outcome, record.failed_step) == ("failed", "read")
        assert "orders.csv" in record.message
        assert "party" not in record.message
        good, failed = last_runs(folder, NAME)
        assert good is not None and good.outcome == "finished"
        assert failed is not None and (failed.outcome, failed.failed_step) == ("failed", "read")
        assert failed.directory == folder / f"{NAME}{FAILED_SUFFIX}"
        assert latest(good, failed) == failed

    def test_a_failure_at_a_later_step_names_that_step_and_its_cause(self, project_root: Path) -> None:
        run(project_root)
        folder = runs_folder(project_root)
        good_before = snapshot(folder / NAME)

        (project_root / "pipelines" / f"{NAME}.yaml").write_text(pipeline_text(predicate="[nope] > 0"), encoding="utf8")
        record = run(project_root)

        assert snapshot(folder / NAME) == good_before
        assert (record.outcome, record.failed_step) == ("failed", "keep")
        assert "nope" in record.message

    def test_a_successful_run_clears_the_failed_slot(self, project_root: Path) -> None:
        data = (project_root / DATA).read_bytes()
        (project_root / DATA).unlink()
        run(project_root)
        folder = runs_folder(project_root)
        assert (folder / f"{NAME}{FAILED_SUFFIX}").is_dir()

        (project_root / DATA).write_bytes(data)
        record = run(project_root)

        assert record.outcome == "finished"
        assert not (folder / f"{NAME}{FAILED_SUFFIX}").exists()
        good, failed = last_runs(folder, NAME)
        assert failed is None
        assert good is not None and latest(good, failed) == good

    def test_a_load_refusal_is_saved_as_refused_and_leaves_the_good_run(self, project_root: Path) -> None:
        run(project_root)
        folder = runs_folder(project_root)
        good_before = snapshot(folder / NAME)

        (project_root / "pipelines" / f"{NAME}.yaml").write_text(
            pipeline_text(read_config=f"path: {DATA}, nonsense: 1"), encoding="utf8"
        )
        record = run(project_root)

        assert record.outcome == "refused"
        assert "nonsense" in record.message
        assert snapshot(folder / NAME) == good_before
        _, failed = last_runs(folder, NAME)
        assert failed is not None and failed.outcome == "refused"
        assert not (failed.directory / MANIFEST).exists()


# ---------------------------------------------------------------------------
# Reading the run store back
# ---------------------------------------------------------------------------


class TestRunStore:
    def test_a_pipeline_never_run_has_no_runs(self, tmp_path: Path) -> None:
        assert last_runs(tmp_path / "runs", NAME) == (None, None)
        assert latest(None, None) is None

    def test_a_run_saved_by_another_program_reads_as_unknown(self, tmp_path: Path) -> None:
        saved = tmp_path / "runs" / NAME
        persist_ports({"read.rows": pl.DataFrame({"party": ["A"]})}, saved)

        good, failed = last_runs(tmp_path / "runs", NAME)

        assert failed is None
        assert good is not None
        assert (good.kind, good.outcome) == ("unknown", "unknown")
        assert good.directory == saved
        assert good.started


# ---------------------------------------------------------------------------
# A run made by a project's own script, saved where the viewer lists it
# ---------------------------------------------------------------------------


def run_outside(root: Path, *, fail: bool = False) -> tuple[Any, Any, Any]:
    """Run the project's pipeline the way a project's own script would, from its project folder."""
    from dantron_etl import StepFailedError, run_pipeline

    project = load_project(root)
    spec = load_entry(project, project.pipelines[NAME])
    if fail:
        (root / DATA).unlink()
    try:
        return project, spec, (run_pipeline(spec, base_dir=root), None)
    except StepFailedError as failure:
        return project, spec, (failure.trace, failure)


class TestSaveRun:
    def test_a_finished_run_is_saved_as_a_good_real_run_with_its_entities(self, project_root: Path) -> None:
        project, spec, (trace, _) = run_outside(project_root)

        record = save_run(project, NAME, spec, trace)

        good, failed = last_runs(project.runs_folder, NAME)
        assert failed is None and good is not None
        assert (good.kind, good.outcome, good.failed_step) == ("real", "finished", None)
        assert good.steps_ran == good.steps_total == 3
        assert record.directory == project.runs_folder / NAME
        entities = json.loads((record.directory / "entities.json").read_text(encoding="utf8"))
        assert entities["values"] == [str(101 + n) for n in range(7)]
        assert good.focus is not None and good.focus.identity == ("party",)
        assert (record.directory / PAGE_NAME).is_file()

    def test_a_failed_run_goes_to_the_failed_slot_and_keeps_the_good_one(self, project_root: Path) -> None:
        project, spec, (trace, _) = run_outside(project_root)
        save_run(project, NAME, spec, trace)
        good_before = snapshot(project.runs_folder / NAME)

        project, spec, (trace, failure) = run_outside(project_root, fail=True)
        record = save_run(project, NAME, spec, trace, failure=failure)

        assert (record.outcome, record.failed_step) == ("failed", "read")
        assert "orders.csv" in (record.message or "")
        assert snapshot(project.runs_folder / NAME) == good_before
        good, failed = last_runs(project.runs_folder, NAME)
        assert latest(good, failed) == failed

    def test_a_name_the_project_does_not_have_is_refused(self, project_root: Path) -> None:
        project, spec, (trace, _) = run_outside(project_root)
        with pytest.raises(ProjectError, match="nope"):
            save_run(project, "nope", spec, trace)
        assert not (project.runs_folder / "nope").exists()

    def test_the_focus_given_wins_over_the_settings(self, project_root: Path) -> None:
        from dantron_etl import FocusSpec

        project, spec, (trace, _) = run_outside(project_root)
        record = save_run(project, NAME, spec, trace, focus=FocusSpec(label="amount", identity=("amount",)))
        good, _ = last_runs(project.runs_folder, NAME)
        assert good is not None and good.focus is not None and good.focus.identity == ("amount",)
        assert json.loads((record.directory / "entities.json").read_text(encoding="utf8"))["label"] == "amount"
