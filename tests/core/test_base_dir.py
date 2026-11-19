"""``run_pipeline(base_dir=...)``: read a flow's relative paths from a folder, not the working directory.

Opt-in. Without it a relative path is read from the working directory, exactly as before. With it,
each path field a tool marks as a file path is joined onto the folder. An absolute path is never
changed. Both directions are tested: a reader and a writer.
"""

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl import RunError, load_pipeline, run_pipeline
from dantron_etl.registry import REGISTRY


def _document(read: str, write: str) -> dict[str, Any]:
    return {
        "name": "paths",
        "steps": [
            {"id": "rows", "tool": "read_csv", "config": {"path": read}},
            {"id": "out", "tool": "write_csv", "config": {"path": write, "overwrite": True}, "inputs": {"rows": "rows.rows"}},
        ],
    }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    (folder / "data").mkdir(parents=True)
    (folder / "data" / "in.csv").write_text("n\n1\n2\n", encoding="utf8")
    return folder


@pytest.fixture
def elsewhere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "elsewhere"
    (folder / "data").mkdir(parents=True)
    (folder / "data" / "in.csv").write_text("n\n7\n", encoding="utf8")
    monkeypatch.chdir(folder)
    return folder


def test_relative_paths_are_read_and_written_under_base_dir(project: Path, elsewhere: Path) -> None:
    trace = run_pipeline(load_pipeline(_document("data/in.csv", "data/out.csv")), base_dir=project)
    assert trace.frame("rows")["n"].to_list() == [1, 2]
    assert pl.read_csv(project / "data" / "out.csv")["n"].to_list() == [1, 2]
    assert not (elsewhere / "data" / "out.csv").exists()


def test_without_base_dir_the_working_directory_is_used_as_before(project: Path, elsewhere: Path) -> None:
    trace = run_pipeline(load_pipeline(_document("data/in.csv", "data/out.csv")))
    assert trace.frame("rows")["n"].to_list() == [7]
    assert (elsewhere / "data" / "out.csv").exists()
    assert not (project / "data" / "out.csv").exists()


def test_an_absolute_path_is_never_changed(project: Path, elsewhere: Path) -> None:
    absolute = str(elsewhere / "data" / "in.csv")
    trace = run_pipeline(load_pipeline(_document(absolute, "out.csv")), base_dir=project)
    assert trace.frame("rows")["n"].to_list() == [7]
    assert (project / "out.csv").exists()


def test_the_loaded_flow_keeps_the_paths_as_written(project: Path, elsewhere: Path) -> None:
    spec = load_pipeline(_document("data/in.csv", "data/out.csv"))
    run_pipeline(spec, base_dir=project)
    assert spec.step("rows").config["path"] == "data/in.csv"


def test_a_base_dir_that_is_not_a_folder_is_refused_before_anything_runs(tmp_path: Path, elsewhere: Path) -> None:
    with pytest.raises(RunError) as raised:
        run_pipeline(load_pipeline(_document("data/in.csv", "data/out.csv")), base_dir=tmp_path / "missing")
    assert "missing" in str(raised.value)
    assert not (elsewhere / "data" / "out.csv").exists()


def test_every_file_reading_and_writing_tool_marks_its_path_fields() -> None:
    marked = {
        name: sorted(field for field, spec in REGISTRY.get(name).config_template().items() if spec.file_path)
        for name in REGISTRY
    }
    assert {name: fields for name, fields in marked.items() if fields} == {
        "read_csv": ["path"],
        "read_excel": ["path"],
        "write_csv": ["path"],
        "write_excel": ["path", "template"],
    }
