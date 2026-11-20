"""The files a pipeline's steps name, what is known about each, and a first look inside: viewer/files.py.

The set of file settings comes from the tools' own templates (the ``file_path`` marker), and each
path is worked out the way a run works it out. Every test builds a real project on disk and loads
it the way the viewer does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import xlsxwriter
import yaml

from dantron_etl import REGISTRY
from dantron_etl.config import ConfigField
from dantron_etl.viewer.files import (
    PREVIEW_COLUMNS,
    PREVIEW_ROWS,
    FileRefused,
    NamedFile,
    facts,
    file_settings,
    named_file,
    named_files,
    preview,
)
from dantron_etl.viewer.project import load_entry, load_project
from dantron_etl.viewer.runs import run_folder

ORDERS = "party,amount\nA,1\nB,-1\nC,2\n"


def write_workbook(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    book = xlsxwriter.Workbook(str(path))
    for name, rows in sheets.items():
        sheet = book.add_worksheet(name)
        for at, row in enumerate(rows):
            sheet.write_row(at, 0, row)
    book.close()


def make(root: Path, steps: list[dict[str, Any]], *, rules: dict[str, Any] | None = None) -> Path:
    """A project with one pipeline, ``flow``, made of ``steps``. A lone reader gets a writer after it."""
    if not any(step["tool"].startswith("write_") for step in steps):
        sink = {"path": "output/sink.csv", "overwrite": True}
        steps = [*steps, {"id": "sink", "tool": "write_csv", "config": sink, "inputs": {"rows": "read.rows"}}]
    (root / "pipelines").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(exist_ok=True)
    document = {"name": "A flow", "steps": steps}
    (root / "pipelines" / "flow.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf8")
    settings: dict[str, Any] = {"pipelines": {"flow": {"rules": rules}}} if rules else {}
    (root / "etl-project.yaml").write_text(yaml.safe_dump(settings), encoding="utf8")
    return root


def read_step(path: str, **config: Any) -> dict[str, Any]:
    return {"id": "read", "tool": "read_csv", "config": {"path": path, **config}}


def write_step(path: str, source: str = "read.rows") -> dict[str, Any]:
    return {"id": "save", "tool": "write_csv", "config": {"path": path, "overwrite": True}, "inputs": {"rows": source}}


def loaded(root: Path) -> tuple[Any, Any]:
    project = load_project(root)
    return project, load_entry(project, project.pipelines["flow"])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    workbook_step = {
        "id": "pack",
        "tool": "write_excel",
        "config": {
            "path": "output/pack.xlsx",
            "template": "pipelines/template.xlsx",
            "sheets": [{"port": "totals", "sheet": "Totals"}],
            "overwrite": True,
        },
        "inputs": {"totals": "keep.true"},
    }
    keep = {"id": "keep", "tool": "filter", "config": {"predicate": "[amount] > 0"}, "inputs": {"rows": "read.rows"}}
    make(root, [read_step("pipelines/orders.csv"), keep, write_step("output/totals.csv", "keep.true"), workbook_step])
    (root / "pipelines" / "orders.csv").write_text(ORDERS, encoding="utf8")
    write_workbook(root / "pipelines" / "template.xlsx", {"Cover": [["title"]]})
    return root


# ---------------------------------------------------------------------------
# Which settings are file paths, and which files a pipeline names
# ---------------------------------------------------------------------------


def test_file_settings_are_read_from_the_marker_in_each_template() -> None:
    assert file_settings(REGISTRY.get("read_csv").config_template()) == ("path",)
    assert file_settings(REGISTRY.get("write_excel").config_template()) == ("path", "template")
    assert file_settings(REGISTRY.get("filter").config_template()) == ()
    made_up = {"source": ConfigField(type="string", description="x", file_path=True), "sep": ConfigField("string", "y")}
    assert file_settings(made_up) == ("source",)


def test_every_tool_with_a_file_setting_is_found_by_the_marker() -> None:
    marked = {name for name in REGISTRY.names() if file_settings(REGISTRY.get(name).config_template())}
    assert {"read_csv", "read_excel", "write_csv", "write_excel"} <= marked


def test_a_pipeline_names_each_file_once_per_setting_in_run_order(project: Path) -> None:
    found = named_files(*_spec_and_base(project))
    assert [(one.step, one.setting) for one in found] == [
        ("read", "path"),
        ("save", "path"),
        ("pack", "path"),
        ("pack", "template"),
    ]


def test_a_relative_path_is_worked_out_from_the_folder_a_run_reads_from(project: Path) -> None:
    spec, base = _spec_and_base(project)
    read = named_file(spec, base, "read", "path")
    assert read.written == "pipelines/orders.csv"
    assert read.path == str(project.resolve() / "pipelines" / "orders.csv")


def test_an_absolute_path_is_used_as_written(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / "in.csv"
    root = make(tmp_path / "project", [read_step(elsewhere.as_posix())])
    found = named_file(*_spec_and_base(root), "read", "path")
    assert found.path == elsewhere.as_posix()


def test_a_rule_in_a_path_is_filled_in_as_the_run_sees_it(tmp_path: Path) -> None:
    root = make(tmp_path / "project", [read_step("${rules.folder}/in.csv")], rules={"folder": "data"})
    found = named_file(*_spec_and_base(root), "read", "path")
    assert found.written == "data/in.csv"
    assert found.path == str(root.resolve() / "data" / "in.csv")


@pytest.mark.parametrize(
    ("step", "setting"),
    [("nowhere", "path"), ("keep", "predicate"), ("read", "delimiter"), ("read", "nothing"), ("pack", "sheets")],
    ids=["unknown-step", "a-step-with-no-file", "not-a-file-setting", "unknown-setting", "a-list-setting"],
)
def test_anything_but_a_file_setting_of_a_step_is_refused(project: Path, step: str, setting: str) -> None:
    with pytest.raises(FileRefused):
        named_file(*_spec_and_base(project), step, setting)


def test_a_file_setting_the_step_leaves_out_is_refused(tmp_path: Path) -> None:
    root = make(tmp_path / "project", [read_step("in.csv"), write_step("output/out.csv")])
    with pytest.raises(FileRefused):
        named_file(*_spec_and_base(root), "save", "template")


def _spec_and_base(root: Path) -> tuple[Any, Path]:
    project, spec = loaded(root)
    return spec, run_folder(project)


# ---------------------------------------------------------------------------
# What is known about a file
# ---------------------------------------------------------------------------


def test_a_file_that_is_there_has_its_size_and_time(project: Path) -> None:
    project_, spec = loaded(project)
    found = facts(named_file(spec, run_folder(project_), "read", "path"), project_.root)
    assert found["exists"] and not found["outside_project"]
    assert found["size"] == len(ORDERS.encode())
    stamp = os.stat(project / "pipelines" / "orders.csv").st_mtime
    assert found["modified"].startswith(__import__("datetime").datetime.fromtimestamp(stamp).strftime("%Y-%m-%d"))
    assert found["can_preview"] and found["can_open"]


def test_a_file_not_written_yet_is_missing_and_says_whether_its_folder_is_there(project: Path) -> None:
    project_, spec = loaded(project)
    found = facts(named_file(spec, run_folder(project_), "save", "path"), project_.root)
    assert not found["exists"] and found["folder_exists"]
    assert found["size"] is None and not found["can_preview"]
    assert found["can_open"]


def test_a_file_outside_the_project_is_allowed_and_marked(tmp_path: Path) -> None:
    elsewhere = tmp_path / "shared" / "in.csv"
    elsewhere.parent.mkdir()
    elsewhere.write_text(ORDERS, encoding="utf8")
    root = make(tmp_path / "project", [read_step(elsewhere.as_posix())])
    project_, spec = loaded(root)
    found = facts(named_file(spec, run_folder(project_), "read", "path"), project_.root)
    assert found["exists"] and found["outside_project"]
    assert any("outside the project" in line for line in found["said"])


def test_a_link_names_the_file_it_points_to_and_counts_as_outside_when_that_is(tmp_path: Path) -> None:
    target = tmp_path / "shared" / "real.csv"
    target.parent.mkdir()
    target.write_text(ORDERS, encoding="utf8")
    root = make(tmp_path / "project", [read_step("pipelines/link.csv")])
    (root / "pipelines" / "link.csv").symlink_to(target)
    project_, spec = loaded(root)
    found = facts(named_file(spec, run_folder(project_), "read", "path"), project_.root)
    assert found["path"] == str(root.resolve() / "pipelines" / "link.csv")
    assert found["real_path"] == str(target.resolve())
    assert found["outside_project"]


def test_a_path_holding_a_placeholder_cannot_be_worked_out(tmp_path: Path) -> None:
    root = make(tmp_path / "project", [read_step("{scratch}/in.csv")])
    project_, spec = loaded(root)
    found = facts(named_file(spec, run_folder(project_), "read", "path"), project_.root)
    assert found["path"] is None
    assert not found["can_open"] and not found["can_preview"]


# ---------------------------------------------------------------------------
# A first look inside
# ---------------------------------------------------------------------------


def _preview_of(tmp_path: Path, name: str, content: bytes | str, **config: Any) -> dict[str, Any]:
    root = make(tmp_path / "project", [read_step(f"pipelines/{name}", **config)])
    target = root / "pipelines" / name
    if isinstance(content, str):
        target.write_text(content, encoding="utf8")
    else:
        target.write_bytes(content)
    project_, spec = loaded(root)
    return preview(named_file(spec, run_folder(project_), "read", "path"))


def test_a_csv_is_shown_as_the_step_splits_it(tmp_path: Path) -> None:
    shown = _preview_of(tmp_path, "in.csv", 'title line\nparty;note\nA;"x;y"\nB;\n', delimiter=";", skip_rows=1)
    assert shown["kind"] == "table"
    assert shown["columns"] == ["party", "note"]
    assert shown["rows"] == [["A", "x;y"], ["B", None]]
    assert shown["rows_total"] == 2


def test_a_long_csv_shows_its_first_rows_and_says_how_many_were_left_out(tmp_path: Path) -> None:
    lines = "\n".join(f"P{n},{n}" for n in range(PREVIEW_ROWS + 25))
    shown = _preview_of(tmp_path, "in.csv", "party,amount\n" + lines + "\n")
    assert len(shown["rows"]) == PREVIEW_ROWS
    assert shown["rows"][0] == ["P0", "0"]
    assert shown["rows_total"] == PREVIEW_ROWS + 25
    assert f"{PREVIEW_ROWS + 25:,}" in shown["note"]


def test_a_wide_csv_shows_its_first_columns_and_says_how_many_were_left_out(tmp_path: Path) -> None:
    width = PREVIEW_COLUMNS + 7
    header = ",".join(f"c{n}" for n in range(width))
    shown = _preview_of(tmp_path, "in.csv", header + "\n" + ",".join(str(n) for n in range(width)) + "\n")
    assert shown["columns"] == [f"c{n}" for n in range(PREVIEW_COLUMNS)]
    assert shown["rows"] == [[str(n) for n in range(PREVIEW_COLUMNS)]]
    assert shown["columns_total"] == width
    assert str(width) in shown["note"]


def test_a_csv_too_big_to_read_whole_shows_its_start_and_does_not_guess_a_total(tmp_path: Path) -> None:
    row = "party,amount\n" + "".join(f"P{n},{'9' * 60}\n" for n in range(40_000))
    shown = _preview_of(tmp_path, "in.csv", row)
    assert shown["kind"] == "table"
    assert len(shown["rows"]) == PREVIEW_ROWS
    assert shown["rows_total"] is None
    assert all(len(r) == 2 for r in shown["rows"])


def test_an_empty_file_says_so(tmp_path: Path) -> None:
    shown = _preview_of(tmp_path, "in.csv", b"")
    assert shown["kind"] == "none"
    assert "empty" in shown["note"]


def test_a_csv_not_in_the_steps_encoding_says_so_instead_of_guessing(tmp_path: Path) -> None:
    shown = _preview_of(tmp_path, "in.csv", "party\ncafé\n".encode("cp1252"))
    assert shown["kind"] == "none"
    assert "utf8" in shown["note"]


def test_a_workbook_lists_its_sheets_and_shows_the_one_asked_for(tmp_path: Path) -> None:
    root = make(tmp_path / "project", [{"id": "read", "tool": "read_excel", "config": {"path": "in.xlsx", "sheet": "Two"}}])
    write_workbook(
        root / "in.xlsx", {"One": [["a", "b"], [1, "x"]], "Two": [["name"], ["first"], ["second"]], "Three": []}
    )
    project_, spec = loaded(root)
    found = named_file(spec, run_folder(project_), "read", "path")

    first = preview(found)
    assert first["kind"] == "workbook"
    assert first["sheets"] == ["One", "Two", "Three"]
    assert first["sheet"] == "Two"  # the sheet the step reads
    assert first["rows"] == [["name"], ["first"], ["second"]]

    one = preview(found, sheet="One")
    assert one["sheet"] == "One"
    assert one["columns"] == ["A", "B"]
    assert one["rows"] == [["a", "b"], ["1", "x"]]

    empty = preview(found, sheet="Three")
    assert empty["rows"] == []

    with pytest.raises(FileRefused):
        preview(found, sheet="Four")


def test_another_text_file_shows_its_first_lines(tmp_path: Path) -> None:
    # No shipped tool names a plain text file yet; a future one gets this for free.
    notes = tmp_path / "notes.txt"
    notes.write_text("".join(f"line {n}\n" for n in range(500)), encoding="utf8")
    shown = preview(NamedFile("step", "source", "some_tool", str(notes), str(notes), {}))
    assert shown["kind"] == "text"
    assert shown["lines"][:2] == ["line 0", "line 1"]
    assert len(shown["lines"]) < 500
    assert shown["more"]


def test_a_binary_file_is_not_shown(tmp_path: Path) -> None:
    root = make(tmp_path / "project", [read_step("in.csv")])
    (root / "in.csv").write_bytes(b"\x00\x01\x02binary\xff")
    project_, spec = loaded(root)
    shown = preview(named_file(spec, run_folder(project_), "read", "path"))
    assert shown["kind"] == "none"
    assert "cannot" in shown["note"]


def test_a_file_not_there_has_nothing_to_show(project: Path) -> None:
    project_, spec = loaded(project)
    shown = preview(named_file(spec, run_folder(project_), "save", "path"))
    assert shown["kind"] == "none"


def test_a_preview_of_a_link_shows_what_the_step_reads_through_it(tmp_path: Path) -> None:
    target = tmp_path / "shared" / "real.csv"
    target.parent.mkdir()
    target.write_text("party\nthrough-the-link\n", encoding="utf8")
    root = make(tmp_path / "project", [read_step("link.csv")])
    (root / "link.csv").symlink_to(target)
    project_, spec = loaded(root)
    shown = preview(named_file(spec, run_folder(project_), "read", "path"))
    assert shown["rows"] == [["through-the-link"]]
