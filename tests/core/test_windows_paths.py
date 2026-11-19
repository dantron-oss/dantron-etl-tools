"""Paths with backslashes in them, which is every path on Windows.

Two things broke on Windows and on nothing else, so nobody saw them on Linux:

- The conformance check put its scratch folder into a worked example with ``re.sub``. A replacement
  string reads backslashes as escapes, so ``C:\\Users\\...`` crashed the check before any example ran.
- A refusal named its file with ``repr``. Every backslash came out doubled, so the path in the
  message was not the path the author wrote, and could not be pasted back into a folder window.

A folder name may hold a backslash on Linux too. So each case here puts one in the path, and these
tests catch both problems on any machine, not only on the Windows build.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from dantron_etl.db import open_connection
from dantron_etl.example_world import _resolved_scratch
from dantron_etl.authoring import ConfigError, Ports, REGISTRY, check_registry, run_tool
from dantron_etl.tools.read_csv import ReadCsv
from dantron_etl.tools.read_excel import ReadExcel
from dantron_etl.tools.write_csv import WriteCsv
from dantron_etl.tools.write_excel import WriteExcel

# On Linux this is one folder with a backslash in its name. On Windows it is two folders. Either way,
# the path as written holds a backslash.
_FOLDER = "back\\slash"


def _folder(tmp_path: Path) -> Path:
    folder = tmp_path / _FOLDER
    folder.mkdir(parents=True)
    return folder


def _rows() -> pl.DataFrame:
    return pl.DataFrame({"key": ["a"], "amount": [1.0]})


def _read_csv_absent(folder: Path) -> Path:
    target = folder / "extract.csv"
    run_tool(ReadCsv(), Ports(), {"path": str(target)})
    return target


def _read_excel_absent(folder: Path) -> Path:
    target = folder / "extract.xlsx"
    run_tool(ReadExcel(), Ports(), {"path": str(target), "sheet": "Figures"})
    return target


def _write_csv_existing(folder: Path) -> Path:
    target = folder / "out.csv"
    target.write_text("key,amount\n", encoding="utf8")
    run_tool(WriteCsv(), Ports({"rows": _rows()}), {"path": str(target)})
    return target


def _write_excel_absent_template(folder: Path) -> Path:
    template = folder / "template.xlsx"
    run_tool(
        WriteExcel(),
        Ports({"one": _rows()}),
        {"path": str(folder / "pack.xlsx"), "template": str(template), "sheets": [{"port": "one", "sheet": "Figures"}]},
    )
    return template


def _sqlite_absent(folder: Path) -> Path:
    target = folder / "store.db"
    with open_connection(f"sqlite:///{target}"):
        pass
    return target


@pytest.mark.parametrize(
    "refused",
    [
        _read_csv_absent,
        _read_excel_absent,
        _write_csv_existing,
        _write_excel_absent_template,
        _sqlite_absent,
    ],
    ids=lambda case: case.__name__.lstrip("_"),
)
def test_a_refusal_names_the_file_exactly_as_it_was_written(
    tmp_path: Path, refused: Callable[[Path], Path]
) -> None:
    folder = _folder(tmp_path)
    with pytest.raises(ConfigError) as caught:
        refused(folder)
    message = str(caught.value)
    written = str(folder)
    assert written in message, f"the message does not show the path as written ({written}): {message}"
    assert "\\\\" not in message, f"a backslash in the path came out doubled: {message}"


def test_the_scratch_folder_goes_into_a_worked_example_as_it_is() -> None:
    directory = Path("C:\\Users\\runner\\Temp\\etl_tool_scratch_1")  # one name on Linux, a real path on Windows
    configuration = {"path": "{scratch}/out.csv", "sheets": [{"path": "{scratch}"}], "keep": "${scratch}"}
    assert _resolved_scratch(configuration, directory) == {
        "path": f"{directory}/out.csv",
        "sheets": [{"path": str(directory)}],
        "keep": "${scratch}",
    }


def test_the_whole_catalog_conforms_when_the_temporary_folder_has_a_backslash_in_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: every worked example that writes a file is run inside a folder like Windows gives."""
    monkeypatch.setattr(tempfile, "tempdir", str(_folder(tmp_path)))
    check_registry(REGISTRY)
