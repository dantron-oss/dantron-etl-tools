"""Importing the package must not load the Excel libraries; only the Excel tools need them."""

import subprocess
import sys
from pathlib import Path

import pytest

from dantron_etl import ConfigError, get_tool
from dantron_etl.ports import Ports

_SRC = Path(__file__).resolve().parents[2] / "src"

_EXCEL_MODULES = ("fastexcel", "xlsxwriter", "openpyxl")


def test_importing_the_package_loads_no_excel_library():
    script = (
        "import sys\n"
        "import dantron_etl\n"
        f"print(','.join(m for m in {_EXCEL_MODULES!r} if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": str(_SRC), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.stdout.strip() == ""


def test_read_excel_names_the_package_to_install_when_fastexcel_is_missing(monkeypatch, tmp_path):
    """A missing reader is refused by name, in the words db.py uses for a missing driver."""
    monkeypatch.setitem(sys.modules, "fastexcel", None)
    tool = get_tool("read_excel")
    config = {"path": str(tmp_path / "book.xlsx"), "sheet": "Sample", "has_header": True, "skip_rows": 0}
    with pytest.raises(ConfigError, match="install 'fastexcel'"):
        tool.execute(Ports({}), config)
