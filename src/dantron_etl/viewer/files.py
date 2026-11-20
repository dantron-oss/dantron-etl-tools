"""The files a pipeline's steps name: where each one is, what is known about it, and its first rows.

A setting is a file path when its tool's template marks it ``file_path``, so a new tool that reads
or writes a file is covered without a change here. A path is worked out the way a run works it out:
the step's settings after its rules are filled in, then ``with_base_dir`` from the folder the
viewer's runs read from.

The viewer only ever looks at a file found this way. A request names a pipeline, a step and a
setting; it never names a path.

Why: docs/design/viewer.md#looking-at-a-file-a-step-names
"""

from __future__ import annotations

import codecs
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import ConfigField, apply_defaults, with_base_dir
from ..spec import Pipeline
from ..tools.read_csv import _normalise_newlines, _parse_records, _skip_lines, _strip_bom

__all__ = [
    "PREVIEW_BYTES",
    "PREVIEW_COLUMNS",
    "PREVIEW_LINES",
    "PREVIEW_ROWS",
    "FileRefused",
    "NamedFile",
    "facts",
    "file_settings",
    "named_file",
    "named_files",
    "preview",
]

#: The most rows and columns a preview shows.
PREVIEW_ROWS = 50
PREVIEW_COLUMNS = 30
#: The most lines of a plain text file a preview shows.
PREVIEW_LINES = 100
#: How much of a text or CSV file is read for a preview. The rest is never read.
PREVIEW_BYTES = 1_000_000
#: A workbook bigger than this is not opened for a preview: it is read whole to find one sheet.
MAX_WORKBOOK_BYTES = 50_000_000

_TABLES = (".csv", ".tsv")
_WORKBOOKS = (".xlsx", ".xlsm", ".xls")


class FileRefused(LookupError):
    """A request for something that is not a file a step names. The message says what was asked."""


@dataclass(frozen=True)
class NamedFile:
    """One file a step names in one setting.

    ``written`` is the setting's value after rules are filled in. ``path`` is the full path a run
    uses, or None when it cannot be worked out. ``config`` is the step's settings as a run sees them,
    defaults included, so a preview can read the file as the step does.
    """

    step: str
    setting: str
    tool: str
    written: str
    path: str | None
    config: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Which files a pipeline names
# ---------------------------------------------------------------------------


def file_settings(template: Mapping[str, ConfigField]) -> tuple[str, ...]:
    """The settings of a tool that hold a file path, in template order."""
    return tuple(name for name, spec in template.items() if spec.file_path)


def _step_files(spec: Pipeline, base: Path, step_id: str) -> list[NamedFile]:
    step = spec.step(step_id)
    registry = spec.registry
    assert registry is not None  # a Pipeline always fills it in
    template = registry.get(step.tool).config_template()
    names = file_settings(template)
    if not names:
        return []
    # The same call the runner makes, from the same folder, resolved as the runner resolves it.
    config = apply_defaults(template, with_base_dir(template, dict(step.config), Path(base).resolve()))
    found = []
    for name in names:
        written = step.config.get(name)
        if not isinstance(written, str) or not written:
            continue
        value = config[name]
        path = value if Path(value).is_absolute() else None
        found.append(NamedFile(step_id, name, step.tool, written, path, config))
    return found


def named_files(spec: Pipeline, base: Path) -> tuple[NamedFile, ...]:
    """Every file the steps of ``spec`` name, in run order. ``base`` is the folder a run reads from."""
    return tuple(one for step_id in spec.order() for one in _step_files(spec, base, step_id))


def named_file(spec: Pipeline, base: Path, step_id: str, setting: str) -> NamedFile:
    """The file step ``step_id`` names in ``setting``, or FileRefused when it names none there."""
    if not isinstance(step_id, str) or step_id not in {step.id for step in spec.steps}:
        raise FileRefused(f"pipeline '{spec.name}' has no step {step_id!r}")
    for one in _step_files(spec, base, step_id):
        if one.setting == setting:
            return one
    raise FileRefused(f"step '{step_id}' names no file in a setting called {setting!r}")


# ---------------------------------------------------------------------------
# What is known about a file
# ---------------------------------------------------------------------------


def _size(count: int) -> str:
    if count < 1024:
        return f"{count:,} bytes"
    for unit in ("KB", "MB", "GB"):
        count /= 1024
        if count < 1024 or unit == "GB":
            return f"{count:,.1f} {unit}"
    raise AssertionError("unreachable")


def _inside(path: str, folder: Path) -> bool:
    try:
        Path(path).relative_to(folder)
    except ValueError:
        return False
    return True


def facts(named: NamedFile, project_root: Path, *, opens: bool = True) -> dict[str, Any]:
    """Where the file is, whether it is there, its size and time, and those facts as sentences.

    ``opens`` says whether this machine has a file manager; without one, ``can_open`` is false.
    """
    shown: dict[str, Any] = {
        "step": named.step,
        "setting": named.setting,
        "tool": named.tool,
        "written": named.written,
        "path": named.path,
        "real_path": None,
        "exists": False,
        "is_folder": False,
        "folder_exists": False,
        "size": None,
        "modified": None,
        "outside_project": False,
        "can_open": opens and named.path is not None,
        "can_preview": False,
        "said": [],
    }
    said: list[str] = shown["said"]
    if named.path is None:
        said.append(
            f"The path {named.written} holds a {{...}} placeholder that only a check of the tool fills in, "
            "so the viewer cannot work out where the file is."
        )
        return shown

    real = os.path.realpath(named.path)
    if real != os.path.normpath(named.path):
        shown["real_path"] = real
        said.append(f"This path is a link. The file it leads to is {real}.")
    shown["outside_project"] = not _inside(real, Path(os.path.realpath(project_root)))
    shown["folder_exists"] = os.path.isdir(os.path.dirname(real))
    if os.path.isdir(real):
        shown["is_folder"] = True
        said.append("A folder is at this path, not a file.")
    elif os.path.isfile(real):
        stat = os.stat(real)
        shown.update(exists=True, size=stat.st_size, can_preview=True)
        shown["modified"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        what = "The file is there but empty" if stat.st_size == 0 else f"The file is there: {_size(stat.st_size)}"
        said.append(f"{what}, last changed {shown['modified'][:16]}.")
    elif shown["folder_exists"]:
        said.append("There is no file here yet. Its folder is there.")
    else:
        said.append("There is no file here, and its folder is not there either.")
    if shown["outside_project"]:
        said.append(f"It is outside the project folder, {project_root}. The pipeline names it there.")
    return shown


# ---------------------------------------------------------------------------
# A first look inside
# ---------------------------------------------------------------------------


def _nothing(path: str | None, note: str) -> dict[str, Any]:
    return {"kind": "none", "path": path, "note": note}


def _cut_note(rows_shown: int, rows_total: int | None, columns_shown: int, columns_total: int) -> str:
    if rows_total is None:
        rows = f"the first {rows_shown:,} rows of a file too long to read whole here"
    elif rows_shown < rows_total:
        rows = f"{rows_shown:,} of {rows_total:,} rows"
    else:
        rows = f"all {rows_total:,} rows"
    columns = f"{columns_shown:,} of {columns_total:,} columns" if columns_shown < columns_total else "every column"
    return f"Showing {rows} and {columns}."


def _head(path: str) -> tuple[bytes, bool]:
    """The first bytes of a file, and whether there was more after them."""
    with open(path, "rb") as handle:
        data = handle.read(PREVIEW_BYTES + 1)
    return data[:PREVIEW_BYTES], len(data) > PREVIEW_BYTES


def _decoded(data: bytes, encoding: str, cut: bool) -> str:
    """``data`` as text. When the file was cut, a character split at the cut is left off, not refused."""
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    return decoder.decode(data, final=not cut)


def _table(named: NamedFile, path: str) -> dict[str, Any]:
    """A CSV split as the step that names it splits it: its delimiter, encoding, skip and header."""
    config = named.config
    delimiter = config.get("delimiter")
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        delimiter = ","
    encoding = str(config.get("encoding") or "utf8")
    skip_rows = config.get("skip_rows")
    skip_rows = skip_rows if isinstance(skip_rows, int) and skip_rows >= 0 else 0
    has_header = config.get("has_header", True) is not False

    data, cut = _head(path)
    if b"\x00" in data:
        return _nothing(path, "The viewer cannot show this file: it holds bytes that are not text.")
    try:
        text = _decoded(data, encoding, cut)
    except (UnicodeDecodeError, LookupError) as error:
        return _nothing(path, f"The viewer cannot show this file as {encoding} text: {error}.")
    text = _normalise_newlines(_strip_bom(text))
    if cut:
        text = text[: text.rfind("\n") + 1]  # a line split at the cut is left off
    records = [fields for _, fields in _parse_records(_skip_lines(text, skip_rows), delimiter, skip_rows)]
    if not records:
        return _nothing(path, f"There is nothing to show after skipping {skip_rows} line(s).")
    if has_header:
        columns, rows = records[0], records[1:]
    else:
        columns, rows = [f"column_{n + 1}" for n in range(len(records[0]))], records
    total = None if cut else len(rows)
    shown_rows = [[cell if cell != "" else None for cell in row[:PREVIEW_COLUMNS]] for row in rows[:PREVIEW_ROWS]]
    shown_columns = columns[:PREVIEW_COLUMNS]
    return {
        "kind": "table",
        "path": path,
        "columns": shown_columns,
        "rows": shown_rows,
        "rows_total": total,
        "columns_total": len(columns),
        "note": _cut_note(len(shown_rows), total, len(shown_columns), len(columns)),
    }


def _letters(index: int) -> str:
    """A spreadsheet column's letters: 0 is A, 26 is AA."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(65 + rest) + name
    return name


def _workbook(named: NamedFile, path: str, sheet: str | None) -> dict[str, Any]:
    """The sheet names, and the first rows of one sheet exactly as its cells hold them."""
    import fastexcel

    if os.path.getsize(path) > MAX_WORKBOOK_BYTES:
        return _nothing(path, f"The viewer cannot show this workbook: it is over {_size(MAX_WORKBOOK_BYTES)}.")
    try:
        reader = fastexcel.read_excel(path)
        names = list(reader.sheet_names)
    except Exception as error:  # noqa: BLE001 -- any reason a workbook will not open is told, not raised
        return _nothing(path, f"The viewer cannot open this workbook: {error}")
    if sheet is None:
        wanted = named.config.get("sheet")
        sheet = wanted if isinstance(wanted, str) and wanted in names else (names[0] if names else None)
    elif sheet not in names:
        raise FileRefused(f"the workbook has no sheet {sheet!r}; it has: {', '.join(names)}")
    shown: dict[str, Any] = {"kind": "workbook", "path": path, "sheets": names, "sheet": sheet}
    if sheet is None:
        return {**shown, "columns": [], "rows": [], "note": "The workbook has no sheets."}
    loaded = reader.load_sheet(
        sheet,
        header_row=None,
        skip_rows=0,
        n_rows=PREVIEW_ROWS,
        use_columns=lambda column: column.index < PREVIEW_COLUMNS,
        dtypes="string",
    )
    frame = loaded.to_polars()
    total_rows, total_columns = loaded.total_height, loaded.width
    rows = [list(row) for row in frame.iter_rows()]
    columns = [_letters(n) for n in range(frame.width)]
    if not rows:
        return {**shown, "columns": [], "rows": [], "note": f"Sheet {sheet} is empty."}
    note = _cut_note(len(rows), total_rows, len(columns), max(total_columns, len(columns)))
    return {**shown, "columns": columns, "rows": rows, "note": note}


def _text(path: str) -> dict[str, Any]:
    data, cut = _head(path)
    if b"\x00" in data:
        return _nothing(path, "The viewer cannot show this file: it holds bytes that are not text.")
    try:
        text = _decoded(data, "utf8", cut)
    except UnicodeDecodeError:
        return _nothing(path, "The viewer cannot show this file: it is not UTF-8 text.")
    lines = _normalise_newlines(_strip_bom(text)).split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    more = cut or len(lines) > PREVIEW_LINES
    note = f"Showing the first {PREVIEW_LINES:,} lines." if more else f"Showing all {len(lines):,} lines."
    return {"kind": "text", "path": path, "lines": lines[:PREVIEW_LINES], "more": more, "note": note}


def preview(named: NamedFile, sheet: str | None = None) -> dict[str, Any]:
    """The first rows of the file, read the way its step reads it, or why there is nothing to show.

    A CSV is a table, a workbook its sheet names and one sheet's first rows, any other file its first
    lines of text. Only the file the step names is read, through any link it is.
    """
    if named.path is None:
        return _nothing(None, "The viewer cannot work out where this file is.")
    path = os.path.realpath(named.path)
    if not os.path.isfile(path):
        return _nothing(path, "There is no file here to show.")
    if os.path.getsize(path) == 0:
        return _nothing(path, "The file is empty.")
    suffix = Path(path).suffix.lower()
    if named.path.lower().endswith(_TABLES) or suffix in _TABLES:
        return _table(named, path)
    if named.path.lower().endswith(_WORKBOOKS) or suffix in _WORKBOOKS:
        return _workbook(named, path, sheet)
    return _text(path)
