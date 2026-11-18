"""WriteExcel: one frame becomes one sheet of one XLSX workbook, and the same rows pass through.

The catalog's second file sink, and the twin of ``write_table`` in one respect and its opposite in
another. Both refuse to overwrite something already there unless ``replace`` says so in writing, and
both hand their rows on unchanged rather than emitting a receipt, for the reason spelled out in
``write_table``'s own module docstring: every field a receipt would carry is derivable from the
configuration and the frame, so a receipt would add shape rather than information. Where the two
differ is what an empty frame means to the destination: replacing a table with nothing empties it,
which is why ``write_table`` refuses an empty frame by default, while a file with nothing in it is
still a workbook somebody can open and see was empty -- so this tool writes a header-only sheet
instead of refusing.

A cell is typed, and that typing is the whole reason this is not a delimited-text sink wearing a
different extension. A double cannot hold a whole number past 2**53 - 1 exactly, so one is refused
rather than silently rounded or silently turned into text behind the frame's back. A sheet holds at
most 1,048,576 rows, the header among them, so a frame taller than that is refused rather than
truncated -- truncation is the one outcome a delivered figure can least afford. And a date is written
with a number format, not because a format is decoration here but because it is the only thing that
tells a cell it holds a date rather than the count of days since 1899; every other number format --
a decimal count, a thousands separator, a colour for a negative figure -- is a claim about how the
sheet will be read that this tool is not entitled to make, so none of it is applied, and the workbook
carries no table object either.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import polars as pl
import xlsxwriter

from ..authoring import (
    REGISTRY,
    SCRATCH_TOKEN,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    scratch_path,
)
from ._spreadsheet import (
    FORBIDDEN_SHEET_CHARACTERS,
    MAX_DATA_ROWS,
    MAX_EXACT_INTEGER,
    MAX_SHEET_NAME_LENGTH,
    MAX_SHEET_ROWS,
    require_xlsx_path,
    unsupported_dtype_reason,
)

_PLAIN_PATH: Final[str] = f"{SCRATCH_TOKEN}/plain.xlsx"
_OVERWRITE_PATH: Final[str] = f"{SCRATCH_TOKEN}/overwrite.xlsx"


class WriteExcel:
    """Write ``rows`` to one named sheet of one XLSX workbook, and hand the same rows on."""

    name = "write_excel"
    description = "Write `rows` to one sheet of an XLSX workbook, and hand the same rows on."
    category = "output"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="Where the workbook is written; must end in .xlsx.",
                required=True,
                non_empty=True,
            ),
            "sheet": ConfigField(
                type="string",
                description="The name of the one sheet the rows are written to; at most 31 "
                "characters, and none of '[' ']' ':' '*' '?' '/' '\\'.",
                required=True,
                non_empty=True,
            ),
            "replace": ConfigField(
                type="boolean",
                description="Whether a file already at 'path' may be overwritten; leave it out and "
                "an existing file is refused.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "'rows' is written to the named sheet of the workbook at 'path', with a header row "
                "and every row of data below it; the same rows leave on the output port unchanged, "
                "so this sink is composable rather than terminal by construction. An existing file "
                "at 'path' is never overwritten unless 'replace' says so in writing -- the same "
                "keystroke this library already uses for a column already in use, though writing "
                "true when nothing occupies the path is accepted rather than refused, so one "
                "configuration writes a report's first run and overwrites every run after it. An "
                "empty frame still writes a header-only sheet, because a file with nothing in it is "
                "a workbook somebody can open, where the file this tool would otherwise not write is "
                "indistinguishable from a run that never happened -- the opposite of the sibling "
                "table sink, which refuses an empty frame because replacing a table with nothing "
                "empties it."
            ),
            examples=(_plain_write_example(), _overwrite_example()),
            notes=(
                "An integer past 2**53 - 1 in either direction is refused rather than rounded or "
                "silently written as text: a cell holds a double, so exactness past that point is a "
                "promise this tool cannot keep, and the fix is one explicit cast upstream, in "
                "writing, rather than a silent precision loss here.",
                "A Time, a Duration, a Decimal, a list or a nested record is refused rather than "
                "stringified: each has a plausible-looking wrong answer in a cell, and none of them "
                "is the value it started as.",
                "A frame taller than a sheet -- more than 1,048,575 rows of data -- is refused "
                "rather than truncated, because a truncated total is the worst outcome a delivered "
                "figure can have.",
                "A date column gets a date number format, because that format is what tells a "
                "person's spreadsheet the cell is a date rather than a count of days; a numeric "
                "column gets no format at all, because any decimal count this tool chose would be a "
                "claim about precision it is not entitled to make.",
            ),
            field_help={
                "replace": "Defaults to false: a path already occupied is refused until an author "
                "says in writing that overwriting it is intended.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong format, an unwritable sheet name, or an overwrite nobody authorised.

        All three are answerable from the configuration alone, before a workbook is ever opened, so
        each is refused here rather than partway through a write that would otherwise have to be
        undone.
        """
        path = config["path"]
        require_xlsx_path("write_excel", path)

        _validate_sheet_name(config["sheet"])

        if Path(path).exists() and not config["replace"]:
            raise ConfigError(
                f"write_excel refuses to overwrite the file already at '{path}'; set replace: true "
                "to overwrite it deliberately"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a frame too tall for a sheet, a dtype a cell cannot hold, or a value it cannot keep.

        Every refusal here is answerable from the schema or from a column's extremes, so every one
        of them happens before the workbook is opened -- a refused write must not leave a half
        written file behind, and the only way to keep that promise is to know everything is writable
        before the first byte is.
        """
        frame = inputs["rows"]

        if frame.height > MAX_DATA_ROWS:
            raise DataError(
                f"write_excel refuses a frame of {frame.height:,} row(s); a sheet's header takes "
                f"one row, leaving {MAX_DATA_ROWS:,} row(s) for data (a worksheet holds at most "
                f"{MAX_SHEET_ROWS:,} rows in total)",
                offending=frame.head(0),
                total=frame.height,
            )

        for name, dtype in frame.schema.items():
            reason = unsupported_dtype_reason(dtype)
            if reason is not None:
                raise ConfigError(
                    f"write_excel cannot write column {name!r} of dtype {dtype}: {reason}; format "
                    "it into something a cell can hold, one step upstream"
                )

        for name, dtype in frame.schema.items():
            if not dtype.is_integer():
                continue
            offending = frame.filter(pl.col(name).abs() > MAX_EXACT_INTEGER)
            if offending.height:
                raise DataError(
                    f"write_excel refuses column {name!r}: a spreadsheet cell holds a double, and "
                    f"cannot keep a whole number larger than {MAX_EXACT_INTEGER} in magnitude "
                    "exactly",
                    offending=offending,
                    total=frame.height,
                )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        _write_workbook(frame, config["path"], config["sheet"])
        return Ports({"rows": frame})


def _validate_sheet_name(sheet: str) -> None:
    """Refuse a sheet name a workbook cannot carry, in this library's own words.

    Left to the writer underneath, the same mistake fails part way through a run with a message
    about worksheets rather than about the flow that named one; caught here, before anything opens,
    the refusal names the rule that was broken and the name that broke it.
    """
    if len(sheet) > MAX_SHEET_NAME_LENGTH:
        raise ConfigError(
            f"write_excel refuses sheet name {sheet!r}: a workbook tab name may be at most "
            f"{MAX_SHEET_NAME_LENGTH} characters"
        )
    for character in FORBIDDEN_SHEET_CHARACTERS:
        if character in sheet:
            raise ConfigError(
                f"write_excel refuses sheet name {sheet!r}: a workbook tab name may not contain "
                f"the character {character}"
            )


def _write_workbook(frame: pl.DataFrame, path: str, sheet: str) -> None:
    """Write ``frame`` to ``sheet``, cell by cell, adding only the one format this tool claims to add.

    Written directly with ``xlsxwriter`` rather than through ``DataFrame.write_excel``, because that
    convenience applies a decimal number format and an autofiltered table object to every column
    whether or not anybody asked for either -- exactly the presentation this tool's rulings refuse.
    """
    with xlsxwriter.Workbook(path) as workbook:
        worksheet = workbook.add_worksheet(sheet)
        date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
        datetime_format = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})

        columns = frame.columns
        for column_index, name in enumerate(columns):
            worksheet.write(0, column_index, name)

        cell_formats = {
            name: date_format
            if isinstance(dtype, pl.Date)
            else datetime_format
            if isinstance(dtype, pl.Datetime)
            else None
            for name, dtype in frame.schema.items()
        }

        for row_index, row in enumerate(frame.iter_rows(), start=1):
            for column_index, (name, value) in enumerate(zip(columns, row)):
                if value == "":
                    value = None
                cell_format = cell_formats[name]
                if cell_format is not None and value is not None:
                    worksheet.write_datetime(row_index, column_index, value, cell_format)
                else:
                    worksheet.write(row_index, column_index, value)


@contextmanager
def _overwrite_setup() -> Iterator[None]:
    """A file already occupying the path, standing in for last month's workbook.

    Planted with :func:`scratch_path`, which resolves the same ``{scratch}`` token the example's own
    configuration holds, into the directory :func:`check_tool` opens for the run. Nothing here tears
    the file back down: the directory is the checker's, and it takes whatever a setup left inside it.
    """
    target = scratch_path(_OVERWRITE_PATH)
    target.write_bytes(b"last month's workbook")
    yield


def _plain_write_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})
    return WorkedExample(
        scenario="write a new workbook where nothing exists yet",
        configuration={"path": _PLAIN_PATH, "sheet": "Data"},
        explanation="'replace' is left out, and the path is empty, so no permission is needed; the "
        "same two rows leave on the output port unchanged.",
        inputs=Ports({"rows": frame}),
        outputs=Ports({"rows": frame}),
        check_row_order=True,
    )


def _overwrite_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})
    return WorkedExample(
        scenario="overwrite a workbook that is already there, on purpose",
        configuration={"path": _OVERWRITE_PATH, "sheet": "Data", "replace": True},
        explanation="A file already occupies the path; 'replace: true' says in writing that "
        "overwriting it is intended, and the same two rows leave on the output port unchanged "
        "either way.",
        inputs=Ports({"rows": frame}),
        outputs=Ports({"rows": frame}),
        check_row_order=True,
        setup=_overwrite_setup,
    )


REGISTRY.register(WriteExcel())
