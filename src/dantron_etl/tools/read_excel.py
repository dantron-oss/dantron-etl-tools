"""ReadExcel: one named tab of one XLSX workbook becomes one frame, with no tab guessed.

The catalog's second file source, and the mirror image of ``read_table``'s own refusal to accept a
query: there, a schema and a table are the whole of what may be named, because a query duplicates
Filter and SelectColumns in a second language nothing here can pre-flight. Here, a sheet is the whole
of what may be named, and deliberately not an index -- "the first tab" is a positional read wearing
a default's clothes, and a delivered pack is laid out with a cover sheet in front of the figures for
exactly the audience who never configures a flow. A tab named in the configuration turns that failure
into a refusal instead: a name that is not there is refused with the names that are, in the same
words this library already uses for a column that is not on a port, and a near miss -- a wrong case,
a trailing space -- is not a match.

What comes back is what the sheet holds, not what somebody once put in. A column of whole numbers
written as floats comes back whole, and a column mixing text and figures comes back entirely as
text, because a cell that once held ``10.25`` next to a cell that holds ``"not a number"`` cannot be
un-mixed by guessing -- a reader that tried would be right most months and wrong in a way no run
reports. A header row is a sheet's schema, so a repeated heading or a blank one is refused rather
than repaired: this library's own vocabulary for either -- ``key_1``, ``__UNNAMED__1`` -- would
otherwise leak into a flow's own column names, silently, the moment a downstream step names the
column that came first.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import fastexcel
import polars as pl
import xlsxwriter

from ..tool import (
    REGISTRY,
    SCRATCH_TOKEN,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    scratch_path,
)

_EXAMPLE_PATH: Final[str] = f"{SCRATCH_TOKEN}/rows.xlsx"


class ReadExcel:
    """Read one named sheet of an XLSX workbook onto the ``rows`` port."""

    name = "read_excel"
    description = "Read one named sheet of an XLSX workbook onto the `rows` port."
    category = "input"
    input_ports = ()
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="The workbook to read; must end in .xlsx.",
                required=True,
                non_empty=True,
            ),
            "sheet": ConfigField(
                type="string",
                description="The name of the one sheet to read; never an index, never guessed.",
                required=True,
                non_empty=True,
            ),
            "has_header": ConfigField(
                type="boolean",
                description="Whether the sheet's first row (after skip_rows) names the columns.",
                default=True,
            ),
            "skip_rows": ConfigField(
                type="integer",
                description="How many rows above the header (or the data, if has_header is false) "
                "to skip -- for a title block laid out above the table.",
                default=0,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "The named sheet arrives on `rows` with the columns and the dtypes the sheet "
                "itself holds: a column of round figures comes back whole, a column mixing text "
                "and figures comes back entirely as text, and a blank cell comes back null rather "
                "than an empty string. `sheet` is required and is never an index, because "
                "defaulting to the first tab is a positional read that fails silently the moment a "
                "delivered pack puts a cover sheet in front of the figures. A row blank in every "
                "column is dropped as layout rather than kept as a row of nulls."
            ),
            examples=(_reads_a_sheet_example(),),
            notes=(
                "A sheet named in the configuration that is not in the workbook is refused with "
                "the sheet names that are there; a name differing only in case or a trailing space "
                "is a different name, not a near match.",
                "has_header defaults to true, since a spreadsheet's first row is a header until an "
                "author says otherwise; a duplicated or a blank heading is refused rather than "
                "repaired, because a sheet's header row is its schema.",
                "There is no cell range: skip_rows reaches a header sitting below a title block, "
                "and everything else about narrowing a read -- which columns, which rows -- is a "
                "SelectColumns or a Filter put after this tool.",
            ),
            field_help={
                "sheet": "Required, by name; refused with the sheet names present if it does not "
                "match one of them exactly.",
                "skip_rows": "Rows above the header (or the data) to skip; must not be negative.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong format or a negative skip count before a workbook is ever opened."""
        path = config["path"]
        if not path.lower().endswith(".xlsx"):
            raise ConfigError(f"read_excel only reads .xlsx workbooks; got path {path!r}")
        if config["skip_rows"] < 0:
            raise ConfigError(f"config field 'skip_rows' must not be negative, got {config['skip_rows']}")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        path = config["path"]
        sheet = config["sheet"]
        has_header = config["has_header"]
        skip_rows = config["skip_rows"]

        target = Path(path)
        if not target.exists():
            raise ConfigError(f"read_excel found no file at {path!r}")

        try:
            reader = fastexcel.read_excel(path)
        except Exception as error:
            raise ConfigError(f"read_excel could not open {path!r} as an XLSX workbook: {error}") from error

        sheet_names = reader.sheet_names
        if sheet not in sheet_names:
            raise ConfigError(
                f"read_excel found no sheet named {sheet!r} in {path!r}; the sheets present are: "
                f"{', '.join(sheet_names)}"
            )

        try:
            raw_header = reader.load_sheet_by_name(sheet, header_row=None, skip_rows=skip_rows, n_rows=1).to_polars()
        except Exception as error:
            raise ConfigError(f"read_excel could not read sheet {sheet!r} in {path!r}: {error}") from error

        if raw_header.height == 0:
            raise ConfigError(f"read_excel found nothing on sheet {sheet!r} in {path!r}")

        if has_header:
            _validate_header_row(raw_header.row(0), sheet)

        read_options: dict[str, Any] = {"header_row": skip_rows if has_header else None}
        if not has_header:
            read_options["skip_rows"] = skip_rows

        try:
            frame = pl.read_excel(
                path,
                sheet_name=sheet,
                has_header=has_header,
                read_options=read_options,
                drop_empty_cols=False,
                raise_if_empty=False,
            )
        except Exception as error:
            raise ConfigError(f"read_excel could not read sheet {sheet!r} in {path!r}: {error}") from error

        return Ports({"rows": frame})


def _validate_header_row(raw_header: Sequence[Any], sheet: str) -> None:
    """Refuse a header row that repeats or omits a name, rather than repair it.

    Left alone, the reader underneath renames a repeated heading to ``key_1`` and a blank one to
    ``__UNNAMED__1`` -- its own vocabulary, leaking into a flow's column names the moment a
    downstream step names the column that happened to come first.
    """
    seen: set[Any] = set()
    for index, value in enumerate(raw_header, start=1):
        if value is None or (isinstance(value, str) and value.strip() == ""):
            raise ConfigError(
                f"read_excel refuses sheet {sheet!r}: header cell {index} is blank, and a header cannot be guessed"
            )
        if value in seen:
            raise ConfigError(
                f"read_excel refuses sheet {sheet!r}: header names {value!r} more than once, and a "
                "duplicate cannot be told apart by guessing"
            )
        seen.add(value)


@contextmanager
def _example_workbook() -> Iterator[None]:
    """A small workbook the example reads, entered around the example and torn down after.

    Built here rather than inside ``help()``, which must describe a tool without changing the
    process doing the describing -- the catalog page generator calls ``help()`` on every tool, and
    writing a file as a side effect of describing one is exactly the disturbance the boundary now
    refuses. Planted with :func:`scratch_path`, into the directory :func:`check_tool` opens for the
    run; nothing here tears it back down, because that directory is the checker's to remove.
    """
    target = scratch_path(_EXAMPLE_PATH)
    with xlsxwriter.Workbook(str(target)) as workbook:
        worksheet = workbook.add_worksheet("Sample")
        worksheet.write_row(0, 0, ["key", "label"])
        worksheet.write_row(1, 0, [1, "first"])
        worksheet.write_row(2, 0, [2, "second"])
    yield


def _reads_a_sheet_example() -> WorkedExample:
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "label": ["first", "second"]})})
    return WorkedExample(
        scenario="read a two-row sheet named in the configuration",
        configuration={"path": _EXAMPLE_PATH, "sheet": "Sample"},
        explanation="'sheet' names the tab to read; the columns and the dtypes arrive exactly as the sheet holds them.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_example_workbook,
    )


REGISTRY.register(ReadExcel())
