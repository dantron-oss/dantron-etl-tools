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
    check_output_column,
    scratch_path,
)

_EXAMPLE_PATH: Final[str] = f"{SCRATCH_TOKEN}/rows.xlsx"
_MONTHS_EXAMPLE_PATH: Final[str] = f"{SCRATCH_TOKEN}/months.xlsx"


class ReadExcel:
    """Read one named sheet, or every sheet whose name matches a pattern, onto the ``rows`` port."""

    name = "read_excel"
    description = "Read one named sheet of an XLSX workbook, or stack every sheet whose name matches a pattern."
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
                description="The name of the one sheet to read; never an index, never guessed. "
                "Give this or sheets_matching, not both.",
                required=False,
                default=None,
                non_empty=True,
            ),
            "sheets_matching": ConfigField(
                type="string",
                description="A pattern instead of a name. Every sheet whose name contains a match is "
                "read, and the sheets are stacked in workbook order. Case-sensitive; anchor with ^ "
                "and $ to match whole names.",
                required=False,
                default=None,
                non_empty=True,
            ),
            "sheet_column": ConfigField(
                type="string",
                description="Name for the column saying which sheet each row came from. Required "
                "with sheets_matching, and refused without it.",
                required=False,
                default=None,
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
                "than an empty string. The sheet is always named, in `sheet`, or found by a pattern, "
                'in `sheets_matching`. It is never an index: "the first tab" silently reads the '
                "wrong sheet the moment a delivered pack puts a cover sheet in front of the figures. "
                "A row blank in every column is dropped as layout rather than kept as a row of nulls."
            ),
            examples=(_reads_a_sheet_example(), _reads_matching_sheets_example()),
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
                "sheets_matching reads every sheet whose name contains a match for the pattern, the "
                "same test as regex_match() in an expression. A pattern that matches no sheet is "
                "refused with the sheets that are there. Each sheet is read exactly as naming it in "
                "'sheet' would read it, and a sheet that does not match is never read.",
                "Stacked sheets are lined up by column name. A column a sheet lacks is null on that "
                "sheet's rows, and columns come in the order they first appear. Whole and fractional "
                "figures stack as fractional figures. A sheet where a column holds no values takes "
                "the other sheets' type. Any other difference, such as text on one sheet and figures "
                "on another, is refused, naming the column and the sheets.",
            ),
            field_help={
                "sheet": "By name. Refused, with the sheet names present, if it does not match one of "
                "them exactly. Give this or sheets_matching.",
                "skip_rows": "Rows above the header (or the data) to skip; must not be negative.",
                "sheets_matching": "Found anywhere in the sheet name, like regex_match(). Refused if no sheet matches.",
                "sheet_column": "Comes after every column read, and holds the sheet's name as text.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong format or a negative skip count before a workbook is ever opened."""
        path = config["path"]
        if not path.lower().endswith(".xlsx"):
            raise ConfigError(f"read_excel only reads .xlsx workbooks; got path {path!r}")
        if config["skip_rows"] < 0:
            raise ConfigError(f"config field 'skip_rows' must not be negative, got {config['skip_rows']}")

        sheet = config.get("sheet")
        pattern = config.get("sheets_matching")
        column = config.get("sheet_column")
        if (sheet is None) == (pattern is None):
            given = "both" if sheet is not None else "neither"
            raise ConfigError(
                f"read_excel needs exactly one of config fields 'sheet' and 'sheets_matching', but "
                f"{given} was given; the sheet to read is never guessed"
            )
        if sheet is not None:
            if column is not None:
                raise ConfigError(
                    f"config field 'sheet_column' names {column!r}, but it is only used with "
                    f"'sheets_matching'; one named sheet needs no column saying which sheet it is"
                )
            return
        if column is None:
            raise ConfigError(
                "config field 'sheet_column' is required with 'sheets_matching': without it, rows "
                "from different sheets cannot be told apart"
            )
        check_output_column(pl.DataFrame(), column, "sheet_column", replace=None)
        try:
            pl.Series([], dtype=pl.String).str.contains(pattern)
        except pl.exceptions.PolarsError as error:
            raise ConfigError(
                f"config field 'sheets_matching' is not a pattern the engine accepts: {error}, in pattern: `{pattern}`"
            ) from error

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        path = config["path"]
        sheet = config.get("sheet")
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
        if sheet is None:
            return Ports({"rows": _read_matching(reader, config)})
        if sheet not in sheet_names:
            raise ConfigError(
                f"read_excel found no sheet named {sheet!r} in {path!r}; the sheets present are: "
                f"{', '.join(sheet_names)}"
            )
        return Ports({"rows": _read_sheet(reader, path, sheet, has_header, skip_rows)})


def _read_matching(reader: Any, config: Mapping[str, Any]) -> pl.DataFrame:
    """Every sheet whose name matches ``sheets_matching``, read one by one and stacked in workbook order."""
    path = config["path"]
    pattern = config["sheets_matching"]
    column = config["sheet_column"]
    sheet_names: list[str] = list(reader.sheet_names)
    hits = pl.Series(sheet_names, dtype=pl.String).str.contains(pattern).to_list()
    matched = [name for name, hit in zip(sheet_names, hits, strict=True) if hit]
    if not matched:
        raise ConfigError(
            f"config field 'sheets_matching' pattern `{pattern}` matches no sheet in {path!r}; the "
            f"sheets present are: {', '.join(sheet_names)}"
        )

    frames: list[tuple[str, pl.DataFrame]] = []
    for name in matched:
        frame = _read_sheet(reader, path, name, config["has_header"], config["skip_rows"])
        check_output_column(frame, column, "sheet_column", replace=None)
        frames.append((name, frame))

    order: list[str] = []
    for _, frame in frames:
        order.extend(item for item in frame.columns if item not in order)
    types = {item: _stacked_type(item, frames) for item in order}

    stacked = [
        frame.select(
            *[
                pl.col(item).cast(types[item]) if item in frame.columns else pl.lit(None, dtype=types[item]).alias(item)
                for item in order
            ],
            pl.lit(name, dtype=pl.String).alias(column),
        )
        for name, frame in frames
    ]
    return pl.concat(stacked, how="vertical")


def _stacked_type(column: str, frames: Sequence[tuple[str, pl.DataFrame]]) -> pl.DataType:
    """The one type ``column`` takes once every sheet is stacked, or a refusal naming two sheets that disagree.

    A sheet where the column holds no values says nothing about its type, so it is left out. Whole
    and fractional figures together become fractional figures. Anything else that differs is refused.
    """
    present = [(name, frame[column]) for name, frame in frames if column in frame.columns]
    typed = [(name, values.dtype) for name, values in present if values.null_count() < values.len()]
    if not typed:
        return present[0][1].dtype
    first_name, first_type = typed[0]
    if all(dtype == first_type for _, dtype in typed):
        return first_type
    if all(dtype.is_numeric() for _, dtype in typed):
        return pl.Float64 if any(dtype.is_float() for _, dtype in typed) else pl.Int64
    other_name, other_type = next((name, dtype) for name, dtype in typed if dtype != first_type)
    raise ConfigError(
        f"read_excel cannot stack column {column!r}: sheet {first_name!r} holds it as {first_type} and "
        f"sheet {other_name!r} holds it as {other_type}; converting one to the other would change what "
        f"every later step compares it with"
    )


def _read_sheet(reader: Any, path: str, sheet: str, has_header: bool, skip_rows: int) -> pl.DataFrame:
    """One sheet, read exactly as naming it in ``sheet`` reads it."""
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

    return frame


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


@contextmanager
def _months_workbook() -> Iterator[None]:
    """A workbook of a cover sheet and two month sheets, for the stacking example."""
    target = scratch_path(_MONTHS_EXAMPLE_PATH)
    with xlsxwriter.Workbook(str(target)) as workbook:
        workbook.add_worksheet("Cover").write_row(0, 0, ["Monthly report"])
        for name, rows in {"Feb 2026": [["f1", 2.5]], "Jan 2026": [["j1", 10.25], ["j2", 4.0]]}.items():
            worksheet = workbook.add_worksheet(name)
            worksheet.write_row(0, 0, ["key", "amount"])
            for index, row in enumerate(rows, start=1):
                worksheet.write_row(index, 0, row)
    yield


def _reads_matching_sheets_example() -> WorkedExample:
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": ["f1", "j1", "j2"],
                    "amount": [2.5, 10.25, 4.0],
                    "sheet": ["Feb 2026", "Jan 2026", "Jan 2026"],
                }
            )
        }
    )
    return WorkedExample(
        scenario="read every month sheet of a report into one frame",
        configuration={
            "path": _MONTHS_EXAMPLE_PATH,
            "sheets_matching": r"^[A-Z][a-z]{2} \d{4}$",
            "sheet_column": "sheet",
        },
        explanation="Both month sheets match the pattern and the cover sheet does not. The months are "
        "stacked in the order the workbook holds them, and 'sheet' says which one each row came from.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_months_workbook,
        check_row_order=True,
    )


REGISTRY.register(ReadExcel())
