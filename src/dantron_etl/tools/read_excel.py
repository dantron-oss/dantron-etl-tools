"""ReadExcel: read one named sheet of an XLSX workbook, or stack the sheets matching a pattern.

The sheet is always named, never picked by position. Each column's type comes from its first 100
rows, and a cell that would read as null is refused. A bad header is refused, not repaired.

Why: docs/design/tools.md#read_excel
"""

import datetime as dt
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    SCRATCH_TOKEN,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    check_output_column,
    scratch_path,
)
from ._files import require_file_to_read

_EXAMPLE_PATH: Final[str] = f"{SCRATCH_TOKEN}/rows.xlsx"
_MONTHS_EXAMPLE_PATH: Final[str] = f"{SCRATCH_TOKEN}/months.xlsx"
_TEXT_EXAMPLE_PATH: Final[str] = f"{SCRATCH_TOKEN}/mixed.xlsx"

AS_TEXT: Final[str] = "as_text"

# Below this, every whole figure a float holds is exact and fits a 64-bit integer, so it can be
# written without a decimal point. Above it, the shortest round-trip spelling ("1e+20") is used.
_EXACT_WHOLE_LIMIT: Final[float] = float(2**53)


class ReadExcel:
    """Read one named sheet, or every sheet whose name matches a pattern, onto the `rows` port."""

    name = "read_excel"
    description = "Read a sheet of an Excel workbook into a table, or stack several matching sheets."
    category = "input"
    input_ports = ()
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="The workbook to read. It must end in .xlsx.",
                required=True,
                non_empty=True,
                file_path=True,
            ),
            "sheet": ConfigField(
                type="string",
                description="The name of the one sheet to read, such as 'Sales'. Give this or "
                "'sheets_matching'.",
                required=False,
                default=None,
                non_empty=True,
            ),
            "sheets_matching": ConfigField(
                type="string",
                description="A pattern instead of a name: every sheet whose name matches is read, and "
                "the sheets are stacked in workbook order.",
                required=False,
                default=None,
                non_empty=True,
            ),
            "sheet_column": ConfigField(
                type="string",
                description="With 'sheets_matching': the name of a column saying which sheet each row "
                "came from.",
                required=False,
                default=None,
            ),
            "has_header": ConfigField(
                type="boolean",
                description="Whether the first row (after skip_rows) holds the column names.",
                default=True,
            ),
            "skip_rows": ConfigField(
                type="integer",
                description="How many rows to skip before the header, such as a title block above the table.",
                default=0,
            ),
            AS_TEXT: ConfigField(
                type="boolean",
                description="Set true to read every cell as text, then convert columns with cast_columns.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use read_excel to start a flow from an Excel workbook. Name the sheet in 'sheet', or "
                "give a pattern in 'sheets_matching' to read and stack several sheets, such as one per "
                "month. The rows go to `rows` with the types the sheet holds: whole figures come back "
                "whole, and blank cells come back null. A row blank in every column is dropped."
            ),
            examples=(_reads_a_sheet_example(), _reads_matching_sheets_example(), _reads_as_text_example()),
            notes=(
                "A sheet is always named or matched, never picked by position, so a cover sheet added "
                "in front does not change what is read.",
                "A sheet name not in the workbook is refused, listing the names that are. A name that "
                "differs only in case or a trailing space is a different name.",
                "Each column's type is decided from its first 100 rows. A cell that holds something "
                "but would read as null is refused, naming the column and the row.",
                "That catches text further down a column of figures, and words read as missing, such "
                "as n/a, #N/A or null, anywhere in a column. Set as_text to read them.",
                "A cell of only spaces reads as null.",
                "The header row is refused, not repaired, if a heading is blank or repeated. The "
                "number 2026 and the text '2026' count as repeated.",
                "A heading that is a date or true/false is refused, because it cannot name a column.",
                "There is no cell range. skip_rows reaches a header below a title block; to narrow "
                "columns or rows, use select_columns or filter afterwards.",
                "sheets_matching reads every sheet whose name contains a match, the same test as "
                "regex_match(); anchor with ^ and $ for whole names. Matching is case sensitive.",
                "A pattern that matches no sheet is refused, listing the sheets. Each matched sheet is "
                "read exactly as naming it in 'sheet' would, and unmatched sheets are never read.",
                "Stacked sheets are lined up by column name. A column a sheet lacks is null on that "
                "sheet's rows, and columns come in the order they first appear.",
                "When stacking, whole and decimal figures become decimals, and a sheet where a column "
                "is empty takes the others' type. Any other difference, such as text on one sheet and "
                "figures on another, is refused, naming the column and the sheets.",
                "as_text makes every column text, each cell as what it stores, not as Excel displays "
                "it: 12, never 12.0; other figures as the shortest text that converts back exactly.",
                "With as_text, a date reads YYYY-MM-DD, a date with a time YYYY-MM-DDTHH:MM:SS, "
                "true/false as true or false, and a blank as null. In a mixed column, each cell keeps "
                "its own text.",
                "Excel stores dates and times as one number, so with as_text a time of exactly midnight "
                "reads as a date, and a cell holding only a time reads as that time on 31 December 1899 "
                "(14:05 reads 1899-12-31T14:05:00).",
                "as_text reads the sheet four times (as text, figures, true/false and dates) and takes "
                "each cell from the right reading, so a large sheet takes a few times longer.",
                "The text reading alone would be quicker, but it rounds figures: 99999999999.99 comes "
                "back 99999999999.990005493.",
            ),
            field_help={
                "path": "For example 'data/sales.xlsx'.",
                "sheet": "For example 'Sales'. It must match exactly.",
                "skip_rows": "For example 3, to skip a title block. Not negative.",
                "sheets_matching": "For example '^[A-Z][a-z]{2} \\d{4}$' for sheets named like 'Jan 2026'.",
                "sheet_column": "For example 'sheet'. It is added after the other columns and holds the "
                "sheet's name.",
                AS_TEXT: "Use it when a column mixes figures and text, then say what each column is with "
                "cast_columns.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a non-.xlsx path, a negative skip, and a sheet setting that is missing, doubled or misplaced."""
        path = config["path"]
        if not path.lower().endswith(".xlsx"):
            raise ConfigError(f"read_excel only reads .xlsx workbooks; got path '{path}'")
        if not isinstance(config.get(AS_TEXT, False), bool):
            raise ConfigError(f"config field '{AS_TEXT}' must be true or false, got {config[AS_TEXT]!r}")
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
        """Refuse a workbook that is not there, before the run starts."""
        require_file_to_read("read_excel", config["path"])

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        path = config["path"]
        sheet = config.get("sheet")
        has_header = config["has_header"]
        skip_rows = config["skip_rows"]

        fastexcel = _import_fastexcel()
        try:
            reader = fastexcel.read_excel(path)
        except Exception as error:
            raise ConfigError(f"read_excel could not open '{path}' as an XLSX workbook: {error}") from error

        sheet_names = reader.sheet_names
        if sheet is None:
            return Ports({"rows": _read_matching(reader, config)})
        if sheet not in sheet_names:
            raise ConfigError(
                f"read_excel found no sheet named {sheet!r} in '{path}'; the sheets present are: "
                f"{', '.join(sheet_names)}"
            )
        return Ports({"rows": _read_sheet(reader, path, sheet, has_header, skip_rows, config[AS_TEXT])})


def _import_fastexcel() -> Any:
    """The workbook reader, imported only when a sheet is read, or a refusal naming the package."""
    try:
        import fastexcel
    except ImportError as error:
        raise ConfigError(
            "read_excel needs fastexcel, which this environment does not have installed; "
            "install 'fastexcel' to read a workbook"
        ) from error
    return fastexcel


def _xlsxwriter() -> Any:
    """The workbook writer the worked examples build their files with, imported when one runs."""
    try:
        import xlsxwriter
    except ImportError as error:
        raise ConfigError(
            "read_excel's examples need xlsxwriter, which this environment does not have installed; "
            "install 'xlsxwriter' to run them"
        ) from error
    return xlsxwriter


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
            f"config field 'sheets_matching' pattern `{pattern}` matches no sheet in '{path}'; the "
            f"sheets present are: {', '.join(sheet_names)}"
        )

    frames: list[tuple[str, pl.DataFrame]] = []
    for name in matched:
        frame = _read_sheet(reader, path, name, config["has_header"], config["skip_rows"], config[AS_TEXT])
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


def _read_sheet(
    reader: Any,
    path: str,
    sheet: str,
    has_header: bool,
    skip_rows: int,
    as_text: bool = False,
) -> pl.DataFrame:
    """One sheet, read exactly as naming it in ``sheet`` reads it."""
    try:
        raw_header = reader.load_sheet_by_name(sheet, header_row=None, skip_rows=skip_rows, n_rows=1).to_polars()
        header_text = reader.load_sheet_by_name(
            sheet, header_row=None, skip_rows=skip_rows, n_rows=1, dtypes="string"
        ).to_polars()
    except Exception as error:
        raise ConfigError(f"read_excel could not read sheet {sheet!r} in '{path}': {error}") from error

    if raw_header.height == 0:
        raise ConfigError(f"read_excel found nothing on sheet {sheet!r} in '{path}'")

    if has_header:
        _validate_header_row(raw_header.row(0), sheet)
        _refuse_headings_the_reader_would_rename(raw_header.row(0), header_text.row(0), sheet)

    read_options: dict[str, Any] = {"header_row": skip_rows if has_header else None}
    if not has_header:
        read_options["skip_rows"] = skip_rows

    def read(options: Mapping[str, Any]) -> pl.DataFrame:
        return pl.read_excel(
            path,
            sheet_name=sheet,
            has_header=has_header,
            read_options=dict(options),
            drop_empty_rows=False,
            drop_empty_cols=False,
            raise_if_empty=False,
        )

    try:
        if as_text:
            return _read_sheet_as_text(path, sheet, has_header, read_options)
        frame = read(read_options)
        text = read({**read_options, "dtypes": "string"})
    except Exception as error:
        raise ConfigError(f"read_excel could not read sheet {sheet!r} in '{path}': {error}") from error

    _refuse_text_the_typed_read_lost(frame, text, sheet, has_header)
    if frame.width == 0:
        return frame
    # A row blank in every column is layout, dropped after the check above so a lost value is still seen.
    return frame.filter(~pl.all_horizontal(pl.all().is_null()))


def _refuse_text_the_typed_read_lost(frame: pl.DataFrame, text: pl.DataFrame, sheet: str, has_header: bool) -> None:
    """Refuse a cell that holds something but reads as null, naming the first one.

    The engine decides a column's type from its first 100 rows and forces every cell into it. Text
    further down a column of figures then reads as null. So do a few words it treats as missing
    (``n/a``, ``#N/A``, ``null``), wherever they sit. ``text`` is the same sheet read as text, row
    for row, so a cell present there and null in ``frame`` was lost. A cell of nothing but spaces
    loses nothing and is let through.
    """
    lost = [
        frame.get_column(name).is_null() & text.get_column(name).str.strip_chars().str.len_chars().gt(0)
        for name in frame.columns
    ]
    if not lost or not any(mask.any() for mask in lost):
        return
    where = pl.DataFrame({name: mask for name, mask in zip(frame.columns, lost, strict=True)})
    offending_rows = where.select(pl.any_horizontal(pl.all())).to_series()
    first_row = offending_rows.arg_true()[0]
    column = next(name for name in frame.columns if where[name][first_row])
    written = text[column][first_row]
    position = f"row {first_row + 1} under the heading" if has_header else f"row {first_row + 1} read"
    raise DataError(
        f"read_excel refuses sheet {sheet!r}: column {column!r} holds {written!r} on {position}, but "
        f"the column reads as {frame.schema[column]}, so that cell would arrive as null. Set "
        f"as_text: true to read every cell as the text it holds, then convert the columns with "
        f"cast_columns",
        offending=text.filter(offending_rows),
        total=text.height,
    )


def _read_sheet_as_text(path: str, sheet: str, has_header: bool, read_options: Mapping[str, Any]) -> pl.DataFrame:
    """Every cell of one sheet as the text of what it stores.

    The engine reads a whole column as one type and forces every cell into it. So the sheet is read
    four times: as text, as figures, as truth values and as dates. Each cell then takes its text
    from the reading that suits what it holds (see ``_cell_as_text``). Empty rows are kept in all
    four readings so their rows line up. A row blank in every column is dropped at the end, as the
    default read drops it.
    """

    def read_as(dtype: str) -> pl.DataFrame:
        return pl.read_excel(
            path,
            sheet_name=sheet,
            has_header=has_header,
            read_options={**read_options, "dtypes": dtype},
            drop_empty_rows=False,
            drop_empty_cols=False,
            raise_if_empty=False,
        )

    text, figure, truth, date = (read_as(dtype) for dtype in ("string", "float", "boolean", "datetime"))

    cell = _cell_as_text()
    columns = [
        pl.DataFrame(
            {
                "text": text[name].cast(pl.String),
                "figure": figure[name].cast(pl.Float64),
                "truth": truth[name].cast(pl.Boolean),
                "date": date[name].cast(pl.Datetime("ms")),
            }
        )
        .select(cell.alias(name))
        .to_series()
        for name in text.columns
    ]

    frame = pl.DataFrame(columns, schema={name: pl.String for name in text.columns})
    if frame.width == 0:
        return frame
    return frame.filter(~pl.all_horizontal(pl.all().is_null()))


def _cell_as_text() -> pl.Expr:
    """One cell's text, picked from its four readings in columns ``text``, ``figure``, ``truth``, ``date``.

    What each reading gives, by what the cell holds. Measured on the engine; the tests in
    ``TestEveryCellReadAsText`` fail if it changes.

        cell holds      text reading            figure    truth          date
        a figure        the figure, rounded     exact     true or false  a day counted from 1900
        a truth value   "true" or "false"       1 or 0    exact          none
        a date or time  "2026-01-31 14:05:00"   none      none           exact
        text            as typed                a figure if it looks like one; none otherwise
        blank           none                    none      none           none

    So the rule, in order:

    1. No text reading: the cell is blank. It reads as null.
    2. A truth reading, and the text reading is "true" or "false": a truth value. Keep the text.
    3. Any other truth reading: a figure. Spell it from the figure reading, since the text reading
       rounds it.
    4. No truth reading, but a date reading: a date or time. Spell it from the date reading.
    5. Anything else is text. Keep it as typed.

    The figure reading cannot decide on its own: a text cell holding "1" has one.
    """
    figure = pl.col("figure")
    # A whole figure is written without a point. Below 2**53 the conversion to a whole number is
    # exact. -0.0 counts as whole, so it reads "0".
    whole = (figure == figure.floor()) & (figure.abs() < _EXACT_WHOLE_LIMIT)
    figure_text = (
        pl.when(whole).then(figure.cast(pl.Int64, strict=False).cast(pl.String)).otherwise(figure.cast(pl.String))
    )

    date = pl.col("date")
    # A cell cannot say whether midnight was meant, so a time of exactly midnight reads as a date.
    date_text = (
        pl.when(date == date.dt.truncate("1d"))
        .then(date.dt.strftime("%Y-%m-%d"))
        .otherwise(date.dt.strftime("%Y-%m-%dT%H:%M:%S%.f"))
    )

    text = pl.col("text")
    has_truth = pl.col("truth").is_not_null()
    return (
        pl.when(text.is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(has_truth & text.is_in(["true", "false"]))
        .then(text)
        .when(has_truth)
        .then(figure_text)
        .when(date.is_not_null())
        .then(date_text)
        .otherwise(text)
    )


def _validate_header_row(raw_header: Sequence[Any], sheet: str) -> None:
    """Refuse a blank or repeated heading, rather than repair it.

    fastexcel would rename them `key_1` or `__UNNAMED__1`, names that then leak into the flow.
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


def _refuse_headings_the_reader_would_rename(raw_header: Sequence[Any], header_text: Sequence[Any], sheet: str) -> None:
    """Refuse a heading that cannot name its column as the sheet shows it.

    A heading must be text or a number. The engine cannot name a column after a date, a time or a
    truth value, and invents a name instead. And two headings that read the same, such as the
    number 2026 and the text "2026", would have the second one renamed. Both are refused here,
    before any name is invented.
    """
    seen: dict[str, int] = {}
    for index, (value, text) in enumerate(zip(raw_header, header_text, strict=True), start=1):
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            raise ConfigError(
                f"read_excel refuses sheet {sheet!r}: header cell {index} holds {text!r}, which is "
                f"not text or a number and cannot name a column. Type the heading as text in the "
                f"sheet, or set has_header: false and name the columns with rename_columns"
            )
        if text in seen:
            raise ConfigError(
                f"read_excel refuses sheet {sheet!r}: header cells {seen[text]} and {index} both "
                f"read {text!r}, and a duplicate cannot be told apart by guessing. Make the two "
                f"headings differ in the sheet, or set has_header: false"
            )
        seen[text] = index


@contextmanager
def _example_workbook() -> Iterator[None]:
    """Write the small workbook the example reads into the example's scratch folder."""
    target = scratch_path(_EXAMPLE_PATH)
    with _xlsxwriter().Workbook(str(target)) as workbook:
        worksheet = workbook.add_worksheet("Sample")
        worksheet.write_row(0, 0, ["key", "label"])
        worksheet.write_row(1, 0, [1, "first"])
        worksheet.write_row(2, 0, [2, "second"])
    yield


def _reads_a_sheet_example() -> WorkedExample:
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "label": ["first", "second"]})})
    return WorkedExample(
        scenario="read one sheet by name",
        configuration={"path": _EXAMPLE_PATH, "sheet": "Sample"},
        explanation="'sheet' names the tab to read. The columns and their types come through as the sheet holds "
        "them.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_example_workbook,
    )


@contextmanager
def _months_workbook() -> Iterator[None]:
    """A workbook of a cover sheet and two month sheets, for the stacking example."""
    target = scratch_path(_MONTHS_EXAMPLE_PATH)
    with _xlsxwriter().Workbook(str(target)) as workbook:
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
        scenario="read every month sheet of a report into one table",
        configuration={
            "path": _MONTHS_EXAMPLE_PATH,
            "sheets_matching": r"^[A-Z][a-z]{2} \d{4}$",
            "sheet_column": "sheet",
        },
        explanation="Both month sheets match the pattern and the cover sheet does not. They are stacked in "
        "workbook order, and 'sheet' says which sheet each row came from.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_months_workbook,
        check_row_order=True,
    )


@contextmanager
def _mixed_workbook() -> Iterator[None]:
    """A workbook whose one column mixes a figure, text and a date, for the as_text example."""
    target = scratch_path(_TEXT_EXAMPLE_PATH)
    with _xlsxwriter().Workbook(str(target)) as workbook:
        worksheet = workbook.add_worksheet("Sample")
        date_format = workbook.add_format({"num_format": "yyyy-mm-dd"})
        worksheet.write_row(0, 0, ["key", "amount"])
        worksheet.write_row(1, 0, ["a", 10.25])
        worksheet.write_row(2, 0, ["b", 12])
        worksheet.write_string(3, 0, "c")
        worksheet.write_string(3, 1, "n/a")
        worksheet.write_string(4, 0, "d")
        worksheet.write_datetime(4, 1, dt.date(2026, 1, 31), date_format)
    yield


def _reads_as_text_example() -> WorkedExample:
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {"key": ["a", "b", "c", "d"], "amount": ["10.25", "12", "n/a", "2026-01-31"]},
            )
        }
    )
    return WorkedExample(
        scenario="read a sheet whose column mixes figures, text and a date",
        configuration={"path": _TEXT_EXAMPLE_PATH, "sheet": "Sample", AS_TEXT: True},
        explanation="Every column comes in as text, each cell as the text of what it stores: the whole figure "
        "without a point, the date as YYYY-MM-DD. Use cast_columns afterwards to say what each column "
        "is.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_mixed_workbook,
    )


REGISTRY.register(ReadExcel())
