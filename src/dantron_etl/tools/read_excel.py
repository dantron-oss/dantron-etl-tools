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
written as floats comes back whole. The engine decides each column's type from its first 100 rows,
so text inside those rows makes the column text. Text further down a column of figures would read
as null, and so would a word the engine treats as missing (``n/a``, ``#N/A``); the read stops
instead and points to ``as_text``.

A header row is a sheet's schema. A repeated heading, a blank one, or one that is a date or a truth
value is refused rather than repaired, and two headings that read the same (the number 2026 and the
text "2026") count as repeated. Repaired, the engine's own names -- ``key_1``, ``__UNNAMED__1`` --
would leak into a flow's column names the moment a downstream step names one of them.
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
                file_path=True,
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
            AS_TEXT: ConfigField(
                type="boolean",
                description="Read every cell as text, so a column holding figures on one sheet and text "
                "on another can be stacked and converted on purpose by cast_columns afterwards.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "The named sheet arrives on `rows` with the columns and the dtypes the sheet "
                "itself holds: a column of round figures comes back whole, a column with text in "
                "its first 100 rows comes back as text, and a blank cell comes back null rather "
                "than an empty string. The sheet is always named, in `sheet`, or found by a pattern, "
                'in `sheets_matching`. It is never an index: "the first tab" silently reads the '
                "wrong sheet the moment a delivered pack puts a cover sheet in front of the figures. "
                "A row blank in every column is dropped as layout rather than kept as a row of nulls."
            ),
            examples=(_reads_a_sheet_example(), _reads_matching_sheets_example(), _reads_as_text_example()),
            notes=(
                "Each column's type is decided from its first 100 rows. A cell that holds something "
                "but would read as null is refused, naming the column and the row: text further "
                "down a column of figures, or a word the engine treats as missing, such as n/a, "
                "#N/A or null, anywhere in it. A cell of only spaces reads as null. Set as_text to "
                "read every cell as its text and convert the columns with cast_columns after.",
                "A sheet named in the configuration that is not in the workbook is refused with "
                "the sheet names that are there; a name differing only in case or a trailing space "
                "is a different name, not a near match.",
                "has_header defaults to true, since a spreadsheet's first row is a header until an "
                "author says otherwise; a duplicated or a blank heading is refused rather than "
                "repaired, because a sheet's header row is its schema. Two headings that read the "
                "same (the number 2026 and the text 2026) are a duplicate, and a heading that is a "
                "date or a truth value is refused, because it cannot name a column.",
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
                "as_text makes every column text. Each cell reads as what it stores, not as Excel "
                "displays it. A whole figure has no point: 12, never 12.0. Any other figure reads as "
                "the shortest text that converts back to exactly the stored number. A date reads "
                "YYYY-MM-DD, and a date with a time of day YYYY-MM-DDTHH:MM:SS. A truth value reads "
                "true or false. A blank cell is null. In a column that mixes kinds, each cell keeps "
                "its own text.",
                "as_text has two limits, both because Excel stores a date or a time as one number. "
                "A time of exactly midnight reads as a date. A cell holding only a time of day "
                "(formatted hh:mm, say) reads as that time on 31 December 1899: 14:05 reads "
                "1899-12-31T14:05:00.",
                "as_text reads the sheet four times instead of once: as text, as figures, as truth "
                "values and as dates. Each cell takes its text from the reading that suits what it "
                "holds. So a large sheet takes a few times longer to read. The text reading alone "
                "would be quicker, but it rounds figures: 99999999999.99 comes back "
                "99999999999.990005493.",
            ),
            field_help={
                "sheet": "By name. Refused, with the sheet names present, if it does not match one of "
                "them exactly. Give this or sheets_matching.",
                "skip_rows": "Rows above the header (or the data) to skip; must not be negative.",
                "sheets_matching": "Found anywhere in the sheet name, like regex_match(). Refused if no sheet matches.",
                "sheet_column": "Comes after every column read, and holds the sheet's name as text.",
                AS_TEXT: "Every column arrives as text; cast_columns afterwards states what each one is. "
                "Off by default.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong format or a negative skip count before a workbook is ever opened."""
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
        """Refuse a workbook that is not there, before any step of the flow does work."""
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
    # A row blank in every column is layout, dropped here as the engine's own option would drop it.
    # It is dropped after the check above, so a row whose only value was lost is still seen there.
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
    """A small workbook the example reads, entered around the example and torn down after.

    Built here rather than inside ``help()``, which must describe a tool without changing the
    process doing the describing -- the catalog page generator calls ``help()`` on every tool, and
    writing a file as a side effect of describing one is exactly the disturbance the boundary now
    refuses. Planted with :func:`scratch_path`, into the directory :func:`check_tool` opens for the
    run; nothing here tears it back down, because that directory is the checker's to remove.
    """
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
        scenario="read a sheet whose column mixes figures, text and a date, every cell as text",
        configuration={"path": _TEXT_EXAMPLE_PATH, "sheet": "Sample", AS_TEXT: True},
        explanation="Every column arrives as text, each cell as the text of what it stores: the whole "
        "figure without a point, the date as YYYY-MM-DD. cast_columns afterwards says what each column is.",
        inputs=Ports({}),
        outputs=outputs,
        setup=_mixed_workbook,
    )


REGISTRY.register(ReadExcel())
