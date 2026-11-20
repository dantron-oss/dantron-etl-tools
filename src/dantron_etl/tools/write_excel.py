"""WriteExcel: write each named frame to its own tab of one XLSX workbook.

The workbook starts from a template, if one is named, or from nothing. It is opened once and saved
once, however many tabs there are. The parts live in sibling modules: `_workbook_checks` (refusals
before the workbook opens), `_workbook_rules` (which columns a rule applies to), `_workbook_file`
(opening it), `_workbook_cells` (values and number formats), `_workbook_layout` (headings, widths,
wrapping, tables) and `_workbook_examples`. What a cell can hold is in `_spreadsheet.py`.

Why: docs/design/tools.md#write_excel
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
)
from ..column_types import DECLARED_TYPES
from ._spreadsheet import (
    MAX_CELL_TEXT,
    MAX_DATA_ROWS,
    MAX_EXACT_INTEGER,
    MAX_SHEET_ROWS,
    require_xlsx_path,
    unsupported_dtype_reason,
)
from ._files import refuse_to_overwrite, require_output_folder
from ._workbook_cells import write_tab
from ._workbook_checks import (
    names_the_same_file,
    validate_format_rules,
    validate_formatted_columns,
    validate_layout,
    validate_layout_columns,
    validate_tab_name,
    validate_table_names,
)
from ._workbook_examples import laid_out_example, plain_write_example, template_example
from ._workbook_file import empty_tab, opened_once, refuse_names_in_use
from ._workbook_layout import lay_out_tab
from ._workbook_rules import format_rules


class WriteExcel:
    """Write each named frame to its own tab of one XLSX workbook. Produces nothing."""

    name = "write_excel"
    description = "Write tables to the tabs of one Excel workbook, optionally into a template."
    category = "output"
    # Empty: the ports come from 'sheets', through input_ports_for.
    input_ports = ()
    output_ports = ()

    def input_ports_for(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        """One input port per sheet entry, in the order written.

        This sees the configuration before it is checked, so it skips anything malformed. The config check
        that follows refuses it by name.
        """
        entries = config.get("sheets")
        if not isinstance(entries, list):
            return ()
        return tuple(
            entry["port"]
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("port"), str) and entry["port"]
        )

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="The workbook file to write. It must end in .xlsx.",
                required=True,
                non_empty=True,
                file_path=True,
            ),
            "sheets": ConfigField(
                type="array",
                description="One entry per tab: the input port to read, the tab to write it to, "
                "and how the tab looks.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One tab: the port its rows come from, the tab name, and its layout.",
                    fields={
                        "port": ConfigField(
                            type="string",
                            description="The input port whose rows go on this tab, such as 'summary'.",
                            required=True,
                            non_empty=True,
                        ),
                        "sheet": ConfigField(
                            type="string",
                            description="The tab name, such as 'Summary'. At most 31 characters, "
                            "and none of '[' ']' ':' '*' '?' '/' '\\'.",
                            required=True,
                            non_empty=True,
                        ),
                        "allow_empty": ConfigField(
                            type="boolean",
                            description="Set true to write a port with no rows as a tab of headings "
                            "only. Left out, an empty port stops the run.",
                        ),
                        "formats": ConfigField(
                            type="array",
                            description="Number formats for this tab, one rule per entry, each for "
                            "a type or one column. Left out, only dates are formatted.",
                            required=False,
                            items=ConfigField(
                                type="object",
                                description="One rule: a type or a column, and its format code.",
                                fields={
                                    "type": ConfigField(
                                        type="string",
                                        description="Apply the format to every column of this type, "
                                        "such as Float64.",
                                        enum=DECLARED_TYPES,
                                    ),
                                    "column": ConfigField(
                                        type="string",
                                        description="Apply the format to this one column. It beats a "
                                        "rule about the column's type.",
                                        non_empty=True,
                                    ),
                                    "format": ConfigField(
                                        type="string",
                                        description="The spreadsheet's format code, such as '#,##0.00' "
                                        "for money or '0.0%' for a percentage.",
                                        required=True,
                                        non_empty=True,
                                    ),
                                },
                            ),
                        ),
                        "widths": ConfigField(
                            type="array",
                            description="Column widths, one rule per entry, each for a type or one "
                            "column. Left out, no width is set.",
                            required=False,
                            items=ConfigField(
                                type="object",
                                description="One rule: what it applies to, and the width it sets.",
                                fields={
                                    "type": ConfigField(
                                        type="string",
                                        description="Set this width on every column of this type.",
                                        enum=DECLARED_TYPES,
                                    ),
                                    "column": ConfigField(
                                        type="string",
                                        description="Set this width on this one column; it beats "
                                        "any rule about the column's type.",
                                        non_empty=True,
                                    ),
                                    "width": ConfigField(
                                        type="number",
                                        description="The width, in characters: more than 0, at most 255.",
                                        required=True,
                                    ),
                                },
                            ),
                        ),
                        "fit_widths": ConfigField(
                            type="boolean",
                            description="Set true to size each column to fit its heading and values, "
                            "up to 60 characters wide.",
                            required=False,
                        ),
                        "wrap": ConfigField(
                            type="array",
                            description="Columns whose text wraps onto several lines, one rule per "
                            "entry, each for a type or one column.",
                            required=False,
                            items=ConfigField(
                                type="object",
                                description="One rule: the type or the column to wrap.",
                                fields={
                                    "type": ConfigField(
                                        type="string",
                                        description="Wrap every column of this type.",
                                        enum=DECLARED_TYPES,
                                    ),
                                    "column": ConfigField(
                                        type="string",
                                        description="Wrap this one column.",
                                        non_empty=True,
                                    ),
                                },
                            ),
                        ),
                        "table": ConfigField(
                            type="object",
                            description="Turn the rows into a spreadsheet table with filter "
                            "dropdowns. Write {} to name it after the tab.",
                            required=False,
                            fields={
                                "name": ConfigField(
                                    type="string",
                                    description="The table's name, such as 'Sales'. Left out, it is made from the tab name.",
                                    non_empty=True,
                                ),
                                "style": ConfigField(
                                    type="string",
                                    description="A built-in table style, such as 'TableStyleLight1'. "
                                    "Left out, the table is plain.",
                                    non_empty=True,
                                ),
                            },
                        ),
                        "header": ConfigField(
                            type="object",
                            description="How the headings look: bold, a font size, or both.",
                            required=False,
                            fields={
                                "bold": ConfigField(
                                    type="boolean",
                                    description="Whether the headings are bold.",
                                ),
                                "size": ConfigField(
                                    type="number",
                                    description="The headings' font size in points: at least 1, at most 409.",
                                ),
                            },
                        ),
                    },
                ),
            ),
            "template": ConfigField(
                type="string",
                description="A workbook to start from. It is only read. Left out, the workbook starts empty.",
                required=False,
                file_path=True,
            ),
            "overwrite": ConfigField(
                type="boolean",
                description="Set true to allow replacing a file already at 'path', or a template "
                "tab that a sheet entry names.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use write_excel to deliver results as a spreadsheet. Each entry in 'sheets' names an "
                "input port and the tab its rows go to, so one step can write many tabs. Name a "
                "'template' to write the tabs into a workbook you have already laid out, with its "
                "charts, notes and formatting; the other tabs come through as they were. The "
                "workbook is saved to 'path'. This step has no output ports: a flow ends here."
            ),
            examples=(plain_write_example(), template_example(), laid_out_example()),
            notes=(
                "The template is only read. A 'path' naming the same file as the template is refused.",
                "The file already at 'path' is never read. Each run builds the workbook again from "
                "the template, or from nothing, so a tab the flow stops writing does not linger.",
                "A file already at 'path', or a template tab that a sheet entry names, is refused "
                "unless 'overwrite' is true.",
                "A template tab that is replaced keeps its place among the other tabs.",
                "The whole template is read into memory and written out again, not copied byte for "
                "byte. Pivot tables, charts, named ranges and the values behind a pivot's filter "
                "dropdowns come through.",
                "One thing is known to be lost: a pivot cache's extension block, where a slicer link "
                "and newer formatting options live. A slicer can come back disconnected.",
                "Write one workbook and open it before using a template you could not rebuild.",
                "Formulas on a template tab come through, but their stored results are cleared. A "
                "spreadsheet recalculates when it opens the file, so a person sees the right figures.",
                "A program that reads the file before a spreadsheet has opened it, read_excel "
                "included, sees blanks where the formulas are.",
                "A port with no rows stops the run, because an empty tab caused by a failure upstream "
                "looks like a quiet month. Set allow_empty: true on its sheet entry to allow it.",
                "An allowed empty port gets its tab with the headings and nothing under them, never "
                "a missing tab and never a phrase such as '(no rows)'.",
                "A whole number beyond 2**53 - 1, either way, is refused, because a cell cannot hold "
                "it exactly. Cast it to text upstream if that is what you want.",
                "A Time, Duration, Decimal, list or nested record column is refused, not turned "
                "into text.",
                "Text longer than 32,767 characters, the most a cell holds, is refused, naming the "
                "column. It is never cut short.",
                "Not-a-number or infinity in a decimal column is refused, naming the tab, the column "
                "and the rows, because a cell would show it as a blank. A null is written as an "
                "empty cell.",
                "A frame with more than 1,048,575 rows, more than a tab holds, is refused, never cut short.",
                "Every frame is checked before the workbook is opened, so a refusal leaves no "
                "half-written file and never touches the template.",
                "A date column always gets a date format, so the spreadsheet shows a date, not a "
                "count of days. Nothing else is formatted unless 'formats' says so.",
                "A rule in 'formats' names either a type or one column, never both and never neither.",
                "A type rule applies to every column of that type, including columns whose names "
                "come from the data, such as one per month after a cross_tab.",
                "A column rule beats a type rule for that column, whichever is written first.",
                "A rule about Int64 also applies to the other whole-number types, such as Int32.",
                "Nothing is decided from a column's name.",
                "A format code goes to the spreadsheet exactly as written. It is not checked here.",
                "A format rule about Time is accepted but does nothing, because a Time column is "
                "refused anyway.",
                "Formats, widths and wrapping apply to the data cells only, not to the whole column, "
                "so a figure typed in below the data is not formatted.",
                "The layout settings 'widths', 'fit_widths', 'wrap', 'table' and 'header' change how "
                "a tab looks, never what it holds. A tab that uses none of them is written plain.",
                "A width or wrap rule names a type or one column, and a column rule beats a type rule. "
                "A rule naming a column the frame does not have is refused.",
                "'fit_widths' sizes each column to its heading and its longest value as shown, so "
                "money is measured as '1,234,567.00'. A fitted column is at most 60 characters wide.",
                "A width in 'widths' beats the fit, and is how a column gets wider than 60.",
                "'wrap' wraps the data cells of the columns it names and aligns them to the top of "
                "the row, empty cells included. The heading is not wrapped.",
                "'table' turns the tab's rows into a spreadsheet table with filter dropdowns. Its name "
                "is the one given, or the tab name with unusable characters turned into underscores.",
                "A table has no colour style unless a built-in style is named.",
                "A table on a port with no rows covers the headings and one empty row, the smallest "
                "table a spreadsheet accepts.",
                "Two tables with the same name, ignoring case, are refused. So is a table name the "
                "template already uses on a tab this step does not replace. A replaced tab loses its "
                "old table.",
                "'header' makes the headings bold, a given size, or both. Anything it does not set "
                "keeps the cell's own font.",
            ),
            field_help={
                "path": "For example 'output/monthly_report.xlsx'. The folder must already exist.",
                "sheets": "For example {port: summary, sheet: Summary}. Each port named here becomes "
                "an input of this step.",
                "overwrite": "Set it to true for a flow that runs again and again: without it, the "
                "second run is refused because the file is already there.",
                "template": "For example 'templates/report.xlsx'. It is never written to, so the "
                "same template works again next run.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse what the configuration and the disk can show is wrong, before any workbook opens.

        That is: a path that is not .xlsx, a bad tab name, a tab or port named twice, a bad format or layout
        rule, a missing template or one equal to `path`, and a file already at `path` without `overwrite`.
        A port named twice is also refused by the loader; it is checked here too for a tool run on its own.
        """
        path = config["path"]
        require_xlsx_path("write_excel", path)

        tabs: dict[str, str] = {}
        ports: set[str] = set()
        for port, sheet in _entries(config):
            validate_tab_name(sheet)
            if sheet in tabs:
                raise ConfigError(
                    f"write_excel was told to write the tab {sheet!r} twice, from ports "
                    f"{tabs[sheet]!r} and {port!r}; one of the two frames would be delivered and "
                    f"the other lost. Give each port its own tab, or join the two frames upstream."
                )
            if port in ports:
                raise ConfigError(
                    f"write_excel was told to read the port {port!r} twice; a port carries one "
                    f"frame, so the same rows would be written to both tabs. Name a different port "
                    f"for one of the two tabs."
                )
            tabs[sheet] = port
            ports.add(port)

        for entry in config["sheets"]:
            validate_format_rules(entry)
            validate_layout(entry)
        validate_table_names(config["sheets"])

        require_output_folder("write_excel", path)
        template = config["template"]
        if template is None:
            refuse_to_overwrite("write_excel", path, "overwrite", config["overwrite"])
            return
        if names_the_same_file(template, path):
            raise ConfigError(
                f"write_excel refuses to write to its own template; 'template' is '{template}' "
                f"and 'path' is '{path}', and the two name the same file on disk. The template is "
                f"read every run, and overwriting it would leave the next run with nothing to start "
                f"from. Change 'path' to name the pack to be delivered, somewhere other than the "
                f"template."
            )
        if not Path(template).exists():
            raise ConfigError(
                f"write_excel cannot find the template '{template}'; a template that is not "
                f"there is not a template. Check the path, or leave 'template' out to start from "
                f"nothing."
            )
        refuse_to_overwrite("write_excel", path, "overwrite", config["overwrite"])

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse any frame a tab cannot hold, before the workbook opens.

        Every frame is checked first, so a refusal about the last tab leaves no file and an untouched
        template. A rule naming a missing column is refused here, the first point the columns are known.
        """
        for entry in config["sheets"]:
            validate_formatted_columns(entry, inputs[entry["port"]])
            validate_layout_columns(entry, inputs[entry["port"]])

        for entry in config["sheets"]:
            frame = inputs[entry["port"]]
            if frame.height == 0 and not entry.get("allow_empty", False):
                raise DataError(
                    f"write_excel refuses port {entry['port']!r}, bound for tab {entry['sheet']!r}: it has "
                    f"no rows, and an empty tab looks like a quiet month even when something upstream "
                    f"failed; set allow_empty: true on that sheet entry if the tab may be empty",
                    offending=frame,
                    total=0,
                )

        for port, sheet in _entries(config):
            frame = inputs[port]

            if frame.height > MAX_DATA_ROWS:
                raise DataError(
                    f"write_excel refuses the {frame.height:,} row(s) on port {port!r}, bound "
                    f"for tab {sheet!r}; a tab's header takes one row, leaving {MAX_DATA_ROWS:,} "
                    f"row(s) for data (a worksheet holds at most {MAX_SHEET_ROWS:,} rows in total)",
                    offending=frame.head(0),
                    total=frame.height,
                )

            for name, dtype in frame.schema.items():
                reason = unsupported_dtype_reason(dtype)
                if reason is not None:
                    raise ConfigError(
                        f"write_excel cannot write column {name!r} of dtype {dtype} on port "
                        f"{port!r}: {reason}; format it into something a cell can hold, one step "
                        f"upstream"
                    )

            for name, dtype in frame.schema.items():
                if dtype != pl.String:
                    continue
                too_long = frame.filter(pl.col(name).str.len_chars() > MAX_CELL_TEXT)
                if too_long.height:
                    raise DataError(
                        f"write_excel refuses column {name!r} on port {port!r}, bound for tab "
                        f"{sheet!r}: a cell holds at most {MAX_CELL_TEXT:,} characters, and a longer "
                        f"text would be cut short without a word. Shorten or split the text upstream "
                        f"(each text below shows its first 40 characters and its length)",
                        offending=too_long.with_columns(
                            pl.col(name).str.slice(0, 40),
                            pl.col(name).str.len_chars().alias(f"{name} length"),
                        ),
                        total=frame.height,
                    )

            for name, dtype in frame.schema.items():
                if not dtype.is_float():
                    continue
                not_a_figure = frame.filter(pl.col(name).is_not_null() & ~pl.col(name).is_finite())
                if not_a_figure.height:
                    raise DataError(
                        f"write_excel refuses column {name!r} on port {port!r}, bound for tab "
                        f"{sheet!r}: it holds not-a-number or infinity, which a cell would show as a "
                        f"blank, as if no figure were there. A figure went wrong upstream; fix it "
                        f"there, or turn those rows into null with case_when if a blank is what the "
                        f"tab should show",
                        offending=not_a_figure,
                        total=frame.height,
                    )

            for name, dtype in frame.schema.items():
                if not dtype.is_integer():
                    continue
                offending = frame.filter(pl.col(name).abs() > MAX_EXACT_INTEGER)
                if offending.height:
                    raise DataError(
                        f"write_excel refuses column {name!r} on port {port!r}: a spreadsheet "
                        f"cell holds a double, and cannot keep a whole number larger than "
                        f"{MAX_EXACT_INTEGER} in magnitude exactly",
                        offending=offending,
                        total=frame.height,
                    )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Open the workbook once, write every tab, and save it once.

        One refusal happens here: a template tab named by a sheet entry without `overwrite`. Only the open
        workbook can answer it, and it comes before the save, so a refused run leaves nothing behind.
        """
        path = config["path"]
        with opened_once(path, config["template"]) as book:
            refuse_names_in_use(book, config["sheets"])
            for entry in config["sheets"]:
                worksheet = empty_tab(book, entry["sheet"], replace=config["overwrite"])
                frame = inputs[entry["port"]]
                # Values and number formats first; then layout, which must never change either.
                write_tab(worksheet, frame, format_rules(entry))
                lay_out_tab(worksheet, frame, entry)
            book.save(path)

        return Ports()


def _entries(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The `(port, tab)` pairs of a checked configuration, in the order written."""
    return tuple((entry["port"], entry["sheet"]) for entry in config["sheets"])


REGISTRY.register(WriteExcel())
