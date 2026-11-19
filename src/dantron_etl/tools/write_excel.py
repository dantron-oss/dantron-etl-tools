"""WriteExcel: one frame per named port, one tab each, in one load and one save.

This tool writes one XLSX workbook: the tabs it is given, on top of a template if one is named.
A delivered pack is usually curated. Somebody built the charts, the formats and the notes by hand.
That curated workbook is the template, and only the tabs holding figures that change every run are
written into it.

Rewritten is not the same as preserved. The template is read into an object model and a new file
is written from it, so every part comes back as that model spells it rather than as the bytes it
arrived as. Most of a pack survives that trip: tabs, formats, formulas, charts, pivot tables and the
cached values behind a pivot's filter dropdowns were all measured coming through. Two things do
not. The result stored with each formula is cleared, so the file recalculates when a spreadsheet
opens it, and a program reading it before then sees blanks. And the extension block on a pivot
cache is lost. That block holds the parts of a pivot added after the original file format, a slicer
link among them, and a test in this package records the loss. Write one pack and open it before
pointing this at a workbook nobody could rebuild.

It reads one input port per tab its configuration names. Eleven tabs is one step, not eleven. That
is not only faster. The workbook is opened once and saved once.

It produces nothing at all, so a flow ends here. Handing eleven frames back out would be a fan-out
this tool invented, and handing back a receipt is shape without information.

The template is read and never written. A ``path`` equal to the template is refused. A template
overwritten by its own first run is gone, and the run after it has nothing to start from.

The file already at ``path`` is never read. Each run builds the workbook from the template, or from
nothing, so a run writes the same file whatever an earlier run left there, and a tab the flow stops
writing does not linger. ``overwrite_sheets`` is the one permission for replacing something that is
already there: the file at ``path`` (refused when the flow loads), and a tab of the template that a
sheet entry names (refused when the template is opened).

How a column's figures are *shown* is stated per tab, in ``formats``, and is stated or it does not
happen. A rule names either a type -- reaching every column of it, including the ones named by the
data -- or one column, which beats the rule about its type. Nothing is read off a column's name: a
writer deciding for itself that a decimal column was money is wrong the first time somebody puts a
reference number in one, and a writer deciding from the heading is wrong the first time somebody
renames it. The format code goes to the spreadsheet as written, unparsed and unchecked.

What a cell can hold is not this tool's ruling to make. Those rules belong to the destination, and
they live in ``_spreadsheet.py``. A whole number past 2**53 - 1
is refused rather than rounded. A frame taller than a sheet is refused rather than truncated. A date
column gets a number format, because that format is the only thing telling a cell it holds a date
rather than a count of days.

The work is split across sibling modules, one job each: ``_workbook_checks`` (refusals answerable
before the workbook opens), ``_workbook_rules`` (which columns a type-or-column rule reaches),
``_workbook_file`` (opening the workbook once), ``_workbook_cells`` (values and number formats),
``_workbook_layout`` (headings, widths, wrapping, tables) and ``_workbook_examples``.

The refusals are split across three places, and the split is deliberate. Anything answerable from
the configuration and the disk is refused in :meth:`WriteExcel.validate_config`. Anything answerable
from the frames is refused in :meth:`WriteExcel.validate_inputs`. One rule is left to
:meth:`WriteExcel.execute`: whether a tab is already in the template. That is the only question
here that cannot be answered without the template open, and asking it any earlier would cost a
second load of a file that can be megabytes.
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
    """Write each named frame to its own tab of one XLSX workbook, and produce nothing."""

    name = "write_excel"
    description = "Write one frame per named port to its own tab of one XLSX workbook."
    category = "output"
    # Empty on purpose: the ports come from 'sheets', through input_ports_for below. A tool answers
    # that question one way or the other, never both.
    input_ports = ()
    output_ports = ()

    def input_ports_for(self, config: Mapping[str, Any]) -> tuple[str, ...]:
        """One port per sheet entry, in the order the entries were written.

        The configuration handed over here is the one a person wrote. No blank in it has been filled
        in yet, and it has not been checked against the config template. So anything malformed is
        passed over rather than refused here. The config check runs a moment later and says what is
        wrong, in words about the field that is wrong.
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
                description="Where the workbook is written; must end in .xlsx.",
                required=True,
                non_empty=True,
                file_path=True,
            ),
            "sheets": ConfigField(
                type="array",
                description="One entry per tab: which input port it reads, which tab it is "
                "written to, and how its figures are shown. These entries are also what say which "
                "ports this step reads.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One tab: the port the rows arrive on, and the tab they are written to.",
                    fields={
                        "port": ConfigField(
                            type="string",
                            description="The input port carrying the rows for this tab.",
                            required=True,
                            non_empty=True,
                        ),
                        "sheet": ConfigField(
                            type="string",
                            description="The tab the rows are written to; at most 31 characters, "
                            "and none of '[' ']' ':' '*' '?' '/' '\\'.",
                            required=True,
                            non_empty=True,
                        ),
                        "formats": ConfigField(
                            type="array",
                            description="How this tab's figures are shown: one rule per entry, "
                            "each naming either a type or a single column, and the format code "
                            "that column is shown under. Leave it out and nothing is formatted "
                            "beyond the date format this tool always writes.",
                            required=False,
                            items=ConfigField(
                                type="object",
                                description="One rule: what it applies to, and the format code it applies.",
                                fields={
                                    "type": ConfigField(
                                        type="string",
                                        description="Show every column of this type under the "
                                        "format; this is what reaches a column the configuration "
                                        "could not have named.",
                                        enum=DECLARED_TYPES,
                                    ),
                                    "column": ConfigField(
                                        type="string",
                                        description="Show this one column under the format; it "
                                        "beats any rule about the column's type.",
                                        non_empty=True,
                                    ),
                                    "format": ConfigField(
                                        type="string",
                                        description="The spreadsheet's own format code, passed "
                                        "through as written: '#,##0.00' for money, '0.000' for a "
                                        "proportion read to three places.",
                                        required=True,
                                        non_empty=True,
                                    ),
                                },
                            ),
                        ),
                        "widths": ConfigField(
                            type="array",
                            description="How wide this tab's columns are: one rule per entry, each "
                            "naming either a type or a single column, and the width. A column rule "
                            "beats a type rule. Leave it out and no width is set.",
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
                            description="Size every column to its heading and its longest value as "
                            "shown, up to a cap of 60 characters. A width stated in 'widths' beats "
                            "the fit. Leave it out and nothing is fitted.",
                            required=False,
                        ),
                        "wrap": ConfigField(
                            type="array",
                            description="Which columns wrap their text: one rule per entry, naming "
                            "either a type or a single column. Wrapped cells sit at the top of the "
                            "row. The heading is never wrapped.",
                            required=False,
                            items=ConfigField(
                                type="object",
                                description="One rule: the type or the column whose data cells wrap.",
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
                            description="Make the tab's rows a spreadsheet table, with filter "
                            "dropdowns on the headings. Write {} to take the name from the tab.",
                            required=False,
                            fields={
                                "name": ConfigField(
                                    type="string",
                                    description="The table's name. Leave it out and it is made from the tab name.",
                                    non_empty=True,
                                ),
                                "style": ConfigField(
                                    type="string",
                                    description="One of the spreadsheet's built-in table styles, such "
                                    "as 'TableStyleLight1'. Leave it out for a plain table.",
                                    non_empty=True,
                                ),
                            },
                        ),
                        "header": ConfigField(
                            type="object",
                            description="How the headings look: bold, a size, or both. What is not "
                            "stated keeps the cell's own font.",
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
                description="A workbook to start from. It is read and never written: the tool "
                "copies it to 'path' and writes there. Leave it out to start from nothing.",
                required=False,
                file_path=True,
            ),
            "overwrite_sheets": ConfigField(
                type="boolean",
                description="Whether something already there may be replaced: a file at 'path', or "
                "a tab of the template that a sheet entry names. Leave it out and either is refused.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Each entry in 'sheets' names an input port and the tab its rows are written to. "
                "Those entries are also what say which ports this step reads: two steps of this "
                "tool with different 'sheets' read different frames. Every tab goes into one "
                "workbook, which is opened once and saved once. Give a 'template' and that workbook "
                "is loaded, the named tabs are written into it, and the whole thing is saved to "
                "'path'. A tab nobody named comes back with its values, its formulas, its "
                "formatting and its column widths. The rest of the workbook is rewritten, not "
                "copied across byte for byte: the whole file is read into memory and a new file is "
                "written out. Pivot tables, charts and named ranges come through, and so do the "
                "cached values behind a pivot's filter dropdowns. One thing is known to be lost. A "
                "pivot cache's extension block is dropped, which is where a slicer link and the "
                "newer formatting options are kept, so a slicer can come back disconnected. Write "
                "one pack and open it before pointing this at a workbook you could not rebuild. "
                "The template itself is never written to, and a 'path' naming the same file is "
                "refused. The file already at 'path' is never read: each run builds the workbook "
                "from the template, or from nothing, so a tab the flow stops writing does not "
                "linger. A file already at 'path', or a template tab a sheet entry names, is "
                "refused until 'overwrite_sheets' says in writing that replacing it is intended. A "
                "replaced template tab keeps its place among the others. Nothing leaves on an "
                "output port: a flow ends here."
            ),
            examples=(plain_write_example(), template_example(), laid_out_example()),
            notes=(
                "Every formula on a template tab comes through, but its stored result is cleared. A "
                "spreadsheet recalculates the file when it opens it, so a person sees the right "
                "figures. A program that reads the file before then, read_excel among them, sees "
                "blanks where the formulas are.",
                "A port with no rows still gets its tab, with the header and nothing under it: a "
                "missing tab reads as a broken report, where an empty one reads as 'nothing this "
                "month'.",
                "A whole number past 2**53 - 1 in either direction is refused rather than rounded "
                "or silently written as text. A cell holds a double, so exactness past that point "
                "is a promise this tool cannot keep, and the fix is one explicit cast upstream "
                "where the next reader can see it.",
                "A Time, a Duration, a Decimal, a list or a nested record is refused rather than "
                "stringified: each has a plausible-looking wrong answer in a cell, and none of them "
                "is the value it started as.",
                "Text longer than 32,767 characters, the most a cell holds, is refused naming the "
                "column, rather than cut short.",
                "A frame taller than a tab -- more than 1,048,575 rows of data -- is refused rather "
                "than truncated, because a truncated total is the worst outcome a delivered figure "
                "can have.",
                "Every frame is checked before the workbook is opened, so a refusal leaves no "
                "half-written pack behind and never touches the template.",
                "A date column gets a date number format, because that format is what tells a "
                "person's spreadsheet the cell holds a date rather than a count of days. Nothing "
                "else is formatted until a tab's 'formats' says so: an unasked-for format is a "
                "claim about how the pack should be read that this tool is not entitled to make.",
                "A rule in 'formats' names either a type or one column, never both and never "
                "neither. A type rule reaches every column of that type, which is the only thing "
                "that reaches a column named by the data -- a cross-tab gains one per period. A "
                "column rule names one column and beats the rule about its type, whichever order "
                "the two were written in. A rule about Int64 reaches a whole number of any width, "
                "Int32 included, since a formula makes a whole-number literal 32-bit. Nothing is "
                "read off a column's name.",
                "A format code is passed to the spreadsheet as written: this library does not parse "
                "it and does not check it. It is in no position to know what a future spreadsheet "
                "accepts, a code wrongly rejected would block a pack for no reason, and a code that "
                "is wrong shows visibly in the file rather than corrupting a figure.",
                "A rule about a type this tool cannot write -- Time is the one -- is accepted and "
                "does nothing, because no column of that type ever reaches a cell: such a column is "
                "refused outright, well before its format would matter. Refusing the rule as well "
                "would tie this field to the list of types a cell cannot hold, and the two would "
                "have to be kept in step for no gain.",
                "A rule about a type or column formats the data cells, and only those. A format set "
                "on the whole column instead would run to the bottom of the tab, so a figure typed "
                "in underneath the data would be shown as money too. Harmless in a delivered pack, "
                "and visible to anybody comparing this pack's cell styles against one written that "
                "way.",
                "A frame with no rows is written as its header with nothing under it, so the tab "
                "carries no formatted cell at all. A writer that instead put a phrase like "
                "'(no rows)' in the first cell would be writing a word where a reader's formula "
                "expects a heading.",
                "The layout options -- 'widths', 'fit_widths', 'wrap', 'table' and 'header' -- change "
                "how a tab looks and never what it holds. No value, cell kind or number format is "
                "different with them than without them. A tab that states none of them is written "
                "as before.",
                "A width rule and a wrap rule name a type or one column, like a format rule, and a "
                "column rule beats a type rule. A rule naming a column the frame does not carry is "
                "refused.",
                "'fit_widths' sizes each column to its heading and its longest value as it is shown, "
                "so a money column is measured as '1,234,567.00', not as the stored digits. A fitted "
                "column is at most 60 characters wide. A width stated in 'widths' beats the fit, and "
                "is how a column gets wider than that.",
                "'wrap' wraps the data cells of the columns it reaches and sits them at the top of "
                "the row, empty cells included. The heading is left alone.",
                "'table' makes the tab's rows a spreadsheet table, with filter dropdowns on the "
                "headings. Its name is stated, or made from the tab name: a character a table name "
                "cannot hold becomes an underscore. It has no colour style unless a built-in one is "
                "named. A port with no rows gets a table over its headings and one empty row, the "
                "smallest table a spreadsheet accepts.",
                "Two tables of one name are refused, ignoring case, as a spreadsheet compares them. "
                "So is a name a template already uses on a tab this step leaves alone, before "
                "anything is saved. A tab this step replaces loses its old table with it.",
                "'header' makes the headings bold, a stated size, or both. What it does not state "
                "keeps the cell's own font.",
            ),
            field_help={
                "overwrite_sheets": "Defaults to false. A file already at 'path' is refused when the flow "
                "loads, and a template tab a sheet entry names is refused when the template is "
                "opened, until an author says in writing that replacing it is intended. A flow that "
                "runs every night needs it from its first night.",
                "template": "Read, never written. The tool writes to 'path' instead, so the same "
                "template can be used again next run.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong file format, a tab a workbook cannot carry, a repeat of any of tab, port
        or format rule, or a template it cannot use.

        Every one of these is answerable from the configuration and the filesystem alone. So every
        one of them is refused here, before a workbook is opened and long before anything is saved.

        The repeated port is checked here as well as by the loader, which is deliberate rather than
        duplication. The loader only speaks when there is a pipeline. A tool driven directly -- from
        a test, from a worked example, from anything running one step -- would otherwise stack one
        frame onto two tabs and report success, because two entries naming the same port arrive as
        one entry of the envelope long before this tool sees them.
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
            refuse_to_overwrite("write_excel", path, "overwrite_sheets", config["overwrite_sheets"])
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
        refuse_to_overwrite("write_excel", path, "overwrite_sheets", config["overwrite_sheets"])

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse any frame a tab cannot hold, before the workbook is opened.

        Every frame is checked, not only the first, and every check reads a schema or a column's
        extremes rather than writing anything. That is what lets a refusal about the last of eleven
        frames leave no pack on disk and no mark on the template.

        A format rule naming a column is checked here for the same reason, and this is the earliest
        phase that can answer it: a schema is the first thing in the run that knows which columns
        arrived. A frame with no rows has one like any other, so an empty period is not the month a
        renamed column slips through.
        """
        for entry in config["sheets"]:
            validate_formatted_columns(entry, inputs[entry["port"]])
            validate_layout_columns(entry, inputs[entry["port"]])

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
        """Open one workbook, write every tab into it, and save it once.

        One load and one save is the tool's central claim, so the whole of it is in plain sight here.
        One ``with`` opens the workbook. The loop only writes tabs into the open book. One ``save``
        is the last thing that happens inside it, and it is the only save on the way to a written
        pack. No branch below reaches a second load.

        One refusal lives here rather than in a validate step: a tab already in the template, with
        no permission to overwrite it. That is the only rule in this tool that cannot be answered
        without the workbook open, and answering it earlier would mean loading a file that can be
        megabytes a second time. Because it is raised before the save, a refused run still leaves
        nothing behind.
        """
        path = config["path"]
        with opened_once(path, config["template"]) as book:
            refuse_names_in_use(book, config["sheets"])
            for entry in config["sheets"]:
                worksheet = empty_tab(book, entry["sheet"], replace=config["overwrite_sheets"])
                frame = inputs[entry["port"]]
                # Two passes, in this order. The first writes every value and number format. The
                # second only changes how the tab looks, and must never set a value or a format.
                write_tab(worksheet, frame, format_rules(entry))
                lay_out_tab(worksheet, frame, entry)
            book.save(path)

        return Ports()


def _entries(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The ``(port, tab)`` pairs of a checked configuration, in the order they were written."""
    return tuple((entry["port"], entry["sheet"]) for entry in config["sheets"])


REGISTRY.register(WriteExcel())
