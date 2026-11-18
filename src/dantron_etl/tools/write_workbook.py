"""WriteWorkbook: one frame per named port, one tab each, in one load and one save.

This tool refreshes the data tabs of one XLSX workbook and rewrites the rest of it unchanged. A
delivered pack is usually curated. Somebody built the charts, the formats and the notes by hand.
Only a few tabs inside it hold figures that change every run, and those are the tabs this writes.

Rewritten is not the same as preserved. The whole file is read into an object model and a new file
is written from it, so every part comes back as that model spells it rather than as the bytes it
arrived as. Most of a pack survives that trip: tabs, formats, formulas, charts, pivot tables and the
cached values behind a pivot's filter dropdowns were all measured coming through. The extension
block on a pivot cache does not survive it. That block holds the parts of a pivot added after the
original file format, a slicer link among them, and a test in this package records the loss. Write
one pack and open it before pointing this at a workbook nobody could rebuild.

It reads one input port per tab its configuration names. Eleven tabs is one step, not eleven. That
is not only faster. A chain of eleven single-sheet writes has to decide, eleven times over, what to
do about the tabs already in the file, and the curated parts of the workbook get eleven chances to
be rebuilt. This tool opens the workbook once and saves it once.

It produces nothing at all, so a flow ends here. ``write_excel`` hands its one frame on so that a
sink can sit mid-flow. That argument does not survive the jump to many frames. Handing eleven frames
back out is a fan-out this tool would have invented, and handing back a receipt is shape without
information.

The template is read and never written. The tool loads it, replaces the tabs it was told about, and
saves the whole thing to ``path``. A ``path`` equal to the template is refused. A template
overwritten by its own first run is gone, and the run after it has nothing to start from.

With no template, a workbook already at ``path`` is written into rather than flattened. The rule
then has no exceptions: this tool never removes a tab it was not told about. The cost of that is
real. A flow whose tab names change between runs accumulates tabs nobody clears out. It is still the
cheaper mistake, because accumulation is visible in the file and silent destruction is not.

``replace`` governs a tab rather than the file. A tab already in the workbook survives until an
author says in writing that overwriting it is intended.

How a column's figures are *shown* is stated per tab, in ``formats``, and is stated or it does not
happen. A rule names either a type -- reaching every column of it, including the ones named by the
data -- or one column, which beats the rule about its type. Nothing is read off a column's name: a
writer deciding for itself that a decimal column was money is wrong the first time somebody puts a
reference number in one, and a writer deciding from the heading is wrong the first time somebody
renames it. The format code goes to the spreadsheet as written, unparsed and unchecked.

What a cell can hold is not this tool's ruling to make. Those rules belong to the destination, and
both workbook sinks read one copy of them, which sits in ``write_excel`` because it was written
first. A whole number past 2**53 - 1 is refused rather than rounded. A frame taller than a sheet is
refused rather than truncated. A date column gets a number format, because that format is the only
thing telling a cell it holds a date rather than a count of days.

The refusals are split across three places, and the split is deliberate. Anything answerable from
the configuration is refused in :meth:`WriteWorkbook.validate_config`. Anything answerable from the
frames is refused in :meth:`WriteWorkbook.validate_inputs`. One rule is left to
:meth:`WriteWorkbook.execute`: whether a tab is already in the workbook. That is the only question
here that cannot be answered without the workbook open, and asking it any earlier would cost a
second load of a file that can be megabytes.
"""

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any, Final

import polars as pl

from ..column_types import DECLARED_DTYPE, DECLARED_TYPES
from ..tool import (
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
from .write_excel import (
    FORBIDDEN_SHEET_CHARACTERS,
    MAX_DATA_ROWS,
    MAX_EXACT_INTEGER,
    MAX_SHEET_NAME_LENGTH,
    MAX_SHEET_ROWS,
    unsupported_dtype_reason,
)

_DATE_FORMAT: Final[str] = "yyyy-mm-dd"
_DATETIME_FORMAT: Final[str] = "yyyy-mm-dd hh:mm:ss"

#: Each dtype a frame can carry, as the one name a rule may call it. Built from the shared
#: vocabulary rather than written out again, so a type this library learns to name is a type a
#: format rule can speak about on the same day. A dtype missing from it -- an Int32, say -- is named
#: by no rule, and a rule about the type it is closest to does not reach it.
_DECLARED_NAME: Final[Mapping[Any, str]] = {DECLARED_DTYPE[name].base_type(): name for name in DECLARED_TYPES}

#: The widest a spreadsheet column can be, in characters.
_MAX_WIDTH: Final[float] = 255
#: The largest font size a spreadsheet accepts, in points.
_MAX_FONT_SIZE: Final[float] = 409
#: A fitted column is never wider than this. A paragraph of text would otherwise make a column
#: hundreds of characters wide. A stated width is how a column goes past it.
_FIT_CAP: Final[float] = 60
#: Room either side of the text in a fitted column.
_FIT_PADDING: Final[float] = 2
#: Room for the filter button a table puts on each heading.
_FILTER_BUTTON: Final[float] = 2
#: The font size a heading is measured at when nobody states one.
_DEFAULT_FONT_SIZE: Final[float] = 11
#: The longest a number is shown under the spreadsheet's General format.
_GENERAL_NUMBER_LENGTH: Final[int] = 11

#: The spreadsheet's built-in table styles. Written out here rather than read from openpyxl, which
#: this module only imports when it opens a workbook.
_TABLE_STYLES: Final[frozenset[str]] = frozenset(
    [f"TableStyleLight{i}" for i in range(1, 22)]
    + [f"TableStyleMedium{i}" for i in range(1, 29)]
    + [f"TableStyleDark{i}" for i in range(1, 12)]
)
_MAX_TABLE_NAME_LENGTH: Final[int] = 255
_TABLE_NAME: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_\\][A-Za-z0-9_.]*$")
_NOT_IN_A_TABLE_NAME: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.]")
#: Names a spreadsheet would read as a cell address, in either of its two address styles.
_A1_ADDRESS: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]{1,3}[0-9]+$")
_R1C1_ADDRESS: Final[re.Pattern[str]] = re.compile(r"^[RrCc]$|^[Rr][0-9]*[Cc][0-9]*$|^[Rr][0-9]+$")

_PLAIN_PATH: Final[str] = f"{SCRATCH_TOKEN}/pack.xlsx"
_LAID_OUT_PATH: Final[str] = f"{SCRATCH_TOKEN}/laid_out.xlsx"
_FROM_TEMPLATE_PATH: Final[str] = f"{SCRATCH_TOKEN}/quarter.xlsx"
_TEMPLATE_PATH: Final[str] = f"{SCRATCH_TOKEN}/template.xlsx"


class WriteWorkbook:
    """Write each named frame to its own tab of one XLSX workbook, and produce nothing."""

    name = "write_workbook"
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
            ),
            "replace": ConfigField(
                type="boolean",
                description="Whether a tab already in the workbook may be overwritten; leave it out "
                "and such a tab is refused.",
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
                "refused. With "
                "no template, a workbook already at 'path' is written into rather than flattened, "
                "so this tool never removes a tab it was not told about. A tab that is already "
                "there is refused until 'replace' says in writing that overwriting it is intended. "
                "A replaced tab keeps its place among the others. Nothing leaves on an output port: "
                "a flow ends here."
            ),
            examples=(_plain_write_example(), _template_example(), _laid_out_example()),
            notes=(
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
                "the two were written in. Nothing is read off a column's name.",
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
                "replace": "Defaults to false, and it is about a tab rather than the file: a tab "
                "already in the workbook is refused until an author says in writing that "
                "overwriting it is intended.",
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
        if not path.lower().endswith(".xlsx"):
            raise ConfigError(f"write_workbook only writes .xlsx workbooks; got path '{path}'")

        tabs: dict[str, str] = {}
        ports: set[str] = set()
        for port, sheet in _entries(config):
            _validate_tab_name(sheet)
            if sheet in tabs:
                raise ConfigError(
                    f"write_workbook was told to write the tab {sheet!r} twice, from ports "
                    f"{tabs[sheet]!r} and {port!r}; one of the two frames would be delivered and "
                    f"the other lost. Give each port its own tab, or join the two frames upstream."
                )
            if port in ports:
                raise ConfigError(
                    f"write_workbook was told to read the port {port!r} twice; a port carries one "
                    f"frame, so the same rows would be written to both tabs. Name a different port "
                    f"for one of the two tabs."
                )
            tabs[sheet] = port
            ports.add(port)

        for entry in config["sheets"]:
            _validate_format_rules(entry)
            _validate_layout(entry)
        _validate_table_names(config["sheets"])

        template = config["template"]
        if template is None:
            return
        if _names_the_same_file(template, path):
            raise ConfigError(
                f"write_workbook refuses to write to its own template; 'template' is '{template}' "
                f"and 'path' is '{path}', and the two name the same file on disk. The template is "
                f"read every run, and overwriting it would leave the next run with nothing to start "
                f"from. Change 'path' to name the pack to be delivered, somewhere other than the "
                f"template."
            )
        if not Path(template).exists():
            raise ConfigError(
                f"write_workbook cannot find the template '{template}'; a template that is not "
                f"there is not a template. Check the path, or leave 'template' out to start from "
                f"nothing."
            )

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
            _validate_formatted_columns(entry, inputs[entry["port"]])
            _validate_layout_columns(entry, inputs[entry["port"]])

        for port, sheet in _entries(config):
            frame = inputs[port]

            if frame.height > MAX_DATA_ROWS:
                raise DataError(
                    f"write_workbook refuses the {frame.height:,} row(s) on port {port!r}, bound "
                    f"for tab {sheet!r}; a tab's header takes one row, leaving {MAX_DATA_ROWS:,} "
                    f"row(s) for data (a worksheet holds at most {MAX_SHEET_ROWS:,} rows in total)",
                    offending=frame.head(0),
                    total=frame.height,
                )

            for name, dtype in frame.schema.items():
                reason = unsupported_dtype_reason(dtype)
                if reason is not None:
                    raise ConfigError(
                        f"write_workbook cannot write column {name!r} of dtype {dtype} on port "
                        f"{port!r}: {reason}; format it into something a cell can hold, one step "
                        f"upstream"
                    )

            for name, dtype in frame.schema.items():
                if not dtype.is_integer():
                    continue
                offending = frame.filter(pl.col(name).abs() > MAX_EXACT_INTEGER)
                if offending.height:
                    raise DataError(
                        f"write_workbook refuses column {name!r} on port {port!r}: a spreadsheet "
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

        One refusal lives here rather than in a validate step: a tab already in the workbook, with
        no permission to overwrite it. That is the only rule in this tool that cannot be answered
        without the workbook open, and answering it earlier would mean loading a file that can be
        megabytes a second time. Because it is raised before the save, a refused run still leaves
        nothing behind.
        """
        path = config["path"]
        with _opened_once(path, config["template"]) as book:
            _refuse_names_in_use(book, config["sheets"])
            for entry in config["sheets"]:
                worksheet = _empty_tab(book, entry["sheet"], replace=config["replace"])
                frame = inputs[entry["port"]]
                # Two passes, in this order. The first writes every value and number format. The
                # second only changes how the tab looks, and must never set a value or a format.
                _write_tab(worksheet, frame, _format_rules(entry))
                _lay_out_tab(worksheet, frame, entry)
            book.save(path)

        return Ports()


def _entries(config: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """The ``(port, tab)`` pairs of a checked configuration, in the order they were written."""
    return tuple((entry["port"], entry["sheet"]) for entry in config["sheets"])


def _format_rules(entry: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The format rules of one sheet entry, in the order they were written.

    An entry that states none reads the same as one that states an empty list: no rules, so nothing
    is formatted beyond the date format this tool writes on its own. A default is never filled
    inside an array's element, so the field is read with ``get`` rather than by subscript.
    """
    return tuple(entry.get("formats") or ())


def _validate_format_rules(entry: Mapping[str, Any]) -> None:
    """Refuse a rule that cannot mean one thing, and a type or column ruled on twice.

    Exactly one of ``type`` and ``column``. Both is two rules written as one, and which of them the
    author meant is not this tool's guess to make. Neither is a format code with nothing to apply
    it to.

    A type or a column named by two rules is refused rather than resolved, because resolving it
    means picking one, and either answer -- the first rule or the last -- is a reading of the list
    the author may not share. A repeat means they believed the list was ordered, or forgot the rule
    they had already written; both are worth a sentence naming the repeated word.
    """
    sheet = entry["sheet"]
    seen: dict[tuple[str, str], str] = {}
    for rule in _format_rules(entry):
        stated = tuple(field for field in ("type", "column") if rule.get(field) is not None)
        if len(stated) == 2:
            raise ConfigError(
                f"write_workbook was given a format rule on tab {sheet!r} carrying both 'type' "
                f"{rule['type']!r} and 'column' {rule['column']!r}; a rule applies to one or the "
                f"other, never both. Write two rules, or drop the one you did not mean."
            )
        if not stated:
            raise ConfigError(
                f"write_workbook was given a format rule on tab {sheet!r} carrying neither 'type' "
                f"nor 'column', so the format {rule['format']!r} has nothing to apply to. Give "
                f"exactly one of them."
            )
        field = stated[0]
        key = (field, rule[field])
        if key in seen:
            raise ConfigError(
                f"write_workbook was told twice how to show {field} {rule[field]!r} on tab "
                f"{sheet!r}, as {seen[key]!r} and as {rule['format']!r}; one of the two would be "
                f"delivered and which one is not a reading of this list to be relied on. Keep the "
                f"rule you meant and take the other out."
            )
        seen[key] = rule["format"]


def _validate_formatted_columns(entry: Mapping[str, Any], frame: pl.DataFrame) -> None:
    """Refuse a rule naming a column the frame does not carry.

    Writing the rule asserts the column is there. Passing over a rule that matched nothing would
    deliver an unformatted pack and say nothing about why, and the author reads the configuration,
    sees the rule, and looks anywhere but at the rule.
    """
    for rule in _format_rules(entry):
        column = rule.get("column")
        if column is not None and column not in frame.columns:
            raise ConfigError(
                f"write_workbook was told to format the column {column!r} on tab "
                f"{entry['sheet']!r}, which port {entry['port']!r} does not carry; it carries: "
                f"{', '.join(frame.columns)}"
            )


def _validate_layout(entry: Mapping[str, Any]) -> None:
    """Refuse a layout option that cannot mean one thing, before anything is written.

    The width and wrap rules follow the format rules: exactly one of ``type`` and ``column``, and no
    type or column named twice. A width, a heading size, a table style and a table name are each
    checked against what a spreadsheet accepts, so the file never needs repairing when it opens.
    """
    sheet = entry["sheet"]
    _validate_layout_rules(sheet, "widths", entry.get("widths") or ())
    _validate_layout_rules(sheet, "wrap", entry.get("wrap") or ())

    for rule in entry.get("widths") or ():
        width = rule["width"]
        if not 0 < width <= _MAX_WIDTH:
            raise ConfigError(
                f"write_workbook was given a column width of {width!r} in 'widths' on tab "
                f"{sheet!r}; a width must be more than 0 and at most {_MAX_WIDTH:g} characters."
            )

    header = entry.get("header")
    if header is not None:
        if header.get("bold") is None and header.get("size") is None:
            raise ConfigError(
                f"write_workbook was given a 'header' on tab {sheet!r} that states nothing; give "
                f"'bold', 'size' or both, or leave 'header' out."
            )
        size = header.get("size")
        if size is not None and not 1 <= size <= _MAX_FONT_SIZE:
            raise ConfigError(
                f"write_workbook was given a heading 'size' of {size!r} on tab {sheet!r}; a font "
                f"size must be at least 1 and at most {_MAX_FONT_SIZE:g} points."
            )

    table = entry.get("table")
    if table is None:
        return
    style = table.get("style")
    if style is not None and style not in _TABLE_STYLES:
        raise ConfigError(
            f"write_workbook was given the table 'style' {style!r} on tab {sheet!r}, which is not "
            f"one of the spreadsheet's built-in table styles. Name one such as 'TableStyleLight1' "
            f"or 'TableStyleMedium2', or leave 'style' out for a plain table."
        )
    name = table.get("name")
    if name is not None and not _is_table_name(name):
        raise ConfigError(
            f"write_workbook cannot use the table 'name' {name!r} on tab {sheet!r}. A table name "
            f"starts with a letter or an underscore, holds only letters, digits, underscores and "
            f"full stops, is at most {_MAX_TABLE_NAME_LENGTH} characters, and must not read as a "
            f"cell address such as 'A1' or 'R1C1'."
        )


def _validate_layout_rules(sheet: str, field: str, rules: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a width or wrap rule naming both or neither of type and column, or repeating one."""
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        stated = tuple(name for name in ("type", "column") if rule.get(name) is not None)
        if len(stated) != 1:
            which = "both" if stated else "neither"
            raise ConfigError(
                f"write_workbook was given a rule in '{field}' on tab {sheet!r} carrying {which} "
                f"'type' and 'column'; a rule applies to exactly one of them."
            )
        kind = stated[0]
        key = (kind, rule[kind])
        if key in seen:
            raise ConfigError(
                f"write_workbook was given two rules in '{field}' on tab {sheet!r} about {kind} "
                f"{rule[kind]!r}; keep the one you meant and take the other out."
            )
        seen.add(key)


def _is_table_name(name: str) -> bool:
    """Whether a spreadsheet accepts ``name`` as a table name."""
    return (
        len(name) <= _MAX_TABLE_NAME_LENGTH
        and _TABLE_NAME.match(name) is not None
        and _A1_ADDRESS.match(name) is None
        and _R1C1_ADDRESS.match(name) is None
    )


def _table_name(entry: Mapping[str, Any]) -> str:
    """The table name of a sheet entry asking for a table: the stated one, or one made from the tab.

    A character a table name cannot hold becomes an underscore. A name that would then start with
    a digit or a full stop, or read as a cell address, gets an underscore in front.
    """
    stated = entry["table"].get("name")
    if stated is not None:
        return stated
    name = _NOT_IN_A_TABLE_NAME.sub("_", entry["sheet"])
    if not _is_table_name(name):
        name = f"_{name}"
    return name


def _validate_table_names(entries: Sequence[Mapping[str, Any]]) -> None:
    """Refuse two tables of one name. A spreadsheet compares table names ignoring case."""
    seen: dict[str, tuple[str, str]] = {}
    for entry in entries:
        if entry.get("table") is None:
            continue
        name = _table_name(entry)
        earlier = seen.get(name.lower())
        if earlier is not None:
            raise ConfigError(
                f"write_workbook would make two tables of one name: {earlier[0]!r} on tab "
                f"{earlier[1]!r} and {name!r} on tab {entry['sheet']!r}. A spreadsheet compares "
                f"table names ignoring case. Give one of them its own 'name'."
            )
        seen[name.lower()] = (name, entry["sheet"])


def _validate_layout_columns(entry: Mapping[str, Any], frame: pl.DataFrame) -> None:
    """Refuse a width or wrap rule naming a column the frame does not carry, and a table over two
    headings that differ only by case, which a spreadsheet table cannot hold."""
    for field in ("widths", "wrap"):
        for rule in entry.get(field) or ():
            column = rule.get("column")
            if column is not None and column not in frame.columns:
                raise ConfigError(
                    f"write_workbook was given a rule in '{field}' about the column {column!r} on "
                    f"tab {entry['sheet']!r}, which port {entry['port']!r} does not carry; it "
                    f"carries: {', '.join(frame.columns)}"
                )

    if entry.get("table") is None:
        return
    seen: dict[str, str] = {}
    for column in frame.columns:
        earlier = seen.get(column.lower())
        if earlier is not None:
            raise ConfigError(
                f"write_workbook cannot make a table on tab {entry['sheet']!r} over the headings "
                f"{earlier!r} and {column!r}: a table's headings must differ ignoring case. Rename "
                f"one of them upstream, or leave 'table' out."
            )
        seen[column.lower()] = column


def _resolve_rules(
    schema: Mapping[str, pl.DataType],
    rules: Sequence[Mapping[str, Any]],
    field: str | None = None,
) -> dict[str, Any]:
    """Which columns a list of type-or-column rules reaches, and what each one gets.

    Formats, widths and wraps all read their rules here, so the three agree on one rule: a column
    rule beats a type rule, whichever order the two were written in. A column no rule reaches is
    left out.

    Each reached column gets its rule's ``field``: its format code, or its width. A wrap rule has
    nothing but its target, so with no ``field`` each reached column gets ``True``.
    """

    def given(rule: Mapping[str, Any]) -> Any:
        return True if field is None else rule[field]

    by_type = {rule["type"]: given(rule) for rule in rules if rule.get("type") is not None}
    by_column = {rule["column"]: given(rule) for rule in rules if rule.get("column") is not None}
    reached: dict[str, Any] = {}
    for name, dtype in schema.items():
        if name in by_column:
            reached[name] = by_column[name]
            continue
        declared = _DECLARED_NAME.get(dtype.base_type())
        if declared in by_type:
            reached[name] = by_type[declared]
    return reached


def column_formats(
    schema: Mapping[str, pl.DataType],
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    """Which format code each column of ``schema`` is shown under, or ``None`` for the default.

    A rule about a column beats a rule about that column's type, whichever order the two were
    written in: precedence is about which kind of rule it is, so the general rule can be stated
    first, where it reads most naturally, and the exception underneath it.

    A rule about a type reaches every column of that type, including the ones nobody could have
    named -- a cross-tab gains a column per period, so naming them one at a time means editing the
    pipeline file every month, and the month somebody forgets, the new column is the only
    unformatted one on the tab.

    A date and an instant start with the format this tool has always written, and a rule about
    either type replaces it. That format is a default rather than a guarantee: it exists because a
    date under no format reads as a count of days since 1899, and a flow writing for a reader who
    expects ``dd/mm/yyyy`` is stating a display choice that nothing downstream depends on.

    Nothing else is guessed. A column no rule matched comes back ``None`` and keeps the workbook's
    own default, because a tool deciding on its own that a decimal column was money would be wrong
    the first time somebody wrote a reference number in one, and a tool reading the column's *name*
    to decide would be wrong the first time somebody renamed it.

    Deliberately about a schema and a list of rules, naming neither a sheet nor a workbook, because
    ``write_excel`` is the intended second caller: the two sinks share every other rule about what a
    cell may hold, and a one-tab flow cannot state a format at all until it shares this one too.
    """
    formats: dict[str, str | None] = {name: _default_format(dtype) for name, dtype in schema.items()}
    formats.update(_resolve_rules(schema, rules, "format"))
    return formats


def _default_format(dtype: pl.DataType) -> str | None:
    """The format a column of ``dtype`` is shown under before any rule is read.

    Only a date and an instant have one, and it is not a display choice made on anybody's behalf: a
    date under no format at all is a count of days since 1899 and reads as one, so the format is
    what tells a cell which of the two it holds. Every other dtype starts under nothing, because a
    format this tool chose for itself would be a claim about how the pack is read that it is not
    entitled to make.

    Read by :func:`column_formats`, which is where the argument for replacing this with a rule is.
    """
    if isinstance(dtype, pl.Date):
        return _DATE_FORMAT
    if isinstance(dtype, pl.Datetime):
        return _DATETIME_FORMAT
    return None


def _names_the_same_file(template: str, path: str) -> bool:
    """Whether two written paths land on the same file.

    Two checks, and either one is enough to refuse.

    First, both paths are resolved and compared. Resolving walks a string out to one spelling: a
    ``..`` segment is taken out, a relative path is joined to the working directory, and a symlink
    is followed to what it points at. Comparing the two strings as written lets all three of those
    through, and the run that follows loads the template and saves over it.

    Resolving has to be the first check, because ``path`` usually does not exist yet. The pack is
    the file this run is about to create, and the filesystem cannot be asked about a file that is
    not there.

    Second, when both files are already on disk, the filesystem is asked directly. A hard link is
    two directory entries for one file. Neither entry points at the other, the way a symlink does,
    so both resolve to themselves and the first check sees two files. ``Path.samefile`` sees one.
    Either file missing means no answer, and the run is allowed: creating a file that is not there
    cannot overwrite one that is.

    What the two checks together still do not cover. The filesystem is read while the configuration
    is checked, not while the workbook is saved, so a link made in between is not seen. And both
    checks trust what the filesystem reports: a mount that gives every file the same identity can
    call two files one, and a refusal here costs one edit either way.
    """
    if Path(template).resolve() == Path(path).resolve():
        return True
    try:
        return Path(template).samefile(path)
    except OSError:
        return False


def _validate_tab_name(sheet: str) -> None:
    """Refuse a tab name a workbook cannot carry, in this library's own words.

    A tab name is written by a person for a person, so it attracts exactly the characters a workbook
    forbids: a date with slashes, a heading with a colon, a draft marker in brackets. Left to the
    library underneath, a whole pack fails part way through with a message about worksheets. Left to
    a writer that quietly renames, the pack is delivered with a tab nobody can find under the name
    the flow calls it.
    """
    if len(sheet) > MAX_SHEET_NAME_LENGTH:
        raise ConfigError(
            f"write_workbook refuses the tab name {sheet!r}: a workbook tab name may be at most "
            f"{MAX_SHEET_NAME_LENGTH} characters. Shorten it."
        )
    for character in FORBIDDEN_SHEET_CHARACTERS:
        if character in sheet:
            raise ConfigError(
                f"write_workbook refuses the tab name {sheet!r}: a workbook tab name may not "
                f"contain the character {character}. Take it out of the name."
            )


@contextmanager
def _opened_once(path: str, template: str | None) -> Iterator[Any]:
    """The one workbook every tab is written into, opened once and closed however the run ends.

    Three ways in, and exactly one load among them. A template is loaded and saved elsewhere, so it
    survives the run. A workbook already at ``path`` and no template means loading that file and
    writing into it, because this tool never removes a tab it was not told about. Otherwise the book
    is new.

    ``openpyxl`` is imported here rather than at the top of the module, and named if it is missing,
    the way this library reaches for a database driver only when a database is asked for. Every tool
    module is imported when the catalog is, so a top-level import would cost every consumer,
    including the many tools that never open a workbook.
    """
    try:
        import openpyxl
    except ImportError as error:
        raise ConfigError(
            "write_workbook needs openpyxl, which this environment does not have installed; "
            "install 'openpyxl' to write a workbook"
        ) from error

    if template is not None:
        book = openpyxl.load_workbook(template)
    elif Path(path).exists():
        book = openpyxl.load_workbook(path)
    else:
        book = openpyxl.Workbook()
        # A new workbook arrives with one empty tab nobody asked for.
        book.remove(book.active)

    try:
        yield book
    finally:
        book.close()


def _empty_tab(book: Any, sheet: str, *, replace: bool) -> Any:
    """A tab named ``sheet``, empty and ready to be written, or a refusal if one is already there.

    This is the tool's one refusal that needs the workbook open, so it is the one refusal left this
    late. It is raised while the book is still only in memory, so a run that ends here writes
    nothing.

    A tab being replaced is removed and remade in the place it already held. A tab that moved to the
    end of the pack is a tab a reader reports as missing.
    """
    if sheet not in book.sheetnames:
        return book.create_sheet(sheet)

    if not replace:
        raise ConfigError(
            f"write_workbook refuses to overwrite the tab {sheet!r}, which is already in the "
            f"workbook; set replace: true to overwrite it deliberately, or write to a tab of "
            f"another name"
        )

    where = book.sheetnames.index(sheet)
    book.remove(book[sheet])
    return book.create_sheet(sheet, where)


def _write_tab(worksheet: Any, frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> None:
    """Write one frame to one tab: a header row, the rows under it, and the number formats.

    A date and an instant carry a format whether the tab asked for one or not, because that format
    is the only thing telling a cell it holds a date rather than a count of days. Everything else is
    formatted only where a rule says so: a format this tool chose for itself would be a claim about
    how the pack is read that it is not entitled to make.

    The header is written before the formats are worked out and is never given one. A heading is a
    word, and a money format on a word shows nothing while saying the writer cannot tell a heading
    from a figure.

    A cell with nothing in it is formatted like the rest of its column. The format belongs to the
    cell rather than to the value that happens to be in it: a money column with a gap in it is still
    a money column, and formatting around the gaps leaves a chequerboard that shows the moment
    anybody types into the file.
    """
    for index, name in enumerate(frame.columns, start=1):
        _as_text_if_formula_like(worksheet.cell(row=1, column=index, value=name))

    formats = column_formats(frame.schema, rules)
    # Once per column, in the order the cells are written. The loop under this one runs for every
    # cell of the pack, and a tab holds up to a million rows, so a lookup by name in there is a
    # million lookups for an answer that cannot change between rows.
    codes = tuple(formats[name] for name in frame.columns)

    for row_number, row in enumerate(frame.iter_rows(), start=2):
        for index, (value, code) in enumerate(zip(row, codes, strict=True), start=1):
            if value == "":
                value = None
            cell = worksheet.cell(row=row_number, column=index, value=value)
            if isinstance(value, str):
                _as_text_if_formula_like(cell)
            if code is not None:
                cell.number_format = code


def _as_text_if_formula_like(cell: Any) -> None:
    """Keep a text value that starts with ``=`` as text, not a formula.

    openpyxl stores any string starting with ``=`` as a formula, so a free-text value such as
    ``=HYPERLINK(...)`` would run on the reader's machine. The cell is forced back to text, and the
    quote prefix stops a spreadsheet program reading it as a formula when the cell is edited.
    """
    if cell.data_type == "f":
        cell.data_type = "s"
        cell.quotePrefix = True


def _refuse_names_in_use(book: Any, entries: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a table name the workbook already uses on a tab this step leaves alone.

    A tab this step writes is remade, so its old tables go with it and their names are free. A tab
    it leaves alone keeps its tables, and so does every name the workbook defines. This is only
    known with the workbook open. It is still refused before the save, so nothing is written.
    """
    written = {entry["sheet"] for entry in entries}
    in_use: dict[str, str] = {}
    for worksheet in book.worksheets:
        if worksheet.title in written:
            continue
        for existing in worksheet.tables:
            in_use[existing.lower()] = f"the table {existing!r} on tab {worksheet.title!r}"
    for defined in book.defined_names:
        in_use.setdefault(defined.lower(), f"the workbook's defined name {defined!r}")

    for entry in entries:
        if entry.get("table") is None:
            continue
        name = _table_name(entry)
        clash = in_use.get(name.lower())
        if clash is not None:
            raise ConfigError(
                f"write_workbook cannot name the table on tab {entry['sheet']!r} {name!r}: the "
                f"workbook already has {clash}, and a spreadsheet compares these names ignoring "
                f"case. Give the table another 'name'."
            )


def _lay_out_tab(worksheet: Any, frame: pl.DataFrame, entry: Mapping[str, Any]) -> None:
    """Apply a sheet entry's layout options to a tab whose values are already written.

    Layout never touches a value. Each step below sets a font, a column width, an alignment or a
    table, and nothing else. None of them may set a cell's value or its number format, so every
    figure is the same with layout as without it. ``TestLayoutNeverTouchesAFigure`` fails if that
    stops being true. A new layout option belongs here only if it keeps that promise.
    """
    header = entry.get("header")
    if header is not None:
        _style_headings(worksheet, frame.width, header)
    _set_widths(worksheet, frame, entry)
    _wrap_columns(worksheet, frame, entry.get("wrap") or ())
    if entry.get("table") is not None:
        _add_table(worksheet, frame, entry)


def _style_headings(worksheet: Any, width: int, header: Mapping[str, Any]) -> None:
    """Make the heading row bold, a stated size, or both. A setting not stated keeps the cell's font."""
    for index in range(1, width + 1):
        cell = worksheet.cell(row=1, column=index)
        font = copy(cell.font)
        if header.get("bold") is not None:
            font.b = header["bold"]
        if header.get("size") is not None:
            font.sz = header["size"]
        cell.font = font


def _set_widths(worksheet: Any, frame: pl.DataFrame, entry: Mapping[str, Any]) -> None:
    """Set each column's width: fitted first if asked for, then any stated width on top."""
    from openpyxl.utils import get_column_letter

    widths = _fitted_widths(frame, entry) if entry.get("fit_widths") else {}
    widths.update(_resolve_rules(frame.schema, entry.get("widths") or (), "width"))
    for name, width in widths.items():
        letter = get_column_letter(frame.columns.index(name) + 1)
        worksheet.column_dimensions[letter].width = width


def _fitted_widths(frame: pl.DataFrame, entry: Mapping[str, Any]) -> dict[str, float]:
    """The width 'fit_widths' gives each column.

    A column is as wide as its heading or its longest value as shown, whichever is longer. Then it
    gets room either side, and it is never wider than the cap.

    The heading is measured at its own font size, so a 14-point heading takes more room than an
    11-point one. On a table, each heading also needs room for its filter button.
    """
    formats = column_formats(frame.schema, _format_rules(entry))
    heading_size = (entry.get("header") or {}).get("size") or _DEFAULT_FONT_SIZE
    button = _FILTER_BUTTON if entry.get("table") is not None else 0

    widths: dict[str, float] = {}
    for name in frame.columns:
        heading = len(name) * heading_size / _DEFAULT_FONT_SIZE + button
        longest = max(heading, _longest_shown(frame[name], formats[name]))
        widths[name] = min(longest + _FIT_PADDING, _FIT_CAP)
    return widths


def _wrap_columns(worksheet: Any, frame: pl.DataFrame, rules: Sequence[Mapping[str, Any]]) -> None:
    """Wrap the data cells of each column the rules reach, and sit them at the top of the row.

    Every data cell is set, empty ones included. The heading is left alone.
    """
    from openpyxl.styles import Alignment

    wrapped = Alignment(wrap_text=True, vertical="top")
    for name in _resolve_rules(frame.schema, rules):
        column = frame.columns.index(name) + 1
        for row in range(2, frame.height + 2):
            worksheet.cell(row=row, column=column).alignment = wrapped


def _add_table(worksheet: Any, frame: pl.DataFrame, entry: Mapping[str, Any]) -> None:
    """Make the tab's heading and rows a spreadsheet table, plain unless a built-in style is named.

    A port with no rows still gets a table: its heading and one empty row, the smallest table a
    spreadsheet accepts.
    """
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    last_row = max(frame.height + 1, 2)
    ref = f"A1:{get_column_letter(frame.width)}{last_row}"
    style = entry["table"].get("style")
    info = None if style is None else TableStyleInfo(name=style, showRowStripes=True)
    worksheet.add_table(Table(displayName=_table_name(entry), ref=ref, tableStyleInfo=info))


def _longest_shown(column: pl.Series, code: str | None) -> float:
    """How many characters the longest value of ``column`` takes on screen, under ``code``.

    A figure is measured as it is shown, not as it is stored: 1234567.0 under '#,##0.00' shows as
    '1,234,567.00', and a column fitted to the stored digits shows '####'. Read from the column's
    extremes, not row by row.
    """
    values = column.drop_nulls()
    if values.is_empty():
        return 0
    dtype = column.dtype
    if dtype == pl.String:
        return float(values.str.len_chars().max() or 0)
    if dtype == pl.Boolean:
        return len("FALSE")
    if dtype.is_temporal():
        return _date_shown_length(code or "")
    if dtype.is_numeric():
        if dtype.is_float():
            values = values.filter(values.is_finite())
            if values.is_empty():
                return _GENERAL_NUMBER_LENGTH
        if code is None:
            return min(float(values.cast(pl.String).str.len_chars().max() or 0), _GENERAL_NUMBER_LENGTH)
        largest = float(values.abs().max())
        negative = float(values.min()) < 0
        return _number_shown_length(largest, negative, code)
    return float(values.cast(pl.String).str.len_chars().max() or 0)


def _number_shown_length(largest: float, negative: bool, code: str) -> int:
    """How long ``largest`` is shown under a number format code, erring on the wide side.

    Reads the decimals, the thousands separator and a percent sign from the code, and counts every
    other character in it as shown. That over-counts quotes and escapes, which only makes a column a
    little wider.
    """
    sections = code.split(";")
    section = re.sub(r"\[[^\]]*\]", "", sections[0])
    integer_part, _, decimal_part = section.partition(".")
    decimals = sum(1 for character in decimal_part if character in "0#?")
    if "%" in section:
        largest *= 100
    separator = "," if "," in integer_part else ""
    shown = len(f"{largest:{separator}.{decimals}f}")
    literal = sum(1 for character in section if character not in "0#?.,%")
    if negative:
        negative_section = re.sub(r"\[[^\]]*\]", "", sections[1]) if len(sections) > 1 else section
        literal = max(literal, sum(1 for character in negative_section if character not in "0#?.,%")) + 1
    return shown + literal + ("%" in section)


def _date_shown_length(code: str) -> int:
    """How long a date is shown under a date format code: the code's own length, with a month or
    day written out in full counted at its longest."""
    shown = re.sub(r'\[[^\]]*\]|["\\]', "", code)
    return len(re.sub(r"(?i)mmmm|dddd", "x" * len("September"), shown))


@contextmanager
def _template_setup() -> Iterator[None]:
    """A curated workbook already on disk, standing in for the pack somebody built by hand.

    Planted with :func:`scratch_path`, which resolves the same ``{scratch}`` token the example's own
    configuration holds, into the directory :func:`check_tool` opens for the run. Nothing here tears
    the file back down: the directory is the checker's, and it takes whatever a setup left inside it.
    """
    import openpyxl

    book = openpyxl.Workbook()
    book.remove(book.active)
    notes = book.create_sheet("Notes")
    notes["A1"] = "Written by hand; the flow never names this tab."
    book.save(scratch_path(_TEMPLATE_PATH))
    book.close()
    yield


def _plain_write_example() -> WorkedExample:
    return WorkedExample(
        scenario="write a pack of two tabs where no workbook exists yet",
        configuration={
            "path": _PLAIN_PATH,
            "sheets": [
                {
                    "port": "figures",
                    "sheet": "Figures",
                    "formats": [{"type": "Float64", "format": "#,##0.00"}],
                },
                {"port": "detail", "sheet": "Detail"},
            ],
        },
        explanation="The two entries in 'sheets' are what say this step reads 'figures' and "
        "'detail'. Each frame lands on its own tab, in the order the entries name them, and "
        "nothing leaves on an output port. The first tab states one format rule, so every decimal "
        "column on it is shown as money -- 'amount' reads 10.25 and would read 1,234.57 for a "
        "figure that long, while the cell keeps every digit that went in. The second tab states "
        "none, so its cells are shown however the workbook shows them.",
        inputs=Ports(
            {
                "figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]}),
                "detail": pl.DataFrame({"key": ["r1", "r2"], "code": ["A", "B"]}),
            }
        ),
        outputs=Ports(),
    )


def _template_example() -> WorkedExample:
    return WorkedExample(
        scenario="refresh one tab of a workbook somebody built by hand",
        configuration={
            "path": _FROM_TEMPLATE_PATH,
            "template": _TEMPLATE_PATH,
            "sheets": [{"port": "figures", "sheet": "Figures"}],
        },
        explanation="The template already holds a 'Notes' tab. It is loaded, the 'Figures' tab is "
        "written into it, and the whole workbook is saved to 'path' -- so 'Notes' arrives in the "
        "pack exactly as it was, and the template itself is untouched and ready for the next run.",
        inputs=Ports({"figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})}),
        outputs=Ports(),
        setup=_template_setup,
    )


def _laid_out_example() -> WorkedExample:
    return WorkedExample(
        scenario="lay out a figures tab as a table and a notes tab as a title over paragraphs",
        configuration={
            "path": _LAID_OUT_PATH,
            "sheets": [
                {
                    "port": "figures",
                    "sheet": "Figures",
                    "formats": [{"type": "Float64", "format": "#,##0.00"}],
                    "fit_widths": True,
                    "table": {"style": "TableStyleLight1"},
                    "header": {"bold": True},
                },
                {
                    "port": "notes",
                    "sheet": "Notes",
                    "widths": [{"type": "String", "width": 100}],
                    "wrap": [{"type": "String"}],
                    "header": {"bold": True, "size": 14},
                },
            ],
        },
        explanation="The 'Figures' tab becomes a table called 'Figures', with filter dropdowns on "
        "its bold headings, and each column is as wide as its heading or its longest figure as "
        "shown. The 'Notes' tab has one column 100 characters wide, a bold 14-point heading that "
        "reads as its title, and paragraphs that wrap. Every value and number format is the same "
        "as it would be without these options.",
        inputs=Ports(
            {
                "figures": pl.DataFrame({"key": ["r1", "r2"], "amount": [1234567.0, -5.5]}),
                "notes": pl.DataFrame({"About these figures": ["The first paragraph.", "The second paragraph."]}),
            }
        ),
        outputs=Ports(),
    )


REGISTRY.register(WriteWorkbook())
