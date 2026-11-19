"""write_excel: the refusals that can be made before the workbook is opened.

Each function reads the configuration, or a frame's schema, and refuses what cannot be written: a
bad tab name, a format, width or wrap rule that is ambiguous, a table name a spreadsheet would
reject, and a template that is the output file itself.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import polars as pl

from ..authoring import ConfigError
from ._spreadsheet import FORBIDDEN_SHEET_CHARACTERS, MAX_SHEET_NAME_LENGTH
from ._workbook_rules import format_rules

#: The widest a spreadsheet column can be, in characters.
_MAX_WIDTH: Final[float] = 255
#: The largest font size a spreadsheet accepts, in points.
_MAX_FONT_SIZE: Final[float] = 409

#: The built-in table styles, listed here so openpyxl is not imported to check one.
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


def validate_tab_name(sheet: str) -> None:
    """Refuse a tab name that is too long or holds a character a workbook forbids.

    People write dates with slashes and headings with colons into tab names. Renaming quietly would
    deliver a tab nobody can find; refusing names the problem.
    """
    if len(sheet) > MAX_SHEET_NAME_LENGTH:
        raise ConfigError(
            f"write_excel refuses the tab name {sheet!r}: a workbook tab name may be at most "
            f"{MAX_SHEET_NAME_LENGTH} characters. Shorten it."
        )
    for character in FORBIDDEN_SHEET_CHARACTERS:
        if character in sheet:
            raise ConfigError(
                f"write_excel refuses the tab name {sheet!r}: a workbook tab name may not "
                f"contain the character {character}. Take it out of the name."
            )


def validate_format_rules(entry: Mapping[str, Any]) -> None:
    """Refuse a format rule naming both or neither of type and column, and a type or column ruled on twice.

    A repeat is refused, not resolved: picking the first or the last is a guess about what was meant.
    """
    sheet = entry["sheet"]
    seen: dict[tuple[str, str], str] = {}
    for rule in format_rules(entry):
        stated = tuple(field for field in ("type", "column") if rule.get(field) is not None)
        if len(stated) == 2:
            raise ConfigError(
                f"write_excel was given a format rule on tab {sheet!r} carrying both 'type' "
                f"{rule['type']!r} and 'column' {rule['column']!r}; a rule applies to one or the "
                f"other, never both. Write two rules, or drop the one you did not mean."
            )
        if not stated:
            raise ConfigError(
                f"write_excel was given a format rule on tab {sheet!r} carrying neither 'type' "
                f"nor 'column', so the format {rule['format']!r} has nothing to apply to. Give "
                f"exactly one of them."
            )
        field = stated[0]
        key = (field, rule[field])
        if key in seen:
            raise ConfigError(
                f"write_excel was told twice how to show {field} {rule[field]!r} on tab "
                f"{sheet!r}, as {seen[key]!r} and as {rule['format']!r}; one of the two would be "
                f"delivered and which one is not a reading of this list to be relied on. Keep the "
                f"rule you meant and take the other out."
            )
        seen[key] = rule["format"]


def validate_formatted_columns(entry: Mapping[str, Any], frame: pl.DataFrame) -> None:
    """Refuse a format rule naming a column the frame does not have, rather than skipping it silently."""
    for rule in format_rules(entry):
        column = rule.get("column")
        if column is not None and column not in frame.columns:
            raise ConfigError(
                f"write_excel was told to format the column {column!r} on tab "
                f"{entry['sheet']!r}, which port {entry['port']!r} does not carry; it carries: "
                f"{', '.join(frame.columns)}"
            )


def validate_layout(entry: Mapping[str, Any]) -> None:
    """Refuse a layout setting that is ambiguous or that a spreadsheet would not accept.

    Width and wrap rules follow the format rules: one of type or column, none repeated. Widths, heading
    sizes, table styles and table names are checked against the spreadsheet's limits.
    """
    sheet = entry["sheet"]
    _validate_layout_rules(sheet, "widths", entry.get("widths") or ())
    _validate_layout_rules(sheet, "wrap", entry.get("wrap") or ())

    for rule in entry.get("widths") or ():
        width = rule["width"]
        if not 0 < width <= _MAX_WIDTH:
            raise ConfigError(
                f"write_excel was given a column width of {width!r} in 'widths' on tab "
                f"{sheet!r}; a width must be more than 0 and at most {_MAX_WIDTH:g} characters."
            )

    header = entry.get("header")
    if header is not None:
        if header.get("bold") is None and header.get("size") is None:
            raise ConfigError(
                f"write_excel was given a 'header' on tab {sheet!r} that states nothing; give "
                f"'bold', 'size' or both, or leave 'header' out."
            )
        size = header.get("size")
        if size is not None and not 1 <= size <= _MAX_FONT_SIZE:
            raise ConfigError(
                f"write_excel was given a heading 'size' of {size!r} on tab {sheet!r}; a font "
                f"size must be at least 1 and at most {_MAX_FONT_SIZE:g} points."
            )

    table = entry.get("table")
    if table is None:
        return
    style = table.get("style")
    if style is not None and style not in _TABLE_STYLES:
        raise ConfigError(
            f"write_excel was given the table 'style' {style!r} on tab {sheet!r}, which is not "
            f"one of the spreadsheet's built-in table styles. Name one such as 'TableStyleLight1' "
            f"or 'TableStyleMedium2', or leave 'style' out for a plain table."
        )
    name = table.get("name")
    if name is not None and not _is_table_name(name):
        raise ConfigError(
            f"write_excel cannot use the table 'name' {name!r} on tab {sheet!r}. A table name "
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
                f"write_excel was given a rule in '{field}' on tab {sheet!r} carrying {which} "
                f"'type' and 'column'; a rule applies to exactly one of them."
            )
        kind = stated[0]
        key = (kind, rule[kind])
        if key in seen:
            raise ConfigError(
                f"write_excel was given two rules in '{field}' on tab {sheet!r} about {kind} "
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


def table_name(entry: Mapping[str, Any]) -> str:
    """The table name for a sheet entry: the one given, or one made from the tab name.

    Characters a table name cannot hold become underscores. A name that would start with a digit or a
    full stop, or read as a cell address, gets an underscore in front.
    """
    stated = entry["table"].get("name")
    if stated is not None:
        return stated
    name = _NOT_IN_A_TABLE_NAME.sub("_", entry["sheet"])
    if not _is_table_name(name):
        name = f"_{name}"
    return name


def validate_table_names(entries: Sequence[Mapping[str, Any]]) -> None:
    """Refuse two tables of one name. A spreadsheet compares table names ignoring case."""
    seen: dict[str, tuple[str, str]] = {}
    for entry in entries:
        if entry.get("table") is None:
            continue
        name = table_name(entry)
        earlier = seen.get(name.lower())
        if earlier is not None:
            raise ConfigError(
                f"write_excel would make two tables of one name: {earlier[0]!r} on tab "
                f"{earlier[1]!r} and {name!r} on tab {entry['sheet']!r}. A spreadsheet compares "
                f"table names ignoring case. Give one of them its own 'name'."
            )
        seen[name.lower()] = (name, entry["sheet"])


def validate_layout_columns(entry: Mapping[str, Any], frame: pl.DataFrame) -> None:
    """Refuse a width or wrap rule naming a missing column, and a table over two headings that differ only by case."""
    for field in ("widths", "wrap"):
        for rule in entry.get(field) or ():
            column = rule.get("column")
            if column is not None and column not in frame.columns:
                raise ConfigError(
                    f"write_excel was given a rule in '{field}' about the column {column!r} on "
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
                f"write_excel cannot make a table on tab {entry['sheet']!r} over the headings "
                f"{earlier!r} and {column!r}: a table's headings must differ ignoring case. Rename "
                f"one of them upstream, or leave 'table' out."
            )
        seen[column.lower()] = column


def names_the_same_file(template: str, path: str) -> bool:
    """Whether the template and the output path are the same file.

    Two checks, either enough to refuse. First the resolved paths are compared, which catches `..`,
    relative paths and symlinks, and works when `path` does not exist yet. Then, when both exist, the
    filesystem is asked (`samefile`), which catches hard links.

    Why: docs/design/tools.md#write_excel
    """
    if Path(template).resolve() == Path(path).resolve():
        return True
    try:
        return Path(template).samefile(path)
    except OSError:
        return False
