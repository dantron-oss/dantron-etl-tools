"""write_excel: open the one workbook, and make room for each tab in it.

The refusals that need the workbook open are here: a template tab named without `overwrite`, and a
table name the workbook already uses.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ..authoring import ConfigError
from ._workbook_checks import table_name


@contextmanager
def opened_once(path: str, template: str | None) -> Iterator[Any]:
    """Open the workbook every tab is written into: the template, or a new one. Always closed.

    The file already at `path` is never read. openpyxl is imported here, and named if missing, so
    importing the package does not load it.
    """
    try:
        import openpyxl
    except ImportError as error:
        raise ConfigError(
            "write_excel needs openpyxl, which this environment does not have installed; "
            "install 'openpyxl' to write a workbook"
        ) from error

    if template is not None:
        book = openpyxl.load_workbook(template)
    else:
        book = openpyxl.Workbook()
        # A new workbook arrives with one empty tab nobody asked for.
        book.remove(book.active)

    try:
        yield book
    finally:
        book.close()


def empty_tab(book: Any, sheet: str, *, replace: bool) -> Any:
    """An empty tab named `sheet`, ready to write; or a refusal if it exists and `replace` is false.

    A replaced tab is remade in the same position, so it does not move to the end of the workbook.
    """
    if sheet not in book.sheetnames:
        return book.create_sheet(sheet)

    if not replace:
        raise ConfigError(
            f"write_excel refuses to overwrite the tab {sheet!r}, which is already in the "
            f"template; set overwrite: true to overwrite it deliberately, or write to a tab of "
            f"another name"
        )

    where = book.sheetnames.index(sheet)
    book.remove(book[sheet])
    return book.create_sheet(sheet, where)


def refuse_names_in_use(book: Any, entries: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a table name already used on a tab this step leaves alone, or as a defined name.

    A tab this step writes is remade, so its old table names are free again.
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
        name = table_name(entry)
        clash = in_use.get(name.lower())
        if clash is not None:
            raise ConfigError(
                f"write_excel cannot name the table on tab {entry['sheet']!r} {name!r}: the "
                f"workbook already has {clash}, and a spreadsheet compares these names ignoring "
                f"case. Give the table another 'name'."
            )
