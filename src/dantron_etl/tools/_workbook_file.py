"""write_excel: opening the one workbook, and making room for each tab in it.

The workbook is loaded from the template, or started new, and it is opened exactly once. The file
already at ``path`` is never read. The refusals that can only be answered with it open live here too: a tab
already there with no permission to replace it, and a table name the workbook already uses. Part of
``write_excel``; the leading underscore keeps it out of the tool catalog.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ..authoring import ConfigError
from ._workbook_checks import table_name


@contextmanager
def opened_once(path: str, template: str | None) -> Iterator[Any]:
    """The one workbook every tab is written into, opened once and closed however the run ends.

    Two ways in. A template is loaded, and saved to ``path``, so the template survives the run.
    Otherwise the book is new. The file already at ``path`` is never read: each run writes the same
    file whatever an earlier run left there.

    ``openpyxl`` is imported here rather than at the top of the module, and named if it is missing,
    the way this library reaches for a database driver only when a database is asked for. Every tool
    module is imported when the catalog is, so a top-level import would cost every consumer,
    including the many tools that never open a workbook.
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
            f"write_excel refuses to overwrite the tab {sheet!r}, which is already in the "
            f"template; set overwrite: true to overwrite it deliberately, or write to a tab of "
            f"another name"
        )

    where = book.sheetnames.index(sheet)
    book.remove(book[sheet])
    return book.create_sheet(sheet, where)


def refuse_names_in_use(book: Any, entries: Sequence[Mapping[str, Any]]) -> None:
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
        name = table_name(entry)
        clash = in_use.get(name.lower())
        if clash is not None:
            raise ConfigError(
                f"write_excel cannot name the table on tab {entry['sheet']!r} {name!r}: the "
                f"workbook already has {clash}, and a spreadsheet compares these names ignoring "
                f"case. Give the table another 'name'."
            )
