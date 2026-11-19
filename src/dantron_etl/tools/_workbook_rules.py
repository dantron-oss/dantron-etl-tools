"""write_excel: which columns a type-or-column rule reaches.

Formats, widths and wraps all name their targets the same way: a type, or one column. This module
answers which columns each list of rules reaches, and what a column is shown under before any rule
is read. Part of ``write_excel``; the leading underscore keeps it out of the tool catalog.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..column_types import DECLARED_DTYPE, DECLARED_TYPES

_DATE_FORMAT: Final[str] = "yyyy-mm-dd"
_DATETIME_FORMAT: Final[str] = "yyyy-mm-dd hh:mm:ss"

#: Each dtype a frame can carry, as the one name a rule may call it. Built from the shared
#: vocabulary rather than written out again, so a type this library learns to name is a type a
#: format rule can speak about on the same day. A whole number of any width answers to Int64 (see
#: ``_declared_name``). Any other dtype missing from it -- a Float32, say -- is named by no rule.
_DECLARED_NAME: Final[Mapping[Any, str]] = {DECLARED_DTYPE[name].base_type(): name for name in DECLARED_TYPES}


def format_rules(entry: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The format rules of one sheet entry, in the order they were written.

    An entry that states none reads the same as one that states an empty list: no rules, so nothing
    is formatted beyond the date format this tool writes on its own. A default is never filled
    inside an array's element, so the field is read with ``get`` rather than by subscript.
    """
    return tuple(entry.get("formats") or ())


def resolve_rules(
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
        declared = _declared_name(dtype)
        if declared in by_type:
            reached[name] = by_type[declared]
    return reached


def _declared_name(dtype: pl.DataType) -> str | None:
    """The name a rule calls ``dtype`` by. Every width of whole number is Int64.

    A formula makes a whole-number literal 32-bit, so a rule about Int64 that skipped Int32 would
    miss a column like ``flag = 1``. Only how the cell is shown depends on this, never its value.
    """
    if dtype.is_integer():
        return "Int64"
    return _DECLARED_NAME.get(dtype.base_type())


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

    Deliberately about a schema and a list of rules, naming neither a sheet nor a workbook, so that
    another writer can call it too.
    """
    formats: dict[str, str | None] = {name: _default_format(dtype) for name, dtype in schema.items()}
    formats.update(resolve_rules(schema, rules, "format"))
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
