"""write_excel: which columns a type-or-column rule applies to.

Formats, widths and wrapping all name their targets the same way, by a type or by one column. This
module works out which columns each list of rules reaches, and the format a column starts with.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..column_types import DECLARED_DTYPE

_DATE_FORMAT: Final[str] = "yyyy-mm-dd"
_DATETIME_FORMAT: Final[str] = "yyyy-mm-dd hh:mm:ss"

#: Each dtype, by the type name a rule uses for it. A dtype not listed (Float32, say) matches no rule.
_DECLARED_NAME: Final[Mapping[Any, str]] = {dtype.base_type(): name for name, dtype in DECLARED_DTYPE.items()}


def format_rules(entry: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The format rules of one sheet entry, in the order written; none if the entry states none."""
    return tuple(entry.get("formats") or ())


def resolve_rules(
    schema: Mapping[str, pl.DataType],
    rules: Sequence[Mapping[str, Any]],
    field: str | None = None,
) -> dict[str, Any]:
    """Which columns a list of type-or-column rules reaches, and what each one gets.

    A column rule beats a type rule, whichever is written first. Each reached column gets the rule's
    `field` (its format or width); with no `field`, as for wrapping, it gets True. Columns no rule
    reaches are left out.
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
    """The type name a rule uses for `dtype`. Every whole-number width counts as Int64.

    A formula makes a whole-number literal 32-bit, so an Int64 rule must also reach Int32 columns.
    """
    if dtype.is_integer():
        return "Int64"
    return _DECLARED_NAME.get(dtype.base_type())


def column_formats(
    schema: Mapping[str, pl.DataType],
    rules: Sequence[Mapping[str, Any]],
) -> dict[str, str | None]:
    """The format code each column is shown under, or None for the workbook's default.

    Dates and timestamps start with a date format; a rule about their type replaces it. A column rule
    beats a type rule. A type rule reaches every column of that type, including columns named by the
    data. Nothing is decided from a column's name. It takes a schema, not a sheet, so another writer
    can use it.
    """
    formats: dict[str, str | None] = {name: _default_format(dtype) for name, dtype in schema.items()}
    formats.update(resolve_rules(schema, rules, "format"))
    return formats


def _default_format(dtype: pl.DataType) -> str | None:
    """The format a column of `dtype` starts with: a date format for dates and timestamps, otherwise none."""
    if isinstance(dtype, pl.Date):
        return _DATE_FORMAT
    if isinstance(dtype, pl.Datetime):
        return _DATETIME_FORMAT
    return None
