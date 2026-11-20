"""The expression language: text a person writes, such as `[amount] > 0`, in; a polars expression out.

The text is scanned, parsed with `ast`, checked against a whitelist of tree nodes and functions, and
only then compiled. Nothing in it is ever executed. Mistakes are refused here, in the author's words.

The modules, each depending only on those below it: `compiler` (tree to expression), `functions`
(every function, defined once), `arguments` (checking a function's arguments), `spelling` (quoting
the author's text back), `parse` (text to checked tree; the limits and node whitelist), `scan`
(rewriting the author's syntax first), `families` (what a value holds), `formatting` (format_number's
writer), `quoting` (spelling a value as text of the language), `errors` (`ExpressionError`).

Why: docs/design/expressions.md
"""

from collections.abc import Collection, Mapping

import polars as pl

from .compiler import Division, RowCheck, build
from .compiler import divisions as _divisions
from .compiler import row_checks as _row_checks
from .errors import ExpressionError
from .functions import ALLOWED_FUNCTIONS
from .parse import ALLOWED_NODES, MAX_DEPTH, MAX_NODES, MAX_TEXT_LENGTH
from .quoting import escape_text, quote_text

__all__ = [
    "ALLOWED_FUNCTIONS",
    "ALLOWED_NODES",
    "MAX_DEPTH",
    "MAX_NODES",
    "MAX_TEXT_LENGTH",
    "Division",
    "ExpressionError",
    "RowCheck",
    "compile_expr",
    "divisions",
    "escape_text",
    "quote_text",
    "references",
    "row_checks",
]


def compile_expr(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None = None) -> pl.Expr:
    """Compile author-written `text` into a polars expression.

    `columns` decides how much is checked before a row is read: a schema (name to dtype) checks that each
    column exists and that each operation suits it; a collection of names checks existence only; None
    checks neither, for a caller with no frame. An empty schema is a schema with no columns.

    With a schema, a Categorical or Enum column is cast to String where it is read, so the result is a
    plain String column and will not pair with the uncast column as a join key.
    """
    expression, _ = build(text, columns)
    return expression


def divisions(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None = None) -> tuple[Division, ...]:
    """Every '/' in `text`, in order: the divisor as written, and its column.

    Tools use this to stop on a row whose divisor is zero (`refuse_division_by_zero`). The text is checked
    exactly as `compile_expr` checks it.
    """
    return _divisions(text, columns)


def row_checks(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None = None) -> tuple[RowCheck, ...]:
    """Every function call in `text` with rows it has no answer for, such as text parse_date() cannot read.

    Tools use this to stop on such a row (`refuse_division_by_zero` checks these too). The text is checked
    exactly as `compile_expr` checks it.
    """
    return _row_checks(text, columns)


def references(text: str) -> frozenset[str]:
    """The column names `text` reads, for a caller checking them before a run.

    Runs the same checks as `compile_expr` and raises on anything invalid, so a check and a compile can
    never disagree.
    """
    _, names = build(text, None)
    return names
