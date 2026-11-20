"""Reading the scanned tree back, the way the author wrote it.

The scan replaced every column name and quoted value with a numbered call. This module turns those
calls back into the name or value they stand for, says whether a node is a value written in the
file, and spells any part of the expression back for a refusal that has to point at it.
"""

import ast
from datetime import date
from typing import Any, Final

from .errors import error
from .quoting import quote_text
from .scan import COLUMN_FN, LITERAL_FN, Scanned

KEYWORD_LITERALS: Final[dict[str, bool | None]] = {"true": True, "false": False, "null": None}

# The author's spelling of each operator, for a refusal that has to quote back the part of the line
# that is wrong rather than the whole line.
OPERATOR_SYMBOLS: Final[dict[type[ast.AST], str]] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.And: "and",
    ast.Or: "or",
}

MISSING: Final[Any] = object()


def render_value(value: Any) -> str:
    """A literal spelled the way the author spelled it, for quoting back in a refusal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return quote_text(value)
    if isinstance(value, date):
        return f"date'{value.isoformat()}'"
    return repr(value)


class Spelling:
    """One expression's scanned tables, and the author's own spelling of any part of it."""

    __slots__ = ("_columns", "_literals", "_text")

    def __init__(self, text: str, scanned: Scanned) -> None:
        self._text = text
        self._columns = scanned.columns
        self._literals = scanned.literals

    def render(self, node: ast.expr) -> str:
        """The author's own spelling of one part of the expression, for a refusal that points at it.

        Rebuilt from the scanned tree, since only the scan knows which index is which name or value.
        """
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == COLUMN_FN:
                return f"[{self.column_name(node)}]"
            if name == LITERAL_FN:
                return render_value(self.interned(node))
            return f"{name}({', '.join(self.render(argument) for argument in node.args)})"
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp):
            rendered = self.render(node.operand)
            return f"not {rendered}" if isinstance(node.op, ast.Not) else f"-{rendered}"
        if isinstance(node, ast.BinOp):
            symbol = OPERATOR_SYMBOLS.get(type(node.op), "?")
            return f"{self.render(node.left)} {symbol} {self.render(node.right)}"
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            symbol = OPERATOR_SYMBOLS.get(type(node.ops[0]), "?")
            return f"{self.render(node.left)} {symbol} {self.render(node.comparators[0])}"
        if isinstance(node, ast.BoolOp):
            joiner = f" {OPERATOR_SYMBOLS.get(type(node.op), '?')} "
            return joiner.join(self.render(value) for value in node.values)
        if isinstance(node, ast.Tuple):
            return f"({', '.join(self.render(element) for element in node.elts)})"
        return "that value"  # pragma: no cover - every whitelisted shape is rendered above

    def column_name(self, node: ast.Call) -> str:
        index = self._table_index(node, len(self._columns))
        return self._columns[index]

    def interned(self, node: ast.Call) -> Any:
        index = self._table_index(node, len(self._literals))
        return self._literals[index]

    def _table_index(self, node: ast.Call, size: int) -> int:
        """The index carried by one of the scan's own calls; fails closed, since an author cannot write these."""
        if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            index = node.args[0].value
            if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < size:
                return index
        raise error(self._text, "the expression could not be understood")  # pragma: no cover

    def literal(self, node: ast.expr) -> Any:
        """The value of `node` if it is a literal, else MISSING, so each caller can word its own refusal."""
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or value is None:
                return MISSING
            return value if isinstance(value, (int, float)) else MISSING
        if isinstance(node, ast.Name) and node.id in KEYWORD_LITERALS:
            return KEYWORD_LITERALS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self.literal(node.operand)
            return -inner if isinstance(inner, (int, float)) and not isinstance(inner, bool) else MISSING
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == LITERAL_FN:
            return self.interned(node)
        return MISSING
