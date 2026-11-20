"""Text in, checked tree out.

This module holds the size limits and the whitelist of tree nodes. It scans the text, parses the
result with ``ast`` in eval mode, and refuses anything too long, too deep, too wide or outside the
whitelist, naming what it found. Nothing is ever executed.
"""

import ast
import re
from collections.abc import Collection, Mapping
from typing import Final

import polars as pl

from .errors import ExpressionError, error
from .scan import Scanned, scan


# Public size limits: an author can see them. Why: docs/design/expressions.md#limits-come-before-parsing
MAX_TEXT_LENGTH: Final[int] = 2_000
MAX_DEPTH: Final[int] = 40
MAX_NODES: Final[int] = 500

# The whitelist. Expression and Load are housekeeping; everything not listed is refused.
ALLOWED_NODES: Final[frozenset[type[ast.AST]]] = frozenset(
    {
        ast.Expression,
        ast.Load,
        # shapes
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.Constant,
        ast.Name,
        ast.Tuple,
        # boolean, unary and arithmetic operators
        ast.And,
        ast.Or,
        ast.Not,
        ast.USub,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        # comparisons
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
    }
)

# An operator refusal gives both the characters typed and the node type.
_OPERATOR_SPELLINGS: Final[dict[str, str]] = {
    "Pow": "**",
    "Mod": "%",
    "FloorDiv": "//",
    "MatMult": "@",
    "BitAnd": "&",
    "BitOr": "|",
    "BitXor": "^",
    "LShift": "<<",
    "RShift": ">>",
    "Invert": "~",
    "UAdd": "+",
}


# Words from SQL, Excel or Python, and what to write instead; offered only after the parser refuses.
_WORD_HINTS: Final[dict[str, str]] = {
    **{
        word: f"write {word.lower()!r} in lower case"
        for word in ("AND", "OR", "NOT", "IN", "IS", "NULL", "TRUE", "FALSE", "And", "Or", "Not", "Null", "True", "False")
    },
    "like": "this language has no 'like': use starts_with(), ends_with(), contains() or regex_match()",
    "LIKE": "this language has no 'like': use starts_with(), ends_with(), contains() or regex_match()",
    "if": "this language has no if(): use the case_when tool to choose a value by condition",
    "IF": "this language has no if(): use the case_when tool to choose a value by condition",
    "iif": "this language has no iif(): use the case_when tool to choose a value by condition",
    "IIF": "this language has no iif(): use the case_when tool to choose a value by condition",
}

_SYMBOL_HINTS: Final[dict[str, str]] = {
    "<>": "write '!=' for 'not equal'",
    "&&": "write 'and' instead of '&&'",
    "||": "write 'or' instead of '||'",
}


def _parse(text: str, scanned: Scanned) -> ast.Expression:
    """Build a tree from the rewritten text, turning every parser failure into one refusal.

    A syntax error points at the author's text, with a hint for words from other languages (`AND`, `<>`).
    `RecursionError` and `MemoryError` are caught too, since the parser is what fails on deep nesting.
    """
    try:
        return ast.parse(scanned.source, mode="eval")
    except SyntaxError as failure:
        where = _where(text, scanned, failure)
        hint = _hint(text)
        raise error(
            text,
            f"the expression could not be understood: it is not valid syntax for this language{where}{hint}",
        ) from None
    except (RecursionError, MemoryError, ValueError):
        raise error(text, f"the expression is nested too deeply to analyse, past the limit of {MAX_DEPTH}") from None


def _where(text: str, scanned: Scanned, failure: SyntaxError) -> str:
    """`` near character N (...)``: where the parser stopped, in the author's own text."""
    if failure.lineno is None or not failure.offset or not scanned.origins:
        # The parser gives no place when the text simply stops: 'a and', 'a +', 'a ='.
        return ": it ends before the expression is complete"
    lines = scanned.source.split("\n")
    line = min(max(failure.lineno, 1), len(lines))
    at = sum(len(earlier) + 1 for earlier in lines[: line - 1]) + failure.offset - 1
    if at >= len(scanned.origins):
        return ": it ends before the expression is complete"
    start = scanned.origins[at]
    if "never closed" in (failure.msg or ""):
        return f": the '(' at character {start + 1} is never closed"
    shown = text[start : start + 20].strip()
    return f" near character {start + 1}, at `{shown}`" if shown else f" near character {start + 1}"


def _hint(text: str) -> str:
    """A hint for the first spelling from another language found outside quotes and brackets."""
    plain = _outside_quotes_and_brackets(text)
    for symbol, hint in _SYMBOL_HINTS.items():
        if symbol in plain:
            return f"; {hint}"
    for word in re.findall(r"[A-Za-z_]+", plain):
        hint = _WORD_HINTS.get(word)
        if hint is not None:
            return f"; {hint}"
    return ""


def _outside_quotes_and_brackets(text: str) -> str:
    """``text`` with quoted text and bracketed column names blanked out, so a hint never looks inside."""
    kept: list[str] = []
    inside_quote = inside_bracket = False
    for char in text:
        if inside_bracket:
            inside_bracket = char != "]"
            kept.append(" ")
        elif char == "'":
            inside_quote = not inside_quote
            kept.append(" ")
        elif inside_quote:
            kept.append(" ")
        elif char == "[":
            inside_bracket = True
            kept.append(" ")
        else:
            kept.append(char)
    return "".join(kept)


def _check_depth(text: str, tree: ast.Expression) -> None:
    """Refuse a tree nested past MAX_DEPTH, walking it with an explicit stack so the check cannot itself overflow."""
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise error(text, f"the expression is nested more than {MAX_DEPTH} levels deep")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _check_breadth(text: str, tree: ast.Expression) -> None:
    """Refuse a tree with more parts than MAX_NODES, which a flat list can reach inside the depth limit."""
    parts = sum(1 for _ in ast.walk(tree))
    if parts > MAX_NODES:
        raise error(
            text,
            f"the expression has {parts} parts, which is over the {MAX_NODES} part limit. Each quoted value in "
            f"a list after 'in' costs four parts, so such a list can hold about {MAX_NODES // 4 - 5} quoted "
            f"values; match a longer list with a join against a table of the values instead",
        )


def _check_nodes(text: str, tree: ast.Expression) -> None:
    """Refuse any node type outside ALLOWED_NODES, naming the outermost one found."""
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            raise node_refusal(text, node)


def node_refusal(text: str, node: ast.AST) -> ExpressionError:
    """The refusal for one disallowed node, as specific as the node type allows."""
    name = type(node).__name__

    if isinstance(node, ast.Attribute):
        return error(
            text,
            f"attribute access is not part of this language: '.{node.attr}' ({name}) cannot be reached from here",
        )
    if isinstance(node, ast.keyword):
        spelled = f"'{node.arg}='" if node.arg is not None else "'**'"
        return error(
            text,
            f"a keyword argument is not part of this language: {spelled} (keyword) cannot be used, because every "
            f"function here takes a fixed number of positional arguments",
        )
    spelling = _OPERATOR_SPELLINGS.get(name)
    if spelling is not None:
        return error(text, f"the operator '{spelling}' ({name}) is not part of this language")
    return error(text, f"{name} is not part of this language")


def front_door(
    text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None
) -> tuple[ast.Expression, Scanned, dict[str, pl.DataType] | None]:
    """Gate, scan, parse, refuse: everything that happens before a tree is compiled, in one place."""
    if not text or not text.strip():
        raise error(text, "the expression is empty")
    # In front of the parser, deliberately: see the package docstring.
    if len(text) > MAX_TEXT_LENGTH:
        raise error(
            text,
            f"the expression is {len(text)} characters long, which is over the {MAX_TEXT_LENGTH} character limit",
        )

    scanned = scan(text)
    if columns is not None:
        _check_columns(text, scanned.columns, columns)

    tree = _parse(text, scanned)
    _check_depth(text, tree)
    _check_breadth(text, tree)
    _check_nodes(text, tree)

    return tree, scanned, dict(columns) if isinstance(columns, Mapping) else None


def _check_columns(text: str, referenced: tuple[str, ...], columns: Collection[str]) -> None:
    """Refuse a column the frame does not have, naming it and listing what is there, first mistake first."""
    available = set(columns)
    for name in referenced:
        if name not in available:
            raise error(
                text,
                f"[{name}] is not a column in the data: the columns are {', '.join(sorted(available))}",
            )
