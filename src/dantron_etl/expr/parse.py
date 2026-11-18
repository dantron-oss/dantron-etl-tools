"""Text in, checked tree out.

This module holds the size limits and the whitelist of tree nodes. It scans the text, parses the
result with ``ast`` in eval mode, and refuses anything too long, too deep, too wide or outside the
whitelist, naming what it found. Nothing is ever executed.
"""

import ast
from collections.abc import Collection, Mapping
from typing import Final

import polars as pl

from .errors import ExpressionError, error
from .scan import Scanned, scan


# The size limits are published rather than private because they are part of the contract with
# whoever writes an expression: a limit an author cannot see is a limit they cannot design around.
# All three are generous next to any real predicate -- the widest honest one is a membership test
# against a list of codes -- and they exist to bound work on untrusted text, not to shape the
# language. MAX_NODES has to sit below half of MAX_TEXT_LENGTH to mean anything at all: roughly two
# characters buy one node, so a breadth cap above that could never be reached.
MAX_TEXT_LENGTH: Final[int] = 2_000
MAX_DEPTH: Final[int] = 40
MAX_NODES: Final[int] = 500

# The whitelist. ``Expression`` and ``Load`` are housekeeping: the root of an eval tree, and the
# context flag every name carries. Everything else here is either a shape the language uses or an
# operator it exposes -- and everything absent is refused, including every form of attribute
# access, subscripting, comprehension, binding, unpacking and string formatting.
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

# An operator refusal has to be readable by an author, who typed characters, and precise for
# whoever reads the log, who wants the node type. So a refusal carries both spellings.
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


def _parse(text: str, source: str) -> ast.Expression:
    """Build a tree from the rewritten text, mapping every parser failure onto one refusal.

    ``RecursionError`` and ``MemoryError`` are caught alongside the syntax errors because the
    parser is the component that fails on pathological nesting, and a caller of this module should
    never have to catch an error type that leaks how the text was analysed.
    """
    try:
        return ast.parse(source, mode="eval")
    except SyntaxError:
        raise error(text, "the expression could not be understood: it is not valid syntax for this language") from None
    except (RecursionError, MemoryError, ValueError):
        raise error(text, f"the expression is nested too deeply to analyse, past the limit of {MAX_DEPTH}") from None


def _check_depth(text: str, tree: ast.Expression) -> None:
    """Refuse a tree nested past :data:`MAX_DEPTH`, walking it with an explicit stack.

    Iterative on purpose: a recursive depth check would itself blow the stack on the input it
    exists to reject, which is exactly the failure it is meant to convert into a clear refusal.
    """
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise error(text, f"the expression is nested more than {MAX_DEPTH} levels deep")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _check_breadth(text: str, tree: ast.Expression) -> None:
    """Refuse a tree with more parts than :data:`MAX_NODES`, naming the limit that fired.

    Depth says nothing about a flat expression, so a list of ten thousand values sits inside the
    depth limit and is bounded only by the character limit -- which then reports a number unrelated
    to what the author wrote. A published breadth cap refuses it by the limit that actually applies.
    """
    parts = sum(1 for _ in ast.walk(tree))
    if parts > MAX_NODES:
        raise error(
            text,
            f"the expression has {parts} parts, which is over the {MAX_NODES} part limit",
        )


def _check_nodes(text: str, tree: ast.Expression) -> None:
    """Refuse any node type outside :data:`ALLOWED_NODES`, naming it.

    A whitelist, so the question a future reader asks is "why is this node allowed?" rather than
    "did anyone think of this node?". ``ast.walk`` is breadth-first and iterative, so the outermost
    offending node in the text is the one reported and no recursion is involved.
    """
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

    tree = _parse(text, scanned.source)
    _check_depth(text, tree)
    _check_breadth(text, tree)
    _check_nodes(text, tree)

    return tree, scanned, dict(columns) if isinstance(columns, Mapping) else None


def _check_columns(text: str, referenced: tuple[str, ...], columns: Collection[str]) -> None:
    """Refuse a column the frame does not have, naming it and listing what is there.

    Checked in the order written, so the first mistake in the line is the one reported. Listing the
    available names is the whole point: an author who misspells a column cannot see the frame, and
    a message that only says "unknown column" leaves them guessing.
    """
    available = set(columns)
    for name in referenced:
        if name not in available:
            raise error(
                text,
                f"[{name}] is not a column in the data: the columns are {', '.join(sorted(available))}",
            )
