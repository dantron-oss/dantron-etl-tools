"""A small expression language: author-written text in, one column expression out.

A tool's configuration is a file on disk. A filter needs a predicate, a derived column needs a
formula, and neither can be a callable, because the person writing the file is not writing code --
they are writing ``[status] = 'PAID' and [amount] > 0``. This module is the one place that turns
such text into a column expression, and it is therefore an execution boundary: every character in
it arrives from outside the program.

So the design is fixed, and it is a whitelist end to end:

1. A forward scan rewrites the author's surface syntax -- ``[name]`` becomes an internal column
   reference, ``=`` becomes ``==`` -- and lifts every quoted literal out of the text entirely, so
   that no author-supplied character survives into the text that gets parsed.
2. ``ast.parse(..., mode="eval")`` builds a tree. Nothing is ever executed: no ``eval``, no
   ``exec``, no ``compile``, no attribute lookup driven by author text, and no hand-off to a
   database engine's own expression parser.
3. Every node in the tree is checked against :data:`ALLOWED_NODES` and anything else is refused
   *by name*, so a node type nobody thought about is refused by default rather than allowed by
   default.
4. Only then are the surviving nodes translated into a column expression, with function calls
   limited to :data:`ALLOWED_FUNCTIONS`.

Four consequences worth stating plainly, because each one is easy to get wrong:

*Length is checked before parsing.* Tree construction is itself the fragile step -- a long enough
operator chain exhausts the parser's own stack -- so the size gate has to sit in front of it, and
the depth gate that follows has to walk the tree with an explicit stack rather than by recursion.
Breadth gets its own published cap for the same reason depth does: a limit an author cannot see is
a limit they cannot design around.

*Null follows the engine, not intuition.* A comparison against a null value is null, not false, and
a filter keeps only rows that are true. Nothing here quietly turns a null into a false; if an
author wants a null row to participate they write ``coalesce([code], 'Z') = 'Z'``, and if they want
to find null rows they write ``[code] is null``. Papering over this would make every predicate
disagree with the engine that stores the data, in the direction of silently including rows.

*A mistake has to fail here, in the author's words, and never later in the engine's.* So a caller
holding the incoming frame passes its schema, and every operation is checked against the type
family its operands actually hold: text, number, date, boolean, with ``null`` belonging to all of
them. Four of those refusals are stricter than the engine, which answers them silently -- and a
predicate that matches nothing while raising nothing is indistinguishable, in a log, from a day on
which nothing happened. A pattern is checked the same way, by asking the engine that will run it
whether it compiles, at compile time rather than at row time.

*A literal has to mean what it looks like.* The parser underneath reads more number spellings than
this language wants: ``0x10`` is sixteen, ``1_000`` carries a separator this language does not have,
``1e400`` is silently infinity and ``1e-400`` is silently zero. All of them are refused, because
each one arrives from an author who was thinking about something else.

*One language, two targets.* The same text compiles to a column expression and, for a caller that
can push a predicate into the database it is reading from, to a boolean SQL clause plus the values
bound into it -- so a read can cut rows at the engine instead of dragging a table into memory to cut
them there. Both targets come through the front door above: one size gate, one scan that lifts every
literal out before the parser sees a character, one node whitelist, one set of type checks. There is
no second parser, and no author-written character reaches a SQL string -- a column becomes a quoted
identifier only by passing :data:`SQL_IDENTIFIER`, and everything else the author wrote is a bound
parameter, where the worst a hostile value can do is match no row.

*What can be pushed down is a subset, and it is published per dialect.* An engine that answers a
function differently from the frame target turns a pushdown into a quietly different predicate, so
every entry in :func:`sql_subset` was measured against the real engine and compared row for row --
and the exclusions differ by engine, which is why one lowest-common-denominator subset would have
been either wrong somewhere or poorer than it needed to be everywhere.
"""

import ast
import enum
import math
import operator
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import date
from decimal import Decimal
from itertools import chain
from typing import Any, Final, NamedTuple

import polars as pl

__all__ = [
    "ALLOWED_FUNCTIONS",
    "ALLOWED_NODES",
    "MAX_DEPTH",
    "MAX_NODES",
    "MAX_TEXT_LENGTH",
    "SQL_DIALECTS",
    "SQL_IDENTIFIER",
    "SQL_PARAMSTYLES",
    "ExpressionError",
    "SqlSubset",
    "compile_expr",
    "compile_sql",
    "escape_text",
    "quote_text",
    "references",
    "sql_subset",
]


class ExpressionError(ValueError):
    """An expression could not be compiled, with a message written for whoever wrote the text.

    A ``ValueError`` subclass because a bad expression is bad input, and because a caller that
    already handles configuration errors as ``ValueError`` keeps working without knowing this type
    exists. Every message names the problem, says what to write instead where there is an obvious
    answer, and quotes the offending expression so a log line points at the line of the file.
    """


# The size limits are published rather than private because they are part of the contract with
# whoever writes an expression: a limit an author cannot see is a limit they cannot design around.
# All three are generous next to any real predicate -- the widest honest one is a membership test
# against a list of codes -- and they exist to bound work on untrusted text, not to shape the
# language. MAX_NODES has to sit below half of MAX_TEXT_LENGTH to mean anything at all: roughly two
# characters buy one node, so a breadth cap above that could never be reached.
MAX_TEXT_LENGTH: Final[int] = 2_000
MAX_DEPTH: Final[int] = 40
MAX_NODES: Final[int] = 500

# Extending this language means adding a name to this set and a branch to the compiler below. It
# never means adding grammar: an author-facing operator has to be spelled, documented, reviewed in
# a file and read by a non-developer, whereas one more named function is free of all of that.
ALLOWED_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "coalesce",
        "round",
        "year",
        "add_months",
        "abs",
        "lower",
        "upper",
        "strip",
        "len",
        "starts_with",
        "ends_with",
        "contains",
        "regex_extract",
        "regex_replace",
        "regex_match",
        "concat",
        "format_number",
    }
)

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

# The published surface of the second target. A dialect is an engine's spelling; a placeholder
# style is a driver's way of binding a value, and the two are separate axes because two drivers for
# one engine do not agree on the second. Both are whitelists for the same reason everything else
# here is: a dialect nobody has run this compiler against would get confident SQL and no warning.
SQL_DIALECTS: Final[frozenset[str]] = frozenset({"sqlite", "postgres"})
SQL_PARAMSTYLES: Final[frozenset[str]] = frozenset({"qmark", "format"})

# The grammar a column name must pass to be written into a clause, and the whole defence at that
# boundary: an identifier is the one thing a driver cannot bind as a parameter. One grammar for
# every dialect, deliberately -- a per-dialect quoting rule is how a name that is safe at one
# engine ends up spliced into a statement for another -- and a name outside it is refused rather
# than escaped, so there is no escaping rule to get wrong and no doubled quote to reason about.
# Stricter than the frame target on purpose: [order status] filters perfectly well in memory.
SQL_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SqlSubset(NamedTuple):
    """What one dialect can be given, as three facts that have to stay together.

    ``functions`` is the language's functions this engine spells the way the frame target answers
    them. ``ordered_families`` is the type families whose ``<`` it agrees with -- text is the one
    that differs between engines, because a byte order and a locale order are not the same order.
    ``paramstyle`` is what that engine's usual driver binds with.
    """

    functions: frozenset[str]
    ordered_families: frozenset[str]
    paramstyle: str


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

# The scan's own output vocabulary. A column reference and a quoted literal are both rewritten to a
# call on one of these names carrying nothing but an integer index into a table the scan built, so
# the text handed to the parser contains no author-supplied characters at all. These names are
# refused in the incoming text (see ``_scan``) precisely so that the only way to produce one is to
# write ``[name]``, ``'text'`` or ``date'...'`` -- the internal spelling is unreachable by hand.
_COLUMN_FN: Final[str] = "col"
_LITERAL_FN: Final[str] = "lit"
_RESERVED_CALLS: Final[frozenset[str]] = frozenset({_COLUMN_FN, _LITERAL_FN, "date"})

_KEYWORD_LITERALS: Final[dict[str, bool | None]] = {"true": True, "false": False, "null": None}

# A quote may be preceded by ``date`` and nothing else. Every other prefix Python would accept is
# named here so the refusal can say what the author reached for.
_QUOTE_PREFIX_HINTS: Final[dict[str, str]] = {
    "f": "an f-string (JoinedStr / FormattedValue) is not part of this language",
    "b": "a bytes literal (Bytes) is not part of this language",
    "r": "a raw-string prefix is not needed, because quoted text is already taken verbatim",
    "u": "a unicode prefix is not supported, because quoted text is already unicode",
}

_BINARY_OPERATORS: Final[dict[type[ast.AST], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_COMPARISONS: Final[dict[type[ast.AST], Callable[[Any, Any], Any]]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# The author's spelling of each operator, for a refusal that has to quote back the part of the line
# that is wrong rather than the whole line.
_OPERATOR_SYMBOLS: Final[dict[type[ast.AST], str]] = {
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

# Arity is fixed, with no defaulted arguments: one spelling per call keeps a configuration file
# reviewable, and a reviewer should not have to know a default to know what a line does.
_FUNCTION_ARITY: Final[dict[str, int]] = {
    "coalesce": -1,  # variadic, at least two
    "round": 2,
    "year": 1,
    "add_months": 2,
    "abs": 1,
    "lower": 1,
    "upper": 1,
    "strip": 1,
    "len": 1,
    "starts_with": 2,
    "ends_with": 2,
    "contains": 2,
    "regex_extract": 3,
    "regex_replace": 3,
    "regex_match": 2,
    "concat": -1,  # variadic, at least two
    "format_number": 4,
}

_NUMBER_FUNCTIONS: Final[frozenset[str]] = frozenset({"abs", "round"})
_PATTERN_FUNCTIONS: Final[frozenset[str]] = frozenset({"regex_extract", "regex_replace", "regex_match"})
_TEXT_PAIR_FUNCTIONS: Final[frozenset[str]] = frozenset({"starts_with", "ends_with", "contains"})

# Two numeric bounds, both about keeping a refusal a refusal. A whole number outside the signed
# 64-bit range has no column that could hold it, and a decimal place or capture group beyond a few
# thousand has no meaning -- and in both cases the engine below rejects the value with an error of
# its own type, which would escape this module and defeat the promise that every bad expression
# raises one thing. The bound is measured on the value the author wrote, sign folded in: the most
# negative whole number is spellable, and negating it twice is not.
_WHOLE_NUMBER_MIN: Final[int] = -(2**63)
_WHOLE_NUMBER_MAX: Final[int] = 2**63 - 1
_SMALL_NUMBER_LIMIT: Final[int] = 65_535

_DIGITS: Final[str] = "0123456789"

_MISSING: Final[Any] = object()
_QUOTED_TEXT_ADVICE: Final[str] = "write text in single quotes, like 'PAID'"

# Why an argument has to be quoted text written in the file. Two wordings, because the two
# arguments that carry this rule worry about different things: one is what the data becomes, the
# other is how the data is spelled.
_REWRITE_REASON: Final[str] = (
    "a rewrite whose replacement came out of the data would let the data decide what the data becomes"
)
_SEPARATOR_REASON: Final[str] = (
    "a separator is a decision written in the file, the way the rest of a report's spelling is, and never a value a "
    "row supplies -- figures that chose their own separators would be deciding how they read"
)


class _Family(enum.Enum):
    """The type families an operation may work within, named as an author would name them.

    Coarse on purpose, and the coarseness is the design. A whole number and a decimal are one
    family because separating them would refuse ``[qty] > 2.5``, which the engine answers perfectly
    well -- and a check stricter than the engine costs an author capability they cannot argue with.
    ``NULL`` belongs to every family, because a null is the one value that may meet anything.
    ``UNKNOWN`` is what a column has when no schema was supplied: unknown means unchecked, never
    refused, so a caller with no frame in hand still gets every other check.
    """

    TEXT = "text"
    NUMBER = "a number"
    DATE = "a date"
    BOOLEAN = "a truth value"
    NULL = "null"
    UNKNOWN = "an unknown type"


# The two families that meet anything: a null value belongs to every family, and an unchecked one
# cannot be argued with. Both are skipped by every requirement below.
_WILDCARD_FAMILIES: Final[frozenset[_Family]] = frozenset({_Family.NULL, _Family.UNKNOWN})

# What each function hands back, so that nesting is checked rather than assumed. Without this,
# ``upper(len([label]))`` compiles and fails in the engine: checking a column reference while
# treating every call's result as unknown leaves exactly that hole. ``coalesce`` is absent because
# its result is whatever family its arguments agree on.
_FUNCTION_RESULTS: Final[dict[str, _Family]] = {
    "round": _Family.NUMBER,
    "year": _Family.NUMBER,
    "abs": _Family.NUMBER,
    "len": _Family.NUMBER,
    "lower": _Family.TEXT,
    "upper": _Family.TEXT,
    "strip": _Family.TEXT,
    "regex_extract": _Family.TEXT,
    "regex_replace": _Family.TEXT,
    "concat": _Family.TEXT,
    "add_months": _Family.DATE,
    "starts_with": _Family.BOOLEAN,
    "ends_with": _Family.BOOLEAN,
    "contains": _Family.BOOLEAN,
    "regex_match": _Family.BOOLEAN,
    "format_number": _Family.TEXT,
}


# What three of those functions hand back exactly, beside the family. year() and len() always give
# a whole number, so a count of months may be built out of either, and format_number() always gives
# a plain string. abs() and round() are not here: they hand back whatever their subject holds,
# which is passed in from where the subject is in view.
_FUNCTION_RESULT_DTYPES: Final[dict[str, pl.DataType]] = {
    "year": pl.Int32(),
    "len": pl.UInt32(),
    "format_number": pl.String(),
}


def compile_expr(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None = None) -> pl.Expr:
    """Compile author-written ``text`` into a column expression.

    ``columns`` says how much can be checked before a row is read, and there are three honest
    answers because there are three kinds of caller:

    * a schema -- name to dtype, as a frame reports it -- checks that every column exists *and*
      that every operation suits what the column holds, so ``upper([qty])`` is refused here instead
      of failing inside the engine later;
    * a plain collection of names checks existence only, because a list of names cannot say
      anything about a dtype and guessing at one would be worse than admitting the gap;
    * ``None`` checks neither, which is what a caller validating a definition against a template
      has to do, since it has no frame at all.

    An empty schema is a schema listing no columns, not the absence of one.

    A schema does one more thing besides checking: two dtypes store text compactly -- a
    dictionary-encoded column, and one restricted to a fixed set of members -- and wherever the
    schema says a column is one of these, the leaf reference to it is cast to a plain string the
    moment it is read, before any operation touches it. Two consequences follow from that, and
    neither is optional. First, the compact encoding does not survive: a result built from such a
    column is a plain string column, not the dictionary-encoded or member-restricted one it came
    from, so it is not carried through to a write. Second, that result no longer shares a dtype with
    a column that never passed through an expression, so the two will not silently pair as a join
    or group-by key -- a caller who needs the original encoding preserved keeps the source column
    alongside the expression's result rather than relying on the two to match.
    """
    expression, _ = _build(text, columns)
    return expression


def compile_sql(
    text: str,
    columns: Mapping[str, pl.DataType] | Collection[str] | None = None,
    *,
    dialect: str,
    paramstyle: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """Compile author-written ``text`` into a boolean SQL clause and the values to bind into it.

    The clause is a self-contained condition with no ``where`` of its own, so whoever is building
    the statement composes it -- a read tool puts its own incremental-sync condition beside the
    author's predicate with an ``and``, and a clause whose top-level ``or`` had lost its grouping
    would silently widen to keep rows the author excluded.

    ``dialect`` is keyword-only and has no default, because there is nothing to compile for an
    engine nobody named: the spelling of a substring test, the placeholder a driver binds, and the
    order an engine puts text in are all the engine's, and a compiler that guessed at them would
    emit confident SQL nobody has run. What each dialect can be given is published by
    :func:`sql_subset`, and anything outside it is refused by name, naming the dialect, and routed
    to a Filter step after the read -- which answers the same question in memory with the frame
    target's exact semantics.

    ``paramstyle`` is the second axis and belongs to the *driver* rather than the engine, because
    two drivers for one engine do not agree on it. It defaults to the one that dialect's usual
    driver publishes, and may only be named from :data:`SQL_PARAMSTYLES`, both of which bind a
    sequence; a style that binds a mapping would take a tuple and mean something else.

    ``columns`` is read exactly as :func:`compile_expr` reads it, and carries one extra duty here:
    a WHERE clause has to be a *condition*, so an expression that produces a value instead --
    ``[amount]`` where the author meant ``[amount] > 0`` -- is refused against the schema rather
    than left to an engine that would answer it by truthiness or raise in a driver's vocabulary.

    The values come back as the author wrote them, adapted by nobody: a date is a
    :class:`datetime.date`, which one driver binds into a date column directly and another needs
    converted to ISO-8601 text. Converting here for one driver would have broken the other.
    """
    subset = sql_subset(dialect)
    if paramstyle is None:
        paramstyle = subset.paramstyle
    elif paramstyle not in SQL_PARAMSTYLES:
        raise _unpublished("placeholder style", paramstyle, SQL_PARAMSTYLES)

    tree, scanned, schema = _front_door(text, columns)
    compiler = _Compiler(text, scanned, schema, _SqlTarget(text, dialect, paramstyle))
    clause: _Fragment = compiler.compile_condition(tree.body)
    return clause.sql, clause.params


def sql_subset(dialect: str) -> SqlSubset:
    """What one dialect can be given: its functions, what it may order, and how it binds values.

    Published for the same reason :data:`ALLOWED_FUNCTIONS` is -- an author cannot design around a
    boundary they cannot see, and a loader cannot tell them in advance which of their expressions
    will reach the database and which will be answered in memory.

    One accessor rather than three lookups keyed by dialect, because three lookups is how three
    facts about one engine drift apart: a dialect added to the function table and forgotten in the
    placeholder table is a mistake that only shows up as a driver error at read time.
    """
    published = _DIALECTS.get(dialect)
    if published is None:
        raise _unpublished("dialect", dialect, SQL_DIALECTS)
    return published.subset


def references(text: str) -> frozenset[str]:
    """The column names ``text`` reads, for a caller that wants to check them before running.

    This runs the *same* validation as :func:`compile_expr`, and raises on anything invalid rather
    than reporting the names it managed to find. A lenient version would let a caller pre-flight an
    expression successfully and then fail on compilation later -- the exact disagreement between
    check and execution that this function exists to prevent.
    """
    _, names = _build(text, None)
    return names


def escape_text(value: str) -> str:
    """``value`` with every quote doubled, ready to sit *inside* a literal this language reads.

    For a caller writing a value into text that already carries the author's own quotes, or the
    language's own ``date'...'`` tag. See :func:`quote_text` for the standalone spelling.
    """
    return value.replace("'", "''")


def quote_text(value: str) -> str:
    """``value`` spelled as a complete text literal of this language: quoted and escaped.

    Published because escaping is the caller's job to hand off, not to reinvent: a value that
    reaches an expression through someone else's string formatting is the one way author-supplied
    characters get into the text this module parses.
    """
    return f"'{escape_text(value)}'"


# ---------------------------------------------------------------------------
# Scanning: the author's syntax rewritten, with every literal lifted out
# ---------------------------------------------------------------------------


class _Scanned:
    """The result of one forward scan: parseable text, plus the tables its indices point into."""

    __slots__ = ("columns", "literals", "source")

    def __init__(self, source: str, columns: tuple[str, ...], literals: tuple[Any, ...]) -> None:
        self.source = source
        self.columns = columns
        self.literals = literals


class _Table:
    """An append-only table of values that returns a stable index, reusing one for a repeat value."""

    __slots__ = ("_index", "values")

    def __init__(self) -> None:
        self.values: list[Any] = []
        self._index: dict[tuple[str, Any], int] = {}

    def intern(self, value: Any) -> int:
        key = (type(value).__name__, value)
        found = self._index.get(key)
        if found is None:
            found = len(self.values)
            self._index[key] = found
            self.values.append(value)
        return found


def _scan(text: str) -> _Scanned:
    """Rewrite the author's syntax in one forward pass, tracking whether we are inside a literal.

    A scan rather than a regular expression, because this is where a rewrite-then-parse design
    goes wrong. ``[note] = 'a=b'`` and ``[note] = 'x [y]'`` both contain characters this function
    rewrites; a blind substitution corrupts the literal, and the corruption then looks like a data
    problem rather than a parser problem. Quoted text is opaque here: it is lifted out into a table
    and replaced by an index, which also means an escape sequence, a stray backslash or a smuggled
    quote cannot change the text that reaches the parser.

    A number is the one author-written value that passes through as characters, so the scan is also
    where a number is held to the spellings this language has -- the parser underneath reads several
    it does not.
    """
    pieces: list[str] = []
    columns = _Table()
    literals = _Table()
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == "'":
            value, index = _read_quoted(text, index)
            pieces.append(f"{_LITERAL_FN}({literals.intern(value)})")
            continue

        if char == '"':
            raise _error(text, f"double quotes are not part of this language, so {_QUOTED_TEXT_ADVICE}")

        if char == "[":
            end = text.find("]", index + 1)
            if end == -1:
                raise _error(text, "a '[' has no matching ']', so the bracket around a column name is unbalanced")
            name = text[index + 1 : end].strip()
            if not name:
                raise _error(text, "'[]' is an empty column reference, so name a column between the brackets")
            pieces.append(f"{_COLUMN_FN}({columns.intern(name)})")
            index = end + 1
            continue

        if char == "]":
            raise _error(text, "a ']' has no matching '[', so the bracket around a column name is unbalanced")

        if char in _DIGITS:
            span, index = _read_number(text, index)
            pieces.append(span)
            continue

        if char.isalpha() or char == "_":
            end = index + 1
            while end < length and (text[end].isalnum() or text[end] == "_"):
                end += 1
            word = text[index:end]

            if end < length and text[end] == "'":
                if word != "date":
                    fallback = f"{_QUOTED_TEXT_ADVICE}, or a date as date'2026-01-01'"
                    hint = _QUOTE_PREFIX_HINTS.get(word.lower(), fallback)
                    raise _error(text, f"an unsupported literal prefix {word!r} appears before quoted text: {hint}")
                raw, index = _read_quoted(text, end)
                pieces.append(f"{_LITERAL_FN}({literals.intern(_parse_date(text, raw))})")
                continue

            if word in _RESERVED_CALLS and _next_visible(text, end) == "(":
                raise _error(
                    text,
                    f"{word!r} is reserved for this language's own use and cannot be called: write a column as "
                    f"[name], text as 'value', and a date as date'2026-01-01'",
                )

            pieces.append(word)
            index = end
            continue

        if char == "=":
            if text[index + 1 : index + 2] == "=":
                raise _error(text, "'==' is not how equality is written here, so use a single '=' instead")
            # ``!=``, ``<=`` and ``>=`` already mean what they say; only a lone '=' is a comparison
            # in this language and has to be doubled for the parser.
            pieces.append("=" if index > 0 and text[index - 1] in "!<>" else "==")
            index += 1
            continue

        if char == "!":
            if text[index + 1 : index + 2] != "=":
                raise _error(text, "'!' on its own is not an operator here, so write '!=' to test for inequality")
            pieces.append("!")
            index += 1
            continue

        pieces.append(char)
        index += 1

    return _Scanned("".join(pieces), tuple(columns.values), tuple(literals.values))


def _read_quoted(text: str, quote_at: int) -> tuple[str, int]:
    """The text a literal holds, and the index just past the quote that closed it.

    A doubled quote inside the literal is one quote, the way SQL and Excel spell it: ``'O''Brian'``
    is the seven characters ``O'Brian``. The doubling is undone here, before the parser underneath
    ever sees the text, because that parser would read two adjacent literals as implicit
    concatenation -- ``O`` followed by ``Brian`` -- and hand back a plausible value nobody wrote.

    A doubled quote is an escape only when the literal continues past it, so ``''`` on its own is
    still the empty string, and a literal whose last quote was spent on an escape is simply one
    that never closes.
    """
    parts: list[str] = []
    index = quote_at + 1
    length = len(text)
    while True:
        end = text.find("'", index)
        if end == -1:
            raise _error(text, "a quote is never closed, so the quoted text has no end")
        parts.append(text[index:end])
        if end + 1 < length and text[end + 1] == "'":
            parts.append("'")
            index = end + 2
            continue
        return "".join(parts), end + 1


def _read_number(text: str, start: int) -> tuple[str, int]:
    """One number, in the only shapes this language has: digits, a decimal point, an exponent.

    Everything the parser underneath would also accept is refused here instead, because each of
    those spellings arrives from an author who meant something else. ``0x10`` is sixteen, written by
    someone thinking about a code that looks like hex; ``1_000`` is a thousand carrying a separator
    this language does not have. Anything touching the end of a number -- a letter, another digit
    group, a second point -- is therefore refused rather than reinterpreted.
    """
    length = len(text)
    index = start
    while index < length and text[index] in _DIGITS:
        index += 1
    if index < length and text[index] == ".":
        index += 1
        while index < length and text[index] in _DIGITS:
            index += 1
    if index < length and text[index] in "eE":
        after = index + 1
        if after < length and text[after] in "+-":
            after += 1
        if after < length and text[after] in _DIGITS:
            index = after
            while index < length and text[index] in _DIGITS:
                index += 1

    span = text[start:index]
    trailing = text[index : index + 1]
    if trailing and (trailing.isalnum() or trailing in "_."):
        # Quote the whole run the author typed, not just the first character that spoiled it: an
        # author who wrote 0x10 is looking for 0x10 in the message, not for '0x'.
        end = index
        while end < len(text) and (text[end].isalnum() or text[end] in "_."):
            end += 1
        raise _error(
            text,
            f"'{text[start:end]}' is not a number this language reads: a whole number is written in digits alone and a "
            f"decimal as 1.5 or 1e-9, so another base (0x10), a digit separator (1_000) or a trailing letter is "
            f"refused rather than reinterpreted",
        )
    if "." in span or "e" in span or "E" in span:
        _check_decimal(text, span)
    return span, index


def _check_decimal(text: str, span: str) -> None:
    """Refuse a decimal that does not survive as the number written.

    Both directions cost money, and both are silent. ``1e400`` arrives as infinity, so
    ``[amount] < 1e400`` keeps every row for an author who typed one character too many. Its mirror
    ``1e-400`` arrives as exactly zero, which turns a tolerance into "anything at all". Exponent
    notation itself stays legal, because ``1e-9`` is how a tolerance is written and the alternative
    spelling is a row of zeros a reviewer has to count.
    """
    value = float(span)
    if not math.isfinite(value):
        raise _error(
            text,
            f"the decimal {span} is too large to hold, so it would arrive as infinity: a comparison against it would "
            f"match every row",
        )
    mantissa = span.split("e")[0].split("E")[0]
    if value == 0.0 and float(mantissa) != 0.0:
        raise _error(
            text,
            f"the decimal {span} is too small to hold, so it would arrive as 0.0: a tolerance written this way would "
            f"accept anything at all",
        )


def _parse_date(text: str, raw: str) -> date:
    """A calendar date from ``date'YYYY-MM-DD'``.

    The prefix-tagged form is deliberate. A postfix cast (``'2026-01-01'::date``) would introduce a
    cast operator, and the obvious next request after one cast is every other cast -- growing the
    grammar, which this design refuses to do. Tagging keeps a date a literal.
    """
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise _error(text, f"date'{raw}' is not a calendar date, so write a date as date'2026-01-01'") from None


def _next_visible(text: str, index: int) -> str:
    """The next character from ``index`` that is not whitespace, or ``''`` at the end of the text."""
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index : index + 1]


# ---------------------------------------------------------------------------
# Parsing and refusal
# ---------------------------------------------------------------------------


def _parse(text: str, source: str) -> ast.Expression:
    """Build a tree from the rewritten text, mapping every parser failure onto one refusal.

    ``RecursionError`` and ``MemoryError`` are caught alongside the syntax errors because the
    parser is the component that fails on pathological nesting, and a caller of this module should
    never have to catch an error type that leaks how the text was analysed.
    """
    try:
        return ast.parse(source, mode="eval")
    except SyntaxError:
        raise _error(text, "the expression could not be understood: it is not valid syntax for this language") from None
    except (RecursionError, MemoryError, ValueError):
        raise _error(text, f"the expression is nested too deeply to analyse, past the limit of {MAX_DEPTH}") from None


def _check_depth(text: str, tree: ast.Expression) -> None:
    """Refuse a tree nested past :data:`MAX_DEPTH`, walking it with an explicit stack.

    Iterative on purpose: a recursive depth check would itself blow the stack on the input it
    exists to reject, which is exactly the failure it is meant to convert into a clear refusal.
    """
    stack: list[tuple[ast.AST, int]] = [(tree, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_DEPTH:
            raise _error(text, f"the expression is nested more than {MAX_DEPTH} levels deep")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _check_breadth(text: str, tree: ast.Expression) -> None:
    """Refuse a tree with more parts than :data:`MAX_NODES`, naming the limit that fired.

    Depth says nothing about a flat expression, so a list of ten thousand values sits inside the
    depth limit and is bounded only by the character limit -- which then reports a number unrelated
    to what the author wrote. A published breadth cap refuses it by the limit that actually applies.
    """
    parts = sum(1 for _ in ast.walk(tree))
    if parts > MAX_NODES:
        raise _error(
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
            raise _refusal(text, node)


def _refusal(text: str, node: ast.AST) -> ExpressionError:
    """The refusal for one disallowed node, as specific as the node type allows."""
    name = type(node).__name__

    if isinstance(node, ast.Attribute):
        return _error(
            text,
            f"attribute access is not part of this language: '.{node.attr}' ({name}) cannot be reached from here",
        )
    if isinstance(node, ast.keyword):
        spelled = f"'{node.arg}='" if node.arg is not None else "'**'"
        return _error(
            text,
            f"a keyword argument is not part of this language: {spelled} (keyword) cannot be used, because every "
            f"function here takes a fixed number of positional arguments",
        )
    spelling = _OPERATOR_SPELLINGS.get(name)
    if spelling is not None:
        return _error(text, f"the operator '{spelling}' ({name}) is not part of this language")
    return _error(text, f"{name} is not part of this language")


def _build(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None) -> tuple[pl.Expr, frozenset[str]]:
    """The frame target: the front door, then the engine's own spelling of what survived it."""
    tree, scanned, schema = _front_door(text, columns)
    expression = _Compiler(text, scanned, schema, _FrameTarget(text)).compile(tree.body)
    return expression, frozenset(scanned.columns)


def _front_door(
    text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None
) -> tuple[ast.Expression, _Scanned, dict[str, pl.DataType] | None]:
    """Gate, scan, parse, refuse -- everything both targets must agree on, in one place.

    Shared rather than mirrored, and that is the property the second target rests on: an attack
    refused for a frame is refused for a database because it never reaches either compiler, not
    because two lists of refusals were kept in step with one another.
    """
    if not text or not text.strip():
        raise _error(text, "the expression is empty")
    # In front of the parser, deliberately: see the module docstring.
    if len(text) > MAX_TEXT_LENGTH:
        raise _error(
            text,
            f"the expression is {len(text)} characters long, which is over the {MAX_TEXT_LENGTH} character limit",
        )

    scanned = _scan(text)
    if columns is not None:
        _check_columns(text, scanned.columns, columns)

    tree = _parse(text, scanned.source)
    _check_depth(text, tree)
    _check_breadth(text, tree)
    _check_nodes(text, tree)

    return tree, scanned, dict(columns) if isinstance(columns, Mapping) else None


# ---------------------------------------------------------------------------
# Type families
# ---------------------------------------------------------------------------


def _family_of_dtype(dtype: pl.DataType) -> _Family:
    """The family a column belongs to, from what it holds.

    Anything this module does not recognise is ``UNKNOWN`` rather than a guess: an unrecognised
    dtype means the check has nothing true to say, and refusing on a guess would take capability
    away from a caller whose frame is merely unusual.
    """
    if dtype in (pl.String, pl.Categorical, pl.Enum):
        return _Family.TEXT
    if dtype == pl.Boolean:
        return _Family.BOOLEAN
    if dtype.is_numeric():
        return _Family.NUMBER
    if dtype in (pl.Date, pl.Datetime):
        return _Family.DATE
    if dtype == pl.Null:
        return _Family.NULL
    return _Family.UNKNOWN


def _family_of_value(value: Any) -> _Family:
    """The family a literal belongs to, which is knowable from the text with no schema at all.

    This is what lets ``'100' + '20'`` and ``concat([code], 1)`` be refused at the boundary a
    definition crosses first, where there is no frame to consult.
    """
    if value is None:
        return _Family.NULL
    if isinstance(value, bool):
        return _Family.BOOLEAN
    if isinstance(value, (int, float)):
        return _Family.NUMBER
    if isinstance(value, str):
        return _Family.TEXT
    return _Family.DATE


def _dtype_of_value(value: Any) -> pl.DataType:
    """The exact type a literal holds, beside the coarser family it belongs to.

    One check needs this much detail. A count of months may be 3 and may not be 1.5, and the
    family calls both of those a number.
    """
    if value is None:
        return pl.Null()
    if isinstance(value, bool):
        return pl.Boolean()
    if isinstance(value, int):
        return pl.Int64()
    if isinstance(value, float):
        return pl.Float64()
    if isinstance(value, str):
        return pl.String()
    return pl.Date()


def _combined_dtype(dtypes: Sequence[pl.DataType | None]) -> pl.DataType | None:
    """What a value built out of other values holds, worked out from its parts.

    Parts that all hold the same type give that type. Whole numbers mixed together give a whole
    number, and one decimal among them makes the answer a decimal. A null part is skipped, because
    there is no value in it to widen anything.

    None means the answer is not known. One unknown part is enough to make it unknown, which is the
    honest answer rather than a guess.
    """
    if any(dtype is None for dtype in dtypes):
        return None
    stated = [dtype for dtype in dtypes if dtype != pl.Null]
    if not stated:
        return pl.Null()
    if all(dtype == stated[0] for dtype in stated):
        return stated[0]
    if all(dtype.is_integer() for dtype in stated):
        return pl.Int64()
    if all(dtype.is_numeric() for dtype in stated):
        return pl.Float64()
    return None


def _arithmetic_dtype(
    operation: type[ast.AST], left: pl.DataType | None, right: pl.DataType | None
) -> pl.DataType | None:
    """What an arithmetic result holds, worked out from its two operands.

    Division is the one to watch. It gives a decimal even when both sides are whole, because it is
    true division here as everywhere else in this language: 3 / 2 is 1.5. The other three operators
    keep two whole numbers whole.
    """
    if operation is ast.Div:
        return pl.Float64()
    return _combined_dtype([left, right])


def _holding_phrase(value: Any) -> str:
    """How a refusal describes what a literal is, in the words an author used to write it."""
    if value is None:
        return "is null"
    if isinstance(value, bool):
        return "is a truth value"
    if isinstance(value, int):
        return "is a whole number"
    if isinstance(value, float):
        return "is a decimal"
    if isinstance(value, str):
        return "is text"
    return "is a date"


def _compatible(one: _Family, other: _Family) -> bool:
    """Whether two operands may meet: same family, or one of them meets anything."""
    return one in _WILDCARD_FAMILIES or other in _WILDCARD_FAMILIES or one is other


def _render_value(value: Any) -> str:
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


def _grouped(digits: str, separator: str) -> str:
    """The integer digits split into threes from the right and rejoined with ``separator``.

    The sign is not passed in, because a minus is not one of the digits being grouped: grouping a
    rendered '-1832' from the right gives '-,832' -- a figure that still reads like a number and
    is spotted by nobody.
    """
    if not separator:
        return digits
    lead = len(digits) % 3 or 3
    return separator.join([digits[:lead], *(digits[at : at + 3] for at in range(lead, len(digits), 3))])


def _written_as_text(value: Any, decimals: int, thousands: str, point: str) -> str | None:
    """One number written out, faithful to the value the column holds.

    "Faithful to the value" is the single rule, and it has two readings because there are two ways
    a column holds a figure. A whole number is exact, so 9007199254740993 is written with those
    digits. A decimal is held as a double, so the digits written are the double's: the same
    9007199254740993 written as a decimal is held as ...992 and reads that way. Neither is a
    rounding choice made here -- each is the value itself, asked to describe itself to as many
    places as the line asked for.

    Every value goes through an exact decimal on the way, which is what makes both readings come
    out right from one piece of code: it converts a whole number without going near a double, and
    it converts a double to the digits that double really is, rather than to the shorter decimal
    somebody typed to produce it.

    **A figure that is not a number has no written form, so it is written as null.** inf, -inf and
    nan arrive here in the ordinary way -- '/' is true division on doubles, so [total] / [count] is
    inf on the row where the count is zero and 0.0 / 0.0 is nan -- and the answer is the one a null
    already gets, for the same reason: whoever reads the cell has to be able to tell "no figure"
    from a figure, and never be handed a plausible-looking wrong one.

    copy_abs() and not abs(), because abs() on a Decimal is a *context* operation: it rounds its
    result to the process-wide context's prec using its rounding, so a prec set by any library
    anywhere in the process would move a published figure -- 651,623.80 silently becomes
    651,620.00 under prec = 5. copy_abs() only drops the sign and consults nothing.
    """
    number = Decimal(value)
    # Before the sign test at the end, because Decimal('NaN') < 0 *signals* InvalidOperation rather
    # than answering False the way the float comparison would, and a raw decimal exception out of a
    # select is the one thing this module promises cannot happen. Grouping is unsafe here too:
    # Decimal(inf) renders 'Infinity', which _grouped() would put in a report cell as 'In,fin,ity'.
    if not number.is_finite():
        return None
    rendered = format(number.copy_abs(), f".{decimals}f")
    whole, _, fraction = rendered.partition(".")
    text = _grouped(whole, thousands)
    if fraction:
        text = f"{text}{point}{fraction}"
    return f"-{text}" if number < 0 else text


# ---------------------------------------------------------------------------
# The two targets: what a checked tree is built into
# ---------------------------------------------------------------------------


class _Typed(NamedTuple):
    """One compiled sub-expression, carrying what a refusal about it would need to say.

    ``node`` rather than a rendered string because rendering is only ever wanted on the way to an
    error, and rendering every subtree eagerly would do quadratic work on a healthy expression.
    ``holds`` is a phrase and not a family so that a column can report the dtype it actually holds:
    an author told "expected text" without being told what the column holds goes to the schema to
    find out why, which is a round trip the message should have saved them.

    ``expr`` is whatever the target in force builds -- a column expression for a frame, a clause
    fragment for a database -- because everything around it here is about the *language*, and the
    language is the same either way.

    ``dtype`` is what this value holds, where the language can work that out. The family is
    coarse on purpose: a whole number and a decimal are both numbers, and almost every check here
    is happier that way. One check is not. add_months() moves a date by whole months, so it has to
    tell 3 from 1.5, in a column as well as in a line of the file.

    A leaf fills it in from what it is: a column from the schema, a literal from the value written.
    A value built out of other values fills it in from its parts. Arithmetic and coalesce() work
    theirs out, abs() and round() hand their subject's type on, and year() and len() always give a
    whole number. None means the language could not work the type out, which is what a count of
    months is refused for: a count nobody can show is whole is not one to move a date by.
    """

    expr: Any
    family: _Family
    node: ast.expr
    holds: str
    dtype: pl.DataType | None = None


class _FrameTarget:
    """Builds a column expression: the target that answers a predicate in memory.

    Nothing in here refuses anything about the language -- those refusals belong to
    :class:`_Compiler`, which does them once for both targets. What is left is this engine's
    spelling of each operation, and the handful of places where its spelling is not the obvious
    one.
    """

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = text

    # -- leaves ------------------------------------------------------------

    def column(self, name: str, dtype: pl.DataType | None, family: _Family) -> pl.Expr:
        """One column reference, cast at the leaf wherever the schema says it is stored compactly.

        Two dtypes store text compactly -- dictionary-encoded, and restricted to a fixed set of
        members -- and the engine's string kernels and its own coercions only work reliably
        against a plain string. Casting here, on the leaf, is what makes that true for every
        operation built on top of the reference: a call nested three deep still meets a plain
        string, because the column never handed back anything else. Casting inside each text
        function instead is not enough -- it disappears the moment the value passes through
        something that is not a text function, such as coalesce().
        """
        expression = pl.col(name)
        if family is _Family.TEXT and dtype is not None and dtype != pl.String:
            expression = expression.cast(pl.String)
        return expression

    def literal(self, value: Any) -> pl.Expr:
        return pl.lit(value)

    # -- shapes ------------------------------------------------------------

    def logic(self, operands: Sequence[_Typed], *, conjunction: bool) -> pl.Expr:
        """``and`` / ``or``, left to the engine so that its three-valued logic is the one in force.

        The cast is a no-op on a truth value and keeps a null null, which is what makes the logic
        below faithful rather than merely close.
        """
        combine = operator.and_ if conjunction else operator.or_
        result = operands[0].expr.cast(pl.Boolean)
        for operand in operands[1:]:
            result = combine(result, operand.expr.cast(pl.Boolean))
        return result

    def negate(self, operand: _Typed) -> pl.Expr:
        return ~operand.expr.cast(pl.Boolean)

    def negative(self, operand: _Typed) -> pl.Expr:
        # A null of no type at all has no negation in this engine, which raises rather than
        # answering. So the null is handed back as it stands. There is nothing in it to negate, and
        # the answer is the same null either way.
        if operand.dtype == pl.Null:
            return operand.expr
        return -operand.expr

    def arithmetic(self, operation: type[ast.AST], symbol: str, left: _Typed, right: _Typed) -> pl.Expr:
        return _BINARY_OPERATORS[operation](left.expr, right.expr)

    def compare(self, operation: type[ast.AST], symbol: str, left: _Typed, right: _Typed) -> pl.Expr:
        return _COMPARISONS[operation](left.expr, right.expr)

    def null_test(self, operand: _Typed, *, negated: bool) -> pl.Expr:
        return operand.expr.is_not_null() if negated else operand.expr.is_null()

    def membership(self, left: _Typed, values: Sequence[Any], family: _Family, *, negated: bool) -> pl.Expr:
        """``in`` over a list this module built rather than the one the author's text implies.

        The engine types the list from its first value and rejects the rest with an exception of
        its own, which for a list of whole numbers and decimals together refuses something an
        author wrote perfectly sensibly. One family spanning both number spellings is therefore
        widened here to the one that holds them all.
        """
        if family is _Family.NUMBER and any(isinstance(value, float) for value in values):
            values = [value if value is None else float(value) for value in values]
        result = left.expr.is_in(list(values))
        return ~result if negated else result

    # -- calls -------------------------------------------------------------

    def coalesce(self, parts: Sequence[_Typed], family: _Family) -> pl.Expr:
        return pl.coalesce([part.expr for part in parts])

    def concat(self, parts: Sequence[_Typed]) -> pl.Expr:
        """Null in, null out, like everything else here -- and the safer of the two behaviours.

        Where a joined value is a key for matching, dropping a missing part builds a *shorter* key
        that matches the wrong rows, whereas a null key matches nothing and shows up as a null in
        a count. An author who wants the other behaviour writes it: concat(coalesce([code], ''), '-x').
        """
        return pl.concat_str([self._as_text(part) for part in parts], separator="", ignore_nulls=False)

    def number_function(self, name: str, subject: _Typed, digits: int | None) -> pl.Expr:
        return subject.expr.abs() if name == "abs" else subject.expr.round(digits)

    def year(self, subject: _Typed) -> pl.Expr:
        return subject.expr.dt.year()

    def add_months(self, subject: _Typed, count: _Typed) -> pl.Expr:
        """Move a date by a count of months, landing on the last day of a month that is too short.

        The count is built into the engine's own offset text -- '3mo', '-1mo' -- rather than
        passed as a number, because the count can come out of a column and so is not known here.
        The engine clamps in both directions, which is the behaviour wanted: 31 January plus one
        month is 28 February, not 3 March.

        A null on either side gives a null. The offset text is null when the count is null, and an
        offset of null leaves a null date behind.
        """
        offset = pl.concat_str([count.expr.cast(pl.String), pl.lit("mo")], ignore_nulls=False)
        return subject.expr.dt.offset_by(offset)

    def format_number(self, subject: _Typed, decimals: int, thousands: str, point: str) -> pl.Expr:
        """A number written as text: grouped in threes, to exactly ``decimals`` places.

        **Rounding follows the value the column holds, not the decimal somebody typed.** -15.215 at
        two places is -15.21, for three reasons worth stating together:

        * it is what the figure this library is measured against says, because that figure is
          formatted by the same rule;
        * it is the more honest reading -- an amount summed out of decimals *is* the decimal, and
          -15.215 as one is -15.21499999999999985789..., which is below the half;
        * and it is a disagreement with round() in this same language, which takes the other view
          and gives -15.22. The two are written side by side here, and in the manual, so that a
          reader meets the disagreement in the catalog rather than in a finished report.

        **This is the one function here that works a row at a time, and it is accepted only here.**
        The alternatives were measured rather than assumed: round(), a cast to a 2-place decimal
        type and the obvious scale-then-round all answer -15.22, and a very wide decimal type lands
        on the right side of this tie by luck and drifts the other way at wider scales still. There
        is no route through the engine's own kernels that keeps the rule above. What makes the cost
        payable is where this belongs: it writes a figure for somebody to read, so it runs at the
        end of a flow on the rows that reach a report, not in the middle of one on a million rows.
        Put it last, after the filtering and the joining, and the cost is the rows a person sees.

        A null is left null, the way it is everywhere else here: a formatted 0.00 where nothing was
        measured cannot be told from a real zero by whoever reads the column.
        """
        return subject.expr.map_elements(
            lambda value: _written_as_text(value, decimals, thousands, point),
            return_dtype=pl.String,
        )

    def text_function(self, name: str, subject: _Typed) -> pl.Expr:
        text = self._as_text(subject)
        if name == "lower":
            return text.str.to_lowercase()
        if name == "upper":
            return text.str.to_uppercase()
        if name == "strip":
            return text.str.strip_chars()
        # Characters in a piece of text, not rows in a frame: an expression here is one column's
        # worth of values, so a row count would have no meaning an author could use.
        return text.str.len_chars()

    def text_pair(self, name: str, subject: _Typed, other: _Typed) -> pl.Expr:
        text = self._as_text(subject)
        needle = self._as_text(other)
        if name == "starts_with":
            return text.str.starts_with(needle)
        if name == "ends_with":
            return text.str.ends_with(needle)
        # literal=True, deliberately. A substring test that quietly accepted patterns would make a
        # dot in an account code mean "any character", and the wrong rows would match without
        # anyone noticing. The regex functions are the only door to regular expressions.
        return text.str.contains(needle, literal=True)

    def regex_match(self, subject: _Typed, pattern: str) -> pl.Expr:
        # Unanchored, like the extract beside it: does the pattern occur anywhere. contains()
        # cannot answer this, because contains() is literal by design, which leaves no way at all
        # to ask a pattern question and get a truth value back -- and a filter needs one.
        return self._as_text(subject).str.contains(pattern)

    def regex_extract(self, subject: _Typed, pattern: str, group: int) -> pl.Expr:
        return self._as_text(subject).str.extract(pattern, group)

    def regex_replace(self, subject: _Typed, pattern: str, replacement: str) -> pl.Expr:
        """A rewrite whose replacement is text, verbatim, whatever the engine reads into it.

        The engine underneath reads '$name' in a replacement as a capture group and deletes it
        when no such group exists, so a replacement of '$5.00' lands as '.00' -- a hidden
        mini-language inside a string literal, in a language for amounts. Doubling the sign is how
        that engine spells a literal one. The pattern is wrapped so the engine takes its expansion
        path for every pattern rather than only for some, which keeps the doubling meaning one
        thing: a single-character pattern is otherwise handled by a fast path that would leave the
        doubled sign in the result.
        """
        engine_pattern = f"(?:{pattern})"
        engine_replacement = replacement.replace("$", "$$")
        self._probe("regex_replace", pattern, lambda probe: probe.str.replace_all(engine_pattern, engine_replacement))
        return self._as_text(subject).str.replace_all(engine_pattern, engine_replacement)

    def check_pattern(self, name: str, pattern: str) -> None:
        """Ask the engine that will run the pattern whether it compiles, before any row is read."""
        self._probe(name, pattern, lambda probe: probe.str.contains(pattern))

    # -- helpers -----------------------------------------------------------

    def _as_text(self, typed: _Typed) -> pl.Expr:
        """A text operand, cast so an engine string kernel always meets a plain string.

        Redundant whenever a schema was supplied, because the leaf column reference is already
        cast there -- this is what is left for the caller with no schema in hand, where the leaf
        dtype is unknown and a compactly stored column would otherwise reach a string kernel
        uncast. The cast is a no-op on a column already stored as a plain string.
        """
        return typed.expr.cast(pl.String)

    def _probe(self, name: str, pattern: str, build: Callable[[pl.Expr], pl.Expr]) -> None:
        """Compile one pattern against this engine, in both directions.

        It has to be the engine's own judgement: look-behind is legal in the standard library's
        regular expressions and illegal in this engine, and a unicode script class is the reverse
        -- so validating against the wrong one is worse than not validating, because it either
        passes a pattern that fails at row time or refuses one that works.
        """
        try:
            pl.select(build(pl.lit("", dtype=pl.String)))
        except Exception as failure:
            raise _error(
                self._text,
                f"{name}() was given a pattern the engine cannot compile: '{pattern}' ({failure})",
            ) from None


class _Fragment(NamedTuple):
    """One piece of a clause and the values bound into it, in the order its placeholders appear."""

    sql: str
    params: tuple[Any, ...]


# The per-dialect data, and the only thing a third engine should need to add. Each spelling names
# its operands, and a name may appear more than once -- which is the whole reason the parameters
# are read back off the template rather than bound in the order the operands were evaluated.
_TEMPLATE_OPERAND: Final[re.Pattern[str]] = re.compile(r"\{(\w+)\}")

# coalesce and concat take any number of operands, so they have no template; every dialect here
# spells them the same way, and a dialect that did not would need a builder rather than a string.
_VARIADIC_FUNCTIONS: Final[frozenset[str]] = frozenset({"coalesce", "concat"})

_SQLITE_SPELLINGS: Final[dict[str, str]] = {
    "len": "length({subject})",
    "contains": "(instr({subject}, {needle}) > 0)",
    "starts_with": "(instr({subject}, {needle}) = 1)",
    "ends_with": "(substr({subject}, length({subject}) - length({needle}) + 1) = {needle})",
}

_POSTGRES_SPELLINGS: Final[dict[str, str]] = {
    "len": "length({subject})",
    "contains": "(strpos({subject}, {needle}) > 0)",
    "starts_with": "(strpos({subject}, {needle}) = 1)",
    "ends_with": "(right({subject}, length({needle})) = {needle})",
}


class _Dialect(NamedTuple):
    """One engine: what it is published as able to do, and how each of those is spelled there."""

    subset: SqlSubset
    spellings: dict[str, str]


def _published(spellings: dict[str, str], ordered: frozenset[str], paramstyle: str) -> _Dialect:
    """One dialect's published subset, derived from what it can actually spell.

    Derived rather than written out a second time, because a published list of functions that
    disagrees with the functions a target can build is a promise the compiler does not keep --
    and it would be kept in step by hand, which is to say not kept in step.
    """
    return _Dialect(SqlSubset(frozenset(spellings) | _VARIADIC_FUNCTIONS, ordered, paramstyle), spellings)


_DIALECTS: Final[dict[str, _Dialect]] = {
    # Text ordering is SQLite's and not Postgres's, measured rather than reasoned: SQLite's BINARY
    # collation is byte order, which is what the frame target compares, while Postgres orders text
    # by the server's locale, where 'a' sorts before 'B'. Five of six measured ordering cases kept
    # different rows there. Emitting COLLATE "C" would make them agree and defeat any index on a
    # non-C column, trading a silent wrong answer for a silent table scan -- and using the index is
    # the entire reason to push a predicate down.
    "sqlite": _published(_SQLITE_SPELLINGS, frozenset({"text", "number", "date"}), "qmark"),
    "postgres": _published(_POSTGRES_SPELLINGS, frozenset({"number", "date"}), "format"),
}

_PLACEHOLDERS: Final[dict[str, str]] = {"qmark": "?", "format": "%s"}

_SQL_COMPARISONS: Final[dict[type[ast.AST], str]] = {
    ast.Eq: "=",
    ast.NotEq: "<>",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_SQL_ORDERINGS: Final[frozenset[type[ast.AST]]] = frozenset({ast.Lt, ast.LtE, ast.Gt, ast.GtE})

# The families as ``ordered_families`` names them, which is not how a refusal about a value names
# them: "a number" reads well in "[qty] is a number" and badly in "cannot order a number".
_FAMILY_NAMES: Final[dict[_Family, str]] = {
    _Family.TEXT: "text",
    _Family.NUMBER: "number",
    _Family.DATE: "date",
    _Family.BOOLEAN: "truth values",
}

# Why an engine's order for a family is not the frame target's, where the difference has a name
# worth putting in front of an author. Keyed by dialect as well as family, because the next engine
# added here will be wrong in its own way rather than in this one's.
_ORDERING_NOTES: Final[dict[tuple[str, str], str]] = {
    ("postgres", "text"): (
        " -- the server orders text by its own locale, where 'a' sorts before 'B', while the frame target orders it "
        "by bytes"
    ),
}

_PATTERN_REASON: Final[str] = (
    "no engine here spells a pattern the way the frame target's regular expressions are spelled, leaving the same "
    "pattern to match different rows"
)

# Every exclusion below is a measured disagreement between an engine and the frame target, not a
# worry about one. They are keyed by function alone because each was wrong in both engines -- which
# is itself the argument for the per-dialect model, since upper() and lower() are wrong in the two
# engines for entirely different reasons.
_OUTSIDE_THE_SUBSET: Final[dict[str, str]] = {
    "upper": (
        "case folding belongs to the engine: 'straße' uppercases to STRASSE in the frame target and leaves the ß "
        "alone at both engines, one of which folds nothing but ASCII in the first place"
    ),
    "lower": (
        "case folding belongs to the engine: 'İstanbul' lowercases to an 'i' with a combining dot in the frame "
        "target, to a plain 'i' at one engine and not at all at the other"
    ),
    "strip": (
        "TRIM and btrim take spaces where strip() takes every kind of whitespace, leaving a tab-padded value to "
        "survive one and not the other"
    ),
    "round": (
        "a tie rounds away from zero at the engine and to even in the frame target, putting a half-cent on the other "
        "side of the comparison"
    ),
    "abs": (
        "it is arithmetic, and a whole number at the edge of the range is promoted to a decimal by one engine and "
        "refused mid-read by the other"
    ),
    "year": (
        "one engine has no EXTRACT at all and spells it as text through strftime, so the same "
        "expression compares a number on one side against a string of digits on the other"
    ),
    "add_months": (
        "each engine spells date arithmetic its own way -- one adds an interval, the other adds text to a date "
        "function -- and they disagree with the frame target about the day a date lands on when the month it moves "
        "into is too short to hold it, so the same shift keeps different rows"
    ),
    "regex_extract": _PATTERN_REASON,
    "regex_replace": _PATTERN_REASON,
    "regex_match": _PATTERN_REASON,
    "format_number": (
        "there is no portable spelling of a grouped, fixed-decimal number, and the last place is "
        "exactly what each engine rounds its own way"
    ),
}

_ARITHMETIC_REASON: Final[str] = (
    "integer division truncates at both engines, and a whole number that overflows is promoted to a decimal by one "
    "engine (keeping a row the frame target drops) and refused mid-read by the other"
)


def _fill(template: str, **operands: _Fragment) -> _Fragment:
    """Fill one dialect's spelling, binding each operand once per appearance, in appearance order.

    The order is read off the template rather than tracked by hand, because a spelling that names
    an operand twice -- as every ends_with does, and as SQLite's names its subject -- has to bind
    that operand's values once per placeholder, in the order the placeholders appear and not the
    order the operands were evaluated. A driver binds positionally, so getting that wrong raises
    nothing: the values simply land in the wrong comparisons.
    """
    appearances = _TEMPLATE_OPERAND.findall(template)
    sql = template.format(**{name: operand.sql for name, operand in operands.items()})
    return _Fragment(sql, tuple(chain.from_iterable(operands[name].params for name in appearances)))


class _SqlTarget:
    """Builds a boolean clause for one dialect, with every author-written value bound beside it.

    Two properties hold for every method here, and they are the reason a pushdown is defensible at
    all. A column reaches the clause only as a quoted identifier that passed
    :data:`SQL_IDENTIFIER`; a name outside that grammar is refused rather than escaped. Everything
    else the author wrote becomes a parameter, so the clause a hostile literal produces is
    character for character the clause a harmless one produces, and the worst that value can do is
    match no row.

    What this engine cannot answer the way the frame target answers it is refused here, by name and
    naming the dialect, because "upper cannot be pushed down" invites an argument where "upper
    cannot be pushed down to postgres" can be acted on.
    """

    __slots__ = ("_dialect", "_mark", "_spellings", "_subset", "_text")

    def __init__(self, text: str, dialect: str, paramstyle: str) -> None:
        self._text = text
        self._dialect = dialect
        self._subset = _DIALECTS[dialect].subset
        self._spellings = _DIALECTS[dialect].spellings
        self._mark = _PLACEHOLDERS[paramstyle]

    # -- leaves ------------------------------------------------------------

    def column(self, name: str, dtype: pl.DataType | None, family: _Family) -> _Fragment:
        if SQL_IDENTIFIER.fullmatch(name) is None:
            raise _error(
                self._text,
                f"[{name}] cannot be named in a clause read at the database, because a name written into SQL has to "
                f"be letters, digits and underscores and cannot start with a digit -- it is the one thing a driver "
                f"cannot bind as a value, so it is refused here rather than escaped: rename the column, or filter "
                f"after the read",
            )
        return _Fragment(f'"{name}"', ())

    def literal(self, value: Any) -> _Fragment:
        """Every literal, without exception, as a placeholder and the value that belongs in it."""
        return _Fragment(self._mark, (value,))

    # -- shapes ------------------------------------------------------------

    def logic(self, operands: Sequence[_Typed], *, conjunction: bool) -> _Fragment:
        return self._joined("(", " and " if conjunction else " or ", ")", operands)

    def negate(self, operand: _Typed) -> _Fragment:
        return _fill("(not {operand})", operand=operand.expr)

    def negative(self, operand: _Typed) -> _Fragment:
        raise self._arithmetic_refusal("-")

    def arithmetic(self, operation: type[ast.AST], symbol: str, left: _Typed, right: _Typed) -> _Fragment:
        raise self._arithmetic_refusal(symbol)

    def compare(self, operation: type[ast.AST], symbol: str, left: _Typed, right: _Typed) -> _Fragment:
        if operation in _SQL_ORDERINGS:
            self._check_ordering(symbol, left, right)
        template = f"({{left}} {_SQL_COMPARISONS[operation]} {{right}})"
        return _fill(template, left=left.expr, right=right.expr)

    def null_test(self, operand: _Typed, *, negated: bool) -> _Fragment:
        keyword = "is not null" if negated else "is null"
        return _fill(f"({{operand}} {keyword})", operand=operand.expr)

    def membership(self, left: _Typed, values: Sequence[Any], family: _Family, *, negated: bool) -> _Fragment:
        """``in`` over a bound list, with the one shape every engine answers differently refused.

        A null inside the list is the sharpest disagreement in this whole target: ``[code] not in
        ('A', null)`` keeps every non-matching row in the frame and *no row at all* in SQL, because
        a null in the list leaves NOT IN unknown for every row not already excluded. A filter that
        quietly empties a table is the worst shape a disagreement can take, so both spellings are
        refused -- the list is the same construct in each, and a rule that allows one but not the
        other is a rule nobody can keep.
        """
        if any(value is None for value in values):
            raise _error(
                self._text,
                "a null inside a list of values cannot be read at the database, because 'not in' then answers no row "
                "at all where the frame target answers every row that does not match: write the null test on its "
                "own, as in [code] is null or [code] in ('A')",
            )
        if not values:
            raise _error(
                self._text,
                "an empty list of values cannot be read at the database, because there is no clause that says it: "
                "name at least one value to match",
            )
        marks = ", ".join(self._mark for _ in values)
        keyword = "not in" if negated else "in"
        return _Fragment(f"({left.expr.sql} {keyword} ({marks}))", left.expr.params + tuple(values))

    # -- calls -------------------------------------------------------------

    def coalesce(self, parts: Sequence[_Typed], family: _Family) -> _Fragment:
        return self._joined("coalesce(", ", ", ")", parts)

    def concat(self, parts: Sequence[_Typed]) -> _Fragment:
        return self._joined("(", " || ", ")", parts)

    def number_function(self, name: str, subject: _Typed, digits: int | None) -> _Fragment:
        raise self._outside(name)

    def year(self, subject: _Typed) -> _Fragment:
        raise self._outside("year")

    def add_months(self, subject: _Typed, count: _Typed) -> _Fragment:
        raise self._outside("add_months")

    def text_function(self, name: str, subject: _Typed) -> _Fragment:
        return _fill(self._spelling(name), subject=subject.expr)

    def text_pair(self, name: str, subject: _Typed, other: _Typed) -> _Fragment:
        return _fill(self._spelling(name), subject=subject.expr, needle=other.expr)

    def regex_match(self, subject: _Typed, pattern: str) -> _Fragment:
        raise self._outside("regex_match")

    def regex_extract(self, subject: _Typed, pattern: str, group: int) -> _Fragment:
        raise self._outside("regex_extract")

    def regex_replace(self, subject: _Typed, pattern: str, replacement: str) -> _Fragment:
        raise self._outside("regex_replace")

    def format_number(self, subject: _Typed, decimals: int, thousands: str, point: str) -> _Fragment:
        raise self._outside("format_number")

    def check_pattern(self, name: str, pattern: str) -> None:
        """Nothing here will ever run a pattern, so there is no engine to ask and nothing to check.

        The refusal comes when the call is built, from the function's own name and this dialect's,
        which is the message an author can act on -- a complaint about the pattern would send them
        to fix something that was never going to run.
        """

    # -- helpers -----------------------------------------------------------

    def _joined(self, opening: str, joiner: str, closing: str, parts: Sequence[_Typed]) -> _Fragment:
        fragments = [part.expr for part in parts]
        return _Fragment(
            opening + joiner.join(fragment.sql for fragment in fragments) + closing,
            tuple(chain.from_iterable(fragment.params for fragment in fragments)),
        )

    def _spelling(self, name: str) -> str:
        """This dialect's spelling of one function, or the refusal that it has none."""
        template = self._spellings.get(name)
        if template is None:
            raise self._outside(name)
        return template

    def _check_ordering(self, symbol: str, left: _Typed, right: _Typed) -> None:
        """Refuse an ordering this engine was not measured to agree with the frame target about.

        Equality, ``<>`` and ``in`` are not asked: they agree under any deterministic collation,
        and it is only the *order* of two unequal values that an engine decides for itself.
        """
        family = left.family if left.family not in _WILDCARD_FAMILIES else right.family
        if family in _WILDCARD_FAMILIES:
            return
        named = _FAMILY_NAMES[family]
        if named in self._subset.ordered_families:
            return
        raise _error(
            self._text,
            f"'{symbol}' cannot order {named} at the {self._dialect} database, because that engine's order for "
            f"{named} was not measured to be the frame target's"
            f"{_ORDERING_NOTES.get((self._dialect, named), '')}: compare with '=', '<>' or 'in' here, and put the "
            f"ordering in a Filter step after the read",
        )

    def _outside(self, name: str) -> ExpressionError:
        return _error(
            self._text,
            f"{name}() cannot be read at the {self._dialect} database, because {_OUTSIDE_THE_SUBSET[name]} -- so "
            f"apply it in a Filter step after the read, where the frame target's meaning is the one in force",
        )

    def _arithmetic_refusal(self, symbol: str) -> ExpressionError:
        return _error(
            self._text,
            f"'{symbol}' cannot be read at the {self._dialect} database, because arithmetic is outside every subset "
            f"here on measurement rather than on taste -- {_ARITHMETIC_REASON} -- so do it in a Formula or Filter "
            f"step after the read, where the frame target's meaning is the one in force",
        )


# ---------------------------------------------------------------------------
# Compiling the surviving nodes
# ---------------------------------------------------------------------------


class _Compiler:
    """Translates a checked tree into whatever the target in force builds.

    Every refusal about the language lives here and is therefore made once for both targets: what
    a function may be given, what two sides of a comparison may hold, what a list of values may
    mix. The target decides only how the surviving operation is *spelled*, and refuses only what
    its own engine cannot answer the way the other one does.

    Recursive, which is safe only because :func:`_check_depth` has already bounded the tree. The
    order matters: nothing in here may run before the whitelist has, so this class is private and
    is only ever constructed on a tree :func:`_front_door` has already returned.
    """

    __slots__ = ("_columns", "_literals", "_schema", "_target", "_text")

    def __init__(
        self,
        text: str,
        scanned: _Scanned,
        schema: dict[str, pl.DataType] | None,
        target: _FrameTarget | _SqlTarget,
    ) -> None:
        self._text = text
        self._columns = scanned.columns
        self._literals = scanned.literals
        self._schema = schema
        self._target = target

    def compile(self, node: ast.expr) -> Any:
        """What one node builds to, with its type discarded once it has been checked."""
        return self._typed(node).expr

    def compile_condition(self, node: ast.expr) -> Any:
        """:meth:`compile`, plus the refusal that what it built is a condition and not a value.

        ``[amount]`` is what an author writes when they meant ``[amount] > 0``, and it compiles,
        because that is exactly how a column reference is spelled. Handed to a database as a WHERE
        clause it is answered by *truthiness* at one engine -- a non-zero amount keeps the row, a
        zero one drops it, and nothing anywhere raises -- and refused at the other in a driver's
        vocabulary, naming neither the field nor the text. So it is refused here, against the
        schema, naming what the expression holds. With no schema in hand nothing can be said about
        a column, and the hole is left open rather than papered over: whoever holds the real
        column types is the one who can close it.
        """
        typed = self._typed(node)
        if typed.family is not _Family.BOOLEAN and typed.family not in _WILDCARD_FAMILIES:
            raise _error(
                self._text,
                f"{self._describe(typed)}, but a clause read at the database has to be a condition: write the "
                f"comparison the value was meant for, as in [amount] > 0",
            )
        return typed.expr

    def _typed(self, node: ast.expr) -> _Typed:
        """The expression for one node and the family it belongs to, dispatching on its type."""
        if isinstance(node, ast.BoolOp):
            return self._bool_op(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary_op(node)
        if isinstance(node, ast.BinOp):
            return self._binary_op(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Tuple):
            raise _error(
                self._text,
                "a list of values in parentheses is only meaningful on the right of 'in', as in [code] in ('A', 'B')",
            )
        raise _refusal(self._text, node)

    # -- shapes ------------------------------------------------------------

    def _bool_op(self, node: ast.BoolOp) -> _Typed:
        """``and`` / ``or``, left to the engine so that its three-valued logic is the one in force.

        Each operand has to be a truth value. Without that rule a logical operator applied to a
        number would quietly become a bitwise one -- ``[qty] and [flag]`` returning arithmetic
        instead of a truth value -- so a column of counts is refused and the author writes
        ``[qty] > 0``, which is what they meant.
        """
        spelling = _OPERATOR_SYMBOLS[type(node.op)]
        operands: list[_Typed] = []
        for value in node.values:
            part = self._typed(value)
            self._require(part, _Family.BOOLEAN, f"'{spelling}' needs a truth value on both sides")
            operands.append(part)
        result = self._target.logic(operands, conjunction=isinstance(node.op, ast.And))
        return _Typed(result, _Family.BOOLEAN, node, "gives a truth value")

    def _unary_op(self, node: ast.UnaryOp) -> _Typed:
        if isinstance(node.op, ast.Not):
            operand = self._typed(node.operand)
            self._require(operand, _Family.BOOLEAN, "'not' needs a truth value")
            return _Typed(self._target.negate(operand), _Family.BOOLEAN, node, "gives a truth value")

        # A signed number is one literal, not a negation applied to one, so the bound on a whole
        # number is measured here -- on the value the author wrote. That is what makes the most
        # negative whole number spellable, and it holds for a doubled sign too, because folding
        # happens before the bound is applied rather than beneath it.
        folded = self._literal(node)
        if isinstance(folded, (int, float)) and not isinstance(folded, bool):
            self._check_whole_number(folded)
            return self._typed_value(folded, node)

        operand = self._typed(node.operand)
        self._require(operand, _Family.NUMBER, "a minus sign needs a number")
        if operand.family is _Family.NULL:
            # A minus sign in front of a null has nothing to negate, so the answer is still null.
            # It stays in the family that meets anything, the way the null it was written from
            # does, and every check below reads it as the null it is.
            return _Typed(self._target.negative(operand), _Family.NULL, node, operand.holds, operand.dtype)
        # The type comes along, because a minus sign does not change it: the negative of a whole
        # number is a whole number, and the negative of a decimal is a decimal.
        return _Typed(self._target.negative(operand), _Family.NUMBER, node, "gives a number", operand.dtype)

    def _binary_op(self, node: ast.BinOp) -> _Typed:
        operation = type(node.op)
        if operation not in _BINARY_OPERATORS:  # pragma: no cover - the whitelist refused the rest
            raise _refusal(self._text, node.op)
        symbol = _OPERATOR_SYMBOLS[operation]
        left = self._typed(node.left)
        right = self._typed(node.right)
        for operand in (left, right):
            self._require_arithmetic(operand, symbol)
        built = self._target.arithmetic(operation, symbol, left, right)
        dtype = _arithmetic_dtype(operation, left.dtype, right.dtype)
        return _Typed(built, _Family.NUMBER, node, "gives a number", dtype)

    def _compare(self, node: ast.Compare) -> _Typed:
        if len(node.ops) != 1:
            raise _error(
                self._text,
                "a chained comparison like 1 < [qty] < 3 is not part of this language: write the two comparisons "
                "joined with 'and', as in 1 < [qty] and [qty] < 3",
            )
        operation = node.ops[0]
        right_node = node.comparators[0]

        if isinstance(operation, (ast.Is, ast.IsNot)):
            return self._null_test(node, right_node, negated=isinstance(operation, ast.IsNot))
        if isinstance(operation, (ast.In, ast.NotIn)):
            return self._membership(node, right_node, negated=isinstance(operation, ast.NotIn))

        if type(operation) not in _COMPARISONS:  # pragma: no cover - the whitelist refused the rest
            raise _refusal(self._text, operation)
        left = self._typed(node.left)
        right = self._typed(right_node)
        if not _compatible(left.family, right.family):
            raise _error(
                self._text,
                f"a comparison needs both sides in the same type family, but {self._describe(left)} and "
                f"{self._describe(right)}",
            )
        symbol = _OPERATOR_SYMBOLS[type(operation)]
        built = self._target.compare(type(operation), symbol, left, right)
        return _Typed(built, _Family.BOOLEAN, node, "gives a truth value")

    def _null_test(self, node: ast.Compare, right_node: ast.expr, *, negated: bool) -> _Typed:
        """``is null`` / ``is not null`` -- the only expressions in this language never themselves null.

        That is what makes them the escape hatch from three-valued logic: every other test against
        a null row is null, and a filter drops it, so without these two an author could not find a
        missing value at all.
        """
        if not (isinstance(right_node, ast.Name) and right_node.id == "null"):
            raise _error(
                self._text,
                "'is' may only be written as 'is null' or 'is not null': to compare values, use '=' or '!='",
            )
        left = self._typed(node.left)
        return _Typed(self._target.null_test(left, negated=negated), _Family.BOOLEAN, node, "gives a truth value")

    def _membership(self, node: ast.Compare, right_node: ast.expr, *, negated: bool) -> _Typed:
        """``in`` / ``not in`` over a parenthesised list of literals.

        Only literals: a set of values to match is a decision recorded in the file, and allowing a
        column on the right would turn a readable membership test into a row-by-row join nobody
        reviewing the file would see coming.

        A mixed *family* is refused in this language's words rather than in the engine's, which
        would leak an exception type of its own out of compilation; what a target then does with
        one family spelled two ways is the target's business.
        """
        candidates = right_node.elts if isinstance(right_node, ast.Tuple) else [right_node]
        values: list[Any] = []
        parts: list[_Typed] = []
        for candidate in candidates:
            value = self._literal(candidate)
            if value is _MISSING:
                raise _error(
                    self._text,
                    "'in' needs a list of fixed values in parentheses, as in [code] in ('A', 'B')",
                )
            if isinstance(value, int) and not isinstance(value, bool):
                self._check_whole_number(value)
            values.append(value)
            parts.append(self._typed_value(value, candidate))

        listed = self._one_family(parts, "a list of values to match needs every value in the same type family")
        left = self._typed(node.left)
        if not _compatible(left.family, listed):
            raise _error(
                self._text,
                f"a membership test needs the listed values in the same type family as what they are matched "
                f"against, but {self._describe(left)} and the list is {listed.value}",
            )
        built = self._target.membership(left, values, listed, negated=negated)
        return _Typed(built, _Family.BOOLEAN, node, "gives a truth value")

    # -- leaves ------------------------------------------------------------

    def _name(self, node: ast.Name) -> _Typed:
        if node.id in _KEYWORD_LITERALS:
            return self._typed_value(_KEYWORD_LITERALS[node.id], node)
        raise _error(
            self._text,
            f"{node.id!r} is not something this language knows: write a column as [{node.id}], text as 'value', or "
            f"one of true, false and null",
        )

    def _constant(self, node: ast.Constant) -> _Typed:
        value = node.value
        if value is None or isinstance(value, bool):
            raise _error(
                self._text,
                f"the Python literal {value!r} is not part of this language: write true, false or null in lower case",
            )
        if isinstance(value, int):
            self._check_whole_number(value)
        if isinstance(value, (int, float)):
            return self._typed_value(value, node)
        raise _error(
            self._text,
            f"{value!r} is not a literal this language supports (it is a {type(value).__name__}): the literals are "
            f"whole numbers, decimals, single-quoted text, true, false, null and date'2026-01-01'",
        )

    def _column(self, node: ast.Call) -> _Typed:
        name = self._column_name(node)
        if self._schema is None:
            built = self._target.column(name, None, _Family.UNKNOWN)
            return _Typed(built, _Family.UNKNOWN, node, "holds an unknown type")
        dtype = self._schema[name]
        family = _family_of_dtype(dtype)
        return _Typed(self._target.column(name, dtype, family), family, node, f"holds {dtype}", dtype)

    def _typed_value(self, value: Any, node: ast.expr) -> _Typed:
        """One literal, whose family is knowable from the text with or without a schema."""
        return _Typed(
            self._target.literal(value),
            _family_of_value(value),
            node,
            _holding_phrase(value),
            _dtype_of_value(value),
        )

    # -- calls -------------------------------------------------------------

    def _call(self, node: ast.Call) -> _Typed:
        if not isinstance(node.func, ast.Name):  # pragma: no cover - the whitelist refuses any other callee
            raise _refusal(self._text, node.func)
        name = node.func.id

        if name == _COLUMN_FN:
            return self._column(node)
        if name == _LITERAL_FN:
            return self._typed_value(self._interned(node), node)
        if name not in ALLOWED_FUNCTIONS:
            available = ", ".join(sorted(ALLOWED_FUNCTIONS))
            raise _error(self._text, f"{name!r} is not a function this language has: the functions are {available}")

        arity = _FUNCTION_ARITY[name]
        arguments = node.args
        if arity == -1:
            if len(arguments) < 2:
                raise _error(
                    self._text,
                    f"{name}() takes at least 2 arguments, but was given {len(arguments)}",
                )
        elif len(arguments) != arity:
            raise _error(
                self._text,
                f"{name}() takes exactly {arity} argument{'s' if arity > 1 else ''}, but was given {len(arguments)}",
            )
        return self._apply(name, node)

    def _apply(self, name: str, node: ast.Call) -> _Typed:
        """One whitelisted function: its operands checked here, its spelling left to the target."""
        arguments = node.args

        if name == "coalesce":
            parts = [self._typed(argument) for argument in arguments]
            family = self._one_family(parts, "coalesce() needs every value it chooses between in the same type family")
            holds = f"gives {family.value}" if family not in _WILDCARD_FAMILIES else "holds an unknown type"
            dtype = _combined_dtype([part.dtype for part in parts])
            return _Typed(self._target.coalesce(parts, family), family, node, holds, dtype)

        if name == "concat":
            parts = [self._typed(argument) for argument in arguments]
            for part in parts:
                self._require(part, _Family.TEXT, "concat() joins text")
            return self._result(name, self._target.concat(parts), node)

        if name == "year":
            subject = self._typed(arguments[0])
            self._require(subject, _Family.DATE, "year() reads the year of a date")
            return self._result(name, self._target.year(subject), node)

        if name == "add_months":
            subject = self._typed(arguments[0])
            self._require(subject, _Family.DATE, "add_months() moves a date")
            count = self._typed(arguments[1])
            self._require(count, _Family.NUMBER, "add_months() moves a date by a count of months")
            self._require_whole_months(count)
            return self._result(name, self._target.add_months(subject, count), node)

        if name == "format_number":
            subject = self._typed(arguments[0])
            self._require(subject, _Family.NUMBER, "format_number() writes a number as text")
            # The width and both separators are read out of the file and never out of a row. How
            # many places a figure is written to is a decision a reviewer reads in the line, the
            # way round()'s digits and regex_extract()'s group are -- a width that varied row by
            # row would not be a width at all.
            decimals = self._whole_number(name, arguments[1], "second")
            thousands = self._text_literal(name, arguments[2], "third", _SEPARATOR_REASON)
            point = self._text_literal(name, arguments[3], "fourth", _SEPARATOR_REASON)
            built = self._target.format_number(subject, decimals, thousands, point)
            return self._result(name, built, node)

        if name in _NUMBER_FUNCTIONS:
            subject = self._typed(arguments[0])
            self._require(subject, _Family.NUMBER, f"{name}() works on numbers")
            digits = None if name == "abs" else self._whole_number(name, arguments[1], "second")
            built = self._target.number_function(name, subject, digits)
            return self._result(name, built, node, subject.dtype)

        subject = self._typed(arguments[0])
        self._require(subject, _Family.TEXT, f"{name}() works on text")

        if name in _TEXT_PAIR_FUNCTIONS:
            other = self._typed(arguments[1])
            self._require(other, _Family.TEXT, f"{name}() compares text against text")
            return self._result(name, self._target.text_pair(name, subject, other), node)

        if name in _PATTERN_FUNCTIONS:
            return self._regex(name, subject, node)

        return self._result(name, self._target.text_function(name, subject), node)

    def _regex(self, name: str, subject: _Typed, node: ast.Call) -> _Typed:
        """The three pattern functions, each checked against the engine that will run the pattern."""
        arguments = node.args
        pattern = self._pattern(name, arguments[1])

        if name == "regex_match":
            return self._result(name, self._target.regex_match(subject, pattern), node)
        if name == "regex_extract":
            group = self._whole_number(name, arguments[2], "third")
            return self._result(name, self._target.regex_extract(subject, pattern, group), node)

        replacement = self._text_literal(name, arguments[2], "third")
        return self._result(name, self._target.regex_replace(subject, pattern, replacement), node)

    def _result(self, name: str, built: Any, node: ast.Call, dtype: pl.DataType | None = None) -> _Typed:
        """One function's result, carrying the family it hands on so that nesting is checked.

        Some functions know what they hand back exactly, and that is carried too. year() and len()
        always give a whole number. abs() and round() give back whatever their subject holds, which
        is passed in, because the subject is not in view here. The rest leave it unknown.
        """
        family = _FUNCTION_RESULTS[name]
        if dtype is None:
            dtype = _FUNCTION_RESULT_DTYPES.get(name)
        return _Typed(built, family, node, f"gives {family.value}", dtype)

    # -- type requirements -------------------------------------------------

    def _require(self, typed: _Typed, family: _Family, requirement: str) -> None:
        """Refuse an operand outside the family an operation works in, saying all three facts.

        Which operand, what it holds, and what was needed instead. A wildcard family is skipped:
        a null meets anything, and an unchecked column is not something to refuse on.
        """
        if typed.family in _WILDCARD_FAMILIES or typed.family is family:
            return
        raise _error(self._text, f"{self._describe(typed)}, but {requirement}")

    def _require_whole_months(self, typed: _Typed) -> None:
        """Refuse a count of months that cannot be shown to be whole.

        Half a month is not a distance a date can move. A count like 1.5 is not something anyone
        writes on purpose; it arrives because a number was divided by something, and rounding it
        here would bury that division inside a date that looks perfectly ordinary.

        The count is checked however it was built. A decimal column is refused for the same reason
        a decimal typed into the file is: the schema says which one a column is before a row is
        read. So is a decimal that arrives through arithmetic, or through abs(), round() or
        coalesce(), because each of those works out what it hands back from what it was given.
        Division is refused whatever it was given, since 3 / 2 is 1.5.

        A count whose type could not be worked out is refused too. This check would rather say no
        to an expression it cannot read than let one through and have the engine fail on a row.

        Two things are not refusals. A null is not half a month: there is no count, so the date
        comes back null. And nothing is refused when no schema was given, because then there are no
        types to read at all, which is how every other check here behaves.
        """
        if typed.family in _WILDCARD_FAMILIES or self._schema is None:
            return
        if typed.dtype is None:
            raise _error(
                self._text,
                f"{self._describe(typed)}, and nothing here can work out whether that is whole, but add_months() "
                f"moves a date by whole months",
            )
        if typed.dtype.is_integer() or typed.dtype == pl.Null:
            return
        raise _error(self._text, f"{self._describe(typed)}, but add_months() moves a date by whole months")

    def _require_arithmetic(self, typed: _Typed, symbol: str) -> None:
        """Arithmetic needs numbers, and text is the operand worth a sentence of its own.

        The engine joins two pieces of text with '+', so two amount columns that arrived from an
        import as text add up to '10020' instead of 120. Both operands are in the same family, so
        no other rule here would ever catch it; the only way to close it is to refuse the operator
        and move the capability to a name that says what it does.
        """
        if typed.family in _WILDCARD_FAMILIES or typed.family is _Family.NUMBER:
            return
        hint = (
            ", and to join two pieces of text use concat() rather than an arithmetic operator"
            if typed.family is _Family.TEXT
            else ""
        )
        raise _error(
            self._text,
            f"{self._describe(typed)}, but arithmetic with '{symbol}' needs numbers on both sides{hint}",
        )

    def _one_family(self, parts: Sequence[_Typed], requirement: str) -> _Family:
        """The single family a list of operands agrees on, refusing the first that disagrees.

        The engine would rather coerce -- it will turn a number into text to sit beside one without
        a word -- and a number that arrives in a column of text looks like data somebody meant.
        """
        settled: _Typed | None = None
        for part in parts:
            if part.family in _WILDCARD_FAMILIES:
                continue
            if settled is None:
                settled = part
            elif part.family is not settled.family:
                raise _error(
                    self._text,
                    f"{requirement}, but {self._describe(settled)} and {self._describe(part)}",
                )
        if settled is not None:
            return settled.family
        return _Family.NULL if all(part.family is _Family.NULL for part in parts) else _Family.UNKNOWN

    def _describe(self, typed: _Typed) -> str:
        """One operand named the way its author wrote it, with what it holds."""
        return f"{self._render(typed.node)} {typed.holds}"

    def _render(self, node: ast.expr) -> str:
        """The author's own spelling of one subtree, for a refusal that points at the part at fault.

        Rebuilt from the scanned tree rather than sliced out of the incoming text, because the scan
        is the only thing that knows which index belongs to which name or literal. Quoting the whole
        line back is not enough on its own: an author reading a refusal about a line with four
        columns in it should not have to work out which of the four is the problem.
        """
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name == _COLUMN_FN:
                return f"[{self._column_name(node)}]"
            if name == _LITERAL_FN:
                return _render_value(self._interned(node))
            return f"{name}({', '.join(self._render(argument) for argument in node.args)})"
        if isinstance(node, ast.Constant):
            return repr(node.value)
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp):
            rendered = self._render(node.operand)
            return f"not {rendered}" if isinstance(node.op, ast.Not) else f"-{rendered}"
        if isinstance(node, ast.BinOp):
            symbol = _OPERATOR_SYMBOLS.get(type(node.op), "?")
            return f"{self._render(node.left)} {symbol} {self._render(node.right)}"
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            symbol = _OPERATOR_SYMBOLS.get(type(node.ops[0]), "?")
            return f"{self._render(node.left)} {symbol} {self._render(node.comparators[0])}"
        if isinstance(node, ast.BoolOp):
            joiner = f" {_OPERATOR_SYMBOLS.get(type(node.op), '?')} "
            return joiner.join(self._render(value) for value in node.values)
        if isinstance(node, ast.Tuple):
            return f"({', '.join(self._render(element) for element in node.elts)})"
        return "that value"  # pragma: no cover - every whitelisted shape is rendered above

    # -- helpers -----------------------------------------------------------

    def _pattern(self, name: str, node: ast.expr) -> str:
        """The pattern for one of the regex functions: quoted text, checked, or nothing.

        A literal and not a column, which is what makes a check before the run possible at all: a
        pattern that varies row by row can be neither read in the file nor validated up front.
        """
        pattern = self._literal(node)
        if not isinstance(pattern, str):
            raise _error(
                self._text,
                f"{name}() needs a quoted pattern as its second argument, as in {name}([label], '[a-z]+'...): a "
                f"pattern is a fixed decision recorded in the file, so it cannot come out of a column",
            )
        self._target.check_pattern(name, pattern)
        return pattern

    def _text_literal(self, name: str, node: ast.expr, position: str, reason: str = _REWRITE_REASON) -> str:
        """Quoted text, for an argument that has to be a decision in the file rather than data.

        The reason travels with the argument rather than with this method, because the argument is
        what an author has to change: a replacement and a separator are refused by the same rule
        and for two different-sounding versions of the same worry.
        """
        value = self._literal(node)
        if not isinstance(value, str):
            raise _error(self._text, f"{name}() needs quoted text as its {position} argument: {reason}")
        return value

    def _whole_number(self, name: str, node: ast.expr, position: str) -> int:
        value = self._literal(node)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _SMALL_NUMBER_LIMIT:
            raise _error(
                self._text,
                f"{name}() needs a whole number between 0 and {_SMALL_NUMBER_LIMIT} as its {position} argument",
            )
        return value

    def _check_whole_number(self, value: int | float) -> None:
        """Refuse a whole number no column could hold, measured on the value as written."""
        if isinstance(value, int) and not _WHOLE_NUMBER_MIN <= value <= _WHOLE_NUMBER_MAX:
            raise _error(
                self._text,
                f"the whole number {value} is outside the range this language holds, which is {_WHOLE_NUMBER_MIN} to "
                f"{_WHOLE_NUMBER_MAX}",
            )

    def _column_name(self, node: ast.Call) -> str:
        index = self._table_index(node, len(self._columns))
        return self._columns[index]

    def _interned(self, node: ast.Call) -> Any:
        index = self._table_index(node, len(self._literals))
        return self._literals[index]

    def _table_index(self, node: ast.Call, size: int) -> int:
        """The index carried by one of the scan's own calls.

        Unreachable by hand -- the scan refuses these names in incoming text -- so this is a
        consistency check on this module rather than a check on the author, and it fails closed.
        """
        if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
            index = node.args[0].value
            if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < size:
                return index
        raise _error(self._text, "the expression could not be understood")  # pragma: no cover

    def _literal(self, node: ast.expr) -> Any:
        """The value of ``node`` if it is a literal, else :data:`_MISSING`.

        A sentinel rather than an exception because each caller wants to say something different
        about a value it cannot use, in the author's terms rather than the tree's.
        """
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or value is None:
                return _MISSING
            return value if isinstance(value, (int, float)) else _MISSING
        if isinstance(node, ast.Name) and node.id in _KEYWORD_LITERALS:
            return _KEYWORD_LITERALS[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._literal(node.operand)
            return -inner if isinstance(inner, (int, float)) and not isinstance(inner, bool) else _MISSING
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _LITERAL_FN:
            return self._interned(node)
        return _MISSING


# ---------------------------------------------------------------------------
# Column checking and error text
# ---------------------------------------------------------------------------


def _check_columns(text: str, referenced: tuple[str, ...], columns: Collection[str]) -> None:
    """Refuse a column the frame does not have, naming it and listing what is there.

    Checked in the order written, so the first mistake in the line is the one reported. Listing the
    available names is the whole point: an author who misspells a column cannot see the frame, and
    a message that only says "unknown column" leaves them guessing.
    """
    available = set(columns)
    for name in referenced:
        if name not in available:
            raise _error(
                text,
                f"[{name}] is not a column in the data: the columns are {', '.join(sorted(available))}",
            )


def _unpublished(what: str, named: str, published: Collection[str]) -> ExpressionError:
    """The refusal for a dialect or a placeholder style nobody has run this compiler against.

    No expression is quoted, because the mistake is not in one: it is in the call, and the caller
    is a tool rather than an author. What the message has to carry instead is the published list,
    since the whole answer to "which do I name?" is the set of names that have been measured.
    """
    return ExpressionError(
        f"{named!r} is not a {what} this compiler has been run against, so it has no measured spelling: "
        f"the {what}s are {', '.join(sorted(published))}"
    )


def _error(text: str, problem: str) -> ExpressionError:
    """One refusal: what is wrong, and the expression it is wrong in.

    The expression is quoted in every message because these are read in a log, hours later, by
    someone holding a file of many expressions and needing to know which line to open. Long text is
    truncated so that a pathological input cannot turn one refusal into a page of log.
    """
    quoted = text if len(text) <= 120 else f"{text[:120]}..."
    return ExpressionError(f"{problem}, in expression: `{quoted}`")
