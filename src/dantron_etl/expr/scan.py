"""The first pass over an expression: the author's syntax rewritten into text the parser reads.

Column references and quoted values are lifted out into tables and replaced by numbered calls, so
no character the author typed reaches the parser. Numbers are held to the spellings this language
has, and dates are read here too.
"""

import math
import re
from datetime import date
from typing import Any, Final, NamedTuple

from .errors import error


# The scan's own call names. [name] and 'text' become col(n) and lit(n), indexes into tables the
# scan builds. These names are refused in the author's text, so they cannot be written by hand.
COLUMN_FN: Final[str] = "col"
LITERAL_FN: Final[str] = "lit"
_RESERVED_CALLS: Final[frozenset[str]] = frozenset({COLUMN_FN, LITERAL_FN, "date"})

# Only `date` may come before a quote; the other prefixes Python accepts are named for the refusal.
_QUOTE_PREFIX_HINTS: Final[dict[str, str]] = {
    "f": "an f-string (JoinedStr / FormattedValue) is not part of this language",
    "b": "a bytes literal (Bytes) is not part of this language",
    "r": "a raw-string prefix is not needed, because quoted text is already taken verbatim",
    "u": "a unicode prefix is not supported, because quoted text is already unicode",
}

_DIGITS: Final[str] = "0123456789"
_QUOTED_TEXT_ADVICE: Final[str] = "write text in single quotes, like 'PAID'"


# Characters never allowed outside quotes and brackets, with what to write instead. '#' would start
# a comment and a backslash would join lines, both silently.
_REFUSED_CHARACTERS: Final[dict[str, str]] = {
    "#": "'#' is not a comment in this language, and everything after it would be ignored: put a note in "
    "the step's why instead",
    "\\": "a backslash outside quotes is not part of this language",
    "\u2018": "a curly quote is not part of this language, so write text in straight single quotes, like 'PAID'",
    "\u2019": "a curly quote is not part of this language, so write text in straight single quotes, like 'PAID'",
    "\u201c": "a curly quote is not part of this language, so write text in straight single quotes, like 'PAID'",
    "\u201d": "a curly quote is not part of this language, so write text in straight single quotes, like 'PAID'",
}


class Scanned:
    """The result of one forward scan: parseable text, plus the tables its indices point into.

    ``origins`` gives, for each character of ``source``, where it came from in the author's text, so
    a syntax error found in ``source`` can be pointed at in the text the author wrote.
    """

    __slots__ = ("columns", "literals", "origins", "source")

    def __init__(
        self,
        source: str,
        columns: tuple[str, ...],
        literals: tuple[Any, ...],
        origins: tuple[int, ...] = (),
    ) -> None:
        self.source = source
        self.columns = columns
        self.literals = literals
        self.origins = origins


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


def scan(text: str) -> Scanned:
    """Rewrite the author's syntax in one forward pass, tracking whether it is inside a quoted value.

    Quoted text and column names are lifted into tables and replaced by numbered calls, so no character
    the author typed reaches the parser. A number passes through as characters, so it is checked here
    against the spellings this language allows.
    """
    pieces: list[str] = []
    origins: list[int] = []
    columns = _Table()
    literals = _Table()
    index = 0
    length = len(text)

    def emit(piece: str, start: int, verbatim: bool = False) -> None:
        pieces.append(piece)
        origins.extend(range(start, start + len(piece)) if verbatim else [start] * len(piece))

    while index < length:
        char = text[index]

        refused = _REFUSED_CHARACTERS.get(char)
        if refused is not None:
            raise error(text, f"{refused} (character {index + 1})")
        if not char.isascii():
            raise error(
                text,
                f"{char!r} (character {index + 1}) is not part of this language: outside quotes and square "
                f"brackets an expression is written in plain ASCII letters, digits and operators",
            )

        if char == "'":
            start = index
            value, index = _read_quoted(text, index)
            emit(f"{LITERAL_FN}({literals.intern(value)})", start)
            continue

        if char == '"':
            raise error(text, f"double quotes are not part of this language, so {_QUOTED_TEXT_ADVICE}")

        if char == "[":
            end = text.find("]", index + 1)
            if end == -1:
                raise error(text, "a '[' has no matching ']', so the bracket around a column name is unbalanced")
            name = text[index + 1 : end].strip()
            if not name:
                raise error(text, "'[]' is an empty column reference, so name a column between the brackets")
            emit(f"{COLUMN_FN}({columns.intern(name)})", index)
            index = end + 1
            continue

        if char == "]":
            raise error(text, "a ']' has no matching '[', so the bracket around a column name is unbalanced")

        if char in _DIGITS:
            start = index
            span, index = _read_number(text, index)
            emit(span, start, verbatim=True)
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
                    raise error(text, f"an unsupported literal prefix {word!r} appears before quoted text: {hint}")
                start = index
                raw, index = _read_quoted(text, end)
                emit(f"{LITERAL_FN}({literals.intern(_parse_date(text, raw))})", start)
                continue

            # Refused everywhere, not only before '(': '(lit)(0)' would otherwise parse as a call.
            if word in (COLUMN_FN, LITERAL_FN) or (word in _RESERVED_CALLS and _next_visible(text, end) == "("):
                raise error(
                    text,
                    f"{word!r} is reserved for this language's own use and cannot be called: write a column as "
                    f"[name], text as 'value', and a date as date'2026-01-01'",
                )

            emit(word, index, verbatim=True)
            index = end
            continue

        if char == "=":
            if text[index + 1 : index + 2] == "=":
                raise error(text, "'==' is not how equality is written here, so use a single '=' instead")
            # Only a lone '=' needs doubling; '!=', '<=' and '>=' are already right.
            emit("=" if index > 0 and text[index - 1] in "!<>" else "==", index)
            index += 1
            continue

        if char == "!":
            if text[index + 1 : index + 2] != "=":
                raise error(text, "'!' on its own is not an operator here, so write '!=' to test for inequality")
            emit("!", index)
            index += 1
            continue

        emit(char, index)
        index += 1

    return Scanned("".join(pieces), tuple(columns.values), tuple(literals.values), tuple(origins))


def _read_quoted(text: str, quote_at: int) -> tuple[str, int]:
    """The text a quoted value holds, and the index just past its closing quote.

    A doubled quote is one quote, as in SQL and Excel: 'O''Brian' is O'Brian. Undone here, because Python
    would read two adjacent literals as one joined string. `''` alone is still the empty string.
    """
    parts: list[str] = []
    index = quote_at + 1
    length = len(text)
    while True:
        end = text.find("'", index)
        if end == -1:
            raise error(text, "a quote is never closed, so the quoted text has no end")
        parts.append(text[index:end])
        if end + 1 < length and text[end + 1] == "'":
            parts.append("'")
            index = end + 2
            continue
        return "".join(parts), end + 1


def _read_number(text: str, start: int) -> tuple[str, int]:
    """One number, in this language's only shapes: digits, a point, an exponent.

    `0x10`, `1_000`, or anything touching the end of a number is refused rather than reinterpreted.
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
        # Quote the whole run typed (0x10), not just the first bad character.
        end = index
        while end < len(text) and (text[end].isalnum() or text[end] in "_."):
            end += 1
        raise error(
            text,
            f"'{text[start:end]}' is not a number this language reads: a whole number is written in digits alone and a "
            f"decimal as 1.5 or 1e-9, so another base (0x10), a digit separator (1_000) or a trailing letter is "
            f"refused rather than reinterpreted",
        )
    if "." in span or "e" in span or "E" in span:
        _check_decimal(text, span)
    return span, index


def _check_decimal(text: str, span: str) -> None:
    """Refuse a decimal that does not survive as the number written: `1e400` (infinity) and `1e-400` (zero).

    Exponents stay legal, because `1e-9` is how a tolerance is written.
    """
    value = float(span)
    if not math.isfinite(value):
        raise error(
            text,
            f"the decimal {span} is too large to hold, so it would arrive as infinity: a comparison against it would "
            f"match every row",
        )
    mantissa = span.split("e")[0].split("E")[0]
    if value == 0.0 and float(mantissa) != 0.0:
        raise error(
            text,
            f"the decimal {span} is too small to hold, so it would arrive as 0.0: a tolerance written this way would "
            f"accept anything at all",
        )


def _parse_date(text: str, raw: str) -> date:
    """A calendar date from `date'YYYY-MM-DD'`, a tagged literal rather than a cast operator."""
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise error(text, f"date'{raw}' is not a calendar date, so write a date as date'2026-01-01'") from None


def _next_visible(text: str, index: int) -> str:
    """The next character from ``index`` that is not whitespace, or ``''`` at the end of the text."""
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index : index + 1]


# A row reference, read only by the multi_row_formula tool: [Row-1:amount] is 'amount' one row up.
# Every other tool reads the same brackets as a column of that whole name, and refuses it.
ROW_REACH: Final[int] = 1000
_ROW_REFERENCE: Final[re.Pattern[str]] = re.compile(r"Row\s*([+-])\s*([0-9]+)\s*:(.*)", re.DOTALL)


class RowReference(NamedTuple):
    """One ``[Row-N:column]`` or ``[Row+N:column]``: the name as written, the rows away, and the column.

    ``offset`` is negative for a row above and positive for a row below.
    """

    written: str
    offset: int
    column: str


def read_row_reference(text: str, name: str) -> RowReference | None:
    """``name`` read as a row reference, or None when it is a plain column name.

    ``name`` is what stood between the brackets, as the scan lifted it. A reach of 0 rows, a reach
    past ``ROW_REACH``, and a reference that names no column are refused.
    """
    found = _ROW_REFERENCE.fullmatch(name)
    if found is None:
        return None
    sign, digits, column = found[1], found[2], found[3].strip()
    rows = int(digits)
    if not column:
        raise error(text, f"[{name}] names no column after the ':', so write the column to read, like [Row{sign}1:amount]")
    if rows == 0:
        raise error(
            text,
            f"[{name}] reads Row{sign}0, which is the current row: write [{column}] for the current row, or a "
            f"reach from Row{sign}1 to Row{sign}{ROW_REACH} for another row",
        )
    if rows > ROW_REACH:
        raise error(
            text,
            f"[{name}] reaches {rows} rows away, past the limit of {ROW_REACH}: a row reference reads from 1 to "
            f"{ROW_REACH} rows up or down",
        )
    return RowReference(name, rows if sign == "+" else -rows, column)
