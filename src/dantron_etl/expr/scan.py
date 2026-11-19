"""The first pass over an expression: the author's syntax rewritten into text the parser reads.

Column references and quoted values are lifted out into tables and replaced by numbered calls, so
no character the author typed reaches the parser. Numbers are held to the spellings this language
has, and dates are read here too.
"""

import math
from datetime import date
from typing import Any, Final

from .errors import error


# The scan's own output vocabulary. A column reference and a quoted literal are both rewritten to a
# call on one of these names carrying nothing but an integer index into a table the scan built, so
# the text handed to the parser contains no author-supplied characters at all. These names are
# refused in the incoming text (see ``scan``) precisely so that the only way to produce one is to
# write ``[name]``, ``'text'`` or ``date'...'`` -- the internal spelling is unreachable by hand.
COLUMN_FN: Final[str] = "col"
LITERAL_FN: Final[str] = "lit"
_RESERVED_CALLS: Final[frozenset[str]] = frozenset({COLUMN_FN, LITERAL_FN, "date"})

# A quote may be preceded by ``date`` and nothing else. Every other prefix Python would accept is
# named here so the refusal can say what the author reached for.
_QUOTE_PREFIX_HINTS: Final[dict[str, str]] = {
    "f": "an f-string (JoinedStr / FormattedValue) is not part of this language",
    "b": "a bytes literal (Bytes) is not part of this language",
    "r": "a raw-string prefix is not needed, because quoted text is already taken verbatim",
    "u": "a unicode prefix is not supported, because quoted text is already unicode",
}

_DIGITS: Final[str] = "0123456789"
_QUOTED_TEXT_ADVICE: Final[str] = "write text in single quotes, like 'PAID'"


# Characters outside quotes and brackets that are never part of this language, with what to write.
# '#' would otherwise start a comment in the parser underneath and drop the rest of the line, and a
# backslash would join two lines -- both silently.
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

            # col and lit are the scan's own names, so they are refused wherever they appear, not only
            # before '(' -- '(lit)(0)' would otherwise reach the parser as a call.
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
            # ``!=``, ``<=`` and ``>=`` already mean what they say; only a lone '=' is a comparison
            # in this language and has to be doubled for the parser.
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
            raise error(text, "a quote is never closed, so the quoted text has no end")
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
    """Refuse a decimal that does not survive as the number written.

    Both directions cost money, and both are silent. ``1e400`` arrives as infinity, so
    ``[amount] < 1e400`` keeps every row for an author who typed one character too many. Its mirror
    ``1e-400`` arrives as exactly zero, which turns a tolerance into "anything at all". Exponent
    notation itself stays legal, because ``1e-9`` is how a tolerance is written and the alternative
    spelling is a row of zeros a reviewer has to count.
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
    """A calendar date from ``date'YYYY-MM-DD'``.

    The prefix-tagged form is deliberate. A postfix cast (``'2026-01-01'::date``) would introduce a
    cast operator, and the obvious next request after one cast is every other cast -- growing the
    grammar, which this design refuses to do. Tagging keeps a date a literal.
    """
    try:
        return date.fromisoformat(raw)
    except ValueError:
        raise error(text, f"date'{raw}' is not a calendar date, so write a date as date'2026-01-01'") from None


def _next_visible(text: str, index: int) -> str:
    """The next character from ``index`` that is not whitespace, or ``''`` at the end of the text."""
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index : index + 1]
