"""The kinds of argument a function takes, and how each one is checked.

A value argument is worked out for each row, and must belong to a type family. Every other kind is
written in the file as a literal: a whole number, quoted text, a regular expression pattern, or a
UUID namespace. Each of those is read and checked before any row is.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import polars as pl

from .errors import Refusal
from .families import Family

# A decimal place or a capture group beyond a few thousand has no meaning, and the engine rejects
# one with an error of its own type, which would escape this package. See the bounds in compiler.py.
_SMALL_NUMBER_LIMIT: Final[int] = 65_535

# How a refusal names an argument by where it sits.
POSITIONS: Final[tuple[str, ...]] = ("first", "second", "third", "fourth")

# Why an argument has to be quoted text written in the file. Three wordings, because the three
# arguments that carry this rule worry about different things: one is what the data becomes, one is
# how the data is spelled, and one is which key space the data lands in.
REWRITE_REASON: Final[str] = (
    "a rewrite whose replacement came out of the data would let the data decide what the data becomes"
)

SEPARATOR_REASON: Final[str] = (
    "a separator is a decision written in the file, the way the rest of a report's spelling is, and never a value a "
    "row supplies -- figures that chose their own separators would be deciding how they read"
)

_NAMESPACE_REASON: Final[str] = (
    "a namespace is what keeps one flow's ids from colliding with another's, so a namespace out of the data would "
    "let the data choose which key space its own id lands in -- and it could not be checked until the row that is "
    "malformed, by which time the ids before it have been written"
)

# Shown to an author whose namespace did not parse, so that the refusal carries the shape as well
# as the complaint: lowercase hex, hyphenated 8-4-4-4-12. The same one the manual uses, because an
# author who meets it twice should meet it spelled identically both times.
_UUID_EXAMPLE: Final[str] = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


def probe_pattern(name: str, pattern: str, build: Callable[[pl.Expr], pl.Expr]) -> None:
    """Compile one pattern against this engine, in both directions.

    It has to be the engine's own judgement: look-behind is legal in the standard library's
    regular expressions and illegal in this engine, and a unicode script class is the reverse
    -- so validating against the wrong one is worse than not validating, because it either
    passes a pattern that fails at row time or refuses one that works.
    """
    try:
        pl.select(build(pl.lit("", dtype=pl.String)))
    except Exception as failure:
        raise Refusal(f"{name}() was given a pattern the engine cannot compile: '{pattern}' ({failure})") from None


@dataclass(frozen=True)
class Value:
    """An argument worked out for each row: a column, a literal, or another expression.

    ``family`` is the type family the argument must belong to. ``None`` accepts any family, and
    then the arguments have to agree with one another instead, which is what coalesce() needs.
    ``requirement`` finishes the sentence a refusal starts with the function's name, as in
    "lower() works on text". ``whole`` is set where the value must also be a whole number, and
    finishes that refusal's sentence the same way.

    A text argument reaches the function cast to a plain string (see ``_built`` in compiler.py).
    """

    family: Family | None
    requirement: str
    whole: str | None = None


class WholeNumber:
    """A whole number written in the file, between 0 and :data:`_SMALL_NUMBER_LIMIT`."""

    def read(self, name: str, value: Any, position: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _SMALL_NUMBER_LIMIT:
            raise Refusal(f"{name}() needs a whole number between 0 and {_SMALL_NUMBER_LIMIT} as its {position} argument")
        return value


@dataclass(frozen=True)
class QuotedText:
    """Quoted text, for an argument that has to be a decision in the file rather than data.

    The reason travels with the argument rather than with this method, because the argument is
    what an author has to change: a replacement and a separator are refused by the same rule
    and for two different-sounding versions of the same worry.
    """

    reason: str = REWRITE_REASON

    def read(self, name: str, value: Any, position: str) -> str:
        if not isinstance(value, str):
            raise Refusal(f"{name}() needs quoted text as its {position} argument: {self.reason}")
        return value


class Pattern:
    """The pattern for one of the regex functions: quoted text, checked, or nothing.

    A literal and not a column, which is what makes a check before the run possible at all: a
    pattern that varies row by row can be neither read in the file nor validated up front.
    """

    def read(self, name: str, value: Any, position: str) -> str:
        if not isinstance(value, str):
            raise Refusal(
                f"{name}() needs a quoted pattern as its {position} argument, as in {name}([label], '[a-z]+'...): a "
                f"pattern is a fixed decision recorded in the file, so it cannot come out of a column"
            )
        # Ask the engine that will run the pattern whether it compiles, before any row is read.
        probe_pattern(name, value, lambda probe: probe.str.contains(value))
        return value


class Namespace:
    """uuid5()'s first argument: quoted text in the file, and a UUID, checked before any row.

    Parsed rather than pattern-matched, and parsed by the same library that will derive the
    ids, so what is accepted here is exactly what works there -- a rule of this module's own
    would either refuse a namespace that derives perfectly good ids or pass one that fails at
    row time, and both defeat the point of checking at all.
    """

    def read(self, name: str, value: Any, position: str) -> uuid.UUID:
        written = QuotedText(_NAMESPACE_REASON).read(name, value, position)
        try:
            return uuid.UUID(written)
        except ValueError as failure:
            raise Refusal(
                f"{name}() was given a namespace that is not a UUID: {written!r} ({failure}) -- write one as "
                f"'{_UUID_EXAMPLE}'"
            ) from None


Param = Value | WholeNumber | QuotedText | Pattern | Namespace
