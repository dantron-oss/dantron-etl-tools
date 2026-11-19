"""A small expression language: author-written text in, one column expression out.

A tool's configuration is a file on disk. A filter needs a predicate, a derived column needs a
formula, and neither can be a callable, because the person writing the file is not writing code --
they are writing ``[status] = 'PAID' and [amount] > 0``. This package is the one place that turns
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

Each module does one job, and they import one way, from the bottom of this list to the top:

* :mod:`.compiler` walks a checked tree and builds the column expression;
* :mod:`.functions` defines every function, once;
* :mod:`.arguments` reads and checks the arguments a function is given;
* :mod:`.spelling` reads the scanned tree back the way the author wrote it;
* :mod:`.parse` turns text into a checked tree, and holds the limits and the node whitelist;
* :mod:`.scan` rewrites the author's syntax before parsing;
* :mod:`.families` works out what a value holds;
* :mod:`.formatting` writes a number as text for format_number();
* :mod:`.quoting` spells a value as text of this language;
* :mod:`.errors` holds :class:`ExpressionError` and the wording of a refusal.
"""

from collections.abc import Collection, Mapping

import polars as pl

from .compiler import Division, build
from .compiler import divisions as _divisions
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
    "compile_expr",
    "divisions",
    "escape_text",
    "quote_text",
    "references",
]


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
    expression, _ = build(text, columns)
    return expression


def divisions(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None = None) -> tuple[Division, ...]:
    """Every '/' in ``text``, in the order written: the divisor as the author wrote it, and its column.

    A tool that runs an expression asks this so it can stop on a row whose divisor is zero (see
    ``refuse_division_by_zero`` in ``frame_checks``). The expression is checked exactly as
    :func:`compile_expr` checks it, so a bad one is refused here in the same words.
    """
    return _divisions(text, columns)


def references(text: str) -> frozenset[str]:
    """The column names ``text`` reads, for a caller that wants to check them before running.

    This runs the *same* validation as :func:`compile_expr`, and raises on anything invalid rather
    than reporting the names it managed to find. A lenient version would let a caller pre-flight an
    expression successfully and then fail on compilation later -- the exact disagreement between
    check and execution that this function exists to prevent.
    """
    _, names = build(text, None)
    return names
