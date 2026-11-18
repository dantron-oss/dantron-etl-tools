"""Helpers a tool calls to check its configuration against the frames it was given.

A tool's ``validate_inputs`` and ``execute`` see the configuration and the frames together. The
helpers here answer the questions most tools ask at that point: does this column exist on this
port, does this expression compile against these columns, what type does it produce, may this step
write a column of this name. Each one refuses in the library's own words, naming the field.

Everything here answers a question that needs both halves in front of it -- the configuration an
author wrote, and the schema of the frame that turned up -- which is why none of it can live in a
template and none of it can wait for ``execute``. Each one exists because several tools were
already asking it, and the third or fourth spelling of a question is where the answers diverge.
"""

from collections.abc import Sequence
from typing import Final

import polars as pl

from .config import ConfigField
from .errors import ConfigError
from .expr import ExpressionError, compile_expr

__all__ = [
    "REPLACE_FIELD",
    "check_output_column",
    "compile_for",
    "compile_predicate_for",
    "compiled_dtype",
    "require_columns",
]


def compile_for(frame: pl.DataFrame, text: str, field: str) -> pl.Expr:
    """Compile ``text`` against ``frame``'s schema, for a tool's own ``execute``.

    Every tool that declares an ``expression`` field ends up doing the same two things at the
    boundary between its configuration and the frame it runs against, and each is the kind of thing
    that is easy to get right once and easy to forget once per tool: pass the frame's *schema*
    (name to dtype), not merely its column names, because a schema is what lets a misspelled column
    and a function applied to the wrong dtype both fail here, in the author's words, rather than one
    of them surviving to fail three layers down inside the query engine; and translate
    :class:`~.expr.ExpressionError` into this module's own error family, because a tool's caller
    already catches :class:`ToolError` and an expression failure that arrived as a bare
    ``ExpressionError`` would slip past it -- :func:`check_tool`'s own example runner is exactly such
    a caller, so a worked example with a bad predicate would otherwise escape catalog conformance
    entirely instead of being reported as a drifted example.

    ``field`` is the config key ``text`` came from, so the refusal names which field to open, and
    the offending text is quoted alongside it for the same reason every refusal in this module and
    in :mod:`~.expr` quotes the text it is about: this is read later, in a log, by someone holding a
    file of many expressions.
    """
    try:
        return compile_expr(text, frame.schema)
    except ExpressionError as error:
        raise ConfigError(f"config field {field!r} does not compile: {error}") from error


def compiled_dtype(frame: pl.DataFrame, text: str, field: str) -> tuple[str, pl.DataType]:
    """What ``text`` produces against ``frame``'s schema: the column's name, and its dtype.

    Asked by every tool that has to decide something about an expression before it runs it -- a
    condition must be a truth value, the branches of an ordered rule must agree with one another, a
    date part must be a whole number -- and answered off the *schema*, so the answer is the same on
    a frame of a million rows and a frame of none. That property is not incidental: a tool that
    refuses a whole run on the strength of this must not start behaving differently once a frame
    happens to arrive empty.

    The pair, rather than the dtype alone, because every caller that refuses a dtype has to name the
    column in its refusal. The pair, rather than a whole schema, because a compiled expression
    produces exactly one column, and handing back a mapping would make each caller re-derive "the
    first and only entry" -- which is the line this function exists to delete.
    """
    return _produced(frame, compile_for(frame, text, field), text, field)


def compile_predicate_for(frame: pl.DataFrame, text: str, field: str) -> pl.Expr:
    """:func:`compile_for`, plus the refusal that what it compiled is a condition.

    ``[amount]`` is what an author writes when they meant ``[amount] > 0``. It compiles, because
    that is exactly how a column reference is spelled, and handed to the query engine as a mask it
    fails there instead -- as a complaint about a dtype, from a library the author has never heard
    of, naming neither the field to open nor the text to change. A bare truth column, by contrast,
    *is* a predicate as written, because a plain status filter is one of the real uses of every
    filtering tool: the question is what the expression produces, never what it looks like.

    The refusal names the column, the config field, the offending text and the dtype it produced
    instead, because all four are what a reader needs while holding a file of many expressions.
    """
    expression = compile_for(frame, text, field)
    column, dtype = _produced(frame, expression, text, field)
    if dtype != pl.Boolean:
        raise ConfigError(
            f"config field {field!r} does not give a truth value: column {column!r} holds "
            f"{dtype}, not a boolean, in expression: `{text}`"
        )
    return expression


def _produced(frame: pl.DataFrame, expression: pl.Expr, text: str, field: str) -> tuple[str, pl.DataType]:
    """Resolve one compiled expression against a schema, in this module's error family.

    The resolution is asked of the query engine, which is entitled to raise its own exception types
    at this point as much as anywhere else, so this is a translating boundary like every other place
    this module reaches into it -- otherwise a probe would hand a caller catching
    :class:`ToolError` something it has never heard of.
    """
    try:
        produced = frame.lazy().select(expression).collect_schema()
    except pl.exceptions.PolarsError as error:
        raise ConfigError(
            f"config field {field!r} cannot be applied to these columns: {type(error).__name__}: "
            f"{error}, in expression: `{text}`"
        ) from error
    name, dtype = next(iter(produced.items()))
    return name, dtype


def require_columns(frame: pl.DataFrame, names: Sequence[str], field: str, port: str) -> None:
    """Refuse a configuration naming a column that is not on the port it will be read from.

    Every tool whose configuration names columns asks this, and each one that spelled its own
    version spelled it differently. ``port`` is the reason a shared version is worth more than four
    private ones: a single-input tool passes its one port name, a pairing tool passes ``left`` or
    ``right``, and "column not found" without a side is a refusal that sends the reader to both.

    Listing the columns that *are* present is the other half, and it is what turns the refusal into
    the answer rather than merely the complaint: the two mistakes made here are a misspelling and a
    wrong case, and both are obvious the moment the real list sits beside them.
    """
    present = frame.columns
    absent = [name for name in names if name not in present]
    if absent:
        raise ConfigError(
            f"config field {field!r} names column(s) {', '.join(absent)} that are not on input port "
            f"{port!r}; the columns on {port!r} are: {', '.join(present)}"
        )


# The one canonical declaration of the flag that answers :func:`check_output_column`. Shared so the
# wording is identical on every catalog page rather than reinvented per tool, and carrying no
# default on purpose: a default would be inert wherever this field sits inside an array element, so
# absent means "not asked", which is the reading the check takes.
REPLACE_FIELD: Final[ConfigField] = ConfigField(
    type="boolean",
    description="Whether writing this column may overwrite a column of that name that is already "
    "there; leave it out and a name already in use is refused.",
)


def check_output_column(frame: pl.DataFrame, name: str, field: str, *, replace: bool | None) -> None:
    """Refuse a name that is not a name, refuse writing ``name`` into ``frame`` silently, and refuse
    repairing a column that is absent.

    A blank name is ruled on first, and it is ruled on here rather than left to each tool. Blank
    means empty or nothing but whitespace. A column called ``"   "`` is invisible in a rendering,
    sits in a header row looking exactly like its neighbours, and cannot be referred to at all in an
    expression, because ``[   ]`` is not something an author can write and mean. Measured across the
    catalog before this check existed, seven tools were asked and gave three different answers, so the
    rule sits in the one place every output-naming tool already reaches: a tool written next year
    inherits it by calling this function, which it has to call anyway for the collision rule below.
    Nothing is trimmed or tidied, only refused -- ``" total "`` is a strange name and still a name,
    and a rule that quietly reshaped it would be writing a column nobody asked for.

    The blank is refused before ``replace`` is read, deliberately. A name made of nothing is wrong
    either way, and reading the flag first is what used to give one mistake two unrelated
    explanations: under ``replace: true`` a blank was reported as a column the input does not carry,
    which is a sentence about the frame and sends the author to the wrong file.

    A field that declares ``non_empty`` in its own template is answered earlier than this and more
    tersely -- ``must not be empty``, at pipeline load, with no frame in front of it -- so ``""`` on
    such a field never reaches here and only whitespace does. That fork in the wording is kept on
    purpose. The load-time refusal is the better of the two: it needs no frame to be right, and it
    catches a tool whose ports were never wired. Moving it here to make two correct messages read
    alike would trade a refusal at load for one at run. This function is the backstop for the fields
    that do not declare it, which is every field whose tool author never thought to.

    The rest is the question about a name already in use, where the answer is neither of the two
    obvious ones. Refusing replacement outright is too strict, because repairing a column in place
    is a legitimate step. Taking the name silently is the real hazard, because whoever writes one
    step does not know all the columns of the frame they are writing into, and the value they
    overwrote is gone.

    So both directions are refused. A collision nobody asked for names the column, the field and the
    flag, so the fix is one keystroke. And ``replace`` on a column that is not there is refused too,
    because unchecked it is the subtler defect of the two: a misspelling quietly creates a new
    column while the column the author meant to repair keeps its old values, and the run reports
    success having changed nothing.
    """
    if not name.strip():
        # The offending value is not quoted back: three spaces and an empty string print the same,
        # so repeating it tells the reader nothing and would fork one refusal into six.
        raise ConfigError(
            f"config field {field!r} must not be blank: a column name has to carry at least one "
            f"character that is not whitespace, or nothing can point at the column it writes"
        )

    present = name in frame.columns
    if replace and not present:
        raise ConfigError(
            f"config field {field!r} asks to replace column {name!r}, which the input does not "
            f"have; replacing a name nothing carries would write a new column and leave the one "
            f"meant for repair untouched. The columns present are: {', '.join(frame.columns)}"
        )
    if present and not replace:
        raise ConfigError(
            f"config field {field!r} names output column {name!r}, which the input already carries; "
            f"say replace: true to overwrite it deliberately, or write to a name nothing uses"
        )
