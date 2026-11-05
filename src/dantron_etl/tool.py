"""The tool contract: a named-frame envelope, a described unit of work, and a catalog of them.

A pipeline is built from small steps -- keep some rows, pair two sets of rows, and so on -- and this
module is where the shape every step must share is fixed once, rather than reinvented per step.
Three problems are solved here, each because the alternative has a specific failure mode:

**Ports** replace a bare ``dict`` of frames in and out. A step may have more than one output -- the
rows that matched a condition and the rows that did not, say -- and a side output that a caller
forgets to read should not surface three steps later as a bare ``KeyError``. So every input and
output is named, the set of names a step declares is fixed at authoring time, and the envelope is
checked at the boundary of every call: a declared port that a step forgot to produce, an output
nobody declared, an output with no columns at all (which would make a downstream check that depends
on the column existing fail for the wrong reason), a step that handed back something other than
ports, and a step that wrote into one of its own inputs are all refused right there, by name.

A step may also send back a **per-row status** beside a port: one row per row of that port, in the
same order, saying which rows it degraded and why. That is the half a note cannot carry. A note says
595 values would not read as a number, which is a quantity and nothing else -- it cannot say whether
those were figures the run destroyed or text that was never a figure, and the output column cannot
say either, because after the step every one of them is null. The status can, so the sentence and
the rows behind it travel together. It rides in the envelope rather than in the frame for the same
reason a note does: what a step did is not a fact about the data, and a column would flow into the
next join and out into the published workbook. Almost every step reports none, and the boundary
refuses one that says only that every row is fine.

**A tool** is the unit that reads ports and writes ports: a machine name, a one-line description, a
category drawn from a published set, detailed help carrying worked examples, a typed configuration
template, a way to validate whatever a template alone cannot express, and the work itself. The
template itself is enforced by the boundary rather than by each tool, because a tool with no
cross-field rules has no reason to write anything of its own, and a rule that holds only when an
author remembers to apply it is a convention rather than a rule. A configuration field can declare
itself an *expression*, which is the seam to the small expression language this package also ships:
a caller that loads a pipeline definition can lift the column references out of a field like that
and check them against an upstream frame before a single row runs through it. A field can also
describe the shape *below* itself -- an array's element type, an object's members -- because the
common configuration shape is an ordered list of small objects, and a seam that reaches only the
top level of such a configuration reaches nothing at all.

**A registry** is a catalog resolved by name, and a catalog is only as trustworthy as its examples
are true. So this module also ships the check that keeps it that way: every worked example a tool
documents is actually run, and its output is compared against what the tool claims it produces. An
example that has quietly drifted from its tool is worse than no example, because someone will read
it and copy it.

**One error family** carries all of it, and it says five different things: the configuration is
wrong, the wiring is wrong, the catalog is wrong, the *data* is wrong, or something nobody expected
broke and was wrapped on the way out. The fourth is worth naming separately because a reader's next
move differs -- a data fault means look at the data, an unexpected failure means look at the run,
and the latter may be wrapping a problem the machine had rather than anything a step decided. Every
one of the five descends from a single base, so a flow driver has one type to catch.

**The boundary calls into a tool's own code four times per run**, in an order that is part of the
contract: build the template, validate the configuration, check that configuration against the
schemas of the frames that actually arrived, and only then run. The third is the least obvious and
was the last to exist: neither a template nor a cross-field rule has a frame in front of it, so
"does the column this configuration names exist, on the port it will be read from?" had nowhere to
live but the first lines of every tool's own ``execute`` -- which is a rule that holds while every
author remembers it, and therefore not a rule.

**A worked example's world** is a directory this module opens for the run, not a path frozen onto
the page. A tool that touches the outside world -- a reader that needs a file already there, a
writer that needs somewhere real to put one -- used to have that path frozen into its example's own
configuration before any check ever entered, because :attr:`WorkedExample.setup` builds its world
too late for the example's own fields to see it. The generated catalog showed a reader
``path: /tmp/etl_worked_example_figures.xlsx``, for exactly the third of it that touches disk. The
``{scratch}`` token is what an example may hold instead: :func:`check_tool` opens a fresh directory
for the run, resolves the token onto a private copy of the configuration, and puts the directory
away again before returning -- the stored configuration a rendered page reads is never touched, only
the copy the run itself sees. What this does not do is watch the filesystem at large: a file written
anywhere other than inside that directory is exactly as invisible as it always was, the same
admission the process-disturbance check below already makes about the environment and the working
directory.
"""

import contextvars
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

import polars as pl
from polars.testing import assert_frame_equal

from .expr import ExpressionError, compile_expr

__all__ = [
    "CATEGORIES",
    "FIELD_TYPES",
    "MAX_REPORTED_ROWS",
    "REGISTRY",
    "REPLACE_FIELD",
    "SCRATCH_TOKEN",
    "STATUS_LEVELS",
    "STATUS_REASON",
    "STATUS_SCHEMA",
    "STATUS_VALUE_CHARS",
    "YAML_UNSAFE_VALUES",
    "ConfigError",
    "ConfigField",
    "ConformanceError",
    "DataError",
    "ExecutionError",
    "PortError",
    "Ports",
    "RegistryError",
    "Tool",
    "ToolError",
    "ToolHelp",
    "ToolRegistry",
    "WorkedExample",
    "apply_defaults",
    "check_output_column",
    "check_registry",
    "check_tool",
    "compile_for",
    "compile_predicate_for",
    "compiled_dtype",
    "expression_fields",
    "expression_values",
    "require_columns",
    "row_status",
    "run_tool",
    "scratch_path",
    "temporary_environment",
    "unreadable_report",
    "validate_against_template",
    "value_caption",
]


class ToolError(ValueError):
    """The base of every error this module raises, mirroring :class:`~.expr.ExpressionError`.

    A ``ValueError`` subclass so that a caller already treating configuration problems as
    ``ValueError`` keeps working without knowing any of the more specific types below exist.
    """


class PortError(ToolError):
    """An input or output port was missing, undeclared, unnamed, or untyped when it should not be."""


class ConfigError(ToolError):
    """A configuration does not satisfy its tool's template, or a rule the template cannot express."""


class RegistryError(ToolError):
    """A tool could not be registered or resolved by name."""


class ConformanceError(ToolError):
    """A tool does not satisfy the catalog's rules -- undocumented, undeclared, or drifted."""


class ExecutionError(ToolError):
    """A tool's own code failed with something other than a deliberate refusal, wrapped so it stays in the family.

    Every tool in a catalog reaches into the query engine somewhere, and the engine raises its own
    exception types -- a ``SchemaError``, an ``InvalidOperationError`` -- that are not part of this
    module's family. Asking each tool to catch and translate its own calls is a convention, not a
    rule, and the first tool that forgets breaks the promise that a flow driver has one exception
    family to catch. So the boundary translates instead, and this is what it translates into.

    That consistency has a cost that must stay visible: this error may be wrapping a failure that
    has nothing whatsoever to do with the tool that raised it -- an allocation the machine could not
    meet, a frame that could not be read, a resource problem far below anything the tool decided.
    The family deliberately does not distinguish that case from an actual defect in the tool's own
    logic, because a hand-maintained list of "not really the tool's fault" exception types would
    always have a gap nobody noticed until it mattered. A driver that wants to continue past one of
    these rather than stop must decide by looking at ``__cause__``, never by the fact that it caught
    ``ExecutionError``.
    """


# How many offending rows a refusal about data prints. Deliberately small: the message is read
# later, in a log, and an invariant that fails on every row of a large frame must not put that frame
# into it. The counts are reported separately and are never capped, so nothing about the size of the
# problem is lost by capping the illustration of it.
MAX_REPORTED_ROWS: Final[int] = 5


class DataError(ToolError):
    """The rows themselves were wrong -- a deliberate refusal about values, not about a set-up.

    Deliberately not a kind of :class:`ExecutionError`, in either direction, because a reader's next
    move differs: this one means go and look at the data, where an execution fault means go and look
    at the run -- and that fault may be wrapping something the machine did rather than anything a
    step decided, where this one never is. That is also why the boundary hands this error back
    exactly as it was raised instead of wrapping it: wrapped, it would announce that a tool "failed
    while running" and send the reader to the run, hiding the one message that mattered.

    The offending rows are reduced to two counts and a few example rows here, in the constructor,
    and the frame itself is not kept. Both halves are load-bearing. The counts and the examples are
    what make the refusal actionable for somebody who was not there when it ran -- "3 of 5 rows,
    starting with ..." sends them to the rows, where "an invariant failed" sends them to a query
    they have to write first. And retaining the frame would keep it alive for as long as the
    exception is reachable, which for an exception travelling out of a long flow is the rest of the
    run: the one way an error type can do real damage.
    """

    def __init__(self, refusal: str, *, offending: pl.DataFrame, total: int) -> None:
        self.offending_rows: int = offending.height
        self.total_rows: int = total
        self.examples: tuple[dict[str, Any], ...] = tuple(offending.head(MAX_REPORTED_ROWS).to_dicts())
        shown = f", starting with {'; '.join(str(row) for row in self.examples)}" if self.examples else ""
        super().__init__(f"{refusal}: {self.offending_rows} of {self.total_rows} row(s) offend{shown}")


# One check per declared type. A boolean is refused by both numeric types because ``bool`` is an
# ``int`` subclass in Python, which is the same trap the expression language had to sidestep for its
# own whole-number arguments; the two must agree, or a template accepts ``true`` where it says it
# takes a count.
_TYPE_CHECKS: Final[dict[str, Any]] = {
    "string": lambda value: isinstance(value, str),
    "expression": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
}

# The field types a template may declare. Kept as a published set, the same way expr.py publishes
# its own whitelists, so that a template author sees exactly what is available rather than
# discovering the boundary by trial and error. "expression" is one of these types rather than a
# separate flag on the field, so a field cannot claim to be an expression in one place and a plain
# string in another. Derived from the checks above rather than written out beside them, so the
# published set and the set that is actually enforced cannot drift apart -- a type in one and not
# the other would be a field that validates nothing while looking as if it does.
FIELD_TYPES: Final[frozenset[str]] = frozenset(_TYPE_CHECKS)

# The groups a rendered catalog is organised into. Published and checked for the same reason as the
# types: assembled tool by tool from whatever each author typed, one group in reality becomes
# "transform", "Transform", "transformation" and "" in the rendering.
CATEGORIES: Final[frozenset[str]] = frozenset(
    {"input", "transform", "combine", "preparation", "text", "output", "assert"}
)

# The types ``non_empty`` can mean anything about. A count of zero or a number of zero are values in
# full, not an absence of one, so ``non_empty`` on either would constrain nothing -- the same silent
# no-op ``_check_spec`` already refuses for ``items`` outside an array and ``fields`` outside an
# object.
_SIZED_TYPES: Final[frozenset[str]] = frozenset({"string", "expression", "array", "object"})

_MACHINE_NAME: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")

# The bare words a definition file written in YAML resolves into another type before this module
# ever sees them, in every spelling such a loader recognises. Published rather than private,
# because the person who needs the list is the tool author choosing what to call an option.
YAML_UNSAFE_VALUES: Final[frozenset[str]] = frozenset(
    {
        spelling
        for word in ("true", "false", "yes", "no", "on", "off", "y", "n", "null")
        for spelling in (word, word.capitalize(), word.upper())
    }
    | {"~", ""}
)

# Text such a loader hands over as a number. A pattern rather than a call to ``float``, so that a
# word like ``nan`` or ``inf`` -- which a loader leaves as text, and either of which is a reasonable
# name for a fill policy -- is not refused for a reason its author could do nothing about.
_NUMERIC_TEXT: Final[re.Pattern[str]] = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")

# What a reader is told about which piece of a tool's own code broke. Named here, once, so the
# three (four, counting help()) call sites that reach into tool-authored code cannot each spell the
# phase differently and so that a reader can search a log for one of these words and land on the
# right part of the boundary.
_PHASE_VERBS: Final[dict[str, str]] = {
    "config_template": "building its configuration template",
    "validate_config": "validating its configuration",
    "validate_inputs": "checking its configuration against the frames it was given",
    "execute": "running",
    "help": "documenting itself",
    "input_ports_for": "working out which input ports it reads",
}


# ---------------------------------------------------------------------------
# Ports -- the envelope
# ---------------------------------------------------------------------------


# What a status row may say about the row it describes. Published rather than left to each tool,
# because the reader of a status is a renderer: a level it has no drawing for shows as neither a
# tick nor a warning nor a cross, so the row shows nothing at all and reads as fine. Three words
# only -- "this row came through", "something about it was degraded", "something about it failed" --
# and the boundary refuses a fourth.
STATUS_LEVELS: Final[tuple[str, ...]] = ("ok", "warning", "error")

# One published shape for every status frame in the library, so nothing reading one has to work out
# per port what it is looking at.
#
# ``reason`` is a *list* because one row can be degraded twice over -- two columns cast in the same
# step, both unreadable on the same row -- and a single reason per row would drop the second in
# silence, sending a reader to the wrong column. It holds structs rather than sentences because the
# question asked of a status is "which rows does this warning cover", answered by matching
# ``reason.column`` exactly; against prose that match is a substring search, which is a different
# question wearing the same answer.
STATUS_REASON: Final[pl.DataType] = pl.List(pl.Struct({"column": pl.String, "detail": pl.String}))
STATUS_SCHEMA: Final[pl.Schema] = pl.Schema({"level": pl.String, "reason": STATUS_REASON})

# How much of an offending value a reason quotes. A reason is read in a table cell beside the row it
# marks, so it is a caption rather than a copy: this is about one line of one, which was enough to
# recognise every value the run that prompted this had emptied. The rest is not lost -- it is on the
# input port, one panel away -- and an uncapped quote would put a whole cell of free text into every
# reason of every marked row, which on a column that failed six hundred times is the frame again.
STATUS_VALUE_CHARS: Final[int] = 60


def value_caption(column: str, *, limit: int = STATUS_VALUE_CHARS) -> pl.Expr:
    """``column``'s value as short quoted text, for a reason that says what was there.

    The whole of what a marked row is asked: *what did this hold before the step emptied it?* An
    output column cannot answer -- every value that failed to read is null afterwards, which is the
    defect this channel closes -- and sending the reader to the input port to find out costs them
    the hunt, and for a step that reshapes rows (an unpivot melts one row into several) there is no
    row to find. So the value is quoted here, where the two sides can be put together honestly.

    Two properties, both load-bearing. It never answers null: a reason built with
    :func:`polars.format` around a null propagates the null through the whole sentence, and a marked
    row whose reason vanished is a row the status calls degraded and cannot say a thing about. And
    it is capped at ``limit`` characters, saying so when it cuts, so the length of a reason does not
    follow the length of whatever text was in the cell.

    Both of those are about a *value*. A column whose type has no text form at all -- a list, a
    struct -- refuses the cast outright rather than answering null, and this refuses with it, which
    is the honest answer: what the reader would be shown is not one value. Nothing reaches here that
    way in practice, because a column is captioned only where a cast to a scalar type has just read
    it, and that cast refuses first.
    """
    text = pl.col(column).cast(pl.String, strict=False)
    shortened = (
        pl.when(text.str.len_chars() > limit)
        .then(pl.format("{}... (shortened from {} characters)", text.str.slice(0, limit), text.str.len_chars()))
        .otherwise(text)
    )
    return (
        pl.when(text.is_null())
        .then(pl.lit("a value that cannot be shown as text"))
        .otherwise(pl.format("'{}'", shortened))
    )


def unreadable_report(
    frame: pl.DataFrame,
    column: str,
    failed: pl.Series,
    declared: str,
) -> tuple[pl.Series, str]:
    """One reason per row for the values of ``column`` that would not read as ``declared``, and the note counting them.

    Written once for the two tools that read a column as a type the flow states. They already answer
    that question in the same words down to the wording of their notes, deliberately, and a second
    spelling of the same sentence is how a reader ends up trusting the rule they learned second.

    The reasons and the note come back from one call because they are two accounts of one fact, and
    the failure worth designing out is them drifting apart: an edit that changed how the rows are
    marked while leaving the count to be worked out a few lines away would leave a reader looking at
    595 marks under a sentence that says something else. Here the count, the share, the example and
    the reasons are all read off ``failed``, so there is nothing left for them to disagree about.

    ``failed`` is the step's own mask, taken before the values were replaced, so this never has to
    re-derive which rows lost something -- afterwards it could not: a null that failed to read and a
    null that was always there are the same null, which is the whole defect this channel closes. A
    blank therefore gets no reason here, because it never failed.

    The note quotes the first offending value, captioned exactly as the reasons caption it, for the
    one sentence a reader sees without opening anything: "595 values failed" sends them looking,
    "595 values like 'Canadian Business Growth Fund' failed" usually settles it where they stand.

    Called with something marked. A mask with nothing set describes no rows, so the note it would
    write counts nothing and quotes nothing, and the all-``ok`` status built beside it is refused at
    the boundary as inert -- which is why both callers ask the mask before they ask for this.
    """
    captioned = frame.select(value_caption(column).alias("caption")).with_columns(failed.alias("failed"))
    details = captioned.select(
        pl.when(pl.col("failed")).then(
            pl.format("held {}, which did not read as {}", pl.col("caption"), pl.lit(declared))
        )
    ).to_series()
    example = captioned.filter(pl.col("failed")).get_column("caption").head(1).to_list()

    lost, total = int(failed.sum()), failed.len()
    # Two decimal places rather than two significant figures. `.2g` is a general format, and the
    # share this sentence reaches most often is exactly the one it renders unreadably: a column that
    # is text from top to bottom fails on every row, and 100 comes out as `1e+02`. A reader who has
    # to decode the number in a warning is back to not trusting the warning.
    share = lost / total * 100 if total else 0.0
    note = (
        f"{lost:,} of {total:,} values in {column!r} ({share:.2f}%) did not "
        f"read as {declared} and are now empty (for example {example[0] if example else None})"
    )
    return details, note


def row_status(marks: Sequence[tuple[str, pl.Series]], height: int) -> pl.DataFrame:
    """A status frame of ``height`` rows, built from one detail column per thing that could degrade.

    A mark is a column name and a series of reasons the same height as the port, holding the
    sentence for a row that was degraded and null for a row that was not. One series rather than a
    mask beside a sentence, because the two would be free to disagree: a row the mask called
    degraded and the sentence had nothing to say about would be marked and unexplained, which is the
    old count again with extra steps.

    Marks are applied in the order they are given, which is the order the step did the work in, so a
    row degraded by two columns reads in the order they were processed rather than in whatever order
    a set iterated.

    With no marks at all this hands back a frame of ``ok``. That is not a status worth publishing --
    the boundary refuses an inert one -- and it is exactly what a caller stacking a status per block
    needs for the blocks that came through clean.

    Every mark is measured against ``height`` rather than taken on trust, and the check belongs here
    rather than at the boundary: the boundary compares a finished status against the port it
    describes, which catches a wrong total and nothing finer. A caller stacking one block per melted
    column hands over several blocks, and a block one row short beside a block one row long adds up
    to a total the boundary is happy with, having slid every mark between them onto the wrong row --
    the exact failure this channel exists to make impossible, arriving inside the tool that builds
    it.
    """
    misaligned = [column for column, detail in marks if detail.len() != height]
    if misaligned:
        raise PortError(
            f"a per-row status of {height:,} row(s) cannot be built from mark(s) on "
            f"{', '.join(misaligned)} of another length; a mark carries one reason per row of the "
            f"port it describes, and one of another height marks the wrong rows rather than fewer"
        )

    if not marks:
        return pl.DataFrame({"level": ["ok"] * height, "reason": [[]] * height}, schema=STATUS_SCHEMA)

    # Keyed by position rather than by column name: two marks may name the same column -- a column
    # read twice, a column melted alongside itself -- and a frame keyed by name would silently keep
    # one of them.
    details = pl.DataFrame({str(index): detail for index, (_, detail) in enumerate(marks)})
    reason = pl.concat_list(
        *(
            pl.when(pl.col(str(index)).is_not_null())
            .then(pl.concat_list(pl.struct(column=pl.lit(column, dtype=pl.String), detail=pl.col(str(index)))))
            .otherwise(pl.lit([], dtype=STATUS_REASON))
            for index, (column, _) in enumerate(marks)
        )
    )
    return details.select(reason=reason).select(
        level=pl.when(pl.col("reason").list.len() > 0).then(pl.lit("warning")).otherwise(pl.lit("ok")),
        reason=pl.col("reason"),
    )


class Ports:
    """An ordered, read-only set of named frames -- a step's inputs, or its outputs.

    Deliberately not a :class:`collections.abc.Mapping`: ``Mapping.get`` is built on
    ``__getitem__`` catching ``KeyError``, and looking up a port that is not there raises
    :class:`PortError` here instead, so inheriting the mixin would hand back a ``get`` that is
    broken in a way nothing would notice until it silently swallowed the wrong exception. A small,
    closed surface is easier to trust than a partially-working abstract base.

    The source mapping is copied at construction, and ``__slots__`` together with a read-only
    ``names`` property is what keeps an instance immutable afterwards: there is nowhere on the
    object for a new attribute to land, and the one property that looks like an attribute has no
    setter.
    """

    __slots__ = ("_frames", "_notes", "_status")

    def __init__(
        self,
        frames: Mapping[str, pl.DataFrame] | None = None,
        *,
        notes: Sequence[str] = (),
        status: Mapping[str, pl.DataFrame] | None = None,
    ) -> None:
        source = dict(frames) if frames is not None else {}
        for name, frame in source.items():
            if not isinstance(frame, pl.DataFrame):
                raise PortError(f"port {name!r} must hold a DataFrame, got {type(frame).__name__} instead")
        marked = dict(status) if status is not None else {}
        for name, frame in marked.items():
            if not isinstance(frame, pl.DataFrame):
                raise PortError(f"the status for port {name!r} must be a DataFrame, got {type(frame).__name__} instead")
        self._frames: tuple[tuple[str, pl.DataFrame], ...] = tuple(source.items())
        self._notes: tuple[str, ...] = tuple(notes)
        self._status: Mapping[str, pl.DataFrame] = MappingProxyType(marked)

    @property
    def names(self) -> tuple[str, ...]:
        """Port names, in the order they were given to the constructor."""
        return tuple(name for name, _ in self._frames)

    @property
    def notes(self) -> tuple[str, ...]:
        """What the step wants a reader to know, short of refusing.

        The channel between "this ran" and "this failed", which nothing else here provides. A tool
        that quietly dropped three values out of forty-six thousand has done something a reader
        would want to know the size of, and an exception is the wrong way to say it -- the run
        should carry on and the fact should travel with it. The runner raises a step carrying notes
        to a warning, which is what puts it in the log and marks it on the canvas.

        Frames only in the data plane, as everywhere else: a note is for a person, and nothing
        downstream may read one.
        """
        return self._notes

    @property
    def status(self) -> Mapping[str, pl.DataFrame]:
        """What the step did to each row of a port it degraded, keyed by port name.

        Beside the frame and never inside it. A column named ``status`` would be a column: it flows
        into the next join, survives a ``select_columns`` that never heard of it, turns up in a
        schema comparison against somebody else's engine, and lands in the published workbook -- so
        it would have to be carried and stripped at every one of those places, and the first place
        that forgot would ship it. What a step did is not a fact about the data, so it travels where
        ``notes`` travels.

        The pair is the point: a note says *how many* values a step emptied, which is a quantity and
        nothing more, and this says *which rows* -- one row per data row, in the same order, never
        re-derived afterwards. A reader given only the count cannot tell a figure the run destroyed
        from text that was never a figure, and that is the difference between a broken pipeline and
        a correctly reported one.

        Empty for almost every step, and deliberately so: a step that degraded nothing says nothing,
        the boundary refuses an inert status, and the presence of a status on a port is itself the
        signal.
        """
        return self._status

    def items(self) -> tuple[tuple[str, pl.DataFrame], ...]:
        """``(name, frame)`` pairs, in construction order."""
        return self._frames

    def __getitem__(self, name: str) -> pl.DataFrame:
        for key, frame in self._frames:
            if key == name:
                return frame
        available = ", ".join(self.names) if self._frames else "none"
        raise PortError(f"no port named {name!r}; the ports present are: {available}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self._frames)

    def __contains__(self, name: object) -> bool:
        return name in self.names


# ---------------------------------------------------------------------------
# Tool anatomy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfigField:
    """One entry in a tool's configuration template.

    ``type`` is a member of :data:`FIELD_TYPES`, and ``"expression"`` is one of those types rather
    than a separate boolean, so :func:`expression_fields` has exactly one place to read instead of
    two that could disagree.

    ``items`` describes an array's element and ``fields`` describes an object's members, which is
    the smallest pair of additions that makes the common configuration shape -- an ordered list of
    small objects, as used for ordered rules, lists of aggregations, lists of key pairs -- say
    anything about itself. Without them such a configuration is one untyped ``array`` field, so
    nothing inside it is type checked, required checked, or compiled, and the expressions it holds
    are invisible to a caller pre-flighting them.

    A default is not filled inside ``items`` or ``fields``: an element that does not exist has
    nothing to fill, and inventing one would put a rule in an ordered list that nobody wrote.

    ``required`` and ``non_empty`` answer two different questions: whether a value arrived at all,
    and whether the value that arrived carries anything. ``required=True`` accepts ``[]``, ``""``
    and ``{}`` -- each is a value -- and ``non_empty=True`` is how a field refuses them, on its own
    or alongside ``required``, without teaching ``required`` a second meaning.
    """

    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Sequence[str] | None = None
    items: "ConfigField | None" = None
    fields: "Mapping[str, ConfigField] | None" = None
    non_empty: bool = False


@dataclass(frozen=True, slots=True)
class WorkedExample:
    """One demonstration of a tool: a scenario, the configuration and inputs it runs on, the output
    it claims to produce, and the explanation a reader needs to trust the claim without running it.

    The claim is not taken on faith: :func:`check_tool` runs every worked example through
    :func:`run_tool` and compares the result against ``outputs``, so a documented example is a
    checked example.

    ``check_row_order`` defaults to ``False``: for a tool that pairs or groups rows, the order they
    come back in is an artefact of the query engine, and pinning it would make an ordinary example
    fail whenever the engine's own internals changed rather than when the tool did. An example that
    documents an order-preserving tool opts in, and doing so applies to every port the example
    documents, not one.

    ``setup`` is how an example that needs a world -- a connection, a file already on disk --
    declares it, rather than a tool building that world inside ``help()`` as a side effect of
    describing itself. Left as ``None``, an example is checked exactly as one of the shipped,
    total-function tools always has been: no world entered, none to tear down.

    ``setup`` is a callable returning a context manager, not a context manager itself, and that
    indirection carries the whole guarantee. The same example is checked more than once in a
    process -- once on its own, and again every time the catalog it belongs to is checked -- so
    whatever prepares the world must be able to do so more than once. A context manager built once
    and stored, such as a bare ``@contextmanager`` generator object, is good for exactly one
    ``with`` block; asked for a second time it is already exhausted. A callable has no such limit:
    every check calls it again and gets a fresh one.

    When ``configuration`` holds the ``{scratch}`` token, :func:`check_tool` opens a directory
    before this callable is entered and resolves the token onto a private copy of ``configuration``
    for the run -- ``setup`` reaches the very same directory through :func:`scratch_path`, and the
    directory is removed again once the check returns, whatever it decided. The directory is the
    checker's, not the example's: anything a setup writes inside it goes with it, but a file written
    anywhere else is exactly as invisible to every check here as it always was, and this does not
    close that hole and is not claiming to.
    """

    scenario: str
    configuration: Mapping[str, Any]
    explanation: str
    inputs: Ports
    outputs: Ports
    check_row_order: bool = False
    setup: Callable[[], AbstractContextManager[object]] | None = None


@dataclass(frozen=True, slots=True)
class ToolHelp:
    """The detailed half of a tool's documentation -- what its one-line description leaves out."""

    detailed_description: str
    examples: Sequence[WorkedExample] = ()
    notes: Sequence[str] = ()
    field_help: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """The shape a step must have to be registered and run: described, and callable.

    A :class:`typing.Protocol` rather than an abstract base class, because the registry only needs
    "a described, callable thing" and structural typing says exactly that without forcing a caller
    to inherit from a class of ours. ``runtime_checkable`` gives the registry a cheap gate at
    registration time, checking that every named member is present without dictating how a caller's
    tool is implemented.

    There are two validation members rather than one because they can answer different questions.
    ``validate_config`` sees the configuration alone, so it is where a cross-field rule lives.
    ``validate_inputs`` also sees the frames, so it is where "the column this configuration names
    has to be on this port" lives -- and it is on the protocol, called on every run, rather than
    called wherever it happens to exist, because the second shape is a rule that holds while an
    author remembers. A tool with nothing to add writes ``return None``; the cost of the gate being
    real is that a tool without the member is not a :class:`Tool` at all.
    """

    name: str
    description: str
    category: str
    input_ports: Sequence[str]
    output_ports: Sequence[str]

    def config_template(self) -> Mapping[str, ConfigField]: ...

    def help(self) -> ToolHelp: ...

    def validate_config(self, config: Mapping[str, Any]) -> None: ...

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None: ...

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports: ...


@runtime_checkable
class DerivesInputPorts(Protocol):
    """The optional member for a tool whose input ports depend on how the step is configured.

    Almost every tool reads a fixed set of ports and names them in ``input_ports``. One shape
    cannot: a step that writes a sheet per frame reads as many frames as its configuration names.
    Such a tool leaves ``input_ports`` empty and writes this member instead. A tool without the
    member is a :class:`Tool` like any other, and behaves exactly as it always has.

    Three rules for an author writing it. Two of them are checked, and this docstring says which.

    **A tool answers the question one way or the other, never both.** Declare static
    ``input_ports`` or implement this member. A tool doing both gives two answers to "what does
    this step read", and different parts of the library ask different ones -- so a supplied frame
    can be accepted by the one that asks the configuration and then thrown away by the one that
    reads the attribute, with nothing raised. :func:`check_tool` refuses a tool that does both.

    **Never name one port twice.** A port carries one frame, so the second use of a name replaces
    the first and a frame is lost without a word. Refused in two places: :func:`check_tool` refuses
    a worked example that does it, and the loader refuses an authored step that does it.

    **Return the ports in the order the step reads them.** The configuration handed over is the one
    a person wrote, with no default filled in -- the loader checks a step's edges before it
    validates its configuration, and the boundary checks the envelope before it fills defaults in.
    So the field naming the ports must be one the author always writes. Put the port list behind a
    default and this member sees nothing. **Nothing checks this one**, because a field that is
    absent and a field that is empty look the same from here.
    """

    def input_ports_for(self, config: Mapping[str, Any]) -> Sequence[str]: ...


def ports_from_config(tool: Tool) -> bool:
    """Say whether this tool works its input ports out from a step's configuration."""
    return isinstance(tool, DerivesInputPorts)


def input_ports_for(tool: Tool, config: Mapping[str, Any]) -> tuple[str, ...]:
    """The input ports a step reads, given its tool and the configuration it was written with.

    Every reader of a port asks this one question, so no two of them can get different answers. A
    tool without the optional member hands back the attribute it declares, whatever the
    configuration says.
    """
    if not isinstance(tool, DerivesInputPorts):
        return tuple(tool.input_ports)
    return tuple(_boundary_call(tool, "input_ports_for", lambda: tool.input_ports_for(config)))


def catalog_input_ports(tool: Tool) -> tuple[str, ...]:
    """The input ports to show in a catalog, which has no step's configuration in hand.

    A tool with fixed ports shows them. A tool that derives its ports shows the ones its first
    worked example reads. Those names cannot drift: they come from the same member the runner asks,
    and from a configuration :func:`check_tool` executes every time it runs. They are true of that
    one example only, so show :func:`ports_from_config` beside them and say where they came from.
    """
    if not ports_from_config(tool):
        return tuple(tool.input_ports)
    examples = _boundary_call(tool, "help", tool.help).examples
    if not examples:
        return ()
    return input_ports_for(tool, examples[0].configuration)


# ---------------------------------------------------------------------------
# Running a tool through the boundary
# ---------------------------------------------------------------------------


def _boundary_call[T](tool: Tool, phase: str, action: Callable[[], T]) -> T:
    """Run one piece of a tool's own code, translating a stray failure into :class:`ExecutionError`.

    Every call site that reaches into code an author wrote -- building a template, validating a
    configuration, running against real data, describing itself -- is a call site the query engine
    can raise through, with an exception type this module's caller has never heard of. The
    ``except ToolError: raise`` arm has to come first, and it has to come first here rather than
    only around ``execute``: a tool that already raises a precise :class:`ConfigError` from inside
    its own ``validate_config`` must reach the caller as itself, not buried under a phase name.
    """
    try:
        return action()
    except ToolError:
        raise
    except Exception as error:  # noqa: BLE001 - translating whatever a tool's own code raises is the point
        raise ExecutionError(
            f"{tool.name!r} failed while {_PHASE_VERBS[phase]}: {type(error).__name__}: {error}"
        ) from error


def run_tool(tool: Tool, inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Run ``tool`` on ``inputs`` with ``config``, checking the envelope on the way in and out.

    The order below is part of the contract, not an implementation detail. Ports come first,
    because a port mistake belongs to the flow that wired the step rather than to the configuration
    anybody wrote. The template comes next, and it is checked here rather than left to the tool: a
    tool with nothing to add beyond its template writes an empty ``validate_config``, which is the
    natural thing to write and which used to buy it no validation whatsoever. A tool's own
    ``validate_config`` is therefore for what a template cannot express, it runs last, and it runs
    on the *filled* configuration, so a cross-field rule reads a defaulted value by subscript
    instead of guessing with ``.get``. ``execute`` never sees a configuration that fails
    validation -- a caller relying on that is relying on something this function guarantees. A tool
    that derives its ports is therefore asked about the configuration as written, before any
    default is filled in, which is the same configuration the loader asks it about.

    On the way out, what came back must be :class:`Ports`. Returning a bare mapping of frames is
    the likeliest authoring slip, and left unchecked it surfaces as an attribute error raised from
    inside this function, naming neither the tool nor the mistake.

    The inputs are then checked for having been written to. This *detects and reports after the
    fact*; it is not a rollback, and by the time it raises, the caller's frame has already been
    changed. It is still worth raising, because ``Ports`` copies the mapping and not the frames, so
    an in-place column operation reaches straight through the envelope: where one output fans into
    two branches, one branch rewrites the other branch's input, and a step's own record of what it
    produced stops being true. Only the schema is compared, because copying every frame would cost
    a copy per step and a flow is many steps long.

    Between the tool's own configuration rules and ``execute`` sits ``validate_inputs``, the only
    phase that sees the configuration and the frames together. It runs on the filled configuration,
    like ``validate_config``, and it exists because the question it answers -- do the columns this
    configuration names exist, on the ports they will be read from -- can be answered by neither of
    its neighbours, and so used to be answered by the first few lines of every tool's ``execute``,
    in each tool's own words, after this boundary had already reported that the tool was running.

    Each of the four calls into the tool's own code -- ``config_template``, ``validate_config``,
    ``validate_inputs``, ``execute`` -- goes through :func:`_boundary_call`, so a stray exception
    from any of them, including one raised from inside the query engine a tool called into, arrives
    here as :class:`ExecutionError` naming which phase failed, rather than as whatever exception
    type that phase happened to raise. Each phase names itself distinctly, because a reader sent to
    the wrong one of two neighbouring functions has been told nothing. A :class:`ToolError` a tool
    raises on purpose is never touched by this -- including a :class:`DataError`, which is a
    diagnosis about the rows and would be buried by a phase name: it reaches the caller exactly as
    the tool wrote it.
    """
    declared_inputs = input_ports_for(tool, config)
    declared_input_set = set(declared_inputs)
    supplied = set(inputs.names)

    undeclared_inputs = supplied - declared_input_set
    if undeclared_inputs:
        raise PortError(
            f"{tool.name!r} was given undeclared input port(s) {', '.join(sorted(undeclared_inputs))}; "
            f"it declares: {', '.join(declared_inputs)}"
        )
    missing_inputs = declared_input_set - supplied
    if missing_inputs:
        raise PortError(
            f"{tool.name!r} is missing declared input port(s) {', '.join(sorted(missing_inputs))}; "
            f"it declares: {', '.join(declared_inputs)}"
        )

    template = _boundary_call(tool, "config_template", tool.config_template)
    validate_against_template(template, config)
    filled_config = apply_defaults(template, config)
    _boundary_call(tool, "validate_config", lambda: tool.validate_config(filled_config))
    _boundary_call(tool, "validate_inputs", lambda: tool.validate_inputs(inputs, filled_config))

    schemas_before = {name: frame.schema for name, frame in inputs.items()}

    result = _boundary_call(tool, "execute", lambda: tool.execute(inputs, filled_config))
    if not isinstance(result, Ports):
        raise PortError(
            f"{tool.name!r} handed back {type(result).__name__} rather than Ports; a tool must return "
            f"its outputs as named ports"
        )

    declared_outputs = tuple(tool.output_ports)
    declared_output_set = set(declared_outputs)
    produced = set(result.names)

    undeclared_outputs = produced - declared_output_set
    if undeclared_outputs:
        raise PortError(
            f"{tool.name!r} produced undeclared output port(s) {', '.join(sorted(undeclared_outputs))}; "
            f"it declares: {', '.join(declared_outputs)}"
        )
    missing_outputs = declared_output_set - produced
    if missing_outputs:
        raise PortError(
            f"{tool.name!r} never produced declared output port(s) {', '.join(sorted(missing_outputs))}; "
            f"it declares: {', '.join(declared_outputs)}"
        )

    for name in declared_outputs:
        if result[name].width == 0:
            raise PortError(
                f"{tool.name!r} output port {name!r} has no columns at all; an output must carry its "
                f"schema even when it carries no rows"
            )

    for name, frame in inputs.items():
        if frame.schema != schemas_before[name]:
            raise PortError(
                f"{tool.name!r} changed the columns of its input port {name!r} in place; a tool must "
                f"treat its inputs as read-only (columns before: {', '.join(schemas_before[name])}; "
                f"after: {', '.join(frame.schema)})"
            )

    _check_status(tool, result, declared_outputs)

    # Rebuilt so the ports come back in the tool's declared order whatever order it produced them
    # in -- and carrying the notes and the per-row status across, which rebuilding would otherwise
    # drop. A tool that reported something worth knowing and had it silently discarded at the
    # boundary is the exact failure the boundary exists to prevent, one level up, and it has
    # happened here once already: the notes were dropped by this very rebuild.
    return Ports(
        {name: result[name] for name in declared_outputs},
        notes=result.notes,
        status=result.status,
    )


def _check_status(tool: Tool, result: Ports, declared_outputs: tuple[str, ...]) -> None:
    """Refuse a per-row status that cannot mean what it says, before anything downstream reads it.

    Checked on every call rather than trusted, because the correspondence between a status row and
    a data row is positional and is never re-derived afterwards: a status one row short does not
    report less, it reports the *wrong* rows, sliding every mark after the gap onto its neighbour,
    and nothing built from it later could tell. Both directions are refused, since a check written
    against the short case alone passes a status a tool built per *input* row after filtering.

    The shape is refused too, and so is a level outside :data:`STATUS_LEVELS`: what reads a status
    is a renderer, and a level it has no drawing for shows as neither a tick nor a warning nor a
    cross, so the row shows nothing and reads as fine -- the same defect as a missing column,
    arriving more quietly.

    An all-``ok`` status is refused last and on sight. Seven million rows asserting that nothing
    happened would make the presence of a status mean nothing, and a mark that is always there is a
    mark nobody reads. This is the ruling this library already takes on a cast to the type a column
    already holds: an inert declaration is refused rather than agreed with.
    """
    for name, status in result.status.items():
        if name not in declared_outputs:
            raise PortError(
                f"{tool.name!r} reported a per-row status for {name!r}, which is not one of its "
                f"output ports; it declares: {', '.join(declared_outputs)}"
            )

        if status.schema != STATUS_SCHEMA:
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} in the wrong shape: "
                f"expected {dict(STATUS_SCHEMA)}, got {dict(status.schema)}"
            )

        unknown = sorted(str(level) for level in status["level"].unique().to_list() if level not in STATUS_LEVELS)
        if unknown:
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} holding level(s) "
                f"{', '.join(unknown)}, which nothing can draw; the published levels are: "
                f"{', '.join(STATUS_LEVELS)}"
            )

        height = result[name].height
        if status.height != height:
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} of {status.height:,} "
                f"row(s) against a port of {height:,}; the two line up row for row, and a status of "
                f"another height marks the wrong rows rather than fewer of them"
            )

        if not (status["level"] != "ok").any():
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} saying only that every "
                f"row is fine; a status says what a step degraded, and one that degraded nothing "
                f"reports no status at all"
            )


# ---------------------------------------------------------------------------
# A configuration read against a frame
#
# Everything here answers a question that needs both halves in front of it -- the configuration an
# author wrote, and the schema of the frame that turned up -- which is why none of it can live in a
# template and none of it can wait for ``execute``. Each one exists because several tools were
# already asking it, and the third or fourth spelling of a question is where the answers diverge.
# ---------------------------------------------------------------------------


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


def expression_fields(template: Mapping[str, ConfigField]) -> tuple[str, ...]:
    """Paths of the fields in ``template`` that hold an expression, in template order.

    The declared field type is the whole seam to the expression language, and the path is how a
    field nested inside an array or an object is named: ``"rules[].when"`` is the ``when`` member
    of an element of the ``rules`` array. A path carries no index, because a template describes
    what *may* be there and no particular element exists in it -- which is exactly the question a
    form renderer or a documentation page asks. For what a configuration *does* hold, and where,
    use :func:`expression_values`.
    """
    return tuple(_expression_paths(template, prefix=""))


def _expression_paths(template: Mapping[str, ConfigField], prefix: str) -> Iterator[str]:
    for name, spec in template.items():
        yield from _spec_expression_paths(_joined(prefix, name), spec)


def _spec_expression_paths(path: str, spec: ConfigField) -> Iterator[str]:
    if spec.type == "expression":
        yield path
    elif spec.type == "array" and spec.items is not None:
        yield from _spec_expression_paths(f"{path}[]", spec.items)
    elif spec.type == "object" and spec.fields is not None:
        yield from _expression_paths(spec.fields, prefix=path)


def expression_values(
    template: Mapping[str, ConfigField],
    config: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Every expression ``config`` actually holds, as ``(path, text)`` pairs in template order.

    Separate from :func:`expression_fields` because the two answer different questions and only one
    of them can be answered by a template. A concrete index -- ``rules[0].when`` -- is the only
    form of path a caller can put in front of a person who has to go and fix the expression, and an
    index cannot exist in a template, so the reportable path needs the configuration.

    This is what makes the seam reach a configuration that has no top-level expression field at
    all: a caller lifts the references out of every pair, checks them against the upstream frame,
    and reports the ones that miss by the path they sit at, before a single row is read.
    """
    return tuple(_expression_values(template, config, prefix=""))


def _expression_values(
    template: Mapping[str, ConfigField],
    config: Mapping[str, Any],
    prefix: str,
) -> Iterator[tuple[str, str]]:
    for name, spec in template.items():
        if name in config:
            yield from _spec_expression_values(_joined(prefix, name), spec, config[name])


def _spec_expression_values(path: str, spec: ConfigField, value: Any) -> Iterator[tuple[str, str]]:
    if value is None:
        return
    if spec.type == "expression" and isinstance(value, str):
        yield path, value
    elif spec.type == "array" and spec.items is not None and isinstance(value, list):
        for index, item in enumerate(value):
            yield from _spec_expression_values(f"{path}[{index}]", spec.items, item)
    elif spec.type == "object" and spec.fields is not None and isinstance(value, dict):
        yield from _expression_values(spec.fields, value, prefix=path)


def apply_defaults(template: Mapping[str, ConfigField], config: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` with every field of ``template`` present -- its default, or ``None``.

    Every field, not only the ones carrying a declared default, because the alternative makes
    ``execute`` read its own configuration two ways: subscript the field with a default, ``.get``
    the field beside it without one, a distinction with no meaning that every author has to
    rediscover field by field. Filling all of them makes "was this configured?" one expression,
    ``config["x"] is None``, everywhere.

    That only composes because :func:`validate_against_template` accepts ``None`` for an optional
    field: a filled configuration is handed to a tool's own ``validate_config``, which validates it
    against the same template a second time, and it has to survive that.

    Nothing is filled inside an array or an object. There is no element to fill a member of, and
    inventing one would add an entry to an ordered list that nobody wrote.

    Public because a loader assembling a configuration from a file wants the same filling the
    boundary does.
    """
    filled = dict(config)
    for name, spec in template.items():
        if name not in filled:
            filled[name] = spec.default
    return filled


def _joined(prefix: str, name: str) -> str:
    """A field's reportable path: ``name`` at the top level, ``rules[0].when`` inside a value."""
    return f"{prefix}.{name}" if prefix else name


def validate_against_template(template: Mapping[str, ConfigField], config: Mapping[str, Any]) -> None:
    """Everything a template can express about ``config``, checked uniformly for every field.

    An unknown field, a missing required one, a value of the wrong type, a value outside its
    ``enum``, and an expression that will not compile are all refused here -- the last one compiled
    with no known columns, so bad *syntax* is caught at this boundary while a missing *column* stays
    the caller's problem, because only the caller knows the frame the tool is about to run against.

    A string still holding ``${...}`` anywhere inside it -- including nested inside an array or an
    object value -- is refused too, regardless of the field's declared type. Such a placeholder is
    meant to be resolved by whoever assembles the configuration; it is not this module's job to
    resolve it, but it is exactly this module's job to refuse one that arrives unresolved, because a
    placeholder like that would otherwise compile as an ordinary piece of text, match nothing, and
    leave the run looking as if it had succeeded.

    Wherever a field declares the shape below itself, a nested value is held to exactly these same
    rules and the refusal names it by its path -- ``rules[0].when`` -- because a configuration whose
    expressions all sit inside an array of objects is the common case, not the exotic one.

    A field whose declared type is not in :data:`FIELD_TYPES` is refused rather than skipped. The
    lookup used to fail open, which meant a template author reaching for the obvious name of a type
    that does not exist got a field accepting anything at all.

    ``None`` is accepted for an optional field and still refused for a required one. That is what
    makes filling and validating composable: :func:`apply_defaults` fills every field, so a
    configuration on its way to a tool carries ``None`` for everything nobody configured and has to
    survive its own type, enum and expression checks.

    ``required`` and ``non_empty`` are checked separately and answer different questions.
    ``required`` refuses ``None`` -- did a value arrive at all -- and is satisfied by ``[]``,
    ``""`` or ``{}``, each of which is a value. ``non_empty`` refuses exactly those empty values on
    a field that declares it, whether or not the field is also ``required``, because knowing a
    value arrived says nothing about whether it carries anything.
    """
    _check_level(template, config, prefix="")


def _check_level(template: Mapping[str, ConfigField], config: Mapping[str, Any], prefix: str) -> None:
    """One level of a configuration against one level of a template, reporting paths under ``prefix``."""
    _refuse_key_that_is_not_a_name(config, prefix)

    unknown = set(config) - set(template)
    if unknown:
        raise ConfigError(f"unknown config field(s): {', '.join(sorted(_joined(prefix, name) for name in unknown))}")

    for name, spec in template.items():
        path = _joined(prefix, name)
        _refuse_unknown_type(path, spec)
        if spec.required and config.get(name) is None:
            raise ConfigError(f"missing required config field {path!r}")

    for name, spec in template.items():
        if name in config:
            _check_value(_joined(prefix, name), spec, config[name])


def _refuse_key_that_is_not_a_name(config: Mapping[str, Any], prefix: str) -> None:
    """Refuse a configuration key that is not a string, before anything tries to report it.

    A key like this comes from a definition file. The author wrote ``on: period``, the loader
    turned the bare word into ``True``, and the key stopped being a name. So this is its own
    refusal rather than an unknown field. An unknown field is a typo, and the fix is a different
    word. This is a name the file ate, and the fix is a pair of quotes.

    The message carries the type as well as the key, because ``True`` and ``"True"`` print the
    same and the difference between them is the whole problem.
    """
    for key in config:
        if isinstance(key, str):
            continue
        where = f"config field {prefix!r} has" if prefix else "config has"
        raise ConfigError(
            f"{where} a key that is not a name: {key!r} ({type(key).__name__}). YAML turns bare "
            f"words like on/no/true into other types -- quote the key in the file."
        )


def _check_value(path: str, spec: ConfigField, value: Any) -> None:
    """One value against one field spec, walking into it wherever the spec describes what is inside."""
    _refuse_unknown_type(path, spec)
    if value is None:
        return

    if not _TYPE_CHECKS[spec.type](value):
        raise ConfigError(f"config field {path!r} must be of type {spec.type!r}, got {type(value).__name__}")

    if spec.non_empty and spec.type in _SIZED_TYPES and len(value) == 0:
        raise ConfigError(f"config field {path!r} must not be empty")

    if spec.type == "array" and spec.items is not None:
        for index, item in enumerate(value):
            _check_value(f"{path}[{index}]", spec.items, item)
        return

    if spec.type == "object" and spec.fields is not None:
        _check_level(spec.fields, value, prefix=path)
        return

    _check_no_placeholder(path, value)

    if spec.enum is not None and value not in spec.enum:
        raise ConfigError(f"config field {path!r} has value {value!r}, but must be one of: {', '.join(spec.enum)}")

    if spec.type == "expression":
        try:
            compile_expr(value, columns=None)
        except ExpressionError as error:
            raise ConfigError(f"config field {path!r} does not compile: {error}") from error


def _refuse_unknown_type(path: str, spec: ConfigField) -> None:
    """Refuse a declared type nothing knows how to check, rather than checking nothing."""
    if spec.type not in FIELD_TYPES:
        raise ConfigError(
            f"config field {path!r} declares unknown type {spec.type!r}; the types a template may "
            f"declare are: {', '.join(sorted(FIELD_TYPES))}"
        )


def _check_no_placeholder(path: str, value: Any) -> None:
    """Refuse ``${...}`` anywhere inside ``value``, walking into an array or object at any depth."""
    if isinstance(value, str):
        if "${" in value:
            raise ConfigError(f"config field {path!r} still holds an unresolved parameter placeholder: {value!r}")
    elif isinstance(value, list):
        for item in value:
            _check_no_placeholder(path, item)
    elif isinstance(value, dict):
        for item in value.values():
            _check_no_placeholder(path, item)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """A catalog of tools resolved by name.

    A repeat name is refused rather than left to overwrite whatever registered first, because a
    second tool silently taking over a name is a defect a caller would only find by noticing the
    wrong tool ran.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise RegistryError(f"{tool!r} does not conform to the Tool protocol and cannot be registered")
        if tool.name in self._tools:
            raise RegistryError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(self.names()) if self._tools else "none"
            raise RegistryError(f"no tool named {name!r} is registered; registered tools are: {available}") from None

    def names(self) -> tuple[str, ...]:
        """Registered names, sorted -- so a rendered catalog does not reshuffle as tools are added."""
        return tuple(sorted(self._tools))

    def tools(self) -> tuple[Tool, ...]:
        """Registered tools, in :meth:`names` order."""
        return tuple(self._tools[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


# The shipped catalog. Empty at this stage -- tools are added to it as they are built -- and
# :func:`check_registry` passes on it precisely because the conformance rules hold vacuously over
# nothing.
REGISTRY: Final[ToolRegistry] = ToolRegistry()


# ---------------------------------------------------------------------------
# Catalog conformance
# ---------------------------------------------------------------------------


@contextmanager
def temporary_environment(**values: str) -> Iterator[None]:
    """Set each of ``values`` in the environment for the block, then put back what was there.

    A source's or a sink's declared setup needs exactly this once it stops building its own world
    inside ``help()``: somewhere to resolve a connection from for the duration of one checked
    example, without touching whatever a variable of the same name already meant to the process
    that is doing the checking. "Put back" is not "unset" -- a variable that already carried a real
    value is restored to it, never removed, because the process asking for the check keeps running
    after it. A variable that was absent is left absent again. Both directions hold whether the
    block finishes cleanly or raises, which is the pairing a tool would otherwise have to hand-write
    once per source and once per sink, getting one of the two halves right and the other half not.
    """
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _process_state() -> tuple[dict[str, str], str]:
    """A snapshot of the two things a description or a declared world must not permanently move."""
    return dict(os.environ), os.getcwd()


def _refuse_disturbance(refusal: str, before: tuple[dict[str, str], str]) -> None:
    """Compare the process against a snapshot taken earlier, and raise ``refusal`` naming what moved.

    Only the environment and the working directory are compared, because those are the two
    disturbances that are both realistic here and cheap to observe from outside a tool's own code.
    A file left on disk, a connection left open, an in-memory database still referenced from a
    module global, a rewritten ``sys.path`` -- none of it is visible from here, and this function
    does not claim otherwise. It catches the mistake this module was built to stop, not every
    mistake a tool's own code could make.
    """
    environment_before, directory_before = before
    environment_after, directory_after = _process_state()
    if directory_after != directory_before:
        raise ConformanceError(
            f"{refusal}: the working directory changed, from '{directory_before}' to '{directory_after}'"
        )
    changed = sorted(
        name
        for name in set(environment_before) | set(environment_after)
        if environment_before.get(name) != environment_after.get(name)
    )
    if changed:
        raise ConformanceError(f"{refusal}: environment variable(s) changed: {', '.join(changed)}")


# ---------------------------------------------------------------------------
# The scratch directory -- see the module docstring for what this does and does not buy.
#
# ``{scratch}`` is one token with one fixed meaning, not the start of a substitution language: a
# label like ``{total}`` is not this module's business and reaches a tool exactly as written. And it
# is deliberately not ``${scratch}`` -- that spelling means a pipeline parameter nobody bound, and it
# must still be refused by the existing placeholder rule even though it contains the token's own
# name, which is why the pattern below never fires on a ``{scratch}`` preceded by ``$``.
# ---------------------------------------------------------------------------

SCRATCH_TOKEN: Final[str] = "{scratch}"

# The lookbehind is the sharp bit: substituting before ``_check_no_placeholder`` ever ran would
# rescue ``${scratch}``, turning a configuration nobody bound into one that runs -- against a
# directory this module is about to remove.
_SCRATCH_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\$)\{scratch\}")

# The directory the checker opened for whichever example is running right now, if any. A
# ``ContextVar`` rather than a plain module global, so a resolver reached from inside a tool's own
# declared setup sees exactly the directory the current check opened and never one left over from an
# earlier check in the same process.
_current_scratch_directory: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "_current_scratch_directory", default=None
)


def scratch_path(value: str) -> Path:
    """``value`` with every ``{scratch}`` token resolved against the directory the checker opened.

    Public so a declared ``setup`` can resolve the same token its example's own configuration holds,
    rather than a second, private way of spelling "the directory this check is using". Raises
    :class:`ConformanceError` when no example is being checked: a setup that cannot be entered
    outside :func:`check_tool` is correct, not a limitation, because there is no directory yet for it
    to resolve against.
    """
    directory = _current_scratch_directory.get()
    if directory is None:
        raise ConformanceError(
            "scratch_path() has nothing to resolve against: it may only be called while check_tool "
            "is checking a worked example whose configuration holds the {scratch} token"
        )
    return Path(_SCRATCH_PATTERN.sub(lambda _: str(directory), value))  # a function: see _resolved_scratch


def _uses_scratch(value: Any) -> bool:
    """Whether ``{scratch}`` (never ``${scratch}``) appears anywhere inside ``value``.

    Recurses into a list or a dict at any depth, mirroring the walk :func:`_check_no_placeholder`
    already does over a configuration for the same reason: the shape this catalog's configurations
    actually have is an ordered list of small objects, and stopping at the top level would miss the
    common case rather than the exotic one.
    """
    if isinstance(value, str):
        return _SCRATCH_PATTERN.search(value) is not None
    if isinstance(value, list):
        return any(_uses_scratch(item) for item in value)
    if isinstance(value, dict):
        return any(_uses_scratch(item) for item in value.values())
    return False


def _resolved_scratch(value: Any, directory: Path) -> Any:
    """``value`` with every ``{scratch}`` token replaced by ``directory``, rebuilt rather than rewritten.

    A list or a dict is rebuilt into a new list or dict rather than mutated in place, because the
    stored configuration -- the one a rendered catalog page reads back -- must still show the token
    afterwards. Writing into the original container would resolve it for whichever check happened to
    run first and leave the page showing a directory that existed for a few milliseconds on somebody
    else's machine.
    """
    if isinstance(value, str):
        # A function, not the string itself: re.sub reads backslashes in a replacement string as
        # escapes, and every Windows path is full of them (C:\Users\... is a bad \U escape).
        return _SCRATCH_PATTERN.sub(lambda _: str(directory), value)
    if isinstance(value, list):
        return [_resolved_scratch(item, directory) for item in value]
    if isinstance(value, dict):
        return {key: _resolved_scratch(item, directory) for key, item in value.items()}
    return value


def check_tool(tool: Tool) -> None:
    """Check ``tool`` against every rule that makes the catalog trustworthy.

    The structural rules -- a machine name, a real description, an output port unless the tool is a
    sink, unique port names, a well-formed template, documentation that matches the template -- run
    first, because there is no point running a worked example against a tool that does not even
    have a name worth reporting.

    The rule that runs last is the one that matters most: every worked example is run through
    :func:`run_tool` and its result is compared, port by port over every port the tool *declares*,
    against what the tool claims. A tool whose example has quietly drifted from its own behaviour
    is worse than one with no example at all, because someone will read the example and copy it.

    Row order is unchecked by default: a tool that pairs or groups rows produces an order that is an
    artefact of the query engine underneath, and pinning it in the catalog would make such an
    example flaky against nothing the tool actually got wrong. An example opts into the comparison
    with :attr:`WorkedExample.check_row_order`, and doing so covers every port that example
    documents, so an order-preserving tool can have the one property a reader most wants to trust
    actually checked here rather than merely claimed in prose.

    ``help()`` itself is held to one further rule: describing a tool must not change the process
    doing the describing. A snapshot is taken immediately before the call and compared immediately
    after, and a ``help()`` that exported an environment variable or changed the working directory
    while building its worked examples is refused right here, before any example runs -- a
    description that disturbs the process is the defect, not anything an example does later. The
    same comparison runs again around each example's own declared setup, in
    :func:`_check_example_with_setup`, which is this rule's mirror rather than a separate one.
    """
    if not isinstance(tool, Tool):
        raise ConformanceError(f"{tool!r} does not conform to the Tool protocol")

    if not _MACHINE_NAME.match(tool.name):
        raise ConformanceError(
            f"tool name {tool.name!r} is not a machine name (expected lowercase letters, digits and "
            f"underscores, starting with a letter)"
        )

    if not tool.description.strip():
        raise ConformanceError(f"tool {tool.name!r} has a blank description")

    if tool.category not in CATEGORIES:
        raise ConformanceError(
            f"tool {tool.name!r} has category {tool.category!r}, which is not one of the published "
            f"categories: {', '.join(sorted(CATEGORIES))}"
        )

    # A sink may be terminal, and nothing else may. A transform, a combine or a read that produces
    # nothing has been wired into a flow that can never use it, which is a defect; a sink that
    # produces nothing is the whole point of a sink -- a workbook writer has nothing honest to hand
    # on, and inventing a receipt to hand back would be shape without information. So the rule is
    # about the category rather than about every tool.
    if not tool.output_ports and tool.category != "output":
        raise ConformanceError(
            f"tool {tool.name!r} declares no output ports, and only a tool in the 'output' category "
            f"may be terminal; it is in {tool.category!r}"
        )

    # A tool says what it reads in one of two ways: a static attribute, or the optional member that
    # works the ports out from a step's configuration. Doing both makes the tool answer differently
    # depending on which one a reader asked, and the cost is silent. Configure such a tool to derive
    # no ports and the run guard accepts a supplied frame, while the decision to skip the step reads
    # the static attribute and says the step is computed. The step then runs and the supplied frame
    # is dropped, with nothing said. This used to be a line in a docstring, and a rule that holds
    # only while an author remembers it is a convention.
    if ports_from_config(tool) and tuple(tool.input_ports):
        raise ConformanceError(
            f"tool {tool.name!r} declares static input ports ({', '.join(tool.input_ports)}) and "
            f"implements input_ports_for as well. A tool does one or the other. Either leave "
            f"input_ports empty, or drop input_ports_for and keep the static ports."
        )

    if len(set(tool.input_ports)) != len(tool.input_ports):
        raise ConformanceError(f"tool {tool.name!r} declares duplicate input port names: {tool.input_ports}")
    if len(set(tool.output_ports)) != len(tool.output_ports):
        raise ConformanceError(f"tool {tool.name!r} declares duplicate output port names: {tool.output_ports}")

    for port in tool.input_ports:
        _refuse_rewritten_input_port(tool.name, port)

    template = tool.config_template()
    _check_specs(tool.name, template, prefix="")

    before_help = _process_state()
    help_ = _boundary_call(tool, "help", tool.help)
    _refuse_disturbance(f"tool {tool.name!r} disturbed the process while documenting itself", before_help)
    if not help_.detailed_description.strip():
        raise ConformanceError(f"tool {tool.name!r} has a blank detailed description")

    undocumented = set(help_.field_help) - set(template)
    if undocumented:
        raise ConformanceError(
            f"tool {tool.name!r} documents field_help for field(s) not in its config template: "
            f"{', '.join(sorted(undocumented))}"
        )

    if not help_.examples:
        raise ConformanceError(f"tool {tool.name!r} ships no worked examples")

    # A tool that derives its ports can name one port twice, and nothing further down would say so.
    # Ports are built from a mapping, so a repeated name arrives as one frame: the declared ports
    # and the supplied ports collapse to the same set and the boundary sees nothing wrong. The
    # example then runs, stacks the same frame twice, and what a reader is told is that the example
    # has drifted -- which sends them to compare output rows when the defect is the port list. So
    # this check is explicit, and it runs before any example does.
    if ports_from_config(tool):
        for example in help_.examples:
            derived = input_ports_for(tool, example.configuration)
            duplicated = sorted({port for port in derived if derived.count(port) > 1})
            if duplicated:
                raise ConformanceError(
                    f"tool {tool.name!r} derives duplicate input port name(s) {', '.join(duplicated)} "
                    f"from the configuration of its worked example {example.scenario!r}; a port "
                    f"carries one frame"
                )

    for example in help_.examples:
        _check_example(tool, template, example)


def _check_specs(
    tool_name: str,
    template: Mapping[str, ConfigField],
    prefix: str,
    *,
    nested: bool = False,
) -> None:
    """One level of a template, and then whatever each of its fields describes below itself."""
    for name, spec in template.items():
        _refuse_rewritten_name(tool_name, _joined(prefix, name), name)
        _check_spec(tool_name, _joined(prefix, name), spec, nested=nested)


def _check_spec(
    tool_name: str,
    path: str,
    spec: ConfigField,
    *,
    nested: bool = False,
    element: bool = False,
) -> None:
    """One field spec, held to the same rules wherever it sits -- top level or nested.

    Nested included, because a nested declaration is where a template is least read and most
    trusted. Every rule below refuses a declaration that reads as if it constrains something and
    does not, which is the worst outcome available to a template: an element type on a field that is
    not an array, members on a field with none, ``non_empty`` on a field with no notion of
    emptiness, and the two that depend on where the field sits.

    Those two are ``default`` anywhere below the top level and ``required`` on an element spec
    itself. :func:`apply_defaults` fills the top level only -- an element that does not exist has
    nothing to fill, and inventing one would add an entry to an ordered list nobody wrote -- so a
    nested default is never applied, while reading exactly as if it were, leaving ``execute`` to
    write ``.get`` against a field its own template promised would always be there. ``required`` on
    an element spec is read by nothing at all, because an element that is absent is not an element;
    what an author means by it is ``non_empty`` on the array itself, so the refusal says so.
    ``required`` on a *member* of an element is a different declaration and stays: the validator
    enforces it per element, and an ordered-rules tool depends on it.

    A declared default is also checked against its own field here, and this is the only place it can
    be checked, because validation runs on what a caller supplied and the default is added
    afterwards. Left out, a default is the one value in a configuration that never meets the
    validator, and a default of the wrong type reaches ``execute`` as something its own template
    says it will never receive.
    """
    if spec.type not in FIELD_TYPES:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} has unknown type {spec.type!r}; the types a template "
            f"may declare are: {', '.join(sorted(FIELD_TYPES))}"
        )

    if not spec.description.strip():
        raise ConformanceError(f"tool {tool_name!r} field {path!r} has a blank description")

    if spec.items is not None and spec.type != "array":
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares items but has type {spec.type!r}; an element "
            f"type constrains nothing outside an array"
        )

    if spec.fields is not None and spec.type != "object":
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares fields but has type {spec.type!r}; members "
            f"constrain nothing outside an object"
        )

    if spec.non_empty and spec.type not in _SIZED_TYPES:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares non_empty but has type {spec.type!r}, which "
            f"has no notion of emptiness; non_empty may only be declared on: {', '.join(sorted(_SIZED_TYPES))}"
        )

    if nested and spec.default is not None:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares a default that is never applied: defaults "
            f"are filled at the top level of a configuration only, because an element or a member "
            f"that nobody wrote has nothing to fill. Give the value in the tool's own code instead "
            f"of declaring it here, where the next reader will trust it."
        )

    if element and spec.required:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares required on an element, which nothing "
            f"reads: an element that is absent is not an element. Declare non_empty on the array "
            f"itself to refuse a list with nothing in it."
        )

    _refuse_rewritten_enum(tool_name, path, spec)

    if spec.default is not None:
        try:
            _check_value(path, spec, spec.default)
        except ConfigError as error:
            raise ConformanceError(
                f"tool {tool_name!r} field {path!r} has a default its own declaration refuses: {error}"
            ) from error

    if spec.items is not None:
        _check_spec(tool_name, f"{path}[]", spec.items, nested=True, element=True)
    if spec.fields is not None:
        _check_specs(tool_name, spec.fields, prefix=path, nested=True)


def _yaml_rewrites(text: str) -> bool:
    """Whether a definition file hands this text over as something other than the text itself."""
    return text in YAML_UNSAFE_VALUES or _NUMERIC_TEXT.match(text) is not None


def _refuse_rewritten_name(tool_name: str, path: str, name: str) -> None:
    """Refuse a config field name a definition file would turn into something else.

    A field name is a key in the file. ``cross_tab`` used to have a field called ``on``, so the
    obvious way to write that step handed the loader the key ``True``. Quoting the key in the file
    works, but that puts the fix on every author of every pipeline using the tool, and nothing
    tells them. The tool has one author, so the refusal belongs here.

    Nested names included. A member of an object and a member of an element of an array are both
    keys in the file, and a template is least read where it is deepest.
    """
    if _yaml_rewrites(name):
        raise ConformanceError(
            f"tool {tool_name!r} declares a config field {path!r}, which YAML resolves to "
            f"something other than that text before this library sees it -- so a step written the "
            f"obvious way hands over a key that is not a name at all. Choose a name that survives "
            f"a definition file."
        )


def _refuse_rewritten_input_port(tool_name: str, port: str) -> None:
    """Refuse an input port name a definition file would turn into something else.

    A step says what it reads as ``inputs: {rows: earlier_step.rows}``. The port name is the key,
    on the left of the colon, which is the same place a config field name sits. A port called
    ``on`` hands the loader the key ``True``, and the name has stopped being a name.

    Quoting the key in the file works. It also puts the fix on every author of every pipeline using
    the tool, and nothing tells them. The tool has one author, so the refusal belongs here.

    Output port names are left alone. A step names one as a value, on the right of the colon, where
    a definition file hands the word over as written -- ``filter`` ships ports called ``true`` and
    ``false``.

    This covers the ports a tool declares. A tool that works its input ports out from a step's
    configuration can still end up with an unsafe name, and nothing checks that yet.
    """
    if _yaml_rewrites(port):
        raise ConformanceError(
            f"tool {tool_name!r} declares an input port {port!r}, which YAML resolves to something "
            f"other than that text before this library sees it -- so a step wiring that port hands "
            f"over a key that is not a name at all. Choose a port name that survives a definition "
            f"file. An output port may keep such a name, because a step names one as a value, "
            f"where YAML leaves it alone."
        )


def _refuse_rewritten_enum(tool_name: str, path: str, spec: ConfigField) -> None:
    """Refuse an option a definition file would hand over as something other than the word written.

    A pipeline definition is authored in YAML, which resolves a handful of bare words into other
    types before this library sees any of them, and the two ways that lands are not equally bad. An
    option arriving as ``True`` is refused by its own field's type check, which is merely confusing.
    An option arriving as ``None`` -- which is what the obvious spellings of "nothing" do -- is
    treated as a field nobody configured, filled with the field's default, and the run proceeds
    under a policy its author did not choose and was never told about. So the value is refused here,
    at authoring time, where there is a person to tell. Every such word has a safe synonym:
    ``exclude``/``include`` rather than ``false``/``true``, ``first``/``last`` rather than
    ``no``/``yes``.

    Field names are held to the same rule, by :func:`_refuse_rewritten_name`, and both read the one
    published set. Names used to be exempt on the argument that a bad one announces itself. It does
    not. A key of another type used to crash the code that reports an unknown field, and one level
    down it produced a refusal naming a field nobody had written. Declared input port names are
    held to it too, by :func:`_refuse_rewritten_input_port`, because a step names one on the left
    of a colon as well. A port worked out from a step's configuration is not checked yet. Output
    port names are exempt: a step names one as a value, where YAML leaves it alone.

    The numeric clause refuses a tool that wanted members ``"1"`` and ``"2"``, and that is correct
    rather than over-strict: such a definition arrives carrying integers, so the enum would never
    match. The fix is a field type that is not a string -- not a looser rule here.
    """
    for value in spec.enum or ():
        if isinstance(value, str) and _yaml_rewrites(value):
            raise ConformanceError(
                f"tool {tool_name!r} field {path!r} offers enum value {value!r}, which YAML resolves "
                f"to something other than that text before this library sees it -- so the value can "
                f"never arrive, and a spelling of nothing would silently become the field's default. "
                f"Choose a word that survives a definition file."
            )


def _run_example(tool: Tool, template: Mapping[str, ConfigField], example: WorkedExample) -> None:
    """One worked example: readable, valid, documenting every declared output, and true.

    Validating the configuration before running it is what separates two defects a single "does
    not run" refusal would report identically: an example whose configuration is wrong, and an
    example that uses a port its tool does not declare. They have two different fixes, and a
    reader told only that the example failed has to work out which one they are looking at.

    The configuration is validated against the example's own inputs too, in that same block and for
    that same reason. An example naming a column its own frames do not carry is a configuration
    nobody could ever have run, and a documented example is the thing somebody copies: reported as
    a run failure it reads as if the tool were broken, and reported as a configuration failure it
    points at the line to change. This is also why a declared setup, when there is one, has to be
    open for validation and not merely for the run that follows: resolving a connection is exactly
    the kind of thing a source's ``validate_config`` does, and a bad example must still be reported
    as a configuration defect rather than as a run failure that only happened to occur first.

    Every declared output port is compared, not merely the ports the example documents, because the
    undocumented ones are the side outputs -- the rows that did not match, the leftovers from each
    side -- which are the whole reason the envelope exists and, being anti-joins, the likeliest to
    be wrong. An example documenting one of three outputs has been run rather than checked.

    The comparison is unordered by default, and only by default: row order is an artefact of the
    query engine for a tool that pairs or groups rows, so pinning it would make such an example fail
    whenever the engine's own internals changed. ``example.check_row_order`` is passed straight
    through to the comparison, so an example that opts in still has values, dtypes, column order and
    row count checked exactly as before -- only the one additional property changes.
    """
    if not example.scenario.strip():
        raise ConformanceError(
            f"tool {tool.name!r} ships a worked example with a blank scenario; every refusal below "
            f"quotes the scenario to say which example it means"
        )
    if not example.explanation.strip():
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} has a blank explanation, which is "
            f"the thing an example exists to be"
        )

    try:
        validate_against_template(template, example.configuration)
        filled = apply_defaults(template, example.configuration)
        tool.validate_config(filled)
        tool.validate_inputs(example.inputs, filled)
    except ConfigError as error:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} does not validate: {error}"
        ) from error

    declared = tuple(tool.output_ports)
    undeclared = set(example.outputs.names) - set(declared)
    if undeclared:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} documents output port(s) the tool "
            f"does not declare: {', '.join(sorted(undeclared))}; it declares: {', '.join(declared)}"
        )
    undocumented = set(declared) - set(example.outputs.names)
    if undocumented:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} documents no output for declared "
            f"port(s): {', '.join(sorted(undocumented))}"
        )

    try:
        produced = run_tool(tool, example.inputs, example.configuration)
    except ToolError as error:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} does not run: {error}"
        ) from error

    for name in declared:
        try:
            assert_frame_equal(produced[name], example.outputs[name], check_row_order=example.check_row_order)
        except AssertionError as error:
            raise ConformanceError(
                f"tool {tool.name!r} worked example {example.scenario!r} port {name!r} has drifted from "
                f"its documented output: {error}"
            ) from error


def _check_example_with_setup(tool: Tool, template: Mapping[str, ConfigField], example: WorkedExample) -> None:
    """Check one worked example, entering its declared world first if it has one.

    ``example.setup`` is ``None`` for every one of the shipped, total-function tools, and for those
    this is exactly :func:`_run_example` and nothing more -- the fence that keeps this seam from
    changing behaviour nobody asked it to change.

    Where a setup is declared, it is entered before :func:`_run_example` is called and left open for
    the whole of it -- validation and the run both -- because validation is where a source resolves
    its connection, and closing the world after only the run would leave ``validate_config`` and
    ``validate_inputs`` looking at a process with nothing prepared. The teardown runs in a
    ``finally``, so a comparison that fails or a tool that raises still tears down the world exactly
    as a clean check would: a conformance failure that left a database or a variable behind would
    poison every check that runs after it in the same process.

    A :class:`ToolError` -- including the :class:`ConformanceError` that :func:`_run_example` itself
    raises -- is never touched here and reaches the caller exactly as raised. Only a failure of the
    setup's own entry or exit, which is never a :class:`ToolError`, is translated, because that
    failure belongs to the declaration and not to the example it was checking.

    After the world is closed, the process is compared against the snapshot taken before it was
    opened, the mirror of the same check :func:`check_tool` runs around ``help()``: a setup that put
    back everything it changed passes silently, and a teardown that left a variable set or the
    working directory moved is refused by name.
    """
    if example.setup is None:
        _run_example(tool, template, example)
        return

    before = _process_state()
    try:
        with example.setup():
            _run_example(tool, template, example)
    except ToolError:
        raise
    except Exception as error:  # noqa: BLE001 - a setup failure is translated, not swallowed
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} has a setup that failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        _refuse_disturbance(
            f"tool {tool.name!r} worked example {example.scenario!r} left the process disturbed after "
            f"its setup was torn down",
            before,
        )


def _check_example(tool: Tool, template: Mapping[str, ConfigField], example: WorkedExample) -> None:
    """Give ``example`` a directory to work in when its configuration asks for one, then check it.

    ``example.configuration`` is the page a reader copies, so it is inspected for the ``{scratch}``
    token exactly as written, before anything else runs. A configuration that holds no token at all
    -- thirteen of the shipped tools, and the whole of :class:`_StubTool`'s default shape in this
    module's own tests -- is delegated to :func:`_check_example_with_setup` unchanged: this is the
    fence that keeps this seam from changing behaviour nobody asked it to change.

    Where the token is present, a fresh directory is created before anything else -- before the
    example's own configuration is resolved, and before its declared ``setup``, if it has one, is
    ever entered -- because ``setup`` reaches the same directory through :func:`scratch_path`, and a
    source resolves what it is about to read as early as ``validate_config``. The configuration
    :func:`_check_example_with_setup` goes on to check is a copy with the token resolved, built with
    :func:`dataclasses.replace` rather than written into ``example`` itself, so the object this
    module was handed -- and the page it came from -- still reads ``{scratch}`` afterwards.

    The directory is removed in a ``finally``, on every path out of the check, and that removal must
    not swallow the check's own refusal: whether the directory still exists is captured before it is
    removed, and the "destroyed the checker's directory" refusal is raised only *after* the whole
    try/finally has run its course -- which means only when nothing else already raised, so a drifted
    example or a tool that threw still reports as exactly that, not as a directory gone missing.

    A fresh directory per call, never a cached one: the conformance tests in this catalog run
    :func:`check_tool` twice on purpose, and a shared directory would let a file a sink's example
    wrote on the first pass satisfy the second pass's own "nothing exists yet" without writing
    anything at all.
    """
    if not _uses_scratch(example.configuration):
        _check_example_with_setup(tool, template, example)
        return

    directory = Path(tempfile.mkdtemp(prefix="etl_tool_scratch_"))
    token = _current_scratch_directory.set(directory)
    destroyed = False
    try:
        resolved_example = replace(example, configuration=_resolved_scratch(example.configuration, directory))
        try:
            _check_example_with_setup(tool, template, resolved_example)
        finally:
            destroyed = not directory.exists()
            shutil.rmtree(directory, ignore_errors=True)
    finally:
        _current_scratch_directory.reset(token)

    if destroyed:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} destroyed the directory "
            f"({directory}) the checker opened for it to work in; a file left inside the directory "
            "is the example's own to leave behind, but the directory itself belongs to the checker"
        )


def check_registry(registry: ToolRegistry) -> None:
    """Check every tool in ``registry`` against :func:`check_tool`, so a whole catalog is one call.

    Every refusal :func:`check_tool` raises names the tool it came from, which is what makes a
    catalog-wide call actionable rather than merely red.
    """
    for tool in registry.tools():
        check_tool(tool)
