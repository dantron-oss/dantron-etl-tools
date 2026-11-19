"""The errors this library raises, all in one family.

Every error here descends from ``ToolError``, so a caller who wants to stop on any refusal catches
that one type. Each subclass points at a different thing to fix: a port, a configuration, the
wiring, the catalog, the data, or the run. The two run failures that carry a partial trace,
``StepFailedError`` and ``ColumnCheckError``, live in ``trace.py`` beside the trace they carry.

**One error family** carries all of it, and it says several different things: a port is wrong, the
configuration is wrong, the wiring is wrong, the catalog is wrong, the *data* is wrong, a run could
not proceed, or something nobody expected broke and was wrapped on the way out. The data fault is
worth naming separately because a reader's next move differs -- a data fault means look at the
data, an unexpected failure means look at the run, and the latter may be wrapping a problem the
machine had rather than anything a step decided. Every one of them descends from a single base, so a
flow driver has one type to catch.
"""

import difflib
from collections.abc import Iterable
from typing import Any, Final

import polars as pl

__all__ = [
    "MAX_REPORTED_ROWS",
    "ConfigError",
    "ConformanceError",
    "DataError",
    "ExecutionError",
    "PipelineError",
    "PortError",
    "RegistryError",
    "RunError",
    "ToolError",
    "name_choices",
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


# How many names a refusal lists when it says what there is to choose from. A flow can have a
# thousand steps, and a message listing all of them is a page nobody reads.
MAX_LISTED_NAMES: Final[int] = 10


def name_choices(asked: str, available: Iterable[str], noun: str = "steps") -> str:
    """The end of a refusal about a name that was not found: the closest names, and what there is.

    Lists every name when there are a few, and the first ten and the total when there are many.
    """
    names = [str(name) for name in available]
    if not names:
        return f"there are no {noun}"
    near = difflib.get_close_matches(str(asked), names, n=3)
    guess = f"did you mean {' or '.join(repr(name) for name in near)}? " if near else ""
    if len(names) <= MAX_LISTED_NAMES:
        return f"{guess}{noun} are: {', '.join(names)}"
    shown = ", ".join(names[:MAX_LISTED_NAMES])
    return f"{guess}there are {len(names)} {noun}; the first {MAX_LISTED_NAMES} are: {shown}, ..."


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


class PipelineError(ToolError):
    """A wiring fault: a duplicate id, an unresolved parameter, a bad edge, a lane, a group, a cycle,
    a dead end, a step id the pipeline does not have, or a value that cannot be fingerprinted.

    Distinct from :class:`ConfigError`, which a delegated ``validate_config`` or
    ``validate_against_template`` call raises for a configuration a tool refuses on its own terms.
    Both descend from :class:`ToolError`, so a driver has one family to catch, but the two
    name different next moves: this one sends a reader to the arrows, the other to the config block.
    """


class RunError(ToolError):
    """A run could not proceed: a bad argument, an unresolved reference, evidence asked for that a
    retention mode already freed. Stays inside the one family every tool boundary already raises so
    a caller has one type to catch, whichever layer refused.
    """
