"""The errors this library raises, all in one family.

Every error here descends from ``ToolError``, so a caller who wants to stop on any refusal catches
that one type. Each subclass points at a different thing to fix: a port, a configuration, the
wiring, the catalog, the data, or the run. The two run failures that carry a partial trace,
``StepFailedError`` and ``ColumnCheckError``, live in ``trace.py``.

Why: docs/design/core.md#one-error-family
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
    """The base of every error the library raises. Catch this to stop on any refusal.

    It is a ``ValueError``, so code that already catches ``ValueError`` keeps working.
    """


class PortError(ToolError):
    """An input or output port was missing, undeclared, unnamed, or untyped when it should not be."""


class ConfigError(ToolError):
    """A configuration does not satisfy its tool's template, or a rule the template cannot express."""


class RegistryError(ToolError):
    """A tool could not be registered or resolved by name."""


class ConformanceError(ToolError):
    """A tool breaks a catalog rule: something undocumented, undeclared, or no longer true."""


class ExecutionError(ToolError):
    """A tool's own code failed unexpectedly. The original exception is in ``__cause__``.

    The boundary wraps any exception that is not a ``ToolError``, such as a polars
    ``SchemaError``. The cause may have nothing to do with the tool (the machine ran out of
    memory, say). To decide whether to carry on, look at ``__cause__``.

    Why: docs/design/core.md#unexpected-failures-are-wrapped
    """


# How many example rows a refusal about data prints. The counts are always given in full.
MAX_REPORTED_ROWS: Final[int] = 5


# How many names a refusal lists when it says what there is to choose from.
MAX_LISTED_NAMES: Final[int] = 10


def name_choices(asked: str, available: Iterable[str], noun: str = "steps") -> str:
    """Return the end of a refusal about a name that was not found.

    It suggests the closest names, then lists what there is: every name when there are ten or
    fewer, otherwise the first ten and the total. ``noun`` is the word for the names ("steps").
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
    """The rows themselves were wrong. Go and look at the data.

    Raise it as ``DataError(refusal, offending=rows, total=n)``. The message gives how many rows
    offend out of how many, and the first few of them. Pass ``offending=None`` when the rows were
    refused as a whole. The frame itself is not kept. The boundary passes this error through
    unwrapped.

    Why: docs/design/core.md#a-data-error-keeps-counts-and-examples
    """

    def __init__(self, refusal: str, *, offending: pl.DataFrame | None, total: int) -> None:
        self.total_rows: int = total
        if offending is None:
            # No rows were examined one by one; "0 of N" would wrongly say no row was at fault.
            self.offending_rows: int | None = None
            self.examples: tuple[dict[str, Any], ...] = ()
            super().__init__(f"{refusal}: {total} row(s), not examined one by one")
            return
        self.offending_rows = offending.height
        self.examples = tuple(offending.head(MAX_REPORTED_ROWS).to_dicts())
        shown = f", starting with {'; '.join(str(row) for row in self.examples)}" if self.examples else ""
        super().__init__(f"{refusal}: {self.offending_rows} of {self.total_rows} row(s) offend{shown}")


class PipelineError(ToolError):
    """The flow's wiring is wrong. Go and look at the steps and the arrows between them.

    Examples: a duplicate step id, a bad edge, a cycle, a dead end, an unresolved reference, a
    step id the flow does not have. A step's own settings being wrong is a :class:`ConfigError`.
    """


class RunError(ToolError):
    """A run could not go ahead: a bad argument to the runner, a reference it could not resolve,
    or a frame asked for that the run did not keep.
    """
