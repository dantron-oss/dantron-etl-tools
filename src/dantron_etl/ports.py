"""``Ports``: the named frames a step reads and the named frames it hands back.

Every input and every output of a tool travels in a ``Ports`` envelope. Each frame has a name, the
envelope cannot be changed once built, and asking for a name that is not there is refused by name.
A step's notes and its per-row status ride in the same envelope, beside the frames.

**Ports** replace a bare ``dict`` of frames in and out. A step may have more than one output -- the
rows that matched a condition and the rows that did not, say -- and a side output that a caller
forgets to read should not surface three steps later as a bare ``KeyError``. So every input and
output is named, the set of names a step declares is fixed at authoring time, and the envelope is
checked at the boundary of every call: a declared port that a step forgot to produce, an output
nobody declared, an output with no columns at all (which would make a downstream check that depends
on the column existing fail for the wrong reason), a step that handed back something other than
ports, and a step that wrote into one of its own inputs are all refused right there, by name.
"""

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType

import polars as pl

from .errors import PortError

__all__ = ["Ports"]


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
