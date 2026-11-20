"""``Ports``: the named frames a step reads, and the named frames it returns.

Every input and output of a tool travels in a ``Ports``. Each frame has a name, a ``Ports`` cannot be
changed once built, and asking for a missing name is refused with the names that are there. A
step's notes and per-row status travel in the same ``Ports``, beside the frames.

Why: docs/design/core.md#named-ports
"""

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType

import polars as pl

from .errors import PortError

__all__ = ["Ports"]


class Ports:
    """An ordered, read-only set of named frames: a step's inputs, or its outputs.

    Build one with ``Ports({"rows": frame})``, optionally with ``notes=[...]`` and
    ``status={"rows": status_frame}``. Read a frame with ``ports["rows"]``; a missing name raises
    :class:`PortError`. The constructor refuses a value that is not a polars ``DataFrame``.

    Why: docs/design/core.md#ports-is-a-closed-read-only-class
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
        """Messages for a reader about something the step did, short of refusing.

        For example, "3 values could not be read as dates and were left empty". The runner turns a
        step with notes into a warning on the log and the canvas. Nothing downstream may read them.

        Why: docs/design/core.md#notes-and-per-row-status
        """
        return self._notes

    @property
    def status(self) -> Mapping[str, pl.DataFrame]:
        """Which rows a step changed, per output port that has any. Empty for most steps.

        Each status frame has one row per data row, in the same order. A note says how many values
        were changed; the status says which rows. It travels beside the frame, never as a column
        in it.

        Why: docs/design/core.md#notes-and-per-row-status
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
