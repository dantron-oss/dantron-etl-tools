"""What a run leaves behind: one event per step, and the trace that holds them all.

A ``StepEvent`` says what one step did: the shapes it read and wrote, how long it took, and a
message a log can print. A ``PipelineTrace`` holds every event of a run, the shape of every port,
and the frames the run kept. ``StepFailedError`` and ``ColumnCheckError`` carry the trace of a run
that stopped part way.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

import polars as pl

from .errors import RunError, name_choices
from .spec import PortRef

__all__ = [
    "CachedFrame",
    "ColumnCheckError",
    "FrameShape",
    "Level",
    "PipelineTrace",
    "StepEvent",
    "StepFailedError",
]


#: How serious one event is. ``"warning"``: the step ran but has something to say, such as a note
#: from its tool or an empty frame a later step will read. ``"error"``: the step produced nothing;
#: it is always the last event of a run.
Level = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class FrameShape:
    """The size of the frame on one port: its row count and its column names.

    Column names are kept, not only a count, so a page can still draw a port whose frame was
    freed under ``retain="live"``.
    """

    port: PortRef
    rows: int
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CachedFrame:
    """A frame from an earlier run, with the fingerprint of the steps that produced it.

    Build one with ``CachedFrame(frame, pipeline.fingerprint(port))`` and pass it to
    :func:`run_pipeline` in ``cached=``. The run refuses it if the fingerprint no longer matches
    the flow, so a frame made by a since-edited flow is never reused.

    Why: docs/design/loading-and-running.md#a-cached-frame-carries-its-fingerprint
    """

    frame: pl.DataFrame
    fingerprint: str


@dataclass(frozen=True, slots=True)
class StepEvent:
    """What one step did in a run.

    - ``inputs`` and ``outputs``: one :class:`FrameShape` per port, in the tool's port order.
    - ``message``: the sentence a log line or an exception prints.
    - ``level``: ``"info"``, ``"warning"`` or ``"error"``.
    - ``supplied``: the run was handed this step's frame instead of reading its source.
    - ``cached``: the step's frames came from an earlier run. Its ``inputs`` is empty, because it
      read nothing in this run. ``supplied`` and ``cached`` are never both true.
    - ``released``: ports whose frames were freed after this step under ``retain="live"``.
    """

    pipeline: str
    step_id: str
    tool: str
    level: Level
    inputs: tuple[FrameShape, ...]
    outputs: tuple[FrameShape, ...]
    duration_ms: float
    at: datetime
    message: str
    lane: str | None = None
    stage: str | None = None
    supplied: bool = False
    cached: bool = False
    released: tuple[PortRef, ...] = ()
    notes: tuple[str, ...] = ()
    error: str | None = None
    run_id: str | None = None


class StepFailedError(RunError):
    """Raised when a step fails during a run.

    ``__cause__`` is the tool's own error, unchanged. A :class:`~.errors.DataError` means look at
    the rows; an :class:`~.errors.ExecutionError` means look at the run.

    ``step_id``, ``event`` and ``trace`` are read-only. ``trace`` is the run up to the failure,
    frames included, so a caller can still save or show what ran before it.
    """

    def __init__(self, step_id: str, event: StepEvent, trace: "PipelineTrace") -> None:
        super().__init__(event.message)
        self._step_id = step_id
        self._event = event
        self._trace = trace

    @property
    def step_id(self) -> str:
        return self._step_id

    @property
    def event(self) -> StepEvent:
        return self._event

    @property
    def trace(self) -> "PipelineTrace":
        return self._trace

    def __reduce__(self) -> tuple[Any, ...]:
        # Rebuilt from its three fields, so it can be copied and sent to another process.
        return (type(self), (self._step_id, self._event, self._trace))


class ColumnCheckError(StepFailedError):
    """Raised when a step's settings name a column that none of its input frames have.

    It is raised before the tool runs. ``except StepFailedError`` catches it too.
    """


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    """Everything one run left behind. :func:`run_pipeline` returns it.

    Read a port's frame with ``trace.frame("step.port")`` and a step's event with
    ``trace.event("step")``. Both refuse a name the run did not produce, listing the choices.

    - ``events``: one :class:`StepEvent` per step, in the order the steps ran.
    - ``shapes``: the size of every port's frame, kept whatever ``retain`` says.
    - ``frames``: the frames the run kept. Under ``retain="live"`` most are freed.
    - ``status``: per-row status for a port where a step emptied or changed values. Ports with
      nothing to report have no entry. A status is freed with its frame.
    - ``until``: the steps a partial run was asked to reach, in dependency order. Empty for a
      whole run.
    - ``cached``: the ports this run took from a cache instead of computing. Empty when the run
      computed everything. Check it before publishing a resumed run's figures.

    The mappings are read-only.

    Why: docs/design/loading-and-running.md#what-the-trace-says-about-itself
    """

    pipeline: str
    retain: Literal["all", "live"]
    events: tuple[StepEvent, ...]
    shapes: Mapping[PortRef, FrameShape]
    frames: Mapping[PortRef, pl.DataFrame]
    status: Mapping[PortRef, pl.DataFrame] = field(default_factory=lambda: MappingProxyType({}))
    until: tuple[str, ...] = ()
    cached: tuple[PortRef, ...] = ()
    run_id: str | None = None
    # Each step's event by its id, built once so looking one up does not scan the run.
    _by_step: Mapping[str, StepEvent] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        by_step: dict[str, StepEvent] = {}
        for event in self.events:
            by_step.setdefault(event.step_id, event)
        object.__setattr__(self, "_by_step", MappingProxyType(by_step))

    def __reduce__(self) -> tuple[Any, ...]:
        # Read-only mappings cannot be pickled, so they travel as plain dicts.
        return (
            _rebuilt_trace,
            (
                self.pipeline,
                self.retain,
                self.events,
                dict(self.shapes),
                dict(self.frames),
                dict(self.status),
                self.until,
                self.cached,
                self.run_id,
            ),
        )

    def frame(self, port: PortRef | str) -> pl.DataFrame:
        """Return the frame on ``port``.

        ``port`` is a :class:`PortRef`, the text ``"step.port"``, or a step id alone when that step
        has one port. Refuses a port the run did not produce, or one whose frame was freed under
        ``retain="live"`` (pass it in ``keep=`` to hold on to it).
        """
        ref = self._port(port)
        if ref in self.frames:
            return self.frames[ref]
        if ref in self.shapes:
            raise RunError(
                f"port {ref!s} produced a frame, but it was released under retain={self.retain!r} "
                f"before this trace was read; its shape is still on trace.shapes. Pass it in keep= to "
                f"hold on to it"
            )
        raise RunError(f"no port {ref!s} on this trace; {name_choices(str(ref), map(str, self.shapes), 'ports')}")

    def _port(self, port: PortRef | str) -> PortRef:
        """Turn any of the three spellings :meth:`frame` takes into a :class:`PortRef`."""
        if isinstance(port, PortRef):
            return port
        if not isinstance(port, str):
            raise RunError(f"a port is a PortRef or the text 'step.port'; got {port!r}")
        if "." in port:
            step_id, port_name = port.split(".", 1)
            return PortRef(step_id, port_name)
        produced = [ref for ref in self.shapes if ref.step == port]
        if len(produced) == 1:
            return produced[0]
        if produced:
            raise RunError(
                f"step {port!r} produced {len(produced)} ports, so name one: "
                f"{', '.join(str(ref) for ref in produced)}"
            )
        raise RunError(f"no step {port!r} produced a port on this trace; {name_choices(port, self._by_step, 'steps that ran')}")

    def event(self, step_id: str) -> StepEvent:
        """Return the event ``step_id`` reported. Refuses a step that did not run, listing those that did."""
        found = self._by_step.get(step_id)
        if found is not None:
            return found
        raise RunError(f"no event for step {step_id!r}; {name_choices(step_id, self._by_step, 'steps that ran')}")

    def event_or_none(self, step_id: str) -> StepEvent | None:
        """Return the event ``step_id`` reported, or ``None`` when it did not run."""
        return self._by_step.get(step_id)

def _rebuilt_trace(
    pipeline: str,
    retain: Literal["all", "live"],
    events: tuple[StepEvent, ...],
    shapes: dict[PortRef, FrameShape],
    frames: dict[PortRef, pl.DataFrame],
    status: dict[PortRef, pl.DataFrame],
    until: tuple[str, ...],
    cached: tuple[PortRef, ...],
    run_id: str | None,
) -> PipelineTrace:
    """Rebuild a trace from its pickled parts. See :meth:`PipelineTrace.__reduce__`."""
    return PipelineTrace(
        pipeline=pipeline,
        retain=retain,
        events=events,
        shapes=MappingProxyType(shapes),
        frames=MappingProxyType(frames),
        status=MappingProxyType(status),
        until=until,
        cached=cached,
        run_id=run_id,
    )
