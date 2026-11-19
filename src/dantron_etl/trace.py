"""What a run leaves behind: one event per step, and the trace that holds them all.

A ``StepEvent`` says what one step did: the shapes it read and wrote, how long it took, and one
line a log can print. A ``PipelineTrace`` holds every event of a run, the shape of every port, and
the frames the run kept. ``StepFailedError`` and ``ColumnCheckError`` carry the partial trace of a
run that stopped.
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


#: How a reader should treat one event. ``"warning"`` is a step that ran and produced a frame but
#: has something to say: a note its tool reported, or an empty frame on a port a later step will
#: read. ``"error"`` is a step that produced nothing, and is always the last event a run reports.
Level = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class FrameShape:
    """A named frame's size, reported against the port that produced it rather than the step.

    ``columns`` carries names rather than a count: a released port's frame is gone under
    ``retain="live"``, but its schema is still worth drawing, and a width alone cannot be drawn.
    """

    port: PortRef
    rows: int
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CachedFrame:
    """A frame a previous run left on a computed port, and the fingerprint recorded beside it.

    Two fields because neither is any use alone. The frame without the fingerprint is a number with
    no provenance, and the danger a cache carries is not one that fails to load -- it is one that
    loads perfectly against a pipeline somebody has since edited. The fingerprint is
    :meth:`~.spec.Pipeline.fingerprint` of the port, taken over the closure that produced it, and
    :func:`run_pipeline` compares it before it walks a step.

    Frozen because it is evidence of a run that already happened, not a value the run holding it may
    edit on its way past.
    """

    frame: pl.DataFrame
    fingerprint: str


@dataclass(frozen=True, slots=True)
class StepEvent:
    """What one step did, for a reader who was not there when it ran.

    ``inputs`` and ``outputs`` are one shape per declared port, in the tool's own declared order --
    a pairing reports one input shape per side and one output shape per branch, never a single
    number standing in for all of them. ``message`` is the whole sentence a log line or an exception
    reads; nothing here is assembled twice into two different wordings.

    ``supplied`` and ``cached`` are separate facts and are never both true. A supplied source is
    the routine case -- a run handed the frame a reader would otherwise have opened. A cached step
    is a claim about a *previous* run, whose ancestors did not run in this one, so its event reports
    no input shapes: it read nothing, and reporting what it would have read would be inventing
    evidence.
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
    """A step failed mid-run. ``__cause__`` is the tool's own refusal, exactly as it was raised.

    A :class:`~.errors.DataError` cause says go and look at the rows; an
    :class:`~.errors.ExecutionError` says go and look at the run.

    ``step_id``, ``event`` and ``trace`` are read-only. ``trace`` is the run up to the failure,
    frames included, so a caller can still save or show what ran before it.

    A plain exception class, not a frozen dataclass: Python and pytest write to an exception's
    ``__traceback__`` and notes, and a frozen dataclass refuses those writes with a second error.
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
    """A step's configuration names a column that none of its input frames carry.

    Caught the moment those frames exist, before the tool runs. A kind of :class:`StepFailedError`,
    so ``except StepFailedError`` catches it too.
    """


@dataclass(frozen=True, slots=True)
class PipelineTrace:
    """One run: every frame it produced (subject to ``retain``), every shape, every event, in order.

    ``shapes`` is complete regardless of ``retain`` -- how large a released frame was is exactly
    what a failure needs -- while ``frames`` holds only what the retention mode kept. Both are
    read-only mappings: a trace is evidence of what happened, not a value a reader can rewrite.

    ``until`` is what the run was asked to reach, and is empty for a whole run. A reader of a trace
    is not the caller that produced it and cannot ask it anything, so a partial run has to say on
    its own face that it was one: rendering a seven-step closure of a nine-step flow as the flow
    would quietly tell somebody that two steps do not exist. The steps are reported in dependency
    order, whatever order or container the caller wrote them in -- a set has no order of its own,
    and a field on a piece of evidence that renders differently run to run is not evidence.

    ``cached`` names the ports this run took from a cache instead of computing, in the order it
    reached them, and is empty for a run that computed everything it reports. It is the only thing
    that tells an honest run from a resumed one, because a resumed run's figures are as plausible
    as any other's -- so a report, a page or a scheduled job that must not publish a resumed run
    reads this field, and a caller who merely offered a cache the run never reached does not
    trip it.

    ``status`` holds the per-row mark a step reported beside a port it degraded -- one row per row
    of that port, in the same order -- and holds nothing for every port that degraded nothing, which
    on a real flow is almost all of them. It does not travel downstream: a step handing on the very
    rows an earlier step emptied reports nothing, because a status is a fact about what a step did
    and that step did nothing to those values. Propagating it would turn one degradation into a mark
    on every step after it and make the last step of a flow look like the culprit. It is released in
    lockstep with the frame it describes, since a status kept after its rows were freed is a row
    reference with nothing to point at -- ``shapes``, which is a summary rather than a reference,
    survives as it always did.
    """

    pipeline: str
    retain: Literal["all", "live"]
    events: tuple[StepEvent, ...]
    shapes: Mapping[PortRef, FrameShape]
    frames: Mapping[PortRef, pl.DataFrame]
    status: Mapping[PortRef, pl.DataFrame] = MappingProxyType({})
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
        # The read-only mappings cannot be pickled as they are, so they travel as plain dicts and are
        # wrapped again on the way back in.
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
        """The frame ``port`` produced, or a refusal naming why it cannot be handed back.

        ``port`` is a :class:`PortRef`, the text ``"step.port"``, or a step id alone when that step
        produced exactly one port.
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
        """``port`` as a :class:`PortRef`, read from any of the three spellings :meth:`frame` takes."""
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
        """The event ``step_id`` reported, or a refusal listing the steps that actually ran."""
        found = self._by_step.get(step_id)
        if found is not None:
            return found
        raise RunError(f"no event for step {step_id!r}; {name_choices(step_id, self._by_step, 'steps that ran')}")

    def event_or_none(self, step_id: str) -> StepEvent | None:
        """The event ``step_id`` reported, or ``None`` when it did not run."""
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
    """A trace from its pickled parts. See :meth:`PipelineTrace.__reduce__`."""
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
