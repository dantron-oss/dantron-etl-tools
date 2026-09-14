"""Walking a loaded flow and leaving behind the one thing a failed run can be debugged from.

The loader says what a flow *is*. This module says what happens when it is run: a frame for every
port, one event for every step, and an accessible record of both -- whichever step a run stops on.
Two readers share that one artefact rather than each getting a bespoke one. An operator watching an
orchestrator's log at three in the morning wants each event as it happens, worded so the failing one
explains itself without a stack trace. A tool built to slice and render a run afterwards wants every
frame, keyed precisely enough that a split's two sides and a pairing's three are never one entry
fighting over a single key. Both are served by the same walk: events are streamed to a sink the
moment they exist and are also kept, in order, on the trace this function returns.

Two retention modes exist because the two readers pull in opposite directions on memory. Every
intermediate frame is exactly what a real run must not hold for long, and exactly what makes a run
inspectable after the fact. ``retain="live"`` frees a port's frame the moment nothing will read it
again; ``retain="all"`` never frees one. The schedule that decides *when* is computed from the
wiring alone -- from which steps still have to consume a port -- so it is a property of the flow
being run rather than a choice a caller could get subtly wrong, and the one property both modes are
required to share is that every count, shape and message they report is identical.

Refusals are ordered so that none of them can arrive after a side effect. A parameter placeholder
that survived into a configuration is refused before a single step runs, because a refusal after
several steps have already written somewhere is not a refusal. A column an expression names is
checked against a step's own input frames the moment they exist -- not for the whole flow at once,
because no tool declares the columns it produces, so an up-front pass could only ever reach the
steps fed by frames supplied before the run began.

A source step is handed its frame rather than made to read one, because every zero-input tool in
the catalog opens a file or a database, and a run driven from tests, from a notebook, or from an
upstream asset that already has the frame in memory should not have to fabricate one for a reader
to open. A step with declared inputs may never be supplied this way: accepting a frame for a
computed port would let a caller substitute the very thing a run exists to prove happened.

A run need not be a whole flow. A pipeline is built by growing it, and a flow that can only be run
whole cannot be grown: the only way to look at step ten would be to survive step six hundred.
``until`` names the steps a run is to reach, and what runs is their ancestor closure -- the steps
that feed them, and nothing else. Everything outside that closure leaves no event, no shape and no
frame, because a step that ran and produced nothing and a step that was never asked to run are
opposite facts a reader of the trace must not have to tell apart by guessing.

``cached`` is the other end of that cut. A frame a previous run left on a computed port may be
handed back in, with the fingerprint of the closure that produced it, and then neither that step nor
the ancestors nothing else needs runs again -- which is what makes 'run the first ten tools, look at
them, write the next ten' worth doing, since the saving is the files those ten opened. The
fingerprint covers configuration and cannot cover data, so a cache is a development convenience and
a run that used one is not evidence about today's figures; the trace says on its own face that it
rested on one, so a reader who was not there can refuse it.
"""

import logging
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

import polars as pl

from .pipeline import PipelineSpec, PortRef
from .tool import REGISTRY, Ports, Tool, ToolError, ToolRegistry, input_ports_for, run_tool

__all__ = [
    "CachedFrame",
    "ColumnCheckError",
    "FrameShape",
    "Level",
    "PipelineTrace",
    "RunError",
    "StepEvent",
    "StepFailedError",
    "logging_sink",
    "run_pipeline",
]

#: How a reader should treat one event. ``"warning"`` is a step that ran and produced a frame while
#: something it declared went quiet -- an empty frame on a port a later step will read. ``"error"``
#: is a step that produced nothing, and is always the last event a run reports.
Level = Literal["info", "warning", "error"]

#: Where one step's output frames came from. Not a pair of booleans on the walk, because "ran" and
#: "was skipped for two different reasons" is one question with three answers.
_Origin = Literal["ran", "supplied", "cached"]

#: How each origin is worded in the one line an event carries. A cached step says 'cache' and says
#: nothing about being supplied: the two are different claims and a reader must not have to tell
#: them apart by tone.
_ORIGIN_VERBS: Mapping[_Origin, str] = {
    "ran": "produced",
    "supplied": "supplied, not read",
    "cached": "read from a cache, not re-run",
}

#: The retention modes :func:`run_pipeline` accepts. Not a bare string, so the refusal below can
#: name the two spellings that exist rather than merely say "invalid".
_RETAIN_MODES: tuple[str, ...] = ("all", "live")


class RunError(ToolError):
    """A run could not proceed: a bad argument, an unresolved reference, evidence asked for that a
    retention mode already freed. Stays inside the one family every tool boundary already raises so
    a caller has one type to catch, whichever layer refused.
    """


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
    :meth:`~.pipeline.PipelineSpec.fingerprint` taken over the closure that produced the port, and
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


@dataclass(frozen=True, slots=True)
class StepFailedError(RunError):
    """A step's own tool refused, mid-run. ``__cause__`` is that refusal, exactly as the tool raised
    it -- a :class:`~.tool.DataError` says go and look at the rows, an
    :class:`~.tool.ExecutionError` says go and look at the run, and flattening the two into one
    wording would erase that choice.
    """

    step_id: str
    event: StepEvent
    trace: "PipelineTrace"

    def __init__(self, step_id: str, event: StepEvent, trace: "PipelineTrace") -> None:
        RunError.__init__(self, event.message)
        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "trace", trace)


@dataclass(frozen=True, slots=True)
class ColumnCheckError(RunError):
    """A step's configuration names a column that none of its input frames carry, caught the moment
    those frames exist rather than by letting the tool discover it.
    """

    step_id: str
    event: StepEvent
    trace: "PipelineTrace"

    def __init__(self, step_id: str, event: StepEvent, trace: "PipelineTrace") -> None:
        RunError.__init__(self, event.message)
        object.__setattr__(self, "step_id", step_id)
        object.__setattr__(self, "event", event)
        object.__setattr__(self, "trace", trace)


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
    """

    pipeline: str
    retain: Literal["all", "live"]
    events: tuple[StepEvent, ...]
    shapes: Mapping[PortRef, FrameShape]
    frames: Mapping[PortRef, pl.DataFrame]
    until: tuple[str, ...] = ()
    cached: tuple[PortRef, ...] = ()
    run_id: str | None = None

    def frame(self, port: PortRef) -> pl.DataFrame:
        """The frame ``port`` produced, or a refusal naming why it cannot be handed back."""
        if port in self.frames:
            return self.frames[port]
        if port in self.shapes:
            raise RunError(
                f"port {port!s} produced a frame, but it was released under retain={self.retain!r} "
                f"before this trace was read; its shape is still on trace.shapes"
            )
        raise RunError(f"no port {port!s} on this trace; ports are: {', '.join(str(p) for p in self.shapes)}")

    def event(self, step_id: str) -> StepEvent:
        """The event ``step_id`` reported, or a refusal listing the steps that actually ran."""
        for candidate in self.events:
            if candidate.step_id == step_id:
                return candidate
        ran = ", ".join(event.step_id for event in self.events) or "none"
        raise RunError(f"no event for step {step_id!r}; steps that ran: {ran}")


def logging_sink(logger: logging.Logger | None = None) -> Callable[[StepEvent], None]:
    """A sink that writes each event's own message, verbatim, at the log level its own level names.

    Verbatim on purpose: a sink that reworded an event would give a reader a second account of the
    same run to square against the first, and the log line, the trace and a raised exception must
    stay one wording wherever any of them is read.
    """
    target = logger if logger is not None else logging.getLogger(__name__)
    levels: Mapping[Level, int] = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}

    def sink(event: StepEvent) -> None:
        target.log(levels[event.level], event.message)

    return sink


def run_pipeline(
    spec: PipelineSpec,
    *,
    sources: Mapping[PortRef | str, pl.DataFrame] | None = None,
    retain: Literal["all", "live"] = "all",
    until: str | Collection[str] | None = None,
    cached: Mapping[PortRef | str, CachedFrame] | None = None,
    sink: Callable[[StepEvent], None] | None = None,
    run_id: str | None = None,
) -> PipelineTrace:
    """Run ``spec`` in dependency order -- whole, or as far as ``until`` -- reporting what each step did.

    Refused before a single step runs, in this order: a retention mode that is neither ``"all"``
    nor ``"live"``; an ``until`` that is empty or names a step this pipeline does not have; a
    configuration still holding an unresolved ``${...}`` placeholder, because a refusal that
    arrives after several steps have already written somewhere is not a refusal; a supplied source
    naming a step, a port or a computed (rather than source) port that does not exist to be
    supplied.

    The placeholder scan reads every step, not only the ones ``until`` will reach. A placeholder
    that survived means the loader never finished resolving the document, which is a fact about the
    file rather than about this run; a scan that narrowed with ``until`` would call the same file
    clean today and unresolved tomorrow for a reason nobody had touched.

    ``until`` names one step id or several, and what runs is their ancestor closure: every step
    that feeds a named step, transitively, and the named steps themselves. A closure, emphatically
    not a prefix of the walk -- truncating the order at a position would be the easy reading and
    the wrong one, because the order is a topological sort with ties broken by declaration order,
    so a step written early sits early in the walk whether or not the named step needs it. Name the
    last step of a flow and a side branch nothing on the way to it reads still does not run. Naming
    a step together with one of its own ancestors is a union rather than a refusal, and runs
    neither of them twice; ``None`` and an omitted argument both run everything, event for event,
    as a run with no stopping point always did. A source supplied for a step the closure never
    reaches is accepted in silence, because a caller who assembles sources once for a flow and then
    truncates the run would otherwise have to work the closure out themselves, and again every time
    the wiring changed. The trace says which steps were asked for, because its reader cannot ask.

    ``cached`` is the other half of the same idea: ``until`` stops a run at a step, ``cached``
    starts one there. Each entry is a frame a previous run left on a computed port, with the
    fingerprint recorded beside it, and a step whose output ports are all cached does not run --
    nor do its ancestors, unless some other step that is running still reads them, which is the
    whole saving. Caching one port near the end of a fifty-step flow buys nothing if the
    forty-nine steps behind it still open every file they read. The cached frame becomes that
    port's frame, so everything downstream reads it exactly as though the step had run. Refused
    before the walk, by name: a port this pipeline has not got, a source port (that is what
    ``sources`` is for, and the two arguments mean different things), a step cached on some but not
    all of its output ports -- it produced them in one pass, and obtaining the rest would mean
    running it and its ancestors, which is the cost the cache was taken to avoid -- and a
    fingerprint that no longer matches the configuration in the file. An empty mapping is accepted,
    unlike an empty ``until``: 'resume from nowhere' is a complete run, which is exactly what a
    caller assembling a cache from whatever they have on hand should get when they have none. A
    cache entry the closure never reaches is accepted in silence and leaves no mark.

    The limit of that mechanism, stated plainly because it decides where a cache may be used: a
    fingerprint covers configuration, not data. It cannot cover data -- the expensive reads are
    exactly the ones an author wants to skip, so hashing what they returned would mean performing
    them. A cache built from yesterday's database pull and resumed against today's is a wrong
    number nothing here can see: every configuration matches, nothing is refused, and the figures
    are fiction. So a cache is a development convenience for building a pipeline without re-running
    it, and not a production mechanism, and a run that used one is not evidence about today's
    figures. :attr:`PipelineTrace.cached` is what lets a report or a scheduled job enforce that
    without having to ask the caller anything.

    Per step, once its turn in the walk comes: its declared input frames are gathered (already
    produced by an earlier step, or handed in through ``sources``); every column an expression in
    its configuration names is checked against those frames, the one check that can only happen
    once the frames exist; the tool is run -- or, for a fully supplied source, is not run at all;
    each output frame's shape is recorded, keyed by the port that produced it; an empty frame on a
    port a later step will read is noted as a warning, because a quiet failure and a quiet day
    otherwise read identically in a log; a port nothing will read again is released, if ``retain``
    says to; and one event is appended to the trace and, if given, passed to ``sink``.

    A step that raises is reported -- its error event is emitted before the exception leaves this
    function, so a crashed run still has a log of everything up to it -- and then raises
    :class:`StepFailedError`, carrying the partial trace. A sink that raises is reported once, on
    this module's own logger, and never allowed to end the run it is merely watching.
    """
    if retain not in _RETAIN_MODES:
        raise RunError(f"retain must be one of {', '.join(_RETAIN_MODES)}; got {retain!r}")

    requested, order = _walk_for(spec, until)

    registry: ToolRegistry = spec.registry if spec.registry is not None else REGISTRY
    tools_by_step: dict[str, Tool] = {step.id: registry.get(step.tool) for step in spec.steps}

    for step in spec.steps:  # every step, not only the ones in `order`; see the docstring above
        placeholder = _find_placeholder(step.config)
        if placeholder is not None:
            raise RunError(
                f"step {step.id!r} still holds an unresolved parameter placeholder: {placeholder!r}; "
                f"a run must not start on a configuration the loader was meant to have finished resolving"
            )

    supplied_frames = _resolve_sources(spec, tools_by_step, sources or {})
    cached_frames = _resolve_cached(spec, tools_by_step, cached or {})
    cached_steps = {ref.step for ref in cached_frames}

    # The walk the cache leaves to do, which is the closure minus whatever is now only wanted by a
    # step that will not run. Computed after the cache is resolved and before anything else reads
    # `order`, so the release schedule below and the trace both describe the run that happens.
    order = _walk_around_cache(spec, order, requested, cached_steps)

    column_refs = spec.column_references()

    schedule = _release_schedule(spec, tools_by_step, order) if retain == "live" else {}

    frames: dict[PortRef, pl.DataFrame] = {}
    shapes: dict[PortRef, FrameShape] = {}
    events: list[StepEvent] = []
    resumed_from: list[PortRef] = []
    log = logging.getLogger(__name__)

    def emit(event: StepEvent) -> None:
        events.append(event)
        if sink is not None:
            try:
                sink(event)
            except Exception as failure:  # noqa: BLE001 - telemetry must never take the run down
                log.warning("a sink raised while reporting step %r: %s", event.step_id, failure)

    def snapshot() -> PipelineTrace:
        return PipelineTrace(
            pipeline=spec.name,
            retain=retain,
            events=tuple(events),
            shapes=MappingProxyType(dict(shapes)),
            frames=MappingProxyType(dict(frames)),
            until=requested,
            cached=tuple(resumed_from),
            run_id=run_id,
        )

    for step_id in order:
        step = spec.step(step_id)
        tool = tools_by_step[step_id]
        started = time.perf_counter()

        # Asked once, and used for both gathering the frames and deciding whether this is a source.
        # A tool that derives its ports answers from this step's configuration, so two steps using
        # it can read different ports.
        step_input_ports = input_ports_for(tool, step.config)

        # A cached step reads nothing this run: the steps that would have filled its input ports did
        # not run, so there is no frame to gather and none to check a column reference against.
        # Both are skipped rather than emptied, because a missing column on a frame that was never
        # read is not a fact about this run.
        from_cache = step_id in cached_steps

        input_shapes: list[FrameShape] = []
        input_frames: dict[str, pl.DataFrame] = {}
        column_checks: Mapping[str, frozenset[str]] = {}
        if not from_cache:
            for port_name in step_input_ports:
                source_ref = step.inputs[port_name]
                input_shapes.append(shapes[source_ref])
                input_frames[port_name] = frames[source_ref]
            column_checks = column_refs.get(step_id, {})

        for path, names in column_checks.items():
            available = {name for frame in input_frames.values() for name in frame.columns}
            missing = sorted(name for name in names if name not in available)
            if missing:
                message = _column_check_message(step_id, path, missing, input_frames)
                event = StepEvent(
                    pipeline=spec.name,
                    step_id=step_id,
                    tool=tool.name,
                    level="error",
                    inputs=tuple(input_shapes),
                    outputs=(),
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    at=datetime.now(UTC),
                    message=message,
                    lane=step.lane,
                    stage=step.stage,
                    error=message,
                    run_id=run_id,
                )
                emit(event)
                raise ColumnCheckError(step_id, event, snapshot())

        # A source reads nothing. That is a question about this step, not about its tool: a tool
        # that derives its ports declares none statically, and a step of it that names three parts
        # reads three frames. Asking the attribute here would call such a step a source, let a run
        # hand it frames, and skip gathering the inputs it really has.
        is_source_step = not step_input_ports
        supplied_refs = [
            PortRef(step_id, port) for port in tool.output_ports if PortRef(step_id, port) in supplied_frames
        ]

        if from_cache:
            origin: _Origin = "cached"
            reported: tuple[str, ...] = ()
            output_frames = {port: cached_frames[PortRef(step_id, port)].frame for port in tool.output_ports}
        elif is_source_step and supplied_refs:
            if len(supplied_refs) != len(tool.output_ports):
                missing_ports = [
                    str(PortRef(step_id, port))
                    for port in tool.output_ports
                    if PortRef(step_id, port) not in supplied_frames
                ]
                raise RunError(
                    f"step {step_id!r} was supplied some but not all of its output ports; still "
                    f"missing: {', '.join(missing_ports)}"
                )
            origin = "supplied"
            reported = ()
            output_frames = {port: supplied_frames[PortRef(step_id, port)] for port in tool.output_ports}
        else:
            origin = "ran"
            try:
                result = run_tool(tool, Ports(input_frames), dict(step.config))
            except ToolError as failure:
                message = f"step {step_id!r} ({tool.name}) failed: {failure}"
                event = StepEvent(
                    pipeline=spec.name,
                    step_id=step_id,
                    tool=tool.name,
                    level="error",
                    inputs=tuple(input_shapes),
                    outputs=(),
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    at=datetime.now(UTC),
                    message=message,
                    lane=step.lane,
                    stage=step.stage,
                    error=message,
                    run_id=run_id,
                )
                emit(event)
                raise StepFailedError(step_id, event, snapshot()) from failure
            reported = result.notes
            output_frames = {port: result[port] for port in tool.output_ports}

        output_shapes: list[FrameShape] = []
        # What the step itself wants said, ahead of anything this runner noticed: a tool knows
        # things about its own work that no observer of its ports could reconstruct -- how many
        # values it could not read, and out of how many.
        notes: list[str] = list(reported)

        for port_name in tool.output_ports:
            frame = output_frames[port_name]
            ref = PortRef(step_id, port_name)
            shape = FrameShape(port=ref, rows=frame.height, columns=tuple(frame.columns))
            shapes[ref] = shape
            frames[ref] = frame
            output_shapes.append(shape)
            if from_cache:
                resumed_from.append(ref)
            # Asked of the whole wiring even when the walk is a closure, unlike the release
            # schedule just below. A note is a fact about the flow -- an empty frame something is
            # wired to read is worth saying whether or not this particular run gets that far, and
            # is exactly what an author growing a pipeline a few steps at a time needs told.
            # Releasing a frame is a fact about this run, so it asks the closure instead.
            if frame.height == 0 and spec.consumers(ref):
                notes.append(f"output port {port_name!r} produced no rows, and a later step reads it")

        level: Level = "warning" if notes else "info"

        released: tuple[PortRef, ...] = ()
        if retain == "live":
            due = schedule.get(step_id, ())
            for ref in due:
                frames.pop(ref, None)
            released = due

        event = StepEvent(
            pipeline=spec.name,
            step_id=step_id,
            tool=tool.name,
            level=level,
            inputs=tuple(input_shapes),
            outputs=tuple(output_shapes),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            at=datetime.now(UTC),
            message=_outcome_message(step_id, tool.name, origin, output_shapes, notes),
            lane=step.lane,
            stage=step.stage,
            supplied=origin == "supplied",
            cached=origin == "cached",
            released=released,
            notes=tuple(notes),
            run_id=run_id,
        )
        emit(event)

    return snapshot()


# ---------------------------------------------------------------------------
# Refused before the walk starts
# ---------------------------------------------------------------------------


def _find_placeholder(value: object) -> str | None:
    """The first string anywhere inside ``value`` that still holds ``${...}``, or ``None``.

    Walks a list or a mapping at any depth, the same shape a configuration is built from, because a
    placeholder buried inside an ordered rule is exactly as unresolved as one at the top.
    """
    if isinstance(value, str):
        return value if "${" in value else None
    if isinstance(value, list):
        for item in value:
            found = _find_placeholder(item)
            if found is not None:
                return found
        return None
    if isinstance(value, Mapping):
        for item in value.values():
            found = _find_placeholder(item)
            if found is not None:
                return found
        return None
    return None


def _walk_for(spec: PipelineSpec, until: str | Collection[str] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(the steps asked for, the steps to walk)``, both in dependency order.

    A whole run asks for nothing and walks everything. A truncated one walks the ancestor closure of
    what it named, followed through :attr:`~.pipeline.StepSpec.inputs`, which is the only edge list
    a flow has. Neither tuple is ordered here: both are :meth:`~.pipeline.PipelineSpec.order`
    filtered down to a set of ids. So a step that runs sits in exactly the position it occupies in a
    whole run, a step reached from two directions is still walked once, and what the trace reports
    as asked for is in the same order as what it reports as run -- whatever order, and whatever kind
    of container, the caller wrote them in.

    Both refusals happen here, before a caller can have done anything: a step id this pipeline does
    not have is a typo, and silently running nothing or silently running everything are equally
    defensible readings of one, each leaving an author looking at a result that does not answer the
    question they asked. An empty collection is refused for the same reason -- it almost always
    arrives from a comprehension over something that turned out empty, and a trace with no events
    reads identically to a run that failed on its first step.
    """
    order = spec.order()
    if until is None:
        return (), order

    named = (until,) if isinstance(until, str) else tuple(until)
    known = ", ".join(step.id for step in spec.steps) or "none"
    if not named:
        raise RunError(
            "until was given an empty collection, so this run would have had nothing to do; pass "
            f"None to run the whole pipeline, or name at least one of: {known}"
        )

    known_ids = {step.id for step in spec.steps}
    for step_id in named:
        if step_id not in known_ids:
            raise RunError(f"until names step {step_id!r}, which this pipeline does not have; steps are: {known}")

    closure: set[str] = set()
    pending = list(named)
    while pending:
        step_id = pending.pop()
        if step_id in closure:
            continue
        closure.add(step_id)
        pending.extend(ref.step for ref in spec.step(step_id).inputs.values())

    requested = set(named)
    return (
        tuple(step_id for step_id in order if step_id in requested),
        tuple(step_id for step_id in order if step_id in closure),
    )


def _named_port(
    spec: PipelineSpec,
    tools_by_step: Mapping[str, Tool],
    key: PortRef | str,
    argument: Literal["sources", "cached"],
) -> tuple[PortRef, Tool]:
    """One key of ``sources`` or of ``cached`` read as a port of this flow, with the tool behind it.

    Two arguments hand frames to a run, and a typo has to read the same way in either: the spelling
    of a key -- a :class:`PortRef` or the text a file would carry -- a step this pipeline has not
    got, and a port that step does not declare are the same three mistakes whichever argument they
    were written in, so they are worded once here and told apart only by ``argument``, which names
    the one the caller actually wrote.

    What the two arguments disagree about is left to them: whether the step behind the port reads
    anything is the question they want opposite answers to, and is the whole distinction between
    stating a run's input and standing in for a computation.
    """
    if isinstance(key, PortRef):
        ref = key
    elif isinstance(key, str) and "." in key:
        step_id, port_name = key.split(".", 1)
        ref = PortRef(step_id, port_name)
    else:
        raise RunError(f"a {argument} key must be a PortRef or a 'step.port' string; got {key!r}")

    tool = tools_by_step.get(ref.step)
    if tool is None:
        steps = ", ".join(step.id for step in spec.steps) or "none"
        raise RunError(f"{argument} names step {ref.step!r}, which this pipeline does not have; steps are: {steps}")
    if ref.port not in tool.output_ports:
        ports = ", ".join(tool.output_ports) if tool.output_ports else "none"
        raise RunError(
            f"{argument} names {ref!s}, but step {ref.step!r} (tool {tool.name!r}) has no output "
            f"port {ref.port!r}; its output ports are: {ports}"
        )
    return ref, tool


def _resolve_sources(
    spec: PipelineSpec,
    tools_by_step: Mapping[str, Tool],
    sources: Mapping[PortRef | str, pl.DataFrame],
) -> dict[PortRef, pl.DataFrame]:
    """``sources`` keyed by :class:`PortRef`, refusing anything that is not a real source port.

    A source port is an output of a step that reads nothing -- the only kind of step a run can hand
    a frame to without the frame it hands over silently becoming the flow that actually ran. What a
    step reads is asked of its tool with its own configuration, because a tool that derives its
    ports names none in its attribute even when the step reads three.
    """
    resolved: dict[PortRef, pl.DataFrame] = {}
    for key, frame in sources.items():
        ref, tool = _named_port(spec, tools_by_step, key, "sources")
        step_input_ports = input_ports_for(tool, spec.step(ref.step).config)
        if step_input_ports:
            raise RunError(
                f"sources supplies {ref!s}, but step {ref.step!r} is computed, not a source: its "
                f"tool ({tool.name!r}) reads input port(s) {', '.join(step_input_ports)}, so handing "
                f"it a frame would make the flow that ran a different flow from the one in the file"
            )
        resolved[ref] = frame
    return resolved


def _resolve_cached(
    spec: PipelineSpec,
    tools_by_step: Mapping[str, Tool],
    cached: Mapping[PortRef | str, CachedFrame],
) -> dict[PortRef, CachedFrame]:
    """``cached`` keyed by :class:`PortRef`, refusing every entry that cannot stand in for a step.

    Structure first and fingerprints after, because a fingerprint cannot be taken of a step that
    does not exist, and a refusal about a stale cache would send an author to the wrong problem.

    A source port is refused, and the refusal names ``sources``. The two arguments do not mean the
    same thing: a supplied source is the caller stating this run's input, checked against nothing
    because there is nothing upstream of it to check, while a cached port is a claim that a
    computation which already happened may be skipped. One frame with two doors and two different
    guards would leave every question afterwards -- was this run resumed, is this trace evidence --
    answerable both ways at once.

    A step cached on some but not all of its output ports is refused for a plainer reason: it
    produced all of them in one pass, so obtaining the rest would mean running the step, and its
    ancestors, and the reads they open, which is the entire cost the cache was taken to avoid --
    and would then leave two answers for one step, one of them from a previous run.
    """
    resolved: dict[PortRef, CachedFrame] = {}
    for key, entry in cached.items():
        ref, tool = _named_port(spec, tools_by_step, key, "cached")
        if not input_ports_for(tool, spec.step(ref.step).config):
            raise RunError(
                f"cached names {ref!s}, but step {ref.step!r} is a source rather than a computation: "
                f"there is nothing upstream of it that resuming could skip, and a frame for a source "
                f"port is what sources takes"
            )
        resolved[ref] = entry

    for step_id in dict.fromkeys(ref.step for ref in resolved):
        tool = tools_by_step[step_id]
        missing = [str(PortRef(step_id, port)) for port in tool.output_ports if PortRef(step_id, port) not in resolved]
        if missing:
            raise RunError(
                f"step {step_id!r} was cached on some but not all of its output ports; still "
                f"missing: {', '.join(missing)}. A step produces every one of its ports in one pass, "
                f"so a run would have to perform it -- and everything behind it -- to obtain the rest"
            )

    for ref, entry in resolved.items():
        if entry.fingerprint != spec.fingerprint(ref.step):
            raise RunError(
                f"cached port {ref!s} cannot be used: the configuration that produced it has "
                f"changed. The fingerprint recorded beside the frame is not the one step "
                f"{ref.step!r} and its ancestors produce now, so those rows came from a pipeline "
                f"that is no longer in the file. Take the cache again, or drop it"
            )
    return resolved


def _walk_around_cache(
    spec: PipelineSpec,
    order: tuple[str, ...],
    requested: tuple[str, ...],
    cached_steps: Collection[str],
) -> tuple[str, ...]:
    """``order`` with every step dropped that was only there to produce something now cached.

    The ancestors are the whole point of the feature. Skipping the cached step alone would save one
    step out of fifty, still open every file the flow reads, take just as long, and report that it
    had resumed.

    So the walk is recomputed backwards from what the run is *for*: the steps ``until`` named, or,
    for a whole run, the steps nothing else in the walk reads. From each, every input edge is
    followed -- except through a cached step, which reads nothing this run and so ends the backward
    walk there. What survives is every step some running step still needs, which is why an ancestor
    two further steps read is not dropped merely because one of its consumers was cached. A cache
    that saves nothing at all is left alone rather than refused: it is not a mistake, and a refusal
    would send an author hunting for one they have not made.
    """
    if not cached_steps:
        return order

    in_walk = set(order)
    read_in_walk = {edge.step for step_id in order for edge in spec.step(step_id).inputs.values()}
    roots = list(requested) if requested else [step_id for step_id in order if step_id not in read_in_walk]

    needed: set[str] = set()
    pending = list(roots)
    while pending:
        step_id = pending.pop()
        if step_id in needed:
            continue
        needed.add(step_id)
        if step_id in cached_steps:
            continue
        pending.extend(ref.step for ref in spec.step(step_id).inputs.values() if ref.step in in_walk)

    return tuple(step_id for step_id in order if step_id in needed)


# ---------------------------------------------------------------------------
# Per-step column pre-flight
# ---------------------------------------------------------------------------


def _column_check_message(
    step_id: str,
    path: str,
    missing: list[str],
    input_frames: Mapping[str, pl.DataFrame],
) -> str:
    """Name the field, the missing column(s), and what each input port actually carries.

    Per port rather than as one merged list, because a step with more than one input -- a pairing's
    two sides -- cannot say which side was meant, and showing the columns of each individually is
    what makes that imprecision visible rather than papered over.
    """
    per_port = "; ".join(
        f"port {name!r}: {', '.join(frame.columns) if frame.columns else '(no columns)'}"
        for name, frame in input_frames.items()
    )
    return (
        f"step {step_id!r} field {path!r} references column(s) {', '.join(missing)} that are not "
        f"on any of its input frames ({per_port or '(no inputs)'})"
    )


# ---------------------------------------------------------------------------
# Reporting what a step did
# ---------------------------------------------------------------------------


def _outcome_message(
    step_id: str,
    tool_name: str,
    origin: _Origin,
    outputs: list[FrameShape],
    notes: list[str],
) -> str:
    """One line naming every output port's row count, because a split's two numbers are the fact
    somebody needs and a single figure for the step would hide half of it.

    Three verbs, because where the rows came from is the first thing a reader of a log needs and
    the three answers mean different things: a step ran, a source was handed its frame, or a
    previous run's answer stood in for a step this one did not perform.
    """
    counts = ", ".join(f"{shape.port.port}: {shape.rows} rows" for shape in outputs)
    verb = _ORIGIN_VERBS[origin]
    message = f"step {step_id!r} ({tool_name}) {verb}: {counts}"
    for note in notes:
        message += f"; {note}"
    return message


# ---------------------------------------------------------------------------
# The release schedule -- a property of the wiring, computed once per run
# ---------------------------------------------------------------------------


def _release_schedule(
    spec: PipelineSpec,
    tools_by_step: Mapping[str, Tool],
    order: tuple[str, ...],
) -> dict[str, tuple[PortRef, ...]]:
    """``step id -> ports to free once that step has finished``, for ``retain="live"``.

    A port is due at the last step that still reads it -- read off :meth:`PipelineSpec.consumers`,
    a property of the wiring alone. ``order`` is what decides *last*, and under a truncated run
    that is the closure rather than the whole flow: a consumer that will never run cannot be a
    port's last use, and counting one would leave a source frame -- the largest in the run -- due
    at a step the walk never reaches, which is to say never freed at all. A port nothing remaining
    reads is due at the very step that produced it, for the same reason: a terminal frame kept
    "just in case" is the largest one in the run, and this is what keeps it from being kept.

    Both halves of that are the one question -- is this step in the walk? -- asked of a producer by
    walking ``order`` rather than the spec, and of a consumer by looking its step up in ``position``.
    """
    position = {step_id: index for index, step_id in enumerate(order)}
    schedule: dict[str, list[PortRef]] = {step_id: [] for step_id in order}
    for step_id in order:
        for port_name in tools_by_step[step_id].output_ports:
            ref = PortRef(step_id, port_name)
            read_at = [position[consumer.step] for consumer in spec.consumers(ref) if consumer.step in position]
            due_at = max(read_at, default=position[step_id])
            schedule[order[due_at]].append(ref)
    return {step_id: tuple(refs) for step_id, refs in schedule.items()}
