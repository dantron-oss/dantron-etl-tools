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

Refusals are ordered so that as many as possible arrive before any side effect. A parameter
placeholder that survived into a configuration is refused before a single step runs, because a
refusal after several steps have already written somewhere is not a refusal. A column an
expression names is checked against a step's own input frames the moment they exist -- not for
the whole flow at once, because no tool declares the columns it produces, so an up-front pass
could only ever reach the steps fed by frames supplied before the run began. That check, and a
source supplied on only some of its ports, can therefore refuse after earlier steps have written.

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
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

import polars as pl

# The runner calls the boundary through this module's own name, run_tool. A caller may replace
# runner.run_tool to watch every step's output as the run goes; keep the call a bare run_tool(...).
from .boundary import run_tool
from .contract import Tool, input_ports_for
from .errors import RunError, ToolError
from .ports import Ports
from .registry import REGISTRY, ToolRegistry
from .spec import Pipeline, PortRef
from .trace import CachedFrame, ColumnCheckError, FrameShape, Level, PipelineTrace, StepEvent, StepFailedError
from .walk import release_schedule, resolve_cached, resolve_keep, resolve_sources, walk_around_cache, walk_for

__all__ = ["logging_sink", "run_pipeline"]


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
    spec: Pipeline,
    *,
    sources: Mapping[PortRef | str, pl.DataFrame] | None = None,
    retain: Literal["all", "live"] = "all",
    until: str | Collection[str] | None = None,
    cached: Mapping[PortRef | str, CachedFrame] | None = None,
    keep: Collection[PortRef | str] = (),
    sink: Callable[[StepEvent], None] | None = None,
    run_id: str | None = None,
) -> PipelineTrace:
    """Run ``spec`` in dependency order -- whole, or as far as ``until`` -- reporting what each step did.

    Refused before a single step runs, in this order: a retention mode that is neither ``"all"``
    nor ``"live"``; an ``until`` that is empty or names a step this pipeline does not have; a
    configuration still holding an unresolved ``${...}`` placeholder, because a refusal that
    arrives after several steps have already written somewhere is not a refusal; a supplied source
    naming a step, a port or a computed (rather than source) port that does not exist to be
    supplied, a supplied value that is not a polars DataFrame, or a source step supplied on some but
    not all of its ports.

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

    ``keep`` names ports to hold on to under ``retain="live"``, which otherwise frees a port as soon
    as nothing left in the run reads it -- including the last port of a flow, and the ports of the
    steps ``until`` names. ``retain="live", keep=["publish.receipt"]`` frees everything except the
    receipt. Each entry is a :class:`PortRef`, the text ``"step.port"``, or a step id for all of its
    ports; a name the pipeline does not have is refused before anything runs. Under ``retain="all"``
    every port is kept anyway.

    A step that raises is reported -- its error event is emitted before the exception leaves this
    function, so a crashed run still has a log of everything up to it -- and then raises
    :class:`StepFailedError`, carrying the partial trace. A sink that raises is reported once, on
    this module's own logger, and never allowed to end the run it is merely watching.
    """
    if retain not in _RETAIN_MODES:
        raise RunError(f"retain must be one of {', '.join(_RETAIN_MODES)}; got {retain!r}")

    requested, order = walk_for(spec, until)

    registry: ToolRegistry = spec.registry if spec.registry is not None else REGISTRY
    tools_by_step: dict[str, Tool] = {step.id: registry.get(step.tool) for step in spec.steps}

    for step in spec.steps:  # every step, not only the ones in `order`; see the docstring above
        placeholder = _find_placeholder(step.config)
        if placeholder is not None:
            raise RunError(
                f"step {step.id!r} still holds an unresolved parameter placeholder: {placeholder!r}; "
                f"a run must not start on a configuration the loader was meant to have finished resolving"
            )

    supplied_frames = resolve_sources(spec, tools_by_step, sources or {})
    cached_frames = resolve_cached(spec, tools_by_step, cached or {})
    cached_steps = {ref.step for ref in cached_frames}

    # The walk the cache leaves to do, which is the closure minus whatever is now only wanted by a
    # step that will not run. Computed after the cache is resolved and before anything else reads
    # `order`, so the release schedule below and the trace both describe the run that happens.
    order = walk_around_cache(spec, order, requested, cached_steps)

    column_refs = spec.column_references()

    kept = resolve_keep(spec, tools_by_step, keep)
    schedule = release_schedule(spec, tools_by_step, order, kept) if retain == "live" else {}

    frames: dict[PortRef, pl.DataFrame] = {}
    shapes: dict[PortRef, FrameShape] = {}
    statuses: dict[PortRef, pl.DataFrame] = {}
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
            status=MappingProxyType(dict(statuses)),
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
            # Every port is supplied: resolve_sources refused a partial supply before the walk.
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
            # Only what this step itself marked. A cached or supplied port has no status because no
            # step of this run produced it, and nothing is carried over from the step upstream:
            # those rows are not this step's doing.
            for port, marked in result.status.items():
                statuses[PortRef(step_id, port)] = marked

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
                # With the frame, never after it: a status outliving its rows is a set of row
                # references pointing at nothing, accumulating exactly where this mode exists to
                # stop things accumulating.
                statuses.pop(ref, None)
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
