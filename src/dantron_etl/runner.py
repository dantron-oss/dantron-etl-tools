"""``run_pipeline``: run a loaded flow and return a trace of what every step did.

The run walks the steps in dependency order. It keeps a frame for every port and records one event
per step, and it hands each event to a ``sink`` as it happens. ``until=`` runs only part of a flow,
``cached=`` resumes from frames an earlier run left, ``sources=`` hands frames to source steps, and
``retain=`` chooses between keeping every frame and freeing each one once nothing will read it.

Why: docs/design/loading-and-running.md#the-runner
"""

import logging
import time
from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol

import polars as pl

# Keep the call a bare run_tool(...): a caller may replace runner.run_tool to watch every step.
from .boundary import run_tool
from .config import with_base_dir
from .contract import Tool, call_tool_code, input_ports_for
from .errors import RunError, ToolError
from .ports import Ports
from .registry import REGISTRY, ToolRegistry
from .spec import Pipeline, PortRef
from .trace import CachedFrame, ColumnCheckError, FrameShape, Level, PipelineTrace, StepEvent, StepFailedError
from .walk import release_schedule, resolve_cached, resolve_keep, resolve_sources, walk_around_cache, walk_for

__all__ = ["LogTarget", "logging_sink", "run_pipeline"]


#: Where one step's output frames came from: the step ran, was supplied, or came from a cache.
_Origin = Literal["ran", "supplied", "cached"]

#: How each origin is worded in an event's message.
_ORIGIN_VERBS: Mapping[_Origin, str] = {
    "ran": "produced",
    "supplied": "supplied, not read",
    "cached": "read from a cache, not re-run",
}

#: The retention modes :func:`run_pipeline` accepts, named in its refusal.
_RETAIN_MODES: tuple[str, ...] = ("all", "live")


class LogTarget(Protocol):
    """Anything with ``info``, ``warning`` and ``error`` methods, such as a ``logging.Logger``.

    An orchestrator's job logger works too, even when it is not a ``logging.Logger``.
    """

    def info(self, msg: str, /) -> object: ...

    def warning(self, msg: str, /) -> object: ...

    def error(self, msg: str, /) -> object: ...


def logging_sink(logger: LogTarget | None = None) -> Callable[[StepEvent], None]:
    """Return a sink for :func:`run_pipeline` that logs each event's message as it is.

    ``logger`` is anything with ``info``, ``warning`` and ``error`` methods (see
    :class:`LogTarget`). Left out, it is this module's own logger. Each event is logged at its own
    level, so a warning such as an empty frame stands out from the info lines.

    Use it as ``run_pipeline(spec, sink=logging_sink(logger))``.
    """
    target: LogTarget = logger if logger is not None else logging.getLogger(__name__)
    methods: Mapping[Level, str] = {"info": "info", "warning": "warning", "error": "error"}

    def sink(event: StepEvent) -> None:
        getattr(target, methods[event.level])(event.message)

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
    base_dir: str | Path | None = None,
) -> PipelineTrace:
    """Run a loaded flow and return a :class:`PipelineTrace` of what every step did.

    ``spec`` is the :class:`Pipeline` that :func:`load_pipeline` returns. Every other argument is
    optional:

    - ``sources``: frames to hand to source steps instead of letting them read, keyed by
      ``"step.port"`` or :class:`PortRef`. Only a step with no inputs can be supplied, and on all of
      its ports. A source the run never reaches is ignored.
    - ``retain``: ``"all"`` (default) keeps every frame. ``"live"`` frees a frame as soon as nothing
      left in the run reads it; counts, shapes and messages are the same either way.
    - ``keep``: ports to hold on to under ``retain="live"``, as ``"step.port"``, a :class:`PortRef`,
      or a step id for all its ports. For example ``keep=["publish.receipt"]``.
    - ``until``: one step id or several. Only those steps and the steps that feed them run.
    - ``cached``: :class:`CachedFrame` entries from an earlier run, keyed by computed port. A step
      whose output ports are all cached does not run, and nor do steps only it needed.
    - ``sink``: called with each :class:`StepEvent` as it happens, for example
      :func:`logging_sink`. A sink that raises is logged once and never stops the run.
    - ``run_id``: a label copied onto the trace and every event.
    - ``base_dir``: the folder relative file paths in the flow are read from. Left out, they are read
      from the working directory. Absolute paths are never changed.

    Refused before any step runs: an unknown ``retain``; an empty ``until`` or one naming an unknown
    step; any step (reached or not) still holding a ``${...}`` placeholder; a source that names an
    unknown or computed port, is not a polars DataFrame, or covers only some of a step's ports; a
    cache entry for an unknown port or a source port, covering only some of a step's output ports,
    or whose fingerprint no longer matches; an unknown ``keep`` port; a ``base_dir`` that is not a
    folder.

    During the run, a step whose settings name a column missing from its input frames raises
    :class:`ColumnCheckError`, and a step whose tool fails raises :class:`StepFailedError`. Both
    carry the trace up to that point, and the failing step's event is sent to ``sink`` first.

    A cached frame is checked against the flow's settings, never against the data. Use ``cached``
    while building a flow, not to publish figures; :attr:`PipelineTrace.cached` says whether a run
    used one.

    Why: docs/design/loading-and-running.md#the-runner
    """
    if retain not in _RETAIN_MODES:
        raise RunError(f"retain must be one of {', '.join(_RETAIN_MODES)}; got {retain!r}")

    requested, order = walk_for(spec, until)

    registry: ToolRegistry = spec.registry if spec.registry is not None else REGISTRY
    tools_by_step: dict[str, Tool] = {step.id: registry.get(step.tool) for step in spec.steps}

    # Every step, not only the ones `until` reaches.
    # Why: docs/design/loading-and-running.md#the-placeholder-check-reads-every-step
    for step in spec.steps:
        placeholder = _find_placeholder(step.config)
        if placeholder is not None:
            raise RunError(
                f"step {step.id!r} still holds an unresolved parameter placeholder: {placeholder!r}; "
                f"a run must not start on a configuration the loader was meant to have finished resolving"
            )

    supplied_frames = resolve_sources(spec, tools_by_step, sources or {})
    cached_frames = resolve_cached(spec, tools_by_step, cached or {})
    cached_steps = {ref.step for ref in cached_frames}

    # Drop the steps only a cached step needed. Done before the release schedule reads `order`.
    order = walk_around_cache(spec, order, requested, cached_steps)

    column_refs = spec.column_references()

    kept = resolve_keep(spec, tools_by_step, keep)
    folder = Path(base_dir).resolve() if base_dir is not None else None
    if folder is not None and not folder.is_dir():
        raise RunError(f"base_dir {str(base_dir)!r} is not a folder that exists")
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
            except Exception as failure:  # noqa: BLE001 - a sink must never stop the run
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

        # Asked of the step, not the tool: a tool may take its ports from the step's settings.
        step_input_ports = input_ports_for(tool, step.config)

        # A cached step reads nothing this run, so it gathers no inputs and checks no columns.
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

        # A source is a step with no input ports, judged from the step's own ports (see above).
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
                config = dict(step.config)
                if folder is not None:
                    config = with_base_dir(call_tool_code(tool, "config_template", tool.config_template), config, folder)
                result = run_tool(tool, Ports(input_frames), config)
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
            # Only what this step marked; status never carries over from upstream.
            # Why: docs/design/loading-and-running.md#per-row-status-stays-with-the-step-that-made-it
            for port, marked in result.status.items():
                statuses[PortRef(step_id, port)] = marked

        output_shapes: list[FrameShape] = []
        # The tool's own notes come first, then anything the runner noticed.
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
            # Asks the whole flow, even under `until`: a later step is wired to read this frame.
            if frame.height == 0 and spec.consumers(ref):
                notes.append(f"output port {port_name!r} produced no rows, and a later step reads it")

        level: Level = "warning" if notes else "info"

        released: tuple[PortRef, ...] = ()
        if retain == "live":
            due = schedule.get(step_id, ())
            for ref in due:
                frames.pop(ref, None)
                # A status is freed with its frame.
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
    """Return the first string inside ``value`` that still holds ``${...}``, or ``None``.

    Looks inside lists and mappings at any depth.
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
# Checking a step's columns before its tool runs
# ---------------------------------------------------------------------------


def _column_check_message(
    step_id: str,
    path: str,
    missing: list[str],
    input_frames: Mapping[str, pl.DataFrame],
) -> str:
    """Build the refusal for a missing column: the field, the missing columns, and each port's columns.

    Columns are listed per port, because a step with two inputs cannot say which side was meant.
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
    """Build an event's message: where the rows came from, each output port's row count, and the notes.

    Every port's count is shown, so a split reports both of its sides.
    """
    counts = ", ".join(f"{shape.port.port}: {shape.rows} rows" for shape in outputs)
    verb = _ORIGIN_VERBS[origin]
    message = f"step {step_id!r} ({tool_name}) {verb}: {counts}"
    for note in notes:
        message += f"; {note}"
    return message
