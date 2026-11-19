"""Plan a run before any step of it runs.

``run_pipeline`` asks this module which steps to run (all of them, or what ``until`` needs, less
what ``cached`` makes unnecessary), checks the frames handed in through ``sources``, ``cached``
and ``keep``, and works out when each frame can be freed under ``retain="live"``. Every refusal
here happens before the first step runs.
"""

from collections.abc import Collection, Mapping
from typing import Literal

import polars as pl

from .contract import Tool, input_ports_for
from .errors import RunError, name_choices
from .spec import Pipeline, PortRef
from .trace import CachedFrame

__all__ = ["release_schedule", "resolve_cached", "resolve_keep", "resolve_sources", "walk_around_cache", "walk_for"]


def walk_for(spec: Pipeline, until: str | Collection[str] | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(steps asked for, steps to run)``, both in the order the whole flow runs them.

    With ``until=None`` nothing is asked for and every step runs. Otherwise ``until`` is one step
    id or several, and the steps to run are those and every step that feeds them. Refuses an
    unknown step id and an empty collection.

    Why: docs/design/loading-and-running.md#until-refuses-an-unknown-or-empty-list
    """
    order = spec.order()
    if until is None:
        return (), order

    named = (until,) if isinstance(until, str) else tuple(until)
    known_ids = [step.id for step in spec.steps]
    if not named:
        raise RunError(
            "until was given an empty collection, so this run would have had nothing to do; pass "
            "None to run the whole pipeline, or name at least one step"
        )

    known = set(known_ids)
    for step_id in named:
        if step_id not in known:
            raise RunError(
                f"until names step {step_id!r}, which this pipeline does not have; {name_choices(step_id, known_ids)}"
            )

    closure = spec.ancestor_closure(named)

    requested = set(named)
    return (
        tuple(step_id for step_id in order if step_id in requested),
        tuple(step_id for step_id in order if step_id in closure),
    )


def _named_port(
    spec: Pipeline,
    tools_by_step: Mapping[str, Tool],
    key: PortRef | str,
    argument: Literal["sources", "cached", "keep"],
) -> tuple[PortRef, Tool]:
    """Read one key of ``sources``, ``cached`` or ``keep`` as a port of this flow.

    Returns the :class:`PortRef` and the tool of its step. Refuses a key that is not a
    :class:`PortRef` or ``"step.port"``, an unknown step, and a port the step does not have. The
    refusals name ``argument``, so they read the same for each argument.
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
        raise RunError(
            f"{argument} names step {ref.step!r}, which this pipeline does not have; "
            f"{name_choices(ref.step, tools_by_step)}"
        )
    if ref.port not in tool.output_ports:
        ports = ", ".join(tool.output_ports) if tool.output_ports else "none"
        raise RunError(
            f"{argument} names {ref!s}, but step {ref.step!r} (tool {tool.name!r}) has no output "
            f"port {ref.port!r}; its output ports are: {ports}"
        )
    return ref, tool


def resolve_sources(
    spec: Pipeline,
    tools_by_step: Mapping[str, Tool],
    sources: Mapping[PortRef | str, pl.DataFrame],
) -> dict[PortRef, pl.DataFrame]:
    """Return ``sources`` keyed by :class:`PortRef`, after checking each entry.

    A frame may only be handed to a source step, one that reads nothing. Refuses a computed step,
    a value that is not a polars ``DataFrame``, and a step given some of its output ports but not
    all of them.

    Why: docs/design/loading-and-running.md#sources-and-cached-are-different-doors
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
        if not isinstance(frame, pl.DataFrame):
            hint = "; call .collect() on it first" if isinstance(frame, pl.LazyFrame) else ""
            raise RunError(f"sources supplies {ref!s} with a {type(frame).__name__}, not a polars DataFrame{hint}")
        resolved[ref] = frame

    # Checked before the first step runs, so no earlier step has written anything yet.
    for step_id in dict.fromkeys(ref.step for ref in resolved):
        tool = tools_by_step[step_id]
        missing = [str(PortRef(step_id, port)) for port in tool.output_ports if PortRef(step_id, port) not in resolved]
        if missing:
            raise RunError(
                f"step {step_id!r} was supplied some but not all of its output ports; still missing: "
                f"{', '.join(missing)}. A source step is either read or handed every one of its ports"
            )
    return resolved


def resolve_keep(
    spec: Pipeline,
    tools_by_step: Mapping[str, Tool],
    keep: Collection[PortRef | str],
) -> frozenset[PortRef]:
    """Return the ports ``keep`` names. A step named alone keeps all its output ports.

    Refuses a step or port this flow does not have.
    """
    if isinstance(keep, (str, PortRef)):
        keep = (keep,)
    kept: set[PortRef] = set()
    for key in keep:
        if isinstance(key, str) and "." not in key:
            if key not in tools_by_step:
                raise RunError(f"keep names step {key!r}, which this pipeline does not have; {name_choices(key, tools_by_step)}")
            kept.update(PortRef(key, port) for port in tools_by_step[key].output_ports)
            continue
        ref, _ = _named_port(spec, tools_by_step, key, "keep")
        kept.add(ref)
    return frozenset(kept)


def resolve_cached(
    spec: Pipeline,
    tools_by_step: Mapping[str, Tool],
    cached: Mapping[PortRef | str, CachedFrame],
) -> dict[PortRef, CachedFrame]:
    """Return ``cached`` keyed by :class:`PortRef`, after checking each entry.

    Refuses, in this order: an unknown step or port; a source step (use ``sources`` for those); a
    step cached on some but not all of its output ports; and a frame whose saved fingerprint no
    longer matches the flow in the file.

    Why: docs/design/loading-and-running.md#what-cached-refuses
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
        if entry.fingerprint != spec.fingerprint(ref):
            if len(tools_by_step[ref.step].output_ports) > 1 and entry.fingerprint == spec.fingerprint(ref.step):
                raise RunError(
                    f"cached port {ref!s} carries the fingerprint of step {ref.step!r}, which has "
                    f"{len(tools_by_step[ref.step].output_ports)} output ports. Take each port's own "
                    f"fingerprint, spec.fingerprint(PortRef({ref.step!r}, {ref.port!r})), so a frame put "
                    f"on the wrong port is refused"
                )
            raise RunError(
                f"cached port {ref!s} cannot be used: the configuration that produced it has "
                f"changed. The fingerprint recorded beside the frame is not the one step "
                f"{ref.step!r} and its ancestors produce now, so those rows came from a pipeline "
                f"that is no longer in the file. Take the cache again, or drop it"
            )
    return resolved


def walk_around_cache(
    spec: Pipeline,
    order: tuple[str, ...],
    requested: tuple[str, ...],
    cached_steps: Collection[str],
) -> tuple[str, ...]:
    """Return ``order`` without the steps that only fed something now cached.

    Works backwards from what the run is for (the ``until`` steps, or else the steps nothing reads),
    stopping at each cached step. A step still needed by some other running step stays. A cache
    that saves nothing is allowed.

    Why: docs/design/loading-and-running.md#a-cache-skips-the-steps-behind-it
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
# When each frame can be freed, worked out once per run from the wiring
# ---------------------------------------------------------------------------


def release_schedule(
    spec: Pipeline,
    tools_by_step: Mapping[str, Tool],
    order: tuple[str, ...],
    keep: Collection[PortRef] = frozenset(),
) -> dict[str, tuple[PortRef, ...]]:
    """Return ``{step id: ports to free once that step has finished}``, for ``retain="live"``.

    A port is freed after the last step in ``order`` that reads it. A port that no step in this
    run reads is freed straight after the step that made it. A port in ``keep`` is never freed.

    Why: docs/design/loading-and-running.md#when-a-frame-is-freed
    """
    position = {step_id: index for index, step_id in enumerate(order)}
    schedule: dict[str, list[PortRef]] = {step_id: [] for step_id in order}
    for step_id in order:
        for port_name in tools_by_step[step_id].output_ports:
            ref = PortRef(step_id, port_name)
            if ref in keep:
                continue
            read_at = [position[consumer.step] for consumer in spec.consumers(ref) if consumer.step in position]
            due_at = max(read_at, default=position[step_id])
            schedule[order[due_at]].append(ref)
    return {step_id: tuple(refs) for step_id, refs in schedule.items()}
