"""Working out a run before any step of it runs.

``run_pipeline`` asks this module which steps to walk (all of them, or the ancestor closure of what
``until`` names, less what a cache makes unnecessary), which frames a caller handed in through
``sources`` and ``cached``, and when each frame can be freed under ``retain="live"``. Every refusal
here comes before the first step runs.
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
    """``(the steps asked for, the steps to walk)``, both in dependency order.

    A whole run asks for nothing and walks everything. A truncated one walks the ancestor closure of
    what it named, followed through :attr:`~.spec.Step.inputs`, which is the only edge list
    a flow has. Neither tuple is ordered here: both are :meth:`~.spec.Pipeline.order`
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
        if not isinstance(frame, pl.DataFrame):
            hint = "; call .collect() on it first" if isinstance(frame, pl.LazyFrame) else ""
            raise RunError(f"sources supplies {ref!s} with a {type(frame).__name__}, not a polars DataFrame{hint}")
        resolved[ref] = frame

    # Checked here, before the first step runs, like its mirror in resolve_cached: found mid-walk,
    # the steps before it would already have written their files.
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
    """The ports ``keep`` names, refusing a step or port this flow does not have.

    A step named alone keeps every one of its output ports.
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
# The release schedule -- a property of the wiring, computed once per run
# ---------------------------------------------------------------------------


def release_schedule(
    spec: Pipeline,
    tools_by_step: Mapping[str, Tool],
    order: tuple[str, ...],
    keep: Collection[PortRef] = frozenset(),
) -> dict[str, tuple[PortRef, ...]]:
    """``step id -> ports to free once that step has finished``, for ``retain="live"``.

    A port is due at the last step that still reads it -- read off :meth:`Pipeline.consumers`,
    a property of the wiring alone. ``order`` is what decides *last*, and under a truncated run
    that is the closure rather than the whole flow: a consumer that will never run cannot be a
    port's last use, and counting one would leave a source frame -- the largest in the run -- due
    at a step the walk never reaches, which is to say never freed at all. A port nothing remaining
    reads is due at the very step that produced it, for the same reason: a terminal frame kept
    "just in case" is the largest one in the run, and this is what keeps it from being kept.

    Both halves of that are the one question -- is this step in the walk? -- asked of a producer by
    walking ``order`` rather than the spec, and of a consumer by looking its step up in ``position``.

    A port in ``keep`` is never due: the caller asked to read it once the run is over.
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
