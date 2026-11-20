"""What a loaded flow is: its steps, the edges between them, and the order they run in.

``Pipeline`` is the whole flow and ``Step`` is one step as the file wrote it. An edge is a
``PortRef``: a step id and one of that step's output ports. ``Lane`` and ``Container`` only change
how the flow is drawn. The loader builds these objects; nothing here reads or changes data.
"""

import hashlib
import heapq
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any

from .errors import PipelineError, name_choices
from .expr import references
from .config import expression_values
from .registry import REGISTRY, ToolRegistry

__all__ = ["Container", "Lane", "Pipeline", "PortRef", "Step"]


@dataclass(frozen=True, slots=True)
class PortRef:
    """One end of an edge: a step id and one of that step's ports. ``str()`` gives ``"step.port"``."""

    step: str
    port: str

    def __str__(self) -> str:
        return f"{self.step}.{self.port}"


@dataclass(frozen=True, slots=True)
class Lane:
    """A horizontal band on the drawn page that steps may sit in, shown in ``order``."""

    id: str
    title: str
    order: int


@dataclass(frozen=True, slots=True)
class Container:
    """A box drawn round steps that do one job, with a sentence (``title``) saying what the job is.

    It only changes the drawing, never a frame or a fingerprint. The loader refuses a container
    with no title.

    Why: docs/design/loading-and-running.md#containers
    """

    id: str
    title: str


@dataclass(frozen=True, slots=True)
class Step:
    """One step as the file wrote it.

    ``config`` has its ``${rules.*}`` references filled in and no defaults added; defaults are
    filled in when the step runs. ``inputs`` maps each input port to the :class:`PortRef` it reads.
    ``lane``, ``stage``, ``why`` and ``container`` only affect how the step is shown.

    Why: docs/design/loading-and-running.md#a-step-holds-no-defaults
    """

    id: str
    tool: str
    config: Mapping[str, Any]
    inputs: Mapping[str, PortRef]
    lane: str | None = None
    stage: str | None = None
    why: str | None = None
    #: The container this step is drawn inside, or ``None``.
    container: str | None = None


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A whole loaded flow: its steps in file order, its lanes, containers, and tool registry.

    Get one from :func:`read_pipeline` or :func:`load_pipeline` and pass it to
    :func:`run_pipeline`. Its methods answer questions about the wiring: :meth:`step`,
    :meth:`order`, :meth:`consumers`, :meth:`fingerprint`, the container methods and more. A
    pipeline built in code without a registry uses the shipped one. Two pipelines are equal when
    their steps, lanes and containers are; the registry is not compared.
    """

    name: str
    steps: tuple[Step, ...]
    lanes: tuple[Lane, ...] = ()
    containers: tuple[Container, ...] = ()
    registry: ToolRegistry | None = field(default=None, compare=False, repr=False)
    # Lookups built once: step by id, and the readers of each port.
    _by_id: Mapping[str, Step] = field(init=False, compare=False, repr=False)
    _readers: Mapping[PortRef, tuple[PortRef, ...]] = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.registry is None:
            # Built in code without a registry: use the shipped one.
            object.__setattr__(self, "registry", REGISTRY)
        by_id: dict[str, Step] = {}
        for step in self.steps:
            by_id.setdefault(step.id, step)
        readers: dict[PortRef, list[PortRef]] = {}
        for step in self.steps:
            for port_name, edge in step.inputs.items():
                readers.setdefault(edge, []).append(PortRef(step.id, port_name))
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_readers", MappingProxyType({ref: tuple(found) for ref, found in readers.items()}))

    def step(self, step_id: str) -> Step:
        """Return the step named ``step_id``. Refuse an unknown id, suggesting the closest ones."""
        found = self._by_id.get(step_id)
        if found is not None:
            return found
        raise PipelineError(f"no step {step_id!r}; {name_choices(step_id, self._by_id)}")

    def container(self, container_id: str) -> Container:
        """Return the container named ``container_id``. Refuse an unknown id, listing the real ones."""
        for candidate in self.containers:
            if candidate.id == container_id:
                return candidate
        declared = [container.id for container in self.containers]
        raise PipelineError(f"no container {container_id!r}; {name_choices(container_id, declared, 'containers')}")

    def container_members(self, container_id: str) -> tuple[str, ...]:
        """Return the ids of the steps in ``container_id``, in file order, so every drawing is the same."""
        self.container(container_id)
        return tuple(step.id for step in self.steps if step.container == container_id)

    def container_inputs(self, container_id: str) -> tuple[PortRef, ...]:
        """Return every port made outside ``container_id`` and read inside it, each listed once.

        A port read by two steps in the container is still one set of rows coming in, so it is
        listed once. Ports made by steps inside the container are not inputs to it.
        """
        members = set(self.container_members(container_id))
        entered: list[PortRef] = []
        for step in self.steps:
            if step.id not in members:
                continue
            for source in step.inputs.values():
                if source.step not in members and source not in entered:
                    entered.append(source)
        return tuple(entered)

    def container_outputs(self, container_id: str) -> tuple[PortRef, ...]:
        """Return every port of a step in ``container_id`` that a step outside reads, or nothing reads.

        A port nothing reads still counts as leaving the container, so a container that ends in a
        write still shows what came out of it. Ports used only inside are left out.
        """
        registry = self.registry if self.registry is not None else REGISTRY
        members = set(self.container_members(container_id))
        left: list[PortRef] = []
        for step in self.steps:
            if step.id not in members:
                continue
            for port in registry.get(step.tool).output_ports:
                read_by = self.consumers(PortRef(step.id, port))
                if not read_by or any(ref.step not in members for ref in read_by):
                    left.append(PortRef(step.id, port))
        return tuple(left)

    def fingerprint(self, target: "str | PortRef") -> str:
        """Return a hash of the settings that produce ``target``: the step and every step before it.

        ``target`` is a step id, a :class:`PortRef`, or ``"step.port"``. If the fingerprint saved
        with a cached frame still matches, the flow that made the frame has not changed. For a step
        with several output ports the port name is included too. Each step's ``id``, ``tool``,
        ``config`` and ``inputs`` are covered; ``lane``, ``stage``, ``why`` and ``container`` are not.
        Refuses an unknown step or port.

        It covers the settings, never the data: a frame cached before the source data changed
        still matches.

        Why: docs/design/loading-and-running.md#fingerprints
        """
        step_id, port = self._fingerprint_target(target)
        closure = self.ancestor_closure((step_id,))

        # Sorted by id, so reordering steps in the file does not change the fingerprint.
        payload = [
            {
                "id": step.id,
                "tool": step.tool,
                "config": dict(step.config),
                "inputs": {port: str(ref) for port, ref in step.inputs.items()},
            }
            for step in sorted((self.step(name) for name in closure), key=lambda step: step.id)
        ]
        hashed: Any = payload if port is None else {"steps": payload, "port": port}
        serialised = json.dumps(hashed, default=_canonical_scalar, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf8")).hexdigest()

    def _fingerprint_target(self, target: "str | PortRef") -> tuple[str, str | None]:
        """Return ``(step id, port name or None)`` for :meth:`fingerprint`. A one-port step gives None."""
        if isinstance(target, str) and "." not in target:
            self.step(target)  # refuses an unknown step
            return target, None
        if isinstance(target, str):
            step_id, port = target.split(".", 1)
        else:
            step_id, port = target.step, target.port
        step = self.step(step_id)
        registry = self.registry if self.registry is not None else REGISTRY
        ports = tuple(registry.get(step.tool).output_ports)
        if port not in ports:
            raise PipelineError(
                f"step {step_id!r} (tool {step.tool!r}) has no output port {port!r}; its output ports are: "
                f"{', '.join(ports) or 'none'}"
            )
        return step_id, (port if len(ports) > 1 else None)

    def ancestor_closure(self, step_ids: Iterable[str]) -> frozenset[str]:
        """Return ``step_ids`` and every step that feeds them, however far back. Refuses an unknown id."""
        closure: set[str] = set()
        pending = list(step_ids)
        while pending:
            current = pending.pop()
            if current in closure:
                continue
            closure.add(current)
            pending.extend(ref.step for ref in self.step(current).inputs.values())
        return frozenset(closure)

    def order(self) -> tuple[str, ...]:
        """Return the step ids in the order they run: each after what it reads, otherwise file order."""
        step_ids = tuple(step.id for step in self.steps)
        inputs_by_step = {step.id: step.inputs for step in self.steps}
        return toposort(step_ids, inputs_by_step)

    def consumers(self, source: "PortRef") -> tuple["PortRef", ...]:
        """Return every input port wired to ``source``, in file order of the reading steps."""
        return self._readers.get(source, ())

    def stages(self) -> tuple[str, ...]:
        """Return each stage name once, in the order it first appears in the file."""
        seen: list[str] = []
        for step in self.steps:
            if step.stage is not None and step.stage not in seen:
                seen.append(step.stage)
        return tuple(seen)

    def column_references(self) -> Mapping[str, Mapping[str, frozenset[str]]]:
        """Return the columns each step's expressions name: ``{step id: {field path: column names}}``.

        A step with no expressions is left out.
        """
        registry = self.registry if self.registry is not None else REGISTRY
        result: dict[str, dict[str, frozenset[str]]] = {}
        for step in self.steps:
            tool = registry.get(step.tool)
            values = expression_values(tool.config_template(), step.config)
            if values:
                result[step.id] = {path: references(text) for path, text in values}
        return result


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _canonical_scalar(value: Any) -> str:
    """Write a date or timestamp in a configuration as ISO text for the fingerprint. Refuse anything else.

    Why: docs/design/loading-and-running.md#values-json-cannot-write
    """
    if isinstance(value, date):  # covers datetime too
        return value.isoformat()
    raise PipelineError(
        f"a configuration holds {type(value).__name__} {value!r}, which has no canonical text form, "
        f"so what produced a port cannot be fingerprinted; write the value as text in the pipeline "
        f"or rules file"
    )


# ---------------------------------------------------------------------------
# Cycle detection and the walk order it also answers
# ---------------------------------------------------------------------------


def toposort(
    step_ids: tuple[str, ...],
    inputs_by_step: Mapping[str, Mapping[str, PortRef]],
) -> tuple[str, ...]:
    """Return ``step_ids`` in the order they can run, ties in file order. Refuse a cycle, naming it."""
    position = {step_id: index for index, step_id in enumerate(step_ids)}
    dependents: dict[str, list[str]] = {step_id: [] for step_id in step_ids}
    remaining_deps: dict[str, set[str]] = {step_id: set() for step_id in step_ids}
    for step_id in step_ids:
        for edge in inputs_by_step.get(step_id, {}).values():
            # Two ports read from one step are one dependency; counted twice, the step would run twice.
            if edge.step in position and edge.step not in remaining_deps[step_id]:
                remaining_deps[step_id].add(edge.step)
                dependents[edge.step].append(step_id)

    heap = [position[step_id] for step_id in step_ids if not remaining_deps[step_id]]
    heapq.heapify(heap)
    by_position = {position[step_id]: step_id for step_id in step_ids}

    ordered: list[str] = []
    while heap:
        step_id = by_position[heapq.heappop(heap)]
        ordered.append(step_id)
        for downstream in dependents[step_id]:
            remaining_deps[downstream].discard(step_id)
            if not remaining_deps[downstream]:
                heapq.heappush(heap, position[downstream])

    if len(ordered) != len(step_ids):
        placed = set(ordered)
        unplaced = [s for s in step_ids if s not in placed]
        loop = _one_loop(unplaced, inputs_by_step)
        path = " -> ".join([*(f"{ref.step}.{ref.port} -> {reader}" for reader, ref in loop)])
        stuck = len(unplaced) - len(loop)
        tail = (
            f" {stuck} more step(s) could not be placed either: each is in a loop too, or reads from one."
            if stuck
            else ""
        )
        raise PipelineError(
            f"a cycle among steps: {', '.join(reader for reader, _ in loop)}. The loop is {path}. A flow "
            f"is walked from its sources, so a loop can never reach a consumer.{tail}"
        )
    return tuple(ordered)


def _one_loop(unplaced: list[str], inputs_by_step: Mapping[str, Mapping[str, PortRef]]) -> list[tuple[str, PortRef]]:
    """Return one loop among the steps the sort could not place, as ``(step, edge it reads)`` pairs.

    Each unplaced step reads another unplaced step, so walking upstream must revisit a step. The
    steps from there on are the loop.
    """
    stuck = set(unplaced)
    seen: dict[str, int] = {}
    walk: list[tuple[str, PortRef]] = []
    current = unplaced[0]
    while current not in seen:
        seen[current] = len(walk)
        edge = next(ref for ref in inputs_by_step.get(current, {}).values() if ref.step in stuck)
        walk.append((current, edge))
        current = edge.step
    loop = walk[seen[current] :]
    # In the direction the rows flow.
    return list(reversed(loop))
