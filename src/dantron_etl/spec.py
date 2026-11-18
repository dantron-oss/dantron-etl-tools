"""What a loaded flow is: its steps, the edges between them, and the order they run in.

``Pipeline`` is the whole flow and ``Step`` is one step as the file wrote it. An edge is a
``PortRef``: a step id and one of that step's output ports. ``Lane`` and ``Group`` are how the flow
is drawn and change nothing it computes. The loader in ``loader.py`` builds these; nothing here
moves a row.

A flow as a data file says *which* tools are wired *how* for a given flow -- a document a person can
read and change, where changing it changes what a runner actually does. A flow written as functions
can only be edited by generating functions; a flow written as data is edited by editing it.
"""

import hashlib
import heapq
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .errors import PipelineError
from .expr import references
from .config import expression_values
from .registry import ToolRegistry

__all__ = ["Group", "Lane", "Pipeline", "PortRef", "Step"]


@dataclass(frozen=True, slots=True)
class PortRef:
    """One end of an edge: a step id and one of that step's declared ports, never a step alone."""

    step: str
    port: str

    def __str__(self) -> str:
        return f"{self.step}.{self.port}"


@dataclass(frozen=True, slots=True)
class Lane:
    """A swimlane a step may sit in, carried in the same document as the wiring it lays out."""

    id: str
    label: str
    order: int


@dataclass(frozen=True, slots=True)
class Group:
    """A line drawn round some of a lane's steps, with one sentence written on it.

    Six hundred boxes cannot be read by following them, and a reader after "where does the extract
    get its header row back" is looking for a job rather than for a tool. So a group says
    which steps do one job, and the ``title`` says what that job is -- a sentence, because the whole
    value of the feature is a reader getting one instead of the six boxes it stands over. An id is
    not a title, and the loader refuses a group that offers only the former.

    It reaches no tool and changes no frame: see :meth:`Pipeline.fingerprint` for why that has to
    stay true.
    """

    id: str
    title: str


@dataclass(frozen=True, slots=True)
class Step:
    """One step, as authored: its tool, its configuration with parameters resolved and nothing else
    added, and its inputs as port-to-port edges rather than bare step names.

    ``config`` holds no default a template could have filled in: the file a person edits and the
    object a canvas draws have to stay the same artefact, and a reader seeing a field nobody wrote
    is how a UI starts lying about what is actually there. Filling defaults is :func:`~.boundary.run_tool`'s
    job, done at the one moment they are needed.
    """

    id: str
    tool: str
    config: Mapping[str, Any]
    inputs: Mapping[str, PortRef]
    lane: str | None = None
    stage: str | None = None
    why: str | None = None
    #: The container this step is drawn inside, or ``None`` for a step in none -- which is the state
    #: every step starts in, and the reason containers can be adopted a stage at a time.
    group: str | None = None


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A whole flow: its steps in the order they were authored, its lanes, and the registry it was
    checked against -- needed by :meth:`column_references` and :meth:`group_outputs`, which have to
    ask a tool for its template and its ports.

    ``registry`` does not participate in equality: two specs loaded from the same document against
    the same catalog are the same pipeline, whatever registry object happened to answer for it.
    """

    name: str
    steps: tuple[Step, ...]
    lanes: tuple[Lane, ...] = ()
    groups: tuple[Group, ...] = ()
    registry: ToolRegistry | None = field(default=None, compare=False, repr=False)

    def step(self, step_id: str) -> Step:
        """The step named ``step_id``, or a refusal naming the ids there actually are."""
        for candidate in self.steps:
            if candidate.id == step_id:
                return candidate
        raise PipelineError(f"no step {step_id!r}; steps are: {', '.join(step.id for step in self.steps)}")

    def group(self, group_id: str) -> Group:
        """The container named ``group_id``, or a refusal naming the ones there actually are."""
        for candidate in self.groups:
            if candidate.id == group_id:
                return candidate
        declared = ", ".join(group.id for group in self.groups) or "none"
        raise PipelineError(f"no group {group_id!r}; groups are: {declared}")

    def group_members(self, group_id: str) -> tuple[str, ...]:
        """The steps in ``group_id``, in the order they were declared.

        Declaration order, because that is the order the file reads in and the only order a
        container has. A set would draw the same container over a different arrangement on every
        render, and a reader comparing two renders would be comparing two drawings.
        """
        self.group(group_id)
        return tuple(step.id for step in self.steps if step.group == group_id)

    def group_inputs(self, group_id: str) -> tuple[PortRef, ...]:
        """Every port produced outside ``group_id`` and read inside it, named once.

        Once however many members read it: two steps taking the same upstream port are one lot of
        rows arriving, and counting it twice reports a container that invented rows. A port produced
        by a member is the container's own plumbing and is no part of what entered it.
        """
        members = set(self.group_members(group_id))
        entered: list[PortRef] = []
        for step in self.steps:
            if step.id not in members:
                continue
            for source in step.inputs.values():
                if source.step not in members and source not in entered:
                    entered.append(source)
        return tuple(entered)

    def group_outputs(self, group_id: str) -> tuple[PortRef, ...]:
        """Every port of a member read by a non-member, or by nobody at all.

        The second half is the one that is easy to leave out and wrong to leave out: rows that stop
        at a member's port have still left the container -- they went to a file, to a report, or
        nowhere -- and a container whose last step is a write would otherwise claim nothing came out
        of it. What is excluded is the ports used up inside, which is the difference between a figure
        a reader can check a closed container against and a bigger number that answers to
        nothing.
        """
        assert self.registry is not None
        members = set(self.group_members(group_id))
        left: list[PortRef] = []
        for step in self.steps:
            if step.id not in members:
                continue
            for port in self.registry.get(step.tool).output_ports:
                read_by = self.consumers(PortRef(step.id, port))
                if not read_by or any(ref.step not in members for ref in read_by):
                    left.append(PortRef(step.id, port))
        return tuple(left)

    def fingerprint(self, step_id: str) -> str:
        """One string answering "is the configuration that produced this step's ports still the
        configuration in the file?" -- taken over ``step_id`` and every step behind it.

        The closure, not the step alone, because the rows on a port are the product of everything
        upstream of them: a step whose own config block is untouched produces different rows the
        moment a filter feeding it keeps different ones, and a hash that could not see that would
        report a match and hand back yesterday's numbers under today's rules. Four things per step
        are covered -- ``id``, ``tool``, ``config`` and ``inputs`` -- and ``lane``, ``stage``,
        ``why`` and ``group`` deliberately are not. None of the four reaches a tool, so a frame
        produced under one ``why`` is the identical frame under another, and a cache invalidated by
        an improved comment is a cache nobody trusts, which is to say one everybody works around.
        ``group`` joins that list for the same reason and one of its own: boxing three steps together
        is a line drawn on the canvas, and if it threw away every cached frame behind them the first
        person to tidy the canvas would pay for it with an hour of re-running -- and the second
        person would not tidy the canvas.

        Hashed over a canonical JSON serialisation rather than through :func:`hash`, which is
        salted per process for strings: the whole use of this string is that it is written down
        beside a frame in one run and compared in another, and those are two processes. Sorting the
        keys is what makes a configuration whose keys were retyped in a different order the same
        configuration, which a mapping has no order to disagree about.

        The limit is the thing to say plainly: this covers configuration, not data. It cannot cover
        data -- the expensive reads are exactly the ones an author wants to skip, so hashing what
        they returned would mean performing them, which is the work being avoided. A frame kept from
        yesterday's database pull matches this fingerprint against today's database, and the number
        it produces is wrong in a way nothing here can see. See :func:`~.runner.run_pipeline` for
        what follows from that.
        """
        self.step(step_id)  # a name this pipeline does not have is refused here, in one wording

        closure = self.ancestor_closure((step_id,))

        # Sorted by id rather than taken in declaration or dependency order: where a step sits in
        # the file changes nothing about what it computes, and a hash that moved when somebody
        # reordered two independent steps would go stale on a tidy-up.
        payload = [
            {
                "id": step.id,
                "tool": step.tool,
                "config": dict(step.config),
                "inputs": {port: str(ref) for port, ref in step.inputs.items()},
            }
            for step in sorted((self.step(name) for name in closure), key=lambda step: step.id)
        ]
        serialised = json.dumps(payload, default=_canonical_scalar, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf8")).hexdigest()

    def ancestor_closure(self, step_ids: Iterable[str]) -> frozenset[str]:
        """``step_ids`` and every step that feeds them, however far back, followed through ``inputs``.

        This is what a fingerprint covers and what a run stopped by ``until`` walks. A step id this
        pipeline does not have is refused by :meth:`step`.
        """
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
        """Dependency order, ties broken by the order the steps were declared in."""
        step_ids = tuple(step.id for step in self.steps)
        inputs_by_step = {step.id: step.inputs for step in self.steps}
        return toposort(step_ids, inputs_by_step)

    def consumers(self, source: "PortRef") -> tuple["PortRef", ...]:
        """Every input that is wired to ``source``, in the order the consuming steps were declared."""
        found: list[PortRef] = []
        for step in self.steps:
            for port_name, edge in step.inputs.items():
                if edge == source:
                    found.append(PortRef(step.id, port_name))
        return tuple(found)

    def stages(self) -> tuple[str, ...]:
        """Distinct stages, in the order each is first seen among the steps as declared."""
        seen: list[str] = []
        for step in self.steps:
            if step.stage is not None and step.stage not in seen:
                seen.append(step.stage)
        return tuple(seen)

    def column_references(self) -> Mapping[str, Mapping[str, frozenset[str]]]:
        """``step id -> config field path -> column names``, for whoever holds the frames to check.

        Lifted through :func:`~.config.expression_values` and :func:`~.expr.references`, and nothing
        more: a step whose template has no expression field, or whose expression fields carry no
        value, is simply absent -- there is nothing here to publish about it.
        """
        assert self.registry is not None
        result: dict[str, dict[str, frozenset[str]]] = {}
        for step in self.steps:
            tool = self.registry.get(step.tool)
            values = expression_values(tool.config_template(), step.config)
            if values:
                result[step.id] = {path: references(text) for path, text in values}
        return result


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _canonical_scalar(value: Any) -> str:
    """A configuration value the standard library's JSON encoder has no spelling for, given one.

    Reached for a date or a timestamp, which is what YAML hands back for an unquoted ``2026-04-01``.
    That cannot arrive through :func:`load_pipeline` today -- measured: every field the catalog
    declares is typed, none of them is a date, and a bare date in a pipeline or rules file is
    refused as the wrong type long before anything is fingerprinted. The branch is here for a
    :class:`Pipeline` assembled in code, which is a published way to build one, and for the day
    a tool declares a field that is a date. Handling it costs two lines; the alternative is a
    fingerprint that raises on a spec nothing else objects to.

    Anything else is refused rather than handed to ``str``: ``str`` of an arbitrary object is a repr
    carrying a memory address, which would differ between the run that wrote a fingerprint down and
    the run that compares it -- so every cache would report itself stale, and the failure would look
    like the feature not working rather than like a value nobody can canonically write down.
    """
    if isinstance(value, date):  # a datetime is one of these too; isoformat tells the two apart
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
    """Dependency order over ``step_ids``, ties broken by declaration order; a cycle is refused."""
    position = {step_id: index for index, step_id in enumerate(step_ids)}
    dependents: dict[str, list[str]] = {step_id: [] for step_id in step_ids}
    remaining_deps: dict[str, set[str]] = {step_id: set() for step_id in step_ids}
    for step_id in step_ids:
        for edge in inputs_by_step.get(step_id, {}).values():
            # A step can read two ports of one upstream step. That is one dependency, not two, so
            # it is recorded once. Recording it twice would put this step in the walk twice.
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
        cycle_steps = sorted((s for s in step_ids if s not in ordered), key=lambda s: position[s])
        raise PipelineError(
            f"a cycle among steps: {', '.join(cycle_steps)}; a flow is walked from its sources, so a "
            f"loop can never reach a consumer"
        )
    return tuple(ordered)
