"""A flow as a data file: which tools, wired how, walkable in one order.

Nineteen tools exist, each with a typed configuration template, declared ports and a boundary that
validates whatever it can without a frame. What none of them says is *which* tools are wired *how*
for a given flow -- and that is the whole point of this module: a document a person can read and
change, where changing it changes what a runner actually does. A flow written as functions can only
be edited by generating functions; a flow written as data is edited by editing it.

So this module owns exactly two things. First, the shape: :class:`StepSpec` as authored, an edge as
a :class:`PortRef` naming a step and one of its declared ports, and :class:`PipelineSpec` as the
walkable whole. Second, the loader: every refusal reachable before a single frame exists -- a
duplicate id, a parameter nobody defined, a tool nobody registered, an edge that names a port that
was never declared, a cycle, a configuration a tool's own rules refuse, and a step whose output
nothing reads and which has no effect of its own. Nothing here moves a row; running a step still
belongs to :func:`~.tool.run_tool`, once a runner holds real frames to check the column references
this module publishes but cannot itself resolve.
"""

import hashlib
import heapq
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .expr import escape_text, quote_text, references
from .tool import (
    REGISTRY,
    ConfigError,
    ConfigField,
    RegistryError,
    ToolError,
    ToolRegistry,
    apply_defaults,
    expression_values,
    input_ports_for,
    validate_against_template,
)

__all__ = [
    "Group",
    "Lane",
    "PipelineError",
    "PipelineSpec",
    "PortRef",
    "StepSpec",
    "load_pipeline",
    "read_pipeline_file",
]

# Any ``${...}`` reference, wherever it sits in a string. Whether it is the whole field or text
# embedded inside a larger one is decided by the caller, matching it with ``fullmatch`` or ``sub``.
_PLACEHOLDER: re.Pattern[str] = re.compile(r"\$\{([^}]*)\}")

# The categories whose effect is not one of their ports, so a step nothing consumes is not a dead
# end when it carries one of these -- an 'output' writes somewhere else, an 'assert' raises or not.
# This is the tool's own published category, not a per-step flag an author could set and forget.
_TERMINAL_CATEGORIES: frozenset[str] = frozenset({"output", "assert"})


class PipelineError(ToolError):
    """A wiring fault: a duplicate id, an unresolved parameter, a bad edge, a lane, or a cycle.

    Distinct from :class:`~.tool.ConfigError`, which a delegated ``validate_config`` or
    ``validate_against_template`` call raises for a configuration a tool refuses on its own terms.
    Both descend from :class:`~.tool.ToolError`, so a driver has one family to catch, but the two
    name different next moves: this one sends a reader to the arrows, the other to the config block.
    """


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

    It reaches no tool and changes no frame: see :meth:`PipelineSpec.fingerprint` for why that has to
    stay true.
    """

    id: str
    title: str


@dataclass(frozen=True, slots=True)
class StepSpec:
    """One step, as authored: its tool, its configuration with parameters resolved and nothing else
    added, and its inputs as port-to-port edges rather than bare step names.

    ``config`` holds no default a template could have filled in: the file a person edits and the
    object a canvas draws have to stay the same artefact, and a reader seeing a field nobody wrote
    is how a UI starts lying about what is actually there. Filling defaults is :func:`~.tool.run_tool`'s
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
class PipelineSpec:
    """A whole flow: its steps in the order they were authored, its lanes, and the registry it was
    checked against -- needed by :meth:`column_references`, which has to ask a tool for its template.

    ``registry`` does not participate in equality: two specs loaded from the same document against
    the same catalog are the same pipeline, whatever registry object happened to answer for it.
    """

    name: str
    steps: tuple[StepSpec, ...]
    lanes: tuple[Lane, ...] = ()
    groups: tuple[Group, ...] = ()
    registry: ToolRegistry | None = field(default=None, compare=False, repr=False)

    def step(self, step_id: str) -> StepSpec:
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

        closure: set[str] = set()
        pending = [step_id]
        while pending:
            current = pending.pop()
            if current in closure:
                continue
            closure.add(current)
            pending.extend(ref.step for ref in self.step(current).inputs.values())

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

    def order(self) -> tuple[str, ...]:
        """Dependency order, ties broken by the order the steps were declared in."""
        step_ids = tuple(step.id for step in self.steps)
        inputs_by_step = {step.id: step.inputs for step in self.steps}
        return _toposort(step_ids, inputs_by_step)

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

        Lifted through :func:`~.tool.expression_values` and :func:`~.expr.references`, and nothing
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


def load_pipeline(
    document: Mapping[str, Any],
    *,
    rules: Mapping[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> PipelineSpec:
    """Build a :class:`PipelineSpec` from ``document``, refusing everything reachable without a frame.

    Refused in this order, because each phase needs the one before it to have already succeeded:
    a duplicate id (an edge cannot name its far end unambiguously otherwise), a tool nobody
    registered (its template is what says how a rule value is spelled into each field, and a port
    check needs a tool to ask), a parameter nobody defined (a configuration cannot be read until it
    is resolved, and resolution still finishes before anything validates it), a lane nobody
    declared, a container that is mis-declared or mis-named (both spellings of the same typo, since
    either one draws a box that is silently wrong rather than one that is missing), a port or edge
    that does not exist, a cycle (reachability is meaningless in a graph
    with one), a configuration its own tool refuses, and finally a step nothing consumes and that
    has no effect of its own.

    ``document`` is a plain mapping because a file format is not part of what a pipeline *is* --
    see :func:`read_pipeline_file` for the file-reading convenience built on top of this.
    """
    rules = rules if rules is not None else {}
    active_registry = registry if registry is not None else REGISTRY

    raw_steps: list[Mapping[str, Any]] = list(document.get("steps", ()))
    raw_lanes: list[Mapping[str, Any]] = list(document.get("lanes", ()))

    _refuse_duplicate_ids(raw_steps)

    # The tool is looked up before the rules are resolved, because its template is what says which
    # of its fields are expressions and which are patterns, paths and names. The lookup reads only
    # the step's 'tool' and 'id', neither of which sits inside a configuration a rule could reach.
    tools_by_step = {raw["id"]: _resolve_tool(active_registry, raw) for raw in raw_steps}

    resolved_steps = [
        {
            **raw,
            # config_template() takes no configuration, and must keep taking none: it is read here,
            # before the configuration is final, so a template that consulted its own config would
            # be a cycle rather than an ordering to get right.
            "config": _resolve_rules(
                tools_by_step[raw["id"]].config_template(),
                raw.get("config", {}) or {},
                rules,
                raw["id"],
            ),
        }
        for raw in raw_steps
    ]

    lanes = tuple(Lane(raw["id"], raw["label"], raw["order"]) for raw in raw_lanes)
    _refuse_undeclared_lanes(resolved_steps, lanes)

    groups = _read_groups(document.get("groups", ()) or ())
    _refuse_bad_groups(resolved_steps, groups)

    step_ids = tuple(raw["id"] for raw in resolved_steps)
    inputs_by_step = {raw["id"]: _resolve_edges(raw, tools_by_step, step_ids) for raw in resolved_steps}

    _toposort(step_ids, inputs_by_step)  # raises PipelineError by naming the cycle, if there is one

    for raw in resolved_steps:
        _validate_config(raw["id"], tools_by_step[raw["id"]], raw["config"])

    _refuse_dead_ends(resolved_steps, tools_by_step, inputs_by_step)

    steps = tuple(
        StepSpec(
            id=raw["id"],
            tool=raw["tool"],
            config=MappingProxyType(dict(raw["config"])),
            inputs=MappingProxyType(inputs_by_step[raw["id"]]),
            lane=raw.get("lane"),
            stage=raw.get("stage"),
            why=raw.get("why"),
            group=raw.get("group"),
        )
        for raw in resolved_steps
    )

    return PipelineSpec(name=document["name"], steps=steps, lanes=lanes, groups=groups, registry=active_registry)


def read_pipeline_file(
    path: str | Path,
    *,
    rules: Mapping[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> PipelineSpec:
    """:func:`load_pipeline`, for a document parsed from a YAML file rather than assembled in memory.

    The core loader takes a mapping because this library ships into repositories that did not
    choose its dependencies, and a file format is not one of the things a pipeline *is*. PyYAML is
    imported here, inside the call, exactly the way ``db.py`` reaches for a postgres driver only
    when a postgres connection is actually asked for -- and refused by naming what to install, not
    with the bare ``ImportError`` a caller has never heard of.
    """
    try:
        import yaml
    except ImportError as error:
        raise PipelineError(
            "reading a pipeline file needs PyYAML, which this library does not depend on; install "
            "'pyyaml', or call load_pipeline directly with an already-parsed mapping"
        ) from error

    document = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    return load_pipeline(document, rules=rules, registry=registry)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _canonical_scalar(value: Any) -> str:
    """A configuration value the standard library's JSON encoder has no spelling for, given one.

    Reached for a date or a timestamp, which is what YAML hands back for an unquoted ``2026-04-01``.
    That cannot arrive through :func:`load_pipeline` today -- measured: every field the catalog
    declares is typed, none of them is a date, and a bare date in a pipeline or rules file is
    refused as the wrong type long before anything is fingerprinted. The branch is here for a
    :class:`PipelineSpec` assembled in code, which is a published way to build one, and for the day
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
# Duplicate ids
# ---------------------------------------------------------------------------


def _refuse_duplicate_ids(raw_steps: list[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for raw in raw_steps:
        step_id = raw["id"]
        if step_id in seen:
            raise PipelineError(
                f"duplicate step id {step_id!r}: every edge names its far end by id, so two steps cannot share one"
            )
        seen.add(step_id)


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------


def _resolve_rules(
    template: Mapping[str, ConfigField],
    config: Mapping[str, Any],
    rules: Mapping[str, Any],
    step_id: str,
) -> dict[str, Any]:
    """``config`` with every ``${rules.*}`` resolved, at any depth, before validation ever sees it.

    A reference that is the whole field takes the rule's own value and type, so a list-valued rule
    stays a list. That is what it does in every field, whatever the field is for.

    A reference sitting inside surrounding text is written into that text, and how it is written is
    the field's question rather than this loader's. ``template`` is what answers it. A field
    declared ``type="expression"`` gets the value in the expression language's own spelling --
    quoted, escaped, a list parenthesised as ``('DRAFT', 'SUBMITTED')`` because square brackets in
    an expression mean a column -- since the person editing a pipeline file is stating a value, not
    writing code, and has no reason to know that ``O'Brian`` is spelled differently from
    ``Parkside``. Every other field gets the value and nothing else: a pattern, a path, a sheet
    name, a column name and a table name are none of them expressions, and a quote written into one
    of those is not an escape. It is a character to match, or a character in a file name.

    A field the template does not describe is written verbatim too, and is never refused here. A
    field nobody declared is a misspelled name, and ``validate_against_template`` is what says so;
    a refusal about rules would send the author to the wrong half of the file.

    The ``{scratch}`` token carries no dollar sign and is never touched here. It belongs to the
    tool boundary, not to this loader.
    """
    return {key: _resolve_value(value, template.get(key), rules, step_id) for key, value in config.items()}


def _resolve_value(value: Any, spec: ConfigField | None, rules: Mapping[str, Any], step_id: str) -> Any:
    """One value and the template entry describing it, walked down together.

    Together rather than separately, because the fields this distinction matters to are nested: a
    ladder's conditions, an aggregation's value and a derived column's name all sit two levels down,
    and that last one sits one element away from an expression wanting the opposite spelling.

    ``spec`` is ``None`` wherever the template stops -- an undeclared field, an array with no
    declared element, an object with no declared members -- and everything under it is verbatim.
    """
    if isinstance(value, str):
        return _resolve_text(value, spec, rules, step_id)
    if isinstance(value, list):
        element = spec.items if spec is not None and spec.type == "array" else None
        return [_resolve_value(item, element, rules, step_id) for item in value]
    if isinstance(value, dict):
        declared = spec.fields if spec is not None and spec.type == "object" else None
        members: Mapping[str, ConfigField] = declared if declared is not None else {}
        return {key: _resolve_value(item, members.get(key), rules, step_id) for key, item in value.items()}
    return value


def _resolve_text(text: str, spec: ConfigField | None, rules: Mapping[str, Any], step_id: str) -> Any:
    """One string: the rule's own value where the reference is the whole of it, or written into it.

    Each reference is spelled for where it sits, one at a time, because a field may hold two of them
    wanting opposite spellings -- one inside the author's quotes and one outside. The parity that
    answers that is read from the string as the author wrote it, never from the text produced so far.
    """
    if "${" not in text:
        return text
    whole = _PLACEHOLDER.fullmatch(text)
    if whole is not None:
        return _rule_value(whole.group(1), rules, step_id)
    is_expression = spec is not None and spec.type == "expression"

    def written(match: re.Match[str]) -> str:
        value = _rule_value(match.group(1), rules, step_id)
        if not is_expression:
            return value if isinstance(value, str) else str(value)
        return _as_expression(
            value,
            inside_literal=text[: match.start()].count("'") % 2 == 1,
            rule=match.group(1).removeprefix("rules."),
            step_id=step_id,
        )

    return _PLACEHOLDER.sub(written, text)


def _as_expression(value: Any, *, inside_literal: bool, rule: str, step_id: str) -> str:
    """One rule value, written into an expression field in the expression language's own spelling.

    Each element of a list is spelled by its own type: text is quoted and escaped, a number is not,
    because a number in quotes is text and text against a number column is refused.

    ``inside_literal`` says whether the reference sits inside quotes the author wrote themselves.
    ``[code] = '${rules.kept_code}'`` and ``[month] >= date'${rules.start_month}'`` are both written
    that way in real pipeline files, and both have to keep working. Inside, the value needs the
    escaping only; outside, it needs the quotes as well. The parity of ``'`` before the reference
    answers that, and stays right under doubling because every literal is balanced.

    A truth value and a null have a spelling in this language and get it -- but only outside a
    literal. Inside one, the author asked for text and the rules file holds none, so that is refused
    rather than written as ``'False'``, which compiles, matches nothing and reports success.
    """
    if value is None or isinstance(value, bool):
        if inside_literal:
            raise PipelineError(
                f"step {step_id!r} writes rule {rule!r} inside quotes, where this language reads "
                f"text, but the rule's value is {value!r} -- a truth value or a null, which this "
                f"language spells without quotes. YAML reads bare words like yes/no/on/off/null "
                f"that way too: quote the value in the rules file to state the text you meant."
            )
        if value is None:
            return "null"
        return "true" if value else "false"
    if inside_literal:
        return escape_text(str(value))
    if isinstance(value, (list, tuple)):
        elements = (_as_expression(item, inside_literal=False, rule=rule, step_id=step_id) for item in value)
        return "(" + ", ".join(elements) + ")"
    if isinstance(value, str):
        return quote_text(value)
    return str(value)


def _rule_value(reference: str, rules: Mapping[str, Any], step_id: str) -> Any:
    """The value ``${reference}`` names, or a refusal by name -- 'rules' is the only namespace."""
    if not reference.startswith("rules."):
        raise PipelineError(
            f"step {step_id!r} references '${{{reference}}}', but 'rules' is the only namespace a "
            f"pipeline configuration may reference"
        )
    rule_name = reference[len("rules.") :]
    if rule_name not in rules:
        available = ", ".join(sorted(rules)) if rules else "none"
        raise PipelineError(
            f"step {step_id!r} references undefined rule {rule_name!r}; rules on offer are: {available}"
        )
    return rules[rule_name]


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------


def _resolve_tool(registry: ToolRegistry, raw: Mapping[str, Any]) -> Any:
    try:
        return registry.get(raw["tool"])
    except RegistryError as error:
        raise PipelineError(f"step {raw['id']!r}: {error}") from error


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


def _refuse_undeclared_lanes(resolved_steps: list[Mapping[str, Any]], lanes: tuple[Lane, ...]) -> None:
    lane_ids = {lane.id for lane in lanes}
    for raw in resolved_steps:
        lane = raw.get("lane")
        if lane is not None and lane not in lane_ids:
            available = ", ".join(sorted(lane_ids)) if lane_ids else "none"
            raise PipelineError(
                f"step {raw['id']!r} declares lane {lane!r}, which is not declared; lanes are: {available}"
            )


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def _read_groups(raw_groups: Any) -> tuple[Group, ...]:
    """The declared containers, refusing any that offers no sentence to write on itself.

    Blank counts as missing. A title of spaces satisfies every check that asks whether the key is
    there and draws a box with nothing written on it, which is the same silent failure as falling
    back on the id: the container looks finished and nothing says a sentence was owed.
    """
    groups: list[Group] = []
    for raw in raw_groups:
        title = raw.get("title")
        if not isinstance(title, str) or not title.strip():
            raise PipelineError(
                f"group {raw['id']!r} declares no title; a container's whole value is the one "
                f"sentence a reader gets instead of the steps it stands over, and an id is not a "
                f"sentence"
            )
        groups.append(Group(raw["id"], title))
    return tuple(groups)


def _refuse_bad_groups(resolved_steps: list[Mapping[str, Any]], groups: tuple[Group, ...]) -> None:
    """Both spellings of a mis-boxed step, and a container a lane band cannot be drawn round.

    The first two are mirrors of each other and the pair is the point: a group named by a step and
    not declared leaves a container one step short with a stray node beside it, and a group declared
    and named by no step draws an empty rectangle with a title over nothing. Neither says anything is
    wrong, and on six hundred steps nobody finds either by looking.

    The third is about the drawing. A lane is a horizontal band, and a box across two of them is a
    lie whichever way it is drawn: stretched over both it swallows every step of the intervening lane
    that is not in it, and drawn round one it leaves a member outside the box that claims to hold it.
    Refused rather than silently redrawn, because the fix is one line -- move the step, or split the
    container -- and only the author knows which of the two they meant.

    Connectedness is deliberately not checked. The commonest container there is -- *read the three
    workbooks* -- is three reads that touch nothing of each other and meet in a join outside the box,
    and a connectedness rule refuses exactly it.
    """
    declared = {group.id for group in groups}
    joined: dict[str, list[Mapping[str, Any]]] = {}
    for raw in resolved_steps:
        group_id = raw.get("group")
        if group_id is None:
            continue
        if group_id not in declared:
            available = ", ".join(sorted(declared)) if declared else "none"
            raise PipelineError(
                f"step {raw['id']!r} declares group {group_id!r}, which is not declared; groups are: {available}"
            )
        joined.setdefault(group_id, []).append(raw)

    for group in groups:
        if group.id not in joined:
            raise PipelineError(
                f"group {group.id!r} is declared and no step belongs to it, so it would be drawn as "
                f"an empty box with a title over nothing"
            )
        lanes = sorted({raw.get("lane") or "no lane" for raw in joined[group.id]})
        if len(lanes) > 1:
            raise PipelineError(
                f"group {group.id!r} holds steps in lanes {' and '.join(repr(lane) for lane in lanes)}; "
                f"a lane is a horizontal band and one box cannot honestly be drawn across two. Move "
                f"the step, or split the container in two"
            )


# ---------------------------------------------------------------------------
# Ports and edges
# ---------------------------------------------------------------------------


def _resolve_edges(
    raw: Mapping[str, Any],
    tools_by_step: Mapping[str, Any],
    step_ids: tuple[str, ...],
) -> dict[str, PortRef]:
    step_id = raw["id"]
    tool = tools_by_step[step_id]
    raw_inputs: Mapping[str, str] = raw.get("inputs", {}) or {}
    # The ports this step declares, which for some tools is a question about this step's own
    # configuration. The configuration is the one written in the file: edges are resolved before
    # the configuration is validated, so no default has been filled in yet. The runner reads the
    # same configuration, so both get the same answer.
    declared_inputs = input_ports_for(tool, raw.get("config", {}) or {})

    # Refuse a port this step names twice, before anything else is compared. A port carries one
    # frame, so the second part on a port would quietly take the place of the first. The checks
    # below both work over sets, where a repeated name disappears, so neither of them would say a
    # word. This one runs first for a second reason: an unwired duplicate also trips the "nothing
    # feeds this port" refusal underneath, which would send the author to the arrows when the thing
    # to fix is the configuration.
    duplicated = sorted({port for port in declared_inputs if declared_inputs.count(port) > 1})
    if duplicated:
        raise PipelineError(
            f"step {step_id!r} (tool {tool.name!r}) names duplicate input port {duplicated[0]!r} in "
            f"its configuration. A port carries one frame, so the second use of the name would "
            f"replace the first. Give each one its own port name. This step configures input "
            f"ports: {', '.join(declared_inputs)}"
        )

    unknown = sorted(set(raw_inputs) - set(declared_inputs))
    if unknown:
        available = ", ".join(declared_inputs) if declared_inputs else "none"
        raise PipelineError(
            f"step {step_id!r} declares input port {unknown[0]!r}, which tool {tool.name!r} does "
            f"not have; its input ports are: {available}"
        )

    missing = [port for port in declared_inputs if port not in raw_inputs]
    if missing:
        raise PipelineError(
            f"step {step_id!r} (tool {tool.name!r}) does not feed input port {missing[0]!r}; "
            f"{tool.name} declares input ports: {', '.join(declared_inputs)}"
        )

    step_ids_set = set(step_ids)
    edges: dict[str, PortRef] = {}
    for port_name, source in raw_inputs.items():
        if "." not in source:
            raise PipelineError(
                f"step {step_id!r} input {port_name!r} names {source!r}, which is not a step.port "
                f"reference: an edge names a step and one of its output ports, never a step alone"
            )
        source_step, source_port = source.split(".", 1)
        if source_step not in step_ids_set:
            raise PipelineError(
                f"step {step_id!r} input {port_name!r} refers to step {source_step!r}, which does "
                f"not exist; steps are: {', '.join(step_ids)}"
            )
        source_tool = tools_by_step[source_step]
        if source_port not in source_tool.output_ports:
            raise PipelineError(
                f"step {step_id!r} input {port_name!r} refers to {source!r}, but step {source_step!r} "
                f"(tool {source_tool.name!r}) has no output port {source_port!r}; its output ports "
                f"are: {', '.join(source_tool.output_ports)}"
            )
        edges[port_name] = PortRef(source_step, source_port)
    return edges


# ---------------------------------------------------------------------------
# Cycle detection and the walk order it also answers
# ---------------------------------------------------------------------------


def _toposort(
    step_ids: tuple[str, ...],
    inputs_by_step: Mapping[str, Mapping[str, PortRef]],
) -> tuple[str, ...]:
    """Dependency order over ``step_ids``, ties broken by declaration order; a cycle is refused."""
    position = {step_id: index for index, step_id in enumerate(step_ids)}
    dependents: dict[str, list[str]] = {step_id: [] for step_id in step_ids}
    remaining_deps: dict[str, set[str]] = {step_id: set() for step_id in step_ids}
    for step_id in step_ids:
        for edge in inputs_by_step.get(step_id, {}).values():
            if edge.step in position:
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _validate_config(step_id: str, tool: Any, config: Mapping[str, Any]) -> None:
    """Both halves a tool can refuse without a frame, delegated rather than re-implemented here."""
    template = tool.config_template()
    try:
        validate_against_template(template, config)
        tool.validate_config(apply_defaults(template, config))
    except ConfigError as error:
        raise ConfigError(f"step {step_id!r}: {error}") from error


# ---------------------------------------------------------------------------
# Dead ends
# ---------------------------------------------------------------------------


def _refuse_dead_ends(
    resolved_steps: list[Mapping[str, Any]],
    tools_by_step: Mapping[str, Any],
    inputs_by_step: Mapping[str, Mapping[str, PortRef]],
) -> None:
    consumed = {edge.step for edges in inputs_by_step.values() for edge in edges.values()}
    for raw in resolved_steps:
        step_id = raw["id"]
        tool = tools_by_step[step_id]
        if step_id in consumed or tool.category in _TERMINAL_CATEGORIES:
            continue
        raise PipelineError(
            f"step {step_id!r} (tool {tool.name!r}) is a dead end: nothing consumes any of its "
            f"output and its tool has no effect beyond its ports"
        )
