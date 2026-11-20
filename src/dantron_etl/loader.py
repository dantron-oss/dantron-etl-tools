"""Read a pipeline file into a ``Pipeline``, refusing anything wrong with it before a run starts.

``read_pipeline(path)`` reads a YAML file. ``load_pipeline(mapping)`` takes a document already in
memory. Both refuse, by name: an unknown or missing key, a duplicate id, an unknown tool, an
undefined rule, a bad lane or container, an edge to a port that does not exist, a cycle, settings
the tool refuses, and a step whose output nothing reads. Loading reads no data; checks that need
the frames happen when the step runs.
"""

import difflib
from collections.abc import Collection, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .config import apply_defaults, validate_against_template
from .contract import TERMINAL_CATEGORIES, call_tool_code, input_ports_for
from .errors import ConfigError, ExecutionError, PipelineError, RegistryError, ToolError, name_choices
from .expr import ExpressionError
from .registry import REGISTRY, ToolRegistry
from .renames import MIGRATE_COMMAND, find_renames, rename_document, summarise
from .rules import resolve_rules
from .spec import Container, Lane, Pipeline, PortRef, Step, toposort

__all__ = ["dead_ends", "load_pipeline", "read_pipeline"]


def load_pipeline(
    document: Mapping[str, Any],
    *,
    rules: Mapping[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> Pipeline:
    """Build a :class:`Pipeline` from a pipeline document already in memory.

    - ``document``: the parsed file, a mapping with ``name``, ``steps`` and optionally ``lanes`` and
      ``containers``.
    - ``rules``: the values ``${rules.*}`` references are filled from.
    - ``registry``: the tools to look names up in. Defaults to the shipped ``REGISTRY``.

    Raises a :class:`ToolError` (usually :class:`PipelineError` or :class:`ConfigError`) naming the
    first fault found. The checks run in a fixed order: the file's keys, duplicate ids, unknown
    tools, rule references, lanes, containers, edges, cycles, each step's settings, dead ends. If
    the document still uses names from before a rename, the refusal says which, and gives the
    command that rewrites the file.

    Why: docs/design/loading-and-running.md#the-order-of-the-loaders-checks
    """
    return _load_explained(document, rules, registry, source=None)


def _load_explained(
    document: Mapping[str, Any],
    rules: Mapping[str, Any] | None,
    registry: ToolRegistry | None,
    source: str | None,
) -> Pipeline:
    """Load the document. If it is refused and uses retired names, explain that instead.

    The explanation gives the migrate command (naming ``source`` when known) and one line per kind
    of rename. The renamed document is then tried too; if that is also refused, its refusal is added.

    Why: docs/design/loading-and-running.md#a-refusal-that-mentions-old-names
    """
    try:
        return _load_pipeline(document, rules, registry)
    except (ToolError, ExpressionError) as error:
        found = find_renames(document, registry)
        if not any(item.retired for item in found):
            raise
        lines = [
            "This file uses names from before the library renamed them, so it cannot be loaded as it is.",
            f"To rewrite it in the new names, run: {MIGRATE_COMMAND} {source or '<file>'}",
            *(f"  - {line}" for line in summarise(found)),
        ]
        try:
            _load_pipeline(rename_document(document, found), rules, registry)
        except (ToolError, ExpressionError) as remaining:
            lines += ["Once renamed, it is still refused for another reason:", f"  {remaining}"]
        raise type(error)("\n".join(lines)) from error


def _load_pipeline(
    document: Mapping[str, Any],
    rules: Mapping[str, Any] | None,
    registry: ToolRegistry | None,
) -> Pipeline:
    rules = rules if rules is not None else {}
    active_registry = registry if registry is not None else REGISTRY

    _refuse_bad_shape(document)

    raw_steps: list[Mapping[str, Any]] = list(document["steps"])
    raw_lanes: list[Mapping[str, Any]] = list(document.get("lanes") or ())

    _refuse_duplicate_ids(raw_steps)

    # Tools first: a tool's template says how a rule value is written into each field.
    tools_by_step = {raw["id"]: _resolve_tool(active_registry, raw) for raw in raw_steps}

    resolved_steps = [
        {
            **raw,
            # config_template() must take no settings: it is read before the settings are final.
            "config": resolve_rules(
                _template(raw["id"], tools_by_step[raw["id"]]),
                raw.get("config", {}) or {},
                rules,
                raw["id"],
            ),
        }
        for raw in raw_steps
    ]

    lanes = tuple(Lane(raw["id"], raw["title"], raw["order"]) for raw in raw_lanes)
    _refuse_undeclared_lanes(resolved_steps, lanes)

    containers = _read_containers(document.get("containers", ()) or ())
    _refuse_bad_containers(resolved_steps, containers)

    step_ids = tuple(raw["id"] for raw in resolved_steps)
    inputs_by_step = {raw["id"]: _resolve_edges(raw, tools_by_step) for raw in resolved_steps}

    toposort(step_ids, inputs_by_step)  # refuses a cycle, naming it

    for raw in resolved_steps:
        _validate_config(raw["id"], tools_by_step[raw["id"]], raw["config"])

    _refuse_dead_ends(resolved_steps, tools_by_step, inputs_by_step)

    steps = tuple(
        Step(
            id=raw["id"],
            tool=raw["tool"],
            config=MappingProxyType(dict(raw["config"])),
            inputs=MappingProxyType(inputs_by_step[raw["id"]]),
            lane=raw.get("lane"),
            stage=raw.get("stage"),
            why=raw.get("why"),
            container=raw.get("container"),
        )
        for raw in resolved_steps
    )

    return Pipeline(name=document["name"], steps=steps, lanes=lanes, containers=containers, registry=active_registry)


def read_pipeline(
    path: str | Path,
    *,
    rules: Mapping[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> Pipeline:
    """Read a pipeline from a YAML file and build a :class:`Pipeline`.

    Call it as ``read_pipeline("pipelines/sales.yaml", rules=...)``. It takes the same keyword
    arguments as :func:`load_pipeline`, returns the same thing, and refuses the same faults. A
    refusal about retired names names this file in the migrate command it suggests.
    """
    import yaml

    document = yaml.load(Path(path).read_text(encoding="utf8"), Loader=yaml_loader())
    return _load_explained(document, rules, registry, source=str(path))


def yaml_loader() -> Any:
    """Return PyYAML's fast C safe loader if it is installed, otherwise its Python safe loader.

    Why: docs/design/loading-and-running.md#yaml-is-read-with-the-c-loader-when-there-is-one
    """
    import yaml

    return getattr(yaml, "CSafeLoader", yaml.SafeLoader)


# ---------------------------------------------------------------------------
# The shape of the file
# ---------------------------------------------------------------------------

# The keys each part of a pipeline file may hold. Any other key is refused, so a misspelling is caught.
_DOCUMENT_KEYS: tuple[str, ...] = ("name", "steps", "lanes", "containers")
_STEP_KEYS: tuple[str, ...] = ("id", "tool", "config", "inputs", "lane", "stage", "container", "why")
_LANE_KEYS: tuple[str, ...] = ("id", "title", "order")
_CONTAINER_KEYS: tuple[str, ...] = ("id", "title")

# The keys that must be there. A container's title is checked by _read_containers.
_DOCUMENT_REQUIRED: tuple[str, ...] = ("name", "steps")
_STEP_REQUIRED: tuple[str, ...] = ("id", "tool")
_LANE_REQUIRED: tuple[str, ...] = ("id", "title", "order")
_CONTAINER_REQUIRED: tuple[str, ...] = ("id",)


def _refuse_bad_shape(document: Any) -> None:
    """Refuse a file whose keys or value types are wrong, before anything else is read.

    Checks the top level, every step, lane and container for unknown keys, missing keys and values
    of the wrong type. Also refuses a flow with no steps, a bad step id, and two lanes or two
    containers sharing an id.
    """
    if not isinstance(document, Mapping):
        raise PipelineError(
            f"a pipeline file must hold a mapping with the keys {', '.join(_DOCUMENT_KEYS)}; "
            f"this one holds a {type(document).__name__}"
        )
    _refuse_unknown_keys(document, _DOCUMENT_KEYS, "the pipeline file", "the top level of a pipeline file")
    _refuse_missing_keys(document, _DOCUMENT_REQUIRED, "the pipeline file")
    _refuse_wrong_types(document, {"name": (str,)}, "the pipeline file")

    steps = _entries(document, "steps")
    if not steps:
        raise PipelineError("the pipeline file has no steps; a flow needs at least one")

    for kind, keys, required in (
        ("steps", _STEP_KEYS, _STEP_REQUIRED),
        ("lanes", _LANE_KEYS, _LANE_REQUIRED),
        ("containers", _CONTAINER_KEYS, _CONTAINER_REQUIRED),
    ):
        singular = kind[:-1]
        seen: set[Any] = set()
        for position, entry in enumerate(_entries(document, kind), start=1):
            if not isinstance(entry, Mapping):
                raise PipelineError(
                    f"{singular} number {position} under '{kind}' must be a mapping of keys; "
                    f"it is a {type(entry).__name__}"
                )
            where = f"{singular} {entry['id']!r}" if "id" in entry else f"{singular} number {position}"
            _refuse_unknown_keys(entry, keys, where, f"a {singular}")
            _refuse_missing_keys(entry, required, where)
            _refuse_wrong_types(entry, _VALUE_TYPES[kind], where)
            if kind == "steps":
                _refuse_bad_step_id(entry["id"], where)
            # Duplicate step ids are refused later, by _refuse_duplicate_ids.
            if kind != "steps" and entry["id"] in seen:
                raise PipelineError(
                    f"duplicate {singular} id {entry['id']!r}: only the first would be used, so the "
                    f"second would never be drawn"
                )
            seen.add(entry["id"])


# What each key's value must be. Checked here so a wrong type is refused by name, not as a raw TypeError.
_TEXT_OR_NONE: tuple[type, ...] = (str, type(None))
_VALUE_TYPES: dict[str, dict[str, tuple[type, ...]]] = {
    "steps": {
        "id": (str,),
        "tool": (str,),
        "config": (Mapping, type(None)),
        "inputs": (Mapping, type(None)),
        "lane": _TEXT_OR_NONE,
        "stage": _TEXT_OR_NONE,
        "why": _TEXT_OR_NONE,
        "container": _TEXT_OR_NONE,
    },
    "lanes": {"id": (str,), "title": (str,), "order": (int,)},
    "containers": {"id": (str,)},
}

_TYPE_WORDS: dict[type, str] = {str: "text", int: "a whole number", Mapping: "a mapping of keys", type(None): "empty"}


def _refuse_wrong_types(entry: Mapping[str, Any], expected: Mapping[str, tuple[type, ...]], where: str) -> None:
    """Refuse a key whose value is of the wrong type, naming the key and what it should hold."""
    for key, types in expected.items():
        if key not in entry:
            continue
        value = entry[key]
        if isinstance(value, bool) or not isinstance(value, types):
            wanted = " or ".join(_TYPE_WORDS[kind] for kind in types)
            raise PipelineError(f"{where}: {key!r} must be {wanted}; it holds {type(value).__name__} {value!r}")
    inputs = entry.get("inputs") if "tool" in expected else None
    if isinstance(inputs, Mapping):
        for port, source in inputs.items():
            if not isinstance(port, str) or not isinstance(source, str):
                raise PipelineError(
                    f"{where}: every entry under 'inputs' is a port name and the text 'step.port'; "
                    f"found {port!r}: {source!r}"
                )


def _refuse_bad_step_id(step_id: str, where: str) -> None:
    """Refuse a blank step id, or one with a dot in it, which no ``step.port`` edge could reach."""
    if not step_id.strip():
        raise PipelineError(f"{where}: a step id may not be blank")
    if "." in step_id:
        raise PipelineError(
            f"{where}: a step id may not hold a dot, because an edge is written 'step.port' and is split "
            f"at the first dot, so no edge could reach this step. Use an underscore instead"
        )


def _entries(document: Mapping[str, Any], kind: str) -> list[Any]:
    """Return the list under ``kind``, or an empty list if it is absent. Refuse anything but a list."""
    value = document.get(kind)
    if value is None:
        return []
    if not isinstance(value, list):
        raise PipelineError(f"'{kind}' must be a list; this file has a {type(value).__name__} there")
    return value


def _refuse_unknown_keys(entry: Mapping[str, Any], allowed: tuple[str, ...], where: str, kind: str) -> None:
    unknown = [key for key in entry if key not in allowed]
    if not unknown:
        return
    key = unknown[0]
    near = difflib.get_close_matches(str(key), allowed, n=1)
    guess = f"; did you mean {near[0]!r}?" if near else "."
    raise PipelineError(f"{where} has an unknown key {key!r}{guess} The keys {kind} may hold are: {', '.join(allowed)}")


def _refuse_missing_keys(entry: Mapping[str, Any], required: tuple[str, ...], where: str) -> None:
    missing = [key for key in required if key not in entry]
    if missing:
        raise PipelineError(f"{where} is missing the key {missing[0]!r}; it needs: {', '.join(required)}")


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
# Containers
# ---------------------------------------------------------------------------


def _read_containers(raw_containers: Any) -> tuple[Container, ...]:
    """Return the declared containers. Refuse one with no title, or a blank one.

    Why: docs/design/loading-and-running.md#a-container-must-have-a-title
    """
    containers: list[Container] = []
    for raw in raw_containers:
        title = raw.get("title")
        if not isinstance(title, str) or not title.strip():
            raise PipelineError(
                f"container {raw['id']!r} declares no title; a container's whole value is the one "
                f"sentence a reader gets instead of the steps it stands over, and an id is not a "
                f"sentence"
            )
        containers.append(Container(raw["id"], title))
    return tuple(containers)


def _refuse_bad_containers(resolved_steps: list[Mapping[str, Any]], containers: tuple[Container, ...]) -> None:
    """Refuse a step in an undeclared container, a container with no steps, and a container
    whose steps sit in more than one lane.

    The steps of a container do not have to be connected to each other.

    Why: docs/design/loading-and-running.md#what-the-loader-checks-about-containers
    """
    declared = {container.id for container in containers}
    joined: dict[str, list[Mapping[str, Any]]] = {}
    for raw in resolved_steps:
        container_id = raw.get("container")
        if container_id is None:
            continue
        if container_id not in declared:
            available = ", ".join(sorted(declared)) if declared else "none"
            raise PipelineError(
                f"step {raw['id']!r} declares container {container_id!r}, which is not declared; containers are: {available}"
            )
        joined.setdefault(container_id, []).append(raw)

    for container in containers:
        if container.id not in joined:
            raise PipelineError(
                f"container {container.id!r} is declared and no step belongs to it, so it would be drawn as "
                f"an empty box with a title over nothing"
            )
        lanes = sorted({raw.get("lane") or "no lane" for raw in joined[container.id]})
        if len(lanes) > 1:
            raise PipelineError(
                f"container {container.id!r} holds steps in lanes {' and '.join(repr(lane) for lane in lanes)}; "
                f"a lane is a horizontal band and one box cannot honestly be drawn across two. Move "
                f"the step, or split the container in two"
            )


# ---------------------------------------------------------------------------
# Ports and edges
# ---------------------------------------------------------------------------


def _resolve_edges(raw: Mapping[str, Any], tools_by_step: Mapping[str, Any]) -> dict[str, PortRef]:
    step_id = raw["id"]
    tool = tools_by_step[step_id]
    raw_inputs: Mapping[str, str] = raw.get("inputs", {}) or {}
    # The ports this step reads. Some tools work them out from the settings as written (no defaults).
    try:
        declared_inputs = input_ports_for(tool, raw.get("config", {}) or {})
    except ToolError as error:
        raise PipelineError(f"step {step_id!r}: {error}") from error

    # A port named twice is refused first; the checks below work on sets and would miss it.
    # Why: docs/design/loading-and-running.md#a-port-named-twice-is-refused-first
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

    edges: dict[str, PortRef] = {}
    for port_name, source in raw_inputs.items():
        if "." not in source:
            raise PipelineError(
                f"step {step_id!r} input {port_name!r} names {source!r}, which is not a step.port "
                f"reference: an edge names a step and one of its output ports, never a step alone"
            )
        source_step, source_port = source.split(".", 1)
        if source_step not in tools_by_step:
            raise PipelineError(
                f"step {step_id!r} input {port_name!r} refers to step {source_step!r}, which does "
                f"not exist; {name_choices(source_step, tools_by_step)}"
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
# Configuration
# ---------------------------------------------------------------------------


def _template(step_id: str, tool: Any) -> Mapping[str, Any]:
    """Return the tool's configuration template. A failure in the tool's code names the step."""
    try:
        return call_tool_code(tool, "config_template", tool.config_template)
    except ToolError as error:
        raise PipelineError(f"step {step_id!r}: {error}") from error


def _validate_config(step_id: str, tool: Any, config: Mapping[str, Any]) -> None:
    """Check a step's settings against the tool's template, then with the tool's ``validate_config``.

    Defaults are filled in before ``validate_config`` sees the settings. Every refusal names the
    step. An unexpected exception from the tool becomes an :class:`~.errors.ExecutionError`.
    """
    template = _template(step_id, tool)
    try:
        validate_against_template(template, config)
        filled = apply_defaults(template, config)
        call_tool_code(tool, "validate_config", lambda: tool.validate_config(filled))
    except ExecutionError as error:
        raise ExecutionError(f"step {step_id!r}: {error}") from error
    except ToolError as error:
        raise ConfigError(f"step {step_id!r}: {error}") from error


# ---------------------------------------------------------------------------
# Dead ends
# ---------------------------------------------------------------------------


def dead_ends(
    document: Mapping[str, Any],
    *,
    registry: ToolRegistry | None = None,
    without: Collection[str] = (),
) -> tuple[str, ...]:
    """Return every step of ``document`` that is a dead end, including the chains behind them.

    A dead end is a step whose output nothing reads and whose tool is not an ``output`` or an
    ``assert``. Removing one can leave the steps that fed it unread, so those are listed too: first
    the steps nothing reads, then the ones that feed only those, and so on.

    - ``document``: the parsed pipeline file. Rules are not resolved, so a file the loader refuses
      for its dead ends can still be asked.
    - ``without``: input edges to remove first, to ask "what if". Each is ``"step.port"``, naming
      the reading step and its input port, as in ``without=["report.summary"]``. An edge the
      document does not have is refused.

    Runs no step and reads no data.
    """
    active_registry = registry if registry is not None else REGISTRY
    _refuse_bad_shape(document)
    raw_steps: list[Mapping[str, Any]] = list(document["steps"])
    tools_by_step = {raw["id"]: _resolve_tool(active_registry, raw) for raw in raw_steps}
    reads: dict[str, dict[str, str]] = {raw["id"]: dict(raw.get("inputs") or {}) for raw in raw_steps}

    for edge in without:
        step_id, _, port = str(edge).partition(".")
        if step_id not in reads or port not in reads[step_id]:
            wired = [f"{step}.{name}" for step, inputs in reads.items() for name in inputs]
            raise PipelineError(f"without names input {edge!r}, which this document does not wire; {name_choices(str(edge), wired, 'inputs')}")
        del reads[step_id][port]

    return tuple(step_id for round_ in _dead_rounds(reads, tools_by_step) for step_id in round_)


def _dead_rounds(
    reads: Mapping[str, Mapping[str, Any]],
    tools_by_step: Mapping[str, Any],
) -> list[list[str]]:
    """Return the dead steps round by round: those nothing reads, then those that fed only them."""
    alive = list(reads)
    rounds: list[list[str]] = []
    while True:
        read = {_source_step(source) for step_id in alive for source in reads[step_id].values()}
        dead = [
            step_id
            for step_id in alive
            if step_id not in read and tools_by_step[step_id].category not in TERMINAL_CATEGORIES
        ]
        if not dead:
            return rounds
        rounds.append(dead)
        gone = set(dead)
        alive = [step_id for step_id in alive if step_id not in gone]


def _source_step(source: Any) -> str:
    """Return the step an edge reads from, given ``"step.port"`` text or a :class:`PortRef`."""
    return source.step if isinstance(source, PortRef) else str(source).split(".", 1)[0]


def _refuse_dead_ends(
    resolved_steps: list[Mapping[str, Any]],
    tools_by_step: Mapping[str, Any],
    inputs_by_step: Mapping[str, Mapping[str, PortRef]],
) -> None:
    """Refuse a flow with dead ends, naming each one and the steps that feed only them."""
    rounds = _dead_rounds(inputs_by_step, tools_by_step)
    if not rounds:
        return
    first, behind = rounds[0], [step_id for round_ in rounds[1:] for step_id in round_]
    named = ", ".join(f"{step_id!r} (tool {tools_by_step[step_id].name!r})" for step_id in first)
    if len(first) == 1:
        message = (
            f"step {named} is a dead end: nothing consumes any of its output and its tool has no "
            f"effect beyond its ports"
        )
    else:
        message = (
            f"{len(first)} steps are dead ends: nothing consumes any of their output and their tools have "
            f"no effect beyond their ports: {named}"
        )
    if behind:
        message += (
            f". Removing {'it' if len(first) == 1 else 'them'} would leave {len(behind)} more with no reader, "
            f"which feed only {'it' if len(first) == 1 else 'them'}: {', '.join(repr(step_id) for step_id in behind)}"
        )
    raise PipelineError(message)
