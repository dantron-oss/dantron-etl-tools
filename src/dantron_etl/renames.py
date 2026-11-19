"""The names a pipeline file used to be written in, and what each one is called now.

This is the one rename table. The loader reads it to say what an old name became when it refuses a
file. The migration tool (``migrate.py``) reads it to rewrite an old file in the new names.

A name is renamed by where it sits: which tool, which field, which part of the file. The same word
somewhere else is not a retired name. So the table is keyed by tool and by path, and
:func:`find_renames` walks a parsed document the same way.

A path names a field inside a step's ``config``. ``tables[].key`` is the ``key`` member of each
entry of the ``tables`` list. The parts of a path before its last one are written in the new names.
"""

import ast
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MIGRATE_COMMAND",
    "Found",
    "find_renames",
    "function_calls",
    "port_map",
    "retired_names",
]

MIGRATE_COMMAND: str = f"python -m {__package__}.migrate"

# Old tool name -> new tool name.
TOOL_RENAMES: dict[str, str] = {
    "connected_components": "make_groups",
    "write_workbook": "write_excel",
}

# New tool name -> old field path -> new field name.
FIELD_RENAMES: dict[str, dict[str, str]] = {
    "join": {"left_on": "left_keys", "right_on": "right_keys"},
    "unpivot": {"index": "by", "into": "names_to", "values": "values_to"},
    "cross_tab": {"index": "by", "columns_from": "names_from", "values": "values_from", "aggregate": "function"},
    "date_diff": {"endpoints": "counting"},
    "expand_date_range": {"interval": "unit"},
    "case_when": {"rules": "cases"},
    "remove_duplicates": {"columns": "by"},
    "publish_tables": {"stamp": "stamp_column", "tables[].key": "keys"},
    "write_csv": {"replace": "overwrite"},
    "write_excel": {"replace": "overwrite_sheets"},
    "match_one_to_one": {"left": "left_key", "right": "right_key"},
    "normalize_text": {"strip_suffixes": "suffixes"},
    "make_groups": {"node_column": "record_column", "component_column": "group_column"},
}

# (new tool name, new field path) -> old value -> new value.
VALUE_RENAMES: dict[tuple[str, str], dict[str, str]] = {
    ("date_diff", "counting"): {"both": "inclusive"},
    ("date_diff", "unit"): {"months": "month", "days": "day"},
    ("assemble_date", "on_invalid"): {"empty": "set_null"},
    ("cast_columns", "on_invalid"): {"empty": "set_null"},
    ("unpivot", "on_invalid"): {"empty": "set_null"},
    ("group_by", "aggregations[].function"): {"count_values": "count_non_null"},
    ("cross_tab", "function"): {"count_values": "count_non_null"},
}

# Old defaults that differ from the new ones. Keyed by the tool name as the file writes it, then by
# the new field name. A step with neither the field nor its old name gets the old value written in.
OLD_DEFAULTS: dict[str, dict[str, str]] = {
    "date_diff": {"counting": "inclusive"},
    "connected_components": {"record_column": "node", "group_column": "component"},
}

# New tool name -> old port -> new port.
OUTPUT_PORT_RENAMES: dict[str, dict[str, str]] = {"join": {"matched": "joined"}}
INPUT_PORT_RENAMES: dict[str, dict[str, str]] = {"make_groups": {"edges": "pairs"}}

# Keys of the file itself, outside any step's config.
DOCUMENT_KEY_RENAMES: dict[str, str] = {"groups": "containers"}
STEP_KEY_RENAMES: dict[str, str] = {"group": "container"}
LANE_KEY_RENAMES: dict[str, str] = {"label": "title"}

# Expression functions: old name -> new name.
FUNCTION_RENAMES: dict[str, str] = {"strip": "trim"}

# The old single-sheet workbook writer: this tool with this config key. It cannot be migrated.
SINGLE_SHEET_TOOL: tuple[str, str] = ("write_excel", "sheet")

# Python names. Code cannot be rewritten by the migration, so these are printed as guidance.
PYTHON_RENAMES: tuple[tuple[str, str], ...] = (
    ("read_pipeline_file", "read_pipeline"),
    ("PipelineSpec", "Pipeline"),
    ("StepSpec", "Step"),
    (
        f"imports from {__package__}.pipeline, {__package__}.tool or {__package__}.runner",
        f"imports from {__package__} ({__package__}.authoring for a tool author)",
    ),
    ("Group", "Container"),
    ("Step.group", "Step.container"),
    ("Pipeline.groups", "Pipeline.containers"),
    ("group_members, group_inputs, group_outputs", "container_members, container_inputs, container_outputs"),
    ("Lane.label", "Lane.title"),
    ('PortRef(step, "matched") for a join step', 'PortRef(step, "joined")'),
)

_PLACEHOLDER = re.compile(r"\$\{[^}]*\}")


def port_map() -> dict[str, Any]:
    """The renamed output ports, keyed by the old tool name, for a figure check outside the library."""
    old_names = {new: old for old, new in TOOL_RENAMES.items()}
    return {"tools": {old_names.get(tool, tool): dict(ports) for tool, ports in OUTPUT_PORT_RENAMES.items()}}


@dataclass(frozen=True)
class Found:
    """One retired name in a document, or one thing to write in or to check by hand.

    ``kind`` says what to do at ``path``:

    - ``key``: the mapping at ``path[:-1]`` has the key ``old``; it becomes ``new``.
    - ``value``: the string at ``path`` is ``old``; it becomes ``new``.
    - ``call``: the expression at ``path`` calls function ``old``; each call becomes ``new``.
    - ``insert``: the mapping at ``path`` gets the key ``new`` with the value ``old``.
    - ``manual``: nothing changes; ``sentence`` says what to check by hand.

    ``retired`` is false for an insert and for a manual note that is not about an old name.
    """

    kind: str
    path: tuple[Any, ...]
    old: str
    new: str
    label: str
    sentence: str
    step: str | None = None
    retired: bool = True


def retired_names(document: Mapping[str, Any], registry: Any = None) -> list[str]:
    """One sentence per retired name ``document`` uses. Empty when it uses only current names."""
    return [found.sentence for found in find_renames(document, registry) if found.retired]


def find_renames(document: Any, registry: Any = None) -> list[Found]:
    """Every retired name in a parsed pipeline document, walked per tool and per path."""
    if not isinstance(document, Mapping):
        return []
    found: list[Found] = []
    for old, new in DOCUMENT_KEY_RENAMES.items():
        if old in document:
            found.append(_key((old,), old, new, f"{old} -> {new}", f"the top-level key {old!r} is now {new!r}"))

    for index, lane in _entries(document, "lanes"):
        for old, new in LANE_KEY_RENAMES.items():
            if old in lane:
                where = f"lane {lane.get('id')!r}" if "id" in lane else f"lane number {index + 1}"
                found.append(
                    _key(("lanes", index, old), old, new, f"lane {old} -> {new}", f"{where}: the key {old!r} is now {new!r}")
                )

    steps = _entries(document, "steps")
    tool_by_id = {
        step["id"]: TOOL_RENAMES.get(step["tool"], step["tool"])
        for _, step in steps
        if isinstance(step.get("id"), str) and isinstance(step.get("tool"), str)
    }
    for index, step in steps:
        found.extend(_step_renames(index, step, tool_by_id, registry))
    return found


def _entries(document: Mapping[str, Any], kind: str) -> list[tuple[int, Mapping[str, Any]]]:
    value = document.get(kind)
    if not isinstance(value, list):
        return []
    return [(index, entry) for index, entry in enumerate(value) if isinstance(entry, Mapping)]


def _key(path: tuple[Any, ...], old: str, new: str, label: str, sentence: str, step: str | None = None) -> Found:
    return Found("key", path, old, new, label, sentence, step)


def _step_renames(index: int, step: Mapping[str, Any], tool_by_id: Mapping[str, str], registry: Any) -> list[Found]:
    step_id = step.get("id") if isinstance(step.get("id"), str) else None
    here = f"step {step_id!r}" if step_id is not None else f"step number {index + 1}"
    at = ("steps", index)
    written = step.get("tool")
    config = step.get("config")
    if not isinstance(written, str):
        return []

    single_tool, single_key = SINGLE_SHEET_TOOL
    if written == single_tool and isinstance(config, Mapping) and single_key in config:
        sentence = (
            f"{here}: this is the old single-sheet {written} (its config has {single_key!r}), which "
            f"was retired. Rewrite it by hand as {written} with a 'sheets' list"
        )
        return [Found("manual", (*at, "tool"), written, written, "", sentence, step_id)]

    found: list[Found] = []
    tool = TOOL_RENAMES.get(written, written)
    if tool != written:
        found.append(
            Found("value", (*at, "tool"), written, tool, f"tool {written} -> {tool}", f"{here}: tool {written!r} is now {tool!r}", step_id)
        )

    for old, new in STEP_KEY_RENAMES.items():
        if old in step:
            found.append(_key((*at, old), old, new, f"step {old} -> {new}", f"{here}: the key {old!r} is now {new!r}", step_id))

    if isinstance(config, Mapping):
        found.extend(_config_renames(tool, written, config, (*at, "config"), here, step_id, registry))

    inputs = step.get("inputs")
    if isinstance(inputs, Mapping):
        for port, source in inputs.items():
            renamed_port = INPUT_PORT_RENAMES.get(tool, {}).get(port) if isinstance(port, str) else None
            if renamed_port is not None:
                found.append(
                    _key(
                        (*at, "inputs", port),
                        port,
                        renamed_port,
                        f"input port {tool}.{port} -> {renamed_port}",
                        f"{here}: input port {port!r} is now {renamed_port!r}",
                        step_id,
                    )
                )
            if not isinstance(source, str) or "." not in source:
                continue
            source_step, source_port = source.split(".", 1)
            source_tool = tool_by_id.get(source_step)
            new_port = OUTPUT_PORT_RENAMES.get(source_tool or "", {}).get(source_port)
            if new_port is not None:
                found.append(
                    Found(
                        "value",
                        (*at, "inputs", port),
                        source,
                        f"{source_step}.{new_port}",
                        f"port {source_tool}.{source_port} -> {new_port}",
                        f"{here}: input {port!r} reads {source!r}, and the {source_tool} output port "
                        f"{source_port!r} is now {new_port!r}",
                        step_id,
                    )
                )
    return found


def _config_renames(
    tool: str,
    written: str,
    config: Mapping[str, Any],
    at: tuple[Any, ...],
    here: str,
    step_id: str | None,
    registry: Any,
) -> list[Found]:
    found: list[Found] = []

    for (owner, path), values in VALUE_RENAMES.items():
        if owner != tool:
            continue
        for value, where, shown in _reach(config, _split(path), tool, at, ""):
            if not isinstance(value, str):
                continue
            if value in values:
                new = values[value]
                found.append(
                    Found(
                        "value",
                        where,
                        value,
                        new,
                        f"{tool}.{path}: {value} -> {new}",
                        f"{here}: config field {shown!r} has the value {value!r}, which is now {new!r}",
                        step_id,
                    )
                )
            elif _PLACEHOLDER.fullmatch(value):
                pairs = ", ".join(f"{old} -> {new}" for old, new in values.items())
                found.append(
                    Found(
                        "manual",
                        where,
                        value,
                        value,
                        "",
                        f"{here}: config field {shown!r} takes its value from {value!r}, and the values of "
                        f"this field were renamed ({pairs}). Check that value by hand",
                        step_id,
                        retired=False,
                    )
                )

    for template_path in _expression_paths(tool, registry):
        for value, where, shown in _reach(config, _split(template_path), tool, at, ""):
            if not isinstance(value, str):
                continue
            for old, new in FUNCTION_RENAMES.items():
                offsets, _ = function_calls(value, old)
                if offsets:
                    found.append(
                        Found(
                            "call",
                            where,
                            old,
                            new,
                            f"{old}() -> {new}()",
                            f"{here}: config field {shown!r} calls {old!r}, which was renamed to {new!r}",
                            step_id,
                        )
                    )

    for old_path, new in FIELD_RENAMES.get(tool, {}).items():
        parent, _, old = old_path.rpartition(".")
        for mapping, where, shown in _reach(config, _split(parent), tool, at, ""):
            if not isinstance(mapping, Mapping) or old not in mapping:
                continue
            field = f"{shown}.{old}" if shown else old
            if new in mapping:
                sentence = f"{here}: config field {field!r} is now {new!r}, and {new!r} is written too. Keep one by hand"
                found.append(Found("manual", (*where, old), old, new, "", sentence, step_id))
                continue
            found.append(
                _key((*where, old), old, new, f"{tool}.{old_path} -> {new}", f"{here}: config field {field!r} is now {new!r}", step_id)
            )

    for new, value in OLD_DEFAULTS.get(written, {}).items():
        olds = _old_names(tool, "", new)
        if new not in config and not any(old in config for old in olds):
            found.append(
                Found(
                    "insert",
                    at,
                    value,
                    new,
                    f"{tool}.{new}: old default {value} written",
                    f"{here}: config field {new!r} is written as {value!r}, the old default",
                    step_id,
                    retired=False,
                )
            )
    return found


def _split(path: str) -> list[str]:
    return path.split(".") if path else []


def _old_names(tool: str, parent: str, new: str) -> list[str]:
    """The old names of field ``new`` under ``parent`` (a path in new names) for ``tool``."""
    olds = []
    for old_path, renamed in FIELD_RENAMES.get(tool, {}).items():
        old_parent, _, old = old_path.rpartition(".")
        if renamed == new and old_parent == parent:
            olds.append(old)
    return olds


def _reach(
    value: Any, segments: list[str], tool: str, path: tuple[Any, ...], parent: str, shown: str = ""
) -> Iterator[tuple[Any, tuple[Any, ...], str]]:
    """Each value ``segments`` reaches from ``value``, with its path and the path to show a person.

    A segment is written in the new names. The walk also accepts the field's old names, so it works
    on a document in old names, new names or a mix.
    """
    if not segments:
        yield value, path, shown
        return
    if not isinstance(value, Mapping):
        return
    head, rest = segments[0], segments[1:]
    name = head.removesuffix("[]")
    below = f"{parent}.{head}" if parent else head
    for key in (name, *_old_names(tool, parent, name)):
        if key not in value:
            continue
        child = value[key]
        here = f"{shown}.{key}" if shown else key
        if head.endswith("[]"):
            if isinstance(child, list):
                for index, item in enumerate(child):
                    yield from _reach(item, rest, tool, (*path, key, index), below, f"{here}[{index}]")
        else:
            yield from _reach(child, rest, tool, (*path, key), below, here)


def _expression_paths(tool: str, registry: Any) -> tuple[str, ...]:
    """The paths of a tool's expression fields, read from its template. None for an unknown tool."""
    from .config import expression_fields
    from .errors import RegistryError
    from .registry import REGISTRY

    try:
        found = (registry if registry is not None else REGISTRY).get(tool)
    except RegistryError:
        return ()
    return expression_fields(found.config_template())


# ---------------------------------------------------------------------------
# Function calls inside an expression
# ---------------------------------------------------------------------------


def call_pattern(name: str) -> re.Pattern[str]:
    """``name`` followed by an opening bracket, not part of a longer word or after a dot."""
    return re.compile(rf"(?<![\w.]){re.escape(name)}(?=\s*\()")


def function_calls(text: str, name: str) -> tuple[list[int], int | None]:
    """Where ``text`` calls function ``name``, and how many calls the library's own parser counts.

    The offsets skip quoted text, column references and ``${...}`` placeholders. The count comes
    from the expression scanner and Python's parser. It is ``None`` when ``text`` is not an
    expression at all. When the two disagree, a person should look.
    """
    from .expr.errors import ExpressionError
    from .expr.scan import scan

    hidden = _hidden_spans(text)
    offsets = [
        match.start()
        for match in call_pattern(name).finditer(text)
        if not any(start <= match.start() < end for start, end in hidden)
    ]
    # A placeholder stands in for a value. A number reads anywhere outside quotes, a date inside date'...'.
    for stand_in in ("0", "2000-01-01"):
        try:
            tree = ast.parse(scan(_PLACEHOLDER.sub(stand_in, text)).source, mode="eval")
            break
        except (ExpressionError, SyntaxError, ValueError):
            continue
    else:
        return offsets, None
    count = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    )
    return offsets, count


def _hidden_spans(text: str) -> list[tuple[int, int]]:
    """The spans of quoted text, column references and placeholders, read as the scanner reads them."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text.startswith("${", index):
            end = text.find("}", index)
        elif text[index] == "[":
            end = text.find("]", index)
        elif text[index] == "'":
            end = _quote_end(text, index)
        else:
            index += 1
            continue
        end = len(text) - 1 if end == -1 else end
        spans.append((index, end + 1))
        index = end + 1
    return spans


def _quote_end(text: str, start: int) -> int:
    """Where the quote opened at ``start`` closes. A doubled quote inside is one quote."""
    index = start + 1
    while True:
        end = text.find("'", index)
        if end == -1 or text[end + 1 : end + 2] != "'":
            return end
        index = end + 2
