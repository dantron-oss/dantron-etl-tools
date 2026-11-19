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
    "RENAME_NOTES",
    "STATED_NOW",
    "as_written",
    "Found",
    "call_pattern",
    "find_renames",
    "function_calls",
    "port_map",
    "rename_document",
    "retired_names",
    "summarise",
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
    ("group_by", "aggregations[].function"): {"count_values": "count_non_null", "n_unique": "count_distinct"},
    ("cross_tab", "function"): {"count_values": "count_non_null", "n_unique": "count_distinct"},
}

# A renamed value whose meaning moved with its name, and what moved. The migration's report and the
# loader's refusal both say it, so nobody takes the new name for the old answer.
RENAME_NOTES: dict[tuple[str, str], str] = {
    ("n_unique", "count_distinct"): (
        "count_distinct skips nulls, as SQL's COUNT(DISTINCT) does, where n_unique counted a null as "
        "one more value: a group holding a null counts one fewer"
    ),
}

# Old defaults that differ from the new ones. Keyed by the tool name as the file writes it, then by
# the new field name. A step with neither the field nor its old name gets the old value written in.
OLD_DEFAULTS: dict[str, dict[str, str]] = {
    "date_diff": {"counting": "inclusive"},
    "connected_components": {"record_column": "node", "group_column": "component"},
}

# Settings a step must now state, where the file alone says which value keeps what the step did
# before. The migration writes that value in. Each entry: the tool, the path of the mapping the
# setting goes in (written in the new names, "" for the step's config itself), the values that
# mapping must hold for the setting to apply, the setting, and the value that keeps the old behaviour.
STATED_NOW: tuple[tuple[str, str, Mapping[str, Any], str, Any], ...] = (
    ("publish_tables", "tables[]", {"mode": "merge"}, "when_incoming_is_null", "keep_published"),
)

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
    old: Any
    new: str
    label: str
    sentence: str
    step: str | None = None
    retired: bool = True


def retired_names(document: Mapping[str, Any], registry: Any = None) -> list[str]:
    """One sentence per retired name ``document`` uses. Empty when it uses only current names."""
    return [found.sentence for found in find_renames(document, registry) if found.retired]


def summarise(found: list[Found], *, examples: int = 3, limit: int = 12) -> list[str]:
    """One line per kind of rename among the retired names in ``found``, with a count and examples.

    A file of five hundred steps that all use one old key gives one line, not five hundred. At most
    ``limit`` lines; any kinds beyond that are counted on the last one.
    """
    groups: dict[tuple[str, str], list[str | None]] = {}
    for item in found:
        if item.retired:
            groups.setdefault(_kind_of(item), []).append(item.step)
    lines = []
    for (noun, text), steps in groups.items():
        if noun:
            count = len(steps)
            head = f"{count} {noun if count != 1 else noun.removesuffix('s')}: {text}"
        else:
            head = text
        named = list(dict.fromkeys(step for step in steps if step is not None))
        if named:
            shown = ", ".join(repr(step) for step in named[:examples])
            head += f" (for example {shown})" if len(named) > examples else f" ({shown})"
        lines.append(head)
    if len(lines) > limit:
        lines = [*lines[: limit - 1], f"and {len(lines) - limit + 1} other kinds of rename"]
    return lines


def _kind_of(item: Found) -> tuple[str, str]:
    """What a summary line counts (steps, inputs, ...) and what it says, for one retired name."""
    old, new, label = repr(item.old), repr(item.new), item.label
    if item.kind == "manual":
        if item.path and item.path[-1] == "tool":
            return "steps", f"the old single-sheet {item.old} was retired; rewrite it by hand with a 'sheets' list"
        return "steps", f"config field {old} is written beside its new name {new}; keep one by hand"
    if item.kind == "call":
        return "expressions", f"function {item.old}() is now {item.new}()"
    if label.startswith("port "):
        tool, _, port = label.removeprefix("port ").partition(" -> ")[0].partition(".")
        return "inputs", f"{tool}'s output port {port!r} is now {item.new.rpartition('.')[2]!r}"
    if label.startswith("input port "):
        tool = label.removeprefix("input port ").partition(".")[0]
        return "steps", f"{tool}'s input port {old} is now {new}"
    if label.startswith("tool "):
        return "steps", f"tool {old} is now {new}"
    if label.startswith("lane "):
        return "lanes", f"key {old} is now {new}"
    if label.startswith("step "):
        return "steps", f"key {old} is now {new}"
    if item.kind == "key" and len(item.path) == 1:
        return "", f"the top-level key {old} is now {new}"
    tool, _, rest = label.partition(".")
    if item.kind == "value":
        field = rest.partition(":")[0]
        moved = RENAME_NOTES.get((item.old, item.new))
        return "steps", f"{tool}'s config field {field!r}: the value {old} is now {new}" + (f"; {moved}" if moved else "")
    return "steps", f"{tool}'s config field {old} is now {new}"


def rename_document(document: Any, found: list[Found]) -> Any:
    """A copy of the parsed ``document`` with every rename and old default in ``found`` applied.

    Notes for a person (``manual``) change nothing. Values, calls and inserts go first, because their
    paths are written in the names the file used; keys are renamed last, the deepest first.
    """
    import copy

    renamed = copy.deepcopy(document)

    def at(path: tuple[Any, ...]) -> Any:
        value = renamed
        for part in path:
            value = value[part]
        return value

    for item in found:
        try:
            if item.kind == "value" and at(item.path) == item.old:
                at(item.path[:-1])[item.path[-1]] = item.new
            elif item.kind == "call":
                text = at(item.path)
                for offset in reversed(function_calls(text, item.old)[0]):
                    text = text[:offset] + item.new + text[offset + len(item.old) :]
                at(item.path[:-1])[item.path[-1]] = text
            elif item.kind == "insert":
                at(item.path).setdefault(item.new, item.old)
        except (KeyError, IndexError, TypeError):
            continue  # a value an alias already changed, or a key a rename above already moved
    keys = [item for item in found if item.kind == "key"]
    for item in sorted(keys, key=lambda item: len(item.path), reverse=True):
        try:
            mapping = at(item.path[:-1])
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(mapping, dict) and item.old in mapping:
            items = [(item.new if key == item.old else key, value) for key, value in mapping.items()]
            mapping.clear()
            mapping.update(items)
    return renamed


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
    written = step.get("tool")
    if not isinstance(written, str):
        return []
    step_id = step.get("id") if isinstance(step.get("id"), str) else None
    here = f"step {step_id!r}" if step_id is not None else f"step number {index + 1}"
    at = ("steps", index)
    config = step.get("config")

    single_tool, single_key = SINGLE_SHEET_TOOL
    if written == single_tool and isinstance(config, Mapping) and single_key in config:
        sentence = (
            f"{here}: this is the old single-sheet {written} (its config has {single_key!r}), which "
            f"was retired. Rewrite it by hand as {written} with a 'sheets' list"
        )
        return [Found("manual", (*at, "tool"), written, written, "", sentence, step_id)]

    tool = TOOL_RENAMES.get(written, written)
    if isinstance(config, Mapping):
        # A field written in both its old and its new name: which one was meant cannot be known.
        # Nothing in the step changes; a person decides.
        collisions = _collisions(tool, config, (*at, "config"), here, step_id)
        if collisions:
            return collisions

    found: list[Found] = []
    if tool != written:
        found.append(
            Found("value", (*at, "tool"), written, tool, f"tool {written} -> {tool}", f"{here}: tool {written!r} is now {tool!r}", step_id)
        )

    for old, new in STEP_KEY_RENAMES.items():
        if old in step:
            found.append(_key((*at, old), old, new, f"step {old} -> {new}", f"{here}: the key {old!r} is now {new!r}", step_id))

    if isinstance(config, Mapping):
        found.extend(_config_renames(tool, written, config, (*at, "config"), here, step_id, registry))
        found.extend(_unknown_defaults(tool, written, config, (*at, "config"), here, step_id))

    inputs = step.get("inputs")
    if isinstance(inputs, Mapping):
        found.extend(_input_renames(tool, inputs, (*at, "inputs"), here, step_id, tool_by_id))
    return found


def _collisions(tool: str, config: Mapping[str, Any], at: tuple[Any, ...], here: str, step_id: str | None) -> list[Found]:
    """A note for each config field written in both its old and its new name."""
    found: list[Found] = []
    for old_path, new in FIELD_RENAMES.get(tool, {}).items():
        parent, _, old = old_path.rpartition(".")
        for mapping, where, shown in _reach(config, _split(parent), tool, at, ""):
            if isinstance(mapping, Mapping) and old in mapping and new in mapping:
                field = f"{shown}.{old}" if shown else old
                sentence = (
                    f"{here}: config field {field!r} is now {new!r}, and {new!r} is written too. "
                    "Keep one by hand; nothing in this step was changed"
                )
                found.append(Found("manual", (*where, old), old, new, "", sentence, step_id))
    return found


def _unknown_defaults(
    tool: str, written: str, config: Mapping[str, Any], at: tuple[Any, ...], here: str, step_id: str | None
) -> list[Found]:
    """A note for each old default a half-migrated step may still rely on.

    The step names the new tool but still uses an old field name. So it may have been written for
    the old tool, whose defaults differ. Whether it meant the old default or the new one cannot be
    known, so nothing is written in.
    """
    old_tools = [old for old, new in TOOL_RENAMES.items() if new == written]
    if not old_tools:
        return []
    old_fields = [old_path for old_path in FIELD_RENAMES.get(tool, {}) if "." not in old_path and old_path in config]
    if not old_fields:
        return []
    found: list[Found] = []
    for new, value in OLD_DEFAULTS.get(old_tools[0], {}).items():
        if new in config or any(old in config for old in _old_names(tool, "", new)):
            continue
        sentence = (
            f"{here}: config field {new!r} is not written. The step names {written!r} but still uses the "
            f"old field {old_fields[0]!r}, so it may rely on the old default {value!r}. Write {new!r} by hand"
        )
        found.append(Found("manual", at, value, new, "", sentence, step_id, retired=False))
    return found


def _input_renames(
    tool: str,
    inputs: Mapping[str, Any],
    at: tuple[Any, ...],
    here: str,
    step_id: str | None,
    tool_by_id: Mapping[str, str],
) -> list[Found]:
    """A step's renamed input ports, and the inputs that read a renamed output port of another step."""
    found: list[Found] = []
    for port, source in inputs.items():
        renamed_port = INPUT_PORT_RENAMES.get(tool, {}).get(port) if isinstance(port, str) else None
        if renamed_port is not None:
            found.append(
                _key(
                    (*at, port),
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
                    (*at, port),
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
                moved = RENAME_NOTES.get((value, new))
                found.append(
                    Found(
                        "value",
                        where,
                        value,
                        new,
                        f"{tool}.{path}: {value} -> {new}" + (f" ({moved})" if moved else ""),
                        f"{here}: config field {shown!r} has the value {value!r}, which is now {new!r}"
                        + (f"; {moved}" if moved else ""),
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
            found.append(
                _key((*where, old), old, new, f"{tool}.{old_path} -> {new}", f"{here}: config field {field!r} is now {new!r}", step_id)
            )

    for owner, path, condition, setting, value in STATED_NOW:
        if owner != tool:
            continue
        for mapping, where, shown in _reach(config, _split(path), tool, at, ""):
            if not isinstance(mapping, Mapping) or setting in mapping:
                continue
            if any(mapping.get(key) != wanted for key, wanted in condition.items()):
                continue
            field = f"{shown}.{setting}" if shown else setting
            found.append(
                Found(
                    "insert",
                    where,
                    value,
                    setting,
                    f"{tool}.{path + '.' if path else ''}{setting}: {as_written(value)} written (keeps today's behaviour)",
                    f"{here}: config field {field!r} is written as {as_written(value)!r}, which keeps what it did before",
                    step_id,
                    retired=False,
                )
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


def as_written(value: Any) -> str:
    """A scalar as it is written in a pipeline file: true and false in lower case, text as it is."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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
    """The paths of a tool's expression fields, read from its template. Empty for an unknown tool."""
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
