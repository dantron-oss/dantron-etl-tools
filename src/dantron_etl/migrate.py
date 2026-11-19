"""Rewrite a pipeline file written before the renames, in the new names.

Run it as ``python -m`` this module, with files or folders. By default it only reports what it
would change. ``--write`` changes the files.

The file is edited as text, never loaded and written back. Only the characters of a renamed name
change, so comments, layout, quoting and ``why`` text stay exactly as they were. What to rename
comes from the rename table in ``renames.py``, walked per tool and per path.

After editing, the new text is read back and compared with the old document with the renames
applied. If the two differ, the file is refused rather than written wrong.
"""

import copy
import re
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .renames import (
    FUNCTION_RENAMES,
    as_written,
    MIGRATE_COMMAND,
    PYTHON_RENAMES,
    Found,
    call_pattern,
    find_renames,
    function_calls,
    port_map,
)

__all__ = ["Change", "Migration", "MigrationError", "main", "migrate_text", "rule_uses"]

#: How a pipeline uses a rule: as the whole of an expression field, where its value is read as an
#: expression, or anywhere else, where its value is text (a pattern, a name, or a quoted literal).
EXPRESSION_USE = "expression"
OTHER_USE = "other"


class MigrationError(Exception):
    """A file the migration cannot read, or cannot rewrite safely."""


@dataclass(frozen=True)
class Change:
    """One rename made: its label, its line in the original text, and its step (or None)."""

    rename: str
    line: int
    step: str | None


@dataclass
class Migration:
    """The migrated text, the changes made, a count per rename, and what to do by hand."""

    text: str
    changes: list[Change] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    manual: list[str] = field(default_factory=list)


def migrate_text(text: str, *, rule_uses: Mapping[str, set[str]] | None = None) -> Migration:
    """``text`` rewritten in the new names. Refuses text that is not a YAML mapping.

    For a rules file, ``rule_uses`` says how the pipelines being migrated with it use each rule (see
    :func:`rule_uses`). A rule's value is rewritten only when every use is as a whole expression.
    """
    import yaml

    try:
        root = yaml.compose(text)
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise MigrationError(f"it is not readable as YAML: {error}") from error
    if not isinstance(document, dict) or not isinstance(root, yaml.MappingNode):
        raise MigrationError("it does not hold a mapping of keys, so it is neither a pipeline file nor a rules file")

    found = find_renames(document) if "steps" in document else _settings_calls(document, rule_uses or {})
    editor = _Editor(text, root)
    for item in found:
        editor.apply(item)
    editor.check_shared(document)

    new_text = editor.result()
    expected = _renamed(document, editor.data_edits)
    try:
        reread = yaml.safe_load(new_text)
    except yaml.YAMLError as error:
        raise MigrationError(f"the rewritten text would not be readable YAML: {error}") from error
    if reread != expected:
        raise MigrationError("the rewritten text does not mean what the renames say it should, so it was not written")

    changes = sorted(editor.changes, key=lambda change: change.line)
    return Migration(new_text, changes, dict(Counter(change.rename for change in changes)), editor.manual)


def _settings_calls(document: Mapping[str, Any], uses: Mapping[str, set[str]]) -> list[Found]:
    """In a rules file, the old function calls in each rule's value: rewritten, or noted for a person.

    A rule's value is code only where a pipeline uses the rule as the whole of an expression field.
    Anywhere else it is text: a pattern, a name, or a value quoted into an expression. So a call is
    rewritten only in a rule every use of which, among the pipelines given, is a whole expression.
    Any other rule holding what looks like a call is left alone and noted.
    """
    found: list[Found] = []

    def walk(value: Any, path: tuple[Any, ...]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, (*path, key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, index))
        elif isinstance(value, str):
            for old, new in FUNCTION_RENAMES.items():
                offsets, count = function_calls(value, old)
                if count is None or not (offsets or count):
                    continue
                rule = str(path[0])
                if uses.get(rule) == {EXPRESSION_USE}:
                    found.append(Found("call", path, old, new, f"{old}() -> {new}()", f"rule {rule!r} calls {old!r}, now {new!r}"))
                    continue
                if not uses.get(rule):
                    why = "no pipeline file given uses this rule"
                else:
                    why = "a pipeline file given uses this rule as text, not as a whole expression"
                sentence = (
                    f"not migrated automatically: rule {rule!r} = {value!r} looks like it may contain a "
                    f"renamed function ({old}() is now {new}()); {why}, so check by hand"
                )
                found.append(Found("manual", path, old, new, "", sentence, retired=False))

    walk(document, ())
    return found


def rule_uses(document: Any, registry: Any = None) -> dict[str, set[str]]:
    """How a pipeline document uses each ``${rules.name}`` it references.

    A reference that is the whole of an expression field is an expression use. A reference anywhere
    else, including inside the text of an expression, where the value is quoted in as text, is an
    other use. Fields are read from each tool's template, so the document should be in the new names.
    """
    from .errors import RegistryError
    from .registry import REGISTRY

    active = registry if registry is not None else REGISTRY
    uses: dict[str, set[str]] = {}
    whole = re.compile(r"\$\{rules\.([^}]*)\}")

    def walk(value: Any, spec: Any) -> None:
        if isinstance(value, str):
            matched = whole.fullmatch(value)
            if matched is not None and spec is not None and spec.type == "expression":
                uses.setdefault(matched.group(1), set()).add(EXPRESSION_USE)
                return
            for reference in whole.finditer(value):
                uses.setdefault(reference.group(1), set()).add(OTHER_USE)
        elif isinstance(value, list):
            element = spec.items if spec is not None and spec.type == "array" else None
            for item in value:
                walk(item, element)
        elif isinstance(value, dict):
            members = spec.fields if spec is not None and spec.type == "object" and spec.fields else {}
            for key, item in value.items():
                walk(item, members.get(key))

    for step in document.get("steps") or () if isinstance(document, Mapping) else ():
        if not isinstance(step, Mapping) or not isinstance(step.get("config"), Mapping):
            continue
        try:
            template = active.get(step.get("tool")).config_template()
        except (RegistryError, TypeError):
            template = {}
        for key, item in step["config"].items():
            walk(item, template.get(key))
    return uses


class _Editor:
    """Collects text edits for each rename found, and the same edits as changes to plain data."""

    def __init__(self, text: str, root: Any) -> None:
        self.text = text
        self.root = root
        self.edits: list[tuple[int, int, str]] = []
        self.data_edits: list[tuple[str, tuple[Any, ...], Any, Any]] = []
        self.changes: list[Change] = []
        self.manual: list[str] = []
        # Node id -> (node, step index -> the edits that step makes to that node). A node written
        # once under an anchor and reused by an alias is one node, reached from every step using it.
        self.by_node: dict[int, tuple[Any, dict[int | None, set[tuple[int, int, str]]]]] = {}

    def result(self) -> str:
        text = self.text
        # From the end of the text backwards, so each edit's offsets still hold. Two inserts at one
        # place keep the order they were found in.
        ordered = sorted(enumerate(self.edits), key=lambda item: (item[1][0], item[1][1], item[0]), reverse=True)
        for _, (start, end, replacement) in ordered:
            text = text[:start] + replacement + text[end:]
        return text

    def check_shared(self, document: Mapping[str, Any]) -> None:
        """Refuse a node shared by steps that would edit it differently.

        Steps sharing one node through an alias and making the same edits to it are edited once.
        When one step would change the shared text and another would change it differently, or not
        at all, no single text serves both, so the file is refused.
        """
        import yaml

        steps = _value_of(self.root, "steps")
        if not isinstance(steps, yaml.SequenceNode):
            return
        owners: dict[int, list[int]] = {}
        anchors: dict[int, str] = {}
        for index, step in enumerate(steps.value):
            for node_id in self._reachable(step, None, anchors):
                owners.setdefault(node_id, []).append(index)
        for node_id, (node, made) in self.by_node.items():
            sharing = owners.get(node_id, [])
            if len(sharing) < 2 or len({frozenset(made.get(index, ())) for index in sharing}) == 1:
                continue
            ids = [_step_name(document, index) for index in sharing]
            anchor = anchors.get(node_id)
            written = f"the text under the anchor &{anchor}" if anchor else "the same text"
            raise MigrationError(
                f"steps {', '.join(ids)} share {written} (line {self._line(node)}) and need different "
                "edits to it. Write each step's config out in full by hand, then migrate again"
            )

    def _reachable(self, node: Any, anchor: str | None, anchors: dict[int, str], seen: set[int] | None = None) -> set[int]:
        """The ids of every node inside ``node``, noting the innermost anchor each sits under."""
        import yaml

        seen = set() if seen is None else seen
        if id(node) in seen:
            return seen
        seen.add(id(node))
        match = _ANCHOR.match(self.text, node.start_mark.index)
        anchor = match.group(1) if match else anchor
        if anchor is not None:
            anchors.setdefault(id(node), anchor)
        if isinstance(node, yaml.MappingNode):
            children = [part for pair in node.value for part in pair]
        elif isinstance(node, yaml.SequenceNode):
            children = node.value
        else:
            children = []
        for child in children:
            self._reachable(child, anchor, anchors, seen)
        return seen

    def _edit(self, found: Found, node: Any, edit: tuple[int, int, str]) -> bool:
        """Record ``edit`` to ``node`` for the step of ``found``. False when it was already made."""
        path = found.path
        step = path[1] if len(path) > 1 and path[0] == "steps" and isinstance(path[1], int) else None
        self.by_node.setdefault(id(node), (node, {}))[1].setdefault(step, set()).add(edit)
        if edit in self.edits:
            return False
        self.edits.append(edit)
        return True

    def apply(self, found: Found) -> None:
        key_node, node = self._locate(found.path)
        if found.kind == "manual":
            self._by_hand(node, found.sentence)
        elif found.kind == "key":
            if self._replace(found, key_node):
                self.data_edits.append(("key", found.path[:-1], found.old, found.new))
        elif found.kind == "value":
            if self._replace(found, node):
                self.data_edits.append(("set", found.path, None, found.new))
        elif found.kind == "insert":
            self._insert(found, node)
        elif found.kind == "call":
            self._calls(found, node)

    def _locate(self, path: tuple[Any, ...]) -> tuple[Any, Any]:
        """The key node and the value node at ``path``."""
        import yaml

        key_node, node = None, self.root
        for part in path:
            if isinstance(node, yaml.MappingNode):
                pairs = [(key, value) for key, value in node.value if isinstance(key, yaml.ScalarNode) and key.value == part]
                if not pairs:
                    raise MigrationError(f"cannot find {part!r} in the text (a merged or aliased mapping is not migrated)")
                key_node, node = pairs[-1]
            elif isinstance(node, yaml.SequenceNode) and isinstance(part, int):
                key_node, node = None, node.value[part]
            else:
                raise MigrationError(f"cannot find {part!r} in the text")
        return key_node, node

    def _line(self, node: Any) -> int:
        return node.start_mark.line + 1

    def _by_hand(self, node: Any, sentence: str) -> None:
        """Note something to do by hand. The sentence already names the step."""
        self.manual.append(f"line {self._line(node)}: {sentence}")

    def _change(self, found: Found, line: int) -> None:
        self.changes.append(Change(found.label, line, found.step))

    def _replace(self, found: Found, node: Any) -> bool:
        """Replace the characters of a scalar's value, inside its quotes. False when it cannot."""
        start, end = node.start_mark.index, node.end_mark.index
        if getattr(node, "style", None) in ("'", '"'):
            start, end = start + 1, end - 1
        if self.text[start:end] != found.old:
            self._by_hand(node, f"{found.sentence}; it is written in a way this tool does not edit")
            return False
        if self._edit(found, node, (start, end, found.new)):
            self._change(found, self._line(node))
        return True

    def _insert(self, found: Found, mapping: Any) -> None:
        """Write ``new: old`` into a mapping, as one more entry in its own style."""
        import yaml

        if not isinstance(mapping, yaml.MappingNode):
            self._by_hand(mapping, f"{found.sentence}; the config is not a mapping, so write it by hand")
            return
        entry = f"{found.new}: {as_written(found.old)}"
        if mapping.flow_style:
            if mapping.value:
                at = mapping.value[-1][1].end_mark.index
                edit = (at, at, f", {entry}")
            else:
                at = mapping.end_mark.index - 1
                edit = (at, at, entry)
        else:
            indent = " " * mapping.value[0][0].start_mark.column
            newline = "\r\n" if "\r\n" in self.text else "\n"
            last = _last_written(mapping).end_mark
            if last.column == 0:
                edit = (last.index, last.index, f"{indent}{entry}{newline}")
            else:
                at = self.text.find(newline, last.index)
                at = len(self.text) if at == -1 else at
                edit = (at, at, f"{newline}{indent}{entry}")
        self.data_edits.append(("insert", found.path, found.new, found.old))
        if self._edit(found, mapping, edit):
            self._change(found, self._line(mapping))

    def _calls(self, found: Found, node: Any) -> None:
        """Rename each real call in an expression scalar, or say why a person should look."""
        value = node.value
        offsets, count = function_calls(value, found.old)
        if count is None:
            self._by_hand(node, f"{found.sentence}; the text does not read as an expression, so check it by hand")
            return
        if count != len(offsets):
            self._by_hand(node, f"{found.sentence}; the expression scanner and the text disagree, so check it by hand")
            return
        pattern = call_pattern(found.old)
        parsed = [match.start() for match in pattern.finditer(value)]
        start = node.start_mark.index
        raw = [start + match.start() for match in pattern.finditer(self.text[start : node.end_mark.index])]
        if len(raw) != len(parsed):
            self._by_hand(node, f"{found.sentence}; the text as written and as read disagree, so check it by hand")
            return
        size = len(found.old)
        for written, read in zip(raw, parsed, strict=True):
            if read in offsets and self._edit(found, node, (written, written + size, found.new)):
                self._change(found, self.text.count("\n", 0, written) + 1)
        new_value = value
        for read in sorted(offsets, reverse=True):
            new_value = new_value[:read] + found.new + new_value[read + size :]
        self.data_edits.append(("set", found.path, None, new_value))


_ANCHOR = re.compile(r"&([^\s,\[\]{}]+)")


def _value_of(mapping: Any, key: str) -> Any:
    """The value node under ``key`` in a mapping node, or None."""
    for key_node, value in getattr(mapping, "value", []):
        if getattr(key_node, "value", None) == key:
            return value
    return None


def _step_name(document: Mapping[str, Any], index: int) -> str:
    step = document["steps"][index]
    step_id = step.get("id") if isinstance(step, Mapping) else None
    return repr(step_id) if isinstance(step_id, str) else f"number {index + 1}"


def _last_written(node: Any) -> Any:
    """The node written last inside a block collection.

    A block collection's own end mark sits at the next token, which may be on a later line. A
    scalar's, or a flow collection's, sits right after its last character.
    """
    import yaml

    while isinstance(node, (yaml.MappingNode, yaml.SequenceNode)) and not node.flow_style and node.value:
        last = node.value[-1]
        node = last[1] if isinstance(node, yaml.MappingNode) else last
    return node


def _renamed(document: Any, edits: list[tuple[str, tuple[Any, ...], Any, Any]]) -> Any:
    """``document`` with ``edits`` applied to the plain data: values, then inserts, then keys."""
    document = copy.deepcopy(document)

    def at(path: tuple[Any, ...]) -> Any:
        value = document
        for part in path:
            value = value[part]
        return value

    for kind, path, _, new in edits:
        if kind == "set":
            at(path[:-1])[path[-1]] = new
    for kind, path, key, value in edits:
        if kind == "insert":
            at(path)[key] = value
    keys = [edit for edit in edits if edit[0] == "key"]
    for _, path, old, new in sorted(keys, key=lambda edit: len(edit[1]), reverse=True):
        mapping = at(path)
        items = [(new if key == old else key, value) for key, value in mapping.items()]
        mapping.clear()
        mapping.update(items)
    return document


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Report, or with ``--write`` make, the renames in each file. Returns 0, 1 or 2.

    0: nothing to change, or the changes were written. 1: a dry run found changes to make, or a
    file still has something to do by hand. 2: a file could not be read or rewritten safely. It is
    named and left as it was.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog=MIGRATE_COMMAND,
        description="Rewrite pipeline and rules files written before the renames, in the new names.",
    )
    parser.add_argument("paths", nargs="+", help="files, or folders holding *.yaml and *.yml files")
    parser.add_argument("--write", action="store_true", help="change the files; without it, only report")
    parser.add_argument("--port-map", metavar="PATH", help="also write the renamed output ports to PATH as JSON")
    arguments = parser.parse_args(argv)

    refused = changed = by_hand = False
    texts: dict[Path, str] = {}
    for path in _files(arguments.paths):
        try:
            texts[path] = path.read_bytes().decode("utf8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"{path}: not migrated, {error}")
            refused = True

    # Pipelines first: how they use each rule decides what may change in a rules file.
    results: dict[Path, Migration | MigrationError] = {}
    uses: dict[str, set[str]] = {}
    for path, text in texts.items():
        if not _is_pipeline(text):
            continue
        try:
            results[path] = migrate_text(text)
            for rule, kinds in rule_uses(_parsed(results[path].text)).items():
                uses.setdefault(rule, set()).update(kinds)
        except MigrationError as error:
            results[path] = error
            for rule in re.findall(r"\$\{rules\.([^}]*)\}", text):
                uses.setdefault(rule, set()).add(OTHER_USE)  # unknown use: never rewrite the rule
    for path, text in texts.items():
        if path not in results:
            try:
                results[path] = migrate_text(text, rule_uses=uses)
            except MigrationError as error:
                results[path] = error

    for path, text in texts.items():
        migration = results[path]
        if isinstance(migration, MigrationError):
            print(f"{path}: not migrated, {migration}")
            refused = True
            continue
        _report(path, migration, arguments.write)
        by_hand = by_hand or bool(migration.manual)
        if migration.text != text:
            changed = True
            if arguments.write:
                path.write_bytes(migration.text.encode("utf8"))

    if arguments.port_map:
        Path(arguments.port_map).write_text(json.dumps(port_map(), indent=2) + "\n", encoding="utf8")

    print("\nPython code is not rewritten. Rename these by hand:")
    for old, new in PYTHON_RENAMES:
        print(f"  {old} -> {new}")

    if refused:
        return 2
    return 1 if by_hand or (changed and not arguments.write) else 0


def _parsed(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


def _is_pipeline(text: str) -> bool:
    """Whether ``text`` reads as a pipeline file: a mapping with a list of steps."""
    import yaml

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return "\nsteps:" in f"\n{text}"
    return isinstance(document, dict) and "steps" in document


def _files(paths: list[str]) -> list[Path]:
    """Each file named, and every *.yaml and *.yml file under each folder named."""
    files: list[Path] = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            files.extend(sorted(found for found in path.rglob("*") if found.suffix in (".yaml", ".yml") and found.is_file()))
        else:
            files.append(path)
    return files


def _report(path: Path, migration: Migration, written: bool) -> None:
    if not migration.changes:
        print(f"{path}: nothing to change")
    else:
        print(f"{path}: {len(migration.changes)} change(s) {'made' if written else 'to make'}")
        for label, count in migration.counts.items():
            print(f"  {count} x {label}")
    for entry in migration.manual:
        print(f"  by hand: {entry}")


if __name__ == "__main__":
    sys.exit(main())
