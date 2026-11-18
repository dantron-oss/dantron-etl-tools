"""The words a pipeline file is written in, pinned in one file.

A consumer's pipeline file names tools, ports, config fields, enum values and expression functions.
Renaming any of them, or changing a default, breaks that file at load or quietly changes what it
does. ``tests/fixtures/vocabulary.json`` holds all of those words as the library has them today.
This test fails when the library and the file disagree, and prints the difference.

When a change is deliberate, regenerate the file and commit it with the change:

    python tests/test_dantron_etl_vocabulary.py

Say in the commit message that the change breaks existing pipeline files.
"""

import ast
import difflib
import json
import sys
from pathlib import Path
from typing import Any

VOCABULARY_FILE = Path(__file__).resolve().parent / "fixtures" / "vocabulary.json"
REGENERATE = "python tests/test_dantron_etl_vocabulary.py"


def _field(spec: Any) -> dict[str, Any]:
    """One config field: its type, whether it is required, its default, its allowed values."""
    return {
        "type": spec.type,
        "required": spec.required,
        "default": _plain(spec.default),
        "enum": list(spec.enum) if spec.enum is not None else None,
        "non_empty": spec.non_empty,
        "items": _field(spec.items) if spec.items is not None else None,
        "fields": {name: _field(member) for name, member in spec.fields.items()} if spec.fields is not None else None,
    }


def _plain(value: Any) -> Any:
    """A default as JSON holds it: tuples become lists, mappings become objects."""
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


def _pipeline_file_keys() -> dict[str, list[str]]:
    """The keys the loader reads from a pipeline file, found by reading the loader's own source.

    ``document`` is the whole file. ``raw`` is one entry of its ``steps``, ``lanes`` or ``groups``.
    Every literal key read as ``document["k"]``, ``document.get("k")``, ``raw["k"]`` or
    ``raw.get("k")`` is listed.
    """
    from dantron_etl import pipeline

    keys: dict[str, set[str]] = {"document": set(), "entry": set()}
    roles = {"document": "document", "raw": "entry"}
    for node in ast.walk(ast.parse(Path(pipeline.__file__).read_text(encoding="utf8"))):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            owner, key = node.value.id, node.slice
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.args
        ):
            owner, key = node.func.value.id, node.args[0]
        else:
            continue
        if owner in roles and isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys[roles[owner]].add(key.value)
    return {role: sorted(names) for role, names in keys.items()}


def current_vocabulary() -> dict[str, Any]:
    """Every word a pipeline file can use, read from the live library."""
    import dantron_etl  # noqa: F401 -- importing the package fills the registry
    from dantron_etl.expr import ALLOWED_FUNCTIONS
    from dantron_etl.expr.functions import FUNCTIONS
    from dantron_etl.tool import REGISTRY, ports_from_config

    tools = {
        tool.name: {
            "category": tool.category,
            "input_ports": "from config" if ports_from_config(tool) else list(tool.input_ports),
            "output_ports": list(tool.output_ports),
            "config": {name: _field(spec) for name, spec in tool.config_template().items()},
        }
        for tool in REGISTRY.tools()
    }
    functions = {
        name: "two or more" if FUNCTIONS[name].variadic else len(FUNCTIONS[name].params)
        for name in sorted(ALLOWED_FUNCTIONS)
    }
    return {"tools": tools, "expression_functions": functions, "pipeline_file_keys": _pipeline_file_keys()}


def _as_text(vocabulary: dict[str, Any]) -> str:
    return json.dumps(vocabulary, indent=2, sort_keys=True) + "\n"


def _lines(value: Any, path: str = "") -> list[str]:
    """One line per leaf, each carrying its full path, so a diff line says which tool and field it is."""
    if isinstance(value, dict) and value:
        return [line for key in sorted(value) for line in _lines(value[key], f"{path}.{key}" if path else key)]
    return [f"{path} = {json.dumps(value)}"]


def test_the_vocabulary_is_the_one_written_down() -> None:
    """A rename, a new field, or a changed default shows here as a diff that must be committed on purpose."""
    expected = json.loads(VOCABULARY_FILE.read_text(encoding="utf8"))
    actual = current_vocabulary()
    if actual != expected:
        diff = "\n".join(
            line
            for line in difflib.unified_diff(_lines(expected), _lines(actual), "committed", "library now", n=0, lineterm="")
            if not line.startswith("@@")
        )
        raise AssertionError(
            "The library's pipeline vocabulary no longer matches tests/fixtures/vocabulary.json.\n"
            "Each line below changes what an existing pipeline file means or whether it loads.\n"
            f"If the change is deliberate, regenerate the file with:\n\n    {REGENERATE}\n\n"
            "and say in the commit message that it breaks existing pipeline files.\n\n" + diff
        )
    assert VOCABULARY_FILE.read_text(encoding="utf8") == _as_text(actual), f"the file is not in its canonical form: {REGENERATE}"


def test_the_vocabulary_covers_every_tool_and_function() -> None:
    """The snapshot is not empty, and nothing is left out of it."""
    import dantron_etl  # noqa: F401
    from dantron_etl.expr import ALLOWED_FUNCTIONS
    from dantron_etl.tool import REGISTRY

    vocabulary = current_vocabulary()
    assert sorted(vocabulary["tools"]) == sorted(REGISTRY.names())
    assert sorted(vocabulary["expression_functions"]) == sorted(ALLOWED_FUNCTIONS)
    assert {"id", "tool", "config", "inputs"} <= set(vocabulary["pipeline_file_keys"]["entry"])
    assert {"name", "steps"} <= set(vocabulary["pipeline_file_keys"]["document"])


if __name__ == "__main__":
    # This checkout's own source, ahead of any installed copy.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    VOCABULARY_FILE.write_text(_as_text(current_vocabulary()), encoding="utf8")
    print(f"wrote {VOCABULARY_FILE}")
