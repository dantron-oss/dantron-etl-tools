"""The code examples in README.md and MANUAL.md work as a reader would copy them.

- A block fenced ``yaml`` is a whole pipeline file: the loader must accept it.
- A block fenced ``yaml fragment`` is part of one (a list of steps, an ``inputs:`` map, a settings
  file). It must still parse as YAML.
- A block fenced ``python`` must compile, and every name it uses must be imported or defined in
  the block (a missing import is the commonest way a copied example fails). A block fenced ``python fragment`` continues an earlier one, so only compiling is
  required.

Examples that fail today are marked ``xfail(strict=True)`` with the reason. When the docs are fixed
the mark starts failing, and is removed then.
"""

import ast
import builtins
import re
from typing import Any

import pytest
import yaml

from dantron_etl.paths import project_root
from dantron_etl import load_pipeline

PROJECT_ROOT = project_root(__file__)
DOCUMENTS = (PROJECT_ROOT / "README.md", PROJECT_ROOT / "src" / "dantron_etl" / "MANUAL.md")
_FENCE = re.compile(r"^```(?P<info>[^\n`]*)\n(?P<body>.*?)^```\s*$", re.MULTILINE | re.DOTALL)

# Blocks that fail today, by "<file> <language> #<n>" (n counts that language's blocks in that file,
# from 0). Each needs a docs change, not a test change.
_FAILING_TODAY = {
    "README.md yaml #0": "a step list with no name, reading a step 'source_lines' it never defines",
    "README.md yaml #1": "a settings file, not a pipeline: fence it as `yaml fragment`",
    "MANUAL.md yaml #0": "its last step, 'recent_orders', is a dead end: nothing reads it (F-03)",
    "MANUAL.md yaml #1": "an `inputs:` map on its own: fence it as `yaml fragment`",
    "MANUAL.md yaml #2": "declares no name and uses lane 'sources', which it does not declare",
    "MANUAL.md yaml #3": "a list of steps on its own: fence it as `yaml fragment`",
    "MANUAL.md python #1": "uses run_pipeline and spec without importing or defining them: fence as a fragment",
    "MANUAL.md python #2": "uses PortRef without importing it (G-03), and spec without defining it",
}


def _blocks() -> list[tuple[str, str, bool, str]]:
    """(id, language, is a fragment, body) for every ``yaml`` and ``python`` block in the docs."""
    found: list[tuple[str, str, bool, str]] = []
    for document in DOCUMENTS:
        counts: dict[str, int] = {}
        for match in _FENCE.finditer(document.read_text(encoding="utf8")):
            words = match["info"].split()
            if not words or words[0] not in ("yaml", "python"):
                continue
            language = words[0]
            number = counts.get(language, 0)
            counts[language] = number + 1
            found.append((f"{document.name} {language} #{number}", language, "fragment" in words[1:], match["body"]))
    return found


def _params(language: str) -> list[Any]:
    return [
        pytest.param(
            fragment,
            body,
            id=block_id,
            marks=[pytest.mark.xfail(strict=True, reason=_FAILING_TODAY[block_id])] if block_id in _FAILING_TODAY else [],
        )
        for block_id, block_language, fragment, body in _blocks()
        if block_language == language
    ]


def _bound_names(tree: ast.AST) -> set[str]:
    """Names the block itself imports or assigns."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    return bound


@pytest.mark.parametrize(("fragment", "body"), _params("yaml"))
def test_a_yaml_example_loads_as_a_pipeline_or_is_marked_a_fragment(fragment: bool, body: str) -> None:
    document = yaml.safe_load(body)
    if fragment:
        return
    assert isinstance(document, dict) and "steps" in document, (
        "not a whole pipeline file; fence it as `yaml fragment` if it is meant as a part of one"
    )
    load_pipeline(document)


@pytest.mark.parametrize(("fragment", "body"), _params("python"))
def test_a_python_example_compiles_and_imports_what_it_uses(fragment: bool, body: str) -> None:
    tree = ast.parse(body)
    if fragment:
        return
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    unbound = sorted(used - _bound_names(tree) - set(dir(builtins)) - {"__file__", "__name__"})
    assert not unbound, f"the example uses {unbound} but never imports or defines them"


def test_the_examples_are_found() -> None:
    """Guard against the fence pattern quietly matching nothing, and against a stale entry in the list."""
    found = {block_id for block_id, *_ in _blocks()}
    assert any(block_id.startswith("README.md yaml") for block_id in found)
    assert any(block_id.startswith("MANUAL.md python") for block_id in found)
    stale = sorted(set(_FAILING_TODAY) - found)
    assert not stale, f"_FAILING_TODAY names blocks that no longer exist: {stale}"
