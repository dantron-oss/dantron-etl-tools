"""The generated parts of docs/ match the library.

The tool pages, the function table, the key tables and the command help are written by
docs/_build/generate.py. When a tool, a function, a key or a command changes, these tests fail until
the generator is run again:

    python docs/_build/generate.py
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "docs" / "_build" / "generate.py"
FIX = "run `python docs/_build/generate.py` and commit what it writes"


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docs_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("docs_generator", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    if not GENERATOR.is_file():
        pytest.skip("docs/_build/generate.py is not in this copy of the repository")
    return _generator()


@pytest.fixture(scope="module")
def pages(generator: ModuleType) -> dict[Path, str]:
    return generator.render()


def test_every_generated_page_is_up_to_date(pages: dict[Path, str]) -> None:
    stale = sorted(
        str(path.relative_to(ROOT))
        for path, text in pages.items()
        if not path.is_file() or path.read_text(encoding="utf8") != text
    )
    assert not stale, f"out of date: {stale}; {FIX}"


def test_no_page_describes_a_tool_that_is_gone(generator: ModuleType) -> None:
    extra = [str(path.relative_to(ROOT)) for path in generator.stale_tool_pages()]
    assert not extra, f"pages for tools that no longer exist: {extra}; {FIX}"


def test_there_is_a_page_for_every_tool(pages: dict[Path, str]) -> None:
    from dantron_etl import REGISTRY

    tool_pages = {path.stem for path in pages if path.parent.name == "tools" and path.stem != "README"}
    assert tool_pages == {tool.name for tool in REGISTRY.tools()}


def test_the_function_table_lists_every_function(pages: dict[Path, str], generator: ModuleType) -> None:
    from dantron_etl.expr import ALLOWED_FUNCTIONS

    text = pages[generator.EXPRESSIONS]
    missing = sorted(name for name in ALLOWED_FUNCTIONS if f"| `{name}` |" not in text)
    assert not missing, f"docs/expressions.md has no row for {missing}"


def test_every_key_the_loader_accepts_is_described(pages: dict[Path, str], generator: ModuleType) -> None:
    from dantron_etl import loader

    text = pages[generator.PIPELINE_FILE]
    keys = {*loader._DOCUMENT_KEYS, *loader._STEP_KEYS, *loader._LANE_KEYS, *loader._CONTAINER_KEYS}
    missing = sorted(key for key in keys if f"| `{key}` |" not in text)
    assert not missing, f"docs/pipeline-file.md describes no key {missing}"


def test_the_readme_states_the_real_tool_count() -> None:
    from dantron_etl import REGISTRY

    count = len(list(REGISTRY.tools()))
    readme = (ROOT / "README.md").read_text(encoding="utf8")
    assert f"## The {count} tools" in readme, f"README.md should say 'The {count} tools'"
    missing = sorted(tool.name for tool in REGISTRY.tools() if f"`{tool.name}`" not in readme)
    assert not missing, f"README.md's tool table leaves out {missing}"
