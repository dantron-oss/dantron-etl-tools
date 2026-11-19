"""The import surface: the package root and the tool author's module.

The core is split into small modules, and two modules gather the names people use: the package
root for running a flow, and ``authoring`` for writing a tool. These tests pin that each gathered
name is the very object its home module defines, so there is never a second copy to drift. They
also pin that the old import paths are gone, so nothing keeps a second way in.
"""

import ast
import importlib
from pathlib import Path

import polars as pl
import pytest

import dantron_etl
from dantron_etl import authoring, runner
from dantron_etl.loader import load_pipeline

TOOLS_DIR = Path(dantron_etl.__file__).parent / "tools"

# The modules that hold the tool contract, one job each.
CORE_MODULES = ("boundary", "config", "conformance", "contract", "errors", "example_world", "frame_checks",
                "ports", "registry", "status")

# Where each name the package root exports is defined.
ROOT_HOMES = {
    "dead_ends": "loader",
    "load_pipeline": "loader",
    "read_pipeline": "loader",
    "run_pipeline": "runner",
    "CachedFrame": "trace",
    "Pipeline": "spec",
    "PipelineTrace": "trace",
    "PortRef": "spec",
    "Step": "spec",
    "StepEvent": "trace",
    "FocusSpec": "focus",
    "RelatedKey": "focus",
    "slice_trace": "focus",
    "REGISTRY": "registry",
    "get_tool": "registry",
    "project_root": "paths",
    "ColumnCheckError": "trace",
    "ConfigError": "errors",
    "ConformanceError": "errors",
    "DataError": "errors",
    "ExecutionError": "errors",
    "ExpressionError": "expr",
    "PipelineError": "errors",
    "PortError": "errors",
    "RegistryError": "errors",
    "RunError": "errors",
    "StepFailedError": "trace",
    "ToolError": "errors",
}


def _home(module: str) -> object:
    return importlib.import_module(f"dantron_etl.{module}")


def test_every_root_name_is_its_home_modules_object() -> None:
    """Each name in the root's ``__all__`` is the object its home module defines, not a copy."""
    exported = set(dantron_etl.__all__) - {"__version__", "tools", "COLUMN_TYPES"}
    assert exported == set(ROOT_HOMES), "a root export was added or removed without naming its home here"
    for name, module in ROOT_HOMES.items():
        assert getattr(dantron_etl, name) is getattr(_home(module), name), name


def test_column_types_at_the_root_are_the_declared_types() -> None:
    """``COLUMN_TYPES`` is the root's name for ``column_types.DECLARED_TYPES``, the same tuple."""
    from dantron_etl.column_types import DECLARED_TYPES

    assert dantron_etl.COLUMN_TYPES is DECLARED_TYPES


def test_get_tool_answers_as_the_registry_does() -> None:
    """``get_tool`` is a short form of ``REGISTRY.get``: the same tool, and the same refusal."""
    assert dantron_etl.get_tool("filter") is dantron_etl.REGISTRY.get("filter")
    with pytest.raises(dantron_etl.RegistryError, match="no tool named 'nope' is registered"):
        dantron_etl.get_tool("nope")


def test_every_authoring_name_is_defined_in_one_core_module() -> None:
    """Each name ``authoring`` gathers is the object of exactly one smaller core module."""
    for name in authoring.__all__:
        owners = [module for module in CORE_MODULES if getattr(_home(module), name, None) is getattr(authoring, name)]
        assert owners, f"authoring.{name} is not any core module's object"


def test_every_tool_imports_the_core_only_through_authoring() -> None:
    """A tool file reads everything it needs from ``authoring``, never from the core's inner modules."""
    inner = {*CORE_MODULES, "tool", "spec", "loader", "rules", "runner", "trace", "walk", "pipeline"}
    offenders = []
    for path in sorted(TOOLS_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module in inner:
                offenders.append(f"{path.name}: from ..{node.module}")
    assert not offenders


@pytest.mark.parametrize("retired", ["tool", "pipeline"])
def test_the_retired_import_modules_are_gone(retired: str) -> None:
    """``dantron_etl.tool`` and ``dantron_etl.pipeline`` were old paths; the package root and
    ``authoring`` replace them, and nothing imports them any more."""
    with pytest.raises(ModuleNotFoundError):
        _home(retired)


def test_the_runner_exports_only_what_it_defines() -> None:
    """The trace names live in ``trace`` and are exported from the package root, not again here."""
    assert sorted(runner.__all__) == ["logging_sink", "run_pipeline"]


def test_a_wrapper_put_on_runner_run_tool_sees_every_step_that_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_pipeline`` calls the boundary through ``runner.run_tool``, so replacing it there is seen.

    A figure check outside this library captures each step's output by swapping that one attribute
    while a run is in progress. If the runner called the boundary through another module's name,
    the swap would see nothing and the check would quietly compare nothing.
    """
    real = runner.run_tool
    seen: list[str] = []

    def watching(tool, inputs, config):
        seen.append(tool.name)
        return real(tool, inputs, config)

    monkeypatch.setattr(runner, "run_tool", watching)
    spec = load_pipeline(
        {
            "name": "watched",
            "steps": [
                {"id": "src", "tool": "read_csv", "config": {"path": "unused.csv"}},
                {"id": "big", "tool": "filter", "config": {"predicate": "[n] > 1"}, "inputs": {"rows": "src.rows"}},
                {"id": "out", "tool": "assert_rows", "config": {"predicate": "[n] > 1"}, "inputs": {"rows": "big.true"}},
                {"id": "rest", "tool": "assert_rows", "config": {"predicate": "[n] <= 1"},
                 "inputs": {"rows": "big.false"}},
            ],
        }
    )
    dantron_etl.run_pipeline(spec, sources={"src.rows": pl.DataFrame({"n": [1, 2, 3]})})

    assert seen == ["filter", "assert_rows", "assert_rows"]
