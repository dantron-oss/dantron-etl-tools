"""The conformance check names the tool that broke, and reports every broken tool at once.

A slip inside a tool's own template, ``validate_config`` or ``validate_inputs`` comes out as a
``ConformanceError`` naming the tool, not as the raw Python error. ``check_registry`` lists every
tool that fails rather than stopping at the first. And a tool that changes its own configuration
cannot change the step it was given.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest

from dantron_etl import load_pipeline, run_pipeline
from dantron_etl.authoring import (
    ConfigField,
    ConformanceError,
    Ports,
    ToolHelp,
    ToolRegistry,
    WorkedExample,
    check_registry,
    check_tool,
    run_tool,
)
from dantron_etl.registry import REGISTRY

FRAME = pl.DataFrame({"n": [1]})


class _Tool:
    """A conforming pass-through tool; each subclass breaks one phase."""

    description = "Passes its rows on."
    category = "transform"
    input_ports: Sequence[str] = ("rows",)
    output_ports: Sequence[str] = ("rows",)

    def __init__(self, name: str) -> None:
        self.name = name

    def config_template(self) -> Mapping[str, ConfigField]:
        return {"by": ConfigField(type="array", description="Columns.", default=[], items=ConfigField(type="string", description="One column."))}

    def help(self) -> ToolHelp:
        return ToolHelp(
            "Passes its rows on.",
            examples=(WorkedExample("pass", {}, "Nothing changes.", Ports({"rows": FRAME}), Ports({"rows": FRAME})),),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return Ports({"rows": inputs["rows"]})


class _BadTemplate(_Tool):
    def config_template(self) -> Mapping[str, ConfigField]:
        raise ZeroDivisionError("template")


class _BadValidateConfig(_Tool):
    def validate_config(self, config: Mapping[str, Any]) -> None:
        raise KeyError("by")


class _BadValidateInputs(_Tool):
    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        inputs["nope"]


class _AppendsToItsConfig(_Tool):
    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        config["by"].append("extra")
        return Ports({"rows": inputs["rows"]})


@pytest.mark.parametrize(
    ("tool", "said"),
    [
        (_BadTemplate("bad_template"), "ZeroDivisionError"),
        (_BadValidateConfig("bad_validate_config"), "KeyError"),
        (_BadValidateInputs("bad_validate_inputs"), "nope"),
    ],
)
def test_a_slip_in_a_tools_own_code_names_the_tool(tool: _Tool, said: str) -> None:
    with pytest.raises(ConformanceError) as raised:
        check_tool(tool)
    assert repr(tool.name) in str(raised.value) and said in str(raised.value)


def test_check_registry_reports_every_broken_tool_at_once() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("fine"))
    registry.register(_BadTemplate("bad_template"))
    registry.register(_BadValidateConfig("bad_validate_config"))
    with pytest.raises(ConformanceError) as raised:
        check_registry(registry)
    message = str(raised.value)
    assert "2 tools" in message and "'bad_template'" in message and "'bad_validate_config'" in message
    assert "'fine'" not in message


def test_check_registry_with_one_broken_tool_reports_it_as_it_is() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("fine"))
    registry.register(_BadTemplate("bad_template"))
    with pytest.raises(ConformanceError) as raised:
        check_registry(registry)
    assert str(raised.value).startswith("tool 'bad_template'")


def test_a_conforming_registry_passes() -> None:
    registry = ToolRegistry()
    registry.register(_Tool("fine"))
    check_registry(registry)


class TestAToolCannotChangeTheConfigurationItWasGiven:
    def test_through_run_tool(self) -> None:
        config = {"by": ["n"]}
        run_tool(_AppendsToItsConfig("appends"), Ports({"rows": FRAME}), config)
        assert config == {"by": ["n"]}

    def test_through_a_run_the_steps_fingerprint_does_not_move(self) -> None:
        registry = ToolRegistry()
        for name in REGISTRY:
            registry.register(REGISTRY.get(name))
        registry.register(_AppendsToItsConfig("appends"))
        spec = load_pipeline(
            {
                "name": "appends",
                "steps": [
                    {"id": "s", "tool": "read_csv", "config": {"path": "a.csv"}},
                    {"id": "a", "tool": "appends", "config": {"by": ["n"]}, "inputs": {"rows": "s.rows"}},
                    {"id": "end", "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": "a.rows"}},
                ],
            },
            registry=registry,
        )
        before = spec.fingerprint("a")
        run_pipeline(spec, sources={"s.rows": FRAME})
        run_pipeline(spec, sources={"s.rows": FRAME})
        assert spec.step("a").config["by"] == ["n"]
        assert spec.fingerprint("a") == before
