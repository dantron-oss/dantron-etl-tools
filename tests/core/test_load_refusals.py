"""What the loader says about a file it cannot load.

Every refusal names the step and the key, never a bare Python error. A value of the wrong type, a
step id with a dot in it, and a tool whose own code breaks while its configuration is checked are
each refused by name. Every dead end is named at once, with the steps that would die with them,
and ``dead_ends`` answers the same question for a file before it is edited. A cycle is reported as
the one loop, not as every step downstream of it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest

from dantron_etl import ExecutionError, PipelineError, dead_ends, load_pipeline
from dantron_etl.authoring import ConfigField, Ports, ToolHelp, ToolRegistry, WorkedExample
from dantron_etl.registry import REGISTRY


def _read(step_id: str = "s") -> dict[str, Any]:
    return {"id": step_id, "tool": "read_csv", "config": {"path": "a.csv"}}


def _filter(step_id: str, source: str) -> dict[str, Any]:
    return {"id": step_id, "tool": "filter", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": source}}


def _check(step_id: str, source: str) -> dict[str, Any]:
    return {"id": step_id, "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": source}}


def _document(*steps: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"name": "refusals", "steps": list(steps), **extra}


class TestAValueOfTheWrongTypeIsRefusedByName:
    @pytest.mark.parametrize(
        ("change", "said"),
        [
            ({"id": 5}, "'id' must be text"),
            ({"tool": 5}, "'tool' must be text"),
            ({"config": ["a"]}, "'config' must be a mapping"),
            ({"inputs": ["s.rows"]}, "'inputs' must be a mapping"),
            ({"inputs": {"rows": 5}}, "'step.port'"),
            ({"lane": 3}, "'lane' must be text"),
            ({"why": ["a"]}, "'why' must be text"),
        ],
    )
    def test_a_step_key(self, change: dict[str, Any], said: str) -> None:
        step = {**_check("c", "s.rows"), **change}
        with pytest.raises(PipelineError) as raised:
            load_pipeline(_document(_read(), step))
        assert said in str(raised.value)

    def test_the_pipeline_name(self) -> None:
        with pytest.raises(PipelineError) as raised:
            load_pipeline({"name": 5, "steps": [_read(), _check("c", "s.rows")]})
        assert "'name' must be text" in str(raised.value)

    def test_a_lane_order(self) -> None:
        document = _document(_read(), _check("c", "s.rows"), lanes=[{"id": "a", "title": "A", "order": "first"}])
        with pytest.raises(PipelineError) as raised:
            load_pipeline(document)
        assert "lane 'a'" in str(raised.value) and "'order' must be a whole number" in str(raised.value)


class TestAStepIdCannotHoldADot:
    def test_a_dotted_id_is_refused_saying_why(self) -> None:
        with pytest.raises(PipelineError) as raised:
            load_pipeline(_document(_read("s.v2"), _check("c", "s.v2.rows")))
        assert "'s.v2'" in str(raised.value) and "dot" in str(raised.value)

    def test_a_blank_id_is_refused(self) -> None:
        with pytest.raises(PipelineError) as raised:
            load_pipeline(_document(_read(" "), _check("c", " .rows")))
        assert "blank" in str(raised.value)


class _BrokenValidate:
    """A tool whose validate_config raises a plain KeyError, as a slip in a tool's code would."""

    name = "broken_validate"
    description = "Breaks while checking its configuration."
    category = "assert"
    input_ports: Sequence[str] = ("rows",)
    output_ports: Sequence[str] = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {}

    def help(self) -> ToolHelp:
        frame = pl.DataFrame({"n": [1]})
        return ToolHelp("Breaks.", examples=(WorkedExample("x", {}, "x", Ports({"rows": frame}), Ports({"rows": frame})),))

    def validate_config(self, config: Mapping[str, Any]) -> None:
        raise KeyError("by")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return inputs


def test_a_slip_in_a_tools_own_validate_config_names_the_step_and_the_tool() -> None:
    registry = ToolRegistry()
    for name in REGISTRY:
        registry.register(REGISTRY.get(name))
    registry.register(_BrokenValidate())
    document = _document(_read(), {"id": "broken", "tool": "broken_validate", "inputs": {"rows": "s.rows"}})
    with pytest.raises(ExecutionError) as raised:
        load_pipeline(document, registry=registry)
    message = str(raised.value)
    assert "'broken'" in message and "broken_validate" in message and "KeyError" in message


class TestEveryDeadEndIsNamedAtOnce:
    def _flow(self) -> dict[str, Any]:
        # 'b' and 'c' are read by nothing; 'a' feeds only 'b'; 'kept' ends in a check and is fine.
        return _document(
            _read(), _filter("a", "s.rows"), _filter("b", "a.true"), _filter("c", "s.rows"),
            _filter("kept", "s.rows"), _check("end", "kept.true"),
        )

    def test_the_loader_names_every_dead_step_and_the_ones_behind_them(self) -> None:
        with pytest.raises(PipelineError) as raised:
            load_pipeline(self._flow())
        message = str(raised.value)
        assert "'b'" in message and "'c'" in message
        assert "'a'" in message
        assert "'kept'" not in message and "'end'" not in message

    def test_dead_ends_lists_the_whole_chain_in_rounds(self) -> None:
        assert dead_ends(self._flow()) == ("b", "c", "a")

    def test_a_flow_with_none_has_none(self) -> None:
        assert dead_ends(_document(_read(), _check("end", "s.rows"))) == ()

    def test_without_answers_what_would_die_if_an_input_went(self) -> None:
        document = _document(_read(), _filter("a", "s.rows"), _check("end", "a.true"))
        assert dead_ends(document) == ()
        assert dead_ends(document, without=["end.rows"]) == ("a", "s")

    def test_without_names_an_input_the_file_does_not_wire(self) -> None:
        with pytest.raises(PipelineError) as raised:
            dead_ends(_document(_read(), _check("end", "s.rows")), without=["end.rowz"])
        assert "end.rowz" in str(raised.value) and "end.rows" in str(raised.value)

    def test_a_single_dead_end_is_still_named_as_before(self) -> None:
        with pytest.raises(PipelineError) as raised:
            load_pipeline(_document(_read(), _check("end", "s.rows"), _filter("lone", "s.rows")))
        assert "step 'lone' (tool 'filter') is a dead end" in str(raised.value)


class TestACycleIsReportedAsTheOneLoop:
    def test_steps_downstream_of_the_loop_are_not_named_as_part_of_it(self) -> None:
        document = _document(
            _read(), _filter("a", "b.true"), _filter("b", "a.true"), _filter("after", "a.false"),
            _check("end", "after.true"),
        )
        with pytest.raises(PipelineError) as raised:
            load_pipeline(document)
        message = str(raised.value)
        loop_part = message.split(".")[0]
        assert "'after'" not in loop_part and "after" not in loop_part
        assert "a" in loop_part and "b" in loop_part
        assert "a.true -> b" in message and "b.true -> a" in message
        assert "2 more step(s)" in message

    def test_a_step_that_reads_itself(self) -> None:
        with pytest.raises(PipelineError) as raised:
            load_pipeline(_document(_read(), _filter("me", "me.true"), _check("end", "me.false")))
        message = str(raised.value)
        assert "me.true -> me" in message
        assert "end" not in message.split(".")[0]
