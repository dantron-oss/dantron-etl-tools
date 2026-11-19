"""Looking things up on a loaded flow and on a finished run.

A port can be named the way a caller wrote it anywhere else: a ``PortRef``, ``"step.port"``, or the
step alone when it has one output. A refusal about a name lists the closest names and at most ten
of the rest, so a thousand-step flow does not print a page. Lookups do not scan every step, so they
stay quick on a large flow. A pipeline built in code without a registry uses the shipped one.
"""

import time
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from dantron_etl import Pipeline, PortRef, RunError, Step, load_pipeline, run_pipeline
from dantron_etl.errors import PipelineError
from dantron_etl.loader import read_pipeline, yaml_loader

FIXTURE = Path(__file__).resolve().parents[2] / "pipelines" / "fixture.yaml"


def _chain(length: int) -> dict[str, Any]:
    """A source and ``length`` filters in a row, each split's false side ending in a check."""
    steps: list[dict[str, Any]] = [{"id": "s0", "tool": "read_csv", "config": {"path": "x.csv"}}]
    for number in range(1, length + 1):
        steps.append(
            {
                "id": f"s{number}",
                "tool": "filter",
                "config": {"predicate": "[n] > 0"},
                "inputs": {"rows": f"s{number - 1}.{'rows' if number == 1 else 'true'}"},
            }
        )
        steps.append(
            {
                "id": f"c{number}",
                "tool": "assert_rows",
                "config": {"predicate": "[n] <= 0", "allow_empty": True},
                "inputs": {"rows": f"s{number}.false"},
            }
        )
    steps.append(
        {"id": "end", "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": f"s{length}.true"}}
    )
    return {"name": "chain", "steps": steps}


def _trace():
    spec = load_pipeline(_chain(2))
    return run_pipeline(spec, sources={"s0.rows": pl.DataFrame({"n": [1, -1, 2]})})


class TestTraceFrameTakesEverySpelling:
    def test_a_portref_text_and_a_lone_step_name_find_the_same_frame(self) -> None:
        trace = _trace()
        by_ref = trace.frame(PortRef("s0", "rows"))
        assert trace.frame("s0.rows").equals(by_ref)
        assert trace.frame("s0").equals(by_ref)

    def test_a_lone_step_name_with_two_ports_is_refused_naming_both(self) -> None:
        with pytest.raises(RunError) as raised:
            _trace().frame("s1")
        assert "s1.true" in str(raised.value) and "s1.false" in str(raised.value)

    def test_a_port_that_is_not_there_suggests_the_closest(self) -> None:
        with pytest.raises(RunError) as raised:
            _trace().frame("s1.ture")
        assert "s1.true" in str(raised.value)

    def test_event_or_none_answers_for_a_step_that_did_not_run(self) -> None:
        trace = _trace()
        assert trace.event_or_none("s1") is trace.event("s1")
        assert trace.event_or_none("nowhere") is None


class TestARefusalListsAFewNamesNotAll:
    def test_a_typo_in_a_large_flow_gives_a_short_message_with_a_suggestion(self) -> None:
        spec = load_pipeline(_chain(300))
        with pytest.raises(PipelineError) as raised:
            spec.step("s15O")
        message = str(raised.value)
        assert "did you mean 's15'" in message
        assert len(message) < 400, message

    def test_a_short_flow_still_lists_every_step(self) -> None:
        spec = load_pipeline(_chain(1))
        with pytest.raises(PipelineError) as raised:
            spec.step("zzz")
        for step in spec.steps:
            assert step.id in str(raised.value)


class TestAPipelineBuiltInCode:
    def test_one_built_without_a_registry_still_answers_questions_about_its_columns(self) -> None:
        spec = Pipeline(
            name="in code",
            steps=(
                Step("s", "read_csv", {"path": "x.csv"}, {}),
                Step("f", "filter", {"predicate": "[n] > 0"}, {"rows": PortRef("s", "rows")}),
            ),
        )
        assert spec.column_references() == {"f": {"predicate": frozenset({"n"})}}
        assert spec.consumers(PortRef("s", "rows")) == (PortRef("f", "rows"),)


class TestLookupsStayQuickOnALargeFlow:
    def test_readers_of_every_port_on_two_thousand_steps_take_well_under_a_second(self) -> None:
        spec = load_pipeline(_chain(1000))
        started = time.perf_counter()
        for step in spec.steps:
            for port in ("rows", "true", "false"):
                spec.consumers(PortRef(step.id, port))
            spec.step(step.id)
        assert time.perf_counter() - started < 0.5


class TestTheFastYamlReaderReadsTheSameDocument:
    def test_the_fixture_reads_identically_with_either_parser(self) -> None:
        text = FIXTURE.read_text(encoding="utf8")
        assert yaml.load(text, Loader=yaml_loader()) == yaml.load(text, Loader=yaml.SafeLoader)

    def test_read_pipeline_still_reads_the_fixture(self) -> None:
        assert read_pipeline(FIXTURE, rules=yaml.safe_load((FIXTURE.parent / "fixture" / "settings.yaml").read_text())["rules"]).name
