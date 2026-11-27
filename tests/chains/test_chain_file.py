"""Reading a chain file on its own: its keys, its pipelines, their order, and its parameters.

Each refusal names the file, the entry and what to write instead. A parameter value given as text
(from a command line or a scheduler's form) is read as the same kind as its default.
"""

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from dantron_etl.chains.chain_file import (
    DEFAULT_MAX_PARALLEL,
    ChainError,
    Member,
    parse_chain,
    run_order,
    value_for,
)

PATH = Path("chains/month_end.yaml")


def parse(text: str) -> Any:
    return parse_chain(yaml.safe_load(text), "month_end", PATH)


def refused(text: str) -> str:
    with pytest.raises(ChainError) as caught:
        parse(text)
    return str(caught.value)


class TestTheFile:
    def test_a_small_chain_reads_back_as_written(self) -> None:
        chain = parse(
            "title: Month-end close\n"
            "pipelines:\n"
            "  - name: load\n"
            "  - name: clean\n"
            "    after: [load]\n"
            "    why: It cleans what load staged.\n"
        )
        assert chain.title == "Month-end close"
        assert chain.members == (Member("load", ()), Member("clean", ("load",), "It cleans what load staged."))
        assert dict(chain.parameters) == {}

    def test_the_title_is_the_name_when_left_out(self) -> None:
        assert parse("pipelines: [load]").title == "month_end"

    def test_a_bare_name_and_a_single_after_are_accepted(self) -> None:
        chain = parse("pipelines:\n  - load\n  - name: clean\n    after: load\n")
        assert chain.members == (Member("load", ()), Member("clean", ("load",)))

    def test_a_name_repeated_in_after_counts_once(self) -> None:
        assert parse("pipelines:\n  - load\n  - {name: clean, after: [load, load]}\n").member("clean").after == ("load",)

    def test_a_document_that_is_not_a_mapping_is_refused(self) -> None:
        assert "holds list, not a chain" in refused("- load\n- clean\n")

    def test_an_unknown_key_is_refused_with_the_allowed_ones(self) -> None:
        message = refused("pipelines: [load]\nsteps: []\n")
        assert "'steps'" in message and "title, parameters, max_parallel, pipelines" in message

    def test_an_unknown_key_in_an_entry_is_refused(self) -> None:
        message = refused("pipelines:\n  - name: load\n    inputs: [x]\n")
        assert "entry 1 under 'pipelines'" in message and "'inputs'" in message

    @pytest.mark.parametrize("text", ["title: x\n", "pipelines: []\n", "pipelines: load\n"])
    def test_a_chain_without_pipelines_is_refused(self, text: str) -> None:
        assert "List the pipelines the chain runs" in refused(text)

    def test_an_entry_without_a_name_is_refused(self) -> None:
        assert "Entry 2 under 'pipelines'" in refused("pipelines:\n  - load\n  - after: [load]\n")

    def test_after_that_is_not_a_list_of_names_is_refused(self) -> None:
        assert "'after' for clean" in refused("pipelines:\n  - load\n  - {name: clean, after: {load: 1}}\n")

    def test_a_pipeline_listed_twice_is_refused(self) -> None:
        assert "load is listed twice" in refused("pipelines: [load, clean, load]\n")

    def test_a_pipeline_after_itself_is_refused(self) -> None:
        assert "clean is set to run after itself" in refused("pipelines:\n  - {name: clean, after: [clean]}\n")

    def test_after_a_pipeline_outside_the_chain_is_refused(self) -> None:
        message = refused("pipelines:\n  - {name: clean, after: [load]}\n")
        assert "load is not in this chain" in message and "- name: load" in message

    def test_a_loop_is_refused_naming_every_pipeline_in_it(self) -> None:
        message = refused(
            "pipelines:\n  - load\n  - {name: a, after: [b, load]}\n  - {name: b, after: [a]}\n"
        )
        assert "in a loop" in message and "a, b" in message and "load," not in message


class TestMaxParallel:
    def test_the_default_is_two(self) -> None:
        assert DEFAULT_MAX_PARALLEL == 2
        assert parse("pipelines: [load]").max_parallel == 2

    def test_a_whole_number_is_kept(self) -> None:
        assert parse("max_parallel: 1\npipelines: [load]").max_parallel == 1
        assert parse("max_parallel: 6\npipelines: [load]").max_parallel == 6

    @pytest.mark.parametrize("value", ["0", "-1", "true", "'2'", "1.5"])
    def test_anything_but_a_whole_number_from_one_is_refused(self, value: str) -> None:
        assert "'max_parallel'" in refused(f"max_parallel: {value}\npipelines: [load]")


class TestRunOrder:
    def test_ties_keep_the_order_of_the_file(self) -> None:
        members = [Member("b", ()), Member("a", ()), Member("c", ("a",)), Member("d", ())]
        assert run_order(members, "x") == ("b", "a", "c", "d")

    def test_a_pipeline_comes_after_everything_it_runs_after_wherever_it_is_written(self) -> None:
        members = [Member("report", ("clean",)), Member("clean", ("load",)), Member("load", ())]
        assert run_order(members, "x") == ("load", "clean", "report")


class TestParameters:
    def test_defaults_are_kept_with_their_kinds(self) -> None:
        chain = parse(
            "parameters:\n  period: '2026-11'\n  cutoff: 2026-11-30\n  rate: 1.5\n  days: 3\n  strict: false\n"
            "  codes: [A, B]\npipelines: [load]\n"
        )
        assert dict(chain.parameters) == {
            "period": "2026-11",
            "cutoff": date(2026, 11, 30),
            "rate": 1.5,
            "days": 3,
            "strict": False,
            "codes": ["A", "B"],
        }

    @pytest.mark.parametrize("default", ["", "{a: 1}", "[[1]]"])
    def test_a_default_a_rule_cannot_hold_is_refused(self, default: str) -> None:
        assert "Give it a value" in refused(f"parameters:\n  period: {default}\npipelines: [load]")

    def test_parameters_that_are_not_a_mapping_are_refused(self) -> None:
        assert "'parameters'" in refused("parameters: [period]\npipelines: [load]")


class TestValueFor:
    @pytest.mark.parametrize(
        ("default", "given", "expected"),
        [
            ("2026-11", "2026-12", "2026-12"),
            ("2026-11", "true", "true"),  # text stays as typed, never read as YAML
            ("2026-11", "2026-12-31", "2026-12-31"),
            (date(2026, 11, 30), "2026-12-31", date(2026, 12, 31)),
            (date(2026, 11, 30), date(2026, 12, 31), date(2026, 12, 31)),
            (3, "4", 4),
            (3, 4, 4),
            (1.5, "2", 2.0),
            (1.5, 2, 2.0),
            (False, "true", True),
            (["A"], "[A, B]", ["A", "B"]),
        ],
    )
    def test_a_value_is_read_as_the_kind_of_its_default(self, default: Any, given: Any, expected: Any) -> None:
        value = value_for("p", default, given, "chain 'x'")
        assert value == expected and type(value) is type(expected)

    @pytest.mark.parametrize(
        ("default", "given"),
        [
            (date(2026, 11, 30), "2026-13-01"),
            (date(2026, 11, 30), "November"),
            (3, "3.5"),
            (3, "true"),
            (False, "maybe"),
            ("2026-11", 202611),
            (["A"], "A"),
        ],
    )
    def test_a_value_of_another_kind_is_refused_naming_both(self, default: Any, given: Any) -> None:
        with pytest.raises(ChainError) as caught:
            value_for("period", default, given, "chain 'month_end'")
        message = str(caught.value)
        assert "The parameter period in chain 'month_end'" in message and repr(given) in message
