"""The command, driven the way a person would: run a chain, check one, and have one suggested.

Also how a project finds its chains, and how a pipeline is handed a chain's values.
"""

from pathlib import Path
from typing import Any

import pytest

from dantron_etl.chains import load_chain
from dantron_etl.chains.chain_file import parse_chain
from dantron_etl.cli import COMMAND, main
from dantron_etl.viewer.project import ProjectError, load_project, pipeline_rules


@pytest.fixture
def month_end(chain_project: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    chain_project.month_end()
    monkeypatch.chdir(chain_project.root)
    return chain_project


def printed(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


class TestRun:
    def test_without_write_it_prints_the_plan_and_runs_nothing(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["run", "month_end", "--set", "period=2026-12"]) == 0
        text = printed(capsys)
        assert "Chain month_end: Month-end close" in text
        assert "period: 2026-12 (given for this run), used by clean_ledger, revenue" in text
        assert "Runs up to 2 pipelines at a time" in text
        assert "clean_ledger -> revenue: clean/ledger_2026-12.csv" in text
        assert "inbox/ledger.csv (load_ledger)" in text
        assert "reports/revenue_2026-12.csv (revenue)" in text
        assert f"Nothing was run. To run it for real: {COMMAND} run month_end --write" in text
        assert not (month_end.root / "stage").exists() or not list((month_end.root / "stage").iterdir())

    def test_a_replaced_value_is_shown_beside_the_pipeline_it_replaces(
        self, month_end: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        month_end.settings("pipelines:\n  revenue:\n    rules:\n      period: '2026-01'\n")
        assert main(["run", "month_end"]) == 0
        assert "replaces revenue's own value 2026-01" in printed(capsys)

    def test_write_lists_every_write_asks_for_the_name_and_runs(
        self, month_end: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[str] = []
        monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "month_end")
        assert main(["run", "month_end", "--write"]) == 0
        text = printed(capsys)
        assert asked == ["Type the chain's name (month_end) to go ahead: "]
        assert "Running it will write:" in text and "revenue: " in text and "revenue_2026-11.csv" in text
        assert "Chain month_end finished: 5 finished" in text
        assert (month_end.root / "reports" / "revenue_2026-11.csv").is_file()

    def test_a_wrong_name_runs_nothing(
        self, month_end: Any, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda prompt: "month")
        assert main(["run", "month_end", "--write"]) == 1
        assert "Nothing was run: you typed 'month'" in printed(capsys)
        assert not (month_end.root / "reports").exists() or not list((month_end.root / "reports").iterdir())

    def test_a_failed_chain_exits_with_one_and_names_what_did_not_run(
        self, month_end: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (month_end.root / "inbox" / "customers.csv").unlink()
        assert main(["run", "month_end", "--write", "--yes", "--set", "period=2026-12"]) == 1
        text = printed(capsys)
        assert "Chain month_end failed:" in text
        assert "1 failed (load_customers)" in text and "2 not started (clean_customers, revenue)" in text

    def test_from_runs_part_of_the_chain(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["run", "month_end", "--write", "--yes", "--until", "clean_ledger"]) == 0
        text = printed(capsys)
        assert "load_customers   first   (not in this run)" in text
        assert "2 finished (load_ledger, clean_ledger)" in text

    def test_chain_options_on_a_pipeline_are_refused(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["run", "load_ledger", "--set", "period=2026-12", "--from", "x"]) == 1
        assert "--set and --from are for a chain, and load_ledger is a pipeline." in printed(capsys)

    def test_a_bad_setting_is_refused(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["run", "month_end", "--set", "period"]) == 1
        assert "is not NAME=VALUE" in printed(capsys)

    def test_an_unknown_name_lists_the_chains_too(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["run", "month_ned"]) == 1
        assert "Its chains: month_end." in printed(capsys)


class TestCheck:
    def test_a_good_chain_file_is_ok(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["check", "chains/month_end.yaml"]) == 0
        assert "ok, a chain of 5 pipelines" in printed(capsys)

    def test_a_chain_out_of_order_is_refused_with_the_reason(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        text = (month_end.root / "chains" / "month_end.yaml").read_text(encoding="utf8")
        month_end.chain("month_end", text.replace("after: [clean_ledger, clean_customers]", "after: [clean_ledger]"))
        assert main(["check", "chains/month_end.yaml"]) == 1
        assert "Add clean_customers to revenue's 'after'" in printed(capsys)


class TestSuggest:
    def test_it_writes_a_chain_that_loads_and_has_the_same_hand_offs(
        self, month_end: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["chain", "suggest", "auto", "--set", "period=2026-11"]) == 0
        written = month_end.root / "chains" / "auto.yaml"
        assert f"Wrote {written}" in printed(capsys)
        suggested = load_chain(month_end.root, "auto")
        by_hand = load_chain(month_end.root, "month_end")
        assert {(h.writer, h.reader) for h in suggested.handoffs} == {(h.writer, h.reader) for h in by_hand.handoffs}
        assert {name: suggested.after(name) for name in suggested.order} == {
            name: by_hand.after(name) for name in by_hand.order
        }
        assert dict(suggested.values) == {"period": "2026-11"}
        assert "# from clean_ledger: clean/ledger_2026-11.csv" in written.read_text(encoding="utf8")

    def test_a_date_given_is_kept_as_a_date(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["chain", "suggest", "auto", "--set", "period=2026-11-30", "--print"]) == 0
        chain = parse_chain(__import__("yaml").safe_load(printed(capsys)), "auto", Path("auto.yaml"))
        assert str(type(chain.parameters["period"]).__name__) == "date"

    def test_only_the_pipelines_named_are_put_in_it(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["chain", "suggest", "loads", "load_ledger", "clean_customers", "--print"]) == 0
        text = printed(capsys)
        assert "- name: load_ledger" in text and "- name: clean_customers" in text and "revenue" not in text
        assert "# from outside the chain: stage/customers.csv" in text

    def test_rules_shared_by_several_pipelines_are_pointed_out(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        month_end.settings(
            "pipelines:\n  clean_ledger:\n    rules: {period: '2026-11'}\n  revenue:\n    rules: {period: '2026-11'}\n"
        )
        assert main(["chain", "suggest", "auto", "--print"]) == 0
        assert "#   period: used by clean_ledger, revenue" in printed(capsys)

    def test_an_existing_file_is_never_overwritten(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        before = (month_end.root / "chains" / "month_end.yaml").read_text(encoding="utf8")
        assert main(["chain", "suggest", "month_end", "--set", "period=x"]) == 1
        assert "is already there" in printed(capsys)
        assert (month_end.root / "chains" / "month_end.yaml").read_text(encoding="utf8") == before

    def test_a_loop_in_the_files_is_refused(self, chain_project: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        chain_project.source("a.csv")
        chain_project.source("b.csv")
        chain_project.copy("one", ["a.csv"], "b.csv")
        chain_project.copy("two", ["b.csv"], "a.csv")
        monkeypatch.chdir(chain_project.root)
        assert main(["chain", "suggest", "loop"]) == 1
        assert "in a loop" in printed(capsys)

    def test_a_pipeline_needing_a_value_nobody_gave_is_refused(self, month_end: Any, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["chain", "suggest", "auto"]) == 1
        assert "clean_ledger does not load" in printed(capsys)


class TestTheProject:
    def test_chains_are_found_in_the_chains_folder_with_their_titles(self, month_end: Any) -> None:
        project = load_project(month_end.root)
        assert list(project.chains) == ["month_end"]
        assert project.chains["month_end"].title == "Month-end close"
        assert project.problems == ()

    def test_a_project_without_a_chains_folder_has_no_chains_and_no_problem(self, chain_project: Any) -> None:
        (chain_project.root / "chains").rmdir()
        project = load_project(chain_project.root)
        assert dict(project.chains) == {} and project.problems == ()

    def test_a_chain_folder_the_settings_name_must_exist(self, chain_project: Any) -> None:
        chain_project.settings("chain_folders: [flows]\n")
        assert load_project(chain_project.root).problems == ("The chain folder flows does not exist.",)

    def test_chain_folders_are_read_from_the_settings(self, chain_project: Any) -> None:
        chain_project.month_end()
        (chain_project.root / "flows").mkdir()
        (chain_project.root / "chains" / "month_end.yaml").rename(chain_project.root / "flows" / "month_end.yaml")
        chain_project.settings("chain_folders: [flows]\n")
        assert list(load_project(chain_project.root).chains) == ["month_end"]

    def test_a_chain_named_like_a_pipeline_is_skipped_and_said(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.chain("Revenue", "pipelines: [revenue]\n")
        project = load_project(chain_project.root)
        assert "Revenue" not in project.chains
        assert any("a pipeline is already called 'revenue'" in problem for problem in project.problems)

    def test_a_chain_with_a_reserved_name_is_skipped(self, chain_project: Any) -> None:
        chain_project.chain("api", "pipelines: [x]\n")
        assert "api" not in load_project(chain_project.root).chains

    def test_a_chain_file_that_is_not_yaml_is_kept_with_its_problem(self, chain_project: Any) -> None:
        chain_project.chain("bad", "pipelines: [\n")
        entry = load_project(chain_project.root).chains["bad"]
        assert entry.problem and "could not be read as YAML" in entry.problem


class TestValuesGivenToAPipeline:
    def test_a_given_value_replaces_the_pipelines_own(self, month_end: Any) -> None:
        month_end.settings("pipelines:\n  revenue:\n    rules: {period: '2026-01', other: 1}\n")
        project = load_project(month_end.root)
        assert pipeline_rules(project, project.pipelines["revenue"], {"period": "2026-12"}) == {
            "period": "2026-12",
            "other": 1,
        }

    def test_a_rule_the_chain_gives_needs_no_settings(self, month_end: Any) -> None:
        project = load_project(month_end.root)
        assert pipeline_rules(project, project.pipelines["revenue"], {"period": "2026-12"}) == {"period": "2026-12"}

    def test_without_it_the_refusal_names_the_rule(self, month_end: Any) -> None:
        project = load_project(month_end.root)
        with pytest.raises(ProjectError, match=r"uses \$\{rules.period\} and no rules are set"):
            pipeline_rules(project, project.pipelines["revenue"])
