"""Running a chain: in order, side by side up to max_parallel, and stopping only what a failure feeds.

Most tests run the pipelines in threads, which is quick. One runs them the default way, each in a
process of its own, to prove that path works end to end.
"""

import json
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import pytest
import yaml

from dantron_etl.chains import ChainError, chain_runs, load_chain, run_chain, run_member
from dantron_etl.viewer.runs import RUN_FILE, last_runs


def threads(size: int) -> Executor:
    return ThreadPoolExecutor(size)


def run(chain_project: Any, name: str = "month_end", **options: Any) -> Any:
    plan = load_chain(chain_project.root, name, options.get("parameters"))
    return run_chain(chain_project.root, name, confirm_digest=plan.digest, executor=options.pop("executor", threads),
                     echo_steps=False, **options)


def outcomes(result: Any) -> dict[str, str]:
    return {member.pipeline: member.outcome for member in result.members}


def break_pipeline(chain_project: Any, name: str, reads: str, writes: str) -> None:
    """Make ``name`` fail at its second step, after reading, before writing."""
    steps = [
        {"id": "read", "tool": "read_csv", "config": {"path": reads}},
        {"id": "pick", "tool": "select_columns", "config": {"columns": ["no_such_column"]}, "inputs": {"rows": "read.rows"}},
        {"id": "write", "tool": "write_csv", "config": {"path": writes, "overwrite": True}, "inputs": {"rows": "pick.rows"}},
    ]
    chain_project.pipeline(name, yaml.safe_dump({"name": name, "steps": steps}, sort_keys=False))


class TestAWholeRun:
    def test_every_pipeline_runs_and_writes_with_the_values_given(self, chain_project: Any) -> None:
        chain_project.month_end()
        result = run(chain_project, parameters={"period": "2026-12"})
        assert result.outcome == "finished"
        assert set(outcomes(result).values()) == {"finished"}
        report = chain_project.root / "reports" / "revenue_2026-12.csv"
        assert report.read_text(encoding="utf8").count("\n") == 1 + 3 + 2
        assert not (chain_project.root / "reports" / "revenue_2026-11.csv").exists()

    def test_each_pipelines_run_is_saved_and_says_which_chain_and_values(self, chain_project: Any) -> None:
        chain_project.month_end()
        run(chain_project, parameters={"period": "2026-12"})
        good, failed = last_runs(chain_project.project().runs_folder, "revenue")
        assert failed is None and good is not None and good.kind == "real"
        saved = json.loads((good.directory / RUN_FILE).read_text(encoding="utf8"))
        assert saved["chain"] == {"name": "month_end", "values": {"period": "2026-12"}}
        loaded = json.loads((last_runs(chain_project.project().runs_folder, "load_ledger")[0].directory / RUN_FILE).read_text())
        assert loaded["chain"] == {"name": "month_end", "values": {}}

    def test_the_chains_run_is_saved_and_reads_back(self, chain_project: Any) -> None:
        chain_project.month_end()
        first = run(chain_project)
        second = run(chain_project, parameters={"period": "2026-12"})
        saved = chain_runs(chain_project.project(), "month_end")
        assert [one.saved for one in saved] == [second.saved, first.saved]
        assert saved[0].values == {"period": "2026-12"}
        assert saved[0].members == second.members
        assert saved[0].covered == second.covered

    def test_a_pipeline_starts_only_after_everything_it_runs_after_has_finished(self, chain_project: Any) -> None:
        chain_project.month_end()
        lines: list[str] = []
        run(chain_project, report=lines.append)
        at = {line: n for n, line in enumerate(lines)}
        assert at["Starting revenue."] > at["Finished: clean_ledger, 2 of 2 steps."]
        assert at["Starting revenue."] > at["Finished: clean_customers, 2 of 2 steps."]
        assert at["Starting clean_ledger."] > at["Finished: load_ledger, 2 of 2 steps."]


class TestAFailure:
    @pytest.mark.parametrize(
        ("broken", "reads", "writes", "other"),
        [
            ("clean_ledger", "stage/ledger.csv", "clean/ledger_${rules.period}.csv", "clean_customers"),
            ("clean_customers", "stage/customers.csv", "clean/customers.csv", "clean_ledger"),
        ],
    )
    def test_what_waits_for_a_failure_is_not_started_and_the_other_branch_finishes(
        self, chain_project: Any, broken: str, reads: str, writes: str, other: str
    ) -> None:
        chain_project.month_end()
        break_pipeline(chain_project, broken, reads, writes)
        result = run(chain_project)
        assert result.outcome == "failed"
        assert outcomes(result)[broken] == "failed"
        assert result.member(broken).failed_step == "pick"
        assert outcomes(result)[other] == "finished"
        assert outcomes(result)["revenue"] == "not started"
        assert result.member("revenue").message == f"not started, because {broken} did not finish"
        assert not list((chain_project.root / "reports").glob("*.csv"))

    def test_not_started_reaches_every_pipeline_further_on(self, chain_project: Any) -> None:
        chain_project.month_end()
        break_pipeline(chain_project, "load_customers", "inbox/customers.csv", "stage/customers.csv")
        result = run(chain_project)
        assert outcomes(result) == {
            "load_ledger": "finished",
            "load_customers": "failed",
            "clean_ledger": "finished",
            "clean_customers": "not started",
            "revenue": "not started",
        }
        assert result.member("revenue").message == "not started, because clean_customers did not finish"

    def test_a_process_the_system_stops_is_a_failure_that_names_memory(self, chain_project: Any) -> None:
        chain_project.month_end()

        class Stopping(ThreadPoolExecutor):
            def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
                if args[2] == "clean_ledger":
                    future: Future[Any] = Future()
                    future.set_exception(BrokenProcessPool("killed"))
                    return future
                return super().submit(fn, *args, **kwargs)

        result = run(chain_project, executor=lambda size: Stopping(size))
        assert outcomes(result)["clean_ledger"] == "failed"
        assert "ran out of memory" in (result.member("clean_ledger").message or "")
        assert "max_parallel" in (result.member("clean_ledger").message or "")
        assert outcomes(result)["revenue"] == "not started"
        assert outcomes(result)["clean_customers"] == "finished"

    def test_an_interrupted_run_is_saved_as_stopped_and_the_interruption_goes_on(self, chain_project: Any) -> None:
        chain_project.month_end()

        def interrupt(line: str) -> None:
            if line == "Starting revenue.":
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run(chain_project, report=interrupt)
        saved = chain_runs(chain_project.project(), "month_end")
        assert len(saved) == 1 and saved[0].outcome == "stopped"
        assert {m.pipeline: m.outcome for m in saved[0].members}["revenue"] == "not started"


class TestConfirmation:
    def test_a_digest_that_does_not_match_is_refused_before_anything_runs(self, chain_project: Any) -> None:
        chain_project.month_end()
        with pytest.raises(ChainError, match="changed since it was shown"):
            run_chain(chain_project.root, "month_end", confirm_digest="not the digest", executor=threads)
        assert not (chain_project.root / "stage").exists() or not list((chain_project.root / "stage").iterdir())
        assert chain_runs(chain_project.project(), "month_end") == ()

    def test_a_pipeline_edited_after_the_plan_was_shown_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        shown = load_chain(chain_project.root, "month_end").digest
        chain_project.copy("clean_customers", ["stage/customers.csv"], "clean/elsewhere.csv")
        chain_project.copy("revenue", ["clean/ledger_${rules.period}.csv", "clean/elsewhere.csv"], "reports/r.csv")
        with pytest.raises(ChainError, match="changed since it was shown"):
            run_chain(chain_project.root, "month_end", confirm_digest=shown, executor=threads)

    def test_a_value_other_than_the_one_shown_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        shown = load_chain(chain_project.root, "month_end").digest
        with pytest.raises(ChainError, match="changed since it was shown"):
            run_chain(chain_project.root, "month_end", parameters={"period": "2026-12"}, confirm_digest=shown, executor=threads)


class TestPartOfAChain:
    def test_from_runs_that_pipeline_and_what_follows_and_marks_the_rest(self, chain_project: Any) -> None:
        chain_project.month_end()
        run(chain_project)
        result = run(chain_project, start_from="clean_ledger")
        assert result.covered == ("clean_ledger", "revenue")
        assert outcomes(result) == {
            "load_ledger": "not in this run",
            "load_customers": "not in this run",
            "clean_ledger": "finished",
            "clean_customers": "not in this run",
            "revenue": "finished",
        }

    def test_until_runs_that_pipeline_and_what_comes_before(self, chain_project: Any) -> None:
        chain_project.month_end()
        result = run(chain_project, until="clean_customers")
        assert result.covered == ("load_customers", "clean_customers")
        assert outcomes(result)["revenue"] == "not in this run"


class TestMaxParallel:
    @pytest.mark.parametrize(("setting", "expected"), [("", 2), ("max_parallel: 1\n", 1), ("max_parallel: 3\n", 3)])
    def test_no_more_than_max_parallel_pipelines_run_at_once(self, chain_project: Any, setting: str, expected: int) -> None:
        for n in range(5):
            chain_project.source(f"in_{n}.csv", rows=20_000)
            chain_project.copy(f"p{n}", [f"in_{n}.csv"], f"out_{n}.csv")
        chain_project.chain("wide", setting + "pipelines: [p0, p1, p2, p3, p4]\n")
        sizes: list[int] = []
        lock = threading.Lock()
        now = {"running": 0, "most": 0}

        class Counting(ThreadPoolExecutor):
            def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
                with lock:
                    now["running"] += 1
                    now["most"] = max(now["most"], now["running"])
                future = super().submit(fn, *args, **kwargs)

                def ended(_: Future[Any]) -> None:
                    with lock:
                        now["running"] -= 1

                future.add_done_callback(ended)
                return future

        def make(size: int) -> Executor:
            sizes.append(size)
            return Counting(max_workers=8)  # more workers than allowed: the chain itself must hold back

        result = run(chain_project, "wide", executor=make)
        assert result.outcome == "finished"
        assert sizes == [expected]
        assert now["most"] == expected


@pytest.mark.slow
def test_by_default_each_pipeline_runs_in_a_process_of_its_own(chain_project: Any) -> None:
    chain_project.month_end()
    plan = load_chain(chain_project.root, "month_end")
    result = run_chain(chain_project.root, "month_end", confirm_digest=plan.digest, echo_steps=False)
    assert result.outcome == "finished", result.members
    assert (chain_project.root / "reports" / "revenue_2026-11.csv").is_file()


class TestRunMember:
    def test_one_pipeline_runs_with_the_chains_values(self, chain_project: Any) -> None:
        chain_project.month_end()
        run(chain_project, until=["clean_ledger", "clean_customers"], parameters={"period": "2026-12"})
        record = run_member(chain_project.root, "month_end", "revenue", {"period": "2026-12"})
        assert record.outcome == "finished"
        assert (chain_project.root / "reports" / "revenue_2026-12.csv").is_file()
        saved = json.loads((record.directory / RUN_FILE).read_text(encoding="utf8"))
        assert saved["chain"] == {"name": "month_end", "values": {"period": "2026-12"}}

    def test_its_steps_go_to_the_logger_given(self, chain_project: Any) -> None:
        chain_project.month_end()

        class Logger:
            def __init__(self) -> None:
                self.lines: list[str] = []

            def info(self, message: str) -> None:
                self.lines.append(message)

            warning = error = info

        logger = Logger()
        run_member(chain_project.root, "month_end", "load_ledger", logger=logger)
        assert len(logger.lines) == 2 and "read_0" in logger.lines[0]

    def test_a_pipeline_outside_the_chain_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.copy("other", ["inbox/ledger.csv"], "other.csv")
        with pytest.raises(ChainError, match="'other' is not a pipeline of chain 'month_end'"):
            run_member(chain_project.root, "month_end", "other")

    def test_a_pipeline_the_project_does_not_have_is_refused_by_name(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.chain("c", "pipelines: [load_ledger, ghost]\n")
        with pytest.raises(ChainError, match="runs 'ghost', which is not a pipeline in the project"):
            run_member(chain_project.root, "c", "ghost")

    def test_a_value_for_an_unknown_parameter_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        with pytest.raises(ChainError, match="'perod'"):
            run_member(chain_project.root, "month_end", "revenue", {"perod": "2026-12"})

    def test_a_failure_is_returned_not_raised(self, chain_project: Any) -> None:
        chain_project.month_end()
        record = run_member(chain_project.root, "month_end", "revenue")
        assert record.outcome == "failed"
