"""Working a chain out before it runs: its pipelines loaded with its values, its order checked.

The rule that matters most: a pipeline that reads a file or table another pipeline in the chain
writes must run after it. Otherwise the two may run side by side and the reader takes what an
earlier run left there, and nothing would say so. Each such disagreement refuses the chain by name.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from dantron_etl.chains import ChainError, load_chain
from dantron_etl.chains.handoffs import READER_CATEGORY, WRITER_CATEGORY, Target, _targets
from dantron_etl.registry import REGISTRY


def chain_text(*members: str, parameters: str = "") -> str:
    """A chain file: each member is 'name' or 'name: after1, after2'."""
    lines = [parameters] if parameters else []
    lines.append("pipelines:")
    for member in members:
        name, _, after = member.partition(":")
        lines.append(f"  - name: {name.strip()}")
        if after.strip():
            lines.append(f"    after: [{after.strip()}]")
    return "\n".join(lines) + "\n"


def refused(root: Path, name: str = "c", parameters: dict[str, Any] | None = None) -> str:
    with pytest.raises(ChainError) as caught:
        load_chain(root, name, parameters)
    return str(caught.value)


class TestTheWorkedExample:
    def test_the_order_hand_offs_and_outside_files_are_worked_out(self, chain_project: Any) -> None:
        chain_project.month_end()
        plan = load_chain(chain_project.root, "month_end")
        assert plan.order == ("load_ledger", "load_customers", "clean_ledger", "clean_customers", "revenue")
        assert [(h.writer, h.reader, h.target.shown) for h in plan.handoffs] == [
            ("load_ledger", "clean_ledger", "stage/ledger.csv"),
            ("load_customers", "clean_customers", "stage/customers.csv"),
            ("clean_ledger", "revenue", "clean/ledger_2026-11.csv"),
            ("clean_customers", "revenue", "clean/customers.csv"),
        ]
        assert [(t.shown, r) for t, r in plan.outside] == [
            ("inbox/ledger.csv", ("load_ledger",)),
            ("inbox/customers.csv", ("load_customers",)),
        ]
        assert [(t.shown, w) for t, w in plan.results] == [("reports/revenue_2026-11.csv", "revenue")]
        assert plan.max_parallel == 2

    def test_a_parameter_given_for_the_run_is_used_in_every_path_that_names_it(self, chain_project: Any) -> None:
        chain_project.month_end()
        plan = load_chain(chain_project.root, "month_end", {"period": "2026-12"})
        assert dict(plan.values) == {"period": "2026-12"}
        assert dict(plan.asked) == {"period": "2026-12"}
        shown = {h.target.shown for h in plan.handoffs}
        assert "clean/ledger_2026-12.csv" in shown and "clean/ledger_2026-11.csv" not in shown
        assert dict(plan.members["clean_ledger"].given) == {"period": "2026-12"}
        assert dict(plan.members["revenue"].given) == {"period": "2026-12"}
        assert dict(plan.members["load_ledger"].given) == {}

    def test_the_digest_changes_with_a_value_and_with_a_writer_and_not_otherwise(self, chain_project: Any) -> None:
        chain_project.month_end()
        first = load_chain(chain_project.root, "month_end").digest
        assert load_chain(chain_project.root, "month_end").digest == first
        assert load_chain(chain_project.root, "month_end", {"period": "2026-12"}).digest != first
        chain_project.copy("load_ledger", ["inbox/ledger.csv"], "stage/ledger.csv", title="renamed")
        assert load_chain(chain_project.root, "month_end").digest == first  # the title writes nothing
        chain_project.copy("revenue", ["clean/ledger_${rules.period}.csv", "clean/customers.csv"], "reports/other.csv")
        assert load_chain(chain_project.root, "month_end").digest != first


class TestTheOrderAgainstTheHandOffs:
    def test_a_reader_not_after_its_writer_is_refused_by_name(self, chain_project: Any) -> None:
        chain_project.source("in.csv")
        chain_project.copy("load", ["in.csv"], "stage.csv")
        chain_project.copy("report", ["stage.csv"], "out.csv")
        chain_project.chain("c", chain_text("load", "report"))
        message = refused(chain_project.root)
        assert "report reads stage.csv (step read_0), which load writes" in message
        assert "Add load to report's 'after'" in message
        assert "Nothing was run" in message

    def test_a_reader_before_its_writer_is_refused_as_backwards(self, chain_project: Any) -> None:
        chain_project.source("in.csv")
        chain_project.copy("load", ["in.csv"], "stage.csv")
        chain_project.copy("report", ["stage.csv"], "out.csv")
        chain_project.chain("c", chain_text("report", "load: report"))
        message = refused(chain_project.root)
        assert "load runs after report" in message and "Reverse the order" in message

    def test_every_disagreement_is_listed_at_once(self, chain_project: Any) -> None:
        chain_project.source("in.csv")
        chain_project.copy("load", ["in.csv"], "stage.csv")
        chain_project.copy("a", ["stage.csv"], "a.csv")
        chain_project.copy("b", ["stage.csv"], "b.csv")
        chain_project.chain("c", chain_text("load", "a", "b"))
        message = refused(chain_project.root)
        assert "a reads stage.csv" in message and "b reads stage.csv" in message

    def test_running_after_the_writer_by_way_of_another_pipeline_is_enough(self, chain_project: Any) -> None:
        chain_project.source("in.csv")
        chain_project.copy("load", ["in.csv"], "stage.csv")
        chain_project.copy("clean", ["stage.csv"], "clean.csv")
        chain_project.copy("report", ["stage.csv", "clean.csv"], "out.csv")
        chain_project.chain("c", chain_text("load", "clean: load", "report: clean"))
        plan = load_chain(chain_project.root, "c")
        assert ("load", "report") in {(h.writer, h.reader) for h in plan.handoffs}

    def test_an_order_with_no_file_behind_it_is_allowed(self, chain_project: Any) -> None:
        chain_project.source("a_in.csv")
        chain_project.source("b_in.csv")
        chain_project.copy("a", ["a_in.csv"], "a.csv")
        chain_project.copy("b", ["b_in.csv"], "b.csv")
        chain_project.chain("c", chain_text("a", "b: a"))
        plan = load_chain(chain_project.root, "c")
        assert plan.after("b") == ("a",)
        assert plan.handoffs_between("a", "b") == ()

    @pytest.mark.parametrize("order", [("one", "two"), ("two", "one")])
    def test_two_writers_of_one_file_are_refused_whichever_comes_first(self, chain_project: Any, order: tuple[str, str]) -> None:
        chain_project.source("in.csv")
        chain_project.copy("one", ["in.csv"], "out.csv")
        chain_project.copy("two", ["in.csv"], "./sub/../out.csv")
        chain_project.chain("c", chain_text(*order))
        message = refused(chain_project.root)
        assert "both write out.csv" in message
        assert "one (step write)" in message and "two (step write)" in message

    def test_two_writers_of_one_file_are_refused_even_when_ordered(self, chain_project: Any) -> None:
        chain_project.source("in.csv")
        chain_project.copy("one", ["in.csv"], "out.csv")
        chain_project.copy("two", ["in.csv"], "out.csv")
        chain_project.chain("c", chain_text("one", "two: one"))
        assert "both write out.csv" in refused(chain_project.root)

    def test_a_pipeline_reading_back_its_own_file_is_not_a_hand_off(self, chain_project: Any) -> None:
        chain_project.source("in.csv")
        chain_project.copy("load", ["in.csv"], "in.csv")
        chain_project.chain("c", chain_text("load"))
        plan = load_chain(chain_project.root, "c")
        assert plan.handoffs == () and plan.outside == ()


class TestTables:
    def table_pipelines(self, chain_project: Any, *, write: dict[str, Any], read: dict[str, Any]) -> None:
        chain_project.source("in.csv")
        steps = [
            {"id": "read", "tool": "read_csv", "config": {"path": "in.csv"}},
            {"id": "write", "tool": "write_table", "config": {"mode": "replace", **write}, "inputs": {"rows": "read.rows"}},
        ]
        chain_project.pipeline("load", yaml.safe_dump({"name": "load", "steps": steps}))
        steps = [
            {"id": "read", "tool": "read_table", "config": read},
            {"id": "write", "tool": "write_csv", "config": {"path": "out.csv", "overwrite": True}, "inputs": {"rows": "read.rows"}},
        ]
        chain_project.pipeline("report", yaml.safe_dump({"name": "report", "steps": steps}))
        chain_project.chain("c", chain_text("load", "report"))

    def test_a_table_read_and_written_is_a_hand_off_whatever_the_capitals_and_a_missing_schema(self, chain_project: Any) -> None:
        self.table_pipelines(
            chain_project,
            write={"connection": "TARGET_URL", "table": "Totals"},
            read={"connection": "TARGET_URL", "schema": "sales", "table": "totals"},
        )
        assert "report reads table sales.totals (TARGET_URL)" in refused(chain_project.root)

    def test_the_same_table_name_on_another_connection_is_not(self, chain_project: Any) -> None:
        self.table_pipelines(
            chain_project,
            write={"connection": "TARGET_URL", "table": "totals"},
            read={"connection": "SOURCE_URL", "table": "totals"},
        )
        plan = load_chain(chain_project.root, "c")
        assert [t.shown for t, _ in plan.outside] == ["in.csv", "table totals (SOURCE_URL)"]

    def test_the_same_table_name_in_two_named_schemas_is_not(self, chain_project: Any) -> None:
        self.table_pipelines(
            chain_project,
            write={"connection": "TARGET_URL", "schema": "stage", "table": "totals"},
            read={"connection": "TARGET_URL", "schema": "sales", "table": "totals"},
        )
        assert load_chain(chain_project.root, "c").handoffs == ()

    def test_publish_tables_writes_each_of_its_tables(self, chain_project: Any) -> None:
        chain_project.source("in.csv")
        steps = [
            {"id": "read", "tool": "read_csv", "config": {"path": "in.csv"}},
            {
                "id": "publish",
                "tool": "publish_tables",
                "config": {
                    "connection": "TARGET_URL",
                    "tables": [
                        {"port": "a", "table": "first", "mode": "replace"},
                        {"port": "b", "table": "second", "mode": "replace"},
                    ],
                },
                "inputs": {"a": "read.rows", "b": "read.rows"},
            },
        ]
        chain_project.pipeline("load", yaml.safe_dump({"name": "load", "steps": steps}))
        steps = [
            {"id": "read", "tool": "read_table", "config": {"connection": "TARGET_URL", "table": "second"}},
            {"id": "write", "tool": "write_csv", "config": {"path": "out.csv", "overwrite": True}, "inputs": {"rows": "read.rows"}},
        ]
        chain_project.pipeline("report", yaml.safe_dump({"name": "report", "steps": steps}))
        chain_project.chain("c", chain_text("load", "report"))
        assert "report reads table second (TARGET_URL)" in refused(chain_project.root)


class TestEveryReaderAndWriterIsUnderstood:
    def test_each_worked_example_of_a_reader_or_writer_names_its_file_or_table(self) -> None:
        """A tool that reads or writes under a setting this module does not know would be missed."""
        seen = 0
        for tool in REGISTRY.tools():
            if tool.category not in (READER_CATEGORY, WRITER_CATEGORY):
                continue
            for example in tool.help().examples:
                found = _targets(example.configuration, Path("/project"))
                assert found, f"{tool.name}: cannot tell what this example reads or writes: {example.configuration}"
                assert all(isinstance(target, Target) for target in found)
                seen += 1
        assert seen >= 7


class TestParameters:
    def test_a_parameter_replaces_a_pipelines_own_value_and_says_so(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.settings("pipelines:\n  clean_ledger:\n    rules:\n      period: '2026-10'\n")
        plan = load_chain(chain_project.root, "month_end")
        assert dict(plan.members["clean_ledger"].replaced) == {"period": "2026-10"}
        assert dict(plan.members["revenue"].replaced) == {}

    def test_the_same_own_value_is_not_reported_as_replaced(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.settings("pipelines:\n  clean_ledger:\n    rules:\n      period: '2026-11'\n")
        assert dict(load_chain(chain_project.root, "month_end").members["clean_ledger"].replaced) == {}

    def test_a_pipelines_other_rules_still_come_from_its_settings(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.copy("clean_ledger", ["stage/ledger.csv"], "clean/ledger_${rules.period}_${rules.suffix}.csv")
        chain_project.copy("revenue", ["clean/ledger_${rules.period}_x.csv", "clean/customers.csv"], "reports/r.csv")
        chain_project.settings("pipelines:\n  clean_ledger:\n    rules:\n      suffix: x\n")
        plan = load_chain(chain_project.root, "month_end")
        assert "clean/ledger_2026-11_x.csv" in {h.target.shown for h in plan.handoffs}

    def test_a_rule_neither_the_chain_nor_the_settings_give_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.copy("clean_customers", ["stage/customers.csv"], "clean/customers_${rules.region}.csv")
        message = refused(chain_project.root, "month_end")
        assert "clean_customers" in message and "${rules.region}" in message and "period" not in message.split("Add to")[1]

    def test_a_parameter_no_pipeline_uses_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        text = (chain_project.root / "chains" / "month_end.yaml").read_text(encoding="utf8")
        chain_project.chain("month_end", text.replace("  period: '2026-11'\n", "  period: '2026-11'\n  perod: x\n"))
        message = refused(chain_project.root, "month_end")
        assert "No pipeline in chain 'month_end' uses the parameter perod" in message

    def test_a_value_for_an_unknown_parameter_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        message = refused(chain_project.root, "month_end", {"perod": "2026-12"})
        assert "'perod', which is not a parameter" in message and "period" in message

    def test_a_value_of_the_wrong_kind_is_refused(self, chain_project: Any) -> None:
        chain_project.month_end()
        assert "The parameter period" in refused(chain_project.root, "month_end", {"period": 202612})


class TestWhatIsInTheChain:
    def test_an_unknown_chain_is_refused_with_the_ones_there_are(self, chain_project: Any) -> None:
        chain_project.month_end()
        assert "did you mean 'month_end'" in refused(chain_project.root, "month_ned")

    def test_a_pipeline_not_in_the_project_is_refused_with_a_guess(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.chain("c", chain_text("load_ledgr"))
        assert "'load_ledgr', which is not a pipeline" in refused(chain_project.root)

    def test_a_pipeline_that_does_not_load_is_refused_by_name(self, chain_project: Any) -> None:
        chain_project.pipeline("broken", "name: x\nsteps:\n  - {id: a, tool: no_such_tool}\n")
        chain_project.chain("c", chain_text("broken"))
        message = refused(chain_project.root)
        assert "runs broken, which does not load" in message and "no_such_tool" in message


class TestSelect:
    @pytest.fixture
    def plan(self, chain_project: Any) -> Any:
        chain_project.month_end()
        return load_chain(chain_project.root, "month_end")

    def test_nothing_named_covers_the_whole_chain(self, plan: Any) -> None:
        assert plan.select() == plan.order

    def test_from_covers_the_pipeline_and_everything_after_it(self, plan: Any) -> None:
        assert plan.select("clean_ledger") == ("clean_ledger", "revenue")
        assert plan.select("load_ledger") == ("load_ledger", "clean_ledger", "revenue")
        assert plan.select("revenue") == ("revenue",)

    def test_until_covers_the_pipeline_and_everything_before_it(self, plan: Any) -> None:
        assert plan.select(until="clean_ledger") == ("load_ledger", "clean_ledger")
        assert plan.select(until="revenue") == plan.order
        assert plan.select(until="load_customers") == ("load_customers",)

    def test_both_cover_what_lies_between(self, plan: Any) -> None:
        assert plan.select("load_ledger", "clean_ledger") == ("load_ledger", "clean_ledger")
        assert plan.select(["load_ledger", "load_customers"], "revenue") == plan.order

    def test_nothing_between_is_refused(self, plan: Any) -> None:
        with pytest.raises(ChainError, match="nothing to run"):
            plan.select("clean_ledger", "load_customers")

    def test_an_unknown_name_is_refused(self, plan: Any) -> None:
        with pytest.raises(ChainError, match="--from names 'clean'"):
            plan.select("clean")
