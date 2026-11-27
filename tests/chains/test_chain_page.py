"""A chain's page in the viewer: one box per pipeline, one line per 'after', and how each did.

The tests read the drawn page back and check properties of it (every pipeline has a box, every line
joins the right two boxes, a box stands to the right of everything it runs after), not that some
string appears in it.
"""

import urllib.request
from html.parser import HTMLParser
from typing import Any

import pytest
import yaml

from dantron_etl.chains import load_chain, run_chain
from dantron_etl.viewer.chain_page import chain_page
from dantron_etl.viewer.server import start_server


class Drawn(HTMLParser):
    """The boxes and lines of a chain page's graph, as the page draws them."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: dict[str, dict[str, str]] = {}
        self.edges: dict[tuple[str, str], dict[str, str]] = {}
        self._in: tuple[str, Any] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        found = {key: value or "" for key, value in attrs}
        classes = found.get("class", "").split()
        if tag == "a" and "node" in classes:
            self.nodes[found["data-pipeline"]] = {**found, "kind": classes[1]}
            self._in = ("node", found["data-pipeline"])
        elif tag == "g" and "edge" in classes:
            key = (found["data-from"], found["data-to"])
            self.edges[key] = {**found, "kind": classes[1]}
            self._in = ("edge", key)
        elif tag == "rect" and self._in and self._in[0] == "node":
            self.nodes[self._in[1]]["x"] = found["x"]
        elif tag == "text" and self._in:
            self._text = []

    def handle_data(self, data: str) -> None:
        self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "text" and self._in:
            kind, key = self._in
            target = self.nodes[key] if kind == "node" else self.edges[key]
            target.setdefault("texts", []).append("".join(self._text))  # type: ignore[union-attr]
        if tag in ("a", "g"):
            self._in = None


def drawn(html: str) -> Drawn:
    parser = Drawn()
    parser.feed(html)
    return parser


class TestTheGraph:
    def test_every_pipeline_has_a_box_that_opens_its_page(self, chain_project: Any) -> None:
        chain_project.month_end()
        page = drawn(chain_page(chain_project.project(), "month_end"))
        plan = load_chain(chain_project.root, "month_end")
        assert set(page.nodes) == set(plan.order)
        assert all(node["href"] == f"/{name}/" for name, node in page.nodes.items())

    def test_every_after_is_one_line_between_the_right_boxes(self, chain_project: Any) -> None:
        chain_project.month_end()
        page = drawn(chain_page(chain_project.project(), "month_end"))
        plan = load_chain(chain_project.root, "month_end")
        assert set(page.edges) == {(before, name) for name in plan.order for before in plan.after(name)}

    def test_a_box_stands_right_of_everything_it_runs_after(self, chain_project: Any) -> None:
        chain_project.month_end()
        page = drawn(chain_page(chain_project.project(), "month_end"))
        for before, name in page.edges:
            assert float(page.nodes[name]["x"]) > float(page.nodes[before]["x"])

    def test_a_line_says_what_passes_along_it_or_that_nothing_does(self, chain_project: Any) -> None:
        chain_project.month_end()
        chain_project.source("notes.csv")
        chain_project.copy("notes", ["notes.csv"], "notes_out.csv")
        text = (chain_project.root / "chains" / "month_end.yaml").read_text(encoding="utf8")
        chain_project.chain("month_end", text + "  - name: notes\n    after: [revenue]\n")
        page = drawn(chain_page(chain_project.project(), "month_end"))
        assert page.edges[("clean_ledger", "revenue")]["kind"] == "passes"
        assert page.edges[("clean_ledger", "revenue")]["texts"] == ["clean/ledger_2026-11.csv"]
        assert page.edges[("revenue", "notes")]["kind"] == "order"
        assert page.edges[("revenue", "notes")]["texts"] == ["order only"]

    def test_several_files_on_one_line_are_counted_and_all_named_in_its_title(self, chain_project: Any) -> None:
        chain_project.source("a.csv")
        chain_project.source("b.csv")
        steps = [
            {"id": "ra", "tool": "read_csv", "config": {"path": "a.csv"}},
            {"id": "wa", "tool": "write_csv", "config": {"path": "x.csv", "overwrite": True}, "inputs": {"rows": "ra.rows"}},
            {"id": "wb", "tool": "write_csv", "config": {"path": "y.csv", "overwrite": True}, "inputs": {"rows": "ra.rows"}},
        ]
        chain_project.pipeline("load", yaml.safe_dump({"name": "load", "steps": steps}))
        chain_project.copy("report", ["x.csv", "y.csv"], "out.csv")
        chain_project.chain("c", "pipelines:\n  - load\n  - {name: report, after: [load]}\n")
        html = chain_page(chain_project.project(), "c")
        assert drawn(html).edges[("load", "report")]["texts"] == ["2 files and tables"]
        assert "load writes, and report reads: x.csv, y.csv" in html


class TestHowEachDid:
    def test_before_any_run_every_box_says_so(self, chain_project: Any) -> None:
        chain_project.month_end()
        page = drawn(chain_page(chain_project.project(), "month_end"))
        assert {node["data-outcome"] for node in page.nodes.values()} == {"never"}

    @pytest.mark.parametrize("broken", ["load_ledger", "load_customers"])
    def test_each_box_shows_its_pipelines_outcome_in_the_latest_run(self, chain_project: Any, broken: str) -> None:
        chain_project.month_end()
        (chain_project.root / "inbox" / f"{broken.removeprefix('load_')}.csv").unlink()
        plan = load_chain(chain_project.root, "month_end")
        from concurrent.futures import ThreadPoolExecutor

        result = run_chain(chain_project.root, "month_end", confirm_digest=plan.digest, echo_steps=False,
                           executor=lambda size: ThreadPoolExecutor(size))
        page = drawn(chain_page(chain_project.project(), "month_end"))
        assert {name: node["data-outcome"] for name, node in page.nodes.items()} == {
            member.pipeline: member.outcome.replace(" ", "-") for member in result.members
        }
        assert page.nodes[broken]["kind"] == "failed"
        assert page.nodes["revenue"]["kind"] == "not-started"


class TestARefusedChain:
    def test_the_refusal_is_shown_and_the_boxes_are_still_drawn_from_the_file(self, chain_project: Any) -> None:
        chain_project.month_end()
        text = (chain_project.root / "chains" / "month_end.yaml").read_text(encoding="utf8")
        chain_project.chain("month_end", text.replace("after: [clean_ledger, clean_customers]", "after: [clean_ledger]"))
        html = chain_page(chain_project.project(), "month_end")
        assert "Add clean_customers to revenue&#x27;s &#x27;after&#x27;" in html
        page = drawn(html)
        assert len(page.nodes) == 5
        assert ("clean_customers", "revenue") not in page.edges
        assert "chains/month_end.yaml" in html


class TestTheViewer:
    def get(self, url: str) -> tuple[int, str, str]:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=30) as response:
            return response.status, response.geturl(), response.read().decode("utf-8")

    def test_the_home_page_lists_chains_and_a_chain_has_its_page(self, chain_project: Any) -> None:
        chain_project.month_end()
        server = start_server(chain_project.project(), port=0)
        try:
            _, _, home = self.get(server.url)
            assert '<a href="/month_end/">Month-end close</a>' in home
            status, where, html = self.get(f"{server.url}month_end")
            assert status == 200 and where.endswith("/month_end/")
            assert set(drawn(html).nodes) == set(load_chain(chain_project.root, "month_end").order)
        finally:
            server.shutdown()
