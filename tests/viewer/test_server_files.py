"""Looking at a file a step names, through the project server: viewer/server.py.

Every test sends real HTTP requests to a real server on a port the operating system picks. The
browser names a pipeline, a step and a setting, never a path. Opening a location is handed to a
fake, so no file manager is ever started: the tests check what the server would have opened and
that a refused request opens nothing.
"""

from __future__ import annotations

import http.client
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
import yaml

from dantron_etl.viewer.project import load_project
from dantron_etl.viewer.reveal import Opened, Target
from dantron_etl.viewer.server import ViewerServer, start_server

SECRET = "do-not-serve-5e1b"
ORDERS = "party,amount\nA,1\nB,-1\nC,2\n"


def reader(step_id: str, path: str) -> dict[str, Any]:
    return {"id": step_id, "tool": "read_csv", "config": {"path": path}}


def writer(step_id: str, path: str, source: str) -> dict[str, Any]:
    return {"id": step_id, "tool": "write_csv", "config": {"path": path, "overwrite": True}, "inputs": {"rows": source}}


def make_project(root: Path) -> Path:
    """One pipeline reading four files (plain, outside, through a link, named like an option)."""
    (root / "pipelines").mkdir(parents=True)
    (root / "output").mkdir()
    (root / "pipelines" / "orders.csv").write_text(ORDERS, encoding="utf8")
    (root / "pipelines" / "-n.csv").write_text(ORDERS, encoding="utf8")
    shared = root.parent / "shared"
    shared.mkdir()
    (shared / "rates.csv").write_text("party,rate\nA,2\n", encoding="utf8")
    (shared / "linked.csv").write_text("party\nthrough-the-link\n", encoding="utf8")
    (root / "pipelines" / "link.csv").symlink_to(shared / "linked.csv")
    (root / ".env").write_text(f"TOKEN={SECRET}\n", encoding="utf8")
    steps = [
        reader("read", "pipelines/orders.csv"),
        reader("rates", (shared / "rates.csv").as_posix()),
        reader("linked", "pipelines/link.csv"),
        reader("dashed", "pipelines/-n.csv"),
        {"id": "keep", "tool": "filter", "config": {"predicate": "[amount] > 0"}, "inputs": {"rows": "read.rows"}},
        writer("save", "output/totals.csv", "keep.true"),
        writer("save_rates", "output/rates.csv", "rates.rows"),
        writer("save_linked", "output/linked.csv", "linked.rows"),
        writer("save_dashed", "output/dashed.csv", "dashed.rows"),
    ]
    (root / "pipelines" / "orders.yaml").write_text(
        yaml.safe_dump({"name": "Orders", "steps": steps}, sort_keys=False), encoding="utf8"
    )
    needs = [reader("read", "${rules.folder}/orders.csv"), writer("save", "output/ruled.csv", "read.rows")]
    (root / "pipelines" / "ruled.yaml").write_text(
        yaml.safe_dump({"name": "Ruled", "steps": needs}, sort_keys=False), encoding="utf8"
    )
    return root


@dataclass
class FakeOpener:
    """Stands in for the file manager: records each file it was asked to show."""

    asked: list[Target] = field(default_factory=list)

    def __call__(self, target: Target) -> Opened:
        self.asked.append(target)
        return Opened(True, "Shown.", ("fake-file-manager", target.path))


@dataclass
class Served:
    root: Path
    server: ViewerServer
    opener: FakeOpener


@pytest.fixture(autouse=True)
def _no_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ETL_PROJECT_ROOT", raising=False)


@pytest.fixture(scope="module")
def one_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    root = make_project(tmp_path_factory.mktemp("files") / "project")
    opener = FakeOpener()
    server = start_server(load_project(root), port=0, reveal=opener)
    try:
        yield Served(root.resolve(), server, opener)
    finally:
        server.shutdown()


@pytest.fixture
def served(one_server: Served) -> Served:
    """The module's server, with nothing opened yet."""
    one_server.opener.asked.clear()
    return one_server


@dataclass(frozen=True)
class Answer:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def call(server: ViewerServer, method: str, target: str, *, body: bytes | None = None, **headers: str) -> Answer:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        connection.request(method, target, body=body, headers={k.replace("_", "-"): v for k, v in headers.items()})
        answer = connection.getresponse()
        return Answer(answer.status, {k.lower(): v for k, v in answer.getheaders()}, answer.read())
    finally:
        connection.close()


def ask(served: Served, what: str, pipeline: str = "orders", token: str | None = None, **query: str) -> Answer:
    sent = served.server.token if token is None else token
    return call(served.server, "GET", f"/{pipeline}/api/{what}?{urlencode(query)}", X_Run_Token=sent)


def open_location(served: Served, payload: Any, token: str | None = None, **headers: str) -> Answer:
    sent = served.server.token if token is None else token
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf8")
    return call(
        served.server, "POST", "/api/open-location", body=body,
        Content_Type="application/json", X_Run_Token=sent, **headers,
    )  # fmt: skip


def assert_refused(answer: Answer, status: int) -> None:
    assert answer.status == status, answer.body[:300]
    assert answer.headers["content-type"].startswith("application/json")
    assert answer.json()["error"]
    assert SECRET.encode() not in answer.body


# ---------------------------------------------------------------------------
# What the page is told about a file
# ---------------------------------------------------------------------------


class TestFacts:
    def test_a_file_a_step_reads_is_given_with_the_full_path_the_run_uses(self, served: Served) -> None:
        found = ask(served, "file", step="read", setting="path").json()
        assert found["path"] == str(served.root / "pipelines" / "orders.csv")
        assert found["exists"] and found["size"] == len(ORDERS)
        assert not found["outside_project"]

    def test_a_file_not_written_yet_is_given_as_missing(self, served: Served) -> None:
        found = ask(served, "file", step="save", setting="path").json()
        assert found["path"] == str(served.root / "output" / "totals.csv")
        assert not found["exists"] and found["folder_exists"]

    def test_a_file_outside_the_project_that_a_step_names_is_given_and_marked(self, served: Served) -> None:
        found = ask(served, "file", step="rates", setting="path").json()
        assert found["exists"] and found["outside_project"]

    def test_a_path_sent_by_the_browser_is_ignored(self, served: Served) -> None:
        found = ask(served, "file", step="read", setting="path", path="/etc/passwd").json()
        assert found["path"] == str(served.root / "pipelines" / "orders.csv")

    @pytest.mark.parametrize(
        ("pipeline", "query"),
        [
            ("orders", {"step": "nowhere", "setting": "path"}),
            ("orders", {"step": "keep", "setting": "predicate"}),
            ("orders", {"step": "save", "setting": "overwrite"}),
            ("orders", {"step": "read", "setting": "../.env"}),
            ("orders", {"step": "read"}),
            ("orders", {"path": ".env"}),
            ("nothing", {"step": "read", "setting": "path"}),
            ("..", {"step": "read", "setting": "path"}),
        ],
        ids=["unknown-step", "no-file-setting", "not-a-file-setting", "a-path-as-setting", "no-setting", "a-path-only",
             "unknown-pipeline", "dot-dot-pipeline"],
    )  # fmt: skip
    @pytest.mark.parametrize("what", ["file", "file-preview"])
    def test_anything_but_a_file_a_step_names_is_refused(
        self, served: Served, what: str, pipeline: str, query: dict[str, str]
    ) -> None:
        assert_refused(ask(served, what, pipeline, **query), 404)

    @pytest.mark.parametrize("what", ["file", "file-preview"])
    @pytest.mark.parametrize("token", ["", "wrong"])
    def test_looking_needs_the_run_token_because_it_loads_the_pipeline(
        self, served: Served, what: str, token: str
    ) -> None:
        assert_refused(ask(served, what, token=token, step="read", setting="path"), 403)

    def test_a_pipeline_that_does_not_load_says_why(self, served: Served) -> None:
        answer = ask(served, "file", "ruled", step="read", setting="path")
        assert_refused(answer, 409)
        assert "rules" in answer.json()["error"]


class TestPreview:
    def test_a_csv_shows_its_first_rows(self, served: Served) -> None:
        shown = ask(served, "file-preview", step="read", setting="path").json()
        assert shown["kind"] == "table"
        assert shown["columns"] == ["party", "amount"]
        assert shown["rows"] == [["A", "1"], ["B", "-1"], ["C", "2"]]

    def test_a_file_named_through_a_link_shows_what_the_step_reads(self, served: Served) -> None:
        shown = ask(served, "file-preview", step="linked", setting="path").json()
        assert shown["rows"] == [["through-the-link"]]

    def test_a_file_not_written_yet_has_nothing_to_show(self, served: Served) -> None:
        assert ask(served, "file-preview", step="save", setting="path").json()["kind"] == "none"


# ---------------------------------------------------------------------------
# Opening a file's location
# ---------------------------------------------------------------------------


class TestOpenLocation:
    def test_the_server_shows_the_file_the_step_names_and_says_what_it_did(self, served: Served) -> None:
        answer = open_location(served, {"pipeline": "orders", "step": "read", "setting": "path"})
        assert answer.status == 200, answer.body
        assert answer.json()["opened"] is True
        assert answer.json()["message"] == "Shown."
        assert served.opener.asked == [Target(str(served.root / "pipelines" / "orders.csv"), True, True)]

    def test_a_file_not_written_yet_is_passed_on_as_missing_so_its_folder_is_opened(self, served: Served) -> None:
        assert open_location(served, {"pipeline": "orders", "step": "save", "setting": "path"}).status == 200
        assert served.opener.asked == [Target(str(served.root / "output" / "totals.csv"), False, True)]

    def test_a_link_is_shown_where_the_file_really_is(self, served: Served) -> None:
        assert open_location(served, {"pipeline": "orders", "step": "linked", "setting": "path"}).status == 200
        assert served.opener.asked[0].path == str((served.root.parent / "shared" / "linked.csv").resolve())

    def test_a_file_outside_the_project_that_a_step_names_may_be_shown(self, served: Served) -> None:
        assert open_location(served, {"pipeline": "orders", "step": "rates", "setting": "path"}).status == 200
        assert served.opener.asked[0].path == str((served.root.parent / "shared" / "rates.csv").resolve())

    def test_a_file_named_like_an_option_is_passed_as_its_full_path(self, served: Served) -> None:
        assert open_location(served, {"pipeline": "orders", "step": "dashed", "setting": "path"}).status == 200
        shown = served.opener.asked[0].path
        assert os.path.isabs(shown) and not shown.startswith("-")
        assert shown == str(served.root / "pipelines" / "-n.csv")

    def test_a_path_in_the_request_is_ignored(self, served: Served) -> None:
        payload = {"pipeline": "orders", "step": "read", "setting": "path", "path": "/etc/passwd"}
        assert open_location(served, payload).status == 200
        assert served.opener.asked[0].path == str(served.root / "pipelines" / "orders.csv")

    @pytest.mark.parametrize(
        "payload",
        [
            {"pipeline": "orders", "step": "nowhere", "setting": "path"},
            {"pipeline": "orders", "step": "keep", "setting": "predicate"},
            {"pipeline": "orders", "step": "read", "setting": "delimiter"},
            {"pipeline": "orders", "step": "read", "setting": None},
            {"pipeline": "nothing", "step": "read", "setting": "path"},
            {"pipeline": ["orders"], "step": "read", "setting": "path"},
            {"path": str(Path("/etc/passwd"))},
        ],
        ids=["unknown-step", "no-file-setting", "not-a-file-setting", "no-setting", "unknown-pipeline",
             "pipeline-not-text", "a-path-only"],
    )  # fmt: skip
    def test_anything_but_a_file_a_step_names_is_refused_and_nothing_opens(
        self, served: Served, payload: dict[str, Any]
    ) -> None:
        assert_refused(open_location(served, payload), 404)
        assert served.opener.asked == []

    @pytest.mark.parametrize("token", ["", "wrong"])
    def test_without_the_run_token_nothing_opens(self, served: Served, token: str) -> None:
        answer = open_location(served, {"pipeline": "orders", "step": "read", "setting": "path"}, token=token)
        assert_refused(answer, 403)
        assert served.opener.asked == []

    def test_a_request_from_another_site_is_refused_and_nothing_opens(self, served: Served) -> None:
        payload = {"pipeline": "orders", "step": "read", "setting": "path"}
        assert_refused(open_location(served, payload, Origin="http://elsewhere.example"), 403)
        assert served.opener.asked == []

    def test_a_get_is_refused_and_nothing_opens(self, served: Served) -> None:
        query = urlencode({"pipeline": "orders", "step": "read", "setting": "path"})
        answer = call(served.server, "GET", f"/api/open-location?{query}", X_Run_Token=served.server.token)
        assert_refused(answer, 405)
        assert answer.headers.get("allow") == "POST"
        assert served.opener.asked == []

    def test_a_body_that_is_not_json_is_refused_and_nothing_opens(self, served: Served) -> None:
        assert_refused(open_location(served, b"step=read"), 400)
        assert served.opener.asked == []


# ---------------------------------------------------------------------------
# The real-run confirmation
# ---------------------------------------------------------------------------


def test_the_plan_gives_each_file_a_real_run_writes_with_the_same_facts(served: Served) -> None:
    plan = ask(served, "plan").json()
    by_step = {write["step"]: write for write in plan["writes"]}
    files = by_step["save"]["files"]
    assert [(one["step"], one["setting"]) for one in files] == [("save", "path")]
    assert files[0]["path"] == str(served.root / "output" / "totals.csv")
    assert files[0] == ask(served, "file", step="save", setting="path").json()
    digest = plan["digest"]
    written = served.root / "output" / "totals.csv"
    written.write_text("party\n", encoding="utf8")
    try:
        again = ask(served, "plan").json()
    finally:
        written.unlink()
    assert {w["step"]: w for w in again["writes"]}["save"]["files"][0]["exists"]
    assert again["digest"] == digest  # a file appearing is not a change to what the run writes
