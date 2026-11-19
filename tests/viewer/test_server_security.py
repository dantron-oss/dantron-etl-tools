"""The project server answers only this machine, runs only for its own pages, and serves no file by path.

Every test sends real HTTP requests to a real server on a port the operating system picks. The
project holds the things a real project root holds and the server must never hand out: a `.env`,
a `pyproject.toml`, a hidden pipeline file, and a secret file just outside the root.
"""

from __future__ import annotations

import http.client
import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml

from dantron_etl.preview import PAGE_NAME
from dantron_etl.viewer.project import load_project
from dantron_etl.viewer.server import ViewerServer, start_server

SECRET = "do-not-serve-7f3a"
DONE = {"finished", "failed", "refused"}
PIPELINES = ("orders", "untouched", "origin_ok")


def flow(root: Path, name: str) -> str:
    return (
        f"name: Totals for {name}\n"
        "steps:\n"
        "  - id: read\n"
        "    tool: read_csv\n"
        f"    config: {{path: pipelines/{name}/orders.csv}}\n"
        "  - id: keep\n"
        "    tool: filter\n"
        '    config: {predicate: "[amount] > 0"}\n'
        "    inputs: {rows: read.rows}\n"
        "  - id: save\n"
        "    tool: write_csv\n"
        f'    config: {{path: "{(root / "output" / f"{name}.csv").as_posix()}", overwrite: true}}\n'
        "    inputs: {rows: keep.true}\n"
    )


def make_project(root: Path) -> Path:
    (root / "output").mkdir(parents=True)
    (root / "pipelines").mkdir()
    for name in PIPELINES:
        (root / "pipelines" / f"{name}.yaml").write_text(flow(root, name), encoding="utf8")
        (root / "pipelines" / name).mkdir()
        (root / "pipelines" / name / "orders.csv").write_text("party,amount\nA,1\nB,-1\nC,2\nD,3\n", encoding="utf8")
    settings = {"pipelines": {name: {"focus": "party"} for name in PIPELINES}}
    (root / "etl-project.yaml").write_text(yaml.safe_dump(settings), encoding="utf8")
    (root / "pipelines" / ".hidden.yaml").write_text(flow(root, "hidden"), encoding="utf8")
    (root / ".env").write_text(f"TOKEN={SECRET}\n", encoding="utf8")
    (root / "pyproject.toml").write_text(f'[project]\nname = "{SECRET}"\n', encoding="utf8")
    (root.parent / "outside.txt").write_text(SECRET, encoding="utf8")
    (root.parent / "etc").mkdir()
    (root.parent / "etc" / "passwd").write_text(SECRET, encoding="utf8")
    return root


@dataclass(frozen=True)
class Answer:
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf8", errors="replace")


def call(
    server: ViewerServer, method: str, target: str, *, body: bytes | None = None, headers: dict[str, str] | None = None
) -> Answer:
    """One request, with the target sent exactly as written: nothing re-encoded or tidied."""
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        connection.request(method, target, body=body, headers=headers or {})
        answer = connection.getresponse()
        return Answer(answer.status, {k.lower(): v for k, v in answer.getheaders()}, answer.read())
    finally:
        connection.close()


def post_run(server: ViewerServer, name: str, headers: dict[str, str]) -> Answer:
    body = json.dumps({"pipeline": name, "kind": "preview"}).encode("utf8")
    return call(server, "POST", "/api/run", body=body, headers={"Content-Type": "application/json", **headers})


def state(server: ViewerServer, name: str) -> str:
    answer = call(server, "GET", f"/{name}/api/progress")
    assert answer.status == 200
    return json.loads(answer.body)["state"]


def wait_done(server: ViewerServer, name: str) -> str:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        found = state(server, name)
        if found in DONE:
            return found
        time.sleep(0.05)
    raise AssertionError(f"run of {name!r} did not end")


def meta_token(page: str) -> str | None:
    tag = re.search(r'<meta\b[^>]*\bname="run-token"[^>]*>', page)
    if tag is None:
        return None
    content = re.search(r'\bcontent="([^"]*)"', tag.group(0))
    return content.group(1) if content else None


@dataclass
class Served:
    root: Path
    server: ViewerServer


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    root = make_project(tmp_path_factory.mktemp("security") / "project")
    server = start_server(load_project(root), port=0)
    try:
        answer = post_run(server, "orders", {"X-Run-Token": server.token})
        assert answer.status == 202, answer.text
        assert wait_done(server, "orders") == "finished"
        yield Served(root, server)
    finally:
        server.shutdown()


def nothing_ran(served: Served, name: str) -> None:
    time.sleep(0.2)
    assert state(served.server, name) == "idle"
    assert not (load_project(served.root).runs_folder / name).exists()
    assert not (served.root / "output" / f"{name}.csv").exists()


# ---------------------------------------------------------------------------
# The run token
# ---------------------------------------------------------------------------


class TestRunToken:
    @pytest.mark.parametrize(
        "sent", [None, "", "wrong", "TOKEN+x"], ids=["none", "empty", "wrong", "one-more-character"]
    )
    def test_a_run_without_the_right_token_is_forbidden_and_nothing_runs(
        self, served: Served, sent: str | None
    ) -> None:
        headers = {} if sent is None else {"X-Run-Token": sent.replace("TOKEN", served.server.token)}
        assert post_run(served.server, "untouched", headers).status == 403
        nothing_ran(served, "untouched")

    def test_each_server_gets_its_own_long_token(self, served: Served) -> None:
        other = start_server(load_project(served.root), port=0)
        try:
            assert other.token != served.server.token
            assert len(served.server.token) >= 32
        finally:
            other.shutdown()

    def test_the_token_is_in_the_served_page_and_not_in_the_saved_one(self, served: Served) -> None:
        served_page = call(served.server, "GET", "/orders/").text
        assert meta_token(served_page) == served.server.token
        saved = (load_project(served.root).runs_folder / "orders" / PAGE_NAME).read_text(encoding="utf8")
        assert served.server.token not in saved
        assert meta_token(saved) == ""

    @pytest.mark.parametrize("target", ["/", "/untouched/"])
    def test_pages_offering_a_run_carry_the_token(self, served: Served, target: str) -> None:
        assert meta_token(call(served.server, "GET", target).text) == served.server.token


# ---------------------------------------------------------------------------
# Names are looked up, never joined into a path
# ---------------------------------------------------------------------------


class TestNames:
    @pytest.mark.parametrize(
        "target",
        [
            "/nope/",
            "/../",
            "/..",
            "/%2e%2e/",
            "/%2e%2e/outside.txt",
            "/../outside.txt",
            "/../etc/passwd",
            "/%2e%2e/etc/passwd",
            "/..%2fetc%2fpasswd",
            "/a/b/",
            "/.hidden/",
            "/hidden/",
            "/orders.failed/",
            "/orders/../../outside.txt",
        ],
    )
    def test_an_unknown_or_unsafe_name_is_404_and_serves_no_file(self, served: Served, target: str) -> None:
        answer = call(served.server, "GET", target)
        assert answer.status == 404
        assert SECRET not in answer.text

    @pytest.mark.parametrize(
        "target",
        [
            "/orders/best/",
            "/orders/GOOD/",
            "/orders/good2/api/ports",
            "/orders/good/../../outside.txt",
            "/orders/run.json",
            "/orders/api/nope",
        ],
    )
    def test_a_route_other_than_good_or_failed_is_404(self, served: Served, target: str) -> None:
        answer = call(served.server, "GET", target)
        assert answer.status == 404
        assert SECRET not in answer.text

    def test_a_post_for_an_unknown_pipeline_is_404(self, served: Served) -> None:
        for name in ("nope", "..", "../orders", ".hidden"):
            assert post_run(served.server, name, {"X-Run-Token": served.server.token}).status == 404, name

    @pytest.mark.parametrize(
        "target", ["/api/source?path=pyproject.toml", "/api/source?path=.env", "/api/source?path=pipelines/orders.yaml"]
    )
    def test_source_files_are_not_served(self, served: Served, target: str) -> None:
        answer = call(served.server, "GET", target)
        assert answer.status == 404
        assert SECRET not in answer.text

    @pytest.mark.parametrize(
        "target",
        [
            "/.env",
            "/orders/.env",
            "/orders/good/.env",
            "/catalog/.env",
            "/pyproject.toml",
            "/orders/api/rows?port=../../../.env",
        ],
    )
    def test_a_dotfile_or_project_file_is_never_served(self, served: Served, target: str) -> None:
        answer = call(served.server, "GET", target)
        assert answer.status in {400, 404}
        assert SECRET not in answer.text


# ---------------------------------------------------------------------------
# Only this machine, only its own pages
# ---------------------------------------------------------------------------


class TestThisMachineOnly:
    def test_it_listens_on_loopback_by_default(self, served: Served) -> None:
        address = urlparse(served.server.url)
        assert address.hostname == "127.0.0.1"
        assert address.port == served.server.port != 0

    @pytest.mark.parametrize(
        "host", ["evil.example:80", "evil.example", "127.0.0.1.evil.example", "localhost.evil.example:8899"]
    )
    def test_a_foreign_host_header_is_forbidden_for_reading_and_running(self, served: Served, host: str) -> None:
        answer = call(served.server, "GET", "/api/pipelines", headers={"Host": host})
        assert answer.status == 403
        assert "orders" not in answer.text
        assert post_run(served.server, "untouched", {"X-Run-Token": served.server.token, "Host": host}).status == 403
        nothing_ran(served, "untouched")

    @pytest.mark.parametrize("host", ["localhost:12345", "127.0.0.1:1", "[::1]:9", "localhost"])
    def test_a_loopback_host_on_any_port_is_allowed(self, served: Served, host: str) -> None:
        answer = call(served.server, "GET", "/api/pipelines", headers={"Host": host})
        assert answer.status == 200
        assert {p["name"] for p in json.loads(answer.body)["pipelines"]} == set(PIPELINES)

    @pytest.mark.parametrize("origin", ["http://evil.example", "http://127.0.0.1.evil.example", "null"])
    def test_a_run_from_a_foreign_origin_is_forbidden_even_with_the_token(self, served: Served, origin: str) -> None:
        answer = post_run(served.server, "untouched", {"X-Run-Token": served.server.token, "Origin": origin})
        assert answer.status == 403
        nothing_ran(served, "untouched")

    def test_a_run_from_a_loopback_origin_on_another_port_is_accepted(self, served: Served) -> None:
        answer = post_run(
            served.server, "origin_ok", {"X-Run-Token": served.server.token, "Origin": "http://localhost:12345"}
        )
        assert answer.status == 202
        assert wait_done(served.server, "origin_ok") == "finished"

    @pytest.mark.parametrize("target", ["/", "/api/pipelines", "/orders/", "/orders/api/ports", "/nope/"])
    def test_every_answer_forbids_sniffing_and_caching(self, served: Served, target: str) -> None:
        headers: dict[str, Any] = call(served.server, "GET", target).headers
        assert headers.get("x-content-type-options") == "nosniff"
        assert "no-store" in headers.get("cache-control", "")
