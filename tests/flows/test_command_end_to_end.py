"""A new project, served, previewed and read back through the server, as the command's reader meets it.

``dantron-etl new`` writes a starter project; ``dantron-etl view`` serves it; the reader presses Run
preview and opens every step. These tests do the same through the same parts: the scaffold, the
project server on a port the system picks, a preview started with the run token, and the server's
own answers about what the run saved. The last test starts the real command in a child process and
stops it with ctrl-c.

What is compared is what the server says against what is on disk and in the pipeline file: the
writer's port and its file, the ports the run saved, the panels the page fetches, the record the
page carries, and the entities the picker offers. The page script is not run here. It turns this
data into markup, so the data is what is pinned.
"""

import html
import http.client
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest
import yaml

from dantron_etl.preview import MANIFEST, PAGE_NAME
from dantron_etl.viewer.project import load_project
from dantron_etl.viewer.runs import RUN_FILE
from dantron_etl.viewer.scaffold import write_project
from dantron_etl.viewer.server import ViewerServer, start_server

SRC = Path(__file__).resolve().parents[2] / "src"
NAME = "example"
DONE = {"finished", "failed", "refused"}
RUN_SECONDS = 60.0
STOP_SECONDS = 15.0


@pytest.fixture(autouse=True)
def _no_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ETL_PROJECT_ROOT", raising=False)


# ---------------------------------------------------------------------------
# Talking to the server
# ---------------------------------------------------------------------------


def request(server: ViewerServer, method: str, target: str, body: bytes | None = None, **headers: str) -> Any:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        connection.request(method, target, body=body, headers=headers)
        answer = connection.getresponse()
        return answer.status, answer.read()
    finally:
        connection.close()


def get_text(server: ViewerServer, target: str) -> str:
    status, body = request(server, "GET", target)
    assert status == 200, (target, status, body[:300])
    return body.decode("utf-8")


def get_json(server: ViewerServer, target: str) -> Any:
    return json.loads(get_text(server, target))


def listed(server: ViewerServer) -> dict[str, Any]:
    return {entry["name"]: entry for entry in get_json(server, "/api/pipelines")["pipelines"]}


def page_data(page: str) -> dict[str, Any]:
    found = re.search(r'<script\b[^>]*\bid="page-data"[^>]*>(.*?)</script>', page, re.S)
    assert found, "the page carries no page-data"
    return json.loads(found.group(1))


def drawn_steps(page: str) -> list[str]:
    return re.findall(r'<a class="node[^"]*" id="node-([^"]+)"', page)


def table_ports(markup: str) -> set[str]:
    return {html.unescape(port) for port in re.findall(r'<table class="data"[^>]*\bdata-port="([^"]+)"', markup)}


def every_row(server: ViewerServer, port: str) -> dict[str, Any]:
    first = get_json(server, f"/{NAME}/api/rows?port={quote(port)}&limit=1")
    return get_json(server, f"/{NAME}/api/rows?port={quote(port)}&limit={max(first['total'], 1)}")


# ---------------------------------------------------------------------------
# One starter project, previewed once
# ---------------------------------------------------------------------------


@dataclass
class Previewed:
    root: Path
    server: ViewerServer
    document: dict[str, Any]
    before: dict[str, Any] = field(default_factory=dict)
    home: str = ""
    accepted: int = 0
    progress: dict[str, Any] = field(default_factory=dict)

    @property
    def writers(self) -> list[dict[str, Any]]:
        return [step for step in self.document["steps"] if step["tool"] == "write_csv"]

    @property
    def manifest(self) -> dict[str, Any]:
        folder = load_project(self.root).runs_folder / NAME
        return json.loads((folder / MANIFEST).read_text(encoding="utf8"))


@pytest.fixture(scope="module")
def previewed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Previewed]:
    root = tmp_path_factory.mktemp("command") / "demo"
    write_project(root)
    document = yaml.safe_load((root / "pipelines" / f"{NAME}.yaml").read_text(encoding="utf-8"))
    server = start_server(load_project(root), port=0)
    try:
        found = Previewed(root, server, document)
        found.home = get_text(server, "/")
        found.before = listed(server)[NAME]
        body = json.dumps({"pipeline": NAME, "kind": "preview"}).encode("utf8")
        status, _ = request(
            server, "POST", "/api/run", body, **{"Content-Type": "application/json", "X-Run-Token": server.token}
        )
        found.accepted = status
        deadline = time.monotonic() + RUN_SECONDS
        while time.monotonic() < deadline:
            found.progress = get_json(server, f"/{NAME}/api/progress")
            if found.progress.get("state") in DONE:
                break
            time.sleep(0.05)
        yield found
    finally:
        server.shutdown()


class TestTheNewProjectIsListedAndNeverRun:
    def test_the_home_page_and_the_listing_name_the_example_and_it_has_never_run(self, previewed: Previewed) -> None:
        assert f'href="/{NAME}/"' in previewed.home
        assert previewed.before["last"] is None
        assert previewed.before["running"] is False
        assert previewed.before["problem"] is None
        assert previewed.before["steps"] == len(previewed.document["steps"])


class TestAPreviewOfTheExample:
    def test_the_preview_is_accepted_and_finishes(self, previewed: Previewed) -> None:
        assert previewed.accepted == 202
        assert previewed.progress["state"] == "finished", previewed.progress

    def test_the_listing_now_shows_a_finished_preview_that_held_back_the_writer(self, previewed: Previewed) -> None:
        last = listed(previewed.server)[NAME]["last"]
        assert (last["kind"], last["outcome"]) == ("preview", "finished")
        assert set(last["held_back"]) == {step["id"] for step in previewed.writers}
        assert last["steps_ran"] == len(previewed.document["steps"]) - len(previewed.writers)

    def test_the_writer_has_no_port_and_its_file_was_never_written(self, previewed: Previewed) -> None:
        ports = set(get_json(previewed.server, f"/{NAME}/api/ports")["ports"])
        writers = {step["id"] for step in previewed.writers}
        ran = {step["id"] for step in previewed.document["steps"]} - writers

        assert writers, "the example has no writer, so nothing here is held back"
        assert not {port for port in ports if port.split(".")[0] in writers}
        assert {port.split(".")[0] for port in ports} == ran, "a step that ran saved no port"
        for step in previewed.writers:
            assert not (previewed.root / step["config"]["path"]).exists()

    def test_every_port_serves_every_row_the_run_saved(self, previewed: Previewed) -> None:
        manifest = previewed.manifest
        assert set(get_json(previewed.server, f"/{NAME}/api/ports")["ports"]) == set(manifest)
        for port, entry in manifest.items():
            window = every_row(previewed.server, port)
            assert window["total"] == entry["rows"], port
            assert len(window["rows"]) == entry["rows"], port


class TestThePageAndItsPanels:
    def test_every_step_the_canvas_draws_is_a_step_of_the_file_and_has_a_panel(self, previewed: Previewed) -> None:
        page = get_text(previewed.server, f"/{NAME}/good/")
        drawn = drawn_steps(page)

        assert sorted(drawn) == sorted(step["id"] for step in previewed.document["steps"])
        for step in drawn:
            panel = get_text(previewed.server, f"/{NAME}/api/panel?step={quote(step)}")
            assert re.findall(r'<section class="stepdata" data-step="([^"]+)"', panel) == [step], step

    def test_every_table_a_panel_draws_names_a_saved_port_and_every_saved_port_is_drawn(
        self, previewed: Previewed
    ) -> None:
        drawn: set[str] = set()
        for step in previewed.document["steps"]:
            drawn |= table_ports(get_text(previewed.server, f"/{NAME}/api/panel?step={quote(step['id'])}"))
        saved = set(previewed.manifest)

        assert drawn <= saved, f"panels draw tables for {sorted(drawn - saved)}, which the run never saved"
        assert saved <= drawn, f"the run saved {sorted(saved - drawn)}, which no panel shows"

    def test_the_page_data_in_the_saved_page_is_the_run_record(self, previewed: Previewed) -> None:
        folder = load_project(previewed.root).runs_folder / NAME
        record = json.loads((folder / RUN_FILE).read_text(encoding="utf8"))
        saved = page_data((folder / PAGE_NAME).read_text(encoding="utf-8"))

        assert {key: saved.get(key) for key in record} == record
        assert page_data(get_text(previewed.server, f"/{NAME}/good/")) == saved

    def test_the_entity_picker_offers_every_customer_found_in_any_saved_port(self, previewed: Previewed) -> None:
        offered = get_json(previewed.server, f"/{NAME}/api/entities?limit=500")
        column = offered["label"]
        found: set[str] = set()
        for port in previewed.manifest:
            window = every_row(previewed.server, port)
            if column in window["columns"]:
                at = window["columns"].index(column)
                found |= {str(row[at]) for row in window["rows"] if row[at] is not None}

        assert found, f"no saved port carries {column!r}, so the picker has nothing to be checked against"
        assert set(offered["values"]) == found
        assert offered["total"] == len(offered["values"]) == len(found)
        totals = every_row(previewed.server, "by_customer.rows")
        in_totals = {str(row[totals["columns"].index(column)]) for row in totals["rows"]}
        assert found - in_totals, "every customer has a paid order, so the picker's reach past the totals is unproved"


# ---------------------------------------------------------------------------
# The real command, in a child process
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _lines(stream: Any, into: "queue.Queue[str]") -> None:
    for line in stream:
        into.put(line)


@pytest.mark.skipif(sys.platform == "win32", reason="ctrl-c is sent as SIGINT, which Windows does not deliver this way")
def test_view_serves_the_project_and_stops_cleanly_on_ctrl_c(tmp_path: Path) -> None:
    root = tmp_path / "demo"
    write_project(root)
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONUNBUFFERED": "1"}
    env.pop("ETL_PROJECT_ROOT", None)
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, "-m", "dantron_etl", "view", "--no-browser", "--port", str(port)],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    said: queue.Queue[str] = queue.Queue()
    threading.Thread(target=_lines, args=(process.stdout, said), daemon=True).start()
    try:
        printed: list[str] = []
        address = None
        deadline = time.monotonic() + RUN_SECONDS
        while address is None and time.monotonic() < deadline:
            try:
                line = said.get(timeout=0.5)
            except queue.Empty:
                assert process.poll() is None, f"the command exited early: {''.join(printed)}"
                continue
            printed.append(line)
            found = re.search(r"running at (\S+)", line)
            address = found.group(1) if found else None
        assert address is not None, f"the command never said where it was running: {''.join(printed)}"

        with urllib.request.urlopen(address, timeout=10) as answer:
            assert answer.status == 200
        with urllib.request.urlopen(address + "api/whoami", timeout=10) as answer:
            assert Path(json.loads(answer.read())["project"]).resolve() == root.resolve()

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=STOP_SECONDS) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
