"""The project server: viewer/server.py.

Every test here sends real HTTP requests to a real server started on a port the operating system
picks. Each class builds its own small project in a scratch folder (CSV in, filter, group by party,
write a CSV), so one class's runs never show up in another's answers.

The assertions compare what the server says with what is on disk: the pipeline files, the saved
run's ports.json, the CSV that was or was not written. They do not look for wording.
"""

from __future__ import annotations

import html
import http.client
import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import pytest
import yaml

from dantron_etl.preview import MANIFEST
from dantron_etl.viewer.project import load_project
from dantron_etl.viewer.server import ViewerServer, start_server

PARTIES = ("North Alpha", "South Alpha", "Beta", "Gamma", "Delta", "Epsilon")
DONE = {"finished", "failed", "refused"}
WAIT_SECONDS = 120.0


# ---------------------------------------------------------------------------
# Building a project on disk
# ---------------------------------------------------------------------------


def orders_csv() -> str:
    """Three rows per party: one positive amount, one negative, one of 5."""
    rows = ["party,amount"]
    for n, party in enumerate(PARTIES):
        rows += [f"{party},{10 + n}", f"{party},{-1 - n}", f"{party},5"]
    return "\n".join(rows) + "\n"


def rows_per_party() -> dict[str, dict[str, int]]:
    """How many rows each port holds for one party, worked out from ``orders_csv`` by hand."""
    return {
        "read.rows": {p: 3 for p in PARTIES},
        "keep.true": {p: 2 for p in PARTIES},
        "totals.rows": {p: 1 for p in PARTIES},
    }


def target_of(root: Path, name: str) -> Path:
    return root / "output" / f"{name}_totals.csv"


def orders_flow(
    root: Path, name: str, *, predicate: str = "[amount] > ${rules.min_amount}", target: Path | None = None
) -> str:
    """read_csv -> filter -> group_by -> write_csv -> assert_rows (downstream of the writer)."""
    out = (target or target_of(root, name)).as_posix()
    return (
        f"name: Totals for {name}\n"
        "steps:\n"
        "  - id: read\n"
        "    tool: read_csv\n"
        f"    config: {{path: pipelines/{name}/orders.csv}}\n"
        "  - id: keep\n"
        "    tool: filter\n"
        f'    config: {{predicate: "{predicate}"}}\n'
        "    inputs: {rows: read.rows}\n"
        "  - id: totals\n"
        "    tool: group_by\n"
        "    config:\n"
        "      by: [party]\n"
        "      aggregations:\n"
        '        - {name: total, function: sum, value: "[amount]"}\n'
        "        - {name: lines, function: count}\n"
        "    inputs: {rows: keep.true}\n"
        "  - id: save\n"
        "    tool: write_csv\n"
        f'    config: {{path: "{out}", overwrite: true}}\n'
        "    inputs: {rows: totals.rows}\n"
        "  - id: audit\n"
        "    tool: assert_rows\n"
        '    config: {predicate: "[total] > 0", label: positive}\n'
        "    inputs: {rows: save.rows}\n"
    )


def slow_flow(root: Path, name: str, filters: int = 150) -> str:
    """A long chain of filters over a big file: a preview takes seconds, not milliseconds."""
    steps: list[dict[str, Any]] = [
        {"id": "read", "tool": "read_csv", "config": {"path": f"pipelines/{name}/orders.csv"}}
    ]
    previous = "read.rows"
    for n in range(filters):
        steps.append(
            {
                "id": f"f{n}",
                "tool": "filter",
                "config": {"predicate": f"[amount] > {-1000 - n}"},
                "inputs": {"rows": previous},
            }
        )
        previous = f"f{n}.true"
    steps.append(
        {
            "id": "save",
            "tool": "write_csv",
            "config": {"path": target_of(root, name).as_posix(), "overwrite": True},
            "inputs": {"rows": previous},
        }
    )
    return yaml.safe_dump({"name": "A slow flow", "steps": steps}, sort_keys=False)


def slow_csv(rows: int = 400_000) -> str:
    return "party,amount\n" + "\n".join(f"{PARTIES[n % len(PARTIES)]},{n % 97 - 10}" for n in range(rows)) + "\n"


def add_pipeline(root: Path, name: str, text: str, data: str | None = None) -> None:
    (root / "pipelines").mkdir(parents=True, exist_ok=True)
    (root / "pipelines" / f"{name}.yaml").write_text(text, encoding="utf8")
    if data is not None:
        (root / "pipelines" / name).mkdir(parents=True, exist_ok=True)
        (root / "pipelines" / name / "orders.csv").write_text(data, encoding="utf8")


def make_project(
    root: Path, *, ruled: tuple[str, ...] = (), plain: tuple[str, ...] = (), needs_rules: bool = False
) -> Path:
    """``ruled`` pipelines read ``${rules.min_amount}`` set inline; ``plain`` ones use no rules.

    Every pipeline is focused on ``party``. ``needs_rules`` adds a pipeline using
    ``${rules.cutoff}`` that no settings mention.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(exist_ok=True)
    settings: dict[str, Any] = {"pipelines": {}}
    for name in ruled:
        settings["pipelines"][name] = {"rules": {"min_amount": 0}, "focus": "party"}
        add_pipeline(root, name, orders_flow(root, name), orders_csv())
    for name in plain:
        settings["pipelines"][name] = {"focus": "party"}
        add_pipeline(root, name, orders_flow(root, name, predicate="[amount] > 0"), orders_csv())
    if needs_rules:
        add_pipeline(
            root, "needs_rules", orders_flow(root, "needs_rules", predicate="[amount] > ${rules.cutoff}"), orders_csv()
        )
    (root / "etl-project.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf8")
    return root


def pipeline_files(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted((root / "pipelines").glob("*.yaml"))}


def file_document(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf8"))


def wiring(path: Path) -> set[tuple[str, str, str]]:
    """(source step, reading step, source port) for every input in the pipeline file."""
    found: set[tuple[str, str, str]] = set()
    for step in file_document(path)["steps"]:
        for ref in (step.get("inputs") or {}).values():
            source, _, port = str(ref).partition(".")
            found.add((source, step["id"], port))
    return found


def saved_manifest(root: Path, name: str, slot: str = "") -> dict[str, Any]:
    folder = load_project(root).runs_folder / f"{name}{slot}"
    return json.loads((folder / MANIFEST).read_text(encoding="utf8"))


# ---------------------------------------------------------------------------
# Talking to the server
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def text(self) -> str:
        return self.body.decode("utf8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.body)


def call(
    server: ViewerServer, method: str, target: str, *, body: bytes | None = None, headers: dict[str, str] | None = None
) -> Answer:
    """One request, with the target sent exactly as written."""
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        connection.request(method, target, body=body, headers=headers or {})
        answer = connection.getresponse()
        return Answer(answer.status, {k.lower(): v for k, v in answer.getheaders()}, answer.read())
    finally:
        connection.close()


def get(server: ViewerServer, target: str) -> Answer:
    return call(server, "GET", target)


def get_json(server: ViewerServer, target: str) -> Any:
    answer = get(server, target)
    assert answer.status == 200, (target, answer.status, answer.text[:500])
    return answer.json()


def get_plan(server: ViewerServer, name: str) -> Answer:
    """The plan loads the pipeline, so it is asked for with the run token, as the page does."""
    return call(server, "GET", f"/{name}/api/plan", headers={"X-Run-Token": server.token})


def plan_json(server: ViewerServer, name: str) -> Any:
    answer = get_plan(server, name)
    assert answer.status == 200, (name, answer.status, answer.text[:500])
    return answer.json()


def post_run(server: ViewerServer, payload: dict[str, Any]) -> Answer:
    headers = {"Content-Type": "application/json", "X-Run-Token": server.token}
    return call(server, "POST", "/api/run", body=json.dumps(payload).encode("utf8"), headers=headers)


def progress(server: ViewerServer, name: str) -> dict[str, Any]:
    return get_json(server, f"/{name}/api/progress")


def wait_done(server: ViewerServer, name: str) -> dict[str, Any]:
    deadline = time.monotonic() + WAIT_SECONDS
    seen: dict[str, Any] = {}
    while time.monotonic() < deadline:
        seen = progress(server, name)
        if seen.get("state") in DONE:
            return seen
        time.sleep(0.05)
    raise AssertionError(f"run of {name!r} did not end within {WAIT_SECONDS}s; last progress: {seen}")


def run_preview(server: ViewerServer, name: str) -> dict[str, Any]:
    answer = post_run(server, {"pipeline": name, "kind": "preview"})
    assert answer.status == 202, (answer.status, answer.text[:500])
    return wait_done(server, name)


def listing(server: ViewerServer) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in get_json(server, "/api/pipelines")["pipelines"]}


def port_names(payload: Any) -> set[str]:
    ports = payload.get("ports", payload) if isinstance(payload, dict) else payload
    if isinstance(ports, dict):
        return set(ports)
    return {p if isinstance(p, str) else (p.get("port") or p.get("name")) for p in ports}


def rows_window(server: ViewerServer, name: str, port: str, **query: str) -> dict[str, Any]:
    extra = "".join(f"&{key}={quote(value)}" for key, value in query.items())
    return get_json(server, f"/{name}/api/rows?port={quote(port)}{extra}")


def entities(server: ViewerServer, name: str, q: str = "", limit: int | None = None) -> dict[str, Any]:
    tail = f"&limit={limit}" if limit is not None else ""
    return get_json(server, f"/{name}/api/entities?q={quote(q)}{tail}")


def page_data(page: str) -> dict[str, Any]:
    found = re.search(r'<script\b[^>]*\bid="page-data"[^>]*>(.*?)</script>', page, re.S)
    assert found, "the page carries no page-data script"
    return json.loads(found.group(1))


def find_key(data: Any, key: str) -> Any:
    """The first value stored under ``key`` at any depth, or a KeyError."""
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for value in data.values():
            try:
                return find_key(value, key)
            except KeyError:
                pass
    elif isinstance(data, list):
        for value in data:
            try:
                return find_key(value, key)
            except KeyError:
                pass
    raise KeyError(key)


def edges(page: str) -> set[tuple[str, str, str]]:
    found: set[tuple[str, str, str]] = set()
    for tag in re.findall(r"<g\b[^>]*>", page):
        attrs = {k: html.unescape(v) for k, v in re.findall(r'([\w-]+)="([^"]*)"', tag)}
        if "edge" in attrs.get("class", "").split():
            found.add((attrs["data-from"], attrs["data-to"], attrs["data-port"]))
    return found


def linked_paths(page: str) -> set[str]:
    return {
        urlparse(urljoin("http://127.0.0.1/", html.unescape(href))).path for href in re.findall(r'href="([^"]*)"', page)
    }


@pytest.fixture(autouse=True)
def _no_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ETL_PROJECT_ROOT", raising=False)


@dataclass
class Served:
    root: Path
    server: ViewerServer
    notes: dict[str, Any]


def serve(root: Path) -> Iterator[Served]:
    server = start_server(load_project(root), port=0)
    try:
        yield Served(root, server, {})
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Listing pipelines: nothing has run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def listed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    root = make_project(
        tmp_path_factory.mktemp("listed") / "project", ruled=("orders",), plain=("ledger",), needs_rules=True
    )
    yield from serve(root)


class TestListing:
    def test_pipeline_names_are_the_yaml_files_on_disk(self, listed: Served) -> None:
        assert set(listing(listed.server)) == set(pipeline_files(listed.root))

    def test_each_pipeline_reports_the_number_of_steps_in_its_file(self, listed: Served) -> None:
        found = listing(listed.server)
        for name, path in pipeline_files(listed.root).items():
            assert found[name]["steps"] == len(file_document(path)["steps"]), name

    def test_a_pipeline_never_run_has_no_last_run_and_is_not_running(self, listed: Served) -> None:
        for name, entry in listing(listed.server).items():
            assert entry["last"] is None, name
            assert entry["running"] is False, name
            assert progress(listed.server, name)["state"] == "idle"

    def test_rules_needed_names_what_each_file_uses(self, listed: Served) -> None:
        found = listing(listed.server)
        assert found["orders"]["rules_needed"] == ["min_amount"]
        assert found["ledger"]["rules_needed"] == []
        assert found["needs_rules"]["rules_needed"] == ["cutoff"]
        assert found["orders"]["rules_source"] is not None
        assert found["needs_rules"]["rules_source"] is None

    def test_a_file_added_while_serving_appears_on_the_next_request_and_goes_when_removed(self, listed: Served) -> None:
        added = listed.root / "pipelines" / "added.yaml"
        add_pipeline(listed.root, "added", orders_flow(listed.root, "added", predicate="[amount] > 0"))
        try:
            found = listing(listed.server)
            assert found["added"]["steps"] == 5
            assert found["added"]["last"] is None
            assert "/added/" in linked_paths(get(listed.server, "/").text)
        finally:
            added.unlink()
        assert "added" not in listing(listed.server)

    def test_home_page_names_and_links_every_pipeline(self, listed: Served) -> None:
        answer = get(listed.server, "/")
        assert answer.status == 200
        assert answer.headers["content-type"].startswith("text/html")
        links = linked_paths(answer.text)
        for name in pipeline_files(listed.root):
            assert f"/{name}/" in links, name
        assert "/catalog/" in links
        catalog = get(listed.server, "/catalog/")
        assert catalog.status == 200
        assert catalog.headers["content-type"].startswith("text/html")

    def test_a_never_run_pipeline_row_offers_a_run(self, listed: Served) -> None:
        page = get(listed.server, "/").text
        rows = re.split(r"<tr\b", page)[1:]
        for name in pipeline_files(listed.root):
            mine = [row for row in rows if f"/{name}/" in linked_paths(row)]
            assert len(mine) == 1, f"one table row for {name}"
            assert "<button" in mine[0], name
            assert "never run" in html.unescape(mine[0]).lower(), name

    def test_whoami_names_the_project(self, listed: Served) -> None:
        assert get_json(listed.server, "/api/whoami") == {"app": "dantron-etl viewer", "project": str(listed.root)}

    def test_the_plan_lists_what_a_real_run_would_write(self, listed: Served) -> None:
        plan = plan_json(listed.server, "orders")
        assert plan["steps"] == 5
        assert plan["preview"] == {"runs": ["read", "keep", "totals"], "held_back": ["save", "audit"]}
        assert [(w["step"], w["kind"]) for w in plan["writes"]] == [("save", "file")]
        assert Path(plan["writes"][0]["target"]) == target_of(listed.root, "orders")
        assert re.fullmatch(r"[0-9a-f]{64}", plan["digest"])
        assert plan["problem"] is None

    def test_a_pipeline_needing_rules_says_what_to_add_to_the_settings(self, listed: Served) -> None:
        plan = get_plan(listed.server, "needs_rules").text
        page = html.unescape(get(listed.server, "/needs_rules/").text)
        for told in (plan, page):
            assert "etl-project.yaml" in told
            assert "cutoff" in told


# ---------------------------------------------------------------------------
# A preview run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def previewed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    root = make_project(tmp_path_factory.mktemp("previewed") / "project", ruled=("orders",))
    for served in serve(root):
        answer = post_run(served.server, {"pipeline": "orders", "kind": "preview"})
        served.notes["status"] = answer.status
        served.notes["progress"] = wait_done(served.server, "orders") if answer.status == 202 else None
        yield served


class TestPreviewRun:
    def test_the_run_is_accepted_and_its_progress_ends_finished(self, previewed: Served) -> None:
        assert previewed.notes["status"] == 202
        ended = previewed.notes["progress"]
        assert ended["state"] == "finished"
        assert ended["kind"] == "preview"
        assert {event["step"] for event in ended["events"]} == {"read", "keep", "totals"}

    def test_the_last_run_is_a_finished_preview_that_held_back_the_writer(self, previewed: Served) -> None:
        last = listing(previewed.server)["orders"]["last"]
        assert (last["kind"], last["outcome"]) == ("preview", "finished")
        assert set(last["held_back"]) == {"save", "audit"}
        assert last["steps_ran"] == 3
        assert last["failed_step"] is None
        assert not target_of(previewed.root, "orders").exists()
        assert list((previewed.root / "output").iterdir()) == []

    def test_ports_name_nothing_from_the_writer_or_downstream_of_it(self, previewed: Served) -> None:
        names = port_names(get_json(previewed.server, "/orders/api/ports"))
        assert names == set(saved_manifest(previewed.root, "orders"))
        assert {"read.rows", "keep.true", "totals.rows"} <= names
        assert not {name for name in names if name.split(".")[0] in {"save", "audit"}}

    def test_every_saved_port_is_fetchable_with_the_manifest_row_count(self, previewed: Served) -> None:
        manifest = saved_manifest(previewed.root, "orders")
        assert manifest
        for port, entry in manifest.items():
            window = rows_window(previewed.server, "orders", port)
            assert window["total"] == entry["rows"], port

    def test_the_drawn_graph_equals_the_wiring_in_the_file(self, previewed: Served) -> None:
        page = get(previewed.server, "/orders/good/").text
        assert edges(page) == wiring(previewed.root / "pipelines" / "orders.yaml")

    def test_entities_lists_every_party_and_a_search_can_match_nothing_or_be_capped(self, previewed: Served) -> None:
        found = entities(previewed.server, "orders")
        assert found["label"] == "party"
        assert found["total"] == found["matched"] == len(PARTIES)
        assert set(found["values"]) == set(PARTIES)
        none = entities(previewed.server, "orders", "zzz")
        assert (none["total"], none["matched"], none["values"]) == (len(PARTIES), 0, [])
        capped = entities(previewed.server, "orders", "alpha", limit=1)
        assert capped["matched"] == 2
        assert len(capped["values"]) == 1

    @pytest.mark.parametrize("q", ["alpha", "ALPHA", "Alpha"])
    def test_entity_search_is_a_case_insensitive_substring(self, previewed: Served, q: str) -> None:
        found = entities(previewed.server, "orders", q)
        assert found["total"] == len(PARTIES)
        assert found["matched"] == 2
        assert set(found["values"]) == {"North Alpha", "South Alpha"}

    @pytest.mark.parametrize("port", ["read.rows", "keep.true", "totals.rows"])
    def test_rows_for_one_entity_are_only_that_entitys_rows(self, previewed: Served, port: str) -> None:
        party = "South Alpha"
        window = rows_window(previewed.server, "orders", port, focus=party)
        at = window["columns"].index("party")
        assert window["rows"]
        assert all(str(row[at]) == party for row in window["rows"])
        assert window["matched"] == rows_per_party()[port][party]
        assert window["matched"] < window["total"]

    def test_the_page_carries_the_run_record(self, previewed: Served) -> None:
        data = page_data(get(previewed.server, "/orders/good/").text)
        assert set(find_key(data, "held_back")) == {"save", "audit"}
        assert find_key(data, "outcome") == "finished"
        good = page_data(get(previewed.server, "/orders/good/").text)
        assert good == data


# ---------------------------------------------------------------------------
# A pipeline that keeps no rows: save_rows: false
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def rowless(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    root = make_project(tmp_path_factory.mktemp("rowless") / "project", ruled=("orders",))
    settings = yaml.safe_load((root / "etl-project.yaml").read_text(encoding="utf8"))
    (root / "etl-project.yaml").write_text(yaml.safe_dump({"save_rows": False, **settings}), encoding="utf8")
    for served in serve(root):
        served.notes["progress"] = run_preview(served.server, "orders")
        yield served


class TestAPipelineThatKeepsNoRows:
    def test_the_run_finishes_and_every_port_still_has_its_count_and_columns(self, rowless: Served) -> None:
        assert rowless.notes["progress"]["state"] == "finished"
        ports = get_json(rowless.server, "/orders/api/ports")["ports"]
        assert ports["read.rows"]["rows"] == 3 * len(PARTIES)
        assert ports["totals.rows"]["columns"] == ["party", "total", "lines"]

    def test_asking_for_rows_is_refused_in_words_that_name_the_setting(self, rowless: Served) -> None:
        answer = get(rowless.server, "/orders/api/rows?port=read.rows")

        assert answer.status == 409, answer.text
        assert "save_rows" in answer.json()["error"] and "etl-project.yaml" in answer.json()["error"]
        assert not any(party in answer.text for party in PARTIES)

    def test_following_one_party_says_why_it_cannot(self, rowless: Served) -> None:
        found = entities(rowless.server, "orders")

        assert found["values"] == [] and "save_rows" in found["reason"]
        page = get(rowless.server, "/orders/good/").text
        assert 'class="pick"' not in page and "save_rows" in page

    def test_each_port_shows_its_columns_and_the_reason_and_no_grid_that_would_fetch_rows(self, rowless: Served) -> None:
        panel = get(rowless.server, "/orders/api/panel?step=totals")

        assert panel.status == 200
        assert "save_rows" in panel.text and "<th>total<em>" in panel.text
        assert "data-lazy" not in panel.text and "data-port" not in panel.text
        assert not any(party in panel.text for party in PARTIES)


# ---------------------------------------------------------------------------
# A failing run after a good one
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def failed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    root = make_project(tmp_path_factory.mktemp("failed") / "project", ruled=("orders",), needs_rules=True)
    add_pipeline(
        root,
        "broken",
        orders_flow(root, "broken", predicate="[amount] > 0").replace("broken/orders.csv", "broken/missing.csv"),
    )
    for served in serve(root):
        notes = served.notes
        notes["good_progress"] = run_preview(served.server, "orders")
        notes["good_started"] = listing(served.server)["orders"]["last"]["started"]
        notes["good_ports"] = port_names(get_json(served.server, "/orders/api/ports"))
        notes["good_manifest"] = saved_manifest(root, "orders")
        (root / "pipelines" / "orders" / "orders.csv").write_text("party,qty\nBeta,1\nGamma,2\n", encoding="utf8")
        notes["failed_progress"] = run_preview(served.server, "orders")
        notes["broken_progress"] = run_preview(served.server, "broken")
        notes["rules_progress"] = run_preview(served.server, "needs_rules")
        yield served


class TestFailedRun:
    def test_progress_ends_failed_naming_the_step_and_the_real_cause(self, failed: Served) -> None:
        assert failed.notes["good_progress"]["state"] == "finished"
        ended = failed.notes["failed_progress"]
        assert (ended["state"], ended["failed_step"]) == ("failed", "keep")
        assert "amount" in ended["message"]

    def test_the_last_run_failed_and_points_back_at_the_good_one(self, failed: Served) -> None:
        last = listing(failed.server)["orders"]["last"]
        assert (last["outcome"], last["failed_step"]) == ("failed", "keep")
        assert "amount" in last["message"]
        assert last["previous_good"] == failed.notes["good_started"]

    def test_the_good_slot_keeps_the_earlier_run_and_the_failed_slot_what_ran_before_failing(
        self, failed: Served
    ) -> None:
        assert port_names(get_json(failed.server, "/orders/good/api/ports")) == failed.notes["good_ports"]
        for port, entry in failed.notes["good_manifest"].items():
            window = get_json(failed.server, f"/orders/good/api/rows?port={quote(port)}")
            assert window["total"] == entry["rows"], port
        assert port_names(get_json(failed.server, "/orders/failed/api/ports")) == {"read.rows"}

    def test_the_pipeline_page_shows_the_failure(self, failed: Served) -> None:
        page = get(failed.server, "/orders/failed/").text
        data = page_data(page)
        assert find_key(data, "outcome") == "failed"
        assert find_key(data, "failed_step") == "keep"
        assert "keep" in page
        assert page_data(get(failed.server, "/orders/failed/").text) == data
        assert find_key(page_data(get(failed.server, "/orders/good/").text), "outcome") == "finished"

    def test_a_pipeline_that_never_succeeded_has_no_previous_good_run(self, failed: Served) -> None:
        ended = failed.notes["broken_progress"]
        assert (ended["state"], ended["failed_step"]) == ("failed", "read")
        assert "missing.csv" in ended["message"]
        last = listing(failed.server)["broken"]["last"]
        assert (last["outcome"], last["previous_good"]) == ("failed", None)

    def test_a_pipeline_needing_rules_with_none_set_is_refused(self, failed: Served) -> None:
        ended = failed.notes["rules_progress"]
        assert ended["state"] == "refused"
        assert "cutoff" in ended["message"] and "etl-project.yaml" in ended["message"]
        assert listing(failed.server)["needs_rules"]["last"]["outcome"] == "refused"
        assert not target_of(failed.root, "needs_rules").exists()


# ---------------------------------------------------------------------------
# A real run, and the refusals around it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def real(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    root = make_project(tmp_path_factory.mktemp("real") / "project", ruled=("orders", "ledger", "moving"))
    yield from serve(root)


def nothing_ran(served: Served, name: str) -> None:
    assert not target_of(served.root, name).exists()
    assert not (load_project(served.root).runs_folder / name).exists()
    assert progress(served.server, name)["state"] == "idle"
    assert listing(served.server)[name]["last"] is None


class TestRealRun:
    def test_a_confirmed_real_run_writes_its_target(self, real: Served) -> None:
        digest = plan_json(real.server, "orders")["digest"]
        answer = post_run(real.server, {"pipeline": "orders", "kind": "real", "confirm": "orders", "digest": digest})
        assert answer.status == 202
        assert wait_done(real.server, "orders")["state"] == "finished"
        written = target_of(real.root, "orders").read_text(encoding="utf8").splitlines()
        assert written[0].split(",") == ["party", "total", "lines"]
        assert len(written) - 1 == len(PARTIES)
        last = listing(real.server)["orders"]["last"]
        assert (last["kind"], last["outcome"], last["held_back"]) == ("real", "finished", [])

    @pytest.mark.parametrize(
        ("change", "refused_with"),
        [
            ({"confirm": "orders"}, {400, 409}),
            ({"confirm": "LEDGER"}, {400, 409}),
            ({"confirm": None}, {400, 409}),
            ({"digest": "0" * 64}, {409}),
            ({"digest": None}, {400, 409}),
        ],
        ids=["another-name", "different-case", "no-confirm", "wrong-digest", "no-digest"],
    )
    def test_a_real_run_without_the_right_confirmation_writes_nothing(
        self, real: Served, change: dict[str, Any], refused_with: set[int]
    ) -> None:
        digest = plan_json(real.server, "ledger")["digest"]
        payload: dict[str, Any] = {"pipeline": "ledger", "kind": "real", "confirm": "ledger", "digest": digest}
        payload.update(change)
        payload = {key: value for key, value in payload.items() if value is not None}

        answer = post_run(real.server, payload)

        assert answer.status in refused_with
        time.sleep(0.2)
        nothing_ran(real, "ledger")

    def test_a_stale_digest_is_refused_with_the_fresh_plan(self, real: Served) -> None:
        stale = plan_json(real.server, "moving")["digest"]
        moved = real.root / "output" / "moved_elsewhere.csv"
        (real.root / "pipelines" / "moving.yaml").write_text(
            orders_flow(real.root, "moving", target=moved), encoding="utf8"
        )
        fresh = plan_json(real.server, "moving")["digest"]
        assert fresh != stale

        answer = post_run(real.server, {"pipeline": "moving", "kind": "real", "confirm": "moving", "digest": stale})

        assert answer.status == 409
        assert find_key(answer.json(), "digest") == fresh
        time.sleep(0.2)
        assert not moved.exists()
        nothing_ran(real, "moving")


# ---------------------------------------------------------------------------
# One run at a time per pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def busy(tmp_path: Path) -> Iterator[Served]:
    root = make_project(tmp_path / "project", plain=("ledger",))
    add_pipeline(root, "slow", slow_flow(root, "slow"), slow_csv())
    yield from serve(root)


@pytest.mark.slow
class TestOneRunAtATime:
    """The slow flow takes a few seconds, far longer than the requests made while it runs."""

    def test_a_second_run_of_the_same_pipeline_is_refused_while_the_first_goes(self, busy: Served) -> None:
        assert post_run(busy.server, {"pipeline": "slow", "kind": "preview"}).status == 202
        again = post_run(busy.server, {"pipeline": "slow", "kind": "preview"})
        assert again.status == 409
        assert listing(busy.server)["slow"]["running"] is True
        assert wait_done(busy.server, "slow")["state"] == "finished"

    def test_another_pipeline_may_run_meanwhile_and_the_same_one_again_afterwards(self, busy: Served) -> None:
        assert post_run(busy.server, {"pipeline": "slow", "kind": "preview"}).status == 202
        assert post_run(busy.server, {"pipeline": "ledger", "kind": "preview"}).status == 202
        assert wait_done(busy.server, "ledger")["state"] == "finished"
        assert wait_done(busy.server, "slow")["state"] == "finished"
        assert listing(busy.server)["slow"]["running"] is False
        assert run_preview(busy.server, "slow")["state"] == "finished"
