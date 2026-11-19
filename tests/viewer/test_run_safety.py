"""Run safety in the viewer: what a confirmation covers, which run a page reads, and saved runs.

Each class builds a small project of its own in a scratch folder. The assertions compare what the
viewer says with what is on disk or with what an earlier answer said. They do not look for wording,
except where a finding is that a fact is not mentioned at all.
"""

from __future__ import annotations

import contextlib
import html
import http.client
import json
import re
import socket
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import polars as pl
import pytest
import yaml

from dantron_etl import ToolError, load_pipeline
from dantron_etl.focus import FocusSpec, RelatedKey
from dantron_etl.preview import MANIFEST, PortStore, persist_ports
from dantron_etl.spec import PortRef
from dantron_etl.viewer.project import load_project
from dantron_etl.viewer.runs import RUN_FILE, Write, execute, last_runs, writes, writes_digest
from dantron_etl.viewer.server import ViewerServer, start_server

NAME = "orders"
DATA = "pipelines/orders/orders.csv"
REDIRECTS = {301, 302, 303, 307, 308}


@pytest.fixture(autouse=True)
def _no_root_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ETL_PROJECT_ROOT", raising=False)


# ---------------------------------------------------------------------------
# A project on disk: read_csv -> filter -> write_csv, focused on party
# ---------------------------------------------------------------------------


def pipeline_text(read_config: str = f"path: {DATA}", predicate: str = "[amount] > 0") -> str:
    return (
        "name: Orders kept\n"
        "steps:\n"
        "  - id: read\n"
        "    tool: read_csv\n"
        f"    config: {{{read_config}}}\n"
        "  - id: keep\n"
        "    tool: filter\n"
        f"    config: {{predicate: '{predicate}'}}\n"
        "    inputs: {rows: read.rows}\n"
        "  - id: save\n"
        "    tool: write_csv\n"
        "    config: {path: kept.csv, overwrite: true}\n"
        "    inputs: {rows: keep.true}\n"
    )


def orders_csv(parties: int = 5) -> str:
    rows = ["party,amount"]
    for n in range(parties):
        rows += [f"P{101 + n},{10 + n}", f"P{101 + n},{-1 - n}", f"P{101 + n},5"]
    return "\n".join(rows) + "\n"


def make_project(root: Path, focus: str = "party") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "etl-project.yaml").write_text(f"pipelines:\n  {NAME}:\n    focus: {focus}\n", encoding="utf8")
    (root / "pipelines" / NAME).mkdir(parents=True, exist_ok=True)
    (root / "pipelines" / f"{NAME}.yaml").write_text(pipeline_text(), encoding="utf8")
    (root / DATA).write_text(orders_csv(), encoding="utf8")
    return root


def run(root: Path, kind: str = "preview", name: str = NAME) -> Any:
    project = load_project(root)
    return execute(project, project.pipelines[name], kind)


def break_data(root: Path) -> None:
    """The filter's column goes missing, so a run fails at step 'keep' after 'read' ran."""
    (root / DATA).write_text("party,qty\nP101,1\nP102,2\n", encoding="utf8")


def refuse_at_load(root: Path) -> None:
    """A setting the reader does not know: the pipeline is refused before any step runs."""
    (root / "pipelines" / f"{NAME}.yaml").write_text(
        pipeline_text(read_config=f"path: {DATA}, nonsense: 1"), encoding="utf8"
    )


# ---------------------------------------------------------------------------
# Talking to a server
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


def get(server: ViewerServer, target: str, headers: dict[str, str] | None = None) -> Answer:
    """One GET, redirects not followed."""
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        connection.request("GET", target, headers=headers or {})
        answer = connection.getresponse()
        return Answer(answer.status, {k.lower(): v for k, v in answer.getheaders()}, answer.read())
    finally:
        connection.close()


def follow(server: ViewerServer, target: str) -> tuple[str, Answer]:
    """A GET that follows redirects, as a browser would. Returns the final path and the answer."""
    for _ in range(5):
        answer = get(server, target)
        if answer.status not in REDIRECTS:
            return urlparse(target).path, answer
        target = urljoin(target, answer.headers["location"])
    raise AssertionError(f"too many redirects from {target}")


def get_json(server: ViewerServer, target: str) -> Any:
    answer = get(server, target)
    assert answer.status == 200, (target, answer.status, answer.text[:500])
    return answer.json()


def port_names(server: ViewerServer, target: str) -> set[str]:
    return set(get_json(server, target)["ports"])


def hrefs(page: str) -> list[str]:
    return [html.unescape(href) for href in re.findall(r'href="([^"]*)"', page)]


def linked_paths(page: str, base: str = "/") -> set[str]:
    return {urlparse(urljoin(f"http://127.0.0.1{base}", href)).path for href in hrefs(page)}


def home_row(page: str, name: str) -> str:
    rows = [row for row in re.split(r"<tr\b", page)[1:] if f"/{name}/" in linked_paths(row)]
    assert len(rows) == 1, f"one home table row for {name}"
    return rows[0]


@contextlib.contextmanager
def serving(root: Path) -> Iterator[ViewerServer]:
    server = start_server(load_project(root), port=0)
    try:
        yield server
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# 1. What a real run is confirmed against covers how destructive each write is
# ---------------------------------------------------------------------------

READ = {"id": "read", "tool": "read_csv", "config": {"path": "in.csv"}}


def excel_template(path: Path, tab: str) -> Path:
    import openpyxl

    book = openpyxl.Workbook()
    book.active.title = tab
    book.save(path)
    return path


def writer_flow(tool: str, config: dict[str, Any], inputs: dict[str, str]) -> Any:
    return load_pipeline(
        {"name": "a flow", "steps": [READ, {"id": "w", "tool": tool, "config": config, "inputs": inputs}]}
    )


def write_cases(tmp: Path) -> dict[str, tuple[str, dict[str, str], dict[str, Any], dict[str, Any]]]:
    """Two configurations of one writer step: same target, different consequence for what is there."""
    sheets = [{"port": "figures", "sheet": "Figures"}]
    excel = {"path": str(tmp / "pack.xlsx"), "sheets": sheets}
    table = {"connection": "TARGET_DB", "schema": "finance", "table": "sample", "mode": "replace"}

    def published(**extra: Any) -> dict[str, Any]:
        return {
            "connection": "TARGET_DB",
            "schema": "finance",
            "tables": [{"port": "totals", "table": "totals", "mode": "replace", **extra}],
        }

    first = excel_template(tmp / "first.xlsx", "Notes")
    second = excel_template(tmp / "second.xlsx", "Cover")
    return {
        "write_csv overwrite": (
            "write_csv",
            {"rows": "read.rows"},
            {"path": str(tmp / "a.csv"), "overwrite": False},
            {"path": str(tmp / "a.csv"), "overwrite": True},
        ),
        "write_table mode": ("write_table", {"rows": "read.rows"}, table, {**table, "mode": "append"}),
        "write_table allow_empty": ("write_table", {"rows": "read.rows"}, table, {**table, "allow_empty": True}),
        "write_excel overwrite_sheets": (
            "write_excel",
            {"figures": "read.rows"},
            excel,
            {**excel, "overwrite_sheets": True},
        ),
        "write_excel template": (
            "write_excel",
            {"figures": "read.rows"},
            {**excel, "template": str(first)},
            {**excel, "template": str(second)},
        ),
        "publish_tables allow_empty": (
            "publish_tables",
            {"totals": "read.rows"},
            published(),
            published(allow_empty=True),
        ),
        "publish_tables mode": ("publish_tables", {"totals": "read.rows"}, published(), published(mode="append")),
    }


CASE_NAMES = (
    "write_csv overwrite",
    "write_table mode",
    "write_table allow_empty",
    "write_excel overwrite_sheets",
    "write_excel template",
    "publish_tables allow_empty",
    "publish_tables mode",
)


class TestDigestCoversHowAWriteTreatsWhatIsThere:
    @pytest.mark.parametrize("case", CASE_NAMES)
    def test_digest_changes_when_a_setting_changes_and_the_target_does_not(self, tmp_path: Path, case: str) -> None:
        tool, inputs, before, after = write_cases(tmp_path)[case]
        found_before = writes(writer_flow(tool, before, inputs), tmp_path)
        found_after = writes(writer_flow(tool, after, inputs), tmp_path)

        assert [w.target for w in found_before] == [w.target for w in found_after], "the target is the same"
        assert writes_digest(found_before) != writes_digest(found_after), (
            f"{case}: a real run confirmed for one setting would run with the other",
            found_before,
            found_after,
        )

    def test_write_excel_detail_says_when_it_replaces_what_is_there(self, tmp_path: Path) -> None:
        tool, inputs, before, after = write_cases(tmp_path)["write_excel overwrite_sheets"]
        (kept,) = writes(writer_flow(tool, before, inputs), tmp_path)
        (replaced,) = writes(writer_flow(tool, after, inputs), tmp_path)
        assert isinstance(replaced, Write)
        assert replaced.detail != kept.detail
        assert re.search(r"overwrit|replac", replaced.detail, re.I), replaced.detail


# ---------------------------------------------------------------------------
# 2. A refused or failed attempt still points at the good run it kept
# ---------------------------------------------------------------------------


def good_then(root: Path, how: str) -> None:
    make_project(root)
    assert run(root).outcome == "finished"
    if how == "refused":
        refuse_at_load(root)
    else:
        break_data(root)
    assert run(root).outcome == how


class TestKeptGoodRunIsLinked:
    def test_after_a_refused_attempt_the_pipeline_page_links_the_good_run(self, tmp_path: Path) -> None:
        good_then(tmp_path, "refused")
        with serving(tmp_path) as server:
            at, answer = follow(server, f"/{NAME}/")
            assert answer.status == 200
            assert f"/{NAME}/good/" in linked_paths(answer.text, at), "no link to the good run that was kept"

    def test_after_a_failed_run_the_run_page_links_the_good_run(self, tmp_path: Path) -> None:
        good_then(tmp_path, "failed")
        with serving(tmp_path) as server:
            at, answer = follow(server, f"/{NAME}/")
            assert answer.status == 200
            assert f"/{NAME}/good/" in linked_paths(answer.text, at)

    @pytest.mark.parametrize("how", ["refused", "failed"])
    def test_home_row_links_the_good_run_kept_behind_a_bad_attempt(self, tmp_path: Path, how: str) -> None:
        good_then(tmp_path, how)
        with serving(tmp_path) as server:
            listed = {p["name"]: p for p in get_json(server, "/api/pipelines")["pipelines"]}
            assert listed[NAME]["last"]["previous_good"] is not None
            row = home_row(get(server, "/").text, NAME)
            assert f"/{NAME}/good/" in linked_paths(row), "the home row hides the good run that was kept"


# ---------------------------------------------------------------------------
# 3. A page keeps reading the run it was served for
# ---------------------------------------------------------------------------


class TestPageIsPinnedToOneRun:
    def test_pipeline_address_redirects_to_the_latest_slot_and_that_slot_keeps_answering(self, tmp_path: Path) -> None:
        good_then(tmp_path, "failed")
        with serving(tmp_path) as server:
            failed_ports = port_names(server, f"/{NAME}/failed/api/ports")
            good_ports = port_names(server, f"/{NAME}/good/api/ports")
            assert failed_ports != good_ports

            first = get(server, f"/{NAME}/")
            assert first.status in REDIRECTS, (
                f"/{NAME}/ answered {first.status}, so the page's api/ calls follow whatever run is latest"
            )
            served_at = urlparse(urljoin(f"/{NAME}/", first.headers["location"])).path
            assert served_at == f"/{NAME}/failed/"
            assert port_names(server, urljoin(served_at, "api/ports")) == failed_ports

            (tmp_path / DATA).write_text(orders_csv(), encoding="utf8")
            assert run(tmp_path).outcome == "finished"

            later = get(server, urljoin(served_at, "api/ports"))
            assert later.status == 404 or set(later.json()["ports"]) == failed_ports, (
                "a page served for the failed run is now reading another run"
            )
            again = get(server, f"/{NAME}/")
            assert again.status in REDIRECTS
            assert urlparse(urljoin(f"/{NAME}/", again.headers["location"])).path == f"/{NAME}/good/"

    def test_a_failed_slot_answers_its_own_run_while_a_newer_good_run_is_latest(self, tmp_path: Path) -> None:
        good_then(tmp_path, "failed")
        folder = load_project(tmp_path).runs_folder
        with serving(tmp_path) as server:
            failed_ports = port_names(server, f"/{NAME}/failed/api/ports")
            # The good run becomes the newer one while the failed slot is still there.
            record = json.loads((folder / NAME / RUN_FILE).read_text(encoding="utf8"))
            record["started"] = "2999-01-01T00:00:00+00:00"
            (folder / NAME / RUN_FILE).write_text(json.dumps(record), encoding="utf8")

            assert port_names(server, f"/{NAME}/failed/api/ports") == failed_ports
            moved = get(server, f"/{NAME}/")
            assert moved.status in REDIRECTS
            assert urlparse(urljoin(f"/{NAME}/", moved.headers["location"])).path == f"/{NAME}/good/"

    def test_a_pipeline_never_run_answers_its_page_without_a_redirect(self, tmp_path: Path) -> None:
        make_project(tmp_path)
        with serving(tmp_path) as server:
            answer = get(server, f"/{NAME}/")
            assert answer.status == 200
            assert answer.headers["content-type"].startswith("text/html")


# ---------------------------------------------------------------------------
# 4. and 6. Pipeline names the viewer cannot serve
# ---------------------------------------------------------------------------

PLAIN_FLOW = (
    "steps:\n"
    "  - id: read\n"
    "    tool: read_csv\n"
    "    config: {path: in.csv}\n"
    "  - id: save\n"
    "    tool: write_csv\n"
    "    config: {path: out.csv}\n"
    "    inputs: {rows: read.rows}\n"
)


class TestUnservableNames:
    @pytest.mark.parametrize("reserved", ["catalog", "api"])
    def test_a_pipeline_named_like_a_viewer_address_is_skipped_and_noted(self, tmp_path: Path, reserved: str) -> None:
        (tmp_path / "pipelines").mkdir()
        (tmp_path / "pipelines" / f"{reserved}.yaml").write_text(PLAIN_FLOW, encoding="utf8")
        (tmp_path / "pipelines" / "neighbour.yaml").write_text(PLAIN_FLOW, encoding="utf8")

        project = load_project(tmp_path)

        assert "neighbour" in project.pipelines
        assert reserved not in project.pipelines
        noted = [p for p in project.problems if f"{reserved}.yaml" in p]
        assert noted, project.problems
        assert any("reserved" in p.lower() for p in noted), noted

    def test_two_files_differing_only_by_case_list_the_first_and_note_the_other(self, tmp_path: Path) -> None:
        folder = tmp_path / "pipelines"
        folder.mkdir()
        (folder / "Orders.yaml").write_text(PLAIN_FLOW, encoding="utf8")
        (folder / "orders.yaml").write_text(PLAIN_FLOW, encoding="utf8")
        if len(list(folder.iterdir())) != 2:
            pytest.skip("this file system does not tell names apart by case")

        project = load_project(tmp_path)

        assert list(project.pipelines) == ["Orders"]
        assert any("orders.yaml" in p for p in project.problems), project.problems


# ---------------------------------------------------------------------------
# 5. Every entity the run lists can be followed, whatever the focus column's type
# ---------------------------------------------------------------------------

TYPED_CSV = (
    "party,day,moment,flag,share,count\n"
    "P1,2026-01-05,2026-01-05 10:30:00,true,0.1,7\n"
    "P2,2026-01-06,2026-01-06 08:00:00,false,2.0,-3\n"
    "P1,2026-01-05,2026-01-05 10:30:00,true,0.1,7\n"
    "P3,2026-02-01,2026-02-01 23:59:59,false,1234567.891,1000000\n"
    "P2,2026-02-02,2026-01-06 08:00:00,true,2.0,0\n"
    "P4,2026-03-03,2026-03-03 00:00:00,false,0.30000000000000004,-3\n"
)
TYPED_COLUMNS = {"day": "Date", "moment": "Datetime", "flag": "Boolean", "share": "Float64", "count": "Int64"}


def typed_flow() -> str:
    return yaml.safe_dump(
        {
            "name": "Typed",
            "steps": [
                {
                    "id": "read",
                    "tool": "read_csv",
                    "config": {
                        "path": "data/typed.csv",
                        "column_types": [
                            {"column": "day", "type": "Date"},
                            {"column": "moment", "type": "Datetime", "format": "%Y-%m-%d %H:%M:%S"},
                        ],
                    },
                },
                {
                    "id": "keep",
                    "tool": "filter",
                    "config": {"predicate": "[count] > -1000"},
                    "inputs": {"rows": "read.rows"},
                },
                {
                    "id": "save",
                    "tool": "write_csv",
                    "config": {"path": "output/typed.csv", "overwrite": True},
                    "inputs": {"rows": "keep.true"},
                },
            ],
        },
        sort_keys=False,
    )


@pytest.fixture(scope="module")
def typed(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ViewerServer]:
    root = tmp_path_factory.mktemp("typed")
    (root / "data").mkdir()
    (root / "output").mkdir()
    (root / "pipelines").mkdir()
    (root / "data" / "typed.csv").write_text(TYPED_CSV, encoding="utf8")
    settings = {"pipelines": {f"by_{column}": {"focus": column} for column in TYPED_COLUMNS}}
    (root / "etl-project.yaml").write_text(yaml.safe_dump(settings), encoding="utf8")
    for column in TYPED_COLUMNS:
        (root / "pipelines" / f"by_{column}.yaml").write_text(typed_flow(), encoding="utf8")
    for column in TYPED_COLUMNS:
        record = run(root, name=f"by_{column}")
        assert record.outcome == "finished", record.message
    with serving(root) as server:
        yield server


class TestEntitiesRoundTrip:
    @pytest.mark.parametrize("column", list(TYPED_COLUMNS))
    def test_every_listed_entity_matches_its_own_rows(self, typed: ViewerServer, column: str) -> None:
        name = f"by_{column}"
        listed = get_json(typed, f"/{name}/api/entities?q=&limit=500")
        dtypes = None
        for port in ("read.rows", "keep.true"):
            whole = get_json(typed, f"/{name}/api/rows?port={port}&limit=500")
            at = whole["columns"].index(column)
            dtypes = whole["dtypes"][at]
            followed = 0
            for value in listed["values"]:
                found = get_json(typed, f"/{name}/api/rows?port={port}&limit=500&focus={quote(value)}")
                assert found["matched"] >= 1, f"{column} ({dtypes}) value {value!r} listed but matches no row at {port}"
                assert len({row[at] for row in found["rows"]}) == 1, (value, found["rows"])
                followed += found["matched"]
            assert followed == whole["total"], f"the listed {column} values do not account for every row at {port}"
        assert listed["values"], f"no {column} values were listed"


# ---------------------------------------------------------------------------
# 7. A related-key port that cannot be narrowed is refused, not shown whole
# ---------------------------------------------------------------------------

LINES = pl.DataFrame({"party": ["P1", "P2", "P1"], "invoice": ["I1", "I2", "I3"]})
NO_PARTY = pl.DataFrame({"invoice": ["I1", "I2", "I3"], "amount": [1, 2, 3]})
INVOICES = pl.DataFrame({"invoice": ["I1", "I2", "I3"], "total": [10, 20, 30]})
NO_INVOICE = pl.DataFrame({"total": [10, 20, 30]})


@pytest.fixture
def related_store(tmp_path: Path) -> PortStore:
    persist_ports(
        {"lines.rows": LINES, "loose.rows": NO_PARTY, "invoices.rows": INVOICES, "bare.rows": NO_INVOICE}, tmp_path
    )
    return PortStore(tmp_path)


class TestRelatedKeyRefusals:
    @pytest.mark.parametrize(
        ("port", "origin", "key"),
        [
            ("invoices.rows", "loose.rows", "invoice"),  # the origin names no party
            ("invoices.rows", "lines.rows", "number"),  # the key is missing at the origin
            ("bare.rows", "lines.rows", "invoice"),  # the key is missing at the port itself
        ],
    )
    def test_a_via_port_that_cannot_be_narrowed_refuses_naming_the_port(
        self, related_store: PortStore, port: str, origin: str, key: str
    ) -> None:
        from dantron_etl.viewer.focus_rows import focus_match

        step, _, name = port.partition(".")
        from_step, _, from_name = origin.partition(".")
        focus = FocusSpec(
            label="party",
            identity=("party",),
            via={PortRef(step, name): RelatedKey(PortRef(from_step, from_name), key)},
        )
        with pytest.raises((ToolError, ValueError), match=re.escape(port)):
            outcome = focus_match(related_store, focus, port, "P1")
            pytest.fail(f"returned {outcome!r}, which the viewer shows as the whole port")


# ---------------------------------------------------------------------------
# 8. Leftovers of an interrupted save
# ---------------------------------------------------------------------------


class TestInterruptedSave:
    def test_a_good_run_moved_aside_with_nothing_put_back_is_still_found(self, tmp_path: Path) -> None:
        make_project(tmp_path)
        saved = run(tmp_path)
        folder = load_project(tmp_path).runs_folder
        (folder / NAME).rename(folder / f".{NAME}.old-0123456789ab")

        good, failed = last_runs(folder, NAME)

        assert failed is None
        assert good is not None, "the only good run is on disk but reads as never run"
        assert (good.outcome, good.started) == ("finished", saved.started)

    def test_stale_incoming_folders_are_cleared_by_the_next_run(self, tmp_path: Path) -> None:
        make_project(tmp_path)
        folder = load_project(tmp_path).runs_folder
        folder.mkdir(parents=True)
        stale = [folder / f".{NAME}.incoming-{tag}" for tag in ("aaaaaaaaaaaa", "bbbbbbbbbbbb")]
        for one in stale:
            one.mkdir()
            (one / MANIFEST).write_text("{}", encoding="utf8")
        other = folder / ".another.incoming-cccccccccccc"
        other.mkdir()

        assert run(tmp_path).outcome == "finished"

        assert [one for one in stale if one.exists()] == []
        assert other.is_dir(), "another pipeline's folder is not this run's to clear"


# ---------------------------------------------------------------------------
# 9. A saved run is followed by the focus it was saved with
# ---------------------------------------------------------------------------


class TestSavedFocus:
    def test_changing_the_focus_setting_does_not_change_what_a_saved_run_matches_on(self, tmp_path: Path) -> None:
        make_project(tmp_path, focus="party")
        assert run(tmp_path).outcome == "finished"
        (tmp_path / "etl-project.yaml").write_text(f"pipelines:\n  {NAME}:\n    focus: amount\n", encoding="utf8")

        with serving(tmp_path) as server:
            found = get_json(server, f"/{NAME}/good/api/rows?port=read.rows&limit=500&focus=P101")
            at = found["columns"].index("party")
            assert found["matched"] == 3, "the run was saved following party; it is now matched on another column"
            assert {row[at] for row in found["rows"]} == {"P101"}


# ---------------------------------------------------------------------------
# 10. Finding a viewer already running for this project
# ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class TestChoosePort:
    def test_a_viewer_for_this_project_on_a_spare_port_is_reused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dantron_etl import cli

        make_project(tmp_path)
        with serving(tmp_path) as server, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            preferred = int(taken.getsockname()[1])
            monkeypatch.setattr(cli, "SPARE_PORTS", range(server.port, server.port + 1))

            assert cli.choose_port(tmp_path, preferred=preferred) == (server.port, True)

    def test_the_whoami_probe_ignores_proxy_settings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from dantron_etl import cli

        dead_proxy = f"http://127.0.0.1:{free_port()}"
        for variable in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(variable, raising=False)
        for variable in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            monkeypatch.setenv(variable, dead_proxy)
        monkeypatch.setattr(cli, "SPARE_PORTS", range(0))
        # urlopen keeps the proxy settings it saw first; start from none, as a fresh command does.
        monkeypatch.setattr(urllib.request, "_opener", None)

        make_project(tmp_path)
        with serving(tmp_path) as server:
            assert cli.choose_port(tmp_path, preferred=server.port) == (server.port, True)


# ---------------------------------------------------------------------------
# 11. The plan loads the pipeline, which may run project code: it needs the token
# ---------------------------------------------------------------------------


class TestPlanNeedsToken:
    def test_plan_without_the_run_token_is_refused(self, tmp_path: Path) -> None:
        make_project(tmp_path)
        with serving(tmp_path) as server:
            assert get(server, f"/{NAME}/api/plan").status == 403
            assert get(server, f"/{NAME}/api/plan", {"X-Run-Token": "not-the-token"}).status == 403

    def test_plan_with_the_run_token_is_answered(self, tmp_path: Path) -> None:
        make_project(tmp_path)
        with serving(tmp_path) as server:
            answer = get(server, f"/{NAME}/api/plan", {"X-Run-Token": server.token})
            assert answer.status == 200
            assert answer.json()["digest"]
