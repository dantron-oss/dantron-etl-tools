"""The viewer is reachable only from this machine, and hands out only the files a page reads.

Every test here issues real HTTP requests against a real ``serve()`` on a port the operating system
picks. The source tree each test serves holds the things a project root really holds: a `.env`, a
`.git` directory, data files, and a symlink pointing outside it.
"""

import http.client
import json
import os
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from dantron_etl.preview import PANEL_DIR, PANEL_MANIFEST, persist_panels, persist_ports, serve

PIPELINE = b"name: Copy\nsteps:\n  - id: src\n    tool: read_csv\n    config: {path: in.csv}\n"


def _get(address: tuple[str, int], target: str) -> tuple[int, str, bytes]:
    """One GET, with the request target sent exactly as written: nothing re-encoded or tidied."""
    connection = http.client.HTTPConnection(*address, timeout=5)
    try:
        connection.request("GET", target)
        answer = connection.getresponse()
        return answer.status, answer.getheader("Content-Type") or "", answer.read()
    finally:
        connection.close()


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project root with the files a viewer may show and the files it must never show."""
    tmp_path = tmp_path_factory.mktemp("source")
    root = tmp_path / "project"
    (root / "pipelines" / "copy").mkdir(parents=True)
    (root / "pipelines" / "copy.yaml").write_bytes(PIPELINE)
    (root / "pipelines" / "copy" / "settings.yaml").write_text("rules: {}\n", encoding="utf-8")
    (root / "pipelines" / "copy" / "in.csv").write_text("account,amount\nA,1\n", encoding="utf-8")
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Read me\n", encoding="utf-8")
    (root / ".env").write_text("SECRET_TOKEN=do-not-serve\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[remote]\n\turl = do-not-serve\n", encoding="utf-8")
    (root / ".private").mkdir()
    (root / ".private" / "notes.md").write_text("do-not-serve\n", encoding="utf-8")
    (root / "key.pem").write_text("do-not-serve\n", encoding="utf-8")

    (tmp_path / "outside.yaml").write_text("do-not-serve: true\n", encoding="utf-8")
    (root / "pipelines" / "escape.yaml").symlink_to(tmp_path / "outside.yaml")
    (root / "pipelines" / "env.yaml").symlink_to(root / ".env")
    (root / "pipelines" / "hidden.md").symlink_to(root / ".private" / "notes.md")
    return root


@pytest.fixture(scope="module")
def served(project: Path) -> Iterator[tuple[str, int]]:
    server = serve(project / "runs", port=0, source_root=project)
    yield server.server_address[:2]
    server.shutdown()
    server.server_close()


def _assert_refused(status: int, kind: str, body: bytes) -> None:
    assert status in {403, 404}, f"expected a refusal, got {status}: {body[:200]!r}"
    assert kind.startswith("application/json")
    error = json.loads(body)["error"]
    assert error and "Traceback" not in error
    assert b"do-not-serve" not in body


def test_serve_listens_only_on_this_machine_unless_told_otherwise(tmp_path: Path) -> None:
    server = serve(tmp_path, port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.shutdown()
        server.server_close()


def test_serve_binds_the_host_it_is_given(tmp_path: Path) -> None:
    server = serve(tmp_path, host="0.0.0.0", port=0)
    try:
        assert server.server_address[0] == "0.0.0.0"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "path",
    ["pipelines/copy.yaml", "pipelines/copy/settings.yaml", "src/pkg/module.py", "README.md", "./README.md"],
)
def test_a_file_a_page_shows_is_served_byte_for_byte(served: tuple[str, int], project: Path, path: str) -> None:
    status, kind, body = _get(served, f"/api/source?path={path}")

    assert status == 200
    assert kind.startswith("text/plain")
    assert body == (project / path).read_bytes()


@pytest.mark.parametrize(
    "target",
    [
        "/api/source?path=.env",
        "/api/source?path=%2Eenv",
        "/api/source?path=.git/config",
        "/api/source?path=.git%2Fconfig",
        "/api/source?path=.private/notes.md",
        "/api/source?path=pipelines/../.env",
        "/api/source?path=pipelines%2F..%2F.env",
        "/api/source?path=%2e%2e/outside.yaml",
        "/api/source?path=../outside.yaml",
        "/api/source?path=pipelines/escape.yaml",
        "/api/source?path=pipelines/env.yaml",
        "/api/source?path=pipelines/hidden.md",
        "/api/source?path=key.pem",
        "/api/source?path=pipelines/copy/in.csv",
        "/api/source?path=/etc/passwd",
        "/api/source?path=%2Fetc%2Fpasswd",
        "/api/source?path=%252e%252e/outside.yaml",
        "/api/source?path=pipelines/nothing.yaml",
        "/api/source?path=README.md%00.py",
    ],
)
def test_anything_else_under_or_outside_the_root_is_refused_with_a_short_json_error(
    served: tuple[str, int], target: str
) -> None:
    _assert_refused(*_get(served, target))


def test_an_absolute_path_is_refused_even_when_it_lands_inside_the_root(served: tuple[str, int], project: Path) -> None:
    _assert_refused(*_get(served, f"/api/source?path={project / 'README.md'}"))


@pytest.fixture
def run(tmp_path: Path) -> Iterator[tuple[tuple[str, int], Path]]:
    """One saved run, and a secret beside the runs directory that its index files could point at."""
    runs = tmp_path / "runs"
    persist_ports({"read.rows": pl.DataFrame({"a": [1, 2]})}, runs / "flow")
    persist_panels({"read": "<p>read</p>"}, runs / "flow")
    (tmp_path / "secret.html").write_text("do-not-serve\n", encoding="utf-8")
    server = serve(runs, port=0)
    yield server.server_address[:2], runs / "flow"
    server.shutdown()
    server.server_close()


def test_a_panel_index_pointing_outside_its_run_is_refused(run: tuple[tuple[str, int], Path]) -> None:
    address, directory = run
    (directory / PANEL_MANIFEST).write_text(json.dumps({"read": "../../../secret.html"}), encoding="utf-8")

    status, _, body = _get(address, "/flow/api/panel?step=read")

    assert 400 <= status < 500
    assert b"do-not-serve" not in body


def test_a_panel_index_pointing_inside_its_run_is_still_served(run: tuple[tuple[str, int], Path]) -> None:
    address, directory = run

    status, _, body = _get(address, "/flow/api/panel?step=read")

    assert status == 200
    assert body == (directory / PANEL_DIR / "read.html").read_bytes()


def test_a_port_manifest_pointing_outside_its_run_is_refused(run: tuple[tuple[str, int], Path], tmp_path: Path) -> None:
    address, directory = run
    pl.DataFrame({"a": ["do-not-serve"]}).write_parquet(tmp_path / "elsewhere.parquet")
    manifest = json.loads((directory / "ports.json").read_text(encoding="utf-8"))
    manifest["read.rows"]["file"] = os.path.relpath(tmp_path / "elsewhere.parquet", directory)
    (directory / "ports.json").write_text(json.dumps(manifest), encoding="utf-8")

    status, _, body = _get(address, "/flow/api/rows?port=read.rows")

    assert 400 <= status < 500
    assert b"do-not-serve" not in body


@pytest.mark.parametrize("target", ["/%2e%2e/", "/../", "/..%2f..%2fsecret.html", "/flow/../../secret.html"])
def test_a_run_name_cannot_climb_out_of_the_runs_directory(run: tuple[tuple[str, int], Path], target: str) -> None:
    address, _ = run

    status, _, body = _get(address, target)

    assert status != 200 or b"do-not-serve" not in body
    assert b"do-not-serve" not in body
