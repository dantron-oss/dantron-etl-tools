"""Save a finished run's ports to disk and serve them a window of rows at a time.

``persist_trace(trace, directory)`` writes every kept port to Parquet, with each port's per-row
status beside it, and ``persist_panels`` writes each step's panel. ``PortStore(directory)`` reads
them back: ``window(port, offset=, limit=, filters=)`` returns up to 500 rows, after filtering the
whole port. ``serve(root)`` is a small local web server over every run saved under ``root``.

A page draws only a few rows per port, so this is what lets a reader search all of them.

Why: docs/design/loading-and-running.md#preview
"""

from __future__ import annotations

import html
import json
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, urlparse

import polars as pl

from .status import STATUS_LEVELS

if TYPE_CHECKING:  # pragma: no cover - a type only
    from .trace import PipelineTrace

#: The most rows one request may ask for. For everything, read the Parquet file.
MAX_WINDOW = 500

#: The prefix a per-column filter arrives under: ``f.account_code=2-5100``.
FILTER_PREFIX = "f."

MANIFEST = "ports.json"

#: Where each step's panel (what a reader sees on opening the step) is written, and the index of
#: which file belongs to which step. A step id is looked up in the index, never turned into a path.
PANEL_MANIFEST = "panels.json"
PANEL_DIR = "panels"

#: The kinds of file the source route serves. Everything else (data, keys, workbooks) is refused.
SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md"})


def _inside(directory: Path, name: str) -> Path:
    """Return ``directory / name``. Refuse a name that leads outside ``directory``.

    The names come from the run's own index files, which someone could have edited.
    """
    landed = (directory / name).resolve()
    if not landed.is_relative_to(directory.resolve()):
        raise PreviewError(f"{name!r} is outside the run's directory, so it is not served")
    return landed


#: The name the per-row status travels under beside a port's rows. Lengthened with ``_`` if the
#: port already has a column of that name.
STATUS_COLUMN = "status"


def _beside(columns: list[str]) -> str:
    """Return a name for the status column that no column of the port already uses."""
    name = STATUS_COLUMN
    while name in columns:
        name = "_" + name
    return name


class PreviewError(ValueError):
    """A request the store cannot answer: an unknown port, step or column, or a bad window."""


@dataclass(frozen=True)
class Window:
    """One window of a port: the rows asked for, as text, with the counts that place them.

    - ``rows``: every value as text (or ``None``), with each column's type in ``dtypes``.
    - ``total``: rows in the whole port. ``matched``: rows left after the filters.
    - ``status``: one mark per row of this window, in the same order, or ``None`` when no step
      said anything about this port.

    Why: docs/design/loading-and-running.md#the-status-travels-with-the-rows
    """

    columns: tuple[str, ...]
    dtypes: tuple[str, ...]
    rows: tuple[tuple[str | None, ...], ...]
    total: int
    matched: int
    offset: int
    status: tuple[dict[str, Any], ...] | None = None


def persist_ports(
    frames: dict[str, pl.DataFrame],
    directory: Path,
    *,
    status: dict[str, pl.DataFrame] | None = None,
) -> dict[str, Any]:
    """Write each frame in ``frames`` to Parquet under ``directory``, and return the manifest.

    ``frames`` is keyed by ``"step.port"``. ``status``, keyed the same way, is each port's per-row
    status; it goes in a file of its own beside the port. A port with no status gets no status
    file. The manifest (``ports.json``) lists each port's file, row count, columns and types.

    Why: docs/design/loading-and-running.md#status-is-saved-beside-the-port
    """
    directory.mkdir(parents=True, exist_ok=True)
    marked = status or {}
    manifest: dict[str, Any] = {}
    for key, frame in frames.items():
        stem = key.replace(".", "__")
        name = stem + ".parquet"
        frame.write_parquet(directory / name)
        manifest[key] = {
            "file": name,
            "rows": frame.height,
            "columns": list(frame.columns),
            "dtypes": [str(d) for d in frame.dtypes],
        }
        marks = marked.get(key)
        if marks is not None:
            status_name = stem + "__status.parquet"
            marks.write_parquet(directory / status_name)
            manifest[key]["status"] = {"file": status_name, "rows": marks.height}
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf8")
    return manifest


def persist_trace(trace: PipelineTrace, directory: Path) -> dict[str, Any]:
    """Write a finished run to ``directory``: every port it kept, and the per-row status of each.

    Use this rather than :func:`persist_ports`, so the statuses cannot be forgotten. Returns the
    manifest.
    """
    return persist_ports(
        {f"{ref.step}.{ref.port}": frame for ref, frame in trace.frames.items()},
        directory,
        status={f"{ref.step}.{ref.port}": marks for ref, marks in trace.status.items()},
    )


def persist_panels(panels: dict[str, str], directory: Path) -> dict[str, str]:
    """Write each step's panel markup to its own file under ``directory``, and return the index.

    ``panels`` maps a step id to its HTML. The index (``panels.json``) maps each step id to its
    file name.

    Why: docs/design/loading-and-running.md#panels-are-separate-files
    """
    directory.mkdir(parents=True, exist_ok=True)
    into = directory / PANEL_DIR
    into.mkdir(exist_ok=True)
    index: dict[str, str] = {}
    for step, markup in panels.items():
        # Percent-encoded, so any step id becomes one file name in this directory.
        name = quote(step, safe="") + ".html"
        (into / name).write_text(markup, encoding="utf-8")
        index[step] = name
    (directory / PANEL_MANIFEST).write_text(json.dumps(index, indent=2), encoding="utf8")
    return index


class PortStore:
    """The saved ports of one run, read a window at a time.

    ``PortStore(directory)`` refuses a directory with no ``ports.json``. It keeps no frame in
    memory: each request reads the Parquet file lazily.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        if not (self.directory / MANIFEST).is_file():
            raise PreviewError(f"no run has been written to {self.directory}: {MANIFEST} is not there")
        self._manifest: dict[str, Any] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        """Return the run's port index, read on first use and kept."""
        if self._manifest is None:
            self._manifest = json.loads((self.directory / MANIFEST).read_text(encoding="utf8"))
        return self._manifest

    def ports(self) -> dict[str, Any]:
        return self.manifest

    def panels(self) -> dict[str, str]:
        """Return the index of steps that have a panel. A run with no panels gives an empty dict."""
        index = self.directory / PANEL_MANIFEST
        return json.loads(index.read_text(encoding="utf8")) if index.is_file() else {}

    def panel(self, step: str) -> str:
        """Return the panel markup written for ``step``, exactly as written.

        Refuses (:class:`PreviewError`) a step with no panel, listing the steps that have one.
        """
        written = self.panels()
        name = written.get(step)
        if name is None:
            have = ", ".join(sorted(written)) or "no panels at all"
            raise PreviewError(f"no panel for step {step!r} in this run; it has: {have}")
        return _inside(self.directory / PANEL_DIR, name).read_text(encoding="utf-8")

    def window(
        self,
        port: str,
        *,
        offset: int = 0,
        limit: int = 100,
        filters: dict[str, str] | None = None,
        level: str | None = None,
        match: Mapping[str, Collection[str]] | None = None,
    ) -> Window:
        """Return up to ``limit`` rows of ``port`` from ``offset``, after narrowing, as a :class:`Window`.

        - ``filters``: ``{column: text}``. Keeps rows whose value, as shown, contains the text.
        - ``match``: ``{column: values}``. Keeps rows whose value, as text, is exactly one of them.
        - ``level``: keeps rows the step marked at that status level.

        All narrowings apply together to the same rows. Refuses (:class:`PreviewError`) an unknown
        port or column, a negative offset, a limit outside 1 to 500, an unknown level, and a level
        on a port with no status.

        Why: docs/design/loading-and-running.md#searching-a-saved-run
        """
        entry = self.manifest.get(port)
        if entry is None:
            raise PreviewError(f"no port {port!r} in this run; it has: {', '.join(sorted(self.manifest))}")
        if offset < 0:
            raise PreviewError(f"offset must not be negative; got {offset}")
        if not 1 <= limit <= MAX_WINDOW:
            raise PreviewError(f"limit must be between 1 and {MAX_WINDOW}; got {limit}")

        columns: list[str] = entry["columns"]
        wanted = {name: text for name, text in (filters or {}).items() if text.strip()}
        exact = {name: [str(text) for text in values] for name, values in (match or {}).items()}
        unknown = sorted((set(wanted) | set(exact)) - set(columns))
        if unknown:
            raise PreviewError(f"port {port!r} carries no column {unknown[0]!r}; it has: {', '.join(columns)}")

        marks = entry.get("status")
        if level is not None:
            if level not in STATUS_LEVELS:
                raise PreviewError(f"{level!r} is not a level; a level is one of: {', '.join(STATUS_LEVELS)}")
            if marks is None:
                raise PreviewError(
                    f"no step said anything about the rows of port {port!r}, so it cannot be narrowed to {level!r}"
                )

        scan = pl.scan_parquet(_inside(self.directory, entry["file"]))
        beside = _beside(columns)
        if marks is not None:
            # The status and the port must have the same row count; a mismatch would pad with nulls.
            if int(marks["rows"]) != int(entry["rows"]):
                raise PreviewError(
                    f"port {port!r} holds {entry['rows']:,} row(s) and the status beside it {marks['rows']:,}; "
                    "the two line up row for row, and position is the whole of what joins them"
                )

            # The marks ride beside the rows as one column, so every slice and filter keeps them
            # lined up. They are joined by position: there is no key to join on.
            scan = pl.concat(
                [scan, pl.scan_parquet(_inside(self.directory, marks["file"])).select(pl.struct(pl.all()).alias(beside))],
                how="horizontal",
            )
            if level is not None:
                scan = scan.filter(pl.col(beside).struct.field("level") == level)

        for name, text in wanted.items():
            # Match the text the page shows: 2026 in a date column means the text, not a number.
            scan = scan.filter(pl.col(name).cast(pl.String, strict=False).str.contains(text, literal=True))
        for name, texts in exact.items():
            scan = scan.filter(pl.col(name).cast(pl.String, strict=False).is_in(texts))

        narrowed = bool(wanted) or bool(exact) or level is not None
        matched = int(scan.select(pl.len()).collect().item()) if narrowed else int(entry["rows"])
        page = scan.slice(offset, limit).collect()
        # Only the port's own columns, so the status column is not shown as data.
        as_text = page.select(pl.col(name).cast(pl.String, strict=False) for name in columns)

        return Window(
            columns=tuple(columns),
            dtypes=tuple(entry["dtypes"]),
            rows=tuple(tuple(row) for row in as_text.iter_rows()),
            total=int(entry["rows"]),
            matched=matched,
            offset=offset,
            status=tuple(page.get_column(beside).to_list()) if marks is not None else None,
        )


PAGE_NAME = "page.html"

_TITLE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Run:
    """One saved run: its name, its directory, its page (if any) and its title."""

    name: str
    directory: Path
    page: Path | None
    title: str


def discover(root: Path) -> list[Run]:
    """Return every run saved under ``root``: one per folder that holds a ``ports.json``."""
    root = Path(root)
    runs: list[Run] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        if not (directory / MANIFEST).is_file():
            continue
        page = directory / PAGE_NAME
        found = _TITLE.search(page.read_text(encoding="utf-8")[:8192]) if page.is_file() else None
        runs.append(
            Run(
                name=directory.name,
                directory=directory,
                page=page if page.is_file() else None,
                title=(found.group(1).strip() if found else directory.name),
            )
        )
    return runs


def _index(runs: list[Run]) -> bytes:
    """Return the HTML page listing the runs, served at the root."""
    items = "".join(
        f'<li><a href="/{run.name}/">{html.escape(run.title)}</a>'
        f"<span>{len(PortStore(run.directory).ports())} ports</span></li>"
        for run in runs
    )
    body = items or "<li>nothing has been written here yet</li>"
    return (
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Pipeline runs</title><style>"
        "body{font:15px/1.5 system-ui,sans-serif;max-width:44rem;margin:3rem auto;padding:0 1.25rem;"
        "background:#f4f4f1;color:#16181b}"
        "@media(prefers-color-scheme:dark){body{background:#131618;color:#e8eae6}}"
        "h1{font-size:1.3rem;letter-spacing:-.01em}ul{list-style:none;padding:0}"
        "li{display:flex;justify-content:space-between;gap:1rem;padding:.7rem 0;"
        "border-bottom:1px solid #dfdfd9}"
        "@media(prefers-color-scheme:dark){li{border-bottom-color:#2a2f34}}"
        "a{color:#0d6e70;text-decoration:none;font-weight:600}a:hover{text-decoration:underline}"
        "@media(prefers-color-scheme:dark){a{color:#46b3b0}}"
        "span{color:#5c6470;font-size:.85rem}</style>"
        f"<h1>Pipeline runs</h1><ul>{body}</ul>"
    ).encode()


def _source_file(source_root: Path | None, query: dict[str, list[str]]) -> tuple[int, Path | str]:
    """Return the source file a page asked for, or the HTTP status and sentence refusing it.

    A file is served only when the path is relative, resolves (following ``..`` and links) to a
    file inside the source root, has no part starting with a dot (no ``.env``, nothing in
    ``.git``), and has one of the :data:`SOURCE_SUFFIXES`. A refused kind of file is 403; a file
    that is not there is 404.
    """
    if source_root is None:
        return 404, "this service was not given a source root, so it serves no code"
    asked = (query.get("path") or [""])[0]
    if not asked:
        return 400, "name the file to read, as 'path' relative to the source root"
    if "\x00" in asked:
        return 404, "the path holds a null character, so no file has that name"

    path = Path(asked)
    allowed = ", ".join(sorted(SOURCE_SUFFIXES))
    if path.is_absolute() or path.anchor:
        return 403, f"{asked!r} is absolute; give the path relative to the source root"
    landed = (source_root / path).resolve()
    if not landed.is_relative_to(source_root):
        return 403, f"{asked!r} leads outside the source root; only files under it are served"
    if any(part.startswith(".") for part in landed.relative_to(source_root).parts):
        return 403, f"{asked!r} leads to a hidden file or folder, which is not served"
    if landed.suffix.lower() not in SOURCE_SUFFIXES:
        return 403, f"{asked!r} is not a kind of file this route serves; it serves: {allowed}"
    if not landed.is_file():
        return 404, f"{asked!r} is not a file under the source root"
    return 200, landed


def _handler(root: Path, source_root: Path | None) -> type[BaseHTTPRequestHandler]:
    """Return the request handler. It reads ``root`` on each request, so a new run shows without a restart."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            """Print nothing per request."""

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode("utf8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802 -- the name the base class dispatches on
            parsed = urlparse(self.path)
            runs = {run.name: run for run in discover(root)}
            rest = parsed.path.strip("/")

            if rest in {"", "index.html"}:
                self._send(200, _index(list(runs.values())), "text/html; charset=utf-8")
                return

            if parsed.path == "/api/source":
                status, found = _source_file(source_root, parse_qs(parsed.query))
                if isinstance(found, Path):
                    self._send(200, found.read_bytes(), "text/plain; charset=utf-8")
                else:
                    self._json(status, {"error": found})
                return

            name, _, tail = rest.partition("/")
            run = runs.get(name)
            if run is None:
                self._json(404, {"error": f"no run {name!r}; this service has: {', '.join(sorted(runs)) or 'none'}"})
                return

            # Redirect to the trailing slash, or the page's relative requests miss the run.
            if not parsed.path.endswith("/") and tail == "":
                self.send_response(301)
                self.send_header("Location", f"/{name}/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if run.page is not None and tail in {"", "index.html", run.page.name}:
                self._send(200, run.page.read_bytes(), "text/html; charset=utf-8")
                return
            if tail == "api/ports":
                self._json(200, {"ports": PortStore(run.directory).ports()})
                return
            if tail == "api/rows":
                try:
                    self._json(200, self._window(run, parse_qs(parsed.query)))
                except PreviewError as refusal:
                    self._json(400, {"error": str(refusal)})
                return
            if tail == "api/panel":
                # Sent as HTML: the page puts it straight into the pane.
                try:
                    asked = (parse_qs(parsed.query).get("step") or [""])[0]
                    self._send(200, PortStore(run.directory).panel(asked).encode("utf-8"), "text/html; charset=utf-8")
                except PreviewError as refusal:
                    self._json(400, {"error": str(refusal)})
                return
            serves = f"/{name}/, /{name}/api/ports, /{name}/api/rows?port=..., /{name}/api/panel?step=..."
            self._json(404, {"error": f"no route {parsed.path!r}; run {name!r} serves: {serves}"})

        def _window(self, run: Run, query: dict[str, list[str]]) -> dict[str, Any]:
            port = (query.get("port") or [""])[0]
            try:
                offset = int((query.get("offset") or ["0"])[0])
                limit = int((query.get("limit") or ["100"])[0])
            except ValueError as bad:
                raise PreviewError(f"offset and limit must be whole numbers: {bad}") from bad

            filters = {
                name[len(FILTER_PREFIX) :]: values[0]
                for name, values in query.items()
                if name.startswith(FILTER_PREFIX) and values
            }
            # The level is its own parameter, not ``f.level``, because a port may have a column
            # called ``level``. Why: docs/design/loading-and-running.md#searching-a-saved-run
            level = (query.get("level") or [""])[0] or None
            found = PortStore(run.directory).window(port, offset=offset, limit=limit, filters=filters, level=level)
            return {
                "columns": list(found.columns),
                "dtypes": list(found.dtypes),
                "rows": [list(row) for row in found.rows],
                "total": found.total,
                "matched": found.matched,
                "offset": found.offset,
                "status": list(found.status) if found.status is not None else None,
            }

    return Handler


def serve(
    root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8899,
    source_root: Path | None = None,
) -> ThreadingHTTPServer:
    """Start a web server for every run saved under ``root``, each at ``/<folder name>/``.

    Returns the server, already listening on a background thread; call ``shutdown()`` to stop it.

    - ``host``: ``127.0.0.1`` by default, so only this machine can reach it. ``0.0.0.0`` lets
      every machine on the network read every row; a container publishing its port needs that.
    - ``port``: the port to listen on.
    - ``source_root``: if given, ``/api/source?path=`` serves YAML, Python and Markdown files under
      it, never a hidden file. See :func:`_source_file`.

    There is no login and no encryption. Anything that can reach the port can read every row.

    Why: docs/design/loading-and-running.md#a-local-viewer-with-no-login
    """
    server = ThreadingHTTPServer(
        (host, port), _handler(Path(root), Path(source_root).resolve() if source_root else None)
    )
    threading.Thread(target=server.serve_forever, name="etl-preview", daemon=True).start()
    return server
