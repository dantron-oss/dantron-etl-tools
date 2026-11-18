"""Serving a finished run's ports a window at a time, so a reader can look at all of it.

A rendered page draws a fixed number of rows per port, because a page that draws them all is a page
no browser will lay out -- the twelve ports of one real flow came to 113 MB of plain text. That cap
is honest about what it shows, but it puts every question of the form *does any row anywhere carry
this* out of reach: a search over the drawn rows answers a question about the page rather than about
the data.

So the rows are written down once, beside the page, and served a window at a time. Filtering happens
against the whole port and only the window crosses to the browser, which means the memory a reader
costs does not grow with the size of the flow -- the same reason the runner leaves a step's work to
the query engine rather than gathering rows itself.

Three decisions worth stating, because each has a quieter alternative that is wrong:

**A filter is a substring, matched against the column's text, and never an expression.** This library
already has one expression language with two compile targets, and every check it makes -- column
references, types, refusals -- exists because config is a file a non-developer edits. A search box is
typed into casually by somebody looking at a page, so it gets the construct with no failure modes:
text that is either in a value or is not. A reader who needs a real predicate has Filter, which is a
step, which is checked, and which is part of the flow rather than part of the looking.

**Everything crosses as text, with the dtype named separately.** A date, a decimal and a large
integer each lose something different on the way through JSON, and a page that renders them is going
to render text regardless. Converting once, here, beats three separate roundings nobody chose.

**A port that was released is absent rather than empty.** The runner drops a frame nothing will read
again when it is told to, and reporting that as a port with no rows would be indistinguishable from a
step that produced none -- which is the difference between a quiet day and a lost figure.

Imports the standard library's ``json`` rather than ``orjson``, against this repository's usual
preference, and the reason is what this package is for: it is imported BY other projects, so every
module-level import here is one those projects are made to carry. The rule is polars plus the
standard library at import time, with everything else loaded inside the function that needs it --
which is why :func:`~.loader.read_pipeline` imports ``yaml`` inside itself, and why a
database driver is reached for by URI scheme rather than declared. Measured cost of keeping it:
parsing a 625 KB run manifest takes 4.6 ms rather than 3.0, on a path that is read lazily and so is
touched once a page rather than once a panel.
"""

from __future__ import annotations

import html
import json
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, quote, urlparse

import polars as pl

from .status import STATUS_LEVELS

if TYPE_CHECKING:  # pragma: no cover -- the trace is a type here and never a runtime dependency
    from .trace import PipelineTrace

#: The most rows one request may ask for. A window is for looking at; a caller wanting everything
#: wants the Parquet file, which is sitting right there.
MAX_WINDOW = 500

#: The prefix a per-column filter arrives under: ``f.account_code=2-5100``.
FILTER_PREFIX = "f."

MANIFEST = "ports.json"

#: Where a step's panel -- the markup a reader is shown when they open that step -- is written, and
#: the index naming which file belongs to which step. An index rather than a naming convention the
#: reader of a request could re-derive: a step id arrives in a query string, and anything that turns
#: one into a path is a path a caller writes. Looked up, an unknown step is a refusal before any
#: filename exists.
PANEL_MANIFEST = "panels.json"
PANEL_DIR = "panels"

#: The kinds of file the source route hands out: pipeline and settings YAML, Python source and
#: Markdown. Everything else under the source root -- data, keys, workbooks -- is refused.
SOURCE_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md"})


def _inside(directory: Path, name: str) -> Path:
    """``directory / name``, refused unless it resolves to a file inside ``directory``.

    The names come from a run's own index files. Those are written by this module, but they are
    files on disk, and one edited or copied from elsewhere must not point the viewer at the rest of
    the machine.
    """
    landed = (directory / name).resolve()
    if not landed.is_relative_to(directory.resolve()):
        raise PreviewError(f"{name!r} is outside the run's directory, so it is not served")
    return landed


#: The name a port's marks travel under while they are being sliced beside its rows. A port is a
#: client's data and may well carry a column called ``status`` of its own, so the name is checked
#: against the port and lengthened until it is free rather than assumed to be: a collision here
#: would silently overwrite one of the columns the reader came to read.
STATUS_COLUMN = "status"


def _beside(columns: list[str]) -> str:
    """A name for the marks that no column of this port answers to."""
    name = STATUS_COLUMN
    while name in columns:
        name = "_" + name
    return name


class PreviewError(ValueError):
    """A request this store cannot honour, named so a caller can tell it from a bug."""


@dataclass(frozen=True)
class Window:
    """One answer: the rows asked for, the two counts needed to say where they came from, and what
    the step that made them said about each one.

    ``status`` is one mark per row of *this window*, in the same order, or ``None`` for a port no
    step said anything about. ``None`` rather than a tuple of ``ok``s, because the absence is what a
    viewer reads to decide whether the port has a status column to draw at all -- and almost every
    port in a real flow degraded nothing, so a column of ticks on all of them is a column a reader
    learns to skip, which costs the flagged ports the attention this channel exists to buy.

    Sliced with the rows and never re-derived afterwards: the correspondence is positional, so a
    window that took the first ``limit`` marks for the rows at some later offset would draw a status
    column that is right on the first page and, on every page after it, puts a warning on a row that
    never had one.
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
    """Write every retained port to Parquet under ``directory`` and return the manifest.

    Keyed by ``step.port``, the same spelling the page and the query string use, so no reader has to
    know how a :class:`~.spec.PortRef` is put together to ask for one.

    A port's per-row status is written to a file of its own beside the port rather than as extra
    columns inside it, for the same reason it is not a column in the frame: the port file is what a
    reader is shown as the data, and somebody scrolling it must see what the step produced and
    nothing the step said about it. Its own manifest entry, so the viewer can scan the two lazily
    and side by side -- they are joined by row position and by nothing else, which is why the row
    count is written down here where it can be checked against the port's own.

    A port with nothing to say gets no file and no entry, and the absence is what a viewer reads to
    decide whether the port has a status column to draw at all: an empty file per port would cost a
    pointless read apiece to be told nothing. Having no status and saying so are the same run --
    ``status={}`` is a complete run rather than a refusal, the ruling ``run_pipeline`` already takes
    on an empty cache, because a caller assembling statuses from whatever its run produced should
    not have to write the conditional itself.
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
    """Write one run to ``directory``: every port it retained, and every mark its steps made.

    Both halves off the one object, because they are one run and the writer is where they come
    apart. A caller assembling the frames by hand and calling :func:`persist_ports` has to remember
    the statuses separately -- they sit on the same trace, one attribute away -- and a directory
    written without them loads, serves and draws exactly like a run whose steps degraded nothing.
    The difference is invisible until a reader asks which rows a warning covers and the run cannot
    say. Taking the trace makes forgetting impossible rather than unlikely.

    A status is written only for a port whose frame was retained, which falls out of ``persist_ports``
    keying the two together: marks kept beside rows nobody can read are row references with nothing
    to point at. A run that degraded nothing writes no status files at all.
    """
    return persist_ports(
        {f"{ref.step}.{ref.port}": frame for ref, frame in trace.frames.items()},
        directory,
        status={f"{ref.step}.{ref.port}": marks for ref, marks in trace.status.items()},
    )


def persist_panels(panels: dict[str, str], directory: Path) -> dict[str, str]:
    """Write one file per step's panel under ``directory`` and return the index of them.

    Beside the ports rather than inside them, because a panel is the same kind of thing: written
    once when the run is written, and handed over when a reader asks for that one step. The page
    used to carry all 672 of them, which was 7.76 MB of an 8.98 MB document laid out before the
    browser could paint anything at all.

    One file each rather than one document holding every panel: the service reads a run's directory
    per request, and a single file would be read whole -- megabytes of six hundred other steps --
    to answer for one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    into = directory / PANEL_DIR
    into.mkdir(exist_ok=True)
    index: dict[str, str] = {}
    for step, markup in panels.items():
        # Percent-encoded, so every character a step id may carry -- including the separators that
        # would otherwise spell a directory -- lands as one file in one directory.
        name = quote(step, safe="") + ".html"
        (into / name).write_text(markup, encoding="utf-8")
        index[step] = name
    (directory / PANEL_MANIFEST).write_text(json.dumps(index, indent=2), encoding="utf8")
    return index


class PortStore:
    """The persisted ports of one run, answered a window at a time.

    Holds no frame in memory: every request scans the Parquet lazily, so a port of five million rows
    costs the same as one of five hundred, and two readers asking different questions do not queue
    behind one another's data.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        if not (self.directory / MANIFEST).is_file():
            raise PreviewError(f"no run has been written to {self.directory}: {MANIFEST} is not there")
        self._manifest: dict[str, Any] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        """The run's ports, read the first time one is asked about and kept for the rest of the store.

        Checked for at construction and read here, rather than both at construction: a mistyped
        directory still refuses immediately -- that is the whole of what the check is for -- but a
        store built to hand over one step's panel no longer parses an index of every port in the
        run to do it. On the real flow that manifest is 625 KB, which is 4.6 ms of parsing per
        request to answer a question it has nothing to do with. The same argument, on the other
        index, is written at :meth:`panels`.
        """
        if self._manifest is None:
            self._manifest = json.loads((self.directory / MANIFEST).read_text(encoding="utf8"))
        return self._manifest

    def ports(self) -> dict[str, Any]:
        return self.manifest

    def panels(self) -> dict[str, str]:
        """Which steps this run wrote a panel for. Absent is empty, not an error.

        Every run written before panels existed is a directory with no index in it, and a reader
        asking one of those for a step has asked for something this run does not have -- which is a
        fact about the run. Read per call rather than in ``__init__`` so that a store built for a
        window does not pay for an index it will not open.
        """
        index = self.directory / PANEL_MANIFEST
        return json.loads(index.read_text(encoding="utf8")) if index.is_file() else {}

    def panel(self, step: str) -> str:
        """The markup written for ``step``, exactly as it was written.

        Refused by name, in the shape every other refusal here takes, because the alternatives are
        both worse in the same direction: an empty answer draws a blank pane over a step that has
        rows behind it, and a missing file thrown out of a request handler is a stack trace where a
        sentence belongs. The index is also the only thing consulted -- a step id is a caller's
        text, and nothing here joins it to a path before it has been found to be a step.
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
    ) -> Window:
        """``limit`` rows of ``port`` from ``offset``, after every filter has been applied.

        Refusals name what was asked for: a port that was never written, a column that port does not
        carry, a window that runs backwards. Each is a caller's mistake rather than an empty answer,
        and answering nothing would read exactly like a port that matched nothing.

        ``level`` narrows to the rows a step marked at that level, and is its own parameter rather
        than one more entry in ``filters``. The per-column filters are substring searches, which is
        right for a value somebody half-remembers and wrong for a vocabulary of three: ``warn`` would
        match ``warning``, and ``error`` would match every row of a client's own column holding the
        word -- both reading as a filter that works. So a level is checked against what the library
        publishes and refused when it is not one of them. A port with no marks is refused too rather
        than answered plausibly: every row and no rows are both readings of the data, when what is
        wrong is the question.

        The two narrowings are applied to the same rows rather than one to what the other already
        cut. A reader types a party into one box and picks a level in the other, and the answer they
        expect is the intersection; either order of narrowing-then-narrowing gives a subset that
        looks entirely plausible and is not it.
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
        unknown = sorted(set(wanted) - set(columns))
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
            # A status is written row for row with the port it describes -- the tool boundary
            # refuses one that is not -- so a pair that disagrees by the time it is read back is a
            # directory that was assembled by hand rather than written, or truncated after it was.
            # Worth refusing here because the disagreement is otherwise silent and then wrong: the
            # concat below pads the shorter side with nulls rather than failing, which hands a reader
            # a mark of ``null`` on every row past the end of the status and draws it as a level
            # nobody published. This is the one place the count ``persist_ports`` writes down beside
            # the file is ever read.
            if int(marks["rows"]) != int(entry["rows"]):
                raise PreviewError(
                    f"port {port!r} holds {entry['rows']:,} row(s) and the status beside it {marks['rows']:,}; "
                    "the two line up row for row, and position is the whole of what joins them"
                )

            # Carried alongside as one struct column so that the marks are sliced and filtered by
            # exactly the operations that slice and filter the rows -- the correspondence is
            # positional and is never re-derived, which is what stops a later window being given the
            # marks of the first one. Concatenated by position for the same reason: there is no key
            # joining a row to its mark, and a join on a column either side happens to carry would
            # be a different question with a plausible answer.
            scan = pl.concat(
                [scan, pl.scan_parquet(_inside(self.directory, marks["file"])).select(pl.struct(pl.all()).alias(beside))],
                how="horizontal",
            )
            if level is not None:
                scan = scan.filter(pl.col(beside).struct.field("level") == level)

        for name, text in wanted.items():
            # Cast before matching so a filter reads the value the page shows: a reader typing 2026
            # into a date column means the text they are looking at, not an epoch.
            scan = scan.filter(pl.col(name).cast(pl.String, strict=False).str.contains(text, literal=True))

        narrowed = bool(wanted) or level is not None
        matched = int(scan.select(pl.len()).collect().item()) if narrowed else int(entry["rows"])
        page = scan.slice(offset, limit).collect()
        # Named rather than `pl.all()`, so the marks travelling beside the rows cannot be handed back
        # as one more column of them: what the port produced is what a reader is shown as the data.
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
    """One flow's persisted run: where its ports live, the page that draws them, what to call it."""

    name: str
    directory: Path
    page: Path | None
    title: str


def discover(root: Path) -> list[Run]:
    """Every run written under ``root``, one per directory holding a manifest.

    Discovery rather than registration, so standing up a second flow for the same client is writing
    another directory rather than editing this module -- which is the difference between a viewer
    that serves what exists and one that serves what somebody remembered to list.
    """
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
    """A page listing the runs, for a reader who arrived at the root without a flow in mind."""
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
    """The source file a page asked for, or the status and sentence that refuse it.

    Served only when all of these hold:

    - the path is relative;
    - it resolves, with ``..`` and symlinks followed, to a file inside the source root;
    - no part of where it landed starts with a dot, so no ``.env`` and nothing in ``.git``;
    - the file it landed on is one of :data:`SOURCE_SUFFIXES`.

    A refusal is 403 for a file this route never serves and 404 for one that is not there.
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
    """The request handler, reading ``root`` on each request so a re-run is picked up without a restart."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            """Quiet: this serves one reader looking at their own runs, and a line per request is noise."""

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

            # Without the trailing slash a browser resolves the page's relative requests against the
            # service root instead of against the run, so every window would be asked of nobody.
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
                # As markup, not as JSON carrying markup: the page puts what comes back straight
                # into the pane, so there is nothing here for a caller to decide or to unwrap.
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
            # Its own parameter, and deliberately not reachable through the ``f.`` channel: a port
            # carrying a column called ``level`` is an ordinary thing for a client's data to have,
            # and spelling the level filter as ``f.level`` would make one control mean two different
            # things on two different ports with nothing to tell them apart. Asked for a column this
            # port does not carry, ``f.`` goes on refusing.
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
    """Serve every run written under ``root``, each at ``/<its directory name>/``.

    One service for a client's whole set of flows rather than one process each: they are read side
    by side, a figure in one is chased into another, and a reader should not have to remember which
    number a given flow was started on.

    Returns the server, already listening on its own thread, so a caller keeps the process alive
    however it likes and shuts it down by name rather than by signal.

    **This is a local viewer, and it carries no authentication, no authorisation and no transport
    security.** Every row of every port is readable by anything that can reach the socket. That is a
    deliberate trade, not an omission: the alternative is a dependency chain and a login on a tool
    one person uses to look at their own run. Anyone putting this anywhere a stranger can reach is
    not hardening this module; they are choosing a different one.

    It listens on ``127.0.0.1`` unless ``host`` says otherwise, so only this machine can reach it.
    Passing ``host="0.0.0.0"`` lets every machine on the network read every row of every run; a
    process in a container whose port is published to its host needs exactly that.

    With ``source_root``, ``/api/source?path=`` hands out source files under it: YAML, Python and
    Markdown only, never a hidden file, never anything outside it. See :func:`_source_file`.
    """
    server = ThreadingHTTPServer(
        (host, port), _handler(Path(root), Path(source_root).resolve() if source_root else None)
    )
    threading.Thread(target=server.serve_forever, name="etl-preview", daemon=True).start()
    return server
