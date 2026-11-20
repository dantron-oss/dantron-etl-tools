"""The project viewer: a local web server for one project's pipelines and their saved runs.

It lists the pipelines, shows each one's last run, and starts a preview or a real run from a
button. The project is read from disk again on every request, so a new pipeline file shows up on
the next reload.

It answers only this machine. A request whose ``Host`` is not a loopback name is refused, a run
needs the token the server put in the page, and a pipeline or run name is looked up in what was
found on disk. A name is never joined onto a path.

It also shows a file a step names: its facts, its first rows, and its place in this machine's file
manager. The page names the pipeline, the step and the setting; the server looks the path up.
"""

from __future__ import annotations

import ipaddress
import json
import os
import secrets
import socket
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..cli import VIEWER_APP
from ..errors import ToolError
from ..expr import ExpressionError
from ..focus import FocusError, FocusSpec
from ..preview import FILTER_PREFIX, PAGE_NAME, PortStore, PreviewError
from ..trace import StepEvent
from .assets import escape as _e
from .assets import page, page_data
from .files import FileRefused, NamedFile, facts, named_file, named_files, preview
from .focus_rows import focus_match
from .project import PipelineEntry, Project, ProjectError, load_entry, load_project, pipeline_rules
from .reveal import Opened, Target, open_location
from .runs import (
    ENTITIES_FILE,
    KINDS,
    RunRecord,
    execute,
    last_runs,
    latest,
    preview_plan,
    run_folder,
    writes,
    writes_digest,
)

__all__ = ["MAX_BODY", "ViewerServer", "start_server"]

#: The largest request body a POST may send. A run request is a few short fields.
MAX_BODY = 64 * 1024

LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
SLOTS = ("good", "failed")
SLOT_APIS = frozenset({"ports", "rows", "panel", "entities"})
DONE = ("finished", "failed", "refused")
DEFAULT_ENTITY_LIMIT = 50
MAX_ENTITY_LIMIT = 500
EMPTY_TOKEN = '<meta name="run-token" content="">'
FILE_APIS = frozenset({"file", "file-preview"})
OPEN_LOCATION = "/api/open-location"

#: Shows a file in this machine's file manager. Tests hand the server a fake.
Reveal = Callable[[Target], Opened]
EMPTY_RUNBAR = '<div class="runbar"></div>'


class ViewerServer:
    """A running viewer. ``url`` is where to open it; ``shutdown()`` stops it and frees the port."""

    def __init__(self, httpd: ThreadingHTTPServer, thread: threading.Thread, token: str, host: str) -> None:
        self._httpd = httpd
        self._thread = thread
        self.token = token
        self.host = host

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        shown = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{shown}:{self.port}/"

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def start_server(
    project: Project, *, host: str = "127.0.0.1", port: int = 8899, reveal: Reveal | None = None
) -> ViewerServer:
    """Serve ``project`` on ``host``:``port`` from a background thread and return at once.

    Port 0 lets the system pick a free port; ``ViewerServer.port`` says which. ``reveal`` shows a file
    in the file manager; left out, it is this machine's.
    """
    state = _State(project.root, secrets.token_urlsafe(32), _is_loopback(host), reveal or open_location)

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        daemon_threads = True

    httpd = Server((host, port), _handler(state))
    thread = threading.Thread(target=httpd.serve_forever, name="etl-viewer", daemon=True)
    thread.start()
    return ViewerServer(httpd, thread, state.token, host)


def _is_loopback(host: str) -> bool:
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _hostname(value: str | None) -> str | None:
    """The host part of a ``Host`` header or an origin, lower case, or None when it does not parse."""
    if not value:
        return None
    try:
        parts = urlsplit(value if "//" in value else f"//{value}")
        parts.port  # noqa: B018 -- refuses a port that is not a number
    except ValueError:
        return None
    return parts.hostname


# ---------------------------------------------------------------------------
# Runs in progress
# ---------------------------------------------------------------------------


def _progress(state: str, kind: str | None, total: int) -> dict[str, Any]:
    """A pipeline's progress before any step has ended."""
    return {"state": state, "kind": kind, "events": [], "total": total, "failed_step": None, "message": None}


class _State:
    """What the server holds between requests: the project root, the token, and runs in progress."""

    def __init__(self, root: Path, token: str, loopback: bool, reveal: Reveal) -> None:
        self.root = root
        self.token = token
        self.loopback = loopback
        self.reveal = reveal
        self.lock = threading.Lock()
        self.progress: dict[str, dict[str, Any]] = {}

    def project(self) -> Project:
        return load_project(self.root)

    def running(self, name: str) -> bool:
        with self.lock:
            return self.progress.get(name, {}).get("state") == "running"

    def snapshot(self, name: str, steps: int) -> dict[str, Any]:
        """A copy of a pipeline's progress. A pipeline not run since the server started is idle."""
        with self.lock:
            found = self.progress.get(name)
            if found is None:
                return _progress("idle", None, steps)
            return {**found, "events": list(found["events"])}

    def claim(self, name: str, kind: str, steps: int) -> bool:
        """Mark ``name`` as running, unless it already is. Done before a run is answered 202."""
        with self.lock:
            if self.progress.get(name, {}).get("state") == "running":
                return False
            self.progress[name] = _progress("running", kind, steps)
            return True

    def update(self, name: str, **changes: Any) -> None:
        with self.lock:
            self.progress[name].update(changes)

    def add_event(self, name: str, event: StepEvent) -> None:
        shown = {
            "step": event.step_id,
            "tool": event.tool,
            "level": event.level,
            "rows_out": sum(shape.rows for shape in event.outputs),
            "ms": round(event.duration_ms, 1),
        }
        with self.lock:
            self.progress[name]["events"].append(shown)


def _run_in_background(state: _State, project: Project, entry: PipelineEntry, kind: str, digest: str | None) -> None:
    """Run one pipeline on a thread of its own, feeding its progress as each step ends."""
    name = entry.name

    def work() -> None:
        try:
            if kind == "preview":
                state.update(name, total=_preview_total(project, entry))
            record = execute(
                project, entry, kind, sink=lambda event: state.add_event(name, event), confirm_digest=digest
            )
            state.update(name, state=record.outcome, failed_step=record.failed_step, message=record.message)
        except ProjectError as refusal:
            state.update(name, state="refused", message=str(refusal))
        except Exception as error:  # noqa: BLE001 -- a run must never be left showing as running
            state.update(name, state="failed", message=f"the viewer could not finish the run: {error}")

    threading.Thread(target=work, name=f"etl-run-{name}", daemon=True).start()


def _preview_total(project: Project, entry: PipelineEntry) -> int:
    """How many steps a preview runs, or every step when the pipeline does not load."""
    try:
        return len(preview_plan(load_entry(project, entry)).runs)
    except Exception:  # noqa: BLE001 -- the run itself reports why it does not load
        return entry.steps


# ---------------------------------------------------------------------------
# Plain data for the JSON answers
# ---------------------------------------------------------------------------


def _plan(project: Project, entry: PipelineEntry) -> dict[str, Any]:
    """What a preview runs and what a real run writes. Loading the pipeline may run project code."""
    import yaml

    try:
        spec = load_entry(project, entry)
    except (ToolError, ExpressionError, yaml.YAMLError, OSError) as refusal:
        return {"steps": entry.steps, "preview": None, "writes": [], "digest": None, "problem": str(refusal)}
    plan = preview_plan(spec)
    found = writes(spec, project.root)
    # A file write is named by its step's 'path'. Its facts go beside it, outside the digest: a file
    # appearing or changing is not a change to what the run writes.
    files = [one for one in named_files(spec, run_folder(project)) if one.setting == "path"]
    shown = [
        {
            **asdict(one),
            "files": [facts(named, project.root) for named in files if named.step == one.step and one.kind == "file"],
        }
        for one in found
    ]
    return {
        "steps": len(spec.steps),
        "preview": {"runs": list(plan.runs), "held_back": list(plan.held_back)},
        "writes": shown,
        "digest": writes_digest(found),
        "problem": None,
    }


def _last(good: RunRecord | None, failed: RunRecord | None) -> dict[str, Any] | None:
    last = latest(good, failed)
    if last is None:
        return None
    return {
        "kind": last.kind,
        "outcome": last.outcome,
        "started": last.started,
        "finished": last.finished,
        "failed_step": last.failed_step,
        "message": last.message,
        "steps_ran": last.steps_ran,
        "held_back": list(last.held_back),
        "previous_good": good.started if good is not None and last is failed else None,
    }


def _listing(state: _State, project: Project) -> dict[str, Any]:
    pipelines = []
    for name, entry in project.pipelines.items():
        pipelines.append(
            {
                "name": name,
                "title": entry.title,
                "steps": entry.steps,
                "problem": entry.problem,
                "rules_needed": list(entry.rules_needed),
                "rules_source": entry.rules_source,
                "last": _last(*last_runs(project.runs_folder, name)),
                "running": state.running(name),
            }
        )
    return {"project": str(project.root), "pipelines": pipelines, "problems": list(project.problems)}


def _entities(directory: Path, query: Mapping[str, list[str]]) -> dict[str, Any]:
    """Search a run's saved entity list: a case-insensitive substring, at most ``limit`` values back."""
    asked = _one(query, "q").casefold()
    limit = _whole(query, "limit", DEFAULT_ENTITY_LIMIT)
    if not 1 <= limit <= MAX_ENTITY_LIMIT:
        raise PreviewError(f"limit must be between 1 and {MAX_ENTITY_LIMIT}; got {limit}")
    saved = directory / ENTITIES_FILE
    if not saved.is_file():
        return {"label": None, "total": 0, "matched": 0, "values": [], "reason": "this run follows no entity"}
    listed = json.loads(saved.read_text(encoding="utf8"))
    values = [str(value) for value in listed.get("values") or []]
    matched = [value for value in values if asked in value.casefold()]
    found = {"label": listed.get("label"), "total": len(values), "matched": len(matched), "values": matched[:limit]}
    if listed.get("reason"):
        found["reason"] = listed["reason"]
    return found


def _rows(directory: Path, name: str, focus: FocusSpec | None, query: Mapping[str, list[str]]) -> dict[str, Any]:
    """One window of a saved port, narrowed by the column filters, a level and one entity.

    ``focus`` is the one the run was saved with, so a later change to the settings does not change it.
    """
    store = PortStore(directory)
    port = _one(query, "port")
    filters = {key[len(FILTER_PREFIX) :]: values[0] for key, values in query.items() if key.startswith(FILTER_PREFIX)}
    level = _one(query, "level") or None
    focus_value = query.get("focus")
    match = None
    no_grain = None
    if focus_value is not None:
        if focus is None:
            raise PreviewError(f"this run of pipeline '{name}' follows no entity, so its rows cannot be narrowed to one")
        how, found = focus_match(store, focus, port, focus_value[0])
        if how == "match":
            match = found
        else:
            no_grain = found
    window = store.window(
        port,
        offset=_whole(query, "offset", 0),
        limit=_whole(query, "limit", 100),
        filters=filters,
        level=level,
        match=match,
    )
    return {
        "columns": list(window.columns),
        "dtypes": list(window.dtypes),
        "rows": [list(row) for row in window.rows],
        "total": window.total,
        "matched": window.matched,
        "offset": window.offset,
        "status": list(window.status) if window.status is not None else None,
        "no_grain": no_grain,
    }


def _one(query: Mapping[str, list[str]], key: str) -> str:
    return (query.get(key) or [""])[0]


def _whole(query: Mapping[str, list[str]], key: str, default: int) -> int:
    text = _one(query, key)
    if not text:
        return default
    try:
        return int(text)
    except ValueError as bad:
        raise PreviewError(f"{key} must be a whole number; got {text!r}") from bad


# ---------------------------------------------------------------------------
# A file a step names
# ---------------------------------------------------------------------------


def _named(project: Project, entry: PipelineEntry, step: Any, setting: Any) -> NamedFile:
    """The file ``step`` names in ``setting``, looked up in the pipeline loaded as a run loads it.

    Loading may run project code (a rules loader), so every caller has checked the run token.
    """
    import yaml

    if not isinstance(step, str) or not isinstance(setting, str) or not step or not setting:
        raise _Refused(404, "name the step and the setting that holds the file")
    try:
        spec = load_entry(project, entry)
    except (ToolError, ExpressionError, yaml.YAMLError, OSError) as refusal:
        raise _Refused(409, str(refusal)) from refusal
    try:
        return named_file(spec, run_folder(project), step, setting)
    except FileRefused as refusal:
        raise _Refused(404, str(refusal)) from refusal


def _target(named: NamedFile) -> Target:
    """Where the file really is, through any link, and whether it and its folder are there."""
    if named.path is None:
        raise _Refused(409, "the viewer cannot work out where this file is, so it cannot show it")
    real = os.path.realpath(named.path)
    return Target(real, os.path.exists(real), os.path.isdir(os.path.dirname(real)))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _when(started: str | None) -> str:
    return started[:16].replace("T", " ") if started else ""


def _outcome(last: Mapping[str, Any] | None) -> str:
    if last is None:
        return "never run"
    if last["outcome"] == "failed":
        return f"failed at step {last['failed_step']}"
    return str(last["outcome"])


def _progress_line(name: str, running: bool, then: str | None = None) -> str:
    after = f' data-then="{_e(then)}"' if then else ""
    return (
        f'<span class="progress" data-progress="{_e(name)}" data-running="{1 if running else 0}"{after}></span>'
    )


def _home(state: _State) -> str:
    try:
        project = state.project()
        listed = _listing(state, project)
        problems = list(project.problems)
    except ToolError as refusal:
        listed = {"project": str(state.root), "pipelines": [], "problems": []}
        problems = [str(refusal)]
    rows = []
    for item in listed["pipelines"]:
        name, last = item["name"], item["last"]
        note = f'<p class="note">{_e(item["problem"])}</p>' if item["problem"] else ""
        if last and last["previous_good"]:
            note += (
                f'<p class="note">The last good run, from {_e(_when(last["previous_good"]))}, is kept. '
                f'<a href="/{_e(name)}/good/">Open it</a></p>'
            )
        rows.append(
            "<tr>"
            f'<td><a href="/{_e(name)}/">{_e(item["title"])}</a><code>{_e(name)}</code>{note}</td>'
            f'<td class="num">{item["steps"]}</td>'
            f"<td>{_e(_when(last['started'] if last else None))}</td>"
            f'<td class="outcome {_e(last["outcome"] if last else "never")}">{_e(_outcome(last))}</td>'
            f"<td>{_e(last['kind'] if last else '')}</td>"
            f'<td><button type="button" data-run="preview" data-pipeline="{_e(name)}">Run preview</button>'
            f"{_progress_line(name, item['running'])}</td>"
            "</tr>"
        )
    table = "".join(rows) or '<tr><td colspan="6">No pipelines found.</td></tr>'
    problem_list = (
        '<section class="problems"><h2>Problems</h2><ul>'
        + "".join(f"<li>{_e(problem)}</li>" for problem in problems)
        + "</ul></section>"
        if problems
        else ""
    )
    return page(
        "home",
        scripts=("files", "runbar"),
        root=_e(listed["project"]),
        rows=table,
        problems=problem_list,
        pagedata=page_data({"project": listed["project"], "pipelines": [p["name"] for p in listed["pipelines"]]}),
    )


def _has_page_or_kept(good: RunRecord | None, failed: RunRecord | None) -> bool:
    """Whether ``/<name>/`` moves to a slot: some run has a page, or a bad attempt kept a good run."""
    if good is not None and failed is not None:
        return True
    only = good or failed
    return only is not None and (only.directory / PAGE_NAME).is_file()


def _runbar(name: str, running: bool, kept: RunRecord | None = None) -> str:
    """The run buttons and the progress line, with a link to the last good run when it was kept."""
    kept_line = (
        f'<p class="banner kept">The last good run, from {_e(_when(kept.started))}, is kept. '
        f'<a href="/{_e(name)}/good/">Open the last good run</a></p>'
        if kept is not None
        else ""
    )
    return (
        f'<div class="runbar" data-pipeline="{_e(name)}">'
        f'<button type="button" data-run="preview" data-pipeline="{_e(name)}">Run preview</button>'
        f'<button type="button" class="real" data-run="real" data-pipeline="{_e(name)}">Real run&hellip;</button>'
        f"{_progress_line(name, running, f'/{name}/')}"
        f'<ol class="steps" data-steps="{_e(name)}"></ol>{kept_line}</div>'
    )


def _pipeline_page(
    project: Project, entry: PipelineEntry, record: RunRecord | None, running: bool, kept: RunRecord | None = None
) -> str:
    """The page for a pipeline with no run page to show: never run, refused, or saved without one."""
    if record is None:
        happened = "This pipeline has not been run yet."
    elif record.outcome == "refused":
        happened = "The last run was refused before any step ran."
    else:
        happened = "A run was saved, but without a page to show."
    told = [text for text in (entry.problem, record.message if record else None) if text]
    if entry.rules_needed and entry.rules_source is None and entry.problem is None:
        try:
            pipeline_rules(project, entry)
        except ProjectError as hint:
            if str(hint) not in told:
                told.append(str(hint))
    messages = "".join(f'<pre class="message">{_e(text)}</pre>' for text in told)
    last = None
    if record is not None:
        last = {"kind": record.kind, "outcome": record.outcome, "started": record.started, "message": record.message}
    data = {"pipeline": entry.name, "title": entry.title, "last": last}
    return page(
        "pipeline",
        scripts=("files", "runbar"),
        name=_e(entry.name),
        title=_e(entry.title),
        happened=_e(happened),
        messages=messages,
        runbar=_runbar(entry.name, running, kept),
        pagedata=page_data(data),
    )


def _served(markup: str, token: str, runbar: str | None = None) -> bytes:
    """A page on its way out: the run token filled in, and the run bar when the page has room for one."""
    markup = markup.replace(EMPTY_TOKEN, f'<meta name="run-token" content="{_e(token)}">', 1)
    if runbar is not None:
        markup = markup.replace(EMPTY_RUNBAR, runbar, 1)
    return markup.encode("utf-8")


@cache
def _catalog_page() -> bytes:
    from ..catalog import catalog
    from .catalog_page import build_page

    return build_page(catalog()).encode("utf-8")


# ---------------------------------------------------------------------------
# The request handler
# ---------------------------------------------------------------------------


class _Refused(Exception):
    def __init__(self, status: int, message: str, headers: Mapping[str, str] | None = None, **extra: Any) -> None:
        super().__init__(message)
        self.status = status
        self.headers = dict(headers or {})
        self.payload = {"error": message, **extra}


def _handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            """Quiet: a line per request is noise for one reader."""

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            super().end_headers()

        # -- answers ---------------------------------------------------------------------------

        def _send(self, status: int, body: bytes, kind: str, headers: Mapping[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any, headers: Mapping[str, str] | None = None) -> None:
            body = json.dumps(payload, default=str).encode("utf8")
            self._send(status, body, "application/json; charset=utf-8", headers)

        def _html(self, body: bytes) -> None:
            self._send(200, body, "text/html; charset=utf-8")

        def _require_token(self) -> None:
            sent = self.headers.get("X-Run-Token", "")
            if not secrets.compare_digest(sent.encode("utf8"), state.token.encode("utf8")):
                raise _Refused(403, "this request does not carry the viewer's run token; reload the page")

        def _redirect(self, to: str, status: int = 301) -> None:
            self.send_response(status)
            self.send_header("Location", to)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _guarded(self, answer: Callable[[], None]) -> None:
            host = _hostname(self.headers.get("Host"))
            if state.loopback and host not in LOOPBACK_NAMES:
                self._json(403, {"error": "this viewer answers only this computer, at 127.0.0.1 or localhost"})
                return
            try:
                answer()
            except _Refused as refusal:
                self._json(refusal.status, refusal.payload, refusal.headers)
            except (PreviewError, FocusError) as refusal:
                self._json(400, {"error": str(refusal)})
            except ToolError as refusal:
                self._json(500, {"error": str(refusal)})
            except Exception as error:  # noqa: BLE001 -- a sentence beats a dropped connection
                self._json(500, {"error": f"the viewer could not answer: {error}"})

        def do_GET(self) -> None:  # noqa: N802 -- the name the base class dispatches on
            self._guarded(self._get)

        def do_POST(self) -> None:  # noqa: N802
            self._guarded(self._post)

        # -- GET -------------------------------------------------------------------------------

        def _get(self) -> None:
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)
            if path == "/":
                self._html(_served(_home(state), state.token))
            elif path == "/api/pipelines":
                self._json(200, _listing(state, state.project()))
            elif path == "/api/whoami":
                self._json(200, {"app": VIEWER_APP, "project": str(state.root)})
            elif path == "/catalog/":
                self._html(_catalog_page())
            elif path == "/catalog":
                self._redirect("/catalog/")
            elif path == OPEN_LOCATION:
                raise _Refused(405, "opening a file's location is an action: send it as a POST", {"Allow": "POST"})
            elif not path.startswith("/"):
                raise _Refused(404, "no such page")
            else:
                self._pipeline_route(path.split("/")[1:], query)

        def _pipeline_route(self, segments: list[str], query: Mapping[str, list[str]]) -> None:
            project = state.project()
            name, rest = segments[0], segments[1:]
            entry = project.pipelines.get(name)
            if entry is None:
                raise _Refused(404, f"no pipeline {name!r} in this project")
            if not rest:
                self._redirect(f"/{name}/")
                return
            if rest == ["api", "plan"]:
                self._require_token()
                self._json(200, _plan(project, entry))
                return
            if rest == ["api", "progress"]:
                self._json(200, state.snapshot(name, entry.steps))
                return
            if len(rest) == 2 and rest[0] == "api" and rest[1] in FILE_APIS:
                self._require_token()
                self._file_api(rest[1], project, entry, query)
                return
            slot = None
            if rest[0] in SLOTS:
                slot, rest = rest[0], rest[1:]
                if not rest:
                    self._redirect(f"/{name}/{slot}/")
                    return
            good, failed = last_runs(project.runs_folder, name)
            record = {"good": good, "failed": failed, None: latest(good, failed)}[slot]
            if rest == [""] and slot is None and _has_page_or_kept(good, failed):
                # A page's api/ requests are relative, so serving it from its slot keeps them on its run.
                self._redirect(f"/{name}/{'failed' if record is failed else 'good'}/", 303)
            elif rest == [""]:
                self._run_page(project, entry, record, slot, good if record is failed else None)
            elif len(rest) == 2 and rest[0] == "api" and rest[1] in SLOT_APIS:
                if record is None:
                    raise _Refused(404, f"pipeline '{name}' has no saved run here")
                self._slot_api(rest[1], record, entry, query)
            else:
                raise _Refused(404, f"no page {'/'.join(segments)!r}")

        def _run_page(
            self,
            project: Project,
            entry: PipelineEntry,
            record: RunRecord | None,
            slot: str | None,
            kept: RunRecord | None,
        ) -> None:
            """A run's saved page with the run bar in it, or the pipeline's page when there is none to show."""
            if record is None and slot is not None:
                raise _Refused(404, f"pipeline '{entry.name}' has no {slot} run saved")
            running = state.running(entry.name)
            saved = record.directory / PAGE_NAME if record is not None else None
            if saved is None or not saved.is_file():
                self._html(_served(_pipeline_page(project, entry, record, running, kept), state.token))
                return
            runbar = _runbar(entry.name, running, kept)
            self._html(_served(saved.read_text(encoding="utf-8"), state.token, runbar))

        def _file_api(self, what: str, project: Project, entry: PipelineEntry, query: Mapping[str, list[str]]) -> None:
            """A file's facts, or its first rows. The query names a step and a setting, never a path."""
            named = _named(project, entry, _one(query, "step"), _one(query, "setting"))
            if what == "file":
                self._json(200, facts(named, project.root))
                return
            sheet = query.get("sheet")
            try:
                self._json(200, preview(named, sheet[0] if sheet else None))
            except FileRefused as refusal:
                raise _Refused(404, str(refusal)) from refusal

        def _slot_api(self, what: str, record: RunRecord, entry: PipelineEntry, query: Mapping[str, list[str]]) -> None:
            directory = record.directory
            if what == "entities":
                self._json(200, _entities(directory, query))
                return
            try:
                store = PortStore(directory)
            except PreviewError as missing:
                raise _Refused(404, str(missing)) from missing
            if what == "ports":
                self._json(200, {"ports": store.ports()})
            elif what == "rows":
                focus = record.focus if record.kind != "unknown" else entry.focus
                self._json(200, _rows(directory, entry.name, focus, query))
            else:
                self._html(store.panel(_one(query, "step")).encode("utf-8"))

        # -- POST ------------------------------------------------------------------------------

        def _post(self) -> None:
            actions = {"/api/run": self._start, OPEN_LOCATION: self._open}
            act = actions.get(urlsplit(self.path).path)
            if act is None:
                raise _Refused(404, f"this viewer takes a POST only for {' and '.join(actions)}")
            origin = self.headers.get("Origin")
            if origin is not None:
                allowed = set(LOOPBACK_NAMES)
                if not state.loopback:
                    allowed.add(_hostname(self.headers.get("Host")) or "")
                if _hostname(origin) not in allowed:
                    raise _Refused(403, "this viewer acts only on requests from its own pages")
            self._require_token()
            body = self._body()
            act(body)

        def _body(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as missing:
                self.close_connection = True
                raise _Refused(411, "send the request's length") from missing
            if not 0 <= length <= MAX_BODY:
                self.close_connection = True
                raise _Refused(413, f"a run request is at most {MAX_BODY} bytes")
            try:
                body = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError) as bad:
                raise _Refused(400, "the request is not JSON") from bad
            if not isinstance(body, dict):
                raise _Refused(400, "the request must be a JSON object")
            return body

        def _start(self, body: Mapping[str, Any]) -> None:
            project = state.project()
            name = body.get("pipeline")
            entry = project.pipelines.get(name) if isinstance(name, str) else None
            if entry is None:
                raise _Refused(404, f"no pipeline {name!r} in this project")
            kind = body.get("kind")
            if kind not in KINDS:
                raise _Refused(400, "kind must be 'preview' or 'real'")
            if state.running(entry.name):
                raise _Refused(409, f"pipeline '{entry.name}' is already running")
            digest = None
            if kind == "real":
                digest = body.get("digest")
                if body.get("confirm") != entry.name:
                    raise _Refused(400, f"type the pipeline name, {entry.name}, to confirm a real run")
                if not isinstance(digest, str):
                    raise _Refused(400, "a real run needs the digest of the writes that were confirmed")
                plan = _plan(project, entry)
                if plan["problem"] is not None:
                    raise _Refused(409, plan["problem"], plan=plan)
                if digest != plan["digest"]:
                    raise _Refused(
                        409, "what this pipeline writes has changed since it was shown. Look again.", plan=plan
                    )
            if not state.claim(entry.name, kind, entry.steps):
                raise _Refused(409, f"pipeline '{entry.name}' is already running")
            _run_in_background(state, project, entry, kind, digest)
            self._json(202, {"started": True})

        def _open(self, body: Mapping[str, Any]) -> None:
            """Show a file a step names in this machine's file manager, and say what was done."""
            project = state.project()
            name = body.get("pipeline")
            entry = project.pipelines.get(name) if isinstance(name, str) else None
            if entry is None:
                raise _Refused(404, f"no pipeline {name!r} in this project")
            target = _target(_named(project, entry, body.get("step"), body.get("setting")))
            opened = state.reveal(target)
            self._json(
                200,
                {
                    "opened": opened.opened,
                    "message": opened.message,
                    "path": target.path,
                    "command": list(opened.command or ()),
                },
            )

    return Handler
