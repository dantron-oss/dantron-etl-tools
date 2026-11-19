"""Running a project's pipeline and saving the run, without ever losing the last good one.

A preview runs every step except the writers and what comes after them, so nothing leaves the
project. A real run also writes, and only after the reader has confirmed what it will write.

Each run is written into a hidden folder first and then renamed into place. A run that fails or is
refused goes into ``<name>.failed`` and leaves ``<name>``, the last good run, as it was.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from ..errors import ToolError
from ..expr import ExpressionError
from ..focus import EntityOption, FocusError, FocusSpec, available_entities
from ..preview import MANIFEST, PAGE_NAME, persist_panels, persist_trace
from ..runner import run_pipeline
from ..spec import Pipeline
from ..trace import PipelineTrace, StepEvent, StepFailedError
from .project import Project, PipelineEntry, ProjectError, load_entry
from .project_settings import parse_focus
from .render import build, build_panels

__all__ = [
    "ENTITIES_FILE",
    "FAILED_SUFFIX",
    "KINDS",
    "RUN_FILE",
    "Plan",
    "RunRecord",
    "Write",
    "execute",
    "last_runs",
    "latest",
    "preview_plan",
    "save_run",
    "writes",
    "writes_digest",
]

RUN_FILE = "run.json"
FAILED_SUFFIX = ".failed"
ENTITIES_FILE = "entities.json"
WRITER_CATEGORY = "output"
KINDS = ("preview", "real")

#: Held while a slot folder is moved aside and replaced, so a reader never puts back a folder a
#: save in this process has only just moved aside.
_SWAP_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# What a preview runs, and what a real run writes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Plan:
    """The steps a preview runs, and the steps it holds back, both in run order."""

    runs: tuple[str, ...]
    held_back: tuple[str, ...]


def _is_writer(spec: Pipeline, step_id: str) -> bool:
    registry = spec.registry
    assert registry is not None  # a Pipeline always fills it in
    return registry.get(spec.step(step_id).tool).category == WRITER_CATEGORY


def preview_plan(spec: Pipeline) -> Plan:
    """Hold back every writer and every step fed by one. Everything else runs."""
    held: list[str] = []
    runs: list[str] = []
    for step_id in spec.order():
        fed_by_held = any(ref.step in held for ref in spec.step(step_id).inputs.values())
        if fed_by_held or _is_writer(spec, step_id):
            held.append(step_id)
        else:
            runs.append(step_id)
    return Plan(runs=tuple(runs), held_back=tuple(held))


@dataclass(frozen=True)
class Write:
    """One thing a writer step changes: a file, a table, or something else it names.

    ``settings`` is the writer step's whole configuration as compact JSON, so the digest changes
    when any setting does, not only the ones ``detail`` puts into words.
    """

    step: str
    tool: str
    kind: str
    target: str
    detail: str
    settings: str


def _plain(value: Any) -> Any:
    """A configuration value as plain JSON data."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


def _compact(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), default=str)


def _setting(config: Mapping[str, Any], template: Mapping[str, Any], key: str) -> Any:
    """The value a step sets for ``key``, or the tool's default for it."""
    if key in config:
        return config[key]
    field = template.get(key)
    return getattr(field, "default", None)


def _is_set(config: Mapping[str, Any], template: Mapping[str, Any], key: str) -> bool | None:
    """Whether an on-or-off setting is on, from the step or its tool's default; None if there is none."""
    if key not in config and key not in template:
        return None
    return bool(_setting(config, template, key))


def _what_is_there(config: Mapping[str, Any], template: Mapping[str, Any]) -> list[str]:
    """What a file writer does with a file, or a template tab, already at its target."""
    overwrite = _is_set(config, template, "overwrite")
    if overwrite is None:
        return []
    if not overwrite:
        return ["stops if the file is already there"]
    tabs = " and any template tab a sheet names" if config.get("template") else ""
    return [f"replaces the file if it is already there{tabs}"]


def _when_empty(config: Mapping[str, Any], template: Mapping[str, Any], *, file: bool) -> list[str]:
    """What a writer does when it is handed no rows."""
    allow = _is_set(config, template, "allow_empty")
    if allow is None:
        return []
    if not allow:
        return ["stops the run if there are no rows"]
    return ["writes headings only if there are no rows" if file else "loads even if there are no rows"]


def _sheets(config: Mapping[str, Any]) -> list[str]:
    """Each sheet a workbook writer makes, and what it does when its rows are empty."""
    parts = []
    for item in config.get("sheets") or ():
        if not isinstance(item, Mapping):
            continue
        name = item.get("sheet", item.get("port"))
        if item.get("allow_empty"):
            parts.append(f"sheet {name} is written with headings only if it has no rows")
        else:
            parts.append(f"sheet {name} stops the run if it has no rows")
    return parts


def _table_mode(config: Mapping[str, Any]) -> list[str]:
    """What a table writer does with the rows already in the table."""
    mode = config.get("mode")
    if mode == "replace":
        return ["empties the table, then loads the rows"]
    if mode == "append":
        return ["adds the rows to what is there"]
    if mode == "merge":
        keys = config.get("keys") or ()
        on = ", ".join(map(str, keys)) if isinstance(keys, list | tuple) else str(keys)
        parts = [f"updates the rows matching on {on} and adds the rest"]
        rule = config.get("when_incoming_is_null")
        if rule == "keep_published":
            parts.append("a blank incoming value keeps the published value")
        elif rule == "overwrite":
            parts.append("a blank incoming value blanks the published value")
        return parts
    return [f"mode {mode}"] if mode is not None else []


def _detail(config: Mapping[str, Any], template: Mapping[str, Any], *, file: bool) -> str:
    """In plain words: what the write does with what is already there, with no rows, and where it goes."""
    parts = _what_is_there(config, template)
    if config.get("template"):
        parts.append(f"starts from the template {config['template']}")
    parts += _sheets(config)
    parts += _table_mode(config)
    parts += _when_empty(config, template, file=file)
    connection = config.get("connection")
    if connection:
        parts.append(f"in the database named by {connection}")
    return "; ".join(parts)


def _table_name(schema: Any, table: Any) -> str:
    return f"{schema}.{table}" if schema else str(table)


def writes(spec: Pipeline, root: Path) -> tuple[Write, ...]:
    """Everything a real run would write, one entry per file or table, in run order."""
    registry = spec.registry
    assert registry is not None
    found: list[Write] = []
    for step_id in spec.order():
        step = spec.step(step_id)
        tool = registry.get(step.tool)
        if tool.category != WRITER_CATEGORY:
            continue
        config = step.config
        template = tool.config_template()
        settings = _compact(config)
        tables = config.get("tables")
        if isinstance(config.get("path"), str):
            target = str(Path(root) / config["path"])
            found.append(Write(step_id, step.tool, "file", target, _detail(config, template, file=True), settings))
        elif isinstance(config.get("table"), str):
            target = _table_name(config.get("schema"), config["table"])
            found.append(Write(step_id, step.tool, "table", target, _detail(config, template, file=False), settings))
        elif isinstance(tables, list | tuple) and tables and all(
            isinstance(item, Mapping) and isinstance(item.get("table"), str) for item in tables
        ):
            for item in tables:
                target = _table_name(item.get("schema", config.get("schema")), item["table"])
                item_template = getattr(template.get("tables"), "items", None)
                fields = dict(getattr(item_template, "fields", None) or {})
                detail = _detail({**config, **item}, {**template, **fields}, file=False)
                found.append(Write(step_id, step.tool, "table", target, detail, settings))
        else:
            found.append(Write(step_id, step.tool, "other", settings, "", settings))
    return tuple(found)


def writes_digest(found: Sequence[Write]) -> str:
    """A fingerprint of what a real run writes. It changes when any write or any writer setting changes."""
    text = json.dumps([asdict(one) for one in found], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf8")).hexdigest()


# ---------------------------------------------------------------------------
# The saved runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """What ``run.json`` says about one saved run, and where that run is."""

    pipeline: str
    kind: str
    outcome: str
    started: str
    finished: str | None
    steps_total: int
    steps_ran: int
    held_back: tuple[str, ...]
    failed_step: str | None
    message: str | None
    directory: Path
    focus: FocusSpec | None = None


def _focus_data(focus: FocusSpec | None) -> dict[str, Any] | None:
    """A focus in the form the settings file takes, to save with a run."""
    if focus is None:
        return None
    via = {str(port): {"from": str(rule.from_port), "key": rule.key} for port, rule in focus.via.items()}
    return {"columns": list(focus.identity), "via": via}


def _saved_focus(data: Mapping[str, Any], name: str) -> FocusSpec | None:
    """The focus a run was saved with, or None when it had none or it does not read back."""
    try:
        return parse_focus(data.get("focus"), name, "a saved run")
    except ProjectError:
        return None


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _unknown(name: str, directory: Path, stamp: Path) -> RunRecord:
    """A run saved by something else: it has ports but no record of how it ran."""
    started = datetime.fromtimestamp(stamp.stat().st_mtime).astimezone().isoformat()
    return RunRecord(name, "unknown", "unknown", started, None, 0, 0, (), None, None, directory)


def _read(directory: Path, name: str) -> RunRecord | None:
    if not directory.is_dir():
        return None
    record_file = directory / RUN_FILE
    if record_file.is_file():
        try:
            data = json.loads(record_file.read_text(encoding="utf8"))
            return RunRecord(
                pipeline=str(data.get("pipeline", name)),
                kind=str(data.get("kind", "unknown")),
                outcome=str(data.get("outcome", "unknown")),
                started=str(data["started"]),
                finished=data.get("finished"),
                steps_total=int(data.get("steps_total", 0)),
                steps_ran=int(data.get("steps_ran", 0)),
                held_back=tuple(data.get("held_back") or ()),
                failed_step=data.get("failed_step"),
                message=data.get("message"),
                directory=directory,
                focus=_saved_focus(data, name),
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            return _unknown(name, directory, record_file)
    if (directory / MANIFEST).is_file():
        return _unknown(name, directory, directory / MANIFEST)
    return None


def _recover(folder: Path, slot: str) -> None:
    """Put back a run a save moved aside and then stopped before putting the new one in."""
    with _SWAP_LOCK:
        if (folder / slot).exists():
            return
        aside = sorted(folder.glob(f".{slot}.old-*"), key=lambda path: path.stat().st_mtime)
        if aside:
            with contextlib.suppress(OSError):
                aside[-1].rename(folder / slot)


def last_runs(runs_folder: Path, name: str) -> tuple[RunRecord | None, RunRecord | None]:
    """The last good run and the last failed or refused one, each ``None`` when there is none."""
    folder = Path(runs_folder)
    slots = (name, f"{name}{FAILED_SUFFIX}")
    if folder.is_dir():
        for slot in slots:
            _recover(folder, slot)
    return _read(folder / slots[0], name), _read(folder / slots[1], name)


def _when(record: RunRecord) -> datetime:
    try:
        moment = datetime.fromisoformat(record.started)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return moment if moment.tzinfo else moment.astimezone()


def latest(good: RunRecord | None, failed: RunRecord | None) -> RunRecord | None:
    """Whichever of the two started last."""
    if good is None or failed is None:
        return good or failed
    return failed if _when(failed) >= _when(good) else good


def _swap_in(incoming: Path, target: Path) -> None:
    """Put ``incoming`` where ``target`` is. The old folder is moved aside first, then deleted."""
    aside = None
    with _SWAP_LOCK:
        if target.exists():
            aside = target.with_name(f".{target.name}.old-{secrets.token_hex(6)}")
            target.rename(aside)
        try:
            incoming.rename(target)
        except OSError:
            if aside is not None:
                aside.rename(target)
            raise
    if aside is not None:
        shutil.rmtree(aside, ignore_errors=True)


def _clear_incoming(folder: Path, name: str) -> None:
    """Delete folders an interrupted save of this pipeline left behind."""
    if folder.is_dir():
        for stale in folder.glob(f".{name}.incoming-*"):
            shutil.rmtree(stale, ignore_errors=True)


def _save(project: Project, name: str, good: bool, fill: Callable[[Path], None]) -> Path:
    """Write a run into a hidden folder, then move it into the good or the failed slot."""
    folder = project.runs_folder
    folder.mkdir(parents=True, exist_ok=True)
    failed_slot = folder / f"{name}{FAILED_SUFFIX}"
    target = folder / name if good else failed_slot
    incoming = folder / f".{name}.incoming-{secrets.token_hex(6)}"
    incoming.mkdir()
    try:
        fill(incoming)
        _swap_in(incoming, target)
    except BaseException:
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    # A good run makes the last failed one out of date, so it is deleted.
    if good and failed_slot.exists():
        shutil.rmtree(failed_slot)
    return target


def _write_record(directory: Path, record: Mapping[str, Any]) -> None:
    (directory / RUN_FILE).write_text(json.dumps(record, indent=2), encoding="utf8")


def _record(directory: Path, data: Mapping[str, Any]) -> RunRecord:
    fields = {**data, "held_back": tuple(data["held_back"]), "directory": directory}
    fields["focus"] = _saved_focus(data, str(data["pipeline"]))
    return RunRecord(**fields)


def _as_text(trace: PipelineTrace, focus: FocusSpec, options: Sequence[EntityOption]) -> list[str]:
    """Each entity value as the text the row matching compares with: its column cast to String."""
    by_type: dict[pl.DataType, list[int]] = {}
    for at, option in enumerate(options):
        port = option.ports[0]
        column = focus.identity_column(trace.shapes[port].columns)
        by_type.setdefault(trace.frames[port].schema[column], []).append(at)
    texts = [""] * len(options)
    for dtype, places in by_type.items():
        values = pl.Series([options[at].value for at in places], dtype=dtype).cast(pl.String, strict=False)
        for at, text in zip(places, values.to_list(), strict=True):
            texts[at] = text
    return texts


def _entities(trace: PipelineTrace, focus: FocusSpec) -> dict[str, Any]:
    """Every entity the run can be followed by, as text. A focus that cannot list them says why."""
    found: dict[str, Any] = {"label": focus.label, "columns": list(focus.identity)}
    try:
        found["values"] = _as_text(trace, focus, available_entities(trace, focus))
    except FocusError as reason:
        found["values"] = []
        found["reason"] = str(reason)
    return found


def _refuse(project: Project, entry: PipelineEntry, data: dict[str, Any], message: str) -> RunRecord:
    data.update(outcome="refused", message=message, finished=_now())
    target = _save(project, entry.name, False, lambda into: _write_record(into, data))
    return _record(target, data)


def execute(
    project: Project,
    entry: PipelineEntry,
    kind: str,
    *,
    sink: Callable[[StepEvent], None] | None = None,
    confirm_digest: str | None = None,
) -> RunRecord:
    """Run one pipeline as a preview or for real, save the run, and return its record.

    A real run needs ``confirm_digest`` to equal the digest of what it writes now. Otherwise it is
    refused with a ProjectError before anything runs or is saved.
    """
    import yaml

    if kind not in KINDS:
        raise ProjectError(f"a run is 'preview' or 'real', not {kind!r}")
    _clear_incoming(project.runs_folder, entry.name)
    data: dict[str, Any] = {
        "pipeline": entry.name,
        "kind": kind,
        "outcome": "finished",
        "started": _now(),
        "finished": None,
        "steps_total": entry.steps,
        "steps_ran": 0,
        "held_back": [],
        "failed_step": None,
        "message": None,
        "focus": _focus_data(entry.focus),
    }
    try:
        spec = load_entry(project, entry)
    except (ToolError, ExpressionError, yaml.YAMLError, OSError) as refusal:
        return _refuse(project, entry, data, str(refusal))
    data["steps_total"] = len(spec.steps)

    until: tuple[str, ...] | None = None
    if kind == "real":
        if confirm_digest != writes_digest(writes(spec, project.root)):
            raise ProjectError(
                f"what pipeline '{entry.name}' writes has changed since it was confirmed, or it was "
                "never confirmed. Look at what it writes again and confirm that."
            )
    else:
        plan = preview_plan(spec)
        data["held_back"] = list(plan.held_back)
        if not plan.runs:
            return _refuse(
                project, entry, data,
                f"every step in pipeline '{entry.name}' writes or follows a writer, so a preview has "
                "nothing to run. Run it for real to see it.",
            )
        until = plan.runs if plan.held_back else None

    try:
        trace = run_pipeline(spec, until=until, sink=sink, base_dir=project.root)
    except StepFailedError as failure:
        return _save_trace(project, spec, trace=failure.trace, data=data, focus=entry.focus, failure=failure)
    return _save_trace(project, spec, trace=trace, data=data, focus=entry.focus, failure=None)


def _save_trace(
    project: Project,
    spec: Pipeline,
    *,
    trace: PipelineTrace,
    data: dict[str, Any],
    focus: FocusSpec | None,
    failure: StepFailedError | None,
) -> RunRecord:
    """Save a trace with its record, in the good slot if it finished and the failed slot if not."""
    if failure is not None:
        data.update(outcome="failed", failed_step=failure.step_id, message=str(failure))
    data["steps_ran"] = sum(1 for event in trace.events if event.step_id != data["failed_step"])
    data["focus"] = _focus_data(focus)

    def fill(into: Path) -> None:
        persist_trace(trace, into)
        persist_panels(build_panels(spec, trace, focus), into)
        total = None
        if focus is not None:
            listed = _entities(trace, focus)
            total = len(listed["values"])
            (into / ENTITIES_FILE).write_text(json.dumps(listed), encoding="utf8")
        data["finished"] = _now()
        page = build(spec, trace, focus, run={**data, "entities": total})
        (into / PAGE_NAME).write_text(page, encoding="utf-8")
        _write_record(into, data)

    target = _save(project, str(data["pipeline"]), data["outcome"] == "finished", fill)
    return _record(target, data)


def save_run(
    project: Project,
    name: str,
    spec: Pipeline,
    trace: PipelineTrace,
    *,
    failure: StepFailedError | None = None,
    focus: FocusSpec | None = None,
    kind: str = "real",
    started: str | None = None,
) -> RunRecord:
    """Save a run a project's own script made, so the viewer lists and shows it like its own runs.

    ``name`` must be one of the project's pipelines. Pass the ``StepFailedError`` as ``failure`` when
    the run stopped; it is then saved beside the last good run, which is kept. ``focus`` is the
    column to follow; left out, the project's setting for this pipeline is used. ``started`` is when
    the run began, as ISO 8601 text; left out, it is now.
    """
    entry = project.pipelines.get(name)
    if entry is None:
        known = ", ".join(project.pipelines) or "none"
        raise ProjectError(f"the project has no pipeline '{name}' to save a run for; it has: {known}")
    if kind not in KINDS:
        raise ProjectError(f"a run is 'preview' or 'real', not {kind!r}")
    _clear_incoming(project.runs_folder, name)
    data: dict[str, Any] = {
        "pipeline": name,
        "kind": kind,
        "outcome": "finished",
        "started": started or _now(),
        "finished": None,
        "steps_total": len(spec.steps),
        "steps_ran": 0,
        "held_back": [step for step in spec.order() if step not in trace.until] if trace.until else [],
        "failed_step": None,
        "message": None,
        "focus": None,
    }
    return _save_trace(project, spec, trace=trace, data=data, focus=focus or entry.focus, failure=failure)
