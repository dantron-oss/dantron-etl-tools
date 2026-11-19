"""Running a project's pipeline and saving the run, without ever losing the last good one.

A preview runs every step except the writers and what comes after them, so nothing leaves the
project. A real run also writes, and only after the reader has confirmed what it will write.

Each run is written into a hidden folder first and then renamed into place. A run that fails or is
refused goes into ``<name>.failed`` and leaves ``<name>``, the last good run, as it was.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..expr import ExpressionError
from ..focus import FocusError, FocusSpec, available_entities
from ..preview import MANIFEST, PAGE_NAME, persist_panels, persist_trace
from ..runner import run_pipeline
from ..spec import Pipeline
from ..trace import PipelineTrace, StepEvent, StepFailedError
from .project import Project, PipelineEntry, ProjectError, load_entry
from .render import build, build_panels

__all__ = [
    "ENTITIES_FILE",
    "FAILED_SUFFIX",
    "RUN_FILE",
    "Plan",
    "RunRecord",
    "Write",
    "execute",
    "last_runs",
    "latest",
    "preview_plan",
    "writes",
    "writes_digest",
]

RUN_FILE = "run.json"
FAILED_SUFFIX = ".failed"
ENTITIES_FILE = "entities.json"
WRITER_CATEGORY = "output"
KINDS = ("preview", "real")


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
    """One thing a writer step changes: a file, a table, or something else it names."""

    step: str
    tool: str
    kind: str
    target: str
    detail: str


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


def _sheet_names(sheets: Any) -> list[str]:
    names = []
    for item in sheets if isinstance(sheets, list | tuple) else []:
        if isinstance(item, Mapping):
            name = item.get("sheet", item.get("name", item.get("port")))
        else:
            name = item
        if name is not None:
            names.append(str(name))
    return names


def _detail(config: Mapping[str, Any], template: Mapping[str, Any], mode: Any = None) -> str:
    """The plain words that go with a write: how it treats what is there, and where it goes."""
    parts = []
    if "overwrite" in config or "overwrite" in template:
        if _setting(config, template, "overwrite"):
            parts.append("overwrites the file")
        else:
            parts.append("stops if the file is already there")
    names = _sheet_names(config.get("sheets"))
    if names:
        parts.append(f"sheets: {', '.join(names)}")
    mode = mode if mode is not None else _setting(config, template, "mode")
    if mode is not None:
        parts.append(f"mode {mode}")
    connection = config.get("connection")
    if connection:
        parts.append(f"database named by {connection}")
    return ", ".join(parts)


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
        tables = config.get("tables")
        if isinstance(config.get("path"), str):
            target = str(Path(root) / config["path"])
            found.append(Write(step_id, step.tool, "file", target, _detail(config, template)))
        elif isinstance(config.get("table"), str):
            target = _table_name(config.get("schema"), config["table"])
            found.append(Write(step_id, step.tool, "table", target, _detail(config, template)))
        elif isinstance(tables, list | tuple) and tables and all(
            isinstance(item, Mapping) and isinstance(item.get("table"), str) for item in tables
        ):
            for item in tables:
                target = _table_name(item.get("schema", config.get("schema")), item["table"])
                detail = _detail(config, template, mode=item.get("mode"))
                found.append(Write(step_id, step.tool, "table", target, detail))
        else:
            found.append(Write(step_id, step.tool, "other", _compact(config), ""))
    return tuple(found)


def writes_digest(found: Sequence[Write]) -> str:
    """A fingerprint of what a real run writes. It changes when any target or detail changes."""
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
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            return _unknown(name, directory, record_file)
    if (directory / MANIFEST).is_file():
        return _unknown(name, directory, directory / MANIFEST)
    return None


def last_runs(runs_folder: Path, name: str) -> tuple[RunRecord | None, RunRecord | None]:
    """The last good run and the last failed or refused one, each ``None`` when there is none."""
    folder = Path(runs_folder)
    return _read(folder / name, name), _read(folder / f"{name}{FAILED_SUFFIX}", name)


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


def _save(project: Project, name: str, good: bool, fill: Callable[[Path], None]) -> Path:
    """Write a run into a hidden folder, then move it into the good or the failed slot."""
    folder = project.runs_folder
    folder.mkdir(parents=True, exist_ok=True)
    incoming = folder / f".{name}.incoming-{secrets.token_hex(6)}"
    incoming.mkdir()
    try:
        fill(incoming)
        target = folder / (name if good else f"{name}{FAILED_SUFFIX}")
        _swap_in(incoming, target)
    except BaseException:
        shutil.rmtree(incoming, ignore_errors=True)
        raise
    failed = folder / f"{name}{FAILED_SUFFIX}"
    if good and failed.exists():
        shutil.rmtree(failed)
    return target


def _write_record(directory: Path, record: Mapping[str, Any]) -> None:
    (directory / RUN_FILE).write_text(json.dumps(record, indent=2), encoding="utf8")


def _record(directory: Path, data: Mapping[str, Any]) -> RunRecord:
    return RunRecord(**{**data, "held_back": tuple(data["held_back"]), "directory": directory})


def _entities(trace: PipelineTrace, focus: FocusSpec) -> dict[str, Any]:
    """Every entity the run can be followed by, as text. A focus that cannot list them says why."""
    found: dict[str, Any] = {"label": focus.label, "columns": list(focus.identity)}
    try:
        found["values"] = [str(option.value) for option in available_entities(trace, focus)]
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
        trace = failure.trace
        data.update(outcome="failed", failed_step=failure.step_id, message=str(failure))
    data["steps_ran"] = sum(1 for event in trace.events if event.step_id != data["failed_step"])
    focus = entry.focus

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

    target = _save(project, entry.name, data["outcome"] == "finished", fill)
    return _record(target, data)
