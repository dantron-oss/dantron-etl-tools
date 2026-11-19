"""Finding a project folder and the pipelines in it.

A project is a folder. Its pipelines are YAML files in one or more pipeline folders. An optional
settings file, ``etl-project.yaml``, says where those folders are, where runs are saved, and which
rules and focus each pipeline uses.

Reading a project only reads YAML. Project code runs only when a pipeline's rules are asked for,
and only when the settings name a rules loader.
"""

from __future__ import annotations

import contextlib
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..focus import FocusSpec
from ..loader import read_pipeline
from ..paths import project_root
from ..spec import Pipeline
from .project_settings import (
    SETTINGS_FILE,
    ProjectError,
    parse_focus,
    read_yaml,
    refuse_unknown,
    rule_mapping,
    rules_from_code,
    rules_from_file,
)

__all__ = [
    "DEFAULT_PIPELINE_FOLDERS",
    "DEFAULT_RUNS_FOLDER",
    "PIPELINE_NAME",
    "RESERVED_NAMES",
    "SETTINGS_FILE",
    "PipelineEntry",
    "Project",
    "ProjectError",
    "find_project",
    "load_entry",
    "load_project",
    "pipeline_rules",
]

DEFAULT_PIPELINE_FOLDERS = ("pipelines",)
DEFAULT_RUNS_FOLDER = "output_files/runs"
PIPELINE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

#: Names the viewer uses for its own addresses, so a pipeline cannot have them.
RESERVED_NAMES = frozenset({"api", "catalog"})

#: The older settings file, kept beside a pipeline's inputs: ``<folder>/<name>/settings.yaml``.
OLD_SETTINGS_FILE = "settings.yaml"

_TOP_KEYS = ("pipeline_folders", "runs_folder", "pipelines")
_ENTRY_KEYS = ("rules", "rules_file", "rules_loader", "focus")
_RULES_KEYS = ("rules", "rules_file", "rules_loader")
_OLD_KEYS = ("rules", "focus")
_RULE_REFERENCE = re.compile(r"\$\{rules\.([^}]*)\}")
_LOADER_NAME = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")
_SUFFIXES = (".yaml", ".yml")

#: Held while a pipeline is read with the working folder changed. The working folder belongs to the
#: whole process, and the server loads pipelines from several threads.
_CHDIR_LOCK = threading.Lock()


@dataclass(frozen=True)
class PipelineEntry:
    """One pipeline file, as read from its YAML. Nothing here has been checked by the loader."""

    name: str
    path: Path
    title: str
    steps: int
    rules_needed: tuple[str, ...]
    rules_source: str | None
    focus: FocusSpec | None
    settings: Mapping[str, Any]
    problem: str | None


@dataclass(frozen=True)
class Project:
    """A project folder, its settings and the pipelines found in it."""

    root: Path
    settings_path: Path | None
    pipeline_folders: tuple[Path, ...]
    runs_folder: Path
    pipelines: Mapping[str, PipelineEntry]
    problems: tuple[str, ...]


def find_project(start: Path | str) -> Project:
    """The project holding ``start``, found by searching upward for a project marker."""
    return load_project(project_root(start))


def load_project(root: Path) -> Project:
    """Read a project's settings and list its pipelines.

    This reads YAML and nothing else. It never imports project code and never resolves rules. A
    problem with one pipeline is kept on that entry, so the others still show.
    """
    root = Path(root).resolve()
    settings_path = root / SETTINGS_FILE
    if not settings_path.is_file():
        settings_path = None
    settings = _read_settings(settings_path)
    folders = tuple(root / folder for folder in _text_list(settings, "pipeline_folders", settings_path))
    runs = _text(settings, "runs_folder", DEFAULT_RUNS_FOLDER, settings_path)
    configured = _configured_pipelines(settings, settings_path)

    problems: list[str] = []
    found = _discover(root, folders, problems)
    entries = {name: _entry(root, name, path, configured.get(name)) for name, path in sorted(found.items())}
    for name in configured:
        if name not in entries:
            problems.append(
                f"{SETTINGS_FILE} has settings for pipeline '{name}', but no pipeline file is called "
                f"{name}.yaml in {', '.join(_shown(folder, root) for folder in folders)}. "
                f"Rename the entry or add the file."
            )
    return Project(
        root=root,
        settings_path=settings_path,
        pipeline_folders=folders,
        runs_folder=root / runs,
        pipelines=MappingProxyType(entries),
        problems=tuple(problems),
    )


def pipeline_rules(project: Project, entry: PipelineEntry) -> dict[str, Any]:
    """The values for a pipeline's ``${rules.*}`` references.

    This may run project code. A ``rules_loader`` in the settings is imported and called here.
    """
    if entry.problem:
        raise ProjectError(entry.problem)
    settings = entry.settings
    if "rules" in settings:
        return rule_mapping(settings["rules"] or {}, f"'rules' for pipeline '{entry.name}'")
    if "rules_file" in settings:
        return rules_from_file(project.root, entry.name, settings["rules_file"])
    if "rules_loader" in settings:
        return rules_from_code(project.root, entry.name, settings["rules_loader"])
    if entry.rules_needed:
        raise ProjectError(_missing_rules_hint(entry))
    return {}


def load_entry(project: Project, entry: PipelineEntry) -> Pipeline:
    """Load a pipeline with its rules, ready to run.

    The pipeline is read with the working folder set to the project root. A writer checks its output
    folder relative to the working folder, so this makes a relative path mean the project's folder.
    The working folder belongs to the whole process, so a lock is held while it is changed. This may
    run project code: see :func:`pipeline_rules`.
    """
    rules = pipeline_rules(project, entry)
    with _CHDIR_LOCK, contextlib.chdir(project.root):
        return read_pipeline(entry.path, rules=rules)


# ---------------------------------------------------------------------------
# Reading the settings file
# ---------------------------------------------------------------------------


def _read_settings(path: Path | None) -> Mapping[str, Any]:
    """The project settings file as a mapping. A missing or empty file is no settings."""
    if path is None:
        return {}
    document = read_yaml(path)
    if document is None:
        return {}
    if not isinstance(document, Mapping):
        raise ProjectError(
            f"{path} holds {type(document).__name__}, not settings. Write settings as 'name: value' lines, "
            f"for example 'pipeline_folders: [pipelines]'."
        )
    refuse_unknown(document, _TOP_KEYS, str(path))
    return document


def _text_list(settings: Mapping[str, Any], key: str, path: Path | None) -> tuple[str, ...]:
    value = settings.get(key, list(DEFAULT_PIPELINE_FOLDERS))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ProjectError(
            f"'{key}' in {path} is {value!r}. Write a list of folder names, for example '{key}: [pipelines]'."
        )
    return tuple(value)


def _text(settings: Mapping[str, Any], key: str, default: str, path: Path | None) -> str:
    value = settings.get(key, default)
    if not isinstance(value, str) or not value:
        raise ProjectError(f"'{key}' in {path} is {value!r}. Write a folder name, for example '{key}: {default}'.")
    return value


def _configured_pipelines(settings: Mapping[str, Any], path: Path | None) -> dict[str, Any]:
    """The ``pipelines:`` section, by name. Each value is checked later, on its own entry."""
    section = settings.get("pipelines") or {}
    if not isinstance(section, Mapping):
        raise ProjectError(
            f"'pipelines' in {path} is {section!r}. Write one entry per pipeline name, for example\n"
            f"pipelines:\n  orders:\n    focus: customer"
        )
    return {str(name): value for name, value in section.items()}


# ---------------------------------------------------------------------------
# Finding pipeline files
# ---------------------------------------------------------------------------


def _shown(path: Path, root: Path) -> str:
    """``path`` relative to the root when it is inside it, for messages."""
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


def _discover(root: Path, folders: tuple[Path, ...], problems: list[str]) -> dict[str, Path]:
    """Pipeline files by name, directly inside each folder. The first folder wins a shared name."""
    found: dict[str, Path] = {}
    by_folded: dict[str, Path] = {}
    for folder in folders:
        if not folder.is_dir():
            problems.append(f"The pipeline folder {_shown(folder, root)} does not exist.")
            continue
        for path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix in _SUFFIXES):
            name = path.stem
            if not PIPELINE_NAME.fullmatch(name):
                problems.append(
                    f"Skipped {_shown(path, root)}: '{name}' is not a pipeline name. "
                    f"Use letters, digits, '_' and '-' only, starting with a letter or digit."
                )
            elif name in RESERVED_NAMES:
                problems.append(
                    f"Skipped {_shown(path, root)}: '{name}' is reserved for the viewer's own pages. Rename the file."
                )
            elif name.casefold() in by_folded:
                first = by_folded[name.casefold()]
                same = "" if first.stem == name else " (names that differ only in capitals are the same name)"
                problems.append(
                    f"Skipped {_shown(path, root)}: a pipeline called '{first.stem}' was already found at "
                    f"{_shown(first, root)}{same}. Rename one of them."
                )
            elif _is_pipeline_or_unreadable(path):
                found[name] = path.resolve()
                by_folded[name.casefold()] = path
            else:
                problems.append(f"Skipped {_shown(path, root)}: it has no 'steps' list, so it is not a pipeline.")
    return found


def _is_pipeline_or_unreadable(path: Path) -> bool:
    """Whether the file is a pipeline, or is not readable YAML (listed with a problem)."""
    try:
        document = read_yaml(path)
    except ProjectError:
        return True
    return isinstance(document, Mapping) and isinstance(document.get("steps"), list)


# ---------------------------------------------------------------------------
# One pipeline entry
# ---------------------------------------------------------------------------


def _entry(root: Path, name: str, path: Path, configured: Any) -> PipelineEntry:
    """The entry for one pipeline file. A problem is kept on the entry, never raised."""
    try:
        text = path.read_text(encoding="utf8")
        document = read_yaml(path)
    except (OSError, UnicodeDecodeError, ProjectError) as error:
        return PipelineEntry(name, path, name, 0, (), None, None, MappingProxyType({}), str(error))
    title = document.get("name")
    needed = tuple(sorted(set(_RULE_REFERENCE.findall(text))))
    base = {
        "name": name,
        "path": path,
        "title": title if isinstance(title, str) and title else name,
        "steps": len(document["steps"]),
        "rules_needed": needed,
    }
    try:
        settings, source = _entry_settings(root, name, path, configured)
        focus = parse_focus(settings.get("focus"), name, source)
    except ProjectError as error:
        return PipelineEntry(**base, rules_source=None, focus=None, settings=MappingProxyType({}), problem=str(error))
    return PipelineEntry(
        **base,
        rules_source=_rules_source(settings, source),
        focus=focus,
        settings=MappingProxyType(dict(settings)),
        problem=None,
    )


def _entry_settings(root: Path, name: str, path: Path, configured: Any) -> tuple[Mapping[str, Any], str]:
    """The pipeline's settings and where they were written. Settings in two places are refused."""
    old_path = path.parent / name / OLD_SETTINGS_FILE
    old_shown = _shown(old_path, root)
    if configured is not None and old_path.is_file():
        raise ProjectError(
            f"Pipeline '{name}' has settings in {SETTINGS_FILE} and in {old_shown}. "
            f"Keep one: move what {old_shown} holds under 'pipelines: {name}:' in {SETTINGS_FILE} and delete it."
        )
    if old_path.is_file():
        old = read_yaml(old_path) or {}
        if not isinstance(old, Mapping):
            raise ProjectError(f"{old_shown} holds {type(old).__name__}. Write 'rules:' and 'focus:' in it.")
        refuse_unknown(old, _OLD_KEYS, old_shown)
        return old, old_shown
    where = f"pipeline '{name}' in {SETTINGS_FILE}"
    if configured is None:
        return {}, where
    if not isinstance(configured, Mapping):
        raise ProjectError(f"{where} is {configured!r}. Write its settings as 'rules:' and 'focus:' under the name.")
    refuse_unknown(configured, _ENTRY_KEYS, where)
    used = [key for key in _RULES_KEYS if key in configured]
    if len(used) > 1:
        raise ProjectError(
            f"{where} sets {' and '.join(used)}. A pipeline takes its rules from one place: keep one of "
            f"{', '.join(_RULES_KEYS)}."
        )
    if "rules" in configured:
        rule_mapping(configured["rules"] or {}, f"'rules' for {where}")
    if "rules_file" in configured:
        rules_from_file(root, name, configured["rules_file"])
    if "rules_loader" in configured and not (
        isinstance(configured["rules_loader"], str) and _LOADER_NAME.fullmatch(configured["rules_loader"])
    ):
        raise ProjectError(
            f"'rules_loader' for {where} is {configured['rules_loader']!r}. "
            f"Write the module and function as 'package.module:function'."
        )
    return configured, where


def _rules_source(settings: Mapping[str, Any], source: str) -> str | None:
    """Where the rules come from, in plain words, or None when no rules are set."""
    if "rules" in settings:
        return "settings: inline" if source.startswith("pipeline ") else source
    if "rules_file" in settings:
        return f"file {settings['rules_file']}"
    if "rules_loader" in settings:
        return f"project code {settings['rules_loader']}"
    return None


def _missing_rules_hint(entry: PipelineEntry) -> str:
    """The refusal for a pipeline that uses rules when none are set, with the YAML to add."""
    names = [f"${{rules.{rule}}}" for rule in entry.rules_needed]
    used = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"
    lines = "\n".join(f"      {rule}: <value>" for rule in entry.rules_needed)
    return (
        f"pipeline '{entry.name}' uses {used} and no rules are set for it. Add to {SETTINGS_FILE}:\n"
        f"pipelines:\n  {entry.name}:\n    rules:\n{lines}"
    )
