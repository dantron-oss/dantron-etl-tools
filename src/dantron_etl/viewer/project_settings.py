"""Reading the parts of a project's settings: YAML files, focus and rules.

These helpers serve ``project.py``. Each refusal is a ProjectError that names the file or the
pipeline, what was found, and what to write instead.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..errors import ToolError
from ..focus import FocusSpec, RelatedKey
from ..loader import yaml_loader
from ..spec import PortRef

SETTINGS_FILE = "etl-project.yaml"

_FOCUS_KEYS = ("columns", "via")
_VIA_KEYS = ("from", "key")


class ProjectError(ToolError):
    """A problem with a project or its settings. The message says what to change."""


def read_yaml(path: Path) -> Any:
    """The YAML document in ``path``, or a ProjectError naming the file and what went wrong."""
    import yaml

    try:
        return yaml.load(path.read_text(encoding="utf8"), Loader=yaml_loader())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProjectError(f"{path} could not be read as YAML: {error}") from error


def refuse_unknown(found: Mapping[Any, Any], allowed: tuple[str, ...], where: str) -> None:
    """Refuse any key not in ``allowed``, naming it and listing the allowed keys."""
    unknown = [str(key) for key in found if key not in allowed]
    if unknown:
        raise ProjectError(
            f"{where} has {', '.join(repr(key) for key in unknown)}, which is not a setting. "
            f"The settings allowed there are: {', '.join(allowed)}."
        )


# ---------------------------------------------------------------------------
# Focus
# ---------------------------------------------------------------------------


def parse_focus(value: Any, name: str, where: str) -> FocusSpec | None:
    """The focus written for a pipeline: a column, a list of columns, or columns with ``via``."""
    if value is None:
        return None
    example = "'focus: customer', 'focus: [account, customer]' or 'focus: {columns: [account], via: {...}}'"
    via: Any = {}
    columns = value
    if isinstance(value, Mapping):
        refuse_unknown(value, _FOCUS_KEYS, f"'focus' for {where}")
        columns = value.get("columns")
        via = value.get("via") or {}
    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) and c for c in columns):
        raise ProjectError(f"'focus' for {where} is {value!r}. Name the columns that identify one entity: {example}.")
    if not isinstance(via, Mapping):
        raise ProjectError(
            f"'via' in the focus for {where} is {via!r}. Write 'step.port': {{from: step.port, key: column}}."
        )
    related = {_port_ref(port, where): _related_key(rule, port, where) for port, rule in via.items()}
    try:
        return FocusSpec(label=columns[0], identity=tuple(columns), via=related)
    except ToolError as error:
        raise ProjectError(f"The focus for {where} is refused: {error}") from error


def _port_ref(text: Any, where: str) -> PortRef:
    """``"step.port"`` as a PortRef, or a refusal naming the text."""
    step, dot, port = text.partition(".") if isinstance(text, str) else ("", "", "")
    if not (step and dot and port) or "." in port:
        raise ProjectError(
            f"The focus for {where} names port {text!r}. Write a port as 'step.port', for example 'orders.rows'."
        )
    return PortRef(step, port)


def _related_key(rule: Any, port: Any, where: str) -> RelatedKey:
    if not isinstance(rule, Mapping) or set(rule) != set(_VIA_KEYS) or not isinstance(rule["key"], str):
        raise ProjectError(
            f"'via' for {port!r} in the focus for {where} is {rule!r}. Write {{from: step.port, key: column}}."
        )
    return RelatedKey(_port_ref(rule["from"], where), rule["key"])


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def rule_mapping(value: Any, where: str) -> dict[str, Any]:
    """A mapping of rule name to value, or a refusal naming where it was written."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProjectError(f"{where} is {value!r}. Write rules as 'name: value' lines, for example 'status: PAID'.")
    return dict(value)


def rules_from_file(root: Path, name: str, written: Any) -> dict[str, Any]:
    """The rules in a rules file, in the plain form or the "value and why" form, never mixed."""
    if not isinstance(written, str) or not written:
        raise ProjectError(
            f"'rules_file' for pipeline '{name}' in {SETTINGS_FILE} is {written!r}. "
            f"Write a path from the project folder, for example 'rules_file: config/{name}_rules.yaml'."
        )
    path = root / written
    if not path.is_file():
        raise ProjectError(
            f"'rules_file' for pipeline '{name}' in {SETTINGS_FILE} names {written}, which does not exist."
        )
    rules = rule_mapping(read_yaml(path) or {}, f"the rules file {written}")
    explained = [isinstance(value, Mapping) and "value" in value for value in rules.values()]
    if all(explained):
        return {key: value["value"] for key, value in rules.items()}
    if any(explained):
        plain = sorted(key for key, flag in zip(rules, explained, strict=True) if not flag)
        raise ProjectError(
            f"The rules file {written} mixes two forms. Most entries are 'name: {{value: X, why: ...}}', but "
            f"{', '.join(plain)} {'is' if len(plain) == 1 else 'are'} plain 'name: value'. Use one form for every entry."
        )
    return rules


def rules_from_code(root: Path, name: str, loader: str) -> dict[str, Any]:
    """Import and call a project's rules loader. This runs project code.

    The project root is put at the front of ``sys.path`` while importing, and taken off afterwards.
    """
    module_name, _, function_name = loader.partition(":")
    folder = str(root)
    sys.path.insert(0, folder)
    try:
        module = importlib.import_module(module_name)
        rules = getattr(module, function_name)()
    except Exception as error:
        raise ProjectError(
            f"The rules loader {loader} for pipeline '{name}' failed: {type(error).__name__}: {error}"
        ) from error
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(folder)
    if not isinstance(rules, Mapping):
        raise ProjectError(
            f"The rules loader {loader} for pipeline '{name}' returned {type(rules).__name__}. "
            f"It must return a mapping of rule name to value, such as {{'status': 'PAID'}}."
        )
    return rule_mapping(rules, f"What {loader} returned for pipeline '{name}'")
