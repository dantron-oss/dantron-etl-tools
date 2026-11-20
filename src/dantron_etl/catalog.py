"""The tool catalog as plain data: every tool, its ports, its settings, its notes and its worked examples.

Read from the registry, so it cannot describe a tool that does not exist or miss one that does. The
viewer draws it as a page, and ``catalog()`` hands the same data to anything that reads JSON.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import polars as pl

from .config import ConfigField
from .contract import Tool, catalog_input_ports, ports_from_config
from .errors import ToolError, name_choices
from .expr.functions import DEFINITIONS
from .registry import REGISTRY

__all__ = ["CATEGORY_ORDER", "UnknownToolError", "catalog", "describe", "example_yaml", "functions_text", "tools_text"]

#: The order categories are listed in: reading data first, writing it last.
CATEGORY_ORDER = ("input", "preparation", "transform", "combine", "text", "assert", "output")


def _field_json(spec: ConfigField) -> dict[str, Any]:
    """Return one configuration field as plain data, with every property it sets.

    Why: docs/design/core.md#the-catalog-lists-every-field-property
    """
    out: dict[str, Any] = {}
    for field in fields(spec):
        name, value = field.name, getattr(spec, field.name)
        # Skip only None: a default of False or 0 must still be shown.
        if value is None:
            continue
        if name == "items":
            out["items"] = _field_json(value)
        elif name == "fields":
            out["fields"] = {inner: _field_json(f) for inner, f in value.items()}
        elif name == "enum":
            out["allowed"] = list(value)
        else:
            out[name] = value
    out.setdefault("required", spec.required)
    out.setdefault("description", spec.description)
    return out


def _frame_json(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "rows": frame.to_dicts(),
    }


def catalog() -> dict[str, Any]:
    """Return the whole catalog as plain data (dicts and lists), ready to write as JSON.

    One entry per tool: name, description, category, ports, settings, notes and worked examples.
    """
    tools = []
    for tool in REGISTRY.tools():
        helped = tool.help()
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                # For a tool whose ports come from its settings, these are its first example's.
                "input_ports": list(catalog_input_ports(tool)),
                "ports_from_config": ports_from_config(tool),
                "output_ports": list(tool.output_ports),
                "detail": helped.detailed_description.strip(),
                "notes": list(helped.notes),
                "config": {name: _field_json(spec) for name, spec in tool.config_template().items()},
                "field_help": dict(helped.field_help),
                "examples": [
                    {
                        "scenario": ex.scenario,
                        "configuration": dict(ex.configuration),
                        "explanation": ex.explanation,
                        "row_order_significant": getattr(ex, "check_row_order", False),
                        "inputs": {name: _frame_json(f) for name, f in ex.inputs.items()},
                        "outputs": {name: _frame_json(f) for name, f in ex.outputs.items()},
                    }
                    for ex in helped.examples
                ],
            }
        )
    return {"tool_count": len(tools), "tools": tools}


# ---- the catalog as terminal text -------------------------------------------------------------

WIDTH = 100
EXAMPLE_MARKER = "Example settings (YAML):"


class UnknownToolError(ToolError):
    """A tool name that is not in the catalog."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(inner) for inner in value]
    return value


def example_yaml(configuration: Mapping[str, Any]) -> str:
    """Return a worked example's configuration as YAML, ready to paste under a step's ``config:``."""
    import yaml

    return yaml.safe_dump(_plain(configuration), sort_keys=False, allow_unicode=True, width=WIDTH - 8)


def _wrap(text: str, indent: str = "", first: str | None = None) -> list[str]:
    """Wrap prose to the terminal, one paragraph at a time, never breaking a word."""
    out: list[str] = []
    for paragraph in text.strip().split("\n\n"):
        out += textwrap.wrap(
            " ".join(paragraph.split()),
            WIDTH,
            initial_indent=indent if first is None or out else first,
            subsequent_indent=indent,
            break_on_hyphens=False,
            break_long_words=False,
        )
    return out


def _shown(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value == "":
        return '""'
    return str(value)


def _table(header: list[str], rows: list[list[str]], indent: str = "  ") -> list[str]:
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]

    def line(cells: list[str]) -> str:
        return (indent + "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))).rstrip()

    return [line(header), line(["-" * width for width in widths]), *(line(row) for row in rows)]


def _setting_rows(name: str, spec: ConfigField) -> list[tuple[list[str], str]]:
    """One row per field, then the fields nested inside it, each with its description."""
    if spec.required:
        need = "required"
    elif spec.default is not None:
        need = f"default {_shown(spec.default)}"
    else:
        need = "optional"
    allowed = ", ".join(_shown(value) for value in spec.enum or ())
    rows = [([name, spec.type, need, allowed], spec.description)]
    if spec.items is not None and spec.items.fields:
        for inner, inner_spec in spec.items.fields.items():
            rows += _setting_rows(f"{name}[].{inner}", inner_spec)
    elif spec.items is not None:
        rows += _setting_rows(f"{name}[]", spec.items)
    elif spec.fields:
        for inner, inner_spec in spec.fields.items():
            rows += _setting_rows(f"{name}.{inner}", inner_spec)
    return rows


def _settings(tool: Tool, field_help: Mapping[str, str]) -> list[str]:
    template = tool.config_template()
    if not template:
        return ["Settings: none."]
    rows = [row for name, spec in template.items() for row in _setting_rows(name, spec)]
    lines = _table(["name", "type", "required or default", "allowed values"], [cells for cells, _ in rows])
    out = ["Settings:", *lines[:2]]
    for row_line, (cells, description) in zip(lines[2:], rows, strict=True):
        out.append(row_line)
        out += _wrap(description, "      ")
        if cells[0] in field_help:
            out += _wrap(field_help[cells[0]], "      ")
    return out


def _frame_lines(port: str, frame: pl.DataFrame) -> list[str]:
    rows = [[_shown(value) for value in row] for row in frame.rows()]
    lines = [f"  {port}:"]
    if not frame.columns:
        return [*lines, "    (no columns)"]
    lines += _table(list(frame.columns), rows, indent="    ")
    if not rows:
        lines.append("    (no rows)")
    return lines


def _find_tool(name: str) -> Tool:
    for tool in REGISTRY.tools():
        if tool.name == name:
            return tool
    names = sorted(tool.name for tool in REGISTRY.tools())
    raise UnknownToolError(f"there is no tool called {name!r}; {name_choices(name, names, 'tools')}")


def describe(name: str) -> str:
    """Return everything the catalog says about one tool, as plain text for a terminal.

    Raises :class:`UnknownToolError`, with the closest names, when there is no tool called ``name``.
    """
    tool = _find_tool(name)
    helped = tool.help()
    out = [f"{tool.name}  ({tool.category} tool)", "", *_wrap(tool.description)]
    out += ["", *_wrap(helped.detailed_description)]
    out += ["", f"Reads:  {', '.join(catalog_input_ports(tool)) or 'nothing'}"]
    if ports_from_config(tool):
        out += ["        (these come from the settings: a step reads the ports its own settings name)"]
    out += [f"Gives:  {', '.join(tool.output_ports) or 'nothing'}", "", *_settings(tool, helped.field_help)]
    if helped.notes:
        out += ["", "Notes:"]
        for note in helped.notes:
            out += _wrap(note, "    ", first="  - ")
    if helped.examples:
        example = helped.examples[0]
        out += ["", "Example:", *_wrap(example.scenario, "  "), "", EXAMPLE_MARKER]
        out += ["    " + line for line in example_yaml(example.configuration).splitlines()]
        out += ["", "Rows in:"]
        for port, frame in example.inputs.items():
            out += _frame_lines(port, frame)
        out += ["", "Rows out:"]
        for port, frame in example.outputs.items():
            out += _frame_lines(port, frame)
        if example.explanation:
            out += ["", *_wrap(example.explanation, "  ")]
    return "\n".join(out) + "\n"


def tools_text() -> str:
    """Return every tool on one line each, grouped by category, as plain text."""
    by_category: dict[str, list[Tool]] = {}
    for tool in sorted(REGISTRY.tools(), key=lambda tool: tool.name):
        by_category.setdefault(tool.category, []).append(tool)
    order = [c for c in CATEGORY_ORDER if c in by_category] + sorted(set(by_category) - set(CATEGORY_ORDER))
    width = max(len(tool.name) for tools in by_category.values() for tool in tools)
    out: list[str] = []
    for category in order:
        out += [category, *(f"  {tool.name.ljust(width)}  {tool.description}" for tool in by_category[category]), ""]
    return "\n".join(out)


def functions_text() -> str:
    """Return every expression function with what it gives and one example call, as plain text."""
    out: list[str] = []
    for definition in DEFINITIONS:
        out += [definition.name, *_wrap(definition.summary, "    "), f"    example: {definition.example}", ""]
    return "\n".join(out)
