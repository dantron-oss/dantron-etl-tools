"""The tool catalog as plain data: every tool, its ports, its settings, its notes and its worked examples.

Read from the registry, so it cannot describe a tool that does not exist or miss one that does. The
viewer draws it as a page, and ``catalog()`` hands the same data to anything that reads JSON.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import polars as pl

from .config import ConfigField
from .contract import catalog_input_ports, ports_from_config
from .registry import REGISTRY

__all__ = ["CATEGORY_ORDER", "catalog"]

#: The order categories are listed in: reading data first, writing it last.
CATEGORY_ORDER = ("input", "preparation", "transform", "combine", "text", "assert", "output")


def _field_json(spec: ConfigField) -> dict[str, Any]:
    """One configuration field, flattened for a reader that is not Python.

    Built from the dataclass rather than field by field, so a property added to a configuration
    field reaches the catalogue because it exists rather than because somebody remembered a line
    here. The hand-written version had three branches whose false arm was "say nothing", which is
    a silent omission by construction and exactly the drift this page cannot afford.
    """
    out: dict[str, Any] = {}
    for field in fields(spec):
        name, value = field.name, getattr(spec, field.name)
        # Only absence is skipped. Skipping a falsey value would drop a meaningful
        # default=False -- the very silent omission this rewrite exists to remove.
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
    """The whole catalogue as plain data -- the model-facing interface, and diffable."""
    tools = []
    for tool in REGISTRY.tools():
        helped = tool.help()
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                # A tool that works its ports out from a step's configuration has no fixed ports to
                # show, so the catalogue shows the ones its first worked example reads and says so
                # with the flag beside them. Both halves are needed: the names, and where they
                # came from.
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


