"""What files and tables a pipeline reads and writes, so a chain can check its order against them.

A file is matched by its full path, from the project folder. A table is matched by the environment
variable naming its connection, its schema and its name. Both are compared without regard to
capitals, and a table written without a schema matches the same table in any schema: a hand-off
taken for one that is not there costs a refusal to read, while one missed costs a wrong figure.

Why: docs/design/chains.md#after-is-written-down-and-checked-against-the-hand-offs
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec import Pipeline

__all__ = ["READER_CATEGORY", "WRITER_CATEGORY", "Target", "Touch", "touches"]

READER_CATEGORY = "input"
WRITER_CATEGORY = "output"


@dataclass(frozen=True)
class Target:
    """A file or a table, as a pipeline names it.

    ``kind`` is ``"file"`` or ``"table"``. ``shown`` is how a person would write it. For a file,
    ``where`` is its full path; for a table, it is the connection's environment variable name.
    """

    kind: str
    shown: str
    where: str
    schema: str | None = None
    table: str | None = None

    def matches(self, other: Target) -> bool:
        """Whether the two name the same file or the same table."""
        if self.kind != other.kind or self.where.casefold() != other.where.casefold():
            return False
        if self.kind == "file":
            return True
        assert self.table is not None and other.table is not None
        if self.table.casefold() != other.table.casefold():
            return False
        return self.schema is None or other.schema is None or self.schema.casefold() == other.schema.casefold()


@dataclass(frozen=True)
class Touch:
    """One step reading or writing one file or table."""

    step: str
    tool: str
    target: Target


def touches(spec: Pipeline, root: Path) -> tuple[tuple[Touch, ...], tuple[Touch, ...], tuple[str, ...]]:
    """What a loaded pipeline reads and what it writes, and the steps whose target cannot be told.

    Only tools that read (category ``input``) and tools that write (``output``) touch files and
    tables. A writer's ``template`` is a file it reads.
    """
    registry = spec.registry
    assert registry is not None  # a Pipeline always fills it in
    reads: list[Touch] = []
    written: list[Touch] = []
    unknown: list[str] = []
    for step_id in spec.order():
        step = spec.step(step_id)
        category = registry.get(step.tool).category
        if category not in (READER_CATEGORY, WRITER_CATEGORY):
            continue
        config = step.config
        found = _targets(config, root)
        if not found:
            unknown.append(step_id)
            continue
        into = reads if category == READER_CATEGORY else written
        into.extend(Touch(step_id, step.tool, target) for target in found)
        template = config.get("template")
        if category == WRITER_CATEGORY and isinstance(template, str) and template:
            reads.append(Touch(step_id, step.tool, _file(template, root)))
    return tuple(reads), tuple(written), tuple(unknown)


def _targets(config: Mapping[str, Any], root: Path) -> list[Target]:
    """The files or tables one step's settings name."""
    if isinstance(config.get("path"), str) and config["path"]:
        return [_file(config["path"], root)]
    connection = config.get("connection")
    if not isinstance(connection, str) or not connection:
        return []
    if isinstance(config.get("table"), str) and config["table"]:
        return [_table(connection, config.get("schema"), config["table"])]
    tables = config.get("tables")
    if isinstance(tables, list | tuple) and tables:
        found = []
        for item in tables:
            if not (isinstance(item, Mapping) and isinstance(item.get("table"), str)):
                return []
            found.append(_table(connection, item.get("schema", config.get("schema")), item["table"]))
        return found
    return []


def _file(written: str, root: Path) -> Target:
    full = os.path.normpath(Path(root) / written)
    shown = full if Path(written).is_absolute() else Path(os.path.normpath(written)).as_posix()
    return Target("file", shown, full)


def _table(connection: str, schema: Any, table: str) -> Target:
    schema = schema if isinstance(schema, str) and schema else None
    name = f"{schema}.{table}" if schema else table
    return Target("table", f"table {name} ({connection})", connection, schema, table)
