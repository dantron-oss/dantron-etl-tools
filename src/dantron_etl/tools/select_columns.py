"""SelectColumns: keep or drop columns named by a list of names or `*` patterns.

Under `keep` the output follows the list's order; under `drop` it keeps the frame's order. Every
entry must match a column. Only `*` is a wildcard, and matching is case sensitive.

Why: docs/design/tools.md#select_columns
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
)

_ACTION_ENUM: Final[tuple[str, ...]] = ("keep", "drop")


class SelectColumns:
    """Keep or drop the columns of `rows` that a list names. The rows are untouched."""

    name = "select_columns"
    description = "Keep or remove columns by name, and put the kept ones in the order you list."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="The columns to keep or drop, by name or by a pattern using `*`.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One column name, or a pattern where `*` stands for any characters.",
                ),
            ),
            "action": ConfigField(
                type="string",
                description="'keep' to keep only the listed columns, 'drop' to remove them.",
                default="keep",
                enum=_ACTION_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use select_columns to narrow a table to the columns later steps need, or to remove "
                "the ones you are finished with. The rows are not changed. With 'keep', the output "
                "has the listed columns in the order you list them, so it also reorders. With 'drop', "
                "the remaining columns keep the table's own order."
            ),
            examples=(_wildcard_example(), _drop_example()),
            notes=(
                "Every name or pattern must match at least one column, or the step is refused. A "
                "pattern that quietly matched nothing would hide a column that was renamed upstream.",
                "Only `*` is a wildcard. Brackets and `?` are part of the name, so 'Total [CAD]' "
                "selects exactly that column.",
                "Matching is case sensitive.",
                "A column matched by two entries is kept once, where it first matched.",
                "A 'drop' that would remove every column is refused.",
            ),
            field_help={
                "columns": "For example ['code*', 'amount']: every column starting with 'code', then "
                "'amount'.",
                "action": "Use 'drop' when it is shorter to list what goes than what stays.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse an entry that matches no column, or a drop that removes every column."""
        _selection(inputs["rows"].columns, config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        return Ports({"rows": frame.select(_selection(frame.columns, config))})


def _selection(available: Sequence[str], config: Mapping[str, Any]) -> list[str]:
    """The columns the output keeps, in order, refusing an entry that matches nothing or a drop of everything."""
    selected: list[str] = []
    seen: set[str] = set()
    for pattern in config["columns"]:
        matches = _matches(available, pattern)
        if not matches:
            raise ConfigError(
                f"config field 'columns' pattern {pattern!r} matches no column of the input; the "
                f"columns present are: {', '.join(available)}"
            )
        for name in matches:
            if name not in seen:
                seen.add(name)
                selected.append(name)

    if config["action"] == "drop":
        dropped = set(selected)
        selected = [name for name in available if name not in dropped]

    if not selected:
        raise ConfigError("config field 'columns' selects nothing: 'action: drop' removed every column of the input")

    return selected


def _matches(available: Sequence[str], entry: str) -> list[str]:
    """The columns one entry selects: itself if it is a column's name, else what its `*`s match."""
    if entry in available:
        return [entry]
    if "*" not in entry:
        return []
    pattern = re.compile(".*".join(re.escape(part) for part in entry.split("*")), re.DOTALL)
    return [name for name in available if pattern.fullmatch(name)]


def _wildcard_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "region": ["north", "south"],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="keep the code columns and the amount, dropping the rest",
        configuration={"columns": ["code*", "amount"]},
        explanation="'code*' matches 'code' and 'code_suffix', in the table's order. 'amount' comes next because "
        "it is listed second. 'region' is not listed, so it is dropped.",
        inputs=inputs,
        outputs=outputs,
    )


def _drop_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "region": ["north", "south"],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="remove the region column",
        configuration={"columns": ["region"], "action": "drop"},
        explanation="Only 'region' matches, so only it is removed. The other columns keep the order the table "
        "had.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(SelectColumns())
