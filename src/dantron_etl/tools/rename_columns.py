"""RenameColumns: rename columns and keep each one where it is.

The renames are applied together, so two entries can swap names. A new name that would overwrite a
column that stays is refused, and so is a rename that changes nothing.

Why: docs/design/tools.md#rename_columns
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
)

_FROM = "from"
_TO = "to"


class RenameColumns:
    """Rename columns of `rows`: same rows, same order, same types, new names."""

    name = "rename_columns"
    description = "Give columns new names, keeping each column where it is."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="The renames, each giving a column's current name and its new name.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One rename: the column's current name and its new name.",
                    fields={
                        _FROM: ConfigField(
                            type="string",
                            description="The column's current name, such as 'region'.",
                            required=True,
                        ),
                        _TO: ConfigField(
                            type="string",
                            description="The column's new name, such as 'territory'.",
                            required=True,
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use rename_columns to give columns clearer or matching names, for example before a "
                "join or a union. Only the names change: each column keeps its position, type and "
                "values. All the renames happen at once, so two columns can swap names in one step."
            ),
            examples=(_in_place_example(), _exchange_example()),
            notes=(
                "A new name that is already used by a column that stays is refused, because that "
                "column would be lost. A name that another entry renames away in the same step is free.",
                "Renaming a column to its own name is refused, as is a blank or all-space name. "
                "These usually mean the configuration was edited halfway.",
                "A column to rename that is not on the input is refused.",
            ),
            field_help={
                "columns": "For example [{from: region, to: territory}]. The order of the entries "
                "does not matter, and the output keeps the table's column order.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse what the configuration alone shows: blank names, a rename to the same name, repeats.

        Blank names are refused here so the refusal can cite the entry's position instead of a column name.
        """
        _renames(config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a column that is not on the input, and a new name that collides with a column that stays."""
        _resolved(inputs["rows"].columns, config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        mapping = _resolved(frame.columns, config)
        return Ports({"rows": frame.rename(mapping)})


def _renames(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The `(old, new)` pairs, refusing what the configuration alone shows is wrong."""
    pairs: list[tuple[str, str]] = []
    taken: dict[str, int] = {}
    renamed: dict[str, int] = {}

    for index, entry in enumerate(config["columns"]):
        old, new = entry[_FROM], entry[_TO]
        for value, field in ((old, _FROM), (new, _TO)):
            if not value.strip():
                raise ConfigError(
                    f"config field 'columns[{index}]' states a blank {field!r} name; a rename must "
                    f"name a column and the name it is to carry"
                )
        if old == new:
            raise ConfigError(
                f"config field 'columns' renames {old!r} to the name it already carries; a rename "
                f"that changes nothing is a configuration edited halfway"
            )
        if old in renamed:
            raise ConfigError(
                f"config field 'columns' renames {old!r} twice, at entries {renamed[old]} and "
                f"{index}; a column has one new name or none"
            )
        if new in taken:
            raise ConfigError(
                f"config field 'columns' gives the name {new!r} to two columns, at entries "
                f"{taken[new]} and {index}; each new name must be claimed once"
            )
        renamed[old] = index
        taken[new] = index
        pairs.append((old, new))

    return pairs


def _resolved(available: Sequence[str], config: Mapping[str, Any]) -> dict[str, str]:
    """The rename mapping, checked against the incoming columns.

    A new name collides only with a column that stays; one being renamed away in this step is free.
    """
    pairs = _renames(config)
    vacated = {old for old, _ in pairs}

    for old, new in pairs:
        if old not in available:
            raise ConfigError(
                f"config field 'columns' renames {old!r}, which is not a column of the input; the "
                f"columns present are: {', '.join(available)}"
            )
        if new in available and new not in vacated:
            raise ConfigError(
                f"config field 'columns' renames {old!r} to {new!r}, but {new!r} is already a column "
                f"of the input and is not itself being renamed; it would be written over and lost"
            )

    return dict(pairs)


def _in_place_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "record_id": [11, 12],
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
                    "record_id": [11, 12],
                    "territory": ["north", "south"],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="rename one column",
        configuration={"columns": [{_FROM: "region", _TO: "territory"}]},
        explanation="'region' becomes 'territory' and stays the second column, between 'record_id' and 'amount'.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _exchange_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["X1", "Y2"],
                    "region": ["north", "south"],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "region": ["X1", "Y2"],
                    "code": ["north", "south"],
                }
            )
        }
    )
    return WorkedExample(
        scenario="swap two column names in one step",
        configuration={
            "columns": [
                {_FROM: "code", _TO: "region"},
                {_FROM: "region", _TO: "code"},
            ]
        },
        explanation="Both renames happen at once, so each column can take the other's name without a temporary "
        "name. The values stay where they were; only the headings change.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(RenameColumns())
