"""Sort: put rows in order by one or more keys, and change nothing else.

Each key has its own direction and its own place for nulls. The sort is stable: rows that tie on
every key keep their arrival order.

Why: docs/design/tools.md#sort
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    require_columns,
)

_DIRECTION_ENUM: Final[tuple[str, ...]] = ("asc", "desc")
_NULLS_ENUM: Final[tuple[str, ...]] = ("first", "last")

# A key's defaults: ascending, nulls last. Held here because defaults inside a list element are
# never filled in by the config template.
_DEFAULT_DIRECTION: Final[str] = "asc"
_DEFAULT_NULLS: Final[str] = "last"


class Sort:
    """Order the rows of `rows` by one or more keys, leaving everything else alone."""

    name = "sort"
    description = "Put rows in order by one or more columns, each ascending or descending."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="The columns to sort on, most important first. Each later key only "
                "breaks ties left by the ones before it.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One sort key: a column, its direction, and where its nulls go.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The column to sort on, such as 'amount'.",
                            required=True,
                            non_empty=True,
                        ),
                        "direction": ConfigField(
                            type="string",
                            description="'asc' for smallest first, 'desc' for largest first. Left "
                            "out, it is 'asc'.",
                            enum=_DIRECTION_ENUM,
                        ),
                        "nulls": ConfigField(
                            type="string",
                            description="Where nulls go: 'first' or 'last'. Left out, they go last, "
                            "in either direction.",
                            enum=_NULLS_ENUM,
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use sort to put rows in order, for example by date, or by customer and then by "
                "largest amount. Each key has its own direction and its own place for nulls. Rows "
                "that tie on every key keep the order they arrived in, so the result is the same on "
                "every run. Only the row order changes: the columns and values are untouched."
            ),
            examples=(_descending_example(), _two_key_example()),
            notes=(
                "Nulls go last by default, in either direction. Write 'nulls: first' on a key to put "
                "them on top.",
                "A not-a-number value (what 0/0 gives) sorts as larger than every number: last when "
                "ascending, first when descending. It is not a null, so 'nulls' does not move it.",
                "A 'direction' or 'nulls' written with no value, such as 'direction:' on its own, is "
                "refused, not given the default.",
                "Naming the same column twice in 'by' is refused, because the second key could never "
                "decide anything.",
            ),
            field_help={
                "by": "For example [{column: zone}, {column: amount, direction: desc}]: by zone, then "
                "largest amount first within each zone.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a blank column, a setting written with no value, and a repeated key, by path."""
        seen: dict[str, int] = {}
        for index, key in enumerate(config["by"]):
            path = f"by[{index}]"
            column = key["column"]
            if not column.strip():
                raise ConfigError(f"config field '{path}.column' is blank; a sort key needs a column name")
            for member, choices in (("direction", _DIRECTION_ENUM), ("nulls", _NULLS_ENUM)):
                if member in key and key[member] is None:
                    raise ConfigError(
                        f"config field '{path}.{member}' was given with no value; write "
                        f"{' or '.join(repr(choice) for choice in choices)}, or leave the member out "
                        f"entirely to take its default"
                    )
            if column in seen:
                raise ConfigError(
                    f"config field 'by' names column {column!r} twice, at {path} and at "
                    f"by[{seen[column]}]; the second one can never decide anything, because the "
                    f"first already ordered every row it would look at"
                )
            seen[column] = index

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a key column that is not on the input."""
        require_columns(inputs["rows"], [key["column"] for key in config["by"]], "by", "rows")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Sort, with each row's arrival position as the last key, so ties keep their order."""
        frame = inputs["rows"]
        keys = config["by"]

        # Arrival position as the last key keeps ties in order. maintain_order alone is not enough.
        # Why: docs/design/tools.md#sort
        arrival = pl.int_range(pl.len())
        return Ports(
            {
                "rows": frame.sort(
                    [pl.col(key["column"]) for key in keys] + [arrival],
                    descending=[key.get("direction", _DEFAULT_DIRECTION) == "desc" for key in keys] + [False],
                    nulls_last=[key.get("nulls", _DEFAULT_NULLS) == "last" for key in keys] + [False],
                    maintain_order=True,
                )
            }
        )


def _descending_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["a", "b", "c", "d"],
                    "amount": [10.0, 25.0, 10.0, None],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["b", "a", "c", "d"],
                    "amount": [25.0, 10.0, 10.0, None],
                }
            )
        }
    )
    return WorkedExample(
        scenario="largest amount first",
        configuration={"by": [{"column": "amount", "direction": "desc"}]},
        explanation="Rows 'a' and 'c' both hold 10.0, so they keep the order they came in. The missing amount "
        "goes last, as nulls do by default.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _two_key_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "zone": ["south", "north", None, "north"],
                    "amount": [1.0, 5.0, 7.0, 9.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "zone": ["north", "north", "south", None],
                    "amount": [9.0, 5.0, 1.0, 7.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="by zone, then largest amount first within a zone",
        configuration={"by": [{"column": "zone"}, {"column": "amount", "direction": "desc"}]},
        explanation="'zone' sorts first, ascending, so the two north rows lead and the row with no zone goes "
        "last. 'amount' only decides between rows with the same zone: here, the two north rows.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Sort())
