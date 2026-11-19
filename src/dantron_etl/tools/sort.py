"""Sort: put the rows of `rows` in an order, and change nothing else about them.

The same rows come back. Same columns, same column order, same types. Only their order moves.

**Each key carries its own direction and its own placement for missing values.** One list can say
"zone ascending with missing zones first, then amount descending with missing amounts last". A tool
that read one direction and applied it to every key would look right on a single-key config. That
is the config everybody writes first. It would be wrong on the second key of the next one.

**The sort is stable.** Rows that tie on every key keep the order they arrived in. The rest of the
library builds on that promise. A generated row number is a sort followed by a count. An unstable
sort would make that number depend on how the engine laid the rows out rather than on the data, so
the same pipeline run twice would give two different answers. The query engine does not keep that
promise in every case on its own, so the sort adds each row's arrival position as its last key.

**A direction or a placement written with no value is refused.** A definition file spells that
`direction:`, and it arrives here as nothing at all. Read as "not configured", it would run the
sort under a policy its author never chose and was never told about.
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

# What a key does when it says nothing: sort ascending, put missing values last.
#
# These live here rather than on the template because the contract fills defaults at the top level
# of a configuration only. A default declared inside a sort key would read as if it applied, and
# never be applied. So the tool holds the two policies itself and applies them where it sorts.
_DEFAULT_DIRECTION: Final[str] = "asc"
_DEFAULT_NULLS: Final[str] = "last"


class Sort:
    """Order the rows of `rows` by one or more keys, leaving everything else about the frame alone."""

    name = "sort"
    description = "Order the rows of `rows` by one or more keys, keeping tied rows in their arrival order."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="The sort keys, most important first. The first key decides. Each later "
                "key breaks the ties the ones before it leave.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One sort key: a column, and how to read it.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The column to sort on. It has to be on the input.",
                            required=True,
                            non_empty=True,
                        ),
                        "direction": ConfigField(
                            type="string",
                            description="'asc' for smallest first, 'desc' for largest first. Left "
                            "out, a key sorts ascending.",
                            enum=_DIRECTION_ENUM,
                        ),
                        "nulls": ConfigField(
                            type="string",
                            description="Where missing values go: 'first' or 'last'. Left out, they "
                            "go last. That holds whichever direction the key sorts in.",
                            enum=_NULLS_ENUM,
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Orders the rows by the keys listed in 'by'. The first key decides. Each later key "
                "breaks the ties the ones before it leave. Every key carries its own direction and "
                "its own placement for missing values, so one list can sort one column ascending "
                "and the next descending. Rows that tie on every key keep the order they arrived "
                "in. That is what lets a row number generated afterwards depend on the data rather "
                "than on the engine. Nothing else about the frame changes: same rows, same "
                "columns, same column order, same types."
            ),
            examples=(_descending_example(), _two_key_example()),
            notes=(
                "Missing values go last by default, in either direction. Write 'nulls: first' on a "
                "key to put them on top.",
                "A direction or a placement written with no value is refused, not defaulted. A "
                "definition file spells 'direction:' when its author meant to say something.",
            ),
            field_help={
                "by": "A list of keys, most important first. Each key needs a 'column'. 'direction' "
                "and 'nulls' are optional, and each applies to its own key alone.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """The rules a template cannot state: a blank column, an option with no value, a repeated key.

        Each refusal names the key by its path. A list of eight keys is read in a file, and "one of
        your sort keys is wrong" sends the reader back to look at all eight of them.
        """
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
        """Every key column has to be on the input, which only the incoming schema can say."""
        require_columns(inputs["rows"], [key["column"] for key in config["by"]], "by", "rows")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Sort, with each row's arrival position as the last key, so ties keep their order."""
        frame = inputs["rows"]
        keys = config["by"]

        # The arrival position is the final key, always ascending. Rows that tie on every key the
        # author wrote are then ordered by where they arrived, which is what stability means.
        #
        # Do not replace it with polars' own stability flag alone. With one key that is missing on
        # every row, sorted descending, polars 1.38 hands the rows back in reverse even when asked
        # to keep their order. record_id numbers rows after a sort, so the reversal would reach the
        # identifiers. The flag stays as well: it costs nothing and says what is meant.
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
        scenario="largest amount first, with the two rows that tie left as they arrived",
        configuration={"by": [{"column": "amount", "direction": "desc"}]},
        explanation="Rows 'a' and 'c' both hold 10.0, so nothing in the configuration separates "
        "them. They come back in the order they went in. The missing amount goes last, which is "
        "where missing values go unless a key says otherwise.",
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
        scenario="group by zone, and within a zone put the largest amount first",
        configuration={"by": [{"column": "zone"}, {"column": "amount", "direction": "desc"}]},
        explanation="'zone' decides first and sorts ascending, so the two northern rows lead and "
        "the row with no zone goes last. 'amount' only decides between rows the first key tied. "
        "Here that is the pair of northern rows.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Sort())
