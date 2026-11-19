"""RemoveDuplicates: one input, two outputs, and nothing deleted between them.

A step whose only output is the survivors would make the rows it removed stop being data at all --
a count a downstream reader reconstructs afterwards, if they remember to. So RemoveDuplicates routes
every row onto exactly one of two named ports: `unique`, carrying one member of every group of
identical rows, and `duplicate`, carrying the members it did not pick. The row-conservation
invariant that buys -- `unique.height + duplicate.height == rows.height`, always -- is the reason
there are two output ports here rather than one, the same discipline Filter applies to `true` and
`false`.

**Sameness follows DISTINCT, not `=`.** Two rows whose key is entirely null are one group, matching
the query engine underneath and whatever a downstream group-by or join will do with the same rows,
rather than reporting every null-keyed row as unique -- which is exactly the data this tool exists
to clean up. A null key is still not the empty string: a text key built by filling nulls first would
merge a genuinely blank value into the null group, which is a different mistake in the other
direction.

**Absent `by` means every column of the row.** An empty list means the same thing, for the same
reason: there is nothing to read `["all"]` out of a schema, so a config carrying that sentinel over
from an older tool is refused by name rather than quietly searched for.

**`keep` decides which member of a group joins `unique`,** and never anything a YAML file would
coerce to a boolean -- `first` and `last`, not `true`/`false`/`yes`/`no`/`on`/`off`.

**Order is preserved within each port.** `DataFrame.unique` makes no such promise on its own; a
per-row mask (`is_first_distinct` / `is_last_distinct`), filtered in place, is what both preserves
order and produces the two complementary ports from a single pass.
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    require_columns,
)

_KEEP_ENUM: Final[tuple[str, ...]] = ("first", "last")


class RemoveDuplicates:
    """Split `rows` into one member of each identical group (`unique`) and the rest (`duplicate`)."""

    name = "remove_duplicates"
    description = "Split `rows` into one member of each identical group and the members left over."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("unique", "duplicate")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="Columns defining sameness; absent or empty means every column of the row.",
                required=False,
                items=ConfigField(
                    type="string",
                    description="One column name; a null value is treated as equal to another null.",
                ),
            ),
            "keep": ConfigField(
                type="string",
                description="Which member of a group joins 'unique': 'first' (default) or 'last'.",
                default="first",
                enum=_KEEP_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Sameness follows DISTINCT, not `=`: two rows whose key is null are one group, the "
                "same group a downstream group-by or join would find, rather than two rows that could "
                "not be compared. Absent 'by' means every column of the row defines the group; "
                "the members that do not join 'unique' join 'duplicate' instead of vanishing."
            ),
            examples=(_whole_row_example(), _keyed_example()),
            notes=(
                "A null key is equal to another null key, never an empty string; building a text key "
                "by filling nulls first would merge a genuinely blank value into the null group.",
                "'keep' reads the order the rows arrived in. After a read_table, which hands rows "
                "over in no fixed order, put a Sort in front so the row kept is the one you chose.",
            ),
            field_help={
                "by": "Absent or empty means every column; a column named more than once is counted only once.",
                "keep": "'first' or 'last' decides which member of a group joins 'unique'; the other "
                "member of a pair always joins 'duplicate'.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Every named key column has to be on the input, which only the incoming schema can say.

        A key that is not there is the mistake this tool invites: a config written against a schema
        that has since moved, or the `["all"]` sentinel carried over from an older tool. Left to the
        query engine it becomes a complaint about a column the author never wrote.
        """
        require_columns(inputs["rows"], _key_columns(inputs["rows"], config), "by", "rows")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        key = pl.struct(_key_columns(frame, config))
        unique_mask = key.is_first_distinct() if config["keep"] == "first" else key.is_last_distinct()

        return Ports({"unique": frame.filter(unique_mask), "duplicate": frame.filter(~unique_mask)})


def _key_columns(frame: pl.DataFrame, config: Mapping[str, Any]) -> list[str]:
    """The columns sameness is judged on: what was configured, or every column of the row.

    Derived from the configuration and the schema rather than held on the tool, because one
    registered instance serves every flow that names this tool and must carry nothing from one run
    into the next. Both callers pay one pass over a list of names for that.
    """
    columns = config["by"]
    return list(dict.fromkeys(columns)) if columns else list(frame.columns)


def _whole_row_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A"],
                    "region": ["x", "y", "x"],
                    "amount": [10.0, 20.0, 10.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "unique": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "region": ["x", "y"],
                    "amount": [10.0, 20.0],
                }
            ),
            "duplicate": pl.DataFrame(
                {
                    "code": ["A"],
                    "region": ["x"],
                    "amount": [10.0],
                }
            ),
        }
    )
    return WorkedExample(
        scenario="keep one instance of a row that repeats across every column",
        configuration={},
        explanation="No 'by' at all means every column defines the group; the third row repeats "
        "the first exactly, so it joins 'duplicate' and the first instance joins 'unique'.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _keyed_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"code": ["A", "A", "B"], "amount": [1.0, 2.0, 3.0]})})
    outputs = Ports(
        {
            "unique": pl.DataFrame({"code": ["A", "B"], "amount": [2.0, 3.0]}),
            "duplicate": pl.DataFrame({"code": ["A"], "amount": [1.0]}),
        }
    )
    return WorkedExample(
        scenario="keep the last row of a group defined by a single key column",
        configuration={"by": ["code"], "keep": "last"},
        explanation="Sameness is judged on 'code' alone, so the two rows sharing code 'A' are one "
        "group; 'keep: last' sends the second of them to 'unique' and the first to 'duplicate'.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(RemoveDuplicates())
