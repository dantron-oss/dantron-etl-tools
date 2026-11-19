"""MakeGroups: put every record linked by a chain of pairs, however long, into one group.

Each row of `pairs` links two records. The group is named after its smallest record, so the same
pairs always give the same result. A null endpoint links nothing, as a null key does in join.

Why: docs/design/tools.md#make_groups
"""

from collections.abc import Mapping
from typing import Any

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

_OUTPUT_FIELDS = ("record_column", "group_column")


class MakeGroups:
    """Group every record linked by a chain of pairs, naming each group after its smallest record."""

    name = "make_groups"
    description = "Put records that are linked, directly or through others, into the same group."
    category = "transform"
    input_ports = ("pairs",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "left": ConfigField(
                type="string",
                description="The column holding one record of each pair.",
                required=True,
                non_empty=True,
            ),
            "right": ConfigField(
                type="string",
                description="The column holding the other record of each pair. Same type as 'left'.",
                required=True,
                non_empty=True,
            ),
            "record_column": ConfigField(
                type="string",
                description="The name of the output column holding each record.",
                default="record",
                non_empty=True,
            ),
            "group_column": ConfigField(
                type="string",
                description="The name of the output column holding the group each record belongs to.",
                default="group",
                non_empty=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use make_groups when records are linked in pairs, such as an account merged into "
                "another, and you want every record that is connected, however indirectly, in one "
                "group. Each input row on `pairs` links two records. `rows` has one row per distinct "
                "record with its group, named after the group's smallest record. Only those two "
                "columns are written."
            ),
            examples=(_chain_example(), _null_example()),
            notes=(
                "A group is named after its smallest record, compared by value in the endpoints' own "
                "type: the number 9 comes before 10. Text is compared character by character, so "
                "'R-100' comes before 'R-20'. The same pairs give the same names, whatever order they "
                "arrive in.",
                "A null endpoint links nothing, as a null key pairs with nothing in a SQL join. A row "
                "with one null still lists the record at its other end. A row with two nulls adds "
                "nothing, and no null is ever written as a record.",
                "A pair of a record with itself is how a record with no links is listed. It does not "
                "cut that record off from its other pairs.",
                "Rows come out sorted by group, then by record, so a written file does not change "
                "between runs on the same pairs.",
                "'left' and 'right' must have the same type. The number 2 and the text '2' are not "
                "the same record, so the tool refuses rather than guess.",
            ),
            field_help={
                "left": "For example 'account_id'. Which column is 'left' and which is 'right' makes "
                "no difference.",
                "right": "For example 'merged_into_id', in the same type as 'left'.",
                "record_column": "The output column for the record. It may reuse an input column's name, "
                "since the output is a new frame.",
                "group_column": "The output column for the group. Its value is the smallest record "
                "in the group. Must differ from 'record_column'.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        for field in _OUTPUT_FIELDS:
            if not config[field].strip():
                raise ConfigError(
                    f"config field {field!r} must not be blank: a column name has to carry at least one "
                    f"character that is not whitespace"
                )
        if config["record_column"] == config["group_column"]:
            raise ConfigError(
                f"config fields 'record_column' and 'group_column' both name {config['record_column']!r}; "
                f"the output has two columns and they need two names"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        edges = inputs["pairs"]
        left, right = config["left"], config["right"]
        require_columns(edges, [left], "left", "pairs")
        require_columns(edges, [right], "right", "pairs")
        left_type, right_type = edges.schema[left], edges.schema[right]
        if left_type != right_type:
            raise ConfigError(
                f"config fields 'left' and 'right' name columns of different types on input port 'pairs': "
                f"{left!r} is {left_type} and {right!r} is {right_type}. A value of one type is never the "
                f"same record as a value of another, so cast one column first"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        edges = inputs["pairs"]
        left, right = config["left"], config["right"]
        record_column, group_column = config["record_column"], config["group_column"]

        # Every distinct record, sorted, so a record's position is also its rank.
        nodes = pl.concat([edges[left], edges[right]]).drop_nulls().unique().sort().rename("value")
        index = nodes.to_frame().with_row_index("id")

        pairs = (
            edges.select(pl.col(left).alias("a"), pl.col(right).alias("b"))
            .join(index, left_on="a", right_on="value", how="inner")
            .join(index, left_on="b", right_on="value", how="inner", suffix="_b")
            .select(pl.col("id").alias("u"), pl.col("id_b").alias("v"))
        )

        # parent[i] is the record i points at. Every record starts as its own group.
        parent = pl.int_range(nodes.len(), dtype=pl.get_index_type(), eager=True)
        links = pairs.filter(pl.col("u") != pl.col("v")).unique()
        # Ends: parent[i] <= i always holds, and each round the larger end of every remaining link
        # is pointed at a smaller group, so the set of groups still on a link shrinks every round.
        while links.height:
            # Each group points at the smallest group it shares a link with, if that one is smaller.
            hooks = links.group_by(pl.max_horizontal("u", "v").alias("high")).agg(
                pl.min_horizontal("u", "v").min().alias("low")
            )
            parent = _follow_to_end(parent.clone().scatter(hooks["high"], hooks["low"]))
            # Rewrite each link between the groups its ends now belong to; drop links inside a group.
            links = (
                links.select(u=parent.gather(links["u"]), v=parent.gather(links["v"]))
                .filter(pl.col("u") != pl.col("v"))
                .unique()
            )

        rows = pl.DataFrame([nodes.rename(record_column), nodes.gather(parent).rename(group_column)]).sort(
            [group_column, record_column]
        )
        return Ports({"rows": rows})


def _follow_to_end(parent: pl.Series) -> pl.Series:
    """Point every record straight at the end of its chain of pointers.

    Each pass replaces a record's pointer with its pointer's pointer, so the distance covered
    doubles every pass. It stops when a pass changes nothing. Because a record only points at a
    smaller number, that always happens.
    """
    while True:
        jumped = parent.gather(parent)
        if jumped.equals(parent):
            return parent
        parent = jumped


def _chain_example() -> WorkedExample:
    inputs = Ports(
        {
            "pairs": pl.DataFrame(
                {
                    "item": ["R-30", "R-12", "R-80", "R-55"],
                    "linked_item": ["R-12", "R-45", "R-80", "R-61"],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "record": ["R-12", "R-30", "R-45", "R-55", "R-61", "R-80"],
                    "group": ["R-12", "R-12", "R-12", "R-55", "R-55", "R-80"],
                }
            )
        }
    )
    return WorkedExample(
        scenario="group items linked to each other through a chain of references",
        configuration={"left": "item", "right": "linked_item"},
        explanation="R-30 links to R-12, and R-12 links to R-45. No pair links R-30 to R-45, but the "
        "chain puts all three in one group, named after its smallest item, R-12. R-55 and R-61 form a "
        "second group. R-80 links only to itself, so it is a group of one.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _null_example() -> WorkedExample:
    inputs = Ports(
        {
            "pairs": pl.DataFrame(
                {
                    "item": [3, None, None, 8],
                    "linked_item": [None, 8, None, 11],
                },
                schema={"item": pl.Int64, "linked_item": pl.Int64},
            )
        }
    )
    outputs = Ports({"rows": pl.DataFrame({"record": [3, 8, 11], "group": [3, 8, 8]})})
    return WorkedExample(
        scenario="a missing endpoint links nothing",
        configuration={"left": "item", "right": "linked_item"},
        explanation="3 and 8 each sit beside a null, but a null links nothing, so they stay apart. Each "
        "is still listed. The row with two nulls adds nothing. 8 and 11 share a real pair, so they "
        "share a group.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(MakeGroups())
