"""ConnectedComponents: put every record linked by a chain of edges into one group.

Each row of ``edges`` links two records. Two records are in the same group when a chain of edges
joins them, however long the chain is. A fixed number of joins cannot do this: two joins follow a
chain two links deep, and the third link is lost.

The group is named after its smallest record. That name does not depend on the order the edges
arrive in, or on which side of an edge a record sits, so the same edges always give the same frame.

A null endpoint links nothing, the way a null key pairs with nothing in ``join``. The other
endpoint of that row is still a record, alone unless another edge links it.

The work is done on whole columns, in rounds. Each record is first given a number: its position
when the distinct records are sorted, so a smaller number is a smaller record. In each round, every
group that touches a group with a smaller number points itself at the smallest such neighbour. Then
every record follows its pointers to the end, by repeated doubling. The round ends by rewriting the
edges between the groups that are left.

The loop always ends. A record only ever points at a smaller number, so following pointers cannot
go round in a circle. Every edge still left has a larger end, and that end stops being a group in
the round, so each round leaves fewer groups on edges than the one before. When no edge joins two
different groups, the work is done. On a single chain at least half the groups go each round, so a
chain of 200,000 records needs at most 18 rounds, not 200,000.
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

_OUTPUT_FIELDS = ("node_column", "component_column")


class ConnectedComponents:
    """Group every node linked to another by a chain of edges, naming each group by its smallest node."""

    name = "connected_components"
    description = "Group records linked by a chain of edges, however long, naming each group by its smallest record."
    category = "transform"
    input_ports = ("edges",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "left": ConfigField(
                type="string",
                description="The column holding one end of each edge.",
                required=True,
                non_empty=True,
            ),
            "right": ConfigField(
                type="string",
                description="The column holding the other end of each edge. Same type as 'left'.",
                required=True,
                non_empty=True,
            ),
            "node_column": ConfigField(
                type="string",
                description="The name of the output column holding each record.",
                default="node",
                non_empty=True,
            ),
            "component_column": ConfigField(
                type="string",
                description="The name of the output column holding the group each record belongs to.",
                default="component",
                non_empty=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Each input row is an edge between two records. The output has one row per distinct "
                "record, with the group it belongs to. Two records share a group when a chain of edges "
                "links them, at any length. Only the two output columns are written; other input "
                "columns are not carried, because an output row is a record, not an edge."
            ),
            examples=(_chain_example(), _null_example()),
            notes=(
                "A group is named after its smallest record, compared by value in the endpoints' own "
                "type: the number 9 comes before 10. Text is compared character by character, so "
                "'R-100' comes before 'R-20'. The same edges give the same names, whatever order they "
                "arrive in.",
                "A null endpoint links nothing, as a null key pairs with nothing in a SQL join. A row "
                "with one null still lists the record at its other end. A row with two nulls adds "
                "nothing, and no null is ever written as a record.",
                "An edge from a record to itself is how a record with no links is listed. It does not "
                "cut that record off from its other edges.",
                "Rows come out sorted by group, then by record, so a written file does not change "
                "between runs on the same edges.",
                "'left' and 'right' must have the same type. The number 2 and the text '2' are not "
                "the same record, so the tool refuses rather than guess.",
            ),
            field_help={
                "left": "One end of each edge. Which column is 'left' and which is 'right' makes no difference.",
                "right": "The other end of each edge, in the same type as 'left'.",
                "node_column": "The output column for the record. It may reuse an input column's name, "
                "since the output is a new frame.",
                "component_column": "The output column for the group. Its value is the smallest record "
                "in the group. Must differ from 'node_column'.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        for field in _OUTPUT_FIELDS:
            if not config[field].strip():
                raise ConfigError(
                    f"config field {field!r} must not be blank: a column name has to carry at least one "
                    f"character that is not whitespace"
                )
        if config["node_column"] == config["component_column"]:
            raise ConfigError(
                f"config fields 'node_column' and 'component_column' both name {config['node_column']!r}; "
                f"the output has two columns and they need two names"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        edges = inputs["edges"]
        left, right = config["left"], config["right"]
        require_columns(edges, [left], "left", "edges")
        require_columns(edges, [right], "right", "edges")
        left_type, right_type = edges.schema[left], edges.schema[right]
        if left_type != right_type:
            raise ConfigError(
                f"config fields 'left' and 'right' name columns of different types on input port 'edges': "
                f"{left!r} is {left_type} and {right!r} is {right_type}. A value of one type is never the "
                f"same record as a value of another, so cast one column first"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        edges = inputs["edges"]
        left, right = config["left"], config["right"]
        node_column, component_column = config["node_column"], config["component_column"]

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

        rows = pl.DataFrame([nodes.rename(node_column), nodes.gather(parent).rename(component_column)]).sort(
            [component_column, node_column]
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
            "edges": pl.DataFrame(
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
                    "node": ["R-12", "R-30", "R-45", "R-55", "R-61", "R-80"],
                    "component": ["R-12", "R-12", "R-12", "R-55", "R-55", "R-80"],
                }
            )
        }
    )
    return WorkedExample(
        scenario="group items linked to each other through a chain of references",
        configuration={"left": "item", "right": "linked_item"},
        explanation="R-30 links to R-12, and R-12 links to R-45. No edge links R-30 to R-45, but the "
        "chain puts all three in one group, named after its smallest item, R-12. R-55 and R-61 form a "
        "second group. R-80 links only to itself, so it is a group of one.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _null_example() -> WorkedExample:
    inputs = Ports(
        {
            "edges": pl.DataFrame(
                {
                    "item": [3, None, None, 8],
                    "linked_item": [None, 8, None, 11],
                },
                schema={"item": pl.Int64, "linked_item": pl.Int64},
            )
        }
    )
    outputs = Ports({"rows": pl.DataFrame({"node": [3, 8, 11], "component": [3, 8, 8]})})
    return WorkedExample(
        scenario="a missing endpoint links nothing",
        configuration={"left": "item", "right": "linked_item"},
        explanation="3 and 8 each sit beside a null, but a null links nothing, so they stay apart. Each "
        "is still listed. The row with two nulls adds nothing. 8 and 11 share a real edge, so they "
        "share a group.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(ConnectedComponents())
