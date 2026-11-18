"""Filter: one input port, two output ports, and nothing dropped between them.

A step that only ever produced the rows that matched would make the rows it discarded stop being
data at all -- a number a downstream reader reconstructs afterwards with an anti-join, if they
remember to. So Filter routes every row onto exactly one of two named ports, ``true`` and ``false``,
after the value the predicate took, and the row-conservation invariant this buys --
``true.height + false.height == rows.height``, always -- is the reason there are two output ports
here rather than one.

A predicate is three-valued, not two: a comparison against a missing value is null, matching the
query engine underneath rather than folk intuition, and a null row goes to ``false`` by default
because that is the choice that surprises nobody reading a plain filter. ``nulls: "include"`` sends
it to ``true`` instead. There is no third port for it -- the distinction is one step away downstream,
``[column] is null`` against the ``false`` port, and a third output would put an edge that is almost
always empty on every flow that uses this tool. The enum is spelled ``"exclude"`` / ``"include"``
rather than the more obvious ``"false"`` / ``"true"``, because a configuration is a YAML file, and
YAML turns an unquoted ``true`` or ``false`` into a boolean before this module ever sees it -- an
author writing ``nulls: true`` would be refused with a type error that has nothing to do with what
they meant.

A predicate that is not a condition is refused before a row is read, and the refusal names the
column: ``[amount]`` on a numeric column compiles as an expression -- it is exactly how a column
reference is written -- but it is not something a row can leave on ``true`` or ``false`` for, and
handed to the query engine as a filter it would raise about a dtype from a library the author has
never heard of. A bare boolean column, by contrast, *is* a valid predicate as written, because a
plain status filter is one of the real uses of this tool; the check is on the dtype the predicate
produces, not on the shape of the expression that produced it.
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
    compile_predicate_for,
)

_NULLS_ENUM: Final[tuple[str, ...]] = ("exclude", "include")


class Filter:
    """Route every row of ``rows`` onto ``true`` or ``false`` by a predicate; nothing is dropped."""

    name = "filter"
    description = "Split rows onto `true` and `false` ports by a predicate, keeping every row."
    # "preparation" rather than "transform": this tool selects among the rows it was given
    # and computes nothing, which is what it shares with the two tools it sits beside.
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("true", "false")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "predicate": ConfigField(
                type="expression",
                description="The whole condition a row is tested against; no column/operator/value "
                "structure sits beside it.",
                required=True,
            ),
            "nulls": ConfigField(
                type="string",
                description="Where a row whose predicate is null goes: 'exclude' (default, joins "
                "'false') or 'include' (joins 'true').",
                default="exclude",
                enum=_NULLS_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every row leaves on exactly one of `true` or `false`, so the pair of ports together "
                "is always the input: nothing a flow routes downstream from here needs an anti-join "
                "to recover what this tool discarded."
            ),
            examples=(_plain_example(), _null_routing_example()),
            notes=(
                "A null predicate is not the same as a false one: 'nulls' decides where the "
                "could-not-tell rows go, and defaults to the false port so a plain filter behaves the "
                "way the query engine already does.",
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Compile the predicate against the incoming schema, and refuse it if it is not a condition.

        Both refusals a predicate can earn -- a column that is not there, and an expression that
        does not produce a truth value -- are answers the schema alone can give, so they belong
        here rather than in ``execute``: an author is told before any row moves, and told it as a
        problem with their configuration rather than as a tool that failed while running.
        """
        compile_predicate_for(inputs["rows"], config["predicate"], "predicate")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        predicate = compile_predicate_for(frame, config["predicate"], "predicate")
        routed = predicate.fill_null(config["nulls"] == "include")
        return Ports({"true": frame.filter(routed), "false": frame.filter(~routed)})


def _plain_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [1, 2, 3], "amount": [10.0, -5.0, 0.0]})})
    outputs = Ports(
        {
            "true": pl.DataFrame({"key": [1], "amount": [10.0]}),
            "false": pl.DataFrame({"key": [2, 3], "amount": [-5.0, 0.0]}),
        }
    )
    return WorkedExample(
        scenario="keep only rows with a positive amount",
        configuration={"predicate": "[amount] > 0"},
        explanation="Every row with a positive amount joins 'true'; the rest, including a zero amount, join 'false'.",
        inputs=inputs,
        outputs=outputs,
    )


def _null_routing_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [1, 2, 3], "status": ["active", "closed", None]})})
    outputs = Ports(
        {
            "true": pl.DataFrame({"key": [1, 3], "status": ["active", None]}),
            "false": pl.DataFrame({"key": [2], "status": ["closed"]}),
        }
    )
    return WorkedExample(
        scenario="a missing status is routed to 'true' by request",
        configuration={"predicate": "[status] = 'active'", "nulls": "include"},
        explanation="The row with no status cannot be tested, so it would join 'false' by default; "
        "'nulls: include' sends it to 'true' instead, without moving the row that is genuinely closed.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(Filter())
