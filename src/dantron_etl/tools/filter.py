"""Filter: split rows onto `true` and `false` by a condition, keeping every row.

A row whose condition is null goes to `false` unless `nulls: include`. A condition that does not
give true or false is refused before any row is read.

Why: docs/design/tools.md#filter
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
    refuse_division_by_zero,
)

_NULLS_ENUM: Final[tuple[str, ...]] = ("exclude", "include")


class Filter:
    """Send each row of `rows` to `true` or `false` by a condition. No row is dropped."""

    name = "filter"
    description = "Split rows in two by a condition: rows that meet it, and rows that do not."
    # It selects rows and computes nothing, like the other preparation tools.
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("true", "false")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "predicate": ConfigField(
                type="expression",
                description="The condition each row is tested against, such as '[amount] > 0'.",
                required=True,
            ),
            "nulls": ConfigField(
                type="string",
                description="Where a row goes when a null leaves the condition undecided: "
                "'exclude' sends it to `false`, 'include' to `true`.",
                default="exclude",
                enum=_NULLS_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use filter to split a table by a condition, such as amount above zero or status equal "
                "to 'active'. Rows that meet the condition go to `true`; all the others go to `false`. "
                "No row is dropped, so the two ports together always hold every input row. Use "
                "`false` to check or report what was left out."
            ),
            examples=(_plain_example(), _null_routing_example()),
            notes=(
                "A condition on a null value is neither true nor false. 'nulls' decides where those "
                "rows go, and by default they go to `false`, as with SQL's WHERE.",
                "To route null rows separately, filter the `false` port again on '[column] is null'.",
                "A true/false column can be the whole condition, such as '[is_active]'.",
                "A condition that does not give true or false, such as '[amount]' on a number "
                "column, is refused before any row is read.",
                "Not-a-number (what 0/0 gives) counts as larger than every number and equal to "
                "itself, so '[ratio] > 0.5' sends it to `true`. Guard a division with case_when if "
                "that matters.",
            ),
            field_help={
                "predicate": "For example: [status] = 'active' and [amount] > 0. It is one "
                "expression; 'dantron-etl functions' lists what it may use.",
                "nulls": "Leave it as 'exclude' unless rows with a missing value should be kept.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a condition that names a missing column or does not give true or false."""
        compile_predicate_for(inputs["rows"], config["predicate"], "predicate")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        predicate = compile_predicate_for(frame, config["predicate"], "predicate")
        refuse_division_by_zero(frame, config["predicate"], "predicate")
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
        explanation="Rows with a positive amount go to 'true'. The rest, including the zero, go to 'false'.",
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
        scenario="keep rows with a missing status along with the active ones",
        configuration={"predicate": "[status] = 'active'", "nulls": "include"},
        explanation="The row with no status cannot be tested, so by default it would go to 'false'. "
        "'nulls: include' sends it to 'true' instead. The closed row still goes to 'false'.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(Filter())
