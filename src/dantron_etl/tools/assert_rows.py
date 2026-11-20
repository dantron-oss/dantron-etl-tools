"""AssertRows: stop the run unless every row meets a condition.

A null result is a failure, and so is an empty frame unless `allow_empty`. The failing rows go into
the error message; there is no second port for them.

Why: docs/design/tools.md#assert_rows
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..authoring import (
    MAX_REPORTED_ROWS,
    REGISTRY,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    compile_predicate_for,
    refuse_division_by_zero,
)

# How many failing rows a message shows; the same cap the boundary uses.
REPORTED_ROW_LIMIT: Final[int] = MAX_REPORTED_ROWS


class AssertRows:
    """Stop the run unless every row of `rows` meets `predicate`; otherwise pass `rows` through."""

    name = "assert_rows"
    description = "Check that every row meets a condition, and stop the run if any does not."
    category = "assert"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "predicate": ConfigField(
                type="expression",
                description="The condition every row must meet, such as '[amount] >= 0'.",
                required=True,
            ),
            "label": ConfigField(
                type="string",
                description="A short name for the check, shown in the message when it fails.",
                required=False,
            ),
            "allow_empty": ConfigField(
                type="boolean",
                description="Set true to let a table with no rows pass. Left out, an empty table fails.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use assert_rows to check a rule that must always hold, such as no negative amounts "
                "or every row having a customer. If every row meets the condition, `rows` passes "
                "through unchanged. If any row does not, the run stops with a message naming the "
                "check, how many rows failed out of how many, and some of the failing rows."
            ),
            examples=(_satisfied_example(), _allowed_empty_example()),
            notes=(
                "A row whose condition is null fails. There is no setting to change that: a check "
                "that could not tell has not passed. (filter has a 'nulls' setting because choosing "
                "rows is different.)",
                "A table with no rows fails unless 'allow_empty' is true, because an empty table "
                "meets every condition, and an upstream filter that matched nothing would otherwise "
                "pass in silence.",
                "To keep the failing rows as data instead of stopping, use filter.",
            ),
            field_help={
                "label": "For example 'no negative amounts'. The condition itself is always shown too.",
                "allow_empty": "Set it on a branch that is sometimes empty for a good reason.",
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
        heading = _heading(config)

        if frame.height == 0:
            if config["allow_empty"]:
                return Ports({"rows": frame})
            raise DataError(
                f"{heading}: the input is empty, and every row of an empty frame vacuously "
                f"satisfies any predicate; set allow_empty to accept that",
                offending=frame,
                total=0,
            )

        satisfied = predicate.fill_null(False)
        failing = frame.filter(~satisfied)
        if failing.height > 0:
            raise DataError(heading, offending=failing, total=frame.height)

        return Ports({"rows": frame})


def _heading(config: Mapping[str, Any]) -> str:
    """The start of the failure message, naming the check by its label and condition."""
    label = config.get("label")
    predicate = config["predicate"]
    if label:
        return f"assertion {label!r} (`{predicate}`) failed"
    return f"assertion `{predicate}` failed"


def _satisfied_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.0, 20.5]})
    inputs = Ports({"rows": frame})
    outputs = Ports({"rows": frame})
    return WorkedExample(
        scenario="check that every amount is positive",
        configuration={"predicate": "[amount] > 0", "label": "every figure is positive"},
        explanation="Both rows have a positive amount, so the run carries on and 'rows' passes through unchanged.",
        inputs=inputs,
        outputs=outputs,
    )


def _allowed_empty_example() -> WorkedExample:
    frame = pl.DataFrame({"key": pl.Series([], dtype=pl.String), "amount": pl.Series([], dtype=pl.Float64)})
    inputs = Ports({"rows": frame})
    outputs = Ports({"rows": frame})
    return WorkedExample(
        scenario="let an empty table pass, because this branch is sometimes empty",
        configuration={"predicate": "[amount] > 0", "allow_empty": True},
        explanation="An empty table would fail by default. 'allow_empty: true' says that an empty table is "
        "expected here.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(AssertRows())
