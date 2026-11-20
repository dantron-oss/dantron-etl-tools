"""Formula: add or replace named columns, each worked out by one expression.

Every expression reads the input frame as it arrived, never a column another entry is about to
write. A replaced column keeps its position; new columns are added at the end.

Why: docs/design/tools.md#formula
"""

from collections.abc import Mapping
from typing import Any

import polars as pl

from ..authoring import (
    REGISTRY,
    REPLACE_FIELD,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    check_output_column,
    compile_for,
    refuse_rows_without_an_answer,
    require_not_blank,
)


class Formula:
    """Add or replace named columns of `rows`, each worked out by an expression on the input."""

    name = "formula"
    description = "Add a calculated column, or change one, using an expression such as [price] * [quantity]."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="The columns to work out, one entry each. At least one is needed.",
                required=True,
                items=ConfigField(
                    type="object",
                    description="One column: its name, the expression that works it out, and "
                    "whether it may replace a column already there.",
                    fields={
                        "name": ConfigField(
                            type="string",
                            description="The column to write, such as 'net_amount'.",
                            required=True,
                        ),
                        "value": ConfigField(
                            type="expression",
                            description="The expression, such as '[amount] - [discount]'. It reads "
                            "the input as it arrived.",
                            required=True,
                        ),
                        "replace": REPLACE_FIELD,
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use formula to add a calculated column, or to change the values of one, like a "
                "spreadsheet formula filled down a column. Each entry names a column and the "
                "expression for it. `rows` comes out with the new columns at the end; a replaced "
                "column stays where it was. Every expression reads the input as it arrived, so to "
                "use a column this step works out, read it in a second formula step."
            ),
            examples=(_monthly_figure_example(), _multi_column_with_replace_example()),
            notes=(
                "Every entry reads the input as it arrived, not as the entries above left it. If one "
                "entry replaces 'amount' with '[amount] * 2' and the next works out '[amount] + 1', "
                "the second reads the old value: 10 gives 20 and 11, not 20 and 21.",
                "An entry that reads a column added by an entry above it is refused, because that "
                "column is not on the input. Use two formula steps.",
                "A name already on the input is refused unless its entry says 'replace: true'.",
                "A blank name, or a name written by two entries, is refused. So is an empty list.",
                "Dividing by zero stops the run and names the rows. Guard the division with "
                "case_when first if zeros are expected.",
                "Dividing by a null gives null.",
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse an empty list, a blank name, and a name written twice.

        `required` is satisfied by an empty list, so it is refused here rather than run as a step that does nothing.
        """
        columns = config["columns"]
        if not columns:
            raise ConfigError("config field 'columns' must name at least one column to write")

        seen: set[str] = set()
        for index, entry in enumerate(columns):
            name = entry["name"]
            require_not_blank(name, f"columns[{index}].name")
            if name in seen:
                raise ConfigError(f"config field 'columns' writes column {name!r} more than once")
            seen.add(name)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Compile each expression against the input, and refuse a name already there unless `replace` is set."""
        frame = inputs["rows"]
        for index, entry in enumerate(config["columns"]):
            compile_for(frame, entry["value"], f"columns[{index}].value")
            check_output_column(frame, entry["name"], f"columns[{index}].name", replace=entry.get("replace"))

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        for index, entry in enumerate(config["columns"]):
            refuse_rows_without_an_answer(frame, entry["value"], f"columns[{index}].value")
        written = [
            compile_for(frame, entry["value"], f"columns[{index}].value").alias(entry["name"])
            for index, entry in enumerate(config["columns"])
        ]
        return Ports({"rows": frame.with_columns(written)})


def _monthly_figure_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "line_amount": [1200.0, 600.0],
                    "currency_rate": [2.0, 1.0],
                    "months_in_period": [12, 6],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "line_amount": [1200.0, 600.0],
                    "currency_rate": [2.0, 1.0],
                    "months_in_period": [12, 6],
                    "monthly": [50.0, 100.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="work out a monthly amount from a total, a rate and a number of months",
        configuration={
            "columns": [
                {
                    "name": "monthly",
                    "value": "([line_amount] / [currency_rate]) / [months_in_period]",
                }
            ]
        },
        explanation="The amount is divided by the rate, then by the number of months, and the result is added as "
        "a new column at the end.",
        inputs=inputs,
        outputs=outputs,
    )


def _multi_column_with_replace_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "code": [" a1 ", "b2"],
                    "line_amount": [1200.0, 600.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "code": ["A1", "B2"],
                    "line_amount": [1200.0, 600.0],
                    "doubled": [2400.0, 1200.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="tidy a column in place and add a new one",
        configuration={
            "columns": [
                {"name": "code", "value": "upper(trim([code]))", "replace": True},
                {"name": "doubled", "value": "[line_amount] * 2"},
            ]
        },
        explanation="'code' is replaced because its entry says 'replace: true', and it keeps its position. "
        "'doubled' is a new name, so it is added at the end.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(Formula())
