"""RemoveDuplicates: keep one row per group of matching rows, and route the rest to `duplicate`.

Matching follows SQL's DISTINCT, so two nulls are the same value. Without `by`, every column is
compared. Each port keeps arrival order.

Why: docs/design/tools.md#remove_duplicates
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
    """Split `rows` into one row of each matching group (`unique`) and the rest (`duplicate`)."""

    name = "remove_duplicates"
    description = "Keep one row of each group of matching rows, and set the repeats aside."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("unique", "duplicate")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="The columns that decide whether two rows are the same. Left out or "
                "empty, every column is compared.",
                required=False,
                items=ConfigField(
                    type="string",
                    description="One column to compare, such as 'customer_id'.",
                ),
            ),
            "keep": ConfigField(
                type="string",
                description="Which row of each group goes to `unique`: the 'first' or the 'last' to arrive.",
                default="first",
                enum=_KEEP_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use remove_duplicates to keep one row per customer, per order, or per whole row. "
                "One row of each group of matching rows goes to `unique`; the others go to "
                "`duplicate`, so you can see what was removed. The two ports together hold every "
                "input row, each in its arrival order. Matching follows SQL's DISTINCT."
            ),
            examples=(_whole_row_example(), _keyed_example()),
            notes=(
                "Two null values count as the same, as with SQL's DISTINCT, so rows with a null key "
                "form one group. join treats null keys the opposite way: they never pair.",
                "A null is not the same as empty text. Filling nulls with '' before comparing would "
                "merge real blanks into the null group.",
                "Without 'by', every column is compared. A column named twice in 'by' counts once.",
                "'keep' goes by the order the rows arrived in. After read_table, which gives rows in "
                "no fixed order, sort first so the row kept is the one you chose.",
            ),
            field_help={
                "by": "For example ['order_id']. Leave it out to remove rows that repeat in every column.",
                "keep": "Use 'last' after sorting by date to keep each group's latest row.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a `by` column that is not on the input, such as an old `['all']` placeholder."""
        require_columns(inputs["rows"], _key_columns(inputs["rows"], config), "by", "rows")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        key = pl.struct(_key_columns(frame, config))
        unique_mask = key.is_first_distinct() if config["keep"] == "first" else key.is_last_distinct()

        return Ports({"unique": frame.filter(unique_mask), "duplicate": frame.filter(~unique_mask)})


def _key_columns(frame: pl.DataFrame, config: Mapping[str, Any]) -> list[str]:
    """The columns rows are compared on: those configured, or every column."""
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
        scenario="remove a row that repeats in every column",
        configuration={},
        explanation="With no 'by', every column is compared. The third row repeats the first exactly, so it goes "
        "to 'duplicate' and the first goes to 'unique'.",
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
        scenario="keep the last row for each code",
        configuration={"by": ["code"], "keep": "last"},
        explanation="Only 'code' is compared, so the two 'A' rows are one group. 'keep: last' sends the second to "
        "'unique' and the first to 'duplicate'.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(RemoveDuplicates())
