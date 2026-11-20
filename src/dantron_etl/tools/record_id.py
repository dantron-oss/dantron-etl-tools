"""RecordId: add a column numbering the rows from `start`, in the order they arrived.

It never sorts; put a sort in front for a meaningful order. The column is Int64, even on an empty
frame. A name already in use is refused.

Why: docs/design/tools.md#record_id
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
    check_output_column,
)

_INT64_MIN: Final[int] = -(2**63)
_INT64_MAX: Final[int] = 2**63 - 1


class RecordId:
    """Add a column of whole numbers to `rows`, counting up in arrival order."""

    name = "record_id"
    description = "Number the rows 1, 2, 3 and so on, in a new column."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "output": ConfigField(
                type="string",
                description="The name of the new number column, such as 'row_no'.",
                required=True,
                non_empty=True,
            ),
            "start": ConfigField(
                type="integer",
                description="The number the first row gets. Each row after it gets one more.",
                default=1,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use record_id to number rows 1, 2, 3 and so on, for example to give each row an "
                "id or a rank. The number goes in a new column at the end of `rows`; nothing else "
                "changes. The rows are numbered in the order they arrive, so put a sort in front "
                "for a meaningful order, such as a rank by amount."
            ),
            examples=(_default_start_example(), _zero_start_example()),
            notes=(
                "This step does not sort. Without a sort in front, the numbers follow whatever order "
                "the rows happened to be in.",
                "read_table gives rows in no fixed order, so put a sort between the two.",
                "The new column is a 64-bit whole number, even on a table with no rows.",
                "A column name already in use is refused. There is no setting to overwrite one.",
                "A 'start' too large for a 64-bit number is refused, and so is a numbering that "
                "would pass the largest 64-bit number part way down.",
            ),
            field_help={
                "output": "For example 'row_no'. It must be a name the table does not already have.",
                "start": "Any whole number, including 0 and negative numbers. Left out, it is 1.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a blank name, a start with no value, and a start too big for Int64.

        A start that fits but would overflow part way down depends on the row count, so it waits for the frame.
        """
        if not config["output"].strip():
            raise ConfigError("config field 'output' is blank; the new column needs a name")
        if config["start"] is None:
            raise ConfigError(
                "config field 'start' was given with no value; write the number the first row "
                "should get, or leave the field out entirely to start at 1"
            )
        start = config["start"]
        if not _INT64_MIN <= start <= _INT64_MAX:
            raise ConfigError(
                f"config field 'start' is {start}, which the number column cannot hold; a row "
                f"number is a 64-bit integer, so it must be between {_INT64_MIN} and {_INT64_MAX}"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a name already on the frame, and a numbering that would pass the Int64 ceiling.

        Left alone, such a numbering wraps round to large negative numbers part way down and reports success.
        """
        frame = inputs["rows"]
        check_output_column(frame, config["output"], "output", replace=None)

        if frame.height == 0:
            return
        last = config["start"] + frame.height - 1
        if not _INT64_MIN <= last <= _INT64_MAX:
            raise ConfigError(
                f"config field 'start' is {config['start']}, and this frame has {frame.height} "
                f"rows, so the last row would be numbered {last}; a row number is a 64-bit "
                f"integer, so it must be between {_INT64_MIN} and {_INT64_MAX}"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Number the rows from `start`, in their arrival order."""
        frame = inputs["rows"]
        start = config["start"]
        numbering = (pl.int_range(pl.len(), dtype=pl.Int64) + start).alias(config["output"])

        return Ports({"rows": frame.with_columns(numbering)})


def _default_start_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"code": ["c", "a", "b"], "amount": [10.0, None, 2.5]})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["c", "a", "b"],
                    "amount": [10.0, None, 2.5],
                    "row_no": pl.Series([1, 2, 3], dtype=pl.Int64),
                }
            )
        }
    )
    return WorkedExample(
        scenario="number the rows from 1",
        configuration={"output": "row_no"},
        explanation="The rows are numbered in the order they arrived: 'c' came first, so it gets 1. The new "
        "column goes at the end.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _zero_start_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"code": ["a", "b", "c"]})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["a", "b", "c"],
                    "offset": pl.Series([0, 1, 2], dtype=pl.Int64),
                }
            )
        }
    )
    return WorkedExample(
        scenario="number the rows from 0",
        configuration={"output": "offset", "start": 0},
        explanation="'start: 0' gives the first row 0, not 1.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(RecordId())
