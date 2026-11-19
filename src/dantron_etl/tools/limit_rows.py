"""LimitRows: keep the first `count` rows, and send the rest to `dropped`.

It never sorts; it takes rows in the order they arrived. `count` has no default, and 0 is allowed.

Why: docs/design/tools.md#limit_rows
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
)


class LimitRows:
    """Split `rows` in two: the first `count` rows, and everything after them."""

    name = "limit_rows"
    description = "Keep the first rows of a table, and set the rest aside on a second port."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows", "dropped")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "count": ConfigField(
                type="integer",
                description="How many rows to keep, from the top, in the order they arrived.",
                required=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use limit_rows to keep the first rows of a table, for example the first 10. The kept "
                "rows go to `rows`; everything after them goes to `dropped`. It does not sort: put a "
                "sort in front for a top 10 by amount. Both ports keep the input's columns and types, "
                "and together they hold every input row, in order."
            ),
            examples=(_split_example(), _short_frame_example()),
            notes=(
                "No row is thrown away. The rows past the limit go to `dropped`, so a flow can count, "
                "log or write them.",
                "'count: 0' keeps no rows and sends them all to `dropped`.",
                "A count larger than the table keeps every row. `dropped` is then empty but still has "
                "the columns and their types.",
                "A negative count is refused.",
            ),
            field_help={
                "count": "A whole number, for example 10. There is no default: a limit nobody wrote "
                "is not a limit.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a negative count, quoting it and naming 0 as the way to keep none."""
        count = config["count"]
        if count < 0:
            raise ConfigError(
                f"config field 'count' is {count}, and a count of rows cannot be negative; write 0 to keep none of them"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Nothing to check: this tool names no column."""
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Split the frame in two at `count`, without sorting."""
        frame = inputs["rows"]
        count = config["count"]
        return Ports({"rows": frame.head(count), "dropped": frame.slice(count)})


def _split_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["d", "a", "c", "b"],
                    "amount": [50.0, 10.0, 90.0, 30.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["d", "a"],
                    "amount": [50.0, 10.0],
                }
            ),
            "dropped": pl.DataFrame(
                {
                    "code": ["c", "b"],
                    "amount": [90.0, 30.0],
                }
            ),
        }
    )
    return WorkedExample(
        scenario="keep the first two rows, as they arrived",
        configuration={"count": 2},
        explanation="The first two rows are 'd' and 'a' because that is the order they arrived in. "
        "The step does not sort. The other two rows go to 'dropped'.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _short_frame_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["a", "b"],
                    "amount": [10.0, 30.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["a", "b"],
                    "amount": [10.0, 30.0],
                }
            ),
            "dropped": pl.DataFrame(
                {
                    "code": [],
                    "amount": [],
                },
                schema={"code": pl.String, "amount": pl.Float64},
            ),
        }
    )
    return WorkedExample(
        scenario="a table shorter than the limit comes back whole",
        configuration={"count": 10},
        explanation="There are fewer rows than the limit, so every row is kept. 'dropped' is "
        "empty, but still has both columns and their types.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(LimitRows())
