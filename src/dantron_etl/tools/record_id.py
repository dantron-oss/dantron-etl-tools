"""RecordId: one whole number per row, counting up in the order the rows arrived.

Everything that came in comes back, unmoved, with one more column on the end. The number starts at
`start` and goes up by one per row.

**The numbering follows arrival order.** This tool sorts nothing and decides nothing. That is why
it is worth something only after an explicit sort step. Put a `sort` in front of it and the
numbering follows the order that step chose, which is an order somebody wrote down. Without one it
follows whatever order the rows happened to be in.

**The column is a 64-bit integer, declared rather than inferred.** A frame with no rows has nothing
to infer a type from. A column typed as nothing at all would break the next join on the day an
upstream filter matched nothing, and not before.

**A name already in use is refused.** There is no option here to overwrite, so the shared check the
rest of the catalog uses is asked with nothing to replace. Whoever writes this step does not know
every column of the frame they are writing into. A collision taken quietly destroys a column and
reports success.
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..tool import (
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
    """Add a column of whole numbers to `rows`, counting up in the order the rows arrived."""

    name = "record_id"
    description = "Add a column numbering the rows of `rows`, counting up from a starting number."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "output": ConfigField(
                type="string",
                description="The name of the number column to add. It may not take the name of a "
                "column that is already there.",
                required=True,
                non_empty=True,
            ),
            "start": ConfigField(
                type="integer",
                description="The number the first row gets. Every row after it counts up by one. "
                "Left out, the numbering starts at 1.",
                default=1,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Adds one column to the end of the frame, holding a 64-bit integer per row. The "
                "first row gets 'start'. The next gets one more, and so on. The rows themselves do "
                "not move, so the numbering follows the order they arrived in. Put a sort in front "
                "of this step and the numbering follows the order that sort chose. A start of 0 or "
                "a negative start counts up from there. The type is declared rather than inferred, "
                "so a frame with no rows still comes back carrying the column."
            ),
            examples=(_default_start_example(), _zero_start_example()),
            notes=(
                "This step decides no order of its own. On an unsorted frame the numbers are only "
                "as meaningful as the order the rows happened to be in.",
            ),
            field_help={
                "output": "The new column goes on the end. A name already in use is refused. There "
                "is no option here to overwrite one.",
                "start": "Any whole number, including 0 and negative numbers. Left out, it is 1.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """The rules the template cannot state: a blank name, a start with no value, a start too big.

        A Python integer has no ceiling and the column has one, so a start can be written that
        nothing here could hold. That is wrong on its own, whatever frame turns up, so it is
        refused here. A start that fits but runs off the end part way down the frame is a different
        question, because the answer depends on how many rows arrive. That one waits for the frame.
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
        """Refuse a name the frame already carries, and a numbering that runs off the end.

        The last row is given `start + height - 1`. A frame with no rows is given nothing, so there
        is no last number and nothing to refuse. Left alone, a numbering that passes the ceiling
        wraps round to large negative numbers part way down, and the tool reports success.
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
        """Count up over the rows as they are, without touching their order.

        The range is built from the frame's own height, not from anything in the data. That is what
        keeps the numbering a function of arrival order alone.
        """
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
        scenario="number the rows from 1, in the order they arrived",
        configuration={"output": "row_no"},
        explanation="The frame is in no particular order, and the numbering follows it anyway. 'c' "
        "arrived first, so 'c' gets 1. The new column goes on the end, and nothing else about the "
        "frame changes.",
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
        scenario="number the rows from 0, for a consumer that counts from zero",
        configuration={"output": "offset", "start": 0},
        explanation="'start: 0' is honoured as written. Zero reads as an absence in a lot of code, "
        "so it is worth seeing it work. The first row gets 0, not 1.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(RecordId())
