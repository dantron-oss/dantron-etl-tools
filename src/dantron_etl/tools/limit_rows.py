"""LimitRows: keep the first `count` rows, and hand the rest back rather than dropping them.

Two output ports are the whole design. A tool that quietly discards rows is how a report comes to
show the top five hundred of something while reading as though it showed everything. So the rest of
the frame leaves by ``dropped``, carrying the input's schema exactly. A flow that ignores it is
ignoring it on purpose. Put the two ports back together and you have the input, in order, with
nothing lost and nothing duplicated.

**This tool never sorts.** It takes the first rows as they arrived. "Top N" means a sort followed
by a limit almost everywhere else, and that is the implementation somebody reaches for here. It is
the wrong one. A sort is its own step. It runs in front of this one when a flow wants it, and its
options belong to it.

**``count: 0`` is legal.** Keeping none is a real thing to configure, and refusing it would be a
special case with no reason behind it. A count is often a parameter, and a parameter that reaches
zero should behave like the limit it is rather than stopping the run. Nothing is lost when it does,
because every row leaves by ``dropped``.

**A count is a whole number, and it has no default.** A count declared as a plain number would
accept 2.5 and leave this tool to round it, which is a policy nobody wrote down. And a limit nobody
configured is not a limit. A default of a thousand would truncate a frame silently, which is the
failure the second port exists to make impossible.
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
    description = "Keep the first `count` rows of `rows`, and send the rest out by `dropped`."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows", "dropped")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "count": ConfigField(
                type="integer",
                description="How many rows to keep, counted from the first row as it arrived. "
                "Zero keeps none. A count past the height of the frame keeps all of it.",
                required=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Takes the first 'count' rows of 'rows' in the order they arrived, and sends "
                "everything after them out by 'dropped'. Nothing is sorted here: put a Sort in "
                "front when the order matters, so the ordering is written down where a reader can "
                "see it. Nothing else about the frame changes. Both ports carry the same columns, "
                "in the same order, holding the same types. Stack the two ports back up "
                "and you have the input, in order, with nothing lost and nothing duplicated."
            ),
            examples=(_split_example(), _short_frame_example()),
            notes=(
                "Nothing is discarded. The rows past the limit leave by 'dropped', so a flow that "
                "wants to count them, log them or write them out can, and a flow that ignores them "
                "is ignoring them on purpose.",
                "'count: 0' runs and keeps nothing. It is a limit like any other, and every row "
                "still leaves by 'dropped'.",
                "A count past the height of the frame keeps the whole frame. 'dropped' is then "
                "empty, and still carries the columns and their types, so a later step compiles "
                "against it either way.",
            ),
            field_help={
                "count": "How many rows travel on. It is a whole number, and there is no default: "
                "a limit nobody wrote is not a limit.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """The one rule the template cannot state: a count cannot be negative.

        Somebody writing a negative count meant one of two things. An off-by-one in whatever
        produced the number, or "keep nothing". So the refusal quotes the number, which settles the
        first, and names the count that does keep nothing, which settles the second.
        """
        count = config["count"]
        if count < 0:
            raise ConfigError(
                f"config field 'count' is {count}, and a count of rows cannot be negative; write 0 to keep none of them"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Nothing to check. This tool names no column, so any frame at all can be limited."""
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Slice in two, without sorting and without copying either half's schema by hand."""
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
        scenario="keep the first two rows as they arrived, and see the other two",
        configuration={"count": 2},
        explanation="The first two rows are 'd' and 'a' because that is the order they arrived in. "
        "They are neither the first two codes alphabetically nor the two largest amounts: this "
        "step does not sort. The other two rows are not gone, they leave by 'dropped'.",
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
        scenario="a frame shorter than the limit comes back whole",
        configuration={"count": 10},
        explanation="There are fewer rows than the limit, so every row travels on and 'dropped' is "
        "empty. It still carries both columns and their types, so a step reading it compiles "
        "whether or not a row ever reaches it.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(LimitRows())
