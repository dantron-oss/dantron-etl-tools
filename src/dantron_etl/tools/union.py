"""Union: two inputs, one output, and no column invented on the way through.

The one reshaping every other tool leaves undone. Filter, RemoveDuplicates and Join all split a
frame into named ports and none of them puts two frames back together, so a flow that routes rows
apart to treat them differently has no way to carry them on as one -- and a flow that reads the same
thing out of two sources has to run every later step twice.

**Both sides must carry the same columns, and a missing one is refused rather than filled.** That is
the whole argument of this tool. Aligning two schemas by inventing nulls is the quiet version of a
join that lost half its rows: the frame that comes back has the shape somebody expected and a column
of nothing where the data was, and the step that reads it cannot tell the difference. A flow that
genuinely wants a column the other side lacks says so with a Formula, where the invention is written
down and a reader can see it.

**Column ORDER may differ and is taken from `top`.** Two frames carrying the same columns in a
different order are the same data, and refusing them would send an author to write a SelectColumns
whose only purpose is to satisfy this tool. The order one side happens to have is not a fact about
the data; a missing column is.

**Dtypes must agree, with one exception: a column that is all nulls takes the other side's type.**
A text column stacked onto a number is a column of text that used to be money, and whichever side
got rewritten did so silently, so that is refused and the message names both dtypes -- "they differ"
without saying how sends a reader back to the data. But a column of nothing has no type to disagree
with: null is the one value belonging to every family, and an empty branch or an unpopulated column
is an ordinary frame rather than a mistake.

**Rows keep their order: all of `top`, then all of `bottom`.** No interleaving and no sorting -- a
step that needs a meaningful order puts a Sort in front of what it reads, and a step that does not
gets the one order that can be reasoned about.

**Nothing is de-duplicated.** A row appearing on both sides appears twice, because this tool stacks
frames and does not judge them; a flow wanting one of each says so with RemoveDuplicates, which
reports what it removed rather than dropping it silently.
"""

from collections.abc import Mapping
from typing import Any

import polars as pl

from ..tool import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
)


class Union:
    """Stack `bottom` under `top`: same columns, same types, every row of both."""

    name = "union"
    description = "Stack the rows of `bottom` under the rows of `top`, which must carry the same columns."
    category = "combine"
    input_ports = ("top", "bottom")
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {}

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every row of `top` followed by every row of `bottom`, carrying the columns of "
                "`top` in `top`'s own order. The two sides must name the same columns and give "
                "each the same type; column order may differ, because that is not a fact about "
                "the data."
            ),
            examples=(_stacks_two_frames(), _order_may_differ()),
            notes=(
                "A column on one side and not the other is refused, never filled with nulls: a "
                "frame of the expected shape with a column of nothing where the data was is the "
                "hardest kind of wrong to notice. Invent the column with a Formula, where it is "
                "written down.",
                "Nothing is de-duplicated. A row on both sides arrives twice, because stacking is "
                "not judging; RemoveDuplicates reports what it removes, which is the difference.",
                "Rows keep their order -- all of `top`, then all of `bottom`. A step needing a "
                "meaningful order puts a Sort in front of it, as everywhere else here.",
            ),
            field_help={},
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Both refusals are decided by the two schemas alone, so neither waits for a row."""
        top, bottom = inputs["top"], inputs["bottom"]

        missing = [name for name in top.columns if name not in bottom.columns]
        extra = [name for name in bottom.columns if name not in top.columns]
        if missing or extra:
            raise ConfigError(
                "input ports 'top' and 'bottom' do not carry the same columns, so stacking them "
                "would invent data: "
                + (f"'bottom' is missing {', '.join(missing)}. " if missing else "")
                + (f"'bottom' carries {', '.join(extra)}, which 'top' does not. " if extra else "")
                + "Add the column with a Formula on the side that lacks it, where the invention is visible"
            )

        # A column that is all nulls has no type of its own -- null is the one value belonging to
        # every family, which is what this library's expression language already says about it -- so
        # it takes the other side's. Refusing it would refuse an ordinary frame: a branch that
        # matched nothing, or a column no row on that side happened to populate.
        disagreeing = [
            (name, top.schema[name], bottom.schema[name])
            for name in top.columns
            if top.schema[name] != bottom.schema[name] and pl.Null not in (top.schema[name], bottom.schema[name])
        ]
        if disagreeing:
            named = "; ".join(f"{name} is {left} on `top` and {right} on `bottom`" for name, left, right in disagreeing)
            raise ConfigError(
                f"input ports 'top' and 'bottom' give the same column different types, so one side "
                f"would be rewritten on the way through: {named}"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        top, bottom = inputs["top"], inputs["bottom"]

        # Both sides are read into `top`'s column order rather than one being refused for having its
        # own: they carry the same columns and agreeing types by the time this runs, so the order one
        # side happens to have is not a fact worth stopping for. An all-null column takes the other
        # side's type, in whichever direction it appears, so the result carries the type the data
        # actually has rather than the absence of one.
        settled = {
            name: (bottom.schema[name] if top.schema[name] == pl.Null else top.schema[name]) for name in top.columns
        }
        aligned = [frame.select(pl.col(name).cast(settled[name]) for name in top.columns) for frame in (top, bottom)]
        return Ports({"rows": pl.concat(aligned, how="vertical")})


def _stacks_two_frames() -> WorkedExample:
    inputs = Ports(
        {
            "top": pl.DataFrame({"code": ["A1", "B2"], "amount": [10.0, -5.0]}),
            "bottom": pl.DataFrame({"code": ["C3"], "amount": [7.5]}),
        }
    )
    outputs = Ports({"rows": pl.DataFrame({"code": ["A1", "B2", "C3"], "amount": [10.0, -5.0, 7.5]})})
    return WorkedExample(
        scenario="carry two frames on as one",
        configuration={},
        explanation="Every row of `top`, then every row of `bottom`. Nothing is removed and nothing "
        "is re-ordered, so a reader can point at any row and say which side it came from.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _order_may_differ() -> WorkedExample:
    inputs = Ports(
        {
            "top": pl.DataFrame({"code": ["A1"], "amount": [10.0]}),
            "bottom": pl.DataFrame({"amount": [7.5], "code": ["C3"]}),
        }
    )
    outputs = Ports({"rows": pl.DataFrame({"code": ["A1", "C3"], "amount": [10.0, 7.5]})})
    return WorkedExample(
        scenario="stack a frame whose columns are in a different order",
        configuration={},
        explanation="The same columns written in another order are the same data, so `bottom` is "
        "read into `top`'s order rather than refused. A column one side does not have at all is a "
        "different matter and is refused.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Union())
