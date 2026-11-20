"""Union: stack `bottom` under `top`, matching columns by name and keeping every row.

Both sides must have the same columns with the same types; column order may differ. A column on one
side only is refused, never filled with nulls. This is SQL's UNION ALL BY NAME.

Why: docs/design/tools.md#union
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


class Union:
    """Stack `bottom` under `top`: same columns, same types, every row of both."""

    name = "union"
    description = (
        "Stack two tables with the same columns, one under the other, keeping every row "
        "(SQL's UNION ALL BY NAME)."
    )
    category = "combine"
    input_ports = ("top", "bottom")
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {}

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use union to put two tables with the same columns into one, for example this year's "
                "orders under last year's. `rows` has every row of `top`, then every row of `bottom`, "
                "with the columns in `top`'s order. Columns are matched by name, so their order may "
                "differ between the two. This is SQL's UNION ALL BY NAME: duplicate rows are kept."
            ),
            examples=(_stacks_two_frames(), _order_may_differ()),
            notes=(
                "Both sides must have the same columns. A column on one side only is refused, never "
                "filled with nulls. Add it with a formula step, where it is visible.",
                "Each column must have the same type on both sides, or the step is refused, naming "
                "both types. A column that is all null on one side takes the other side's type.",
                "Nothing is de-duplicated: a row on both sides comes out twice. Use remove_duplicates "
                "after it if you want one of each.",
                "Rows keep their order: all of `top`, then all of `bottom`. Sort afterwards if you "
                "need another order.",
            ),
            field_help={},
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a column on one side only, and a column whose types differ, from the schemas alone."""
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

        # An all-null column has no type of its own, so it may differ; it takes the other side's.
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

        # Both sides in `top`'s column order; an all-null column takes the other side's type.
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
        scenario="stack two tables into one",
        configuration={},
        explanation="Every row of 'top', then every row of 'bottom'. Nothing is removed and nothing is reordered.",
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
        scenario="stack a table whose columns are in a different order",
        configuration={},
        explanation="The columns are matched by name, so 'bottom' is read into 'top''s column order. A column "
        "that only one side has would be refused.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Union())
