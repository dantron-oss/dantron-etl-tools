"""Formula: named columns, each one expression, every expression read against the input alone.

A single ``formula: "[out] = expr"`` field would put the output name, the expression and the count
of columns a step writes into one opaque string, so ``columns`` is instead an ordered array of
``{name, value, replace}`` objects -- the nested shape the contract's ``items``/``fields`` seam
exists for, because one realistic step writes several derived columns at once.

Every ``value`` is compiled against the *input* frame only, never against a frame that already
carries the columns this step is about to add. That matches the query engine's own multi-column
``with_columns``: a single call is handed every expression at once and evaluates each of them
against the frame as it stood before any of them ran, so an author who wants a second value to read
a first writes two Formula steps -- which says so in the pipeline, rather than hiding it in a list's
order. Compiling every expression once, in one ``with_columns`` call, is also what gives a replaced
column its position back: the query engine keeps a name already in the frame exactly where it was
and only appends the names that are new.

A divisor is not guarded here. Money is naturally divided by a rate quoted the other way round, and
what a zero or a null divisor gives back is the query engine's own three-valued, floating-point
answer -- pinned in the tests, not reinterpreted here, because a row-level guarantee that an
infinity never reaches a money column belongs to a rule an author writes downstream (a CaseWhen
guard, an Assert step), not to a silent decision a derived-column step makes on their behalf.
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
    require_not_blank,
)


class Formula:
    """Add or replace named columns of ``rows``, each computed by an expression against the input alone."""

    name = "formula"
    description = "Add or replace named columns, each computed by an expression against the input alone."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="One entry per column to write, applied together against the input "
                "frame; an empty list is refused rather than treated as a no-op step.",
                required=True,
                items=ConfigField(
                    type="object",
                    description="One derived column: the name it is written under, the expression "
                    "that computes it, and whether it may overwrite a column already there.",
                    fields={
                        "name": ConfigField(
                            type="string",
                            description="The column this entry writes; blank, and a name repeated "
                            "by another entry, are both refused.",
                            required=True,
                        ),
                        "value": ConfigField(
                            type="expression",
                            description="Computed against the input frame only -- never against a "
                            "column another entry in this same list is about to write.",
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
                "Every entry's expression is compiled against the frame that arrived, so entry two "
                "can never see entry one's output -- only two Formula steps in sequence can. A "
                "replaced column keeps the position it already had; a new one is appended in the "
                "order the list gives."
            ),
            examples=(_monthly_figure_example(), _multi_column_with_replace_example()),
            notes=(
                "A null divisor gives null; a zero divisor gives the query engine's infinity, and "
                "0/0 gives its not-a-number. Neither is guarded here -- keeping an infinity out of a "
                "money column is a rule an author writes with CaseWhen or Assert, not a decision "
                "this tool makes silently on every row.",
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse what the template's ``required`` cannot: an empty list, a blank or repeated name.

        ``required=True`` on ``columns`` is satisfied by ``[]`` -- it is a value, just an empty one
        -- so a dropped list of entries reaches here rather than the template, and must be refused
        loudly rather than run as a step that writes nothing. A blank name and a name two entries
        both claim are refused the same way, because neither is a question the template's own type
        and required checks can ask.
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
        """Compile every entry's expression against the frame, and rule on its output column.

        Both questions need the frame in hand rather than the template alone: whether the
        expression's columns exist on this particular ``rows`` port, and whether the name it writes
        under is new, already there and meant to be replaced, or already there and not.
        """
        frame = inputs["rows"]
        for index, entry in enumerate(config["columns"]):
            compile_for(frame, entry["value"], f"columns[{index}].value")
            check_output_column(frame, entry["name"], f"columns[{index}].name", replace=entry.get("replace"))

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
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
        scenario="a per-month figure divides by the rate, because the rate is quoted the other way round",
        configuration={
            "columns": [
                {
                    "name": "monthly",
                    "value": "([line_amount] / [currency_rate]) / [months_in_period]",
                }
            ]
        },
        explanation="The amount is divided by the rate and then by the count of months, appended as "
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
        scenario="one entry repairs 'code' in place, another appends a new column",
        configuration={
            "columns": [
                {"name": "code", "value": "upper(strip([code]))", "replace": True},
                {"name": "doubled", "value": "[line_amount] * 2"},
            ]
        },
        explanation="'code' is overwritten because its entry says 'replace: true' and keeps its "
        "original position; 'doubled' has no name collision to resolve and is appended.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(Formula())
