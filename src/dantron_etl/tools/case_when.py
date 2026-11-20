"""CaseWhen: an ordered list of cases; the first one a row meets decides its value.

A null condition is not a match, so the row falls through. Every value must be the same kind (text,
number, date or true/false). The output is a new column unless `replace: true`.

Why: docs/design/tools.md#case_when
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Final

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
    compile_predicate_for,
    compiled_dtype,
    refuse_division_by_zero,
)

# The kinds of value a branch may give, named as a refusal states them. Whole and decimal numbers
# are one kind, because polars mixes them without trouble.
_TEXT: Final[str] = "text"
_NUMBER: Final[str] = "a number"
_DATE: Final[str] = "a date"
_BOOLEAN: Final[str] = "a truth value"
_NULL: Final[str] = "null"


def _family_of(dtype: pl.DataType) -> str:
    """The kind of value (text, number, date, true/false, null) a compiled branch gives."""
    if dtype == pl.Null:
        return _NULL
    if dtype == pl.Boolean:
        return _BOOLEAN
    if dtype.is_numeric():
        return _NUMBER
    if dtype.is_temporal():
        return _DATE
    return _TEXT


class CaseWhen:
    """Set a column of `rows` by the first case each row meets, or `otherwise`, or null."""

    name = "case_when"
    description = "Set a column by rules: if this, then that; else something else."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "cases": ConfigField(
                type="array",
                description="The cases, checked in order. The first one a row meets decides its value.",
                required=True,
                items=ConfigField(
                    type="object",
                    description="One case: a condition, and the value for rows that meet it.",
                    fields={
                        "when": ConfigField(
                            type="expression",
                            description="The condition, such as '[amount] < 0'.",
                            required=True,
                        ),
                        "then": ConfigField(
                            type="expression",
                            description="The value for a row that meets the condition, such as "
                            "\"'refund'\". Text needs quotes inside the expression.",
                            required=True,
                        ),
                    },
                ),
            ),
            "otherwise": ConfigField(
                type="expression",
                description="The value for a row that meets no case. Left out, such a row gets null.",
                required=False,
            ),
            "output": ConfigField(
                type="string",
                description="The column to write, such as 'band'.",
                required=True,
            ),
            # The shared wording of this flag, with the default filled in, since this field is top level.
            "replace": dataclasses.replace(REPLACE_FIELD, default=False),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use case_when to label or work out a column by rules, like a spreadsheet's nested "
                "IF or SQL's CASE WHEN. Each case is a condition and a value. For each row, the first "
                "case it meets gives the value; a row that meets none gets 'otherwise', or null. The "
                "result goes in a new column at the end of `rows`, or replaces a column when "
                "'replace' is true."
            ),
            examples=(_disposition_ladder_example(), _no_otherwise_example(), _replacing_example()),
            notes=(
                "Cases are checked in the order written, and the first one a row meets wins, even "
                "when cases overlap.",
                "A condition that is null because of a missing value does not count as met, so the "
                "row moves on to the next case.",
                "Every 'then' and 'otherwise' must give the same kind of value: text, a number, a date, "
                "or true/false. A mix is refused before any row runs, naming the case.",
                "A value of null fits with any kind, so it is never the mismatch that gets refused.",
                "With 'replace', the whole column is rewritten, so a row no case reached gets "
                "'otherwise', or null if there is none, instead of keeping its old value.",
                "With 'replace', the cases read the column's values as they were before this step, so "
                "a case can read the column it writes.",
                "A name already on the input is refused unless 'replace' is true, and 'replace' on a "
                "column that is not there is refused too.",
                "Dividing by zero stops the run, but only on a row that reaches that division, as in "
                "SQL's CASE: a condition is checked on the rows no earlier case took, a value on the "
                "rows its case takes, and 'otherwise' on the rows no case took.",
                "So a case that takes the zero rows first, such as when: '[months] = 0', guards a "
                "later case that divides by [months].",
                "An empty list of cases is refused.",
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse an empty list of cases.

        `required` is satisfied by an empty list, so this check is needed. A constant column is formula's job.
        """
        if not config["cases"]:
            raise ConfigError("config field 'cases' must name at least one case")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a condition that is not true/false, values of different kinds, and a clash with the output column."""
        frame = inputs["rows"]
        check_output_column(frame, config["output"], "output", replace=config["replace"])

        for index, case in enumerate(config["cases"]):
            compile_predicate_for(frame, case["when"], f"cases[{index}].when")

        reference: tuple[str, str, str] | None = None
        for index, case in enumerate(config["cases"]):
            reference = _check_family(frame, case["then"], f"cases[{index}].then", reference)
        if config.get("otherwise") is not None:
            _check_family(frame, config["otherwise"], "otherwise", reference)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        cases = config["cases"]
        _refuse_division_by_zero_where_used(frame, config)

        ladder = pl.when(compile_predicate_for(frame, cases[0]["when"], "cases[0].when")).then(
            compile_for(frame, cases[0]["then"], "cases[0].then")
        )
        for index, case in enumerate(cases[1:], start=1):
            ladder = ladder.when(compile_predicate_for(frame, case["when"], f"cases[{index}].when")).then(
                compile_for(frame, case["then"], f"cases[{index}].then")
            )
        if config.get("otherwise") is not None:
            ladder = ladder.otherwise(compile_for(frame, config["otherwise"], "otherwise"))

        return Ports({"rows": frame.with_columns(ladder.alias(config["output"]))})


def _refuse_division_by_zero_where_used(frame: pl.DataFrame, config: Mapping[str, Any]) -> None:
    """Check each '/' only on the rows that reach it, as SQL's CASE does.

    A case's condition is used on the rows no earlier case took, its value on the rows it takes, and
    'otherwise' on the rows no case took. So a case that takes the zero rows first guards a later division.
    """
    taken = pl.lit(False)
    for index, case in enumerate(config["cases"]):
        reached = ~taken
        refuse_division_by_zero(frame, case["when"], f"cases[{index}].when", among=reached)
        takes = compile_predicate_for(frame, case["when"], f"cases[{index}].when").fill_null(False)
        refuse_division_by_zero(frame, case["then"], f"cases[{index}].then", among=reached & takes)
        taken = taken | takes
    if config.get("otherwise") is not None:
        refuse_division_by_zero(frame, config["otherwise"], "otherwise", among=~taken)


def _check_family(
    frame: pl.DataFrame,
    text: str,
    path: str,
    reference: tuple[str, str, str] | None,
) -> tuple[str, str, str] | None:
    """Compile one branch and compare its kind of value with the first branch that had one.

    A null branch fits any kind. Returns the reference (kind, column, path) for the next branch.
    """
    column, dtype = compiled_dtype(frame, text, path)
    family = _family_of(dtype)
    if family == _NULL:
        return reference
    if reference is None:
        return (family, path, text)

    reference_family, reference_path, _reference_text = reference
    if family != reference_family:
        raise ConfigError(
            f"config field {path!r} does not agree with the type family every branch of this ladder "
            f"must share: column {column!r} holds {dtype}, {family}, while {reference_path!r} gives "
            f"{reference_family}, in expression: `{text}`"
        )
    return reference


def _disposition_ladder_example() -> WorkedExample:
    schema: Mapping[str, Any] = {
        "key": pl.Int64,
        "status": pl.String,
        "amount": pl.Float64,
        "category": pl.String,
    }
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                [
                    (1, "superseded", 10.0, "standard"),
                    (2, "open", 0.0, "standard"),
                    (3, "open", 10.0, None),
                    (4, "open", 10.0, "standard"),
                ],
                schema=schema,
                orient="row",
            )
        }
    )
    outputs = Ports(
        {
            "rows": inputs["rows"].with_columns(
                pl.Series("disposition", ["superseded", "zero_total", "no_category", "counted"])
            )
        }
    )
    return WorkedExample(
        scenario="label each row by the first rule it meets",
        configuration={
            "cases": [
                {"when": "[status] = 'superseded'", "then": "'superseded'"},
                {"when": "[amount] = 0", "then": "'zero_total'"},
                {"when": "[category] is null", "then": "'no_category'"},
            ],
            "otherwise": "'counted'",
            "output": "disposition",
        },
        explanation="Each row gets the value of the first case it meets. The last row meets none of the three, so "
        "it gets the 'otherwise' value.",
        inputs=inputs,
        outputs=outputs,
    )


def _no_otherwise_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "amount": [-5.0, 5.0]})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2],
                    "amount": [-5.0, 5.0],
                    "band": ["negative", None],
                }
            )
        }
    )
    return WorkedExample(
        scenario="leave a row null when no case fits",
        configuration={"cases": [{"when": "[amount] < 0", "then": "'negative'"}], "output": "band"},
        explanation="The second row meets no case and there is no 'otherwise', so its band is null.",
        inputs=inputs,
        outputs=outputs,
    )


def _replacing_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [1, 2, 3], "company": ["  Acme  ", "", "Beta"]})})
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2, 3], "company": ["Acme", None, "Beta"]})})
    return WorkedExample(
        scenario="tidy a column in place, turning blanks into nulls",
        configuration={
            "cases": [{"when": "trim([company]) = ''", "then": "null"}],
            "otherwise": "trim([company])",
            "output": "company",
            "replace": True,
        },
        explanation="'replace: true' says to overwrite 'company', so it keeps its position and no column is "
        "added. The cases read the values as they were before this step.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(CaseWhen())
