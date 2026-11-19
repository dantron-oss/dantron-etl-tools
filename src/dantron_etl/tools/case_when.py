"""CaseWhen: an ordered ladder of cases, the first one a row satisfies deciding it.

A single condition and value is Formula's job when there is only one; the reason this tool exists
is the ladder that has more than one, and where the cases can overlap -- a disposition that reads
"superseded" before it ever asks whether the amount is missing, even though a superseded row might
also have no amount. So cases are read in the order they are configured, and the row is labelled by
the first one it satisfies. Nothing later "optimises" that into an order-independent form: an
overlapping ladder has exactly one correct reading, and it is the one the author wrote down.

A condition is three-valued, matching the query engine underneath rather than folk intuition: a
comparison against a missing value is null, not false, so a row whose condition cannot be answered
is not a match for that case and falls through to the next one -- and, failing all of them, to
``otherwise`` or to null.

Every ``then`` and the optional ``otherwise`` are required to agree on one coarse family of value --
text, number, date, or truth value, with null belonging to all of them -- and refused at compile
time, naming the offending branch, when they do not. The query engine would otherwise render a
number or a boolean as text the moment it sits beside one, silently, which is a column of
plausible-looking strings that nobody notices until much later.

The output column is new unless the configuration says ``replace: true``, exactly as a Formula
entry does. Without the flag, a name already on the frame is refused: a step that writes over a
column a later step reads is a wrong answer with no failure attached to it. With the flag and no
such column, it is refused too, because the flag claims a column is there to repair and a
misspelling would otherwise write a new one while the column meant for repair kept its old values.

A replaced column is written where it already was, and every branch is compiled against the frame as
it arrived -- so a case may read the column it writes, and reads the values that were there before
this step ran, which is what the blank-to-null and collapse-a-status steps this flag exists for all
do.
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

# The coarse families a branch's value may belong to, named the way an author would read a
# refusal. Kept as small a set as the query engine itself distinguishes for this purpose: a whole
# number and a decimal are one family throughout, because separating them would refuse a ladder the
# engine answers perfectly well.
_TEXT: Final[str] = "text"
_NUMBER: Final[str] = "a number"
_DATE: Final[str] = "a date"
_BOOLEAN: Final[str] = "a truth value"
_NULL: Final[str] = "null"


def _family_of(dtype: pl.DataType) -> str:
    """The coarse family one compiled branch's dtype belongs to, for comparing it against another."""
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
    """Label every row of ``rows`` by the first case it satisfies, or ``otherwise``, or null."""

    name = "case_when"
    description = "Add a column labelling each row by the first case it satisfies in an ordered ladder."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "cases": ConfigField(
                type="array",
                description="The ladder, read in order; the first case a row satisfies decides it. "
                "An empty ladder is refused rather than treated as a constant column.",
                required=True,
                items=ConfigField(
                    type="object",
                    description="One rung of the ladder: the condition a row is tested against, and "
                    "the value a row that satisfies it is given.",
                    fields={
                        "when": ConfigField(
                            type="expression",
                            description="The condition for this case; a row whose condition is "
                            "null, not true, falls through to the next case.",
                            required=True,
                        ),
                        "then": ConfigField(
                            type="expression",
                            description="The value a row matching this case is given; must agree "
                            "in type family with every other 'then' and with 'otherwise'.",
                            required=True,
                        ),
                    },
                ),
            ),
            "otherwise": ConfigField(
                type="expression",
                description="The value a row that satisfies no case is given; left out, such a row is null.",
                required=False,
            ),
            "output": ConfigField(
                type="string",
                description="The column this ladder writes; a name the input already carries is "
                "refused unless 'replace' says the overwrite was meant.",
                required=True,
            ),
            # One canonical wording of this flag across the catalog, given the default an
            # array element could never carry: here the field is top level, so it is filled.
            "replace": dataclasses.replace(REPLACE_FIELD, default=False),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Rules are read in order and the first one a row satisfies decides it, even where "
                "cases overlap. A null condition is not a match, so such a row falls through rather "
                "than being folded into false. Every 'then' and 'otherwise' must agree on one family "
                "of value -- text, number, date, or truth value -- checked before a single row runs. "
                "Under 'replace' the ladder writes the whole column rather than only the rows a case "
                "matched, so a row no rung reached takes 'otherwise', or null when there is none."
            ),
            examples=(_disposition_ladder_example(), _no_otherwise_example(), _replacing_example()),
            notes=(
                "A branch that gives null belongs to every family, because a null does; it is never "
                "the disagreement that gets refused.",
                "A replacement writes the whole column rather than only the rows a case matched, so "
                "a row no case reached takes 'otherwise', or null when there is none, exactly as it "
                "would in a new column, instead of keeping the value it had.",
                "A '/' whose divisor is zero stops the step, but only on a row that reaches it, as "
                "in SQL's CASE: a case's condition is checked on the rows no earlier case took, its "
                "value on the rows it takes, and 'otherwise' on the rows no case took. So a case that "
                "takes the zero rows first, such as when: '[months] = 0', guards a later case that "
                "divides by [months].",
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse what ``required`` cannot: a ladder with no rungs at all.

        ``required=True`` on ``cases`` is satisfied by ``[]`` -- an empty list is a value -- so a
        ladder emptied by a loader reaches here rather than the template. It is refused rather than
        run as a constant column, because a constant column is what Formula is for and says so.

        A blank ``output`` is refused in ``validate_inputs``, by ``check_output_column``, whatever
        ``replace`` says.
        """
        if not config["cases"]:
            raise ConfigError("config field 'cases' must name at least one case")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Every condition must be a truth value, every value must agree with the others, in family.

        The output column is ruled on here too, because whether it collides with a column already on
        the frame is a question only the frame can answer.
        """
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
    """Check each '/' only on the rows that reach the part of the ladder it sits in, as SQL's CASE does.

    A rung's condition is used on the rows no earlier rung took, and its value on the rows its own
    condition then takes. 'otherwise' is used on the rows no rung took. A null condition takes no
    row, as it does when the column is built. So a rung that sends the zero rows elsewhere first
    guards the division in the rung after it.
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
    """Compile one branch and hold its family against every branch already checked.

    A branch whose family is ``null`` belongs to every family and never becomes the reference and
    never disagrees with one, because a null is the one value that meets anything. The first branch
    that does carry a family becomes the reference the rest are read against, and the refusal names
    the offending branch's own path and quotes its own text -- a file may hold a dozen branches, and
    that is the only address that sends someone to the right line.
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
        scenario="an eight-branch ladder reduced to three rungs and a fall-through",
        configuration={
            "cases": [
                {"when": "[status] = 'superseded'", "then": "'superseded'"},
                {"when": "[amount] = 0", "then": "'zero_total'"},
                {"when": "[category] is null", "then": "'no_category'"},
            ],
            "otherwise": "'counted'",
            "output": "disposition",
        },
        explanation="Each row is labelled by the first case it satisfies; the last row satisfies "
        "none of the three and falls through to 'otherwise' rather than being left unlabelled.",
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
        scenario="with no 'otherwise', a row matching no case is left null rather than labelled",
        configuration={"cases": [{"when": "[amount] < 0", "then": "'negative'"}], "output": "band"},
        explanation="The second row satisfies no case and no fallback is configured, so it is null "
        "rather than silently taking a label that was never written for it.",
        inputs=inputs,
        outputs=outputs,
    )


def _replacing_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"key": [1, 2, 3], "company": ["  Acme  ", "", "Beta"]})})
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2, 3], "company": ["Acme", None, "Beta"]})})
    return WorkedExample(
        scenario="a ladder reads the column it writes, and writes it back in place",
        configuration={
            "cases": [{"when": "trim([company]) = ''", "then": "null"}],
            "otherwise": "trim([company])",
            "output": "company",
            "replace": True,
        },
        explanation="'replace: true' says the overwrite was meant, so 'company' keeps its position "
        "and no column is added; the cases read the values that were there before this step ran.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(CaseWhen())
