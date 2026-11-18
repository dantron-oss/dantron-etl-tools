"""AssertRows: an invariant a run must satisfy, and a run that stops when it does not.

**One output port, not two.** "Which rows failed" is the first thing anyone asks of a guard, and a
second port carrying them cannot answer it: when the assertion fails the run stops, so nothing
downstream ever reads that port, and when the assertion passes the port is empty. A port that is
empty whenever it is readable and unreadable whenever it is not has no possible consumer, so the
offending rows go in the raised message instead. A flow that wants them as *data* rather than as a
refusal reaches for Filter, which already keeps both sides of a predicate.

**A violated assertion is a data fault, never a configuration fault and never a bare execution
failure.** It raises :class:`~.authoring.DataError`, because the reader's next move differs from every
other member of the family: go and look at the data, not at the run and not at the file.

**A predicate over no rows is not satisfied.** Every row of an empty frame satisfies every
condition, which is exactly the silence a filter that matched nothing leaves behind -- the most
common real failure these guards exist to catch. So an empty frame fails by default, and
``allow_empty`` is how a flow whose branch is legitimately sometimes empty says so, in writing.

**A null predicate result is a failure, always, with no knob to change that.** Filter offers one for
its null rows because both dispositions are legitimate for a *selection*; for an invariant, "could
not tell" is never a satisfied condition, and a setting that let it pass would exist only to
silence the tool.
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..authoring import (
    MAX_REPORTED_ROWS,
    REGISTRY,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    compile_predicate_for,
)

# Published rather than private, so a caller reading a capped message knows the cap without reading
# this module's source; equal to the boundary's own cap, since a message this long is one log line's
# worth regardless of which tool raised it.
REPORTED_ROW_LIMIT: Final[int] = MAX_REPORTED_ROWS


class AssertRows:
    """Refuse to let `rows` through unless every row satisfies `predicate`."""

    name = "assert_rows"
    description = "Stop the run unless every row of `rows` satisfies a predicate."
    category = "assert"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "predicate": ConfigField(
                type="expression",
                description="The invariant every row must satisfy; a null result counts as a failure.",
                required=True,
            ),
            "label": ConfigField(
                type="string",
                description="What a reader recognises the assertion by in a log; optional.",
                required=False,
            ),
            "allow_empty": ConfigField(
                type="boolean",
                description="Whether a frame with no rows passes instead of failing.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every row of 'rows' is tested against 'predicate'; if every row satisfies it, "
                "'rows' passes through unchanged. If any row does not -- including a row whose "
                "result is null, which is never treated as a pass -- the run stops with a "
                "DataError naming the assertion, how many rows failed out of how many, and a "
                "handful of the offending rows. An empty frame fails by default, because every "
                "row of it vacuously satisfies any predicate; 'allow_empty' is how a flow whose "
                "branch is legitimately sometimes empty says so."
            ),
            examples=(_satisfied_example(), _allowed_empty_example()),
            notes=(
                "A null predicate result is a failure with no knob to change it: an invariant has "
                "no legitimate 'could not tell', unlike Filter's selection, which has a 'nulls' "
                "option for exactly that reason.",
                "An empty frame fails unless 'allow_empty' says otherwise, because a filter that "
                "matched nothing leaves an empty frame behind and every assertion on it would "
                "otherwise pass in silence.",
            ),
            field_help={
                "label": "Shown in the failure message alongside the predicate text; the predicate "
                "itself is always shown, labelled or not.",
                "allow_empty": "Defaults to false: an empty frame fails until a flow says in writing "
                "that this branch may be empty.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Compile the predicate against the incoming schema, refusing anything that is not a condition.

        A column that is not there, and an unresolved placeholder, both belong to a configuration
        fault, never a data fault -- so both are refused here, before a single row is judged.
        """
        compile_predicate_for(inputs["rows"], config["predicate"], "predicate")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        predicate = compile_predicate_for(frame, config["predicate"], "predicate")
        heading = _heading(config)

        if frame.height == 0:
            if config["allow_empty"]:
                return Ports({"rows": frame})
            raise DataError(
                f"{heading}: the input is empty, and every row of an empty frame vacuously "
                f"satisfies any predicate; set allow_empty to accept that",
                offending=frame,
                total=0,
            )

        satisfied = predicate.fill_null(False)
        failing = frame.filter(~satisfied)
        if failing.height > 0:
            raise DataError(heading, offending=failing, total=frame.height)

        return Ports({"rows": frame})


def _heading(config: Mapping[str, Any]) -> str:
    """The part of the message naming the assertion, before the counts and examples are appended."""
    label = config.get("label")
    predicate = config["predicate"]
    if label:
        return f"assertion {label!r} (`{predicate}`) failed"
    return f"assertion `{predicate}` failed"


def _satisfied_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.0, 20.5]})
    inputs = Ports({"rows": frame})
    outputs = Ports({"rows": frame})
    return WorkedExample(
        scenario="every row satisfies the predicate, so it passes through unchanged",
        configuration={"predicate": "[amount] > 0", "label": "every figure is positive"},
        explanation="Both rows have a positive amount, so nothing stops the run and 'rows' arrives "
        "on the other side exactly as it came in.",
        inputs=inputs,
        outputs=outputs,
    )


def _allowed_empty_example() -> WorkedExample:
    frame = pl.DataFrame({"key": pl.Series([], dtype=pl.String), "amount": pl.Series([], dtype=pl.Float64)})
    inputs = Ports({"rows": frame})
    outputs = Ports({"rows": frame})
    return WorkedExample(
        scenario="an empty frame passes because the flow says in writing that it may be",
        configuration={"predicate": "[amount] > 0", "allow_empty": True},
        explanation="An empty frame would otherwise fail, since every row of it vacuously satisfies "
        "the predicate; 'allow_empty: true' is how this branch declares that emptiness is expected.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(AssertRows())
