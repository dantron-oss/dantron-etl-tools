"""RegexExtractGroups: every named part of one pattern, read from one column in a single pass.

Extracting a single group out of a piece of text is already ``regex_extract()`` in the expression
language, and a tool that only did that would be a worse spelling of a function that already
exists. What earns this tool its place is a property an expression cannot have: a pattern with
several named groups produces several columns from one pass over the text, and therefore makes one
decision per row rather than one decision per group. A row that matches the whole pattern fills
every group; a row that does not is null in every group, never in some -- a partial match is exactly
the case this tool exists to refuse, because it would otherwise flow onward as a null count with no
way to tell "did not parse" apart from "genuinely absent".

The output columns are never configured separately from the pattern: they are the pattern's own
named groups, read off the engine that will actually run it, in the order the pattern declares them.
Two places to write the same names is two places for them to disagree, and the pattern is the one
that decides.
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
    compile_for,
    compiled_dtype,
)

_STRUCT_COLUMN: Final[str] = "__regex_extract_groups__"


class RegexExtractGroups:
    """Extract every named group of ``pattern`` out of ``source``, appended as text columns."""

    name = "regex_extract_groups"
    description = "Extract a pattern's named groups from one column at once, all null together on a miss."
    category = "text"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "source": ConfigField(
                type="expression",
                description="The text the pattern is matched against.",
                required=True,
            ),
            "pattern": ConfigField(
                type="string",
                description="A regular expression whose named groups become the output columns, in "
                "pattern order. An unnamed capturing group is refused; write (?:...) for one that "
                "need not be captured.",
                required=True,
                non_empty=True,
            ),
            "position_column": ConfigField(
                type="string",
                description="Name for an extra column carrying how far into the text the pattern "
                "matched, counted from zero, and null where it did not. Left out, no such column.",
                required=False,
                default=None,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every named group of `pattern` becomes a text column, appended after the input "
                "columns in the order the pattern declares them. A row that matches the whole pattern "
                "fills every group; a row that does not is null in every group, because this tool "
                "exists precisely to keep a partial match from surfacing as a silent partial answer."
            ),
            examples=(_period_example(), _code_example()),
            notes=(
                "There is no field listing the output column names beside the pattern: they are read "
                "off the pattern's own named groups, so there is nowhere for the two to disagree.",
                "'position_column' exists for flows that try several patterns and keep whichever "
                "matched earliest in the text. Without it, a flow can tell that two patterns both "
                "matched but not which one the text states first, and 'first stated' is usually what "
                "a reader means: a line names the term it is billing for before it names any other.",
                "The offset counts bytes, not characters, so on text carrying anything outside plain "
                "ASCII it is larger than the position a person would count to. Compare two offsets "
                "against each other, which is what it is for and is always right; do not read one as "
                "'the Nth character', and do not slice text with it.",
            ),
            field_help={
                "source": "An expression, so the text matched need not already be a stored column.",
                "pattern": "Judged by the engine that will run it, not by a second regular-expression "
                "library that agrees with it only most of the time.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a bad `source` or a bad `pattern` before any row is read.

        `source` has to give text, or the pattern has nothing to run against; `pattern` has to
        compile in the engine that will actually run it and has to name at least one group that is
        not already a column, or it would silently produce nothing useful.
        """
        frame = inputs["rows"]
        column, dtype = compiled_dtype(frame, config["source"], "source")
        if dtype != pl.String:
            raise ConfigError(
                f"config field 'source' must give text to match a pattern against, but column "
                f"{column!r} holds {dtype}, not text"
            )
        source_expr = compile_for(frame, config["source"], "source")
        names = _group_names(frame, source_expr, config["pattern"])

        position = config["position_column"]
        if position is not None:
            # Held to the same rule as a group name: an output column may not quietly replace an
            # input, and it may not collide with a group this very pattern is about to produce.
            check_output_column(frame, position, "position_column", replace=None)
            if position in names:
                raise ConfigError(
                    f"config field 'position_column' names {position!r}, which is also a named group "
                    f"of this pattern; one of the two would overwrite the other"
                )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        pattern = config["pattern"]
        source_expr = compile_for(frame, config["source"], "source")
        extracted = (
            frame.lazy()
            .with_columns(source_expr.str.extract_groups(pattern).alias(_STRUCT_COLUMN))
            .unnest(_STRUCT_COLUMN)
            .collect()
        )
        position = config["position_column"]
        if position is not None:
            # The same pattern against the same text, so the offset belongs to the very match whose
            # groups were just unnested. It is asked for separately rather than derived, because a
            # struct of captures has nowhere to put a number that is about the match rather than
            # about any one group.
            extracted = extracted.with_columns(source_expr.str.find(pattern).alias(position))
        return Ports({"rows": extracted})


def _group_names(frame: pl.DataFrame, source_expr: pl.Expr, pattern: str) -> tuple[str, ...]:
    """The pattern's named groups, in pattern order, refusing every way a pattern fails to name its output.

    Read off the engine that will actually run the pattern rather than out of Python's own regular
    expression library, because the two do not judge the same pattern the same way -- a look-behind
    is exactly the case that decides it, compiling in one and not the other.
    """
    try:
        schema = frame.lazy().select(source_expr.str.extract_groups(pattern).alias(_STRUCT_COLUMN)).collect_schema()
    except pl.exceptions.PolarsError as error:
        raise ConfigError(
            f"config field 'pattern' is not a pattern the engine accepts: {type(error).__name__}: "
            f"{error}, in pattern: `{pattern}`"
        ) from error

    fields = schema[_STRUCT_COLUMN].fields
    unnamed = [item.name for item in fields if item.name.isdigit()]
    if unnamed:
        raise ConfigError(
            f"config field 'pattern' has unnamed capturing group(s) {', '.join(unnamed)}, which have "
            f"no column to go to; name every group, or write (?:...) for one that need not be "
            f"captured, in pattern: `{pattern}`"
        )
    if not fields:
        raise ConfigError(
            f"config field 'pattern' names no groups at all, so it would produce no columns; a single "
            f"group is already better written with regex_extract() in an expression, in pattern: "
            f"`{pattern}`"
        )

    names = tuple(item.name for item in fields)
    collisions = [name for name in names if name in frame.columns]
    if collisions:
        raise ConfigError(
            f"config field 'pattern' names group(s) {', '.join(collisions)}, which already exist as "
            f"column(s) on input port 'rows'; writing over them would destroy the text they were "
            f"derived from"
        )
    return names


def _period_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [10, 11],
                    "note": ["support Jan 2026 to Dec 2026", "support Jan 2026 to Smarch 2026"],
                },
                schema={"key": pl.Int64, "note": pl.String},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [10, 11],
                    "note": ["support Jan 2026 to Dec 2026", "support Jan 2026 to Smarch 2026"],
                    "start_month": ["Jan", None],
                    "start_year": ["2026", None],
                    "end_month": ["Dec", None],
                    "end_year": ["2026", None],
                },
                schema={
                    "key": pl.Int64,
                    "note": pl.String,
                    "start_month": pl.String,
                    "start_year": pl.String,
                    "end_month": pl.String,
                    "end_year": pl.String,
                },
            )
        }
    )
    return WorkedExample(
        scenario="read a service period out of a free-text note",
        configuration={
            "source": "[note]",
            "pattern": r"(?P<start_month>[A-Za-z]{3}) (?P<start_year>\d{4}) to "
            r"(?P<end_month>[A-Za-z]{3}) (?P<end_year>\d{4})",
        },
        explanation="The first note matches the whole pattern and fills all four groups. The second "
        "has a month nobody recognises, so the whole pattern fails to match and all four groups come "
        "back null together, rather than a start with no end.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _code_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"code": ["AC-1042", "not a code"]}, schema={"code": pl.String})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["AC-1042", "not a code"],
                    "code_prefix": ["AC", None],
                    "code_number": ["1042", None],
                },
                schema={"code": pl.String, "code_prefix": pl.String, "code_number": pl.String},
            )
        }
    )
    return WorkedExample(
        scenario="split a party code into its prefix and its number",
        configuration={"source": "[code]", "pattern": r"(?P<code_prefix>[A-Z]+)-(?P<code_number>\d+)"},
        explanation="Both groups are text, even though the number looks numeric: a group is the text "
        "that matched, and guessing a dtype would make a column's type depend on which rows arrived.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(RegexExtractGroups())
