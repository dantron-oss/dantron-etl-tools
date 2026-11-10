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

import re
from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..tool import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    check_output_column,
    compile_for,
    compiled_dtype,
)

_STRUCT_COLUMN: Final[str] = "__regex_extract_groups__"
_TEXT_COLUMN: Final[str] = "__regex_extract_groups_text__"
_ROW_COLUMN: Final[str] = "__regex_extract_groups_row__"


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
            "all_matches": ConfigField(
                type="boolean",
                description="When true, every match in the text becomes its own row, not only the "
                "first. A row whose text matches nothing is kept once, with every group null.",
                required=False,
                default=False,
            ),
            "match_number_column": ConfigField(
                type="string",
                description="Name for the column numbering each match from 1, in the order the text "
                "states them. Required with all_matches, and refused without it.",
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
                "exists precisely to keep a partial match from surfacing as a silent partial answer. "
                "With `all_matches`, every match in the text becomes its own row."
            ),
            examples=(_period_example(), _code_example(), _every_match_example()),
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
                "With all_matches, each match becomes a row: the input row repeated, that match's "
                "groups, and its number counted from 1. Matches are found left to right and do not "
                "overlap. A row with no match, an empty text or a missing text is kept once, with the "
                "groups and the number null. No input row is lost.",
                "With all_matches, Python's own regular expressions find the matches, one distinct "
                "text at a time. This is a known exception to working on whole columns. The engine can "
                "list every match, but not every match's groups. Cutting each match out and reading it "
                "again on its own gives wrong groups when the pattern looks at the text beside the "
                "match, as \\b and \\B do. So each match is read where it sits in the full text.",
                "The engine's list of matches is still checked against Python's for every text. A text "
                "where the two find different matches is refused, not given rows. A pattern that can "
                "match empty text, such as a*, is the usual cause.",
                "position_column is refused with all_matches: the match number already says which "
                "match comes first, and one offset for several matches would be wrong for all but one.",
            ),
            field_help={
                "source": "An expression, so the text matched need not already be a stored column.",
                "pattern": "Judged by the engine that will run it, not by a second regular-expression "
                "library that agrees with it only most of the time.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        every = bool(config.get("all_matches"))
        number = config.get("match_number_column")
        if every and number is None:
            raise ConfigError(
                "config field 'match_number_column' is required when 'all_matches' is true: without "
                "it, two rows from one text cannot be told apart or put back in order"
            )
        if not every and number is not None:
            raise ConfigError(
                f"config field 'match_number_column' names {number!r}, but 'all_matches' is not true, "
                f"so there would be no match number to write; set all_matches: true or remove the field"
            )
        if every and config.get("position_column") is not None:
            raise ConfigError(
                "config field 'position_column' cannot be used with 'all_matches': the match number "
                "already says which match the text states first, and one offset cannot describe "
                "several matches"
            )

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

        number = config.get("match_number_column")
        if config.get("all_matches") and number is not None:
            check_output_column(frame, number, "match_number_column", replace=None)
            if number in names:
                raise ConfigError(
                    f"config field 'match_number_column' names {number!r}, which is also a named "
                    f"group of this pattern; one of the two would overwrite the other"
                )
            _python_pattern(config["pattern"], names)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        pattern = config["pattern"]
        source_expr = compile_for(frame, config["source"], "source")
        if config.get("all_matches"):
            return Ports({"rows": _every_match(frame, source_expr, pattern, config["match_number_column"])})
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


def _python_pattern(pattern: str, names: tuple[str, ...]) -> re.Pattern[str]:
    """The pattern compiled by Python's own regular expressions, which all_matches runs it with.

    The engine accepts some spellings Python does not, such as (?<name>...) or a flag in the middle
    of a pattern. Those are refused here, before any row is read, rather than failing mid-run.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        raise ConfigError(
            f"config field 'pattern' cannot be used with 'all_matches': {error}. Every match is found "
            f"with Python's regular expressions, which do not accept this spelling; write named "
            f"groups as (?P<name>...) and put flags at the start, in pattern: `{pattern}`"
        ) from error
    if tuple(compiled.groupindex) != names:
        raise ConfigError(
            f"config field 'pattern' cannot be used with 'all_matches': Python reads its named groups "
            f"as {', '.join(compiled.groupindex)} where the engine reads {', '.join(names)}, in "
            f"pattern: `{pattern}`"
        )
    return compiled


def _every_match(frame: pl.DataFrame, source_expr: pl.Expr, pattern: str, number: str) -> pl.DataFrame:
    """One row per match, the input row repeated; one row with nulls where nothing matches.

    Each distinct text is searched once with ``re.finditer``, over the whole text, so a group that
    depends on the text around its match is read correctly. The engine's own list of whole matches
    is the check: a text where Python finds different matches is refused, not given wrong rows.
    """
    names = _group_names(frame, source_expr, pattern)
    compiled = _python_pattern(pattern, names)

    keyed = frame.with_columns(source_expr.alias(_TEXT_COLUMN)).with_row_index(_ROW_COLUMN)
    distinct = keyed.select(pl.col(_TEXT_COLUMN).drop_nulls().unique())
    engine = distinct.with_columns(pl.col(_TEXT_COLUMN).str.extract_all(pattern).alias(_STRUCT_COLUMN))

    texts: list[str] = []
    numbers: list[int] = []
    groups: dict[str, list[str | None]] = {name: [] for name in names}
    disagree: list[str] = []
    for text, expected in engine.iter_rows():
        found = list(compiled.finditer(text))
        if [match.group(0) for match in found] != expected:
            disagree.append(text)
            continue
        for index, match in enumerate(found, start=1):
            texts.append(text)
            numbers.append(index)
            for name in names:
                groups[name].append(match.group(name))

    if disagree:
        raise DataError(
            f"regex_extract_groups with all_matches: Python and the engine found different matches "
            f"in some texts, so their groups cannot be trusted. A pattern that can match empty "
            f"text, such as a*, is the usual cause, in pattern: `{pattern}`",
            offending=frame.filter(keyed[_TEXT_COLUMN].is_in(disagree)),
            total=frame.height,
        )

    matches = pl.DataFrame(
        {_TEXT_COLUMN: texts, **groups, number: numbers},
        schema={_TEXT_COLUMN: pl.String, **dict.fromkeys(names, pl.String), number: pl.Int64},
    )
    return (
        keyed.join(matches, on=_TEXT_COLUMN, how="left")
        .sort([_ROW_COLUMN, number], nulls_last=True, maintain_order=True)
        .drop(_ROW_COLUMN, _TEXT_COLUMN)
    )


def _every_match_example() -> WorkedExample:
    pattern = r"(?P<ref_prefix>[A-Z]{3})-(?P<ref_number>\d+)"
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {"key": [1, 2, 3], "note": ["ABC-12 then XYZ-345", "no reference", "see DEF-7"]},
                schema={"key": pl.Int64, "note": pl.String},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 1, 2, 3],
                    "note": ["ABC-12 then XYZ-345", "ABC-12 then XYZ-345", "no reference", "see DEF-7"],
                    "ref_prefix": ["ABC", "XYZ", None, "DEF"],
                    "ref_number": ["12", "345", None, "7"],
                    "match": [1, 2, None, 1],
                },
                schema={
                    "key": pl.Int64,
                    "note": pl.String,
                    "ref_prefix": pl.String,
                    "ref_number": pl.String,
                    "match": pl.Int64,
                },
            )
        }
    )
    return WorkedExample(
        scenario="list every reference a note names, one row each",
        configuration={
            "source": "[note]",
            "pattern": pattern,
            "all_matches": True,
            "match_number_column": "match",
        },
        explanation="The first note names two references, so it becomes two rows, numbered 1 and 2. "
        "The second names none and is kept once, with nulls. The third names one.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


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
