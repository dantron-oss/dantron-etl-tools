"""RegexExtractGroups: read every named group of one pattern out of a text column.

Each named group becomes a text column. A row that does not match is null in every group. With
`all_matches`, every match becomes its own row.

Why: docs/design/tools.md#regex_extract_groups
"""

import re
from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..authoring import (
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
    refuse_rows_without_an_answer,
)

_STRUCT_COLUMN: Final[str] = "__regex_extract_groups__"
_TEXT_COLUMN: Final[str] = "__regex_extract_groups_text__"
_ROW_COLUMN: Final[str] = "__regex_extract_groups_row__"


class RegexExtractGroups:
    """Extract every named group of `pattern` from `source` into new text columns."""

    name = "regex_extract_groups"
    description = "Pull several pieces out of a text column into new columns, using a pattern."
    category = "text"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "source": ConfigField(
                type="expression",
                description="The text to search, such as '[note]'.",
                required=True,
            ),
            "pattern": ConfigField(
                type="string",
                description="A regular expression with named groups, written (?P<name>...). Each "
                "group becomes a column of that name.",
                required=True,
                non_empty=True,
            ),
            "position_column": ConfigField(
                type="string",
                description="Optional: the name of a column for where the match starts in the text, "
                "counted from 0, or null where nothing matched.",
                required=False,
                default=None,
            ),
            "all_matches": ConfigField(
                type="boolean",
                description="Set true to give every match in the text its own row, not only the first.",
                required=False,
                default=False,
            ),
            "match_number_column": ConfigField(
                type="string",
                description="With all_matches: the name of a column numbering each match from 1.",
                required=False,
                default=None,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use regex_extract_groups to pull several pieces out of one text column at once, "
                "such as a prefix and a number from 'INV-1042', or the months and years from 'Jan "
                "2026 to Mar 2026'. Each named group in the pattern becomes a new text column at the "
                "end of `rows`, in the pattern's order. With all_matches, every match in the text "
                "becomes its own row. For one piece, regex_extract in a formula is simpler."
            ),
            examples=(_code_example(), _period_example(), _every_match_example()),
            notes=(
                "A row that does not match the whole pattern is null in every group, never in some.",
                "On a match, a group the pattern makes optional (with '?', or one side of an '|') can "
                "still be null, because the match did not need it.",
                "The new columns are always text, even when they look like numbers. Use cast_columns "
                "afterwards.",
                "The column names come only from the pattern's named groups; there is no separate "
                "list to keep in step.",
                "An unnamed group is refused; write (?:...) for a group that should not become a "
                "column. A pattern with no named group is refused, and so is a group name already on "
                "the input.",
                "The pattern is checked by polars, the engine that runs it, before any row is read.",
                "'position_column' is for flows that try several patterns and keep whichever matched "
                "earliest in the text.",
                "The position counts bytes, not characters, so on text with accents or other "
                "non-ASCII characters it is larger than a person would count. Compare two positions; "
                "do not use one to cut text.",
                "With all_matches, each match becomes a row: the input row repeated, that match's "
                "groups, and its number from 1. Matches are found left to right and do not overlap.",
                "With all_matches, a row with no match, empty text or no text is kept once, with "
                "nulls. No input row is lost.",
                "With all_matches, Python's regular expressions find the groups, one distinct text at "
                "a time, a known exception to whole-column work. Patterns using \\b or \\B read the "
                "text around a match, so each match is read in its full text.",
                "With all_matches, polars' list of matches is checked against Python's for each text, "
                "and a text where they differ is refused. A pattern that can match empty text, such as "
                "a*, is the usual cause. A spelling Python does not accept, such as (?<name>...), is "
                "refused before any row is read.",
                "With all_matches, 'match_number_column' is required, and 'position_column' is refused: "
                "one position cannot describe several matches.",
            ),
            field_help={
                "source": "Any expression giving text, so the text need not be a stored column.",
                "pattern": "For example (?P<prefix>[A-Z]+)-(?P<number>\\d+). In YAML, put it in double "
                "quotes and double each backslash.",
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
        """Refuse a source that is not text, and a pattern polars rejects or that names no usable group.

        Also refuses a position or match-number column that clashes with an input column or a group.
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
            # Like a group name, it must be new and must not clash with a group.
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
        refuse_rows_without_an_answer(frame, config["source"], "source")
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
            # Same pattern, same text, so the position belongs to the match whose groups were read.
            extracted = extracted.with_columns(source_expr.str.find(pattern).alias(position))
        return Ports({"rows": extracted})


def _group_names(frame: pl.DataFrame, source_expr: pl.Expr, pattern: str) -> tuple[str, ...]:
    """The pattern's named groups, in order, read from polars, which will run the pattern.

    polars and Python's `re` do not accept the same patterns (look-behind differs), so polars decides.
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
    """The pattern compiled by Python's `re`, which all_matches uses.

    polars accepts some spellings Python does not, such as (?<name>...); those are refused here, before any row.
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
    """One row per match, the input row repeated; one row of nulls where nothing matches.

    Each distinct text is searched once with `re.finditer` over the whole text. polars' list of matches
    is the check: a text where Python finds different matches is refused.
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
        explanation="The first note names two references, so it becomes two rows, numbered 1 and 2. The second "
        "names none and is kept once, with nulls. The third names one.",
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
        scenario="read a period out of a free-text note",
        configuration={
            "source": "[note]",
            "pattern": r"(?P<start_month>[A-Za-z]{3}) (?P<start_year>\d{4}) to "
            r"(?P<end_month>[A-Za-z]{3}) (?P<end_year>\d{4})",
        },
        explanation="The first note matches the whole pattern, so all four groups are filled. The second has a "
        "month the pattern does not accept, so nothing matches and all four groups are null together.",
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
        scenario="split a code into its prefix and its number",
        configuration={"source": "[code]", "pattern": r"(?P<code_prefix>[A-Z]+)-(?P<code_number>\d+)"},
        explanation="Both new columns are text, even though the number looks like a number. Use cast_columns "
        "afterwards if you need a number.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(RegexExtractGroups())
