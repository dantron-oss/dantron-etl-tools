"""NormalizeText: a match key from a human-entered name, in a published, not caller-chosen, order.

This tool exists to be the step before a join, and the join is only as good as the key produced
here. ``operations`` says *which* of five reductions run -- fold accents, lower-case, remove
punctuation, strip configured suffixes, collapse whitespace -- and :data:`OPERATION_ORDER` says the
order they run in, which is not the order the list names them. The operations form an enabling
chain: folding accents is what lets ``SARL`` be recognised as the configured suffix ``sarl``, and
removing punctuation is what lets ``Ltd.`` be recognised as ``ltd``. A caller-chosen order could run
an enabling step after the one it enables, and the tool would stop being idempotent -- the same
configuration would then produce two different keys depending on how many times it had already run,
which is exactly the failure a join run on either side by a different step, at a different time,
cannot afford.

The suffix words themselves are configuration and never a built-in list, because a jurisdiction's
company suffixes are not this library's business to know. A blank key -- a name that reduces to
nothing but whitespace, punctuation or suffixes -- comes back null rather than ``""``, because a
column of empty strings would join every unusable name to every other one.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    check_output_column,
    compile_for,
    compiled_dtype,
    refuse_division_by_zero,
)

# The sequence every selected operation runs in, regardless of the order ``operations`` names them.
# Accent folding and punctuation removal come before suffix stripping because each is what makes a
# suffix recognisable in the first place -- folded, ``SARL`` reads as the configured word ``sarl``;
# with its stop removed, ``Ltd.`` reads as ``ltd``. Whitespace collapsing runs last because removing
# a suffix leaves the whitespace that separated it from the rest of the name, and collapsing before
# that point would have nothing yet to clean up.
OPERATION_ORDER: Final[tuple[str, ...]] = (
    "fold_accents",
    "lower",
    "remove_punctuation",
    "strip_suffixes",
    "collapse_whitespace",
)

# Every mark a decomposition can produce: nonspacing, spacing-combining and enclosing. Stripped after
# NFKD decomposition, which is what turns an accented letter into the plain letter plus the mark that
# is then removed -- and, being a compatibility decomposition, also what turns a full-width or
# ligature spelling into its plain letters, with nothing transliterated.
_MARKS: Final[str] = r"\p{M}"

# Every Unicode punctuation category, deleted rather than replaced with a space: an initialism's
# stops and a name's apostrophe both disappear, which is what makes ``A.B.C.`` meet ``ABC`` and an
# apostrophe typed as ``'`` meet one a word processor curled into ``'``.
_PUNCTUATION: Final[str] = r"\p{P}"

# The Unicode whitespace property, not one character: a non-breaking space is what a name pasted out
# of a spreadsheet carries routinely, and an ASCII-only class would leave it in.
_WHITESPACE_RUN: Final[str] = r"\s+"

# A generous ceiling on how many times a suffix can be stripped from the same value before giving up
# on convergence. Each pass removes at least one trailing word, so a value with this many words would
# be an extraordinary input; the loop itself exits the moment a pass changes nothing; the cap is a
# safety margin for the vectorized loop below, not the property that makes stripping idempotent.
_MAX_SUFFIX_PASSES: Final[int] = 64


class NormalizeText:
    """Reduce ``source`` to a match key in ``output``, in the published operation order."""

    name = "normalize_text"
    description = "Reduce a name to a join-ready match key: fold accents, case, punctuation, suffixes, spacing."
    category = "text"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "source": ConfigField(
                type="expression",
                description="The name to reduce to a match key.",
                required=True,
            ),
            "output": ConfigField(
                type="string",
                description="The column the match key is written to.",
                required=True,
                non_empty=True,
            ),
            "operations": ConfigField(
                type="array",
                description="Which reductions run; the order they run in is OPERATION_ORDER, not "
                "this list's own order.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One of the published operations.",
                    enum=OPERATION_ORDER,
                ),
            ),
            "suffixes": ConfigField(
                type="array",
                description="Whole trailing words to strip, case-insensitively and repeatedly; "
                "required when 'strip_suffixes' is selected, and only then.",
                items=ConfigField(type="string", description="One suffix word, as the caller spells it."),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every selected operation runs in OPERATION_ORDER, never in the order 'operations' "
                "names them, because the operations enable one another: accent folding and "
                "punctuation removal are what make a configured suffix recognisable, so both have to "
                "run before suffix stripping for the tool to stay idempotent. A value that reduces to "
                "nothing is written as null, never as an empty string."
            ),
            examples=(_party_example(), _plain_case_example()),
            notes=(
                "'suffixes' carries no built-in words: a jurisdiction's company suffixes are "
                "the caller's data, not this library's.",
            ),
            field_help={
                "operations": "Selects which reductions run; OPERATION_ORDER decides the sequence.",
                "suffixes": "The caller's own suffix words; there is no default list.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        operations: list[str] = config["operations"] or []
        seen: set[str] = set()
        for operation in operations:
            if operation in seen:
                raise ConfigError(
                    f"config field 'operations' repeats {operation!r}; a repeat means the list was "
                    f"read as a sequence, and this field only selects which operations run -- the "
                    f"order they run in is OPERATION_ORDER"
                )
            seen.add(operation)

        wants_suffixes = "strip_suffixes" in operations
        suffixes = config["suffixes"]
        if wants_suffixes and not suffixes:
            raise ConfigError(
                "config field 'suffixes' must name at least one word when 'strip_suffixes' is "
                "one of 'operations'; stripping nothing would leave every suffix on the key"
            )
        if not wants_suffixes and suffixes is not None:
            raise ConfigError(
                "config field 'suffixes' names words to strip, but 'strip_suffixes' is not one "
                "of 'operations'; the words would silently do nothing"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        frame = inputs["rows"]
        column, dtype = compiled_dtype(frame, config["source"], "source")
        if dtype != pl.String:
            raise ConfigError(
                f"config field 'source' must give text to normalise, but column {column!r} holds {dtype}, not text"
            )
        check_output_column(frame, config["output"], "output", replace=None)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        source_expr = compile_for(frame, config["source"], "source")
        refuse_division_by_zero(frame, config["source"], "source")
        output = config["output"]
        selected = set(config["operations"])

        working = frame.select(source_expr.alias(output))[output]
        for operation in OPERATION_ORDER:
            if operation not in selected:
                continue
            working = _OPERATIONS[operation](working, config["suffixes"])

        working = working.replace("", None)
        return Ports({"rows": frame.with_columns(working.alias(output))})


def _fold_accents(values: pl.Series, _suffixes: Sequence[str] | None) -> pl.Series:
    return values.str.normalize("NFKD").str.replace_all(_MARKS, "", literal=False)


def _lower(values: pl.Series, _suffixes: Sequence[str] | None) -> pl.Series:
    return values.str.to_lowercase()


def _remove_punctuation(values: pl.Series, _suffixes: Sequence[str] | None) -> pl.Series:
    return values.str.replace_all(_PUNCTUATION, "", literal=False)


def _collapse_whitespace(values: pl.Series, _suffixes: Sequence[str] | None) -> pl.Series:
    return values.str.replace_all(_WHITESPACE_RUN, " ", literal=False).str.strip_chars()


def _strip_suffixes(values: pl.Series, suffixes: Sequence[str] | None) -> pl.Series:
    """Strip every configured word that trails the value, case-insensitively and repeatedly.

    One pass removes at most one trailing word -- ``"Acme Pty Ltd"`` genuinely carries two -- so the
    pass repeats until a pass changes nothing, which is also what keeps the operation idempotent on
    its own: run again on an already-stripped value, the first pass finds nothing to remove and the
    loop exits immediately.
    """
    alternatives = "|".join(re.escape(word) for word in suffixes or ())
    pattern = rf"(?i)(?:^|\s)(?:{alternatives})\s*$"
    current = values
    for _ in range(_MAX_SUFFIX_PASSES):
        stripped = current.str.replace_all(pattern, "", literal=False)
        if stripped.equals(current):
            return current
        current = stripped
    return current


_OPERATIONS: Final[dict[str, Any]] = {
    "fold_accents": _fold_accents,
    "lower": _lower,
    "remove_punctuation": _remove_punctuation,
    "strip_suffixes": _strip_suffixes,
    "collapse_whitespace": _collapse_whitespace,
}


def _party_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {"key": [1, 2, 3], "party": ["ACME Pty Ltd.", "acme pty ltd", "..."]},
                schema={"key": pl.Int64, "party": pl.String},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "key": [1, 2, 3],
                    "party": ["ACME Pty Ltd.", "acme pty ltd", "..."],
                    "party_key": ["acme", "acme", None],
                },
                schema={"key": pl.Int64, "party": pl.String, "party_key": pl.String},
            )
        }
    )
    return WorkedExample(
        scenario="two spellings of one party meet, and pure punctuation reduces to nothing",
        configuration={
            "source": "[party]",
            "output": "party_key",
            "operations": list(OPERATION_ORDER),
            "suffixes": ["ltd", "pty"],
        },
        explanation="A shouted spelling with a stop and its plain lowercase equivalent both reduce to "
        "'acme'; a name that is nothing but punctuation reduces to nothing, and is written null "
        "rather than ''.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _plain_case_example() -> WorkedExample:
    inputs = Ports({"rows": pl.DataFrame({"party": ["Acme  Holdings"]}, schema={"party": pl.String})})
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {"party": ["Acme  Holdings"], "party_key": ["acme holdings"]},
                schema={"party": pl.String, "party_key": pl.String},
            )
        }
    )
    return WorkedExample(
        scenario="only case and spacing are asked for, with no suffix list at all",
        configuration={"source": "[party]", "output": "party_key", "operations": ["lower", "collapse_whitespace"]},
        explanation="'strip_suffixes' is absent from the operation list and 'suffixes' from the configuration, "
        "which is allowed: the two travel together only when suffix stripping is actually selected.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(NormalizeText())
