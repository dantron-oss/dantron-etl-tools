"""NormalizeText: turn a typed name into a matching key for a join.

`operations` chooses which clean-ups run; `OPERATION_ORDER` fixes the order they run in, so the
step gives the same key however often it runs. A name that cleans to nothing becomes null.

Why: docs/design/tools.md#normalize_text
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
    refuse_rows_without_an_answer,
)

# The order the selected operations always run in. Why: docs/design/tools.md#normalize_text
OPERATION_ORDER: Final[tuple[str, ...]] = (
    "fold_accents",
    "lower",
    "remove_punctuation",
    "strip_suffixes",
    "collapse_whitespace",
)

# Every combining mark. Removed after NFKD splits an accented letter into letter plus mark.
_MARKS: Final[str] = r"\p{M}"

# Every Unicode punctuation character, deleted, so 'A.B.C.' meets 'ABC' and curly apostrophes meet straight ones.
_PUNCTUATION: Final[str] = r"\p{P}"

# Any Unicode space, including the non-breaking space pasted names often carry.
_WHITESPACE_RUN: Final[str] = r"\s+"

# A safety cap on suffix-stripping passes. The loop stops as soon as a pass changes nothing.
_MAX_SUFFIX_PASSES: Final[int] = 64


class NormalizeText:
    """Clean `source` into a matching key in `output`, in the fixed operation order."""

    name = "normalize_text"
    description = "Clean up names into a matching key: lower case, no punctuation, accents or suffixes."
    category = "text"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "source": ConfigField(
                type="expression",
                description="The text to clean, such as '[customer_name]'.",
                required=True,
            ),
            "output": ConfigField(
                type="string",
                description="The name of the new column for the cleaned key, such as 'name_key'.",
                required=True,
                non_empty=True,
            ),
            "operations": ConfigField(
                type="array",
                description="Which clean-ups to apply. They always run in a fixed order, whatever "
                "order you list them in.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One clean-up, such as 'lower' or 'remove_punctuation'.",
                    enum=OPERATION_ORDER,
                ),
            ),
            "suffixes": ConfigField(
                type="array",
                description="Words to remove from the end, such as 'ltd' or 'inc'. Needed with "
                "'strip_suffixes', and only then.",
                items=ConfigField(type="string", description="One word to remove from the end, such as 'ltd'."),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use normalize_text to make a matching key from names people typed, so that "
                "'ACME Ltd.' and 'acme' can be joined. It adds the cleaned key as a new column at the "
                "end of `rows`. The clean-ups are, in the order they always run: fold_accents (é to "
                "e), lower, remove_punctuation, strip_suffixes (the words you list) and "
                "collapse_whitespace."
            ),
            examples=(_plain_case_example(), _party_example()),
            notes=(
                "The clean-ups always run in the fixed order above, not the order listed, so running "
                "the step twice gives the same key as running it once.",
                "Accents and punctuation go before suffixes so that 'SARL' and 'Ltd.' are recognised "
                "as the suffixes 'sarl' and 'ltd'.",
                "There is no built-in suffix list. Company suffixes differ by country, so you list them.",
                "Suffixes are whole words at the end, matched ignoring case, and removed again until "
                "none is left: 'Acme Pty Ltd' loses both.",
                "Punctuation is deleted, not replaced by a space, so 'A.B.C.' becomes 'abc'.",
                "Any kind of space counts, including the non-breaking space common in pasted text.",
                "A name that cleans down to nothing is null, not empty text, so it never matches "
                "another empty name.",
                "Listing a clean-up twice is refused. 'suffixes' without 'strip_suffixes', or the "
                "other way round, is refused.",
                "The source must be text, and the output name must be new.",
            ),
            field_help={
                "operations": "For example ['lower', 'remove_punctuation', 'collapse_whitespace'].",
                "suffixes": "For example ['ltd', 'limited', 'inc'].",
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
        refuse_rows_without_an_answer(frame, config["source"], "source")
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
    """Remove every listed word from the end of the value, ignoring case, until none is left.

    One pass removes one trailing word ('Acme Pty Ltd' has two), so passes repeat until nothing changes.
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
        scenario="make two spellings of one company match, with every clean-up",
        configuration={
            "source": "[party]",
            "output": "party_key",
            "operations": list(OPERATION_ORDER),
            "suffixes": ["ltd", "pty"],
        },
        explanation="'ACME Ltd.' and 'acme' both become 'acme'. A name of nothing but punctuation cleans down to "
        "nothing, so it is null, not empty text.",
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
        scenario="lower-case names and tidy their spaces",
        configuration={"source": "[party]", "output": "party_key", "operations": ["lower", "collapse_whitespace"]},
        explanation="Only 'lower' and 'collapse_whitespace' are used, so no 'suffixes' list is needed.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(NormalizeText())
