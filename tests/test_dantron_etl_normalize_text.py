"""Tests for NormalizeText: a match key from a human-entered name, and the order that makes it one.

This tool exists to be the step before a join, and the join's correctness is exactly as good as the
key this produces. Both directions cost money. Too aggressive and two different organisations
collide, so a return line is matched back to the wrong original. Too timid and a return line finds
no original at all, and the revenue it was meant to cancel stays on the books. So the tests below
pin a case in each direction, not merely the happy path.

**The operation surface is a selection, not a sequence.** ``operations`` says *which* operations
run; the order they run in is published as ``OPERATION_ORDER`` and is not the order of the list.
That is the decision this file argues for hardest, because the obvious alternatives are both worse:

* A set of booleans (``lower``, ``collapse_whitespace``, ``remove_punctuation``) leaves the order
  implied by whatever order a dataclass happens to declare its fields in -- invisible in the
  configuration file, invisible in a rendered form, and changeable by an unrelated refactor.
* A caller-ordered list makes the order visible but hands the caller a footgun on the one step a
  join depends on. The operations form an *enabling chain*: folding accents is what lets ``SÀRL``
  be recognised as the configured suffix ``sarl``, and removing punctuation is what lets ``Ltd.``
  be recognised as ``ltd``. Any order that ran an enabling operation *after* the operation it
  enables would stop being idempotent -- normalising the output a second time would strip a suffix
  the first pass missed, so the same configuration would produce two different keys depending on
  how many times it had been run. Run on one side of a join by a nightly step and on the other by a
  backfill, that is a silent no-match.

A published constant is visible and pinned in the way a dataclass's field order is not: it can be
read, documented, and asserted against. The test below runs a scrambled list and a canonical list
and pins that they give the same key, which is what makes the claim testable at all. A repeated
operation is refused rather than deduplicated, because a repeat means the author believed the list
was a sequence, and telling them it is not is more useful than quietly agreeing with them.

**Idempotence** -- normalising an already-normalised value changes nothing -- is what makes it safe
to run this on either side of a join, in different steps, at different times. It is also what forces
two smaller rulings: ``strip_suffixes`` must strip *repeatedly* (or ``"Acme Ltd Ltd"`` would need two
runs), and whitespace collapsing must run *last* (or stripping a suffix would leave a trailing space
for the next run to remove).

**The suffix list is configuration, never a built-in.** A list of company suffixes is
jurisdiction-specific and this library ships to other organisations in other countries. The tool
supplies the mechanism; the caller supplies the words. The collision test below is the proof: the
same two names stay distinct or collapse together depending only on the caller's list.

**A blank key is not a key.** A key that collapses to ``""`` would match every other collapsed key
in a join, which is the silent wrong answer this whole build keeps finding. So an empty result is
null, whatever made it empty -- a missing value, an empty string, or a name that was nothing but
punctuation or nothing but a suffix.

Vocabulary is deliberately generic -- party, key, account code, quantity -- because the library
carries no caller's nouns.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    Ports,
    ToolError,
    check_tool,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.normalize_text import OPERATION_ORDER, NormalizeText

# ---------------------------------------------------------------------------
# Test data and helpers
# ---------------------------------------------------------------------------

NBSP = "\u00a0"  # what a name pasted out of a spreadsheet or a web page carries
CURLY = "\u2019"  # what a word processor turns a typed apostrophe into

# Every operation the tool has, asked for as a full reduction to a match key.
FULL: list[str] = ["fold_accents", "lower", "remove_punctuation", "strip_suffixes", "collapse_whitespace"]

# The caller's words, not the library's: a jurisdiction's suffixes belong in a configuration file.
SUFFIXES: list[str] = ["ltd", "limited", "pty", "sarl", "co"]

# The messy shapes a human-entered party name actually arrives in.
MESSY: list[str | None] = [
    "ACME Pty Ltd.",
    f"Acme{NBSP} Pty. Ltd",
    f"O{CURLY}Brien & Sons Ltd",
    "A.B.C. Holdings",
    "Ångström Limited",
    "Pacific Tobacco",
    "Ltd Brands",
    "  Northwind   Trading  ",
    "  acme pty ltd  ",
    "Ltd",
    "...",
    "",
    None,
]


def _frame(values: Sequence[str | None], column: str = "party") -> pl.DataFrame:
    return pl.DataFrame({column: list(values)}, schema={column: pl.String})


def _config(operations: Sequence[str], suffixes: Sequence[str] | None = None, **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {"source": "[party]", "output": "party_key", "operations": list(operations)}
    if suffixes is not None:
        config["strip_suffixes"] = list(suffixes)
    return config | overrides


def _keys(
    values: Sequence[str | None],
    operations: Sequence[str],
    suffixes: Sequence[str] | None = None,
) -> list[str | None]:
    """The ``party_key`` column after one run through the boundary."""
    result = run_tool(NormalizeText(), Ports({"rows": _frame(values)}), _config(operations, suffixes))
    return result["rows"]["party_key"].to_list()


# ---------------------------------------------------------------------------
# The key: what it is, and the order it is built in
# ---------------------------------------------------------------------------


class TestNormalizeTextKeys:
    """What a name reduces to, and why the order it reduces in is the tool's and not the caller's."""

    @pytest.mark.parametrize(
        "operations",
        [
            pytest.param(FULL, id="the_published_order"),
            pytest.param(list(reversed(FULL)), id="written_backwards"),
            pytest.param(
                ["collapse_whitespace", "strip_suffixes", "lower", "fold_accents", "remove_punctuation"],
                id="written_in_no_order_at_all",
            ),
        ],
    )
    def test_the_operations_run_in_the_published_order_whatever_order_the_list_names_them(
        self,
        operations: list[str],
    ) -> None:
        """``operations`` selects; ``OPERATION_ORDER`` sequences. The list's own order carries nothing.

        This is the ruling the file argues for. If the list were a sequence, an author could put
        ``strip_suffixes`` before ``remove_punctuation``, and the tool would stop being idempotent:
        ``"Acme Pty. Ltd."`` would keep its suffix on the first run and lose it on the second, so
        the same configuration would give two keys depending on how often it had run. Reordering the
        list must therefore change nothing at all.
        """
        assert set(operations) == set(OPERATION_ORDER), "the parameters must cover every operation there is"

        keys = _keys(["  ÅNGSTRÖM  Pty. Ltd."], operations, SUFFIXES)

        assert keys == ["angstrom"], "the accents, case, punctuation, suffixes and spacing all resolve, in one order"

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            pytest.param("ACME Pty Ltd.", "acme", id="shouted_with_a_trailing_full_stop"),
            pytest.param("Acme  Pty. Ltd", "acme", id="doubled_space_and_a_stop_mid_suffix"),
            pytest.param(f"Acme{NBSP}Pty{NBSP}Ltd", "acme", id="pasted_with_non_breaking_spaces"),
            pytest.param("  acme pty ltd  ", "acme", id="already_lowercase_but_padded"),
            pytest.param("Acme Pty Limited", "acme", id="the_suffix_spelled_out"),
            pytest.param("ACME", "acme", id="no_suffix_at_all"),
            pytest.param(f"O{CURLY}Brien Ltd", "obrien", id="an_apostrophe_a_word_processor_curled"),
            pytest.param("O'Brien Ltd", "obrien", id="the_same_name_with_a_typed_apostrophe"),
            pytest.param("OBrien Ltd", "obrien", id="and_the_same_name_with_no_apostrophe_at_all"),
            pytest.param("O\u2014Brien\u2026 Ltd", "obrien", id="an_em_dash_and_an_ellipsis"),
        ],
    )
    def test_every_spelling_of_one_party_lands_on_one_key(self, spelling: str, expected: str) -> None:
        """The too-timid direction: a key that failed to meet its counterpart overstates revenue.

        A return line is matched back to the line it reverses on this key plus an account code plus
        a quantity -- deliberately not on price. So a key that keeps a doubled space, a trailing
        full stop or a shouted spelling means the return finds nothing to cancel, and the amount it
        was meant to reverse stays counted.

        The three apostrophe spellings are the case that decides how punctuation is recognised. A
        word processor silently replaces a typed ``'`` with ``’``, so the same name arrives both
        ways within one dataset, and an implementation built on the standard library's ASCII
        punctuation would remove one and keep the other. Punctuation has to be recognised by its
        Unicode category, not by a character list somebody typed -- which is also why the em dash
        and the ellipsis are here.
        """
        assert _keys([spelling], FULL, SUFFIXES) == [expected]

    def test_two_different_parties_keep_different_keys_until_the_caller_says_otherwise(self) -> None:
        """The too-aggressive direction, and the reason the suffix list is configuration.

        ``"Northwind Trading"`` and ``"Northwind Trading Group"`` may be one organisation or two, and
        which one it is depends on the caller's data, not on this library. So the tool keeps them
        apart by default and collapses them only when the caller puts the word in the list -- and a
        library shipping its own list of suffixes would have made that decision for every caller in
        every country at once.
        """
        parties = ["Northwind Trading", "Northwind Trading Group"]

        cautious = _keys(parties, FULL, SUFFIXES)
        deliberate = _keys(parties, FULL, [*SUFFIXES, "group"])

        assert cautious[0] != cautious[1], "two names the caller did not equate must not collide"
        assert cautious == ["northwind trading", "northwind trading group"]
        assert deliberate[0] == deliberate[1] == "northwind trading", "the caller's word, the caller's collision"

    @pytest.mark.parametrize(
        "operations",
        [
            pytest.param(FULL, id="every_operation"),
            pytest.param(["lower", "collapse_whitespace"], id="case_and_spacing_only"),
            pytest.param(["remove_punctuation", "strip_suffixes"], id="punctuation_and_suffixes_only"),
            pytest.param(["fold_accents"], id="accents_only"),
            pytest.param(["strip_suffixes"], id="suffixes_only"),
        ],
    )
    def test_normalising_an_already_normalised_value_changes_nothing(self, operations: list[str]) -> None:
        """Idempotence, for every selection -- which is what makes it safe on either side of a join.

        One side of a join is normalised by a nightly step and the other by a backfill written
        months apart. If a second pass could change a key, the two sides would agree only when they
        had been normalised the same number of times, and nothing in a flow records that.

        Two smaller rulings fall out of this and are pinned by the same assertion: suffixes are
        stripped repeatedly rather than one per run, and whitespace collapsing runs last so that a
        removed suffix cannot leave a trailing space behind for the next run to find.
        """
        suffixes = SUFFIXES if "strip_suffixes" in operations else None
        corpus = [*MESSY, "Acme Ltd Ltd", "Acme Pty Ltd Limited"]

        once = _keys(corpus, operations, suffixes)
        twice = _keys(once, operations, suffixes)

        assert twice == once, "a key run through the same configuration again must come back unchanged"

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(None, id="no_name_at_all"),
            pytest.param("", id="an_empty_name"),
            pytest.param("   ", id="a_name_of_nothing_but_spaces"),
            pytest.param(NBSP, id="a_name_of_one_non_breaking_space"),
            pytest.param("...", id="a_name_of_nothing_but_punctuation"),
            pytest.param("Ltd", id="a_name_that_is_only_a_suffix"),
            pytest.param("Ltd Ltd", id="a_name_that_is_only_suffixes"),
        ],
    )
    def test_a_key_that_would_be_blank_comes_back_null_and_not_an_empty_string(self, value: str | None) -> None:
        """A blank key matches every other blank key. Null is the only safe answer.

        This is the specific silent wrong answer the ruling exists to prevent: a column of ``""``
        keys joins every row that has no usable name to every other row that has no usable name, so
        a return line finds an 'original' it has nothing to do with and the wrong amount is
        cancelled. The engine's own join does not pair nulls, so null is what says 'no key'.

        Note that null in and empty in are deliberately indistinguishable *in the key*: neither
        carries a name. The text they came from is still in the frame beside it for a person to read.
        """
        assert _keys([value], FULL, SUFFIXES) == [None]

    @pytest.mark.parametrize(
        ("value", "operations", "expected"),
        [
            pytest.param("MÜLLER", ["lower"], "müller", id="case_folding_is_unicode_aware"),
            pytest.param("Straße", ["lower"], "straße", id="but_it_is_not_full_case_folding"),
            pytest.param("STRASSE", ["lower"], "strasse", id="so_the_two_spellings_of_that_name_stay_apart"),
            pytest.param("Ångström", ["fold_accents", "lower"], "angstrom", id="marks_are_folded_away"),
            pytest.param("MÜLLER", ["fold_accents", "lower"], "muller", id="folded_before_lowered"),
            pytest.param("Øresund", ["fold_accents", "lower"], "øresund", id="but_nothing_is_transliterated"),
            pytest.param("ＡＣＭＥ", ["fold_accents", "lower"], "acme", id="a_full_width_spelling"),
            pytest.param("ﬁnance", ["fold_accents", "lower"], "finance", id="a_ligature"),
        ],
    )
    def test_case_and_accent_folding_are_unicode_aware_without_transliterating(
        self,
        value: str,
        operations: list[str],
        expected: str,
    ) -> None:
        """Three rulings, two of which go in opposite directions, and each has a reason worth reading.

        *Case folding is Unicode-aware*, not ASCII: an ASCII-only fold leaves ``"MÜLLER"`` and
        ``"müller"`` differing in one character, so two spellings of one organisation fail to meet --
        exactly the failure this tool exists to prevent.

        *It is not full case folding.* ``"Straße"`` does not become ``"strasse"``. Full case folding
        would equate it with ``"STRASSE"``, which is sometimes what a reader wants and is a different
        operation deserving a different name; making ``lower`` mean it quietly would change the
        meaning of an operation callers already have in their files, and it is not what the engine's
        own lowercase kernel does.

        *Accent folding is compatibility folding, and it transliterates nothing.* Marks are removed,
        and so are the differences that are purely about how a character was encoded: a full-width
        spelling and a ligature both arrive from pasted data and both mean the plain letters, so
        folding them is the whole job. What is *not* done is transliteration: ``"Ø"`` has no
        decomposition and survives as itself. A transliteration table is language-specific -- the
        same character is 'o' to one reader and 'oe' to another -- and a library that shipped one
        would have made a linguistic decision on behalf of every country it ships to, which is the
        same line the suffix list draws.
        """
        assert _keys([value], operations) == [expected]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(f"Acme{NBSP}Holdings", "Acme Holdings", id="a_non_breaking_space"),
            pytest.param("Acme\tHoldings", "Acme Holdings", id="a_tab"),
            pytest.param("Acme\nHoldings", "Acme Holdings", id="a_newline"),
            pytest.param("Acme   Holdings", "Acme Holdings", id="a_run_of_spaces"),
            pytest.param(f"{NBSP} Acme Holdings \t", "Acme Holdings", id="padding_of_mixed_kinds"),
        ],
    )
    def test_whitespace_collapsing_covers_every_kind_of_space_and_not_only_a_plain_one(
        self,
        value: str,
        expected: str,
    ) -> None:
        """A non-breaking space is the case that decides the implementation.

        It is what a name pasted out of a spreadsheet or a web page carries, routinely. Anything
        that recognises only the plain space -- an ASCII character class, a hand-typed list of
        space characters, a pattern that switches Unicode matching off -- leaves it in, and then
        the same name typed by hand never meets the same name pasted in. Whitespace has to be
        recognised as the Unicode property, not as one character or a handful.
        """
        assert _keys([value], ["collapse_whitespace"]) == [expected]

    @pytest.mark.parametrize(
        ("value", "suffixes", "expected"),
        [
            pytest.param("Acme Pty Ltd", ["ltd", "pty"], "Acme", id="more_than_one_suffix_in_a_single_run"),
            pytest.param("Acme Ltd Ltd", ["ltd"], "Acme", id="the_same_suffix_twice"),
            pytest.param("Pacific Tobacco", ["co"], "Pacific Tobacco", id="only_a_whole_word_not_a_word_ending"),
            pytest.param("Ltd Brands", ["ltd"], "Ltd Brands", id="only_at_the_end_never_in_the_middle"),
            pytest.param("Acme LTD", ["ltd"], "Acme", id="matching_ignores_case_of_the_data"),
            pytest.param("Acme ltd", ["LTD"], "Acme", id="and_ignores_case_of_the_configured_word"),
            pytest.param(f"Acme{NBSP}Ltd", ["ltd"], "Acme", id="separated_by_a_non_breaking_space"),
            pytest.param("Acme Ltd  ", ["ltd"], "Acme", id="trailing_space_does_not_hide_the_suffix"),
            pytest.param("Acme Ltd.", ["ltd"], "Acme Ltd.", id="a_stop_hides_the_suffix_until_punctuation_goes"),
            pytest.param("Acme SÀRL", ["sarl"], "Acme SÀRL", id="an_accent_hides_it_until_accents_are_folded"),
            pytest.param("Acme Ltd", [], None, id="an_empty_list_is_refused_not_a_no_op"),
        ],
    )
    def test_strip_suffixes_matches_whole_trailing_words_repeatedly_and_case_insensitively(
        self,
        value: str,
        suffixes: list[str],
        expected: str | None,
    ) -> None:
        """The four matching rules, each with a failure mode on the other side of the ruling.

        *Whole trailing words only*: a suffix that matched a word *ending* would reduce
        ``"Pacific Tobacco"`` to ``"Pacific Toba"`` and collide it with an unrelated party.

        *Only at the end*: ``"Ltd Brands"`` is a name, and stripping mid-string would collapse
        distinct organisations onto one key.

        *Repeatedly*: one suffix per run is not idempotent, and ``"Acme Pty Ltd"`` genuinely carries
        two.

        *Case-insensitively, on both sides*: making it case-sensitive would make this operation's
        correctness depend on whether ``lower`` was also selected -- and it would fail *silently*,
        leaving the suffix on the key rather than raising anything.

        Trailing whitespace does not hide the suffix, and that is not cosmetic: whitespace
        collapsing runs *after* this, so a padded name still has its padding here. A stripper
        anchored hard to the end of the text would leave the suffix on ``"  acme pty ltd  "`` on
        the first run and remove it on the second, once collapsing had trimmed the padding -- which
        is a loss of idempotence, arriving through the back door.

        Punctuation is deliberately **not** part of this: ``"Ltd."`` is not the word ``"ltd"``.
        Removing punctuation is a separate operation, and its running first is precisely why the
        order is published rather than configured.
        """
        if expected is None:
            with pytest.raises(ConfigError):
                _keys([value], ["strip_suffixes"], suffixes)
            return

        assert _keys([value], ["strip_suffixes"], suffixes) == [expected]

    def test_the_rows_are_preserved_in_order_and_the_input_frame_is_left_untouched(self) -> None:
        """One key column appended; nothing else moves, and nothing upstream changes.

        A tool that reordered rows here would silently break every downstream step that pairs this
        frame with another by position, and the boundary only compares input *schemas* after a run,
        so an in-place edit of the data has to be caught by a test.
        """
        frame = pl.DataFrame(
            {
                "key": [3, 1, 2],
                "party": ["ACME Pty Ltd.", "Northwind Trading", None],
                "account_code": ["4000", "4000", "4100"],
                "quantity": [2, 1, 5],
            }
        )
        before = frame.clone()

        result = run_tool(NormalizeText(), Ports({"rows": frame}), _config(FULL, SUFFIXES))
        rows = result["rows"]

        assert rows.columns == [*frame.columns, "party_key"], "the key is appended, never inserted"
        assert_frame_equal(rows.select(frame.columns), frame), "every column and every row, in order"
        assert rows["party_key"].to_list() == ["acme", "northwind trading", None]
        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Refusals: a configuration that cannot mean what it says
# ---------------------------------------------------------------------------


def _refused_configurations() -> list[Any]:
    """(configuration, the word its refusal must quote) for each way this config can be self-contradictory."""
    return [
        pytest.param(_config([], []), "operations", id="no_operation_at_all"),
        pytest.param(_config(["shout"]), "shout", id="an_operation_that_does_not_exist"),
        pytest.param(_config(["lower", "lower"]), "lower", id="the_same_operation_twice"),
        pytest.param(_config(FULL, []), "strip_suffixes", id="suffix_stripping_with_no_words_to_strip"),
        pytest.param(_config(FULL), "strip_suffixes", id="suffix_stripping_with_the_list_left_out"),
        pytest.param(_config(["lower"], SUFFIXES), "strip_suffixes", id="words_to_strip_but_no_operation_to_strip"),
        pytest.param(_config(["lower"], output=""), "output", id="a_key_column_with_no_name"),
        pytest.param(
            _config(FULL, ["${rules.company_suffixes}"]),
            "${rules.company_suffixes}",
            id="a_suffix_list_whose_placeholder_was_never_resolved",
        ),
    ]


class TestNormalizeTextRefusals:
    """A configuration whose parts contradict each other, refused with the field to open."""

    @pytest.mark.parametrize(("configuration", "named"), _refused_configurations())
    def test_a_configuration_that_cannot_mean_what_it_says_is_refused(
        self,
        configuration: Mapping[str, Any],
        named: str,
    ) -> None:
        """Each of these would otherwise produce a key quietly, and the key would be wrong.

        No operation at all copies the column and calls it a key. An operation asked for with no
        words to work on leaves the suffix on every key. Words supplied with no operation to use
        them is a caller who believes the suffixes are being stripped and is wrong. A repeated
        operation is a caller who believes the list is a sequence. A key column with no name has
        nowhere to go. A suffix list still holding a ``${...}`` placeholder strips nothing, so every
        key keeps its suffix and the join that follows quietly finds fewer matches than it should.
        Every one of them fails silently if accepted, and every one has a field to open.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(NormalizeText(), Ports({"rows": _frame(["Acme Pty Ltd"])}), configuration)

        assert named in str(caught.value), f"the refusal must name {named!r}: {caught.value}"

    def test_an_output_name_that_is_already_a_column_is_refused(self) -> None:
        """A key is written beside the text it came from, never over it.

        The original spelling is the one thing a person can read when a match goes wrong, and a tool
        that overwrote it would destroy the evidence in the same step that created the problem.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(
                NormalizeText(),
                Ports({"rows": _frame(["Acme Pty Ltd"])}),
                _config(["lower"], output="party"),
            )

        assert "party" in str(caught.value), f"the refusal must name the column that collided: {caught.value}"

    @pytest.mark.parametrize(
        ("source", "named"),
        [
            pytest.param("[quantity]", "quantity", id="a_number_where_text_was_needed"),
            pytest.param("[not_a_column]", "not_a_column", id="a_column_that_is_not_there"),
            pytest.param("upper([quantity])", "quantity", id="a_text_function_on_a_number"),
        ],
    )
    def test_a_source_that_cannot_give_text_is_refused_naming_it(self, source: str, named: str) -> None:
        """The tool holds the frame, so it holds the schema, so the refusal is in the author's words.

        Handed a number, the engine's own string kernel raises about a dtype from inside a library
        the author has never heard of.
        """
        frame = pl.DataFrame({"party": ["Acme"], "quantity": [3]})

        with pytest.raises(ToolError) as caught:
            run_tool(NormalizeText(), Ports({"rows": frame}), _config(["lower"], source=source))

        message = str(caught.value)
        assert named in message, f"the refusal must name {named!r}: {message}"
        assert "source" in message, f"the refusal must name the field to open: {message}"

    def test_the_template_publishes_the_operation_order_and_keeps_the_suffix_words_in_configuration(self) -> None:
        """The surface itself: a selection of operations, a published order, and a caller's word list.

        ``OPERATION_ORDER`` is the whole argument for this design being made of a published constant
        rather than a dataclass's field order: it can be read, rendered and asserted against, and a
        refactor that reorders declarations cannot change it by accident. The enum and the order must
        name the same operations, or one of them is lying about what the tool does.

        The suffix list is an array of plain strings with no default, which is how the genericity
        line is drawn in the template rather than only in prose: there is nowhere for a built-in
        list of a single country's company suffixes to live.
        """
        template = NormalizeText().config_template()

        assert set(template) == {"source", "output", "operations", "strip_suffixes"}
        assert expression_fields(template) == ("source",), "the loader seam is the declared type"
        assert template["source"].required is True
        assert template["output"].required is True
        assert template["output"].non_empty is True
        assert template["operations"].required is True
        assert template["operations"].non_empty is True, "a normalisation that applies nothing is not a key"

        element = template["operations"].items
        assert element is not None, "the loader must be able to see inside the list, not just its type"
        assert set(element.enum or ()) == set(OPERATION_ORDER), "the enum and the published order must agree"
        assert len(set(OPERATION_ORDER)) == len(OPERATION_ORDER), "an operation cannot run twice"
        assert OPERATION_ORDER[-1] == "collapse_whitespace", (
            "collapsing runs last, or removing a suffix leaves a trailing space for the next run to find, "
            "and the tool stops being idempotent"
        )
        assert OPERATION_ORDER.index("remove_punctuation") < OPERATION_ORDER.index("strip_suffixes"), (
            "punctuation removal is what makes 'Ltd.' recognisable as the word 'ltd'"
        )
        assert OPERATION_ORDER.index("fold_accents") < OPERATION_ORDER.index("strip_suffixes"), (
            "accent folding is what makes 'SARL' recognisable as the configured word 'sarl'"
        )

        suffix_list = template["strip_suffixes"]
        assert suffix_list.type == "array"
        assert suffix_list.default is None, "a built-in list of one country's suffixes must have nowhere to live"
        assert suffix_list.items is not None and suffix_list.items.type == "string"


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestNormalizeTextInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_it_conforms_and_its_examples_show_a_key_and_a_key_that_is_not_one(self) -> None:
        """Conformance runs every example, so a documented key is a checked key.

        The two behaviours a reader will otherwise get wrong are the two an example has to carry:
        that two spellings land on one key, and that a name which reduces to nothing comes back null
        rather than blank.
        """
        tool = NormalizeText()

        check_tool(tool)

        assert tool.name == "normalize_text"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert len(examples) >= 2
        assert any("strip_suffixes" in example.configuration for example in examples), (
            "the suffix list is the genericity line, so an example must show a caller supplying one"
        )
        keys = [
            value
            for example in examples
            for name in example.outputs["rows"].columns
            if name == example.configuration.get("output")
            for value in example.outputs["rows"][name].to_list()
        ]
        assert None in keys, "an example must document a name that reduces to no key at all"
        assert any(example.check_row_order for example in examples), (
            "this tool preserves row order, and an example is where that is proved rather than claimed"
        )

    def test_it_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in test_dantron_etl_registry.py."""
        assert "normalize_text" in REGISTRY
        assert isinstance(REGISTRY.get("normalize_text"), NormalizeText)


# ---------------------------------------------------------------------------
# A key column named with nothing but whitespace
#
# ``non_empty`` on the template caught ``output: ""`` and nothing caught ``output: "   "``, so this
# tool would build a whole match key and write it into a column called three spaces. That column
# cannot be referenced by the join that is the only reason the key exists -- ``[   ]`` is not
# something an author can write -- so the step reported success and the step after it could not see
# what it produced.
#
# The rule now lives in the contract's shared seam, which this tool already calls, so the answer is
# the same one every other output-naming tool gives.
# ---------------------------------------------------------------------------


class TestNormalizeTextRefusesAKeyColumnNamedWithNothing:
    """Whitespace is not a name, and this tool used to accept it."""

    def test_a_key_column_named_only_with_whitespace_is_refused_in_the_shared_words(self) -> None:
        """The field named, the shared sentence, and one wording across every whitespace shape.

        ``output: ""`` was already caught by ``non_empty`` on the template. Three spaces were not,
        and this tool would build a whole key and write it into a column the join it exists for
        cannot reference.
        """
        refusals = set()
        for name in (" ", "   ", "\t", " \t "):
            with pytest.raises(ConfigError) as caught:
                run_tool(
                    NormalizeText(),
                    Ports({"rows": _frame(["Acme Pty Ltd"])}),
                    _config(["lower"], output=name),
                )
            message = str(caught.value)
            assert "'output'" in message, f"the refusal must name the field: {message}"
            assert "blank" in message and "whitespace" in message, (
                f"the refusal must say what a name has to be: {message}"
            )
            refusals.add(message)

        assert len(refusals) == 1, f"one rule, one wording; got: {sorted(refusals)}"

    def test_a_key_column_whose_name_is_merely_unusual_is_still_written(self) -> None:
        """Space around real text is a name, and the rule may not trim it away."""
        result = run_tool(
            NormalizeText(),
            Ports({"rows": _frame(["Acme Pty Ltd"])}),
            _config(["lower"], output=" party key "),
        )

        assert " party key " in result["rows"].columns
