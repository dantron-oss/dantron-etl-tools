"""Tests for RegexExtractGroups: one pattern, one pass, one decision per row.

Extracting a *single* group out of a piece of text is already ``regex_extract()`` in the expression
language, and a tool that only did that would be a worse spelling of a function that exists. This
tool earns its place on one property an expression cannot have: **a pattern with several named
groups produces several columns from one pass**, and therefore makes one decision per row rather
than one decision per group.

That distinction is the whole point, and it is measurable. ``"Jan 2026 to Smarch 2026"`` under one
two-part pattern yields *both* parts null -- an atomic "this did not parse". Under two separate
single-part patterns it yields a start and a null end: a **partial parse**, which flows onward as a
null month count and has to be tracked down by hand later. So the ruling pinned below is that a row
which does not match the whole pattern is null in **every** group, never in some, and the test that
pins it runs the same row through both shapes so the difference is demonstrated rather than
asserted.

Four decisions the pattern itself forces, each pinned here:

**An unnamed capturing group is refused.** The group names are the output column names, so a
capturing group with no name has no column to go to. The engine underneath happens to name it
``"1"``, which is not a name anyone wrote and not a name a later step can read off the pattern, and
silently dropping it would leave an author's group invisible. There is always a way to say what was
meant -- ``(?:...)`` groups without capturing -- so the refusal costs the author nothing and names
the fix.

**A pattern with no named groups at all is refused.** It would produce no columns, which is either a
step that does nothing or an output port with no schema, and the single-group case has a better
answer already: ``regex_extract()`` in an expression.

**A group named after an existing column is refused.** Writing over a column the frame already
carries would destroy the text the key was derived from -- the one thing a person needs when a match
goes wrong -- and quietly suffixing the name instead would invent a column nobody wrote.

**A pattern is judged by the engine that will run it, before any row is read.** Look-behind is legal
in the standard library's regular expressions and illegal in the engine, so a tool that read its
group names out of Python's ``re`` would accept a pattern that then failed at row time. The refusal
has to be this library's own ``ConfigError`` -- not a wrapped engine exception -- because a caller
holding a file of patterns needs to be told which field to open.

Vocabulary is deliberately generic -- note, key, code, party, period -- because the library carries
no caller's nouns. The period shape is the real one: a service period is read out of a free-text
note here, and what happens next is two date assemblies, a repair and a month count, every one of
which multiplies a partial parse into a wrong number rather than a missing one.
"""

from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import (
    CATEGORIES,
    ConfigError,
    DataError,
    ExecutionError,
    Ports,
    REGISTRY,
    ToolError,
    WorkedExample,
    check_tool,
    run_tool,
)
from dantron_etl.config import expression_fields
from dantron_etl.tools.regex_extract_groups import RegexExtractGroups

# ---------------------------------------------------------------------------
# Test data -- the shape the real conversion runs against
# ---------------------------------------------------------------------------

# A service period written in free text: four parts, one pattern, one decision.
PERIOD = r"(?P<start_month>[A-Za-z]{3}) (?P<start_year>\d{4}) to (?P<end_month>[A-Za-z]{3}) (?P<end_year>\d{4})"

# The same period, read a part at a time -- the shape this tool exists to replace.
PERIOD_START = r"(?P<start_month>[A-Za-z]{3}) (?P<start_year>\d{4}) to"
PERIOD_END = r"to (?P<end_month>[A-Za-z]{3}) (?P<end_year>\d{4})"

CODE = r"(?P<code_prefix>[A-Z]+)-(?P<code_number>\d+)"


def _notes() -> pl.DataFrame:
    """Four notes: one that parses, one that half-looks as if it does, a missing one, and prose."""
    return pl.DataFrame(
        {
            "key": [10, 11, 12, 13],
            "note": [
                "support Jan 2026 to Dec 2026",
                "support Jan 2026 to Smarch 2026",
                None,
                "annual support, no period stated",
            ],
        },
        schema={"key": pl.Int64, "note": pl.String},
    )


def _extract(frame: pl.DataFrame, pattern: str, source: str = "[note]") -> pl.DataFrame:
    """``rows`` after one run, through the boundary, so the envelope is checked on every case."""
    result = run_tool(RegexExtractGroups(), Ports({"rows": frame}), {"source": source, "pattern": pattern})
    return result["rows"]


def _added(example: WorkedExample) -> tuple[str, ...]:
    """The columns a worked example's output carries that its input did not -- its group columns."""
    before = set(example.inputs["rows"].columns)
    return tuple(name for name in example.outputs["rows"].columns if name not in before)


# ---------------------------------------------------------------------------
# Extraction: what the columns are, and what a row that did not match holds
# ---------------------------------------------------------------------------


class TestRegexExtractGroupsExtraction:
    """One pass over the text, one column per named group, and one decision per row."""

    @pytest.mark.parametrize(
        ("pattern", "expected_columns"),
        [
            pytest.param(PERIOD, ("start_month", "start_year", "end_month", "end_year"), id="four_named_groups"),
            pytest.param(CODE, ("code_prefix", "code_number"), id="two_named_groups"),
            pytest.param(r"(?:support )(?P<word>[a-z]+)", ("word",), id="one_named_and_one_non_capturing"),
            pytest.param(r"(?<start_month>[A-Za-z]{3})", ("start_month",), id="the_engines_own_spelling_of_a_name"),
        ],
    )
    def test_the_named_groups_of_the_pattern_are_the_columns_produced(
        self,
        pattern: str,
        expected_columns: tuple[str, ...],
    ) -> None:
        """Column names come off the pattern, in pattern order, beside every column already there.

        Both spellings of a named group the engine accepts work, because the names are read from the
        engine that will run the pattern rather than from a second regular-expression library that
        agrees with it only most of the time.

        Every produced column is text. A group is the piece of text that matched; nothing here
        decides that ``"1042"`` was meant to be a number, and a tool that guessed would make the
        dtype of a column depend on which rows happened to arrive in the batch.
        """
        frame = _notes()

        rows = _extract(frame, pattern)

        assert rows.columns == [*frame.columns, *expected_columns], "input columns first, groups in pattern order"
        assert [rows.schema[name] for name in expected_columns] == [pl.String] * len(expected_columns)
        assert rows.height == frame.height, "extracting columns adds no rows and drops none"
        assert_frame_equal(rows.select(frame.columns), frame)  # every input column survives, in order

    @pytest.mark.parametrize(
        ("note", "matched"),
        [
            pytest.param("support Jan 2026 to Dec 2026", True, id="a_note_that_parses"),
            pytest.param("support Jan 2026 to Smarch 2026", False, id="a_month_nobody_recognises"),
            pytest.param("annual support, no period stated", False, id="no_period_at_all"),
            pytest.param(None, False, id="no_note_at_all"),
            pytest.param("", False, id="an_empty_note"),
            pytest.param("Jan 2026 to Dec", False, id="a_year_that_is_missing_from_the_end"),
        ],
    )
    def test_a_row_that_does_not_match_the_whole_pattern_is_null_in_every_group(
        self,
        note: str | None,
        matched: bool,
    ) -> None:
        """All four groups, or none of them: there is no such thing as half a parse here.

        This is the atomicity ruling. A row with a start and no end would flow onward as a null
        month count, and a null month count multiplied by a rate is a figure nobody can tell apart
        from a figure that was genuinely zero. All-or-nothing keeps the failure visible in one
        place.
        """
        frame = pl.DataFrame({"note": [note]}, schema={"note": pl.String})

        rows = _extract(frame, PERIOD)

        groups = [rows[name][0] for name in ("start_month", "start_year", "end_month", "end_year")]
        if matched:
            assert groups == ["Jan", "2026", "Dec", "2026"]
        else:
            assert groups == [None, None, None, None], f"a note that does not parse must be null throughout: {note!r}"

    def test_one_pattern_makes_one_decision_where_two_patterns_make_two(self) -> None:
        """The measured reason this is a tool: one pass refuses the row, two passes half-accept it.

        Run against the same note, the two-part pattern says "did not parse" and the two one-part
        patterns say "started in January, ended never". The second answer is the expensive one: it
        survives every downstream step as a null rather than as a refusal.
        """
        frame = pl.DataFrame({"note": ["support Jan 2026 to Smarch 2026"]}, schema={"note": pl.String})

        one_pass = _extract(frame, PERIOD)
        start_only = _extract(frame, PERIOD_START)
        end_only = _extract(frame, PERIOD_END)

        assert one_pass.row(0, named=True) == {
            "note": "support Jan 2026 to Smarch 2026",
            "start_month": None,
            "start_year": None,
            "end_month": None,
            "end_year": None,
        }, "one pattern, one decision: nothing parsed"
        assert start_only["start_month"][0] == "Jan", "a separate pattern per part accepts the half it can read"
        assert end_only["end_month"][0] is None
        assert (start_only["start_month"][0], end_only["end_month"][0]) != (
            one_pass["start_month"][0],
            one_pass["end_month"][0],
        ), "the two shapes must genuinely disagree, or this tool has no reason to exist"

    def test_an_optional_group_that_did_not_participate_is_null_while_its_siblings_are_filled(self) -> None:
        """Atomicity is about the *match*, not about the groups: an optional part is still optional.

        A row that matched leaves the groups it matched filled, and a group the author marked
        optional and the text did not carry is null -- which is what the author asked for by writing
        ``?``. Only a row that failed the whole pattern is null throughout.
        """
        frame = pl.DataFrame({"code": ["AC-1042", "AC-1042/rev", "ZZ"]}, schema={"code": pl.String})
        pattern = r"(?P<code_prefix>[A-Z]+)-(?P<code_number>\d+)(?:/(?P<code_tag>\w+))?"

        rows = _extract(frame, pattern, source="[code]")

        assert rows["code_prefix"].to_list() == ["AC", "AC", None]
        assert rows["code_number"].to_list() == ["1042", "1042", None], "the digits arrive as text"
        assert rows["code_tag"].to_list() == [None, "rev", None], "an optional part is null only where absent"

    def test_the_source_may_be_computed_and_the_input_frame_is_left_untouched(self) -> None:
        """``source`` is an expression, so the text matched need not be a column as stored.

        The boundary compares input schemas after a run; the data has to be unchanged too, because
        ``Ports`` copies the mapping and not the frames.
        """
        frame = pl.DataFrame({"key": [1], "note": ["ac-1042"]}, schema={"key": pl.Int64, "note": pl.String})
        before = frame.clone()

        rows = _extract(frame, CODE, source="upper([note])")

        assert rows["code_prefix"].to_list() == ["AC"], "the pattern sees the computed text, not the stored text"
        assert rows["code_number"].to_list() == ["1042"]
        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Refusals: a pattern's own mistakes, named before any row is read
# ---------------------------------------------------------------------------


def _uncompilable_patterns() -> list[Any]:
    """Patterns the engine that will run them rejects -- including one Python's ``re`` accepts."""
    return [
        pytest.param(r"(?P<start>[)", id="an_unclosed_character_class"),
        pytest.param(r"(?P<start>(?<=to )\d{4})", id="a_look_behind_the_engine_does_not_have"),
        pytest.param(r"(?P<start>x)(?P<start>y)", id="the_same_group_name_twice"),
    ]


class TestRegexExtractGroupsRefusals:
    """Every way a pattern can fail to name its own output, refused in this library's own words."""

    @pytest.mark.parametrize(
        "pattern",
        [
            pytest.param(r"(\d{4}) to (?P<end_year>\d{4})", id="one_unnamed_beside_one_named"),
            pytest.param(r"(?P<start_year>\d{4}) to (\d{4})", id="the_unnamed_one_last"),
            pytest.param(r"((?P<start_year>\d{4}))", id="an_unnamed_group_wrapped_around_a_named_one"),
        ],
    )
    def test_a_capturing_group_with_no_name_is_refused_and_the_fix_is_named(self, pattern: str) -> None:
        """A group with no name has no column to go to, and dropping it silently loses an author's work.

        Refused rather than ignored because the alternatives are both worse: the engine's own
        placeholder name for it is a bare digit, which is not something a later step can read off
        the pattern, and ignoring it makes a group the author wrote produce nothing at all. The
        author never needs the capture -- ``(?:...)`` groups without capturing -- so the refusal has
        to say so.
        """
        with pytest.raises(ConfigError) as caught:
            _extract(_notes(), pattern)

        message = str(caught.value)
        assert "pattern" in message, f"the refusal must name the field to open: {message}"
        assert "(?:" in message, f"the refusal must name the fix, not merely the problem: {message}"

    def test_a_pattern_with_no_named_groups_at_all_is_refused_and_points_at_the_expression_function(self) -> None:
        """Nothing to produce, so the refusal points at the thing that does the one-group job.

        Left to run, this would produce a port with no columns of its own -- reported by the
        boundary as a missing schema, which names neither the pattern nor the better tool for the
        job.
        """
        with pytest.raises(ConfigError) as caught:
            _extract(_notes(), r"\d{4} to \d{4}")

        message = str(caught.value)
        assert "pattern" in message
        assert "regex_extract" in message, f"the single-group job already has an answer: {message}"

    def test_a_group_named_after_a_column_already_in_the_frame_is_refused(self) -> None:
        """Writing over the text a key was derived from destroys the only thing a person can read.

        Quietly suffixing the name instead would invent a column nobody wrote, and the next step's
        expression would reference a name that is not in the pattern or the frame.
        """
        with pytest.raises(ConfigError) as caught:
            _extract(_notes(), r"(?P<note>[A-Za-z]{3}) (?P<start_year>\d{4})")

        message = str(caught.value)
        assert "note" in message, f"the refusal must name the column that collided: {message}"

    @pytest.mark.parametrize("pattern", _uncompilable_patterns())
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_pattern_the_engine_cannot_compile_is_refused_in_this_librarys_own_error_family(
        self,
        pattern: str,
        height: int,
    ) -> None:
        """Refused as a ``ConfigError`` quoting the pattern, and refused before a row is touched.

        ``ConfigError`` specifically, and not merely something in the family: an uncaught engine
        exception would arrive as ``ExecutionError``, which is what the boundary raises for a
        failure that may have nothing to do with the tool at all -- an allocation the machine could
        not meet, a frame that could not be read. A bad pattern is a configuration defect with a
        field to open, and it must not be reported as the kind of failure a driver is entitled to
        retry.

        The look-behind case is the one that decides the implementation: Python's ``re`` compiles it
        happily and the engine does not, so the group names have to be read from the engine that
        will run the pattern. Running against a frame with no rows is what shows the judgement is
        made on the pattern rather than on the data.
        """
        frame = _notes().head(height)

        with pytest.raises(ConfigError) as caught:
            _extract(frame, pattern)

        assert not isinstance(caught.value, ExecutionError), "a bad pattern is a config defect, not a stray failure"
        message = str(caught.value)
        assert "pattern" in message, f"the refusal must name the field to open: {message}"
        assert pattern in message, f"the refusal must quote the offending pattern: {message}"

    @pytest.mark.parametrize(
        ("source", "named"),
        [
            pytest.param("[key]", "key", id="a_number_where_text_was_needed"),
            pytest.param("[not_a_column]", "not_a_column", id="a_column_that_is_not_there"),
            pytest.param("upper([key])", "key", id="a_text_function_on_a_number"),
        ],
    )
    def test_a_source_that_cannot_give_text_is_refused_naming_it(self, source: str, named: str) -> None:
        """A pattern needs text. Handed a number, the engine raises about a dtype three layers down.

        The tool holds the frame, so it holds the schema, so the refusal can name the column here
        rather than leaving the author to read an engine's message about a type they never mentioned.
        """
        with pytest.raises(ToolError) as caught:
            _extract(_notes(), PERIOD, source=source)

        message = str(caught.value)
        assert named in message, f"the refusal must name {named!r}: {message}"
        assert "source" in message, f"the refusal must name the field to open: {message}"

    def test_an_unresolved_parameter_placeholder_reaches_the_tool_and_is_refused(self) -> None:
        """A pattern assembled from a rules file must not run with its placeholder still in it.

        Passed through, ``${...}`` is ordinary text: the pattern matches nothing, every row is null
        in every group, and the run looks exactly like a day on which no note carried a period.
        """
        with pytest.raises(ConfigError) as caught:
            _extract(_notes(), r"(?P<start_month>${rules.months})")

        assert "${rules.months}" in str(caught.value)

    def test_the_template_declares_a_source_expression_and_a_pattern_and_nothing_about_the_columns(self) -> None:
        """The output column names are not configuration: they are read off the pattern.

        There is deliberately no list of column names beside the pattern. Two places to write the
        same names is two places to disagree, and the pattern is the one that decides.

        ``position_column`` is not an exception to that. It names a column the pattern cannot name --
        how far into the text the match began, which is about the match rather than about any group --
        and it is optional, so a flow that does not ask for it gets exactly the columns the pattern
        declares. It exists because a flow trying several patterns has to be able to keep whichever
        one the text states first, and "both matched" is not an answer.
        """
        template = RegexExtractGroups().config_template()

        assert set(template) == {"source", "pattern", "position_column", "all_matches", "match_number_column"}
        assert template["position_column"].required is False
        assert template["position_column"].default is None, "absent must mean no such column"
        assert expression_fields(template) == ("source",), "the loader seam is the declared type"
        assert template["source"].required is True
        assert template["pattern"].required is True
        assert template["pattern"].type == "string", "a pattern is text, not an expression over columns"
        assert template["pattern"].non_empty is True, "an empty pattern names no groups and can only be a slip"


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestRegexExtractGroupsInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_it_conforms_and_an_example_shows_several_groups_and_a_row_that_did_not_match(self) -> None:
        """Conformance runs every example, so the atomicity claim is demonstrated, not asserted.

        A single-group example would document the one case this tool should not be used for, and an
        example whose every row parses would leave the behaviour that matters -- all groups null
        together -- entirely undocumented.
        """
        tool = RegexExtractGroups()

        check_tool(tool)

        assert tool.name == "regex_extract_groups"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert len(examples) >= 2
        multi_group = [example for example in examples if len(_added(example)) >= 2]
        assert multi_group, "the reason this is a tool is several groups from one pass; an example must show it"
        atomic = [
            example
            for example in multi_group
            if any(
                all(row[name] is None for name in _added(example)) for row in example.outputs["rows"].rows(named=True)
            )
        ]
        assert atomic, "an example must carry a row that did not match, so 'null in every group' is checked"
        assert any(example.check_row_order for example in examples), (
            "this tool preserves row order, and an example is where that is proved rather than claimed"
        )

    def test_it_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "regex_extract_groups" in REGISTRY
        assert isinstance(REGISTRY.get("regex_extract_groups"), RegexExtractGroups)


class TestWhereThePatternMatched:
    """`position_column` answers the one question the captured groups cannot: how far in.

    A flow reading a period out of free text has several forms it might be written in, and when two
    of them both match, the one the text states *first* is the one meant -- a line names the term it
    is billing for before it names a renewal note or a parenthetical. Without an offset a flow can
    tell that two patterns matched but not which came first, so it has to fall back on the order the
    patterns happen to be listed in, which is a different rule and quietly the wrong one.
    """

    def test_the_offset_is_where_the_match_begins_counted_from_zero(self) -> None:
        frame = pl.DataFrame({"note": ["Jan 2026 first", "billed Jan 2026", "nothing here"]})
        rows = run_tool(
            RegexExtractGroups(),
            Ports({"rows": frame}),
            {"source": "[note]", "pattern": r"(?P<month>Jan) (?P<year>\d{4})", "position_column": "at"},
        )["rows"]

        assert rows["at"].to_list() == [0, 7, None], "and null where the pattern did not match at all"
        assert rows["month"].to_list() == ["Jan", "Jan", None], "the groups are unaffected"

    def test_the_offset_belongs_to_the_same_match_the_groups_came_from(self) -> None:
        """A pattern occurring twice reports the leftmost, which is the one whose groups were taken."""
        frame = pl.DataFrame({"note": ["Mar 2025 renewed from Jan 2024"]})
        rows = run_tool(
            RegexExtractGroups(),
            Ports({"rows": frame}),
            {"source": "[note]", "pattern": r"(?P<month>\w{3}) (?P<year>\d{4})", "position_column": "at"},
        )["rows"]

        assert rows["at"].to_list() == [0]
        assert rows["year"].to_list() == ["2025"], "the leftmost match, groups and offset agreeing"

    def test_two_patterns_can_be_compared_on_which_matched_earliest(self) -> None:
        """The property the field exists for, stated as the flow that needs it.

        One description states a range and, later, a parenthetical naming a single month; the other
        states the single month first. Ordering the patterns cannot tell those apart -- the offsets
        can, and they disagree in opposite directions, which is the whole point.
        """
        frame = pl.DataFrame(
            {
                "note": [
                    "Jan 2026 to Dec 2026 (renews Apr 2027)",
                    "Apr 2027 credit against Jan 2026 to Dec 2026",
                ]
            }
        )
        ranged = run_tool(
            RegexExtractGroups(),
            Ports({"rows": frame}),
            {
                "source": "[note]",
                "pattern": r"(?P<from_month>\w{3}) (?P<from_year>\d{4}) to \w{3} \d{4}",
                "position_column": "range_at",
            },
        )["rows"]
        single = run_tool(
            RegexExtractGroups(),
            Ports({"rows": ranged}),
            {
                "source": "[note]",
                "pattern": r"(?P<one_month>Apr) (?P<one_year>\d{4})",
                "position_column": "single_at",
            },
        )["rows"]

        assert single["range_at"].to_list() == [0, 24]
        assert single["single_at"].to_list() == [29, 0]
        earliest = single.select(
            pl.when(pl.col("range_at") <= pl.col("single_at")).then(pl.lit("range")).otherwise(pl.lit("single"))
        ).to_series()
        assert earliest.to_list() == ["range", "single"], "leftmost wins, and it is a different answer each way"

    def test_no_position_column_is_asked_for_and_none_appears(self) -> None:
        """A flow that does not want it gets exactly the columns the pattern declares."""
        rows = _extract(_notes(), r"(?P<month>\w{3}) (?P<year>\d{4})")

        assert "at" not in rows.columns
        assert rows.columns == ["key", "note", "month", "year"]

    def test_a_position_column_that_would_replace_an_input_is_refused(self) -> None:
        """Held to the same rule as any output column: no quiet overwriting."""
        with pytest.raises(ConfigError) as caught:
            run_tool(
                RegexExtractGroups(),
                Ports({"rows": _notes()}),
                {"source": "[note]", "pattern": r"(?P<month>\w{3})", "position_column": "note"},
            )

        assert "note" in str(caught.value)

    def test_a_position_column_colliding_with_a_group_of_its_own_pattern_is_refused(self) -> None:
        """One of the two would overwrite the other, and which one is not worth having to know."""
        with pytest.raises(ConfigError) as caught:
            run_tool(
                RegexExtractGroups(),
                Ports({"rows": _notes()}),
                {"source": "[note]", "pattern": r"(?P<month>\w{3})", "position_column": "month"},
            )

        assert "month" in str(caught.value)


class TestTheOffsetCountsBytes:
    """Stated because it is surprising, and because the surprise is harmless only for one use.

    The engine reports where a match begins in bytes. On plain ASCII that is the position a person
    would count to; on anything else it is larger. Comparing two offsets against each other -- which
    is the only thing this field exists for -- is unaffected, because both counts rise together. What
    is NOT safe is reading one as "the Nth character", so the difference is pinned here rather than
    discovered by somebody slicing a string with it.
    """

    def test_a_multi_byte_character_before_the_match_widens_the_offset(self) -> None:
        frame = pl.DataFrame({"note": ["  Jan 2026"]})
        rows = run_tool(
            RegexExtractGroups(),
            Ports({"rows": frame}),
            {"source": "[note]", "pattern": r"(?P<month>Jan)", "position_column": "at"},
        )["rows"]

        assert rows["at"].to_list() == [4], "two non-breaking spaces are two characters and four bytes"

    def test_the_order_of_two_matches_survives_it_which_is_all_that_is_read(self) -> None:
        """The property the leftmost-wins rule rests on: more text before means a larger offset."""
        frame = pl.DataFrame({"note": [" café Jan 2026 later Feb 2026"]})
        rows = run_tool(
            RegexExtractGroups(),
            Ports({"rows": frame}),
            {"source": "[note]", "pattern": r"(?P<a>Jan)", "position_column": "jan_at"},
        )["rows"]
        rows = run_tool(
            RegexExtractGroups(),
            Ports({"rows": rows}),
            {"source": "[note]", "pattern": r"(?P<b>Feb)", "position_column": "feb_at"},
        )["rows"]

        jan, feb = rows["jan_at"][0], rows["feb_at"][0]
        assert jan < feb, "whatever the units, the one stated first has the smaller offset"


# ---------------------------------------------------------------------------
# The one output column this tool takes from configuration rather than from the pattern
#
# The group columns are read off the pattern's own named groups, and a pattern has no way to name a
# group nothing -- so for those, a blank name is not a question anybody can ask. ``position_column``
# is different in kind: it is a plain string an author types, it goes straight to the same seam
# every other output-naming tool reaches, and it had no rule of its own at all. Both ``""`` and
# three spaces were accepted here, appending a column nothing downstream could reference.
# ---------------------------------------------------------------------------


class TestRegexExtractGroupsRefusesAPositionColumnThatIsNotAName:
    """``position_column`` is configuration, so it answers to the shared rule like any other."""

    @pytest.mark.parametrize("name", ["", " ", "   ", "\t"])
    def test_a_blank_position_column_is_refused(self, name: str) -> None:
        """Empty and whitespace alike, and the field named is the one the author typed."""
        with pytest.raises(ConfigError) as caught:
            run_tool(
                RegexExtractGroups(),
                Ports({"rows": _notes()}),
                {"source": "[note]", "pattern": r"(?P<month>[A-Z][a-z]{2})", "position_column": name},
            )

        message = str(caught.value)
        assert "'position_column'" in message, f"the refusal must name the field: {message}"
        assert "blank" in message and "whitespace" in message, f"the refusal must say what a name has to be: {message}"

    def test_leaving_the_field_out_is_still_not_the_same_as_blanking_it(self) -> None:
        """Absent means "no such column", which is legitimate; blank means a name that is missing.

        Without this, an author fixing the refusal above could reasonably conclude the field is
        compulsory, and it is not.
        """
        result = run_tool(
            RegexExtractGroups(),
            Ports({"rows": _notes()}),
            {"source": "[note]", "pattern": r"(?P<month>[A-Z][a-z]{2})"},
        )

        assert result["rows"].columns == [*_notes().columns, "month"], "no position column, and no refusal"


# ---------------------------------------------------------------------------
# Every match, one row each: the `all_matches` option
#
# A free-text description can name any number of reference numbers or product words, and the
# number is not bounded, so "the first match" is not the question and "the first N" is an arbitrary
# limit written into a flow. With `all_matches: true`, each match in the text becomes its own row:
# the input row repeated once per match, the named groups of that match, and a column numbering the
# match from 1 in the order the text states them.
#
# Rulings pinned here:
#
# - A row whose text matches nothing is KEPT, once, with every group null and a null match number.
#   No input row vanishes. A null text and an empty text are the same case.
# - Matches are found left to right and do not overlap, and each one is decided whole: a stretch of
#   text that half-matches is not a match and takes no number.
# - The match-number column is required with the option and refused without it. It obeys the same
#   naming rules as every other output column.
# - `position_column` is refused alongside `all_matches`: the match number already says which match
#   the text states first, and an offset for only one of the matches would be a wrong answer.
# - With the option off -- omitted or written as false -- nothing about the tool changes.
# ---------------------------------------------------------------------------

REF = r"(?P<ref_prefix>[A-Z]{3})-(?P<ref_number>\d+)"


def _every_match(
    frame: pl.DataFrame,
    pattern: str,
    *,
    number_column: str = "match_number",
    source: str = "[note]",
    **extra: Any,
) -> pl.DataFrame:
    """``rows`` after one run with every match asked for, through the boundary."""
    config: dict[str, Any] = {
        "source": source,
        "pattern": pattern,
        "all_matches": True,
        "match_number_column": number_column,
        **extra,
    }
    return run_tool(RegexExtractGroups(), Ports({"rows": frame}), config)["rows"]


def _descriptions() -> pl.DataFrame:
    """Free-text lines naming three references, none, one, and a missing line."""
    return pl.DataFrame(
        {
            "key": [1, 2, 3, 4],
            "note": [
                "ABC-12 then XYZ-345, and later QRS-6",
                "no reference here",
                "see DEF-7",
                None,
            ],
        },
        schema={"key": pl.Int64, "note": pl.String},
    )


class TestEveryMatchBecomesARow:
    """With `all_matches: true`, one output row per match, and no input row lost."""

    def test_each_match_is_its_own_row_numbered_in_the_order_the_text_states_them(self) -> None:
        """The input row is repeated once per match; each copy carries that match's groups."""
        rows = _every_match(_descriptions(), REF)

        expected = pl.DataFrame(
            {
                "key": [1, 1, 1, 2, 3, 4],
                "note": [
                    "ABC-12 then XYZ-345, and later QRS-6",
                    "ABC-12 then XYZ-345, and later QRS-6",
                    "ABC-12 then XYZ-345, and later QRS-6",
                    "no reference here",
                    "see DEF-7",
                    None,
                ],
                "ref_prefix": ["ABC", "XYZ", "QRS", None, "DEF", None],
                "ref_number": ["12", "345", "6", None, "7", None],
                "match_number": [1, 2, 3, None, 1, None],
            },
            schema={
                "key": pl.Int64,
                "note": pl.String,
                "ref_prefix": pl.String,
                "ref_number": pl.String,
                "match_number": rows.schema["match_number"],
            },
        )
        assert rows.schema["match_number"].is_integer(), "a match number is a count, not text"
        assert_frame_equal(rows, expected, check_row_order=True)

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("nothing to find", id="text_with_no_match"),
            pytest.param("", id="empty_text"),
            pytest.param(None, id="missing_text"),
        ],
    )
    def test_a_row_with_no_match_is_kept_once_with_every_group_and_the_number_null(self, text: str | None) -> None:
        """No input row vanishes. A row that disappeared would drop out of every total downstream."""
        frame = pl.DataFrame({"key": [7], "note": [text]}, schema={"key": pl.Int64, "note": pl.String})

        rows = _every_match(frame, REF)

        assert rows.height == 1
        assert rows.row(0, named=True) == {
            "key": 7,
            "note": text,
            "ref_prefix": None,
            "ref_number": None,
            "match_number": None,
        }

    def test_no_input_row_is_lost_and_none_is_invented(self) -> None:
        """Rows out = for each input row, its match count, or one if it has none."""
        frame = pl.DataFrame(
            {"note": ["ABC-1", "ABC-1 ABC-2 ABC-3 ABC-4 ABC-5", "none", None, "ABC-1 x ABC-2"]},
            schema={"note": pl.String},
        )

        rows = _every_match(frame, REF)

        assert rows.height == 1 + 5 + 1 + 1 + 2
        assert rows["match_number"].to_list() == [1, 1, 2, 3, 4, 5, None, None, 1, 2]

    def test_each_match_is_decided_whole_and_a_half_match_takes_no_number(self) -> None:
        """`c=x` looks like a pair and is not one, so the pair after it is the third match, not the fourth."""
        frame = pl.DataFrame({"note": ["a=1 b=2 c=x d=4"]})

        rows = _every_match(frame, r"(?P<name>[a-z]+)=(?P<value>\d+)")

        assert rows["name"].to_list() == ["a", "b", "d"]
        assert rows["value"].to_list() == ["1", "2", "4"]
        assert rows["match_number"].to_list() == [1, 2, 3]

    def test_matches_are_found_left_to_right_and_do_not_overlap(self) -> None:
        """`aaaa` holds two `aa`, not three: the search resumes after the end of each match."""
        frame = pl.DataFrame({"note": ["aaaa"]})

        rows = _every_match(frame, r"(?P<pair>aa)")

        assert rows["pair"].to_list() == ["aa", "aa"]
        assert rows["match_number"].to_list() == [1, 2]

    def test_an_optional_group_missing_from_one_match_is_null_there_and_filled_in_the_next(self) -> None:
        """The mirror of the first-match rule: a group that did not take part is null, never ``""``."""
        frame = pl.DataFrame({"note": ["ABC-12 ABC-13/r2"]})

        rows = _every_match(frame, r"(?P<code>[A-Z]{3}-\d+)(?:/(?P<rev>r\d))?")

        assert rows["code"].to_list() == ["ABC-12", "ABC-13"]
        assert rows["rev"].to_list() == [None, "r2"]

    def test_the_number_column_comes_after_the_groups_and_an_empty_frame_keeps_the_shape(self) -> None:
        """Zero rows in, zero rows out, with the same columns and types a full frame would get."""
        empty = pl.DataFrame({"key": [], "note": []}, schema={"key": pl.Int64, "note": pl.String})

        rows = _every_match(empty, REF)

        assert rows.height == 0
        assert rows.columns == ["key", "note", "ref_prefix", "ref_number", "match_number"]
        assert rows.schema["ref_prefix"] == pl.String
        assert rows.schema["ref_number"] == pl.String
        assert rows.schema["match_number"].is_integer()

    def test_the_source_may_be_computed(self) -> None:
        """The source is an expression with the option on, exactly as it is with it off."""
        frame = pl.DataFrame({"left": ["ABC-1 "], "right": ["XYZ-2"]})

        rows = _every_match(frame, REF, source="concat([left], [right])")

        assert rows["ref_prefix"].to_list() == ["ABC", "XYZ"]
        assert rows.columns == ["left", "right", "ref_prefix", "ref_number", "match_number"]


class TestEveryMatchIsReadInItsOwnText:
    """Each match is read where it sits in the text, and the two regex engines must agree on the matches."""

    def test_a_pattern_that_looks_beside_its_match_reads_each_match_in_place(self) -> None:
        """`\\B` asks about the character before the match, so the match cannot be read cut out alone.

        In "box fox x" the x in "box" and in "fox" follow a letter, and the last x follows a space.
        So two matches. Cut out and re-read on its own, each "x" starts the text, `\\B` fails, and
        the group comes back null. Reference-number patterns lean on word boundaries in just this way.
        """
        frame = pl.DataFrame({"note": ["box fox x"]})

        rows = _every_match(frame, r"\B(?P<l>x)")

        assert rows["l"].to_list() == ["x", "x"]
        assert rows["match_number"].to_list() == [1, 2]

    def test_a_text_where_the_engines_find_different_matches_is_refused_not_given_rows(self) -> None:
        """`a*` on "baaa": the engine finds two matches and Python finds three, so no row is trusted.

        The other text, "bbb", is one where the two agree. It is not reported, so the refusal points
        at the one text that needs looking at.
        """
        frame = pl.DataFrame({"key": [1, 2], "note": ["baaa", "bbb"]})

        with pytest.raises(DataError) as caught:
            _every_match(frame, r"(?P<run>a*)")

        assert caught.value.offending_rows == 1
        assert caught.value.total_rows == 2
        assert caught.value.examples == ({"key": 1, "note": "baaa"},)


class TestFirstMatchIsStillTheDefault:
    """The option changes nothing unless it is switched on."""

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param({}, id="option_left_out"),
            pytest.param({"all_matches": False}, id="option_written_as_false"),
        ],
    )
    def test_text_with_several_matches_still_gives_one_row_with_the_first(self, extra: dict[str, Any]) -> None:
        rows = run_tool(
            RegexExtractGroups(),
            Ports({"rows": _descriptions()}),
            {"source": "[note]", "pattern": REF, **extra},
        )["rows"]

        assert rows.columns == ["key", "note", "ref_prefix", "ref_number"], "no match-number column appears"
        assert rows["key"].to_list() == [1, 2, 3, 4], "one row per input row"
        assert rows["ref_prefix"].to_list() == ["ABC", None, "DEF", None], "the first match only"

    def test_position_column_still_works_with_the_option_written_as_false(self) -> None:
        rows = run_tool(
            RegexExtractGroups(),
            Ports({"rows": _descriptions()}),
            {"source": "[note]", "pattern": REF, "all_matches": False, "position_column": "at"},
        )["rows"]

        assert rows["at"].to_list() == [0, None, 4, None]


class TestEveryMatchRefusals:
    """The configuration mistakes the option makes possible, each refused by name before a row is read."""

    def test_the_option_without_a_match_number_column_is_refused(self) -> None:
        """Without the number, two rows from one text cannot be told apart or put back in order."""
        with pytest.raises(ConfigError) as caught:
            run_tool(
                RegexExtractGroups(),
                Ports({"rows": _descriptions()}),
                {"source": "[note]", "pattern": REF, "all_matches": True},
            )

        assert "match_number_column" in str(caught.value)

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param({}, id="option_left_out"),
            pytest.param({"all_matches": False}, id="option_written_as_false"),
        ],
    )
    def test_a_match_number_column_without_the_option_is_refused(self, extra: dict[str, Any]) -> None:
        """A field that would silently do nothing is a mistake the author should hear about."""
        with pytest.raises(ConfigError) as caught:
            run_tool(
                RegexExtractGroups(),
                Ports({"rows": _descriptions()}),
                {"source": "[note]", "pattern": REF, "match_number_column": "n", **extra},
            )

        message = str(caught.value)
        assert "match_number_column" in message and "all_matches" in message
        assert "unknown" not in message, "both fields are known; the refusal is about using one without the other"

    def test_position_column_together_with_the_option_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            _every_match(_descriptions(), REF, position_column="at")

        message = str(caught.value)
        assert "position_column" in message and "all_matches" in message

    @pytest.mark.parametrize(
        ("name", "rows"),
        [
            pytest.param("note", 4, id="an_input_column"),
            pytest.param("ref_number", 4, id="a_group_of_its_own_pattern"),
            pytest.param("note", 0, id="an_input_column_on_an_empty_frame"),
            pytest.param("ref_number", 0, id="a_group_on_an_empty_frame"),
        ],
    )
    def test_a_match_number_column_that_would_overwrite_another_column_is_refused(self, name: str, rows: int) -> None:
        """Refused on the frame's shape, so an empty frame is refused the same way as a full one."""
        frame = _descriptions().head(rows)

        with pytest.raises(ConfigError) as caught:
            _every_match(frame, REF, number_column=name)

        message = str(caught.value)
        assert "match_number_column" in message and name in message

    @pytest.mark.parametrize("name", ["", " ", "\t"])
    def test_a_blank_match_number_column_is_refused(self, name: str) -> None:
        with pytest.raises(ConfigError) as caught:
            _every_match(_descriptions(), REF, number_column=name)

        assert "'match_number_column'" in str(caught.value)

    def test_the_pattern_rules_still_apply_with_the_option_on(self) -> None:
        """An unnamed capturing group has no column to go to, whichever mode reads it."""
        with pytest.raises(ConfigError) as caught:
            _every_match(_descriptions(), r"(?P<ref_prefix>[A-Z]{3})-(\d+)")

        assert "(?:" in str(caught.value)


class TestEveryMatchInTheCatalog:
    """The option is declared in the template and shown in an example that conformance runs."""

    def test_the_template_declares_the_option_off_by_default(self) -> None:
        template = RegexExtractGroups().config_template()

        assert {"all_matches", "match_number_column"} <= set(template)
        assert template["all_matches"].type == "boolean"
        assert template["all_matches"].required is False
        assert template["all_matches"].default is False, "first match stays the default"
        assert template["match_number_column"].type == "string"
        assert template["match_number_column"].required is False
        assert template["match_number_column"].default is None
        assert expression_fields(template) == ("source",), "neither new field is an expression"

    def test_an_example_shows_a_text_with_several_matches_and_a_row_with_none(self) -> None:
        """Conformance executes it, so the row-kept-on-no-match rule is checked in the catalog too."""
        tool = RegexExtractGroups()

        check_tool(tool)

        every = [example for example in tool.help().examples if example.configuration.get("all_matches") is True]
        assert every, "an example must show the option, or the catalog page never mentions it"
        example = every[0]
        number = example.configuration["match_number_column"]
        out = example.outputs["rows"]
        assert out.height > example.inputs["rows"].height, "some text in the example must match more than once"
        assert out[number].null_count() >= 1, "and some row must match nothing and still be there"
        assert example.check_row_order, "the order of matches is part of the claim"


class TestAGroupThePatternMakesOptional:
    """On a match, a group the pattern itself makes optional can still be null.

    The help used to promise that a match fills every group. A group marked optional with '?', or
    one side of an alternative, is left empty on a match that did not need it. A row that matches
    nothing is still null in every group.
    """

    def test_an_optional_group_is_null_on_a_match_that_skipped_it(self) -> None:
        frame = pl.DataFrame({"note": ["12kg", "12", "kg"]})

        result = _extract(frame, r"(?P<num>\d+)(?P<unit>[a-z]+)?")

        assert result.select("num", "unit").rows() == [("12", "kg"), ("12", None), (None, None)]

    def test_the_side_of_an_alternative_that_did_not_match_is_null(self) -> None:
        frame = pl.DataFrame({"note": ["A1", "B2", "C3"]})

        result = _extract(frame, r"(?P<a>A\d)|(?P<b>B\d)")

        assert result.select("a", "b").rows() == [("A1", None), (None, "B2"), (None, None)]

    def test_the_help_states_it(self) -> None:
        text = RegexExtractGroups().help().detailed_description

        assert "optional" in text
        assert "fills every group" not in text
