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

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    ExecutionError,
    Ports,
    ToolError,
    WorkedExample,
    check_registry,
    check_tool,
    expression_fields,
    run_tool,
)
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
        assert_frame_equal(rows.select(frame.columns), frame), "every input column survives, in order"

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

        assert set(template) == {"source", "pattern", "position_column"}
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

    def test_it_is_in_the_shipped_catalog_and_the_catalog_still_conforms(self) -> None:
        """Registered under its machine name, and adding it leaves the whole catalog conformant."""
        assert "regex_extract_groups" in REGISTRY
        assert isinstance(REGISTRY.get("regex_extract_groups"), RegexExtractGroups)

        check_registry(REGISTRY)


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
