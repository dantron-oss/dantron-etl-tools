"""Tests for the AssembleDate tool: separate year, month and day parts become one real date.

The source of a date is often not a date. It is text that a regular expression has already been
pulled apart -- a year here, a month there, and frequently no day at all, because a period that
names only a month never stated one. This tool is the one step that turns those parts into a date
column, and everything downstream that compares, sorts or counts between dates depends on it having
built the date the parts named rather than a date near it.

Five decisions are pinned here, each because it has a silent failure mode:

**The day defaults to the first of the month.** It is the only day every month has, so it is the one
default that does not quietly depend on the month's length. The other end of a whole month is
written down instead -- ``day: '31'`` with ``on_invalid: clamp`` -- which is a decision a reviewer
can see in the file, and which resolves to the 28th, the 29th, the 30th or the 31st per row.

**``on_invalid`` is about a day the calendar is too short for, and about nothing else.** ``clamp``
pulls a day into the month it names; a month outside 1-12, a year no calendar date could hold, and a
part that is not written in digits are refused under ``clamp`` exactly as under ``error``, because
none of them is a day the calendar was short of and there is no nearest real value that keeps the
period the parts stated. Clamping a month of 13 to December would book a term into a month nobody
wrote.

**A missing part is not an invalid one.** A null in any part gives a null date under every policy,
including ``error``. The two are different questions -- absent data against impossible data -- and
conflating them would either fail a whole run on rows nobody filled in, or hide an impossible date
among the merely missing.

**The policy that gives a null date is spelled ``empty``, never ``null``.** A configuration is a
YAML file, and an unquoted ``null`` there is the null value itself, so ``on_invalid: null`` would
arrive as no value at all -- reaching for the policy by its obvious name is exactly how an author
would land on it. That same shape is why a field written with no value is refused rather than
silently defaulted.

**The output is a real date, and it is added.** Not text, which sorts lexically and compares by
character, and not over the top of an existing column, which would substitute data with nothing in a
configuration diff to show it.

Vocabulary is deliberately generic -- rows, key, parts, period start and end -- because the library
carries no caller's nouns.
"""

from collections.abc import Mapping, Sequence
from datetime import date
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
from dantron_etl.tools.assemble_date import AssembleDate

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _parts() -> pl.DataFrame:
    """One frame carrying the part shapes real sources arrive in: whole numbers and digit text."""
    return pl.DataFrame(
        {
            "key": [4, 1, 3, 2],
            "year_part": [2026, 2024, 2026, 2025],
            "month_part": [1, 2, 12, 6],
            "day_part": [31, 29, 25, 1],
            "year_text": ["2026", "2024", "2026", "2025"],
            "month_text": ["01", "2", "12", "06"],
        }
    )


def _one_row(year: Any, month: Any, day: Any) -> pl.DataFrame:
    """A single row of parts, as whatever type each part is given as."""
    return pl.DataFrame({"key": [1], "year_part": [year], "month_part": [month], "day_part": [day]})


def _config(**overrides: Any) -> dict[str, Any]:
    """The plain configuration -- every part read from its own column -- with fields overridden."""
    base: dict[str, Any] = {
        "year": "[year_part]",
        "month": "[month_part]",
        "day": "[day_part]",
        "output": "period_start",
    }
    base.update(overrides)
    return base


def _built(frame: pl.DataFrame, config: Mapping[str, Any]) -> pl.Series:
    """The date column AssembleDate produces, through the boundary."""
    result = run_tool(AssembleDate(), Ports({"rows": frame}), config)
    return result["rows"][config["output"]]


def _assembly_cases() -> list[Any]:
    """(config, the dates it must produce) for each way a set of parts can be named."""
    return [
        pytest.param(
            _config(),
            [date(2026, 1, 31), date(2024, 2, 29), date(2026, 12, 25), date(2025, 6, 1)],
            id="each_part_is_a_column_of_whole_numbers",
        ),
        pytest.param(
            _config(year="[year_text]", month="[month_text]"),
            [date(2026, 1, 31), date(2024, 2, 29), date(2026, 12, 25), date(2025, 6, 1)],
            id="a_part_written_in_digits_reads_as_the_number_it_spells",
        ),
        pytest.param(
            {"year": "[year_part]", "month": "[month_part]", "output": "period_start"},
            [date(2026, 1, 1), date(2024, 2, 1), date(2026, 12, 1), date(2025, 6, 1)],
            id="no_day_configured_gives_the_first_of_the_month",
        ),
        pytest.param(
            _config(year="2026", month="3", day="5"),
            [date(2026, 3, 5)] * 4,
            id="every_part_a_fixed_value",
        ),
        pytest.param(
            _config(day="coalesce([day_part], 1)"),
            [date(2026, 1, 31), date(2024, 2, 29), date(2026, 12, 25), date(2025, 6, 1)],
            id="a_computed_part_is_an_expression_like_any_other",
        ),
    ]


# ---------------------------------------------------------------------------
# Building a date
# ---------------------------------------------------------------------------


class TestAssembleDateBuildsADate:
    """What the parts name becomes a date column, and nothing else about the frame moves."""

    @pytest.mark.parametrize(("config", "expected"), _assembly_cases())
    def test_parts_become_a_real_date_column_appended_to_the_input(
        self,
        config: Mapping[str, Any],
        expected: Sequence[date | None],
    ) -> None:
        """A date, not text: comparing the whole frame pins the dtype, the position and the order.

        The output has to be a real date, because a date held as text sorts lexically and compares
        character by character, so ``'2026-1-9' > '2026-12-01'`` -- and the mistake surfaces in a
        report, not here. Building the expected frame from the input with one column added pins at
        once that every input column survives, that the new column is the only addition, that it
        arrives last, and that no row moved.
        """
        frame = _parts()
        wanted = frame.with_columns(pl.Series("period_start", list(expected), dtype=pl.Date))

        result = run_tool(AssembleDate(), Ports({"rows": frame}), config)

        assert_frame_equal(result["rows"], wanted)
        assert result["rows"]["period_start"].dtype == pl.Date, "a real date, not text and not a timestamp"
        assert result["rows"].height == frame.height, "a row is never added or dropped here"

    def test_the_last_day_of_a_month_is_written_as_a_day_of_31_and_clamped_per_row(self) -> None:
        """The idiom that replaces a month-length table in the configuration.

        ``day: '31'`` is one fixed value for every row, and the day it resolves to differs by row --
        28, 29, 30 or 31. That is the case a broadcast mistake gets wrong while looking right: one
        month's length applied to every row, or a single-row result standing in for the frame.
        """
        frame = pl.DataFrame(
            {
                "key": [1, 2, 3, 4, 5],
                "year_part": [2026, 2024, 2026, 2026, 2025],
                "month_part": [2, 2, 4, 1, 9],
            }
        )

        built = _built(frame, {**_config(day="31", on_invalid="clamp"), "output": "period_end"})

        assert built.to_list() == [
            date(2026, 2, 28),
            date(2024, 2, 29),
            date(2026, 4, 30),
            date(2026, 1, 31),
            date(2025, 9, 30),
        ], "each row's last day, not the first row's applied to all of them"

    @pytest.mark.parametrize(
        ("year", "month", "day", "expected"),
        [
            pytest.param(2026, 2, 31, date(2026, 2, 28), id="the_31st_of_february_in_an_ordinary_year"),
            pytest.param(2024, 2, 31, date(2024, 2, 29), id="the_31st_of_february_in_a_leap_year"),
            pytest.param(2026, 4, 31, date(2026, 4, 30), id="the_31st_of_a_thirty_day_month"),
            pytest.param(2026, 2, 29, date(2026, 2, 28), id="the_29th_of_february_in_an_ordinary_year"),
            pytest.param(2024, 2, 29, date(2024, 2, 29), id="the_29th_of_february_in_a_leap_year_is_untouched"),
            pytest.param(2026, 12, 32, date(2026, 12, 31), id="a_day_past_the_end_of_a_thirty_one_day_month"),
            pytest.param(2026, 3, 0, date(2026, 3, 1), id="a_day_of_zero_is_pulled_up_to_the_first"),
            pytest.param(2026, 3, 15, date(2026, 3, 15), id="a_day_the_month_has_is_never_moved"),
        ],
    )
    def test_an_impossible_day_is_clamped_into_the_month_it_names(
        self,
        year: int,
        month: int,
        day: int,
        expected: date,
    ) -> None:
        """Pinned by value, because a test that only checked "no error" would pass a wrong date.

        The leap-year pair is the reason the value matters: the same parts, one year apart, must
        clamp to two different days, and the 29th of February in a leap year is a real date that
        must not be touched at all. Clamping is two-sided -- a day below the first is pulled up to
        it -- because a day is clamped into the month that was named, and the month is the part that
        was stated with confidence.
        """
        built = _built(_one_row(year, month, day), _config(on_invalid="clamp"))

        assert built.to_list() == [expected]

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            pytest.param("empty", [None], id="empty_gives_a_null_date_for_that_row"),
            pytest.param("error", None, id="error_refuses_the_run"),
        ],
    )
    def test_an_impossible_day_is_not_clamped_under_the_other_two_policies(
        self,
        policy: str,
        expected: list[date | None] | None,
    ) -> None:
        """The 31st of February is a data defect unless the author said it was an idiom.

        ``clamp`` is how an author says "the end of that month"; without it, the same parts are an
        extraction that went wrong, and the two remaining policies are the two honest answers --
        carry the row with no date, or stop. ``error`` is the default precisely because the other
        two both change what the data says, and a change like that belongs in the file where a
        reviewer sees it.
        """
        frame = _one_row(2026, 2, 31)

        if expected is None:
            with pytest.raises(ToolError) as caught:
                _built(frame, _config(on_invalid=policy))
            message = str(caught.value)
            assert "31" in message, f"the refusal must quote the offending day: {message}"
            assert "2026" in message, f"the refusal must quote the parts it was given: {message}"
        else:
            assert _built(frame, _config(on_invalid=policy)).to_list() == expected

    @pytest.mark.parametrize(
        ("year", "month", "named"),
        [
            pytest.param(2026, 13, "13", id="a_thirteenth_month"),
            pytest.param(2026, 0, "0", id="a_month_of_zero"),
            pytest.param(0, 3, "0", id="a_year_no_calendar_date_can_hold"),
            pytest.param(10000, 3, "10000", id="a_year_past_the_end_of_the_calendar"),
        ],
    )
    def test_a_month_or_year_that_names_nothing_is_refused_even_under_clamp(
        self,
        year: int,
        month: int,
        named: str,
    ) -> None:
        """``clamp`` clamps a day, and only a day. Every other impossibility is refused.

        There is no defensible nearest month for a 13: December keeps the year and loses the month,
        January of the following year keeps neither. Both would book a period into a month nobody
        stated, silently, and the parts came from an extraction that has plainly gone wrong -- which
        is worth a refusal naming the value, so somebody can find the row. ``empty`` is the one
        policy that carries such a row, and it carries it with no date rather than with a guess.
        """
        frame = _one_row(year, month, 1)

        for policy in ("clamp", "error"):
            with pytest.raises(ToolError) as caught:
                _built(frame, _config(on_invalid=policy))
            assert named in str(caught.value), f"the refusal must quote {named!r}: {caught.value}"

        assert _built(frame, _config(on_invalid="empty")).to_list() == [None]

    @pytest.mark.parametrize("policy", ["clamp", "empty", "error"])
    def test_a_missing_part_is_a_null_date_under_every_policy_including_error(self, policy: str) -> None:
        """A part nobody supplied is missing data, which is a different question from an impossible date.

        ``on_invalid`` decides what happens to parts that name no real date. A null part names
        nothing at all, and it is the ordinary state of a source where the day was never written
        down -- so ``error`` must not stop a run over it, and ``empty`` must not be the only way to
        survive it. Conflating the two would also work the other way round: an impossible date would
        arrive as a null indistinguishable from the rows nobody filled in.
        """
        frame = pl.DataFrame(
            {
                "key": [1, 2, 3, 4],
                "year_part": [2026, None, 2026, 2026],
                "month_part": [None, 3, 3, 3],
                "day_part": [1, 1, None, 4],
            },
            schema={"key": pl.Int64, "year_part": pl.Int64, "month_part": pl.Int64, "day_part": pl.Int64},
        )

        built = _built(frame, _config(on_invalid=policy))

        assert built.to_list() == [None, None, None, date(2026, 3, 4)], "a null part, and only it, gives a null date"

    @pytest.mark.parametrize(
        "written",
        [
            pytest.param("Jan", id="a_month_name"),
            pytest.param("1.0", id="a_month_that_arrived_as_a_decimal"),
            pytest.param(" 2", id="a_month_padded_with_a_space"),
            pytest.param("+5", id="a_month_carrying_a_sign_the_query_engine_would_read"),
            pytest.param("", id="a_part_that_is_blank_rather_than_missing"),
        ],
    )
    def test_a_month_name_is_not_read_here_and_text_must_be_digits_alone(self, written: str) -> None:
        """A month is a number. Turning a name into one belongs to a step that can be read and edited.

        The real source says "Jan", and it is tempting to teach this tool the twelve names -- but
        the names are the part that varies: spellings ("Sept"), punctuation ("Sep."), and language.
        A table of accepted spellings is policy, and policy belongs in an ordered rule list or a
        lookup a reviewer can see and change, not baked invisibly into a tool that would then own
        one job and a half. So text is read only where it is written in digits alone, and everything
        else about it is refused with the value quoted -- no trimming, no decimal point, no name.
        """
        frame = pl.DataFrame({"key": [1, 2], "year_part": ["2026", "2026"], "month_part": ["3", written]})
        config = _config(year="[year_part]", month="[month_part]", day="1")

        with pytest.raises(ToolError) as caught:
            _built(frame, config)
        if written:
            assert written in str(caught.value), f"the refusal must quote the value it could not read: {caught.value}"

        assert _built(frame, {**config, "on_invalid": "empty"}).to_list() == [date(2026, 3, 1), None]

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("as_a_decimal", [2026.0], id="a_part_stored_as_a_decimal"),
            pytest.param("as_a_truth_value", [True], id="a_part_stored_as_a_truth_value"),
            pytest.param("as_a_date", [date(2026, 3, 1)], id="a_part_that_is_already_a_date"),
        ],
    )
    def test_a_part_that_is_not_a_whole_number_is_refused_before_any_row_is_read(
        self,
        column: str,
        values: Sequence[Any],
    ) -> None:
        """Read off the schema, so a frame with no rows at all is refused the same way.

        A decimal is the case worth the refusal: rounding a fractional part is a decision, and its
        two directions differ by a day. Refusing is also the reversible direction -- a dtype
        accepted later costs nobody anything, while one accepted now can never be withdrawn.
        """
        frame = pl.DataFrame({"key": [1], "month_part": [3], "day_part": [1], column: list(values)}).head(0)

        with pytest.raises(ConfigError) as caught:
            _built(frame, _config(year=f"[{column}]"))

        message = str(caught.value)
        assert "year" in message, f"the refusal must name the field to open: {message}"
        assert column in message, f"the refusal must quote the expression: {message}"


# ---------------------------------------------------------------------------
# Refusals a template cannot express
# ---------------------------------------------------------------------------


class TestAssembleDateRefusals:
    """What is refused before a row is read, and why each refusal is worth a rule."""

    def test_the_output_column_may_not_replace_an_existing_column(self) -> None:
        """Writing over a column substitutes data with nothing in the configuration to show it.

        A later step reading that column by name gets a different value than the one its author
        looked at, and the flow shows no sign of it. Naming a new column is free; dropping or
        renaming the old one is a step that exists for the purpose.
        """
        frame = _parts()

        with pytest.raises(ConfigError) as caught:
            _built(frame, _config(output="month_part"))

        assert "month_part" in str(caught.value)

    @pytest.mark.parametrize(
        ("config", "named"),
        [
            pytest.param(_config(on_invalid=None), "on_invalid", id="a_policy_written_with_no_value"),
            pytest.param(_config(day=None), "day", id="a_day_written_with_no_value"),
            pytest.param(_config(output=" "), "output", id="an_output_name_that_is_only_whitespace"),
        ],
    )
    def test_a_field_written_with_no_value_is_refused_rather_than_silently_defaulted(
        self,
        config: Mapping[str, Any],
        named: str,
    ) -> None:
        """``on_invalid:`` with nothing after it is how an author reaches for the null policy.

        A configuration is a YAML file, so ``on_invalid: null`` arrives here as no value at all --
        which is why the policy that gives a null date is spelled ``empty``, and why a field written
        with no value has to be refused instead of quietly taking its default. Defaulting it would
        turn an author asking for null dates into a run that refuses every impossible date instead,
        or the reverse, with nothing said either way.
        """
        with pytest.raises(ConfigError) as caught:
            _built(_parts(), config)

        assert named in str(caught.value)

    @pytest.mark.parametrize("spelling", ["null", "none", "blank", "raise", "clip"])
    def test_only_the_three_published_policies_are_accepted(self, spelling: str) -> None:
        """A near-miss spelling is refused with the three real ones listed.

        ``null`` is the one that matters: it is the obvious name for the policy, it is the name this
        tool deliberately does not use, and an author who tries it in quotes must be told what to
        write instead rather than being told nothing and getting the default.
        """
        with pytest.raises(ConfigError) as caught:
            _built(_parts(), _config(on_invalid=spelling))

        message = str(caught.value)
        assert "empty" in message, f"the refusal must list the policy the author wanted: {message}"
        assert "clamp" in message and "error" in message, f"and the other two: {message}"

    def test_the_template_declares_three_part_expressions_and_a_day_that_defaults_to_the_first(self) -> None:
        """Each part is an expression, so a part can be a column, a fixed value or computed.

        The default day is the first of the month because it is the only day every month has: a
        default of "the last day" would be a different day per row, which is a decision, and a
        decision belongs in the file (``day: '31'`` with ``on_invalid: clamp``) rather than in a
        default nobody wrote.
        """
        template = AssembleDate().config_template()

        assert set(template) == {"year", "month", "day", "output", "on_invalid"}
        assert expression_fields(template) == ("year", "month", "day"), "the loader seam is the declared type"
        assert template["year"].required is True
        assert template["month"].required is True
        assert template["output"].required is True
        assert template["day"].required is False
        assert template["day"].default == "1", "the one day every month has"
        assert template["on_invalid"].default == "error", "the two other policies both change what the data says"
        assert set(template["on_invalid"].enum or ()) == {"clamp", "empty", "error"}
        assert "null" not in (template["on_invalid"].enum or ()), "an unquoted null in YAML is the null value"

    def test_assemble_date_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the data must be unchanged too."""
        frame = _parts()
        before = frame.clone()

        run_tool(AssembleDate(), Ports({"rows": frame}), _config(on_invalid="clamp"))

        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestAssembleDateInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_assemble_date_conforms_and_its_examples_document_the_two_surprises(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output."""
        tool = AssembleDate()

        check_tool(tool)

        assert tool.name == "assemble_date"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert len(examples) >= 2, "one plain example, and one showing what a policy does"
        assert any(example.configuration.get("on_invalid") == "clamp" for example in examples), (
            "the last-day-of-the-month idiom is the behaviour a reader cannot guess, so it must be shown"
        )
        assert any("day" not in example.configuration for example in examples), (
            "and the default day, because a period that names only a month is the motivating case"
        )

    def test_assemble_date_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in test_dantron_etl_registry.py."""
        assert "assemble_date" in REGISTRY
        assert isinstance(REGISTRY.get("assemble_date"), AssembleDate)
