"""Tests for the DateDiff tool: how many whole units lie between two dates.

This count is a divisor. A figure billed for a period is divided by the number of units that period
spans to get a figure per unit, so a count that is one too many understates every unit of that
period, a count that is one too few overstates it, and a count of zero divides by nothing at all.
None of those three announce themselves: the figure still looks like a figure.

So the arithmetic is pinned by value, and four decisions are pinned with it:

**Month counting is day-aware.** A month is counted once the end's day of the month has reached the
start's, so January 31 to February 28 is one month and not two, while January 1 to December 31 is a
full twelve. The rule is one sentence, and it has to be in ``help()``, because a reader of a
configuration cannot guess whether a partial month counts and cannot see the answer in the number.

**``counting`` is required.** ``inclusive`` counts the unit each end falls in (January to December
is 12), ``elapsed`` counts only completed units (11), like SQL's DATEDIFF. The two differ by one, and
the count is a divisor, so the tool takes no default: a flow says which reading it means.

**A backwards range counts negative and keeps its size.** A range whose end falls before its start
is a defect in the source, not in the arithmetic, and repairing it is a decision (which end was
mistyped?) that belongs to a step of its own where it can be reviewed. Reported as a negative count
it is visible, filterable in one step, and still carries its magnitude. Refusing the run would stop
a whole flow over a handful of rows; a null would hide them among the rows that were merely missing
a date.

**A null is a null, never a zero.** A missing date on either side gives a null count, which carries
through the division as a null. A zero would divide a figure by nothing at all, and under ``inclusive``
zero is not otherwise reachable at all -- which is what makes a negative count mean exactly one
thing.

Vocabulary is deliberately generic -- rows, from_date, to_date, counts -- because the library
carries no caller's nouns.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import CATEGORIES, ConfigError, Ports, REGISTRY, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.date_diff import DateDiff

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _ranges(pairs: Sequence[tuple[date | datetime | None, date | datetime | None]]) -> pl.DataFrame:
    """One row per (start, end) pair, keyed in a shuffled order so row order is pinned too."""
    return pl.DataFrame(
        {
            "key": list(range(len(pairs), 0, -1)),
            "from_date": [pair[0] for pair in pairs],
            "to_date": [pair[1] for pair in pairs],
        }
    )


def _config(**overrides: Any) -> dict[str, Any]:
    """The plain configuration -- both ends read from their own column -- with fields overridden."""
    base: dict[str, Any] = {
        "start": "[from_date]",
        "end": "[to_date]",
        "unit": "month",
        "counting": "inclusive",
        "output": "counted",
    }
    base.update(overrides)
    return base


def _counted(pairs: Sequence[tuple[Any, Any]], **overrides: Any) -> list[int | None]:
    """The counts DateDiff produces for ``pairs``, through the boundary."""
    config = _config(**overrides)
    result = run_tool(DateDiff(), Ports({"rows": _ranges(pairs)}), config)
    return result["rows"][config["output"]].to_list()


def _month_cases() -> list[Any]:
    """(start, end, months counting both ends) -- the values the whole rebuild ties out against."""
    return [
        pytest.param(date(2026, 1, 1), date(2026, 12, 31), 12, id="a_full_year_stated_first_to_last_day"),
        pytest.param(date(2026, 1, 15), date(2026, 12, 14), 11, id="january_to_december_one_day_short"),
        pytest.param(date(2026, 1, 31), date(2026, 2, 28), 1, id="the_31st_to_the_28th_of_the_next_month"),
        pytest.param(date(2026, 1, 1), date(2026, 1, 31), 1, id="a_range_inside_one_month"),
        pytest.param(date(2026, 1, 1), date(2026, 1, 1), 1, id="the_same_day_at_both_ends"),
        pytest.param(date(2025, 4, 1), date(2026, 3, 31), 12, id="a_year_that_straddles_two_calendar_years"),
        pytest.param(date(2025, 4, 15), date(2026, 4, 14), 12, id="an_anniversary_year_is_12_and_not_13"),
        pytest.param(date(2025, 4, 15), date(2026, 4, 15), 13, id="one_day_past_the_anniversary_is_13"),
        pytest.param(date(2026, 1, 31), date(2026, 3, 1), 2, id="the_31st_to_the_1st_two_months_later"),
    ]


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


class TestDateDiffCounting:
    """The arithmetic, pinned by value in both units and under both readings of the ends."""

    @pytest.mark.parametrize(("start", "end", "expected"), _month_cases())
    def test_a_month_counts_once_the_end_has_reached_the_start_s_day(
        self,
        start: date,
        end: date,
        expected: int,
    ) -> None:
        """Day-aware month counting, by value, because every one of these is a divisor.

        Two pairs carry the whole rule between them. January 1 to December 31 is 12: a period
        stated as a full year must not divide as eleven months, or every month of it is
        overstated. April 15 to April 14 the following year is also 12, not 13: the last month is
        one day short of completing, so counting it would spread the figure one month thinner than
        it was billed. Move the end one day later and it is genuinely 13.
        """
        assert _counted([(start, end)]) == [expected]

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            pytest.param(date(2026, 1, 1), date(2026, 1, 1), 1, id="the_same_day_at_both_ends_is_one_day"),
            pytest.param(date(2026, 1, 1), date(2026, 1, 2), 2, id="two_consecutive_days_are_two_days"),
            pytest.param(date(2026, 1, 1), date(2026, 12, 31), 365, id="an_ordinary_year"),
            pytest.param(date(2024, 1, 1), date(2024, 12, 31), 366, id="a_leap_year_has_the_extra_day"),
            pytest.param(date(2024, 2, 28), date(2024, 3, 1), 3, id="across_a_leap_day"),
        ],
    )
    def test_days_count_the_day_at_each_end_the_same_way_months_do(
        self,
        start: date,
        end: date,
        expected: int,
    ) -> None:
        """Days is shipped because nothing else in the library can subtract two dates.

        The expression language has no arithmetic on dates and no cast, so without this unit a flow
        that needs the length of a range in days has no way to get it at all. The two units read the
        ends the same way, which is the only thing that makes one ``counting`` field honest across
        both: the same day at both ends is one day, exactly as it is one month.
        """
        assert _counted([(start, end)], unit="day") == [expected]

    @pytest.mark.parametrize("unit", ["month", "day"])
    def test_elapsed_is_exactly_one_less_and_is_zero_inside_a_single_unit(self, unit: str) -> None:
        """The other reading of the ends, and the only way this tool produces a zero.

        ``elapsed`` counts what has been completed, so a range that has not completed one unit
        counts none -- which is a correct answer and a hazard, because a figure divided by it
        divides by nothing. Under ``inclusive`` a forward count is at least one, so the divisor is
        always usable.
        """
        pairs = [
            (date(2026, 1, 1), date(2026, 1, 1)),
            (date(2026, 1, 1), date(2026, 1, 31)),
            (date(2026, 1, 1), date(2026, 12, 31)),
            (date(2025, 4, 15), date(2026, 4, 14)),
        ]

        both = _counted(pairs, unit=unit)
        elapsed = _counted(pairs, unit=unit, counting="elapsed")

        assert elapsed == [count - 1 for count in both if count is not None]
        assert elapsed[0] == 0, "the same day at both ends has completed no unit at all"
        assert both[0] == 1, "while counting both ends it is one, which is what makes it safe to divide by"

    @pytest.mark.parametrize(
        ("start", "end", "unit", "expected"),
        [
            pytest.param(date(2026, 1, 1), date(2025, 12, 31), "month", -1, id="a_year_typo_at_the_end"),
            pytest.param(date(2026, 7, 1), date(2025, 9, 30), "month", -10, id="a_quarter_with_the_wrong_end_year"),
            pytest.param(date(2026, 2, 1), date(2026, 1, 31), "month", -1, id="one_day_backwards_across_a_month"),
            pytest.param(date(2026, 1, 2), date(2026, 1, 1), "day", -2, id="one_day_backwards"),
        ],
    )
    def test_a_backwards_range_counts_negative_and_keeps_its_size(
        self,
        start: date,
        end: date,
        unit: str,
        expected: int,
    ) -> None:
        """Reported faithfully, not repaired here, and never confused with a valid count.

        A range that runs backwards is a real thing a source produces -- a mistyped year at one end
        -- and which end was mistyped is a judgement, so repairing it belongs to a step that can be
        read and argued with. What this tool owes such a row is a count that says so: negative, so
        one filter finds every one of them; sized, so the repair step can see how far out it is; and
        never zero under the default, so a negative value means exactly one thing.
        """
        assert _counted([(start, end)], unit=unit) == [expected]
        assert _counted([(end, start)], unit=unit) == [-expected], "the same size, the other way round"

    def test_under_elapsed_a_short_backwards_range_cannot_be_told_from_a_forward_one(self) -> None:
        """The cost of the other reading, pinned so nobody discovers it in a report.

        ``elapsed`` is one less than ``inclusive``, in both directions, so a backwards range that has
        not completed a unit counts zero -- indistinguishable from a forward range that has not
        either. Under ``inclusive`` the same two rows are -1 and 1. This is the second reason ``inclusive`` is
        the default, and the reason a flow that filters on a negative count must not be run under
        ``elapsed``.
        """
        backwards = (date(2026, 2, 1), date(2026, 1, 31))
        forwards = (date(2026, 1, 31), date(2026, 2, 1))

        assert _counted([backwards, forwards], counting="elapsed") == [0, 0]
        assert _counted([backwards, forwards]) == [-1, 1], "counting both ends keeps the two apart"

    @pytest.mark.parametrize("counting", ["inclusive", "elapsed"])
    @pytest.mark.parametrize("unit", ["month", "day"])
    def test_a_missing_date_on_either_side_counts_null_and_never_zero(self, unit: str, counting: str) -> None:
        """A null divisor makes a null figure; a zero divisor makes a division by nothing.

        This is the case a conditional written the obvious way gets wrong: a comparison against a
        missing date is null, not false, and a branch that treats it as false sends the row down the
        arm meant for a backwards range. The count for a row with no date must be null -- and a
        neighbouring row with both dates must be unaffected.
        """
        pairs = [
            (None, date(2026, 1, 1)),
            (date(2026, 1, 1), None),
            (None, None),
            (date(2026, 1, 1), date(2026, 3, 1)),
        ]

        counts = _counted(pairs, unit=unit, counting=counting)

        assert counts[:3] == [None, None, None], "a missing date on either side counts nothing at all"
        assert counts[3] is not None, "and the row beside it still counts"

    def test_a_timestamp_counts_by_the_calendar_date_it_falls_on(self) -> None:
        """A date that arrived as a timestamp must still be countable.

        The expression language has no cast, so refusing a timestamp here would leave a caller
        whose source carries the time of day with no way to use this tool at all. The time is
        dropped rather than rounded: two timestamps on the same pair of dates count the same,
        whichever side of the clock they fall on. The days case is the one that pins it -- a
        subtraction that keeps the clock loses a day whenever the start is later in the day than
        the end, and the row below is exactly that shape.
        """
        pairs = [
            (datetime(2026, 1, 1, 23, 59), datetime(2026, 12, 31, 0, 1)),
            (datetime(2026, 1, 1, 0, 1), datetime(2026, 12, 31, 23, 59)),
        ]

        assert _counted(pairs) == [12, 12]
        assert _counted(pairs, unit="day") == [365, 365]

    def test_the_count_is_a_signed_whole_number_appended_to_an_untouched_input(self) -> None:
        """A signed integer, because the sign carries the backwards case, and only one column is added.

        Comparing the whole frame pins that every input column survives, that the count arrives
        last, that no row moved, and that the count is a whole number rather than a decimal -- a
        decimal count would divide a figure by 11.983870967741936 and look plausible doing it.
        """
        pairs = [(date(2026, 1, 1), date(2026, 12, 31)), (date(2026, 3, 1), date(2026, 3, 1))]
        frame = _ranges(pairs)
        before = frame.clone()

        result = run_tool(DateDiff(), Ports({"rows": frame}), _config())
        counted = result["rows"]["counted"]

        assert counted.dtype.is_integer(), f"a count of whole units, not a decimal: {counted.dtype}"
        assert counted.dtype.is_signed_integer(), "a backwards range counts below zero"
        assert_frame_equal(result["rows"], frame.with_columns(counted.alias("counted")))
        assert result["rows"].columns == [*frame.columns, "counted"]
        assert_frame_equal(frame, before)  # the input frame is read-only


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestDateDiffRefusals:
    """What is refused, and when: off the schema, before a single row is read."""

    @pytest.mark.parametrize(
        ("column", "values"),
        [
            pytest.param("as_text", ["2026-01-01"], id="a_date_still_held_as_text"),
            pytest.param("as_a_number", [20260101], id="a_date_held_as_a_number"),
            pytest.param("as_a_truth_value", [True], id="not_a_date_at_all"),
        ],
    )
    def test_a_side_that_is_not_a_date_is_refused_before_any_row_is_read(
        self,
        column: str,
        values: Sequence[Any],
    ) -> None:
        """Text is the case that matters: it compares, so it would silently count the wrong thing.

        A date held as text sorts and compares character by character, so the arithmetic here would
        either fail inside the query engine with a message about a dtype or, worse, run on a
        comparison that is lexical rather than chronological. Read off the schema, so a frame with
        no rows at all is refused the same way, and the refusal names the field to open.
        """
        frame = pl.DataFrame({"key": [1], "to_date": [date(2026, 1, 1)], column: list(values)}).head(0)

        with pytest.raises(ConfigError) as caught:
            run_tool(DateDiff(), Ports({"rows": frame}), _config(start=f"[{column}]"))

        message = str(caught.value)
        assert "start" in message, f"the refusal must name the field to open: {message}"
        assert column in message, f"the refusal must quote the expression: {message}"

    def test_the_output_column_may_not_replace_an_existing_column(self) -> None:
        """Writing over a column substitutes data with nothing in the configuration to show it."""
        pairs = [(date(2026, 1, 1), date(2026, 3, 1))]

        with pytest.raises(ConfigError) as caught:
            _counted(pairs, output="from_date")

        assert "from_date" in str(caught.value)

    @pytest.mark.parametrize(
        ("config", "named"),
        [
            pytest.param(_config(unit="years"), "years", id="a_unit_with_no_consumer"),
            pytest.param(_config(unit="weeks"), "weeks", id="a_unit_nobody_asked_for"),
            pytest.param(_config(unit="months"), "months", id="a_unit_spelled_in_the_plural"),
            pytest.param(_config(unit=None), "unit", id="a_unit_written_with_no_value"),
            pytest.param(_config(counting=None), "counting", id="a_counting_written_with_no_value"),
            pytest.param({k: v for k, v in _config().items() if k != "counting"}, "counting", id="no_counting_at_all"),
            pytest.param(_config(counting="both"), "both", id="the_old_word_for_inclusive"),
            pytest.param(_config(output=" "), "output", id="an_output_name_that_is_only_whitespace"),
        ],
    )
    def test_a_unit_or_a_reading_of_the_ends_that_is_not_published_is_refused(
        self,
        config: Mapping[str, Any],
        named: str,
    ) -> None:
        """Two shipped units, two readings, and nothing accepted by resemblance.

        ``years`` is refused because nothing consumes it: whether January 2026 to December 2026 is
        one year or two is a question with two defensible answers and no consumer to settle it, and
        a unit shipped on speculation would be answered by whoever implemented it. ``both`` is the
        old spelling of ``inclusive`` and is refused rather than read as it. ``counting`` has no
        default, so leaving it out is refused, and so is a field written with no value, which is
        what an author gets from an empty YAML value.
        """
        with pytest.raises(ConfigError) as caught:
            run_tool(DateDiff(), Ports({"rows": _ranges([(date(2026, 1, 1), date(2026, 3, 1))])}), config)

        assert named in str(caught.value)

    def test_the_template_pins_two_date_expressions_and_a_required_unit_and_counting(self) -> None:
        """Neither ``unit`` nor ``counting`` has a default.

        A divisor's unit is not something to infer from silence: a configuration that does not say
        what it counted produces a figure nobody can check. The same goes for which ends count,
        because the two readings differ by one.
        """
        template = DateDiff().config_template()

        assert set(template) == {"start", "end", "unit", "counting", "output"}
        assert expression_fields(template) == ("start", "end"), "the loader seam is the declared type"
        assert template["start"].required is True
        assert template["end"].required is True
        assert template["output"].required is True
        assert template["unit"].required is True, "a divisor's unit is never inferred from silence"
        assert template["unit"].default is None
        assert set(template["unit"].enum or ()) == {"month", "day"}
        assert template["counting"].required is True, "the two readings differ by one, so neither is assumed"
        assert template["counting"].default is None
        assert set(template["counting"].enum or ()) == {"inclusive", "elapsed"}


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestDateDiffInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_date_diff_conforms_and_its_help_states_the_counting_rule_in_words(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output.

        The examples have to show the two counts a reader cannot guess from the configuration: a
        backwards range, and a row with a missing date. Both are numbers somebody will otherwise
        meet for the first time in a report.
        """
        tool = DateDiff()

        check_tool(tool)

        assert tool.name == "date_diff"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        help_ = tool.help()
        documented = "\n".join([help_.detailed_description, *help_.notes]).lower()
        assert "month" in documented and "day" in documented, (
            "a reader of a configuration cannot guess whether a partial month counts, so the rule "
            "has to be stated in words"
        )

        counts: list[Any] = [
            value
            for example in help_.examples
            for name in example.outputs.names
            for value in example.outputs[name][example.configuration["output"]].to_list()
        ]
        assert len(help_.examples) >= 2, "one plain example, and one showing a count nobody can guess"
        assert any(value is not None and value < 0 for value in counts), "a backwards range must be documented"
        assert any(value is None for value in counts), "and so must a row with a missing date"

    def test_date_diff_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "date_diff" in REGISTRY
        assert isinstance(REGISTRY.get("date_diff"), DateDiff)
