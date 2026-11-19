"""Tests for the ExpandDateRange tool: one row in, one row per period out, and nothing lost.

ExpandDateRange takes one input port, ``rows``, and produces two: ``rows``, carrying one row per
period between a start and an end, and ``unexpanded``, carrying the source rows it could not expand.
It is the only tool in the set that *increases* the row count, which makes it the one where a wrong
answer multiplies rather than merely misreports.

Four decisions are pinned here, each with a silent failure mode:

**Both ends are inclusive, at period granularity.** January to December is 12 rows, not 11 and not
13. The period containing the start counts however far into it the start falls, and so does the
period containing the end: an end spelled as the last day of a month is the overwhelmingly common
spelling, and truncating it would drop a whole period's money from every such row, while an end that
genuinely falls mid-period is a period the range still touches. Same start and end is one row, never
zero -- a range that covers a single day still covers the period that day sits in.

**A backwards range produces no rows and leaves on ``unexpanded``.** A range whose end falls before
its start is real data, not an impossibility: a mistyped year at one end produces one, and the fix
belongs to a repair step where it can be reviewed, not to the arithmetic here. So this tool neither
invents periods for it nor stops the run: it sets the row aside, by name, the way Filter and
RemoveDuplicates set aside what they did not keep. That agrees with what DateDiff owes the same row
-- a negative count rather than a refusal -- with the one difference that a count can be negative and
a row count cannot. Backwards is judged on the dates and not on the periods they fall in, so a range
whose ends sit in the same period the wrong way round is set aside too, rather than quietly becoming
the one plausible row period arithmetic would give it.

**A row that cannot be expanded must not vanish.** Left on ``rows`` alone, a null start or a
backwards range would contribute nothing and the row's figures would leave the pipeline with no
trace at all -- and the trap is worse than that, because the obvious implementation explodes such a
row into a single row whose period is null, which is neither a period nor an omission but a silent
corruption of the schedule. Hence the second output port, and hence the conservation check below:
every source row is accounted for on exactly one port, and the money on both ports together is the
money that arrived.

**``unexpanded`` carries the input schema exactly** -- no period column, no reason column. The
reason is one predicate away downstream (``[start] is null or [end] is null or [end] < [start]``),
and keeping the port union-compatible with the input is what lets a flow route it into a report, an
assertion, or a repair step beside rows from anywhere else. This is the same argument Filter makes
for having no third port for its null rows.

Vocabulary is deliberately generic -- key, amount, term_start, term_end, period_start -- because the
library carries no caller's nouns; this is the shape of a real schedule-building step, not its name.
"""

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import CATEGORIES, ConfigError, Ports, REGISTRY, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.expand_date_range import ExpandDateRange

_TERM: Mapping[str, Any] = {"start": "[term_start]", "end": "[term_end]"}

# ---------------------------------------------------------------------------
# Test data -- the shape a real schedule-building step runs against
# ---------------------------------------------------------------------------


def _rows() -> pl.DataFrame:
    """One frame carrying every shape this step meets: a clean term, a partial one, and two faults.

    Key 1 runs January to March, key 2 mid-March to mid-April, key 3 backwards by four months, and
    key 4 has no start at all. The last two are the rows whose money would otherwise disappear.
    """
    return pl.DataFrame(
        {
            "key": [1, 2, 3, 4],
            "amount": [120.0, 60.0, 10.0, 99.0],
            "term_start": [date(2026, 1, 1), date(2026, 3, 15), date(2026, 5, 1), None],
            "term_end": [date(2026, 3, 31), date(2026, 4, 10), date(2026, 1, 31), date(2026, 2, 1)],
        }
    )


def _expanded(frame: pl.DataFrame, config: Mapping[str, Any] | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``frame`` expanded through the boundary, as ``(rows, unexpanded)``."""
    result = run_tool(ExpandDateRange(), Ports({"rows": frame}), dict(config or _TERM))
    assert result.names == ("rows", "unexpanded"), "the expanded rows first, the ones set aside second"
    return result["rows"], result["unexpanded"]


def _one_range(start: date | None, end: date | None) -> pl.DataFrame:
    """A single-row frame carrying one range, for pinning a boundary."""
    return pl.DataFrame(
        {"key": [1], "term_start": [start], "term_end": [end]},
        schema_overrides={"term_start": pl.Date, "term_end": pl.Date},
    )


def _boundary_cases() -> list[Any]:
    """(start, end, the periods it must produce) for every boundary worth pinning hard."""
    return [
        pytest.param(
            date(2026, 1, 1),
            date(2026, 12, 31),
            [date(2026, month, 1) for month in range(1, 13)],
            id="a_twelve_period_term_is_twelve_rows_not_eleven_or_thirteen",
        ),
        pytest.param(date(2026, 1, 1), date(2026, 1, 1), [date(2026, 1, 1)], id="the_same_day_at_both_ends_is_one_row"),
        pytest.param(
            date(2026, 1, 1),
            date(2026, 1, 31),
            [date(2026, 1, 1)],
            id="an_end_one_day_before_the_next_period_is_still_one_row",
        ),
        pytest.param(
            date(2026, 1, 1),
            date(2026, 2, 1),
            [date(2026, 1, 1), date(2026, 2, 1)],
            id="an_end_on_a_period_boundary_counts_that_period",
        ),
        pytest.param(
            date(2026, 1, 15),
            date(2026, 3, 5),
            [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)],
            id="a_start_and_an_end_both_mid_period_count_both_periods",
        ),
        pytest.param(
            date(2025, 12, 1),
            date(2026, 1, 31),
            [date(2025, 12, 1), date(2026, 1, 1)],
            id="a_range_crossing_a_year_boundary",
        ),
    ]


# ---------------------------------------------------------------------------
# Expansion: how many rows, carrying what, in what order
# ---------------------------------------------------------------------------


class TestExpandDateRangeExpansion:
    """The row count is the whole risk here, so every boundary is pinned by value."""

    @pytest.mark.parametrize(("start", "end", "periods"), _boundary_cases())
    def test_a_range_expands_to_one_row_per_period_inclusive_of_both_ends(
        self,
        start: date,
        end: date,
        periods: Sequence[date],
    ) -> None:
        """One row per period, in ascending order, with the period as a calendar date.

        This is the assertion that a wrong answer multiplies through: a term expanded into 11 periods
        instead of 12 understates every figure built from it by a period, and does so on every row.
        """
        rows, unexpanded = _expanded(_one_range(start, end))

        assert rows["period_start"].to_list() == list(periods)
        assert rows["period_start"].dtype == pl.Date, "a period is named by the first day it covers"
        assert rows.height == len(periods)
        assert unexpanded.height == 0, "a range this tool can expand is never also set aside"

    def test_every_other_column_is_carried_onto_every_produced_row_unchanged(self) -> None:
        """The source row's columns, values and dtypes are repeated onto each period.

        Including the start and end themselves: a schedule row that has lost the term it came from
        cannot be traced back to it, and the period column is added last so an author reading the
        output finds the columns they already knew in the order they already knew them.
        """
        frame = _rows().head(1)

        rows, _ = _expanded(frame)

        assert rows.columns == [*frame.columns, "period_start"]
        assert rows["key"].to_list() == [1, 1, 1]
        assert rows["amount"].to_list() == [120.0, 120.0, 120.0]
        assert rows["term_start"].to_list() == [date(2026, 1, 1)] * 3
        assert rows["term_end"].to_list() == [date(2026, 3, 31)] * 3
        assert rows.select(frame.columns).schema == frame.schema, "no dtype may change on the way through"

    def test_produced_rows_run_in_period_order_within_a_source_row_and_in_input_order_across_them(self) -> None:
        """Both halves of the order promise, checked at once against a frame with two good rows.

        Without the second half a flow cannot rely on the output being grouped by source row at all,
        and a report built on it reshuffles whenever the engine's own internals change.
        """
        rows, _ = _expanded(_rows())

        assert rows["key"].to_list() == [1, 1, 1, 2, 2]
        assert rows["period_start"].to_list() == [
            date(2026, 1, 1),
            date(2026, 2, 1),
            date(2026, 3, 1),
            date(2026, 3, 1),
            date(2026, 4, 1),
        ]

    def test_a_datetime_at_either_end_is_read_as_its_calendar_date(self) -> None:
        """A timestamp is a date this tool can use, and the period it produces is still a date.

        The step in front of this one may hand over either, and refusing a timestamp would push a
        cast into every flow for no gain -- while producing a timestamp period column would make the
        schedule's key disagree with a date column anything else in the flow carries.
        """
        frame = pl.DataFrame(
            {
                "key": [1],
                "term_start": [datetime(2026, 1, 20, 9, 30)],
                "term_end": [datetime(2026, 3, 2, 0, 0)],
            }
        )

        rows, unexpanded = _expanded(frame)

        assert rows["period_start"].to_list() == [date(2026, 1, 1), date(2026, 2, 1), date(2026, 3, 1)]
        assert rows["period_start"].dtype == pl.Date
        assert unexpanded.height == 0


# ---------------------------------------------------------------------------
# The rows that cannot be expanded: set aside, never dropped
# ---------------------------------------------------------------------------


class TestExpandDateRangeUnexpanded:
    """What happens to a row this tool cannot turn into periods, and why it needs its own port."""

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            pytest.param(date(2026, 5, 1), date(2026, 1, 31), id="backwards_across_several_periods"),
            pytest.param(date(2026, 1, 20), date(2026, 1, 5), id="backwards_inside_a_single_period"),
        ],
    )
    def test_a_backwards_range_produces_no_rows_and_leaves_on_the_unexpanded_port(
        self,
        start: date,
        end: date,
    ) -> None:
        """Not one row with a null period, not zero rows anywhere -- one row on ``unexpanded``.

        The obvious implementation explodes an empty period list into a single row whose period is
        null, which is the worst of the three outcomes: it looks like a schedule row, it carries the
        row's whole figure, and it sits in a report as a period that does not exist.

        Backwards is judged on the dates, not on the periods they fall in, which is what the second
        case pins. Both of its ends sit in the same period, so period arithmetic alone would produce
        one perfectly plausible row for a range that contradicts itself -- the shape a transposed day
        and month makes. It is also the answer DateDiff gives the same row: it counts such a range
        negative rather than zero, so a flow reading a negative count and a flow reading this port
        find the same rows.
        """
        frame = _one_range(start, end)

        rows, unexpanded = _expanded(frame)

        assert rows.height == 0, "a backwards range is not a period"
        assert_frame_equal(unexpanded, frame)

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            pytest.param(None, date(2026, 3, 31), id="no_start"),
            pytest.param(date(2026, 1, 1), None, id="no_end"),
            pytest.param(None, None, id="neither_end"),
        ],
    )
    def test_a_missing_start_or_end_leaves_the_row_on_the_unexpanded_port(
        self,
        start: date | None,
        end: date | None,
    ) -> None:
        """A range with a missing end cannot be expanded, and must not therefore be forgotten.

        A null period column would be indistinguishable from a period, and a dropped row would take
        its figures out of the pipeline with nothing left to say it had been there.
        """
        frame = _one_range(start, end)

        rows, unexpanded = _expanded(frame)

        assert rows.height == 0
        assert unexpanded.height == 1
        assert "period_start" not in unexpanded.columns, "the port stays union-compatible with the input"

    def test_every_source_row_is_accounted_for_on_exactly_one_port(self) -> None:
        """The conservation law, in rows and in money: nothing leaves without a trace.

        A source row either contributes at least one period to ``rows`` or exactly one row to
        ``unexpanded``, never both and never neither. Checking the figures as well as the counts is
        what makes this an accounting statement rather than a row-count coincidence.
        """
        frame = _rows()

        rows, unexpanded = _expanded(frame)

        expanded_keys = sorted(set(rows["key"].to_list()))
        set_aside_keys = sorted(unexpanded["key"].to_list())
        assert expanded_keys == [1, 2]
        assert set_aside_keys == [3, 4]
        assert not set(expanded_keys) & set(set_aside_keys), "no row may leave on both ports"
        assert len(expanded_keys) + len(set_aside_keys) == frame.height, "no row may leave on neither"

        once_per_source_row = rows.unique(subset=["key"], keep="first")["amount"].sum()
        assert once_per_source_row + unexpanded["amount"].sum() == frame["amount"].sum(), (
            "the money on both ports together is the money that arrived"
        )

    def test_both_ports_stay_typed_when_they_are_empty(self) -> None:
        """An empty port still carries its schema, because the step below reads its columns.

        ``unexpanded`` is the input's schema exactly; ``rows`` is that plus the period column. Both
        have to hold when nothing reached them, which is the ordinary case for ``unexpanded`` on
        clean data and for ``rows`` on a frame that arrived empty.
        """
        frame = _rows()

        clean_rows, empty_unexpanded = _expanded(frame.head(1))
        assert empty_unexpanded.height == 0
        assert empty_unexpanded.schema == frame.schema

        no_rows, no_unexpanded = _expanded(frame.head(0))
        assert (no_rows.height, no_unexpanded.height) == (0, 0)
        assert no_rows.schema == pl.Schema({**frame.schema, "period_start": pl.Date})
        assert no_unexpanded.schema == frame.schema
        assert clean_rows.schema == no_rows.schema, "a port's schema does not depend on what reached it"


# ---------------------------------------------------------------------------
# Refusals, the template, and the catalog
# ---------------------------------------------------------------------------


class TestExpandDateRangeRefusals:
    """A configuration this tool cannot honour is refused on the schema, before any row is read."""

    @pytest.mark.parametrize(
        ("config", "named"),
        [
            pytest.param({"start": "[key]", "end": "[term_end]"}, "key", id="a_number_as_a_start"),
            pytest.param({"start": "[term_start]", "end": "[label]"}, "label", id="text_as_an_end"),
            pytest.param({"start": "[term_start]", "end": "[term_ed]"}, "term_ed", id="a_column_that_is_not_there"),
        ],
    )
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_an_end_that_is_not_a_date_is_refused_before_any_row_is_read(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """A range built from a number or a piece of text is a configuration fault, named as one.

        Handed to the engine, some of these expand into something -- a number range, or nothing at
        all -- and the run continues with a schedule built from periods that are not dates. The
        refusal has to name the column and quote the expression, because these are read in a log by
        someone holding a file of many steps.
        """
        frame = _rows().with_columns(pl.lit("2026-01").alias("label")).head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(ExpandDateRange(), Ports({"rows": frame}), dict(config))

        assert named in str(caught.value)

    def test_a_period_column_name_that_already_exists_is_refused(self) -> None:
        """Adding a column over one already there would overwrite data nobody asked it to.

        Silently replacing the existing column is the failure this closes: the overwritten values are
        gone, and the run reports success.
        """
        frame = _rows()

        with pytest.raises(ConfigError) as caught:
            run_tool(ExpandDateRange(), Ports({"rows": frame}), {**_TERM, "period_column": "amount"})

        assert "amount" in str(caught.value)

    def test_expand_date_range_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the data must be unchanged too."""
        frame = _rows()
        before = frame.clone()

        _expanded(frame)

        assert_frame_equal(frame, before)

    def test_the_template_declares_two_date_expressions_one_interval_and_the_period_column(self) -> None:
        """Both ends are expressions, so a loader can pre-flight them; the interval is a whitelist.

        ``month`` is the only interval that ships, and the enum has one member on purpose: a whitelist
        with an untested arm in it is worse than a short one, and every other interval brings its own
        boundary question -- what a partial week or a partial year is worth -- which nothing has asked
        for yet. Its spelling avoids everything YAML would read as a boolean.
        """
        template = ExpandDateRange().config_template()

        assert set(template) == {"start", "end", "unit", "period_column", "in_zone"}
        assert (template["start"].required, template["end"].required) == (True, True)
        assert expression_fields(template) == ("start", "end"), "the loader seam is the declared type"
        assert tuple(template["unit"].enum or ()) == ("month",)
        assert template["unit"].default == "month"
        assert template["period_column"].default == "period_start"


class TestExpandDateRangeInTheCatalog:
    """ExpandDateRange is described, documented with examples that run, and resolvable by name."""

    def test_expand_date_range_conforms_and_an_example_documents_a_row_it_cannot_expand(self) -> None:
        """Conformance runs every worked example over *both* ports, so neither can drift.

        The port that most needs an example is ``unexpanded``, because it is the one a reader does
        not expect a tool like this to have.
        """
        tool = ExpandDateRange()

        check_tool(tool)

        assert tool.name == "expand_date_range"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows", "unexpanded")

        examples = tool.help().examples
        assert len(examples) >= 2, "one clean term, and one carrying a row that cannot be expanded"
        assert any(example.check_row_order for example in examples), (
            "the tool promises period order within a source row, so an example must have it checked"
        )
        assert any(example.outputs["unexpanded"].height > 0 for example in examples), (
            "a reader cannot guess what lands on 'unexpanded', so an example must show a row landing there"
        )

    def test_expand_date_range_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "expand_date_range" in REGISTRY
        assert isinstance(REGISTRY.get("expand_date_range"), ExpandDateRange)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_period_column_name_is_refused_when_the_pipeline_loads(blank: str) -> None:
    """A column with no name cannot be referred to by any later step, so it is never written."""
    config = {"start": "[s]", "end": "[e]", "period_column": blank}
    rows = pl.DataFrame({"s": [date(2026, 1, 1)], "e": [date(2026, 2, 1)]})

    with pytest.raises(ConfigError, match="'period_column' must not be blank"):
        ExpandDateRange().validate_config(config)
    with pytest.raises(ConfigError, match="'period_column' must not be blank"):
        run_tool(ExpandDateRange(), Ports({"rows": rows}), config)


class TestTheEndsAreReadAsDates:
    """Both ends are compared as calendar dates, as date_diff reads them.

    Two timestamps on the same day the wrong way round (15:00 then 09:00) used to be called a
    backwards range and set aside, while date_diff counted the same row as one day. A timestamp
    is read as its calendar date in both tools now.
    """

    @staticmethod
    def _expand(start: Any, end: Any) -> Ports:
        rows = pl.DataFrame({"key": [1], "s": [start], "e": [end]})
        return run_tool(ExpandDateRange(), Ports({"rows": rows}), {"start": "[s]", "end": "[e]"})

    def test_two_times_on_one_day_the_wrong_way_round_are_one_period(self) -> None:
        result = self._expand(datetime(2026, 1, 31, 15, 0), datetime(2026, 1, 31, 9, 0))

        assert result["rows"]["period_start"].to_list() == [date(2026, 1, 1)]
        assert result["unexpanded"].height == 0

    def test_date_diff_counts_the_same_row_forwards(self) -> None:
        from dantron_etl.tools.date_diff import DateDiff

        rows = pl.DataFrame({"s": [datetime(2026, 1, 31, 15, 0)], "e": [datetime(2026, 1, 31, 9, 0)]})
        config = {"start": "[s]", "end": "[e]", "unit": "day", "counting": "inclusive", "output": "n"}

        assert run_tool(DateDiff(), Ports({"rows": rows}), config)["rows"]["n"].to_list() == [1]

    def test_a_timestamp_on_the_day_before_its_start_is_still_backwards(self) -> None:
        """The mirror: one calendar day the wrong way round still leaves on 'unexpanded'."""
        result = self._expand(datetime(2026, 1, 31, 0, 30), datetime(2026, 1, 30, 23, 30))

        assert result["rows"].height == 0
        assert result["unexpanded"]["key"].to_list() == [1]

    @pytest.mark.parametrize("value", [time(9, 0), timedelta(days=1)], ids=["time", "duration"])
    @pytest.mark.parametrize("height", [1, 0])
    def test_a_time_of_day_or_a_duration_is_refused_before_any_row_moves(self, value: Any, height: int) -> None:
        rows = pl.DataFrame({"key": [1], "s": [date(2026, 1, 1)], "e": [value]}).head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(ExpandDateRange(), Ports({"rows": rows}), {"start": "[s]", "end": "[e]"})

        message = str(caught.value)
        for word in ("'end'", "'e'", "not a date"):
            assert word in message, f"refusal should name {word!r}: {message}"


class TestATimestampInAZoneIsExpandedInAStatedZone:
    """expand_date_range reads a zone-aware start or end as its date in the zone 'in_zone' states."""

    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"s": [datetime(2026, 1, 1, 5, 0)], "e": [datetime(2026, 2, 1, 3, 0)], "d": [date(2026, 1, 1)]}
        ).with_columns(
            pl.col("s").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Toronto"),
            pl.col("e").dt.replace_time_zone("UTC").dt.convert_time_zone("America/Toronto"),
        )

    def _periods(self, **extra: Any) -> list[Any]:
        config = {"start": "[s]", "end": "[e]", **extra}
        return run_tool(REGISTRY.get("expand_date_range"), Ports({"rows": self._frame()}), config)["rows"]["period_start"].to_list()

    def test_without_a_zone_it_is_refused(self) -> None:
        with pytest.raises(ConfigError) as caught:
            self._periods()
        assert "in_zone" in str(caught.value) and "America/Toronto" in str(caught.value)

    def test_utc_expands_as_before_and_the_local_zone_by_the_local_calendar(self) -> None:
        assert self._periods(in_zone="UTC") == [date(2026, 1, 1), date(2026, 2, 1)]
        assert self._periods(in_zone="America/Toronto") == [date(2026, 1, 1)]

    def test_a_zone_when_neither_side_carries_one_is_refused(self) -> None:
        config = {"start": "[d]", "end": "[d]", "in_zone": "UTC"}
        with pytest.raises(ConfigError):
            run_tool(REGISTRY.get("expand_date_range"), Ports({"rows": self._frame()}), config)
