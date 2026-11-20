"""Tests for the CrossTab tool: a long frame turned wide, and what the new columns mean.

CrossTab takes one input port, ``rows``, and produces one, ``rows``. One output row per distinct
combination of ``by``. One new column per distinct value of ``names_from``. Each cell holds the chosen
aggregate of ``values_from`` over the rows that landed in it.

The real shape is a series: one row per key and period becomes one row per key with a column per
period. Thirteen decisions are pinned here, because each one has a quiet failure mode.

**Column order is the sorted distinct values of ``names_from``.** Not first appearance. A reader scans the
new columns across a page, and first appearance would order them by whatever order the rows
happened to arrive in. The frame below is built so the two orders disagree, so a tool that hands
back appearance order dies here rather than on somebody's report.

**A value of ``names_from`` has to be text already.** A column name is text. Turning a number or a date
into text is a formatting choice with several right answers, and picking one here would freeze it
into every cross tab this library ever produces. So a non-text ``names_from`` column is refused off the
schema, and the step that formats runs in front. That also settles what "sorted" means: sorted as
text, which is why a period written as ``p1``, ``p2``, ``p3`` sorts the way a reader expects.

**Sorted as text is sorted as text, even where that is not the nicer order.** ``p10`` comes before
``p2``. The tool is handed text and has no licence to read a number out of it. An author who wants
``p01``, ``p02``, ``p10`` formats them that way in the step in front. p1, p2, p3 sort the same
either way, so the case that shows this has a frame of its own.

**A cell with no underlying row is null.** Not zero, and there is no ``fill`` field to make it
zero. Three reasons, and the third is the one that decides it.

First, the two facts are the same fact. "No rows for this key in this period" and "nothing was
measured" say the same thing, and a 0.0 is a figure nobody measured. Downstream, nothing can tell a
fabricated zero from a real one. That is the same ruling GroupBy already makes about summing a
group whose values are all missing, and the two tools have to agree or a reader learns one rule per
tool.

Second, a fill value has no single type. The value columns take their type from the aggregate and
from ``values_from``, so a cross tab of text has text cells and a numeric fill is nonsense in it. A
``fill`` field would need a type rule of its own, and would have to say what it does to a cell that
genuinely aggregated to null, which is a different fact from an absent row.

Third, and this is the real cost of the ruling: the value column names come from the data, so no
later step can name them in its own configuration. A caller who wants a paste-ready tab of zeros
cannot write a per-column repair after this step. That is a real gap, and the honest place to close
it is one generic fill step that takes a frame and a policy, written once, rather than a flag on
every tool in the catalog that can produce a null. Until somebody asks for that step, the gap stays
open and visible rather than being papered over here.

**A missing value beside a real one is skipped, not contagious.** A cell holding 10.0, 3.0 and a
blank sums to 13.0. Three cases look alike and are not: a cell with no rows is null, a cell whose
every value is missing is null, and a cell with some values is answered from the ones it has. Only
the third has a figure in it, and blanking it blanks a period somebody was billed. GroupBy publishes
the same rule, so the two tools agree.

**Every published aggregate computes the thing it is named after.** The aggregates on offer are
GroupBy's published set, borrowed rather than copied, so one added there is accepted here the day it
lands. An aggregate with no implementation of its own falls through to whatever the code ends with
and fills the grid with a plausible number under the wrong name. So the case below walks the
published set rather than a list typed out beside it, and every name in it is checked against an
answer written down by hand and against GroupBy's answer for the same rows.

**``join_text`` fills a cell the way GroupBy fills a group, with one cell this tool has that GroupBy
has not.** Same arrival order, same skipping of what is missing, and the same empty string for a
cell whose values are all missing -- which is GroupBy's ruling, and deliberately the opposite of
what ``sum`` says about the same cell. A cell with *no row* under it stays null instead, because
that is this tool's ruling everywhere and it is a distinction GroupBy never has to make: a group
with no rows does not exist. The engine underneath draws no such line -- it joins a filter matching
nothing to ``""`` exactly as it joins a cell of nothing but blanks -- so the null is asked for, and
the case below puts the two cells side by side in one tab where nothing else in this file does.
Joining a column that is not text is refused on the schema, for the reason GroupBy refuses it: a
grid of joined numbers is plausible, unsortable and unsummable, and nobody is told.

**A null in the ``names_from`` column stops the run.** Refused, not dropped, and not given a column of its
own. Dropping the row would delete its figures from the tab with nothing left to say a row had been
there. Giving it a name is worse: every name available is a name the data could also carry, so a
literal ``"unknown"`` in the column would silently merge with the null group and nobody would see
it happen. So the tool refuses, names the column, and reports how many rows offend and what a few
of them look like. The fix is upstream, where the author decides whether those rows are dropped or
labelled.

**A blank in the ``names_from`` column stops the run as well.** An empty string is not a name a reader can
point at, and it is the same fact as a null arriving from a different system. Accepting one and
refusing the other would make the answer depend on who produced the file.

**A new column name that collides with an index column name is refused.** This is the hazard
:func:`~dantron_etl.authoring.check_output_column` exists for, except that here the name comes from the
data, so it can only be checked once rows have been seen. Left unchecked, one of the two columns
wins and the other is gone.

**Index columns keep their order and their types.** They come first, in the order the configuration
lists them, holding what they held. The value columns follow, typed by the aggregate.

**A frame with no rows gives a frame with no rows carrying just the index columns, typed.** There
are no values of ``names_from`` to make columns from, so there are no value columns. A later step still has
a schema to compile against.

**Every refusal is read off the schema and the configuration, not off the values.** Each refusal
case runs twice: once on a populated frame and once on a frame with no rows at all. Passing both is
what shows the refusal came from the column names rather than from what happened to be in them. The
two refusals that are genuinely about values -- a null in ``names_from``, and a blank in it -- are the
exceptions, and each has its own class below.

Vocabulary is deliberately generic -- code, site, period, amount, label -- because the library
carries no caller's nouns.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import CATEGORIES, ConfigError, DataError, Ports, REGISTRY, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.cross_tab import CrossTab
from dantron_etl.tools.group_by import AGGREGATE_FUNCTIONS, GroupBy


def _rows() -> pl.DataFrame:
    """One long frame carrying every shape worth pinning.

    ``period`` first appears in the order p2, p1, p3 and sorts p1, p2, p3. The two orders disagree,
    which is what makes the column-order decision visible.

    ``code`` first appears b, a, c. Rows 1 and 4 share the pair (a, p1), so at least one cell has
    two rows underneath it and an aggregate has something to do. Row 5 is the only row for code c,
    and its ``amount`` is missing, so the cell that does exist for c holds no value -- which is a
    different fact from the cells that have no row at all.

    Row 6 sits at (a, p1) too, and carries no ``amount`` and no ``label``. That puts a missing value
    beside real ones inside a single cell, which is the third case and the one the other two cannot
    stand in for. A cell that is all-present, a cell that is all-missing and a cell with no rows at
    all leave "some of it is missing" unpinned, and that is the case a real extract is full of.

    ``site`` is an integer, so it can stand as an index column that has to keep its type, and as an
    ``names_from`` column that has to be refused for not being text.
    """
    return pl.DataFrame(
        {
            "code": ["b", "a", "b", "a", "a", "c", "a"],
            "site": [1, 1, 1, 2, 1, 1, 1],
            "period": ["p2", "p1", "p1", "p2", "p1", "p3", "p1"],
            "amount": [5.0, 10.0, 2.0, 7.0, 3.0, None, None],
            "label": ["x", "y", "z", "w", "v", "u", None],
        },
        schema={
            "code": pl.String,
            "site": pl.Int64,
            "period": pl.String,
            "amount": pl.Float64,
            "label": pl.String,
        },
    )


def _index_columns(config: Mapping[str, Any]) -> list[str]:
    """The index as the output carries it: in configured order, a repeated column counted once."""
    return list(dict.fromkeys(config["by"]))


def _wide_cases() -> list[Any]:
    """(config, the frame it must produce) for each shape of cross tab worth pinning."""
    return [
        pytest.param(
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": [2.0, 13.0, None],
                    "p2": [5.0, 7.0, None],
                    "p3": [None, None, None],
                },
                schema={"code": pl.String, "p1": pl.Float64, "p2": pl.Float64, "p3": pl.Float64},
            ),
            id="a_total_per_key_and_period_with_the_periods_in_sorted_order",
        ),
        pytest.param(
            {"by": ["code", "code"], "names_from": "period", "values_from": "amount", "function": "sum"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": [2.0, 13.0, None],
                    "p2": [5.0, 7.0, None],
                    "p3": [None, None, None],
                },
                schema={"code": pl.String, "p1": pl.Float64, "p2": pl.Float64, "p3": pl.Float64},
            ),
            id="an_index_column_named_twice_is_counted_once",
        ),
        pytest.param(
            {"by": ["site"], "names_from": "period", "values_from": "amount", "function": "sum"},
            pl.DataFrame(
                {
                    "site": [1, 2],
                    "p1": [15.0, None],
                    "p2": [5.0, 7.0],
                    "p3": [None, None],
                },
                schema={"site": pl.Int64, "p1": pl.Float64, "p2": pl.Float64, "p3": pl.Float64},
            ),
            id="an_integer_index_column_keeps_its_type",
        ),
        pytest.param(
            {"by": ["code", "site"], "names_from": "period", "values_from": "amount", "function": "sum"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "a", "c"],
                    "site": [1, 1, 2, 1],
                    "p1": [2.0, 13.0, None, None],
                    "p2": [5.0, None, 7.0, None],
                    "p3": [None, None, None, None],
                },
                schema={
                    "code": pl.String,
                    "site": pl.Int64,
                    "p1": pl.Float64,
                    "p2": pl.Float64,
                    "p3": pl.Float64,
                },
            ),
            id="two_index_columns_in_the_order_they_were_configured",
        ),
        pytest.param(
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "count"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": [1, 3, None],
                    "p2": [1, 1, None],
                    "p3": [None, None, 1],
                },
                schema={"code": pl.String, "p1": pl.Int64, "p2": pl.Int64, "p3": pl.Int64},
            ),
            id="counting_rows_gives_whole_numbers_and_still_leaves_an_absent_cell_null",
        ),
        pytest.param(
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "count_non_null"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": [1, 2, None],
                    "p2": [1, 1, None],
                    "p3": [None, None, 0],
                },
                schema={"code": pl.String, "p1": pl.Int64, "p2": pl.Int64, "p3": pl.Int64},
            ),
            id="a_cell_with_a_row_but_no_value_counts_zero_where_an_absent_cell_is_null",
        ),
        pytest.param(
            {"by": ["code"], "names_from": "period", "values_from": "label", "function": "min"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": ["z", "v", None],
                    "p2": ["x", "w", None],
                    "p3": [None, None, "u"],
                },
                schema={"code": pl.String, "p1": pl.String, "p2": pl.String, "p3": pl.String},
            ),
            id="a_text_value_column_gives_text_cells",
        ),
        pytest.param(
            {"by": ["code"], "names_from": "period", "values_from": "label", "function": "max"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": ["z", "y", None],
                    "p2": ["x", "w", None],
                    "p3": [None, None, "u"],
                },
                schema={"code": pl.String, "p1": pl.String, "p2": pl.String, "p3": pl.String},
            ),
            id="the_same_cell_answers_differently_for_min_and_for_max",
        ),
        pytest.param(
            {"by": ["code"], "names_from": "period", "values_from": "label", "function": "count_distinct"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": [1, 2, None],
                    "p2": [1, 1, None],
                    "p3": [None, None, 1],
                },
                schema={"code": pl.String, "p1": pl.Int64, "p2": pl.Int64, "p3": pl.Int64},
            ),
            id="an_aggregate_that_counts_retypes_a_text_value_column_to_whole_numbers",
        ),
        pytest.param(
            {"by": ["code"], "names_from": "period", "values_from": "label", "function": "join_text"},
            pl.DataFrame(
                {
                    "code": ["b", "a", "c"],
                    "p1": ["z", "y, v", None],
                    "p2": ["x", "w", None],
                    "p3": [None, None, "u"],
                },
                schema={"code": pl.String, "p1": pl.String, "p2": pl.String, "p3": pl.String},
            ),
            id="joined_text_reads_the_arrival_order_and_skips_the_blank_beside_it",
        ),
    ]


class TestCrossTabTurnsRowsIntoColumns:
    """One row per index combination, one column per value of ``names_from``, in sorted order."""

    @pytest.mark.parametrize(("config", "expected"), _wide_cases())
    def test_the_wide_frame_is_the_one_written_down(
        self,
        config: Mapping[str, Any],
        expected: pl.DataFrame,
    ) -> None:
        """One comparison covers the lot: which columns, in which order, typed how, holding what.

        ``assert_frame_equal`` compares row order too, which is deliberate. The index rows come back
        in the order each combination first appeared in the input, the same ruling GroupBy makes for
        the same reason: a combination spanning several columns has no obvious total order, and the
        reader of the input already holds the order of first appearance. Sorting the rows is the
        sort tool's job.
        """
        frame = _rows()

        result = run_tool(CrossTab(), Ports({"rows": frame}), config)

        assert result.names == ("rows",), "one input, one output, both named for what they carry"
        assert_frame_equal(result["rows"], expected)

    def test_the_new_columns_are_sorted_not_ordered_by_first_appearance(self) -> None:
        """The one decision a naive implementation gets wrong, pinned on its own.

        ``period`` first appears p2, p1, p3. A tool that keeps appearance order hands back a frame
        with the same values in the same cells and the columns in the wrong order, which is easy to
        miss in a large expected frame and impossible to miss here.
        """
        frame = _rows()

        result = run_tool(
            CrossTab(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        assert frame["period"].unique(maintain_order=True).to_list() == ["p2", "p1", "p3"], (
            "the input has to disagree with sorted order, or this test checks nothing"
        )
        assert result["rows"].columns == ["code", "p1", "p2", "p3"]

    def test_the_new_columns_are_sorted_as_text_where_text_and_number_order_disagree(self) -> None:
        """p10 comes before p2, because the sort is on the text and not on the number inside it.

        The module calls sorted-as-text a ruling, and this is the case where it bites. p1, p2 and p3
        sort the same either way, so the shared fixture cannot show it. A period code that reaches
        ten can, and a period code is exactly where a reader meets it first.

        The answer is not obviously the nicer one. It is the one that follows from the rule that
        ``names_from`` has to hold text already: the tool is handed text and has no licence to read numbers
        out of it. An author who wants p01, p02, p10 formats them that way in the step in front,
        where the decision is written down.
        """
        frame = pl.DataFrame(
            {
                "code": ["a", "a", "a"],
                "period": ["p2", "p10", "p9"],
                "amount": [1.0, 2.0, 3.0],
            },
            schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
        )

        result = run_tool(
            CrossTab(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        assert result["rows"].columns == ["code", "p10", "p2", "p9"]

    def test_two_rows_sharing_a_key_and_a_period_are_combined_never_dropped(self) -> None:
        """The pair of rows in one cell is added up, and the total is the whole of both.

        Rows 1 and 4 both sit at (a, p1), holding 10.0 and 3.0. The cell holds 13.0. A tool that
        took one row and discarded the other would hand back 10.0 and look entirely plausible.

        Row 6 sits there too and carries no amount, so it adds nothing. That the total is still the
        whole of the two real figures is pinned properly in :class:`TestAMissingValueBesideARealOne`.
        """
        frame = _rows()

        result = run_tool(
            CrossTab(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        wide = result["rows"]
        assert wide.filter(pl.col("code") == "a")["p1"].to_list() == [13.0]
        assert wide.height == 3, "one row per distinct code, not one row per input row"

    @pytest.mark.parametrize(("config", "expected"), _wide_cases())
    def test_a_frame_with_no_rows_gives_the_index_columns_and_nothing_else(
        self,
        config: Mapping[str, Any],
        expected: pl.DataFrame,
    ) -> None:
        """No rows means no values of ``names_from``, so there is nothing to make a value column out of.

        The index columns still come back, typed, so a later step has a schema to compile against.
        Running every shape again with nothing in it is what catches a tool that reaches for the
        first row to decide something -- which works perfectly until the day a filter upstream
        matches nothing.
        """
        empty = _rows().head(0)

        result = run_tool(CrossTab(), Ports({"rows": empty}), config)

        assert_frame_equal(result["rows"], empty.select(_index_columns(config)))
        assert result["rows"].height == 0
        assert result["rows"].width == len(_index_columns(config))

    def test_cross_tab_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame itself must be unchanged too.

        A tool that added a working column to its input in place would pass a check that only
        compares schemas afterwards, and would quietly change the other branch of any fan-out.
        """
        frame = _rows()
        before = frame.clone()

        run_tool(
            CrossTab(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        assert_frame_equal(frame, before)


class TestAnEmptyCellIsNull:
    """No row for a key in a period means null, and there is no way to configure that away.

    The argument is in this file's docstring. What is pinned here is the behaviour and the absence
    of the escape hatch, because a ``fill`` field added later would be added quietly.
    """

    def test_a_cell_with_no_row_under_it_is_null_rather_than_zero(self) -> None:
        """Code c has one row, in p3. Its p1 and p2 cells hold nothing, and nothing is null.

        A zero there is a figure nobody measured, and once it is in the frame nothing downstream can
        tell it from a total that really came to zero.
        """
        frame = _rows()

        result = run_tool(
            CrossTab(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        row = result["rows"].filter(pl.col("code") == "c")
        assert row["p1"].to_list() == [None]
        assert row["p2"].to_list() == [None]

    def test_a_cell_whose_only_value_is_missing_sums_to_null_not_zero(self) -> None:
        """The same ruling GroupBy makes, so the two tools agree.

        Code c does have a row in p3, and that row's amount is missing. Summing it gives null. The
        query engine underneath sums a group of nothing but nulls to 0, so this is a rule the tool
        has to ask for rather than one it gets for free.
        """
        frame = _rows()

        result = run_tool(
            CrossTab(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        assert result["rows"].filter(pl.col("code") == "c")["p3"].to_list() == [None]

    def test_the_configuration_offers_no_way_to_fill_an_empty_cell(self) -> None:
        """Five fields, and none of them is a fill. The fifth, 'fraction', is read only by 'quantile'.

        This is the assertion that makes the ruling stick. A ``fill`` field would be a one-line
        addition with a plausible reason behind it, and the reason it is refused -- that the cell is
        the last place where "no row" and "zero" are still distinguishable -- lives in prose that
        nobody reads while adding a field.
        """
        template = CrossTab().config_template()

        assert set(template) == {"by", "names_from", "values_from", "function", "fraction"}
        assert expression_fields(template) == (), "every field here names a column; none computes one"


class TestAMissingValueBesideARealOne:
    """A cell holding one missing figure and two real ones answers with the real ones.

    This is the third case, and the other two cannot stand in for it. A cell with no rows is null. A
    cell whose every value is missing is null. A cell holding 10.0, 3.0 and a blank is none of those:
    something was measured there, so the cell has an answer, and the blank is skipped.

    GroupBy publishes the same rule for the same reason, in the same words: nulls are skipped inside
    a group holding any value at all, so ``[10, null]`` sums to ``10``. That is what makes "every
    value missing" the one case that gives null. The two tools have to agree, so it is pinned here
    as well as there.

    The cost of getting this wrong is not an abstract one. This tool builds a grid a finance team
    reads. A cell blanked because one row beside the real ones had no figure is a period that was
    billed, showing as empty.

    The cell is code ``a`` in ``p1``. It holds 10.0, 3.0 and a missing amount, so every aggregate
    gets a different answer out of it.
    """

    def test_the_fixture_really_puts_a_missing_value_beside_real_ones(self) -> None:
        """Without this, the rest of the class checks nothing.

        Every other cell in the frame is all-present, all-missing, or has no rows. If this cell ever
        loses its blank row, the tests below keep passing and stop meaning anything.
        """
        cell = _rows().filter((pl.col("code") == "a") & (pl.col("period") == "p1"))

        assert cell["amount"].to_list() == [10.0, 3.0, None]

    def test_a_sum_beside_a_missing_amount_is_the_real_total_not_null_and_not_zero(self) -> None:
        """13.0. Not null, which would blank a figure somebody was billed, and not 0.0 either.

        The guard that keeps an all-missing cell null is the one place this can go wrong. A guard
        that asks "is every value present" instead of "is any value present" blanks this cell, and
        the frame it hands back looks entirely ordinary.
        """
        result = run_tool(
            CrossTab(),
            Ports({"rows": _rows()}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        assert result["rows"].filter(pl.col("code") == "a")["p1"].to_list() == [13.0]

    @pytest.mark.parametrize(
        ("aggregate", "expected"),
        [
            ("sum", 13.0),
            ("count", 3),
            ("count_non_null", 2),
            ("count_distinct", 2),
            ("min", 3.0),
            ("max", 10.0),
            ("first", 10.0),
        ],
    )
    def test_each_aggregate_reads_the_same_cell_its_own_way(self, aggregate: str, expected: object) -> None:
        """Seven aggregates, one cell, and the blank row means something different to each.

        ``count`` counts rows, so it says 3. ``count_non_null`` counts the values that are there, so it
        says 2. That the two differ is the whole reason they have two names, and only a cell with
        more than one row and a blank among them can show it.

        ``count_distinct`` says 2: it skips the blank, as SQL's ``COUNT(DISTINCT)`` does, so it
        agrees with ``count_non_null`` here, where the two real figures differ.

        ``sum``, ``min`` and ``max`` skip the blank and answer from the real figures.
        """
        result = run_tool(
            CrossTab(),
            Ports({"rows": _rows()}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": aggregate},
        )

        assert result["rows"].filter(pl.col("code") == "a")["p1"].to_list() == [expected]


class TestFirstAndFirstNonNullInACell:
    """``first`` is the first row's value in a cell, null included; ``first_non_null`` skips blanks."""

    def test_a_cell_whose_first_row_is_blank_tells_the_two_apart(self) -> None:
        rows = pl.DataFrame(
            {"code": ["a", "a", "b", "b"], "period": ["p1", "p1", "p1", "p1"], "label": [None, "x", None, None]}
        )
        base = {"by": ["code"], "names_from": "period", "values_from": "label"}

        first = run_tool(CrossTab(), Ports({"rows": rows}), {**base, "function": "first"})["rows"]
        known = run_tool(CrossTab(), Ports({"rows": rows}), {**base, "function": "first_non_null"})["rows"]

        assert first["p1"].to_list() == [None, None]
        assert known["p1"].to_list() == ["x", None]


def _one_cell_frame() -> pl.DataFrame:
    """One key and one period, so the whole frame is a single cell, built so no two aggregates agree.

    The amounts arrive 5.0, missing, 1.0, 9.0, 5.0, 9.0. ``first`` is 5.0, ``min`` is 1.0 and ``max``
    is 9.0, all different, so an aggregate computed under the wrong name comes back as a visibly
    different number. The shared fixture cannot do that: in every cell of it ``first`` equals either
    ``min`` or ``max``, so ``first`` computed as ``max`` would pass unnoticed.

    The repeats and the blank separate the three counts as well: six rows, five values, four distinct
    values counting the blank.

    ``note`` is the same cell in text, because an aggregate that joins text cannot be asked of a
    column of numbers: both tools refuse that on the schema, which is itself a ruling. Its values
    arrive in an order that is not their sorted order and one of them is missing, so a join that
    sorted behind the author's back or joined a blank in as a gap would come back visibly wrong.

    ``size`` is for the four statistics: 11, 3, 15, 7 and 14 with a blank, whose mean is 10, median
    11, sample standard deviation exactly 5 and 0.375 quantile exactly 9. ``late`` starts and ends
    blank, so ``last`` and ``last_non_null`` disagree. ``shipped`` holds true and false.
    """
    return pl.DataFrame(
        {
            "code": ["a", "a", "a", "a", "a", "a"],
            "period": ["p1", "p1", "p1", "p1", "p1", "p1"],
            "amount": [5.0, None, 1.0, 9.0, 5.0, 9.0],
            "note": ["beta", None, "alpha", "gamma", "beta", "delta"],
            "late": [None, 7.5, 1.0, None, 2.0, None],
            "size": [11.0, None, 3.0, 15.0, 7.0, 14.0],
            "shipped": [True, None, False, True, None, False],
        },
        schema={
            "code": pl.String,
            "period": pl.String,
            "amount": pl.Float64,
            "note": pl.String,
            "late": pl.Float64,
            "size": pl.Float64,
            "shipped": pl.Boolean,
        },
    )


def _values_column(aggregate: str) -> str:
    """Which column of :func:`_one_cell_frame` this aggregate is asked of.

    Every aggregate but one reads the amounts. ``join_text`` reads the notes, because joining a
    column of numbers into a string is refused on the schema by both tools -- so asking it of
    ``amount`` would test the refusal, not the answer.
    """
    if aggregate in ("first_non_null", "last", "last_non_null"):
        return "late"  # it starts and ends blank, so each of these disagrees with its sibling
    if aggregate in ("mean", "median", "std", "quantile"):
        return "size"
    if aggregate in ("any_true", "all_true"):
        return "shipped"
    return "note" if aggregate == "join_text" else "amount"


# The fraction a quantile is asked for wherever a test asks every aggregate the same question.
QUANTILE_FRACTION = 0.375


def _cross_tab_config(aggregate: str) -> dict[str, Any]:
    """The CrossTab configuration that asks ``aggregate`` of the one cell, with a fraction for a quantile."""
    config: dict[str, Any] = {
        "by": ["code"],
        "names_from": "period",
        "values_from": _values_column(aggregate),
        "function": aggregate,
    }
    if aggregate == "quantile":
        config["fraction"] = QUANTILE_FRACTION
    return config


def _cell_answers() -> dict[str, Any]:
    """Every published aggregate, and what it makes of the single cell in :func:`_one_cell_frame`.

    Written down by hand rather than computed, so the test has an answer of its own to hold the tool
    against. No two of them are equal. Each is the answer over the column :func:`_values_column`
    names for that aggregate.
    """
    return {
        "sum": 29.0,
        "count": 6,
        "count_non_null": 5,
        "count_distinct": 3,
        "min": 1.0,
        "max": 9.0,
        "first": 5.0,
        "first_non_null": 7.5,
        "join_text": "beta, alpha, gamma, beta, delta",
        "mean": 10.0,
        "median": 11.0,
        "std": 5.0,
        "quantile": 9.0,
        "last": None,
        "last_non_null": 2.0,
        "any_true": True,
        "all_true": False,
    }


def _group_by_config(aggregate: str) -> dict[str, Any]:
    """The GroupBy configuration that asks the same question of the same rows.

    ``count`` counts rows and takes no value at all, which is GroupBy's rule, so the field is left
    out for it. The value is the same column CrossTab is pointed at, and no separator is configured,
    because the claim being checked is that the two tools answer alike where neither was told
    anything extra.
    """
    element: dict[str, Any] = {"name": "answer", "function": aggregate}
    if aggregate != "count":
        element["value"] = f"[{_values_column(aggregate)}]"
    if aggregate == "quantile":
        element["fraction"] = QUANTILE_FRACTION
    return {"by": ["code"], "aggregations": [element]}


class TestEveryPublishedAggregateIsReallyComputed:
    """Each name in the published set computes the thing it is named after, and not something else.

    An aggregate that has no branch of its own falls through to whatever the code ends with, and the
    cell fills with a plausible number under the wrong name. Nothing anywhere says so.

    This class is written to walk the published set rather than a list typed out beside it. The
    template already promises that CrossTab offers exactly GroupBy's aggregates, so an aggregate
    added to that set is accepted in configuration here the day it lands. If nothing here walks the
    set, it is accepted and quietly computed as something else, in a finance grid, with nothing red.
    """

    def test_an_answer_is_written_down_for_every_published_aggregate(self) -> None:
        """The set and the answers have to match exactly, in both directions.

        An aggregate in the set with no answer written down is the drift this class exists to catch:
        it makes somebody decide what the new aggregate does to a cell before it ships. An answer
        written down for an aggregate nobody publishes is dead weight pointing at a name that is gone.
        """
        assert set(_cell_answers()) == set(AGGREGATE_FUNCTIONS)

    @pytest.mark.parametrize("aggregate", sorted(AGGREGATE_FUNCTIONS))
    def test_the_cell_holds_the_answer_written_down_for_this_aggregate(self, aggregate: str) -> None:
        """One cell, one aggregate, against a number written down by hand.

        ``first`` is the one this catches today. It is in the published set and in the template, and
        no worked example uses it, so conformance never reaches it. Computed as ``max`` it gives 9.0
        where the first row holds 5.0.
        """
        answers = _cell_answers()
        assert aggregate in answers, f"no answer written down for {aggregate!r}"

        result = run_tool(
            CrossTab(),
            Ports({"rows": _one_cell_frame()}),
            _cross_tab_config(aggregate),
        )

        assert result["rows"]["p1"].to_list() == [answers[aggregate]]

    @pytest.mark.parametrize("aggregate", sorted(AGGREGATE_FUNCTIONS))
    def test_cross_tab_and_group_by_answer_the_same_rows_the_same_way(self, aggregate: str) -> None:
        """The same rows, the same aggregate, two tools, one answer.

        CrossTab borrows GroupBy's published set rather than keeping a copy, so a reader who learned
        an aggregate from one tool expects it to mean the same thing in the other. Borrowing the
        names is not the same as borrowing the behaviour, and this is what makes the second half true.
        """
        rows = _one_cell_frame()

        wide = run_tool(
            CrossTab(),
            Ports({"rows": rows}),
            _cross_tab_config(aggregate),
        )
        grouped = run_tool(GroupBy(), Ports({"rows": rows}), _group_by_config(aggregate))

        assert wide["rows"]["p1"].to_list() == grouped["rows"]["answer"].to_list()


def _noted_cells() -> pl.DataFrame:
    """Three codes over two periods, holding the three shapes a joined cell can take at once.

    ``(a, p1)`` holds two notes arriving beta then alpha -- which is not their sorted order, so a
    join that sorted behind the author's back comes back visibly wrong -- beside a third row whose
    note is missing, which has to be skipped rather than joined in as a gap.

    ``(c, p1)`` holds one row and that row's note is missing, so the cell has a row under it and no
    value in it at all. ``(b, p2)`` has no row under it whatsoever. Those two cells are the pair the
    module rules on and they answer differently, which nothing else in this file reaches: every
    other ``join_text`` case here runs on a frame that is one cell, and one cell that has both rows
    and values cannot show either answer.
    """
    return pl.DataFrame(
        {
            "code": ["a", "a", "a", "b", "c", "a"],
            "period": ["p1", "p1", "p1", "p1", "p1", "p2"],
            "note": ["beta", "alpha", None, "solo", None, "zed"],
        },
        schema={"code": pl.String, "period": pl.String, "note": pl.String},
    )


class TestJoinedTextFillsACellTheWayItFillsAGroup:
    """The two look-alike cells a join meets, and the one distinction GroupBy never has to make.

    A cell whose values are *all* missing joins to ``""``. That is GroupBy's published ruling for
    the same group, deliberately the opposite of what ``sum`` says about it: joining no strings
    together produces the empty string in any language that has both, while adding no numbers
    together produces a total nobody measured. The two tools have to agree, or a reader learns one
    rule per tool.

    A cell with *no row* under it stays null. That is this tool's ruling everywhere, and it is the
    distinction GroupBy never has to make, because a group with no rows does not exist. The engine
    underneath does not draw it: asked to join a filter that matches nothing, it answers ``""``
    exactly as it answers for a cell of nothing but blanks. So the null is the tool's doing, asked
    for by the guard in front of the aggregate, and removing that guard turns every absent cell of
    a joined tab into an empty string that reads as a period somebody looked at and found nothing
    in.

    Both answers are pinned by one frame comparison, because the point is that the two cells sit
    side by side in the same tab and do not read alike.
    """

    def test_an_all_blank_cell_joins_to_the_empty_string_and_an_absent_cell_stays_null(self) -> None:
        """Six rows, three codes, two periods, and every cell of the answer written down by hand.

        ``(a, p1)`` is ``"beta, alpha"``: arrival order, and the blank row skipped rather than
        joined in as a gap. ``(c, p1)`` is ``""``: rows, no values. ``(b, p2)`` and ``(c, p2)`` are
        null: no rows at all.
        """
        result = run_tool(
            CrossTab(),
            Ports({"rows": _noted_cells()}),
            {"by": ["code"], "names_from": "period", "values_from": "note", "function": "join_text"},
        )

        expected = pl.DataFrame(
            {
                "code": ["a", "b", "c"],
                "p1": ["beta, alpha", "solo", ""],
                "p2": ["zed", None, None],
            },
            schema={"code": pl.String, "p1": pl.String, "p2": pl.String},
        )
        assert_frame_equal(result["rows"], expected)


class TestANullInTheColumnKeyStopsTheRun:
    """A missing value cannot become a column name, so the run refuses rather than guessing."""

    def test_a_null_on_value_is_refused_and_the_rows_are_counted_and_shown(self) -> None:
        """The refusal names the column, counts the offenders, and shows a few of them.

        :class:`~dantron_etl.authoring.DataError` exists for exactly this: a refusal about values rather
        than about a set-up, carrying enough to send the reader to the rows instead of to a query
        they have to write first.

        Dropping the rows instead would delete their figures from the tab with nothing left in the
        output to say a row had been there. Giving them a column of their own is worse, because any
        name chosen is a name the data could also carry, and the two groups would merge in silence.
        """
        frame = _rows().with_columns(pl.Series("period", ["p2", None, "p1", "p2", None, "p3", "p1"], dtype=pl.String))

        with pytest.raises(DataError) as caught:
            run_tool(
                CrossTab(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
            )

        error = caught.value
        assert error.offending_rows == 2, "both rows with no period offend"
        assert error.total_rows == 7, "the size of the problem is reported against the whole frame"
        assert error.examples == (
            {"code": "a", "site": 1, "period": None, "amount": 10.0, "label": "y"},
            {"code": "a", "site": 1, "period": None, "amount": 3.0, "label": "v"},
        ), "the examples have to be the rows that offend, whole, not just some rows off the top"
        assert "period" in str(error), f"refusal should name the column: {error}"

    def test_a_frame_whose_on_column_is_entirely_null_is_refused_the_same_way(self) -> None:
        """Every row offending is still a refusal, not an empty cross tab.

        A tool that filtered the nulls out before looking would hand back a frame of index columns
        and no value columns at all, which reads exactly like a frame that had no rows to begin
        with.
        """
        frame = _rows().with_columns(pl.lit(None, dtype=pl.String).alias("period"))

        with pytest.raises(DataError) as caught:
            run_tool(
                CrossTab(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
            )

        assert caught.value.offending_rows == 7
        assert caught.value.total_rows == 7


class TestABlankColumnKeyStopsTheRunToo:
    """An empty string is not a column name either, so it is refused the same way a null is.

    Today it is accepted, and becomes a column called ``""``. Two reasons that is the wrong answer.

    The first is that null and blank are the same fact arriving from different systems. One extract
    writes a period nobody filled in as null; the next writes it as an empty string. Refusing one and
    accepting the other makes the tool's answer depend on who produced the file rather than on what
    the file says. Real extracts carry both, often in the same week.

    The second is that a blank is not a usable name. The whole reason a null is refused is that a
    column name has to be something a reader can point at, and every substitute the tool could invent
    is a name the data could also carry. ``""`` fails that test exactly as null does: it reads as a
    nameless column in a spreadsheet, and nothing in the output says which rows fell into it.

    So the refusal is a :class:`DataError`, like the null one, with the column named and the rows
    counted. The fix is the same and it is upstream: drop those rows, or label them.
    """

    def test_a_blank_on_value_is_refused_naming_the_column_and_counting_the_rows(self) -> None:
        """One blank period, out of seven rows, and the run stops.

        A refusal about values has to send the reader to the rows. That means the column, the count,
        and the offending row itself, so nobody has to write a query to find out what happened.
        """
        frame = _rows().with_columns(pl.Series("period", ["p2", "p1", "p1", "p2", "p1", "p3", ""], dtype=pl.String))

        with pytest.raises(DataError) as caught:
            run_tool(
                CrossTab(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
            )

        error = caught.value
        assert error.offending_rows == 1, "the one row with a blank period offends"
        assert error.total_rows == 7, "the size of the problem is reported against the whole frame"
        assert error.examples == ({"code": "a", "site": 1, "period": "", "amount": None, "label": None},)
        assert "period" in str(error), f"refusal should name the column: {error}"


class TestANewColumnMayNotTakeAnIndexColumnsName:
    """A column name that comes from the data can still collide, and a collision is refused."""

    def test_a_value_of_on_that_matches_an_index_column_is_refused_naming_the_collision(self) -> None:
        """One of the two columns would win and the other would be gone. Neither is acceptable.

        This is the hazard :func:`~dantron_etl.authoring.check_output_column` exists for, one step
        removed: the name is not in the configuration, it is in the rows, so it can only be checked
        once the rows have been seen. The refusal has to name the offending value and the index
        column it collides with, because the author is looking at a configuration that mentions
        neither.
        """
        frame = _rows().with_columns(pl.Series("period", ["p2", "code", "p1", "p2", "p1", "p3", "p1"], dtype=pl.String))

        with pytest.raises(ConfigError) as caught:
            run_tool(
                CrossTab(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
            )

        message = str(caught.value)
        assert "code" in message, f"refusal should name the colliding name: {message}"
        assert "period" in message, f"refusal should name the column the name came from: {message}"
        assert "by" in message, f"refusal should say the collision is with the index: {message}"

    def test_a_value_of_on_matching_a_column_that_is_not_in_the_index_is_fine(self) -> None:
        """Only the index columns survive into the output, so only they can be collided with.

        ``label`` is on the input and is not an index column, so a period called ``label`` takes
        that name freely. Refusing it would refuse a cross tab for a reason the output cannot show.
        """
        frame = _rows().with_columns(
            pl.Series("period", ["p2", "label", "p1", "p2", "p1", "p3", "p1"], dtype=pl.String)
        )

        result = run_tool(
            CrossTab(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )

        assert result["rows"].columns == ["code", "label", "p1", "p2", "p3"]


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way a cross tab can be configured wrong."""
    base = {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"}
    return [
        pytest.param({**base, "by": []}, "config field 'by' must not be empty", id="an_index_of_nothing"),
        pytest.param({**base, "by": None}, "required config field 'by'", id="an_index_with_no_value"),
        pytest.param({**base, "names_from": None}, "names_from", id="an_on_with_no_value"),
        pytest.param({**base, "values_from": None}, "values_from", id="a_values_with_no_value"),
        pytest.param({**base, "function": None}, "function", id="an_aggregate_with_no_value"),
        pytest.param({**base, "by": ["branch"]}, "branch", id="an_index_column_that_is_not_there"),
        pytest.param({**base, "by": ["Code"]}, "Code", id="an_index_column_whose_case_is_wrong"),
        pytest.param({**base, "names_from": "quarter"}, "quarter", id="an_on_column_that_is_not_there"),
        pytest.param({**base, "values_from": "total"}, "total", id="a_values_column_that_is_not_there"),
        pytest.param({**base, "names_from": "code"}, "code", id="an_on_column_that_is_also_in_the_index"),
        pytest.param({**base, "values_from": "code"}, "code", id="a_values_column_that_is_also_in_the_index"),
        pytest.param({**base, "function": "average"}, "average", id="an_aggregate_nobody_published"),
        pytest.param({**base, "function": "mode"}, "mode", id="an_aggregate_this_release_does_not_have"),
        pytest.param({**base, "function": "quantile"}, "fraction", id="a_quantile_with_no_fraction"),
        pytest.param({**base, "function": "mean", "fraction": 0.5}, "fraction", id="a_fraction_beside_a_mean"),
        pytest.param({**base, "function": "quantile", "fraction": 1.5}, "1.5", id="a_fraction_above_1"),
        pytest.param({**base, "function": "quantile", "fraction": -0.1}, "-0.1", id="a_fraction_below_0"),
        pytest.param({**base, "values_from": "label", "function": "median"}, "label", id="a_median_of_text"),
        pytest.param({**base, "function": "any_true"}, "true/false", id="any_true_of_a_number"),
        pytest.param({**base, "names_from": "site"}, "site", id="an_on_column_that_is_not_text"),
        pytest.param({**base, "values_from": "label"}, "label", id="a_sum_of_something_that_is_not_a_number"),
        pytest.param(
            {**base, "values_from": "amount", "function": "join_text"},
            "amount",
            id="a_text_join_of_something_that_is_not_text",
        ),
        pytest.param({**base, "by": [7]}, "by[0]", id="an_index_entry_that_is_not_text"),
        pytest.param({**base, "by": "code"}, "by", id="an_index_written_as_one_bare_name"),
        pytest.param({**base, "names_from": ["period"]}, "names_from", id="an_on_written_as_a_list"),
        pytest.param({**base, "names_from": ""}, "names_from", id="an_on_that_is_blank"),
        pytest.param({**base, "columns": ["period"]}, "unknown config field", id="a_field_nobody_declared"),
    ]


class TestCrossTabRefusals:
    """What CrossTab refuses, and when: in its own error family, off the schema, before any row moves."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [7, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_configuration_the_schema_cannot_satisfy_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Every one of these has to arrive as a :class:`ConfigError`, never as an engine exception.

        A caller of this library catches one family. A column that is not there, an aggregate nobody
        published, a key column that is also an index column -- left to the query engine, each of
        those surfaces as some other library's exception type naming neither the field to open nor
        the file to fix.

        The same refusal has to hold on a frame with no rows, which is what shows it was read off
        the column names and the configuration rather than off the values. The empty frame is the
        sharper half of this pair for a cross tab, because a tool that waits until it has seen a
        value of ``names_from`` before checking anything would hand back an empty frame and report success.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(CrossTab(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_a_key_column_that_is_also_an_index_column_is_refused_for_that_reason(self) -> None:
        """The refusal has to say which field collided with which, not just name a column.

        An author looking at four fields that all name columns needs to be told that ``names_from`` and
        ``by`` are the two that overlap. "Column 'code' is used twice" sends them to read all
        four.
        """
        frame = _rows()

        with pytest.raises(ConfigError) as caught:
            run_tool(
                CrossTab(),
                Ports({"rows": frame}),
                {"by": ["code", "site"], "names_from": "code", "values_from": "amount", "function": "sum"},
            )

        message = str(caught.value)
        assert "code" in message, f"refusal should name the column: {message}"
        assert "names_from" in message, f"refusal should name the field that collided: {message}"
        assert "by" in message, f"refusal should name the field it collided with: {message}"

    def test_a_non_text_key_column_is_refused_naming_the_column_and_what_it_holds(self) -> None:
        """A column name is text. Formatting a number into one is a decision for the step in front.

        The refusal names the column and the type it holds, because the fix is to put a formatting
        step ahead of this one and the author needs to know what they are formatting.
        """
        frame = _rows()

        with pytest.raises(ConfigError) as caught:
            run_tool(
                CrossTab(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_from": "site", "values_from": "amount", "function": "sum"},
            )

        message = str(caught.value)
        assert "site" in message, f"refusal should name the column: {message}"
        assert "Int64" in message, f"refusal should say what the column holds: {message}"

    def test_a_text_join_of_a_non_text_values_column_is_refused_naming_the_column_and_what_it_holds(
        self,
    ) -> None:
        """The same refusal GroupBy makes, in the same words, because the mistake is the same one.

        The alternative is worse than a wrong answer: it is a plausible one. A grid of
        ``10.0, 5.0, 2.0`` reads as a list of amounts, so the step ships, the report fills, and
        every value column is text from there on -- unsortable, unsummable, and nobody is told.
        ``min`` and ``max`` stay open to anything orderable, because ordering a number is not a
        category error; joining one is.

        The refusal names the column and the type it actually holds, because an author who pointed
        ``join_text`` at ``amount`` already believes that column is text, and it names the aggregate
        because the fix may be to change that field rather than the column.
        """
        frame = _rows()

        with pytest.raises(ConfigError) as caught:
            run_tool(
                CrossTab(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "join_text"},
            )

        message = str(caught.value)
        for word in ("amount", "Float64", "join_text"):
            assert word in message, f"refusal should name {word!r}: {message}"

    def test_the_template_declares_four_fields_and_borrows_the_published_aggregates(self) -> None:
        """The aggregate list is GroupBy's, not a second copy of it.

        Two whitelists in one catalog drift. One of them gains an aggregate, the other does not, and
        a reader who learned the list from one tool writes a configuration the other refuses. So the
        enum here is derived from the same published set, and this is the assertion that keeps it
        that way.

        The rest of the template is the shape the behaviour above depends on: an index that is a
        list of names and cannot be empty, and three single names that all have to be given.
        """
        template = CrossTab().config_template()

        assert template["by"].type == "array"
        assert template["by"].required is True
        assert template["by"].non_empty is True, "an empty list is a value, so 'required' alone accepts it"
        element = template["by"].items
        assert element is not None, "an array with no declared element type checks nothing"
        assert element.type == "string"

        for name in ("names_from", "values_from", "function"):
            assert template[name].type == "string", f"{name} names one column or one aggregate"
            assert template[name].required is True

        assert tuple(template["function"].enum or ()) == tuple(sorted(AGGREGATE_FUNCTIONS)), (
            "the aggregates on offer must be the published set, not a second copy of it"
        )
        assert template["function"].default is None, "there is no obvious aggregate to assume"


class TestCrossTabInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_cross_tab_conforms_and_an_example_shows_an_empty_cell(self) -> None:
        """Conformance runs every worked example, so a documented cross tab is a checked one.

        The extra demand is that the catalog show a null cell. That is the decision a reader is most
        likely to want checked before they trust this tool with a report, and an example where every
        cell is filled shows them nothing about it.
        """
        tool = CrossTab()

        check_tool(tool)

        assert tool.name == "cross_tab"
        assert tool.category == "transform", "the frame changes shape, and the figures change with it"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert examples, "a tool nobody can copy from is a tool nobody uses"
        assert any(
            _holds_an_empty_cell(example.outputs["rows"], _index_columns(example.configuration)) for example in examples
        ), "no worked example shows a cell with no row under it, which is the ruling readers ask about"

    def test_a_worked_example_joins_text_so_conformance_runs_the_two_look_alike_cells(self) -> None:
        """The catalog is where an author meets this aggregate, and the rulings are only in prose.

        ``check_tool`` runs every worked example and compares the result; it never reads the notes.
        So a note claiming an all-blank cell joins to ``""`` and an absent one stays null is a
        sentence nothing executes, and it could be changed to claim the opposite with nothing
        turning red. An example carrying both cells makes the pair of rulings a checked output
        rather than a claim, and puts them where a reader looking for them will find them.

        The distinction is exactly the one a reader of a joined tab needs before they trust it: an
        empty cell that means "there were rows here and none of them said anything" against one
        that means "nothing landed here at all".
        """
        tool = CrossTab()

        check_tool(tool)

        joining = [example for example in tool.help().examples if example.configuration.get("function") == "join_text"]
        assert joining, "an aggregate nobody demonstrates is one conformance never runs"
        assert any(
            _holds_a_blank_cell(example.outputs["rows"], _index_columns(example.configuration))
            and _holds_an_empty_cell(example.outputs["rows"], _index_columns(example.configuration))
            for example in joining
        ), "a joined example must show both cells: the one that joins to '' and the one that stays null"

    def test_cross_tab_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "cross_tab" in REGISTRY
        assert isinstance(REGISTRY.get("cross_tab"), CrossTab)


def _holds_an_empty_cell(frame: pl.DataFrame, index: Sequence[str]) -> bool:
    """Whether any value column of ``frame`` holds a null."""
    value_columns = [name for name in frame.columns if name not in set(index)]
    return any(frame[name].null_count() > 0 for name in value_columns)


def _holds_a_blank_cell(frame: pl.DataFrame, index: Sequence[str]) -> bool:
    """Whether any value column of ``frame`` holds the empty string, which is the other answer."""
    value_columns = [name for name in frame.columns if name not in set(index)]
    return any(frame[name].dtype == pl.String and (frame[name] == "").any() for name in value_columns)


class TestFirstWhenTheFirstValueIsNull:
    """What ``first`` puts in a cell whose first row holds no value: that null, as GroupBy does.

    pandas' ``first`` skips nulls instead, so this is the rule a port or a "fix" would change
    quietly. The owner ruled on 2026-11-18 to keep it: the first row as it arrives, null or not.
    """

    # pins current behaviour, kept by the owner's ruling on finding F-02
    def test_first_returns_the_first_rows_null_rather_than_the_first_value(self) -> None:
        rows = pl.DataFrame(
            {"code": ["a", "a", "b", "b"], "period": ["p1"] * 4, "amount": [None, 7.0, 3.0, None]},
            schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
        )

        result = run_tool(
            CrossTab(),
            Ports({"rows": rows}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "first"},
        )["rows"].sort("code")

        assert_frame_equal(
            result,
            pl.DataFrame({"code": ["a", "b"], "p1": [None, 3.0]}, schema={"code": pl.String, "p1": pl.Float64}),
        )


class TestACellTooLargeForItsTypeStopsTheRun:
    """A cell summing 32-bit whole numbers keeps its type, and a total that does not fit stops the run."""

    def test_a_cell_past_the_types_limit_is_refused_by_name(self) -> None:
        rows = pl.DataFrame(
            {"code": ["a", "a", "b"], "period": ["p1", "p1", "p1"], "v": pl.Series([2_000_000_000] * 2 + [1], dtype=pl.Int32)}
        )
        config = {"by": ["code"], "names_from": "period", "values_from": "v", "function": "sum"}

        with pytest.raises(DataError) as caught:
            run_tool(CrossTab(), Ports({"rows": rows}), config)

        message = str(caught.value)
        for word in ("'p1'", "'v'", "Int32", "cast_columns", "Int64", "1 of 2"):
            assert word in message, f"refusal should name {word!r}: {message}"

    @pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32])
    def test_a_cell_that_fits_keeps_the_columns_type_and_value(self, dtype: pl.DataType) -> None:
        """The mirror: a cell just inside the limit, an absent cell and an empty frame keep the type."""
        limit = 2_147_483_647 if dtype == pl.Int32 else 4_294_967_295
        rows = pl.DataFrame(
            {"code": ["a", "a", "b"], "period": ["p1", "p1", "p2"], "v": pl.Series([limit - 1, 1, 7], dtype=dtype)}
        )
        config = {"by": ["code"], "names_from": "period", "values_from": "v", "function": "sum"}

        result = run_tool(CrossTab(), Ports({"rows": rows}), config)["rows"]
        empty = run_tool(CrossTab(), Ports({"rows": rows.head(0)}), config)["rows"]

        assert result.schema == pl.Schema({"code": pl.String, "p1": dtype, "p2": dtype})
        assert result.rows() == [("a", limit, None), ("b", None, 7)]
        assert empty.schema == pl.Schema({"code": pl.String})


class TestTheNewAggregatesInACell:
    """The statistics and the true/false aggregates fill a cell as they fill a group."""

    ROWS = pl.DataFrame(
        {
            "code": ["a", "a", "a", "b", "b"],
            "period": ["p1", "p1", "p2", "p1", "p1"],
            "amount": [1.0, 3.0, None, 4.0, None],
            "flag": [True, False, None, None, None],
            "money": pl.Series(["1.25", "2.50", "0.20", "1.00", "3.00"]).cast(pl.Decimal(10, 2)),
        },
        schema_overrides={"flag": pl.Boolean},
    )

    def _cells(self, function: str, values_from: str, **extra: Any) -> dict[str, list[Any]]:
        config = {"by": ["code"], "names_from": "period", "values_from": values_from, "function": function, **extra}
        return run_tool(CrossTab(), Ports({"rows": self.ROWS}), config)["rows"].drop("code").to_dict(as_series=False)

    def test_a_cell_of_only_nulls_gives_null_and_a_cell_with_no_rows_stays_null(self) -> None:
        assert self._cells("mean", "amount") == {"p1": [2.0, 4.0], "p2": [None, None]}
        assert self._cells("quantile", "amount", fraction=0.25) == {"p1": [1.5, 4.0], "p2": [None, None]}
        assert self._cells("any_true", "flag") == {"p1": [True, None], "p2": [None, None]}
        assert self._cells("all_true", "flag") == {"p1": [False, None], "p2": [None, None]}

    def test_an_exact_decimal_is_refused_with_what_to_do(self) -> None:
        with pytest.raises(ConfigError) as caught:
            self._cells("mean", "money")
        for word in ("money", "Decimal", "Float64", "cast_columns", "mean"):
            assert word in str(caught.value), str(caught.value)

    def test_the_fraction_is_an_optional_number_with_no_default(self) -> None:
        field = CrossTab().config_template()["fraction"]
        assert (field.type, field.required, field.default) == ("number", False, None)
