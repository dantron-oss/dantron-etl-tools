"""Tests for the Unpivot tool: a wide frame turned long, and the exact inverse of a cross tab.

Unpivot takes one input port, ``rows``, and produces one, ``rows``. Every column is either named in
``by``, in which case it stays as it is and repeats on every output row, or it is melted: its
name goes into a new column and its cell goes into another. There is no third category.

The library already turns long into wide. Having no way back is the asymmetry this closes: a frame
arriving one column per period cannot be filtered by period, joined on period, or grouped by it,
because a period is not a value anywhere in it. Eleven decisions are pinned here.

**There is deliberately no field listing which columns to melt.** Every column is either in
``by`` or melted, so a list of the rest would be redundant -- and would be the one place two
configurations could disagree with each other about the same frame. It matters more than tidiness:
the obvious name for such a field is ``on``, and a definition file that says ``on: [p1, p2]`` is
read by YAML as the boolean ``True`` standing where a name should be. That trap has already cost
this project twice, and the cheapest way not to meet it a third time is to have no field it could
name.

**``by`` is required and cannot be empty.** An unpivot with no identifier hands back a column of
names and a column of values with nothing to say which row each came from, and nobody can join it
back to anything. A flow that genuinely wants that writes a ``record_id`` in front, which is one
step and leaves the identifier visible.

**``names_to`` and ``values_to`` are required and have no default.** The two new columns are the whole
point of the reshape. A default would publish a column called something nobody chose, in the frame
every later step and every report reads from.

**The melted columns must agree on type, with one exception: a column that is all nulls takes the
others' type.** This is Union's ruling, copied on purpose and for the same reason. Stacking a text
column under a number column gives a column of text that used to be money, and whichever one got
rewritten did so in silence -- so it is refused, and the refusal names both columns and both types,
because "they differ" without saying how sends a reader back to the data to find out which. A
column of nothing is the exception because null is the one value belonging to every family, and an
unpopulated column is an ordinary frame rather than a mistake.

**A flow can melt a disagreeing set deliberately, by saying what the values are.** ``values_type``
names the type every melted column is read as, in the same spelling ``cast_columns`` uses, and while
it is given the disagreement is no longer refused. The case is a block that arrives one column per
period with unrelated text columns beside it: the block gains a column each period, so the text
columns shift position and cannot be named in ``by``, and the only way to tell them apart is to
read what each column's heading says -- which needs the melt to have happened first. Stating the
type is what makes the numbers survive as numbers. Melted as text they come back as whatever a
renderer shortens them to, and a figure that arrives a hair under a half-cent comes back a hair over
it and rounds the other way.

**Nulling a value silently is what this library refuses**, so ``on_invalid`` is spelled and reported
exactly the way ``cast_columns`` spells and reports it: ``error`` (the default) stops the run naming
the column, the offending values and the type asked for; ``empty`` leaves them null and reports how
many, out of how many, and what share, which raises the step to a warning. The count is taken before
the values are replaced, because afterwards a null that failed to convert and a null that was always
there are the same null. And ``on_invalid`` is refused without ``values_type``: there is nothing for
it to govern, and a field that silently does nothing is the class of field this library kills on
sight. That is the shape Join uses for ``keys`` against ``left_keys``, and GroupBy for a separator
beside the wrong function.

**Row order is column-major.** All the rows of the first melted column, then all the rows of the
second, in the frame's own column order. Not the input's row order interleaved, and not sorted. The
fixture below is built so the frame's column order disagrees with alphabetical order, because a
tool that sorted the melted columns behind the author's back would otherwise pass.

**Every refusal is read off the schema.** A frame of a million rows and a frame of none are refused
identically, so each refusal case below runs twice. An unpivot whose configuration only fails once
a row is seen is an unpivot that reports success on the day a filter upstream matched nothing.

**An ``by`` that covers every column is refused.** There is nothing left to melt, so the step
reshapes nothing. That is the reading ``cast_columns`` already gives a cast to the type a column
already holds: a configuration edited halfway, and saying so beats agreeing.

**A frame with no rows gives a frame with no rows and the full schema** -- the index columns as
they were, ``names_to`` as text, ``values_to`` at the melted type -- so a later step still has something to
compile against.

**A null in a melted cell survives as a null.** It is data: the row existed and the cell was empty.
Dropping it would delete a row from the long frame with nothing left to say it had been there.

Vocabulary is deliberately generic -- code, site, period, amount, label -- because the library
carries no caller's nouns.
"""

import math
import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import dantron_etl
from dantron_etl.column_types import DECLARED_DTYPE, DECLARED_TYPES
from dantron_etl.authoring import (
    CATEGORIES,
    ConfigError,
    DataError,
    Ports,
    REGISTRY,
    YAML_UNSAFE_VALUES,
    check_tool,
    run_tool,
)
from dantron_etl.config import expression_fields
from dantron_etl.tools.cross_tab import CrossTab
from dantron_etl.tools.unpivot import Unpivot

# The bare words a definition file rewrites into something other than a name. No field of this tool
# may be called any of them, which is the whole reason it has no field naming the columns to melt.
YAML_REWRITES = frozenset({"on", "off", "yes", "no", "true", "false", "null", "y", "n"})

BASE: dict[str, Any] = {"by": ["code", "site", "label"], "names_to": "period", "values_to": "amount"}


def _rows() -> pl.DataFrame:
    """One wide frame carrying every shape worth pinning.

    The melted columns are written ``p2``, ``p1``, ``p10`` on purpose. That is not their alphabetical
    order and not their numeric order, so a tool that sorted them -- either way -- comes back
    visibly wrong rather than plausibly right.

    ``code`` and ``label`` are text and ``site`` is a whole number, so the index carries more than
    one type and a tool that rebuilt the index columns through a common type would be caught.

    ``p2`` has no value for code ``c`` and ``p1`` has none for code ``d``. Those are the two cells
    that have to survive the reshape as rows holding nothing, rather than as rows that are not
    there.
    """
    return pl.DataFrame(
        {
            "code": ["a", "b", "c", "d", "e", "f", "g"],
            "site": [1, 1, 2, 2, 3, 3, 1],
            "p2": [5.0, 2.0, None, 7.0, 1.0, 4.0, 6.0],
            "p1": [10.0, 3.0, 8.0, None, 9.0, 2.5, 1.5],
            "p10": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            "label": ["x", "y", "z", "w", "v", "u", "t"],
        },
        schema={
            "code": pl.String,
            "site": pl.Int64,
            "p2": pl.Float64,
            "p1": pl.Float64,
            "p10": pl.Float64,
            "label": pl.String,
        },
    )


def _long_cases() -> list[Any]:
    """(config, the frame it must produce) for each shape of unpivot worth pinning."""
    return [
        pytest.param(
            BASE,
            pl.DataFrame(
                {
                    # The index repeats once per melted column. That repetition is the reshape.
                    "code": ["a", "b", "c", "d", "e", "f", "g"] * 3,
                    "site": [1, 1, 2, 2, 3, 3, 1] * 3,
                    "label": ["x", "y", "z", "w", "v", "u", "t"] * 3,
                    # Column-major, in the frame's own column order: p2, then p1, then p10.
                    "period": ["p2"] * 7 + ["p1"] * 7 + ["p10"] * 7,
                    # One line per melted column, in the order they are melted: p2, p1, p10.
                    "amount": (
                        [5.0, 2.0, None, 7.0, 1.0, 4.0, 6.0]
                        + [10.0, 3.0, 8.0, None, 9.0, 2.5, 1.5]
                        + [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
                    ),
                },
                schema={
                    "code": pl.String,
                    "site": pl.Int64,
                    "label": pl.String,
                    "period": pl.String,
                    "amount": pl.Float64,
                },
            ),
            id="three_index_columns_and_three_melted_ones",
        ),
        pytest.param(
            {"by": ["code", "site", "p2", "p1", "p10"], "names_to": "field", "values_to": "text"},
            pl.DataFrame(
                {
                    "code": ["a", "b", "c", "d", "e", "f", "g"],
                    "site": [1, 1, 2, 2, 3, 3, 1],
                    "p2": [5.0, 2.0, None, 7.0, 1.0, 4.0, 6.0],
                    "p1": [10.0, 3.0, 8.0, None, 9.0, 2.5, 1.5],
                    "p10": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
                    "field": ["label"] * 7,
                    "text": ["x", "y", "z", "w", "v", "u", "t"],
                },
                schema={
                    "code": pl.String,
                    "site": pl.Int64,
                    "p2": pl.Float64,
                    "p1": pl.Float64,
                    "p10": pl.Float64,
                    "field": pl.String,
                    "text": pl.String,
                },
            ),
            id="one_column_melted_and_every_other_column_left_where_it_was",
        ),
    ]


class TestUnpivotTurnsColumnsIntoRows:
    """One row per input row per melted column, carrying the column's name and the cell's value."""

    @pytest.mark.parametrize(("config", "expected"), _long_cases())
    def test_the_long_frame_is_the_one_written_down(
        self,
        config: Mapping[str, Any],
        expected: pl.DataFrame,
    ) -> None:
        """One comparison covers the lot: which columns, in which order, typed how, holding what.

        The index columns come first, as they were, then ``names_to``, then ``values_to``. Row order is
        compared too, and that is the point rather than an accident of the comparison: the order is
        a published ruling, so it is pinned here and again on its own below.
        """
        result = run_tool(Unpivot(), Ports({"rows": _rows()}), config)

        assert result.names == ("rows",), "one input, one output, both named for what they carry"
        assert_frame_equal(result["rows"], expected, check_row_order=True)

    def test_the_rows_come_back_column_major_in_the_frames_own_column_order(self) -> None:
        """All of the first melted column, then all of the second: not interleaved and not sorted.

        The fixture's melted columns are written p2, p1, p10, which is neither alphabetical order
        nor numeric order. A tool that sorted them would hand back the same rows in a different
        order, which is easy to miss in a large expected frame and impossible to miss here.

        Interleaving is the other plausible answer and is the one a naive implementation gives:
        row 1's three cells, then row 2's three cells. It reads fine and it means a later ``first``
        after a sort picks a different row.
        """
        frame = _rows()
        melted = [name for name in frame.columns if name not in set(BASE["by"])]
        assert melted == ["p2", "p1", "p10"], "the melted columns must be in the fixture's own order"
        assert melted != sorted(melted), "sorted order has to disagree, or this test checks nothing"

        result = run_tool(Unpivot(), Ports({"rows": frame}), BASE)

        assert result["rows"]["period"].to_list() == ["p2"] * 7 + ["p1"] * 7 + ["p10"] * 7
        assert result["rows"]["code"].to_list() == ["a", "b", "c", "d", "e", "f", "g"] * 3

    def test_the_index_columns_come_back_in_the_order_they_were_configured(self) -> None:
        """The configuration decides the output's column order, not the order the frame arrived in.

        A pipeline file is read to find out what a step produces, and ``by`` is the author's own
        written statement of the output's shape. Taking the frame's order instead would make the
        output depend on something that appears nowhere in the file -- an upstream select, or the
        column order of whatever file was read that morning -- so a reader could not tell what this
        step produces by reading it.

        The index here is written label, site, code, which is the reverse of the fixture's own
        order for those three columns, so frame order and configured order cannot both pass.
        """
        frame = _rows()
        index = ["label", "site", "code"]
        assert [name for name in frame.columns if name in set(index)] == ["code", "site", "label"], (
            "the frame's order must disagree with the configured order, or this test checks nothing"
        )

        result = run_tool(Unpivot(), Ports({"rows": frame}), {**BASE, "by": index})

        assert result["rows"].columns == ["label", "site", "code", "period", "amount"]

    def test_a_reordered_index_carries_each_columns_own_values_under_its_own_name(self) -> None:
        """The order ruling is about the values, not only about the headings above them.

        The test above compares column names, and names alone are satisfied by an implementation
        that selects the index columns in the frame's order and renames them into the configured
        one. Every figure then arrives under somebody else's identifier and the frame still looks
        right -- ``code`` and ``label`` are both text, so a type check catches nothing either. That
        is not a contrived mistake: it is what a rewrite onto the query engine's own reshape invites,
        because that hands the index columns back in the frame's order.

        So the whole frame is compared under a reversed index, which pins each column's contents to
        its own name rather than to its position.
        """
        result = run_tool(Unpivot(), Ports({"rows": _rows()}), {**BASE, "by": ["label", "site", "code"]})

        expected = pl.DataFrame(
            {
                "label": ["x", "y", "z", "w", "v", "u", "t"] * 3,
                "site": [1, 1, 2, 2, 3, 3, 1] * 3,
                "code": ["a", "b", "c", "d", "e", "f", "g"] * 3,
                "period": ["p2"] * 7 + ["p1"] * 7 + ["p10"] * 7,
                "amount": (
                    [5.0, 2.0, None, 7.0, 1.0, 4.0, 6.0]
                    + [10.0, 3.0, 8.0, None, 9.0, 2.5, 1.5]
                    + [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
                ),
            },
            schema={
                "label": pl.String,
                "site": pl.Int64,
                "code": pl.String,
                "period": pl.String,
                "amount": pl.Float64,
            },
        )
        assert_frame_equal(result["rows"], expected, check_row_order=True)

    def test_the_rows_arrive_in_the_order_they_came_and_a_repeated_index_is_not_collapsed(self) -> None:
        """Two things the main fixture cannot say, because its rows happen to be sorted and unique.

        Every other frame in this file is in ascending order by its own index and names each index
        value once, so a ``sort`` or a ``unique`` dropped into the reshape would be invisible in all
        of them. Both are plausible additions -- a sort because the output "looks tidier", a unique
        because an index sounds like it ought to be one -- and both are wrong here. Sorting orders
        the output by something the author never wrote down; deduplicating deletes a row of figures
        and reports success.

        The codes below are deliberately out of order, and ``a`` appears twice carrying different
        figures. The whole frame is compared with the row order checked, so within each melted part
        the rows come back in the order they arrived and both ``a`` rows are still there.
        """
        frame = pl.DataFrame(
            {
                "code": ["d", "a", "c", "a"],
                "p2": [5.0, 2.0, 8.0, 3.0],
                "p1": [1.0, 4.0, 6.0, 7.0],
            },
            schema={"code": pl.String, "p2": pl.Float64, "p1": pl.Float64},
        )
        assert frame["code"].to_list() != sorted(frame["code"].to_list()), (
            "the rows must arrive unsorted, or a sort inside the reshape would not show"
        )
        assert frame["code"].n_unique() < frame.height, (
            "an index value must repeat, or a deduplication inside the reshape would not show"
        )

        result = run_tool(
            Unpivot(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_to": "period", "values_to": "amount"},
        )

        expected = pl.DataFrame(
            {
                "code": ["d", "a", "c", "a"] * 2,
                "period": ["p2"] * 4 + ["p1"] * 4,
                "amount": [5.0, 2.0, 8.0, 3.0] + [1.0, 4.0, 6.0, 7.0],
            },
            schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
        )
        assert_frame_equal(result["rows"], expected, check_row_order=True)

    def test_naming_the_same_index_column_twice_counts_it_once(self) -> None:
        """What the template publishes about a duplicate, held to by something that runs.

        ``by`` is a list of names and nothing stops an author writing one of them twice -- a
        merge of two edits to the same file does it without anybody deciding to. The template says
        the second mention counts for nothing, and ``cross_tab`` publishes that same sentence about
        its own index, so the two have to keep agreeing. Left unenforced, a duplicate reaches the
        query engine and dies there with a message about a duplicate column that names neither the
        field to open nor what to do about it.

        Compared against the un-duplicated configuration rather than merely run, so the claim is
        that the two produce the same frame and not just that neither throws.
        """
        plain = run_tool(Unpivot(), Ports({"rows": _rows()}), BASE)
        duplicated = run_tool(
            Unpivot(),
            Ports({"rows": _rows()}),
            {**BASE, "by": ["code", "code", "site", "label"]},
        )

        assert_frame_equal(duplicated["rows"], plain["rows"], check_row_order=True)

    def test_the_name_column_is_text_and_the_value_column_takes_the_melted_type(self) -> None:
        """A column name is text, whatever the cells under it held.

        ``values_to`` is the other half and is not fixed: it takes the type of the columns that were
        melted, so an unpivot of money hands back money and an unpivot of text hands back text. A
        tool that reached for a common type -- text, being the one everything casts to -- would turn
        every figure in the frame into a string, and the report built on it would still fill.
        """
        numbers = run_tool(Unpivot(), Ports({"rows": _rows()}), BASE)

        assert numbers["rows"].schema["period"] == pl.String
        assert numbers["rows"].schema["amount"] == pl.Float64

        text = run_tool(
            Unpivot(),
            Ports({"rows": _rows()}),
            {"by": ["code", "site", "p2", "p1", "p10"], "names_to": "field", "values_to": "text"},
        )

        assert text["rows"].schema["field"] == pl.String
        assert text["rows"].schema["text"] == pl.String

    def test_a_cell_holding_nothing_becomes_a_row_holding_nothing(self) -> None:
        """A null is data. The row existed and the cell was empty, and both facts survive.

        Dropping those rows is the tempting alternative and it is the wrong one: the long frame
        would be missing a code-and-period pair with nothing left to say it had ever been there,
        and a later count by period would be short without anything looking short.
        """
        result = run_tool(Unpivot(), Ports({"rows": _rows()}), BASE)

        long = result["rows"]
        assert long.height == 21, "seven rows times three melted columns, with nothing dropped"
        assert long.filter((pl.col("code") == "c") & (pl.col("period") == "p2"))["amount"].to_list() == [None]
        assert long.filter((pl.col("code") == "d") & (pl.col("period") == "p1"))["amount"].to_list() == [None]
        assert long["amount"].null_count() == 2

    def test_unpivot_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame itself must be unchanged too.

        A tool that added a working column to its input in place would pass a check that only
        compares schemas afterwards, and would quietly change the other branch of any fan-out.
        """
        frame = _rows()
        before = frame.clone()

        run_tool(Unpivot(), Ports({"rows": frame}), BASE)

        assert_frame_equal(frame, before)


def _one_null_column() -> pl.DataFrame:
    """Two melted columns, one of them carrying nothing at all, so it has no type of its own.

    ``closing`` is declared ``Null`` rather than ``Float64`` holding nulls. That is what a branch
    that matched nothing or a column no row populated actually arrives as, and it is the case a
    naive type check refuses.
    """
    return pl.DataFrame(
        {
            "code": ["a", "b", "c"],
            "opening": [1.0, 2.0, None],
            "closing": [None, None, None],
        },
        schema={"code": pl.String, "opening": pl.Float64, "closing": pl.Null},
    )


class TestAColumnOfNothingTakesTheOthersType:
    """Union's ruling, copied on purpose: null is the one value belonging to every family.

    Union already answers this question for two frames stacked on each other, and an unpivot is the
    same question asked of two columns stacked on each other. Answering it differently here would
    mean a reader learns one rule per tool, and the rule they learned second is the one they would
    be relying on at the moment it mattered.
    """

    def test_an_all_null_melted_column_takes_the_type_of_the_others(self) -> None:
        """A column with no values does not disagree with a column of numbers; it has no type yet.

        The whole frame is compared, so the claim is not only that this is accepted but that the
        value column comes back as ``Float64`` and the nulls come back as nulls, in the right rows.
        A tool settling on ``Null`` instead would hand back a value column that is entirely empty,
        including the two real figures.
        """
        result = run_tool(
            Unpivot(),
            Ports({"rows": _one_null_column()}),
            {"by": ["code"], "names_to": "measure", "values_to": "amount"},
        )

        expected = pl.DataFrame(
            {
                "code": ["a", "b", "c", "a", "b", "c"],
                "measure": ["opening"] * 3 + ["closing"] * 3,
                "amount": [1.0, 2.0, None, None, None, None],
            },
            schema={"code": pl.String, "measure": pl.String, "amount": pl.Float64},
        )
        assert_frame_equal(result["rows"], expected, check_row_order=True)

    def test_melting_nothing_but_empty_columns_leaves_the_value_column_with_no_type_either(self) -> None:
        """Where no melted column has a type, there is none to adopt, and that is not a refusal.

        This is the far end of the same ruling. Two empty columns do not disagree, so refusing them
        would refuse an ordinary frame on the one day everything upstream matched nothing -- which
        is exactly the day a run has to keep its shape rather than stop.
        """
        frame = pl.DataFrame(
            {"code": ["a", "b"], "opening": [None, None], "closing": [None, None]},
            schema={"code": pl.String, "opening": pl.Null, "closing": pl.Null},
        )

        result = run_tool(
            Unpivot(),
            Ports({"rows": frame}),
            {"by": ["code"], "names_to": "measure", "values_to": "amount"},
        )

        assert result["rows"].height == 4
        assert result["rows"].schema["amount"] == pl.Null
        assert result["rows"]["measure"].to_list() == ["opening", "opening", "closing", "closing"]


class TestMeltedColumnsMustAgreeOnType:
    """Two types stacked into one column means one of them was rewritten, and nobody was told."""

    @pytest.mark.parametrize("height", [2, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_columns_of_different_types_are_refused_naming_both_columns_and_both_types(self, height: int) -> None:
        """Saying they differ is not a refusal. Which two, holding what, is.

        A reader meeting this refusal is holding a frame of many columns and a configuration that
        names none of the melted ones -- that is the point of the tool having no such field. So the
        message is the only place the two offending columns can be named, and without the types
        beside them the reader has to go and look the frame up before they can act.
        """
        frame = pl.DataFrame(
            {"code": ["a", "b"], "amount": [1.0, 2.0], "note": ["x", "y"]},
            schema={"code": pl.String, "amount": pl.Float64, "note": pl.String},
        ).head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(
                Unpivot(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_to": "measure", "values_to": "value"},
            )

        message = str(caught.value)
        for word in ("amount", "note", "Float64", "String"):
            assert word in message, f"refusal should name {word!r}: {message}"


#: A figure a hair under a half-cent. Written as a literal so the test carries the exact double
#: rather than a rounded copy of it, and chosen because the nearest shortened rendering of it,
#: ``1302.025``, is a different double that rounds to a different cent.
A_HAIR_UNDER_HALF_A_CENT: float = 1302.0249999999999

#: Melt everything outside ``code`` as a figure, and let the text that is not one go empty.
TYPED: dict[str, Any] = {
    "by": ["code"],
    "names_to": "period",
    "values_to": "amount",
    "values_type": "Float64",
    "on_invalid": "set_null",
}


def _mixed_block() -> pl.DataFrame:
    """A wide block whose columns do not agree: two of figures, and one of text beside them.

    This is the shape the stated type exists for. A block arrives one column per period and gains
    another one every period, with unrelated text beside it whose position therefore shifts. The
    text cannot be named in ``by``, because nothing in the configuration can know where it will
    be next time; it is told apart from a period by reading what its heading says, which is only
    possible once the melt has put the heading in a column.

    ``comment`` holds three values and one blank, so a count of what failed to convert is three out
    of four rows rather than four, and a test can tell those two answers apart.
    """
    return pl.DataFrame(
        {
            "code": ["a", "b", "c", "d"],
            "p2": [5.0, 2.0, None, 7.0],
            "p1": [10.0, 3.0, 8.0, A_HAIR_UNDER_HALF_A_CENT],
            "comment": ["ok", None, "check", "ok"],
        },
        schema={"code": pl.String, "p2": pl.Float64, "p1": pl.Float64, "comment": pl.String},
    )


class TestAMeltGivenTheTypeItsValuesAre:
    """``values_type`` melts a disagreeing set on purpose, and says what came out of it.

    The refusal above is right by default and wrong for one shape: a block a flow cannot enumerate,
    because the columns it wants and the columns it does not both move. Stating the type is how such
    a flow says "read all of this as figures" in one place a reader can see, instead of casting
    column by column in a step that has to be edited every period.
    """

    def test_a_block_of_figures_and_text_melts_into_figures_when_the_type_is_stated(self) -> None:
        """One comparison covers the lot: the figures exact, the text empty, the column a Float64.

        Without ``values_type`` this frame is refused, and rightly. With it, the refusal is the
        flow's own decision already taken, so what is left to check is that nothing was rewritten on
        the way: every figure arrives as itself, in the right row, under the right heading, and the
        text arrives as nothing rather than as ``0`` or as the string it used to be.
        """
        result = run_tool(Unpivot(), Ports({"rows": _mixed_block()}), TYPED)

        expected = pl.DataFrame(
            {
                "code": ["a", "b", "c", "d"] * 3,
                "period": ["p2"] * 4 + ["p1"] * 4 + ["comment"] * 4,
                "amount": (
                    [5.0, 2.0, None, 7.0] + [10.0, 3.0, 8.0, A_HAIR_UNDER_HALF_A_CENT] + [None, None, None, None]
                ),
            },
            schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
        )
        assert_frame_equal(result["rows"], expected, check_row_order=True)

    def test_a_melted_figure_comes_back_with_every_digit_it_arrived_with(self) -> None:
        """The defect this field exists to prevent, asserted on the exact double.

        A melt of a disagreeing block has one obvious workaround: read everything as text, since
        every value converts to text. It is the wrong answer, and the reason is arithmetic rather
        than taste. Text is what somebody renders, a renderer shortens, and the shortest rendering
        of this value is a *different* double -- one that rounds to a different cent. A figure that
        went out through text and came back is not the figure that arrived, and nothing about the
        frame looks wrong afterwards.

        So the claim is exact equality, not approximate: ``assert_frame_equal`` compares floats
        within a tolerance by default, and a tolerance is precisely what is not being granted here.
        """
        shortened = float(f"{A_HAIR_UNDER_HALF_A_CENT:.10g}")
        assert shortened != A_HAIR_UNDER_HALF_A_CENT, "the shortened form must differ, or this test checks nothing"
        assert round(shortened, 2) != round(A_HAIR_UNDER_HALF_A_CENT, 2), "and it must move the cent"

        result = run_tool(Unpivot(), Ports({"rows": _mixed_block()}), TYPED)

        melted = result["rows"].filter((pl.col("code") == "d") & (pl.col("period") == "p1"))
        assert melted["amount"].to_list() == [A_HAIR_UNDER_HALF_A_CENT]
        assert melted["amount"].to_list()[0] == A_HAIR_UNDER_HALF_A_CENT
        assert round(melted["amount"].to_list()[0], 2) == round(A_HAIR_UNDER_HALF_A_CENT, 2)

    def test_stating_the_type_columns_already_hold_is_accepted_and_changes_nothing(self) -> None:
        """Converting a column to the type it already has is ordinary here, not a halfway edit.

        ``cast_columns`` refuses exactly this, and the difference is what its input is. There, the
        columns are the ones a flow named one by one, so naming one whose type is already right is a
        line somebody forgot to delete. Here the melt set is *everything the flow did not name* --
        and the whole reason for stating a type is that the flow cannot enumerate that set -- so
        "some of them are already right" is the normal case and refusing it would refuse the feature.

        Compared against the same melt with no type stated, so the claim is that the two agree
        rather than merely that this one runs.
        """
        plain = run_tool(Unpivot(), Ports({"rows": _rows()}), BASE)
        stated = run_tool(Unpivot(), Ports({"rows": _rows()}), {**BASE, "values_type": "Float64"})

        assert_frame_equal(stated["rows"], plain["rows"], check_row_order=True)
        assert stated.notes == (), "nothing failed to convert, so there is nothing to warn about"

    def test_a_melted_column_with_no_type_at_all_is_read_as_the_stated_one(self) -> None:
        """A column declared as nothing has no type to read, and the stated type answers for it too.

        Without ``values_type`` this is Union's ruling: an empty column adopts whatever the others
        settled on. With a type stated there is nothing left to settle, so the question becomes
        whether such a column is read at all -- and it has to be, or the melt hands back a value
        column that cannot hold both it and the figures beside it. Its cells come back as nothing,
        which is what they were, and nothing is reported, because a column that had no values lost
        none.
        """
        result = run_tool(
            Unpivot(),
            Ports({"rows": _one_null_column()}),
            {"by": ["code"], "names_to": "measure", "values_to": "amount", "values_type": "Float64"},
        )

        expected = pl.DataFrame(
            {
                "code": ["a", "b", "c", "a", "b", "c"],
                "measure": ["opening"] * 3 + ["closing"] * 3,
                "amount": [1.0, 2.0, None, None, None, None],
            },
            schema={"code": pl.String, "measure": pl.String, "amount": pl.Float64},
        )
        assert_frame_equal(result["rows"], expected, check_row_order=True)
        assert result.notes == (), "a column that held no values lost none"

        # And where every melted column is like that, the stated type is the only thing left saying
        # what the value column is. Without a type stated this melt gives a value column of Null,
        # which is the honest answer to a frame that settled on nothing; with one stated, answering
        # Null would ignore the field and hand a later step a column it cannot put a figure in.
        nothing_at_all = pl.DataFrame(
            {"code": ["a", "b"], "opening": [None, None], "closing": [None, None]},
            schema={"code": pl.String, "opening": pl.Null, "closing": pl.Null},
        )

        stated = run_tool(
            Unpivot(),
            Ports({"rows": nothing_at_all}),
            {"by": ["code"], "names_to": "measure", "values_to": "amount", "values_type": "Float64"},
        )

        assert stated["rows"].height == 4
        assert stated["rows"].schema["amount"] == pl.Float64, "the stated type answers where nothing else can"
        assert stated["rows"]["amount"].null_count() == 4
        assert stated.notes == ()

    def test_the_name_column_stays_text_whatever_the_values_are_read_as(self) -> None:
        """``names_to`` holds a column heading, and a heading is text however the cells under it are read.

        Worth pinning because the two new columns are created together, and a tool that reached for
        one type for both would pass every test that only looks at the values. Checked at both ends:
        a stated type that is not text, and a stated type that is.
        """
        figures = run_tool(Unpivot(), Ports({"rows": _mixed_block()}), TYPED)

        assert figures["rows"].schema["period"] == pl.String
        assert figures["rows"].schema["amount"] == pl.Float64

        text = run_tool(Unpivot(), Ports({"rows": _mixed_block()}), {**TYPED, "values_type": "String"})

        assert text["rows"].schema["period"] == pl.String
        assert text["rows"].schema["amount"] == pl.String


class TestWhatWillNotConvert:
    """A value that does not become the stated type stops the run, or is counted as it goes empty.

    ``cast_columns`` settled this and its answer is copied whole, down to the words: ``error`` and
    ``empty``, the count taken before the values are replaced, and how many out of how many reported
    rather than the bare fact that something was dropped. Two tools answering one question similarly
    rather than identically is how a reader ends up trusting the rule they learned second.
    """

    def test_a_value_that_will_not_convert_stops_the_run_and_says_which_one(self) -> None:
        """The default is the safe one, and its refusal has to be actionable without the frame.

        Named as a refusal about data rather than about configuration: the configuration is a legal
        thing to ask for, and what stopped it is the rows -- the difference between "go and look at
        the flow" and "go and look at the data". The column is the only thing that identifies where
        the problem is, since no field of this tool ever names a melted column.
        """
        with pytest.raises(DataError) as caught:
            run_tool(
                Unpivot(),
                Ports({"rows": _mixed_block()}),
                {"by": ["code"], "names_to": "period", "values_to": "amount", "values_type": "Float64"},
            )

        message = str(caught.value)
        for word in ("comment", "ok", "Float64"):
            assert word in message, f"refusal should name {word!r}: {message}"

    def test_an_on_invalid_written_as_error_stops_the_run_exactly_as_leaving_it_out_does(self) -> None:
        """The third state of this field, and the only one nothing else here ever executes.

        There are three ways a configuration can arrive: without the field, with ``empty``, and with
        ``error``. This tool deliberately declares no default -- ``cast_columns`` declares
        ``error`` and this one cannot, or the refusal of ``on_invalid`` without ``values_type``
        could never fire -- so "absent" and "error" reach ``execute`` as genuinely different values
        and only one line decides they mean the same thing. Written as anything that merely asks
        whether the field was given, ``error`` takes the lenient path: every unconvertible cell goes
        quietly null and the run reports a warning instead of stopping.

        So the word is executed, and the refusal it gives is compared against the refusal the absent
        field gives. That pins "absent means error" as the two being the same answer, rather than as
        a template default nobody exercises.
        """
        stated: dict[str, Any] = {
            "by": ["code"],
            "names_to": "period",
            "values_to": "amount",
            "values_type": "Float64",
            "on_invalid": "error",
        }

        with pytest.raises(DataError) as caught:
            run_tool(Unpivot(), Ports({"rows": _mixed_block()}), stated)

        message = str(caught.value)
        for word in ("comment", "ok", "Float64"):
            assert word in message, f"refusal should name {word!r}: {message}"

        absent = dict(stated)
        del absent["on_invalid"]
        with pytest.raises(DataError) as without:
            run_tool(Unpivot(), Ports({"rows": _mixed_block()}), absent)

        assert message == str(without.value), "writing 'error' and leaving it out are one behaviour"
        assert caught.value.offending_rows == without.value.offending_rows
        assert caught.value.total_rows == without.value.total_rows

    def test_the_refusal_counts_the_rows_that_offend_and_names_none_that_was_already_empty(self) -> None:
        """A count in the message is only worth having if a blank cell is not in it.

        The refusal reports how many rows offended out of how many, so a reader can tell one stray
        value from a column that is text all the way down without opening the frame. A count that
        included the cells that were empty to begin with would overstate every column that allows
        blanks, and would list a row whose cell held nothing as an example of a value that would not
        convert -- sending the reader to look at a cell with nothing in it.

        ``comment`` holds three values and one blank, so three of four is the right answer and four
        of four is the wrong one, and the two are far enough apart to tell.
        """
        with pytest.raises(DataError) as caught:
            run_tool(
                Unpivot(),
                Ports({"rows": _mixed_block()}),
                {"by": ["code"], "names_to": "period", "values_to": "amount", "values_type": "Float64"},
            )

        assert caught.value.offending_rows == 3, "the blank was always blank and never failed to convert"
        assert caught.value.total_rows == 4
        assert caught.value.examples, "a count without an example sends the reader back to the data"
        assert all(row["comment"] is not None for row in caught.value.examples), (
            f"no row it names had an empty cell to begin with: {caught.value.examples}"
        )

    def test_empty_says_how_many_out_of_how_many_and_what_share(self) -> None:
        """A warning without a size is a warning nobody can act on.

        Three unreadable values out of four here, and three out of forty-six thousand in the case
        this is built for: the same note, and the reader decides. The numbers are asserted rather
        than the fact that something was said, because a note that says "some values were emptied"
        is the silent null this library refuses, with a sentence in front of it.

        One note, not three: the two columns of figures converted cleanly, and a clean conversion is
        not news or every run would be a warning.
        """
        result = run_tool(Unpivot(), Ports({"rows": _mixed_block()}), TYPED)

        assert len(result.notes) == 1, f"only the text column lost anything: {result.notes}"
        note = result.notes[0]
        assert "3 of 4" in note, f"how many, out of how many: {note}"
        assert "75" in note, f"and what share: {note}"
        assert "comment" in note, f"and which column: {note}"
        assert "Float64" in note, f"and what was asked of it: {note}"

    def test_two_columns_that_lose_values_each_get_their_own_note(self) -> None:
        """One note per melted column that lost something, and every such column gets one.

        The rule cuts both ways and only one half is pinned above: a clean column says nothing, and
        a column that lost something says so. An implementation that reported only the first loser
        would pass every other test in this file, because every other frame here has exactly one
        unreadable column -- and in the shape this field exists for, several text columns sit beside
        the block of periods, so the run would be told about one of them and not about the rest.

        The two losers lose different amounts on purpose, so the notes cannot be swapped without
        showing, and they are asserted in the frame's own column order, which is the order the
        columns are melted in and therefore the order the notes come in.
        """
        frame = pl.DataFrame(
            {
                "code": ["a", "b", "c", "d"],
                "p1": [1.0, 2.0, 3.0, 4.0],
                "comment": ["ok", "check", "seen", None],
                "label": ["x", "y", None, "1.5"],
            },
            schema={"code": pl.String, "p1": pl.Float64, "comment": pl.String, "label": pl.String},
        )

        result = run_tool(Unpivot(), Ports({"rows": frame}), TYPED)

        assert len(result.notes) == 2, f"both losing columns have to be reported: {result.notes}"
        assert "comment" in result.notes[0], f"the first melted loser comes first: {result.notes}"
        assert "3 of 4" in result.notes[0], f"how many, out of how many: {result.notes[0]}"
        assert "label" in result.notes[1], f"and the second one after it: {result.notes}"
        assert "2 of 4" in result.notes[1], f"a different count, so the two cannot be swapped: {result.notes[1]}"
        assert "p1" not in "".join(result.notes), "the column that converted cleanly is not news"

    def test_a_cell_that_was_already_empty_is_not_counted_as_one_that_failed(self) -> None:
        """Counting it would report a problem that is not there, on every column that allows blanks.

        This is what makes the count trustworthy rather than merely present, and it is why the
        count has to be taken before the values are replaced: afterwards, a null that failed to
        convert and a null that was always there are the same null.
        """
        frame = pl.DataFrame(
            {"code": ["a", "b"], "p1": [1.0, 2.0], "text": ["3.5", None]},
            schema={"code": pl.String, "p1": pl.Float64, "text": pl.String},
        )

        result = run_tool(Unpivot(), Ports({"rows": frame}), TYPED)

        assert result.notes == (), "every value that was there converted; the blank was always blank"
        assert result["rows"]["amount"].to_list() == [1.0, 2.0, 3.5, None]


def _typed_melt_refusal_cases() -> list[Any]:
    """(config, the words its refusal must carry) for each way a stated melt can be written wrong."""
    return [
        pytest.param(
            {**BASE, "on_invalid": "set_null"},
            ("on_invalid", "values_type"),
            id="an_on_invalid_with_no_values_type_beside_it",
        ),
        pytest.param(
            {**BASE, "on_invalid": "error"},
            ("on_invalid", "values_type"),
            id="an_on_invalid_of_error_with_no_values_type_beside_it",
        ),
        pytest.param(
            {**BASE, "values_type": "Decimal"},
            ("values_type", "Float64", "String"),
            id="a_values_type_that_is_not_one_of_the_declared_types",
        ),
        pytest.param(
            {**BASE, "values_type": "float64"},
            ("values_type", "Float64"),
            id="a_values_type_whose_case_is_wrong",
        ),
        pytest.param(
            {**BASE, "values_type": "Float64", "on_invalid": "ignore"},
            ("on_invalid", "error", "set_null"),
            id="an_on_invalid_that_is_neither_of_the_two_answers",
        ),
        pytest.param(
            {**BASE, "values_type": 7},
            ("values_type",),
            id="a_values_type_that_is_not_text",
        ),
    ]


class TestAStatedMeltRefusals:
    """What the two new fields refuse, and when: off the schema, before a row moves, by name."""

    @pytest.mark.parametrize(("config", "named"), _typed_melt_refusal_cases())
    @pytest.mark.parametrize("height", [7, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_stated_melt_written_wrong_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: tuple[str, ...],
        height: int,
    ) -> None:
        """``on_invalid`` without ``values_type`` is the one that matters most here.

        Nothing governs it: there is no conversion for it to have an opinion about, so whichever
        word the author wrote would do exactly nothing and the run would report success. A field
        that silently does nothing is worse than a missing one, because a reader takes it as the
        answer to the question they had. Both fields are named in the refusal, and so is the fix,
        since either one of the two could be the one that was meant.

        The type name and the two answers are held to the same rule as every other refusal in this
        file: a message that says a value is wrong without saying what would have been right sends a
        reader to the source of a library they were trying not to read.

        Refused off the schema, so it holds on a frame with no rows exactly as on a full one.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(Unpivot(), Ports({"rows": frame}), config)

        message = str(caught.value)
        for word in named:
            assert word in message, f"refusal should name {word!r}: {message}"

    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_columns_that_disagree_are_still_refused_when_no_type_is_stated(self, height: int) -> None:
        """The new field opens a door; it does not remove the wall.

        A flow that has not said what its values are gets the old answer, naming both columns and
        both types, because nothing has told the tool which of the two was right. The whole value of
        ``values_type`` is that the decision is written down somewhere a reader can find it -- which
        is worth nothing if the same melt goes through silently when nobody writes it.
        """
        frame = _mixed_block().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(
                Unpivot(),
                Ports({"rows": frame}),
                {"by": ["code"], "names_to": "period", "values_to": "amount"},
            )

        message = str(caught.value)
        for word in ("p2", "comment", "Float64", "String"):
            assert word in message, f"refusal should name {word!r}: {message}"


class TestTheStatedMeltIsDeclaredAndDocumented:
    """The two fields as a definition file meets them: optional, spelled safely, and demonstrated."""

    def test_the_template_declares_two_optional_fields_a_definition_file_survives(self) -> None:
        """Optional at both ends, and neither the names nor the answers are words YAML rewrites.

        ``values_type`` has no default because a melt that states no type is the ordinary one, and
        the old behaviour has to be exactly what a flow that says nothing still gets.

        ``on_invalid`` has no default either, and that is the less obvious half. ``error`` is the
        behaviour when it is not written, but it cannot be *filled in* as ``error``, because a
        filled configuration is what the cross-field rule reads: a default would make every
        configuration look as though its author had written the field, and the refusal of
        ``on_invalid`` without ``values_type`` could then never fire at all.

        The vocabulary is the one ``cast_columns`` publishes, in the same spelling and the same
        order. Two spellings of one vocabulary is how they drift.
        """
        template = Unpivot().config_template()

        assert set(template) == {"by", "names_to", "values_to", "values_type", "on_invalid", "exclude_nulls"}
        assert not set(template) & YAML_REWRITES, "a field a definition file rewrites is not a field"
        assert expression_fields(template) == (), "every field here names a column or a type; none computes one"

        values_type = template["values_type"]
        assert values_type.type == "string"
        assert values_type.required is False, "a melt that states no type is the ordinary one"
        assert values_type.default is None, "saying nothing has to give exactly the old behaviour"
        assert tuple(values_type.enum or ()) == tuple(DECLARED_TYPES), "one vocabulary, one spelling"

        on_invalid = template["on_invalid"]
        assert on_invalid.type == "string"
        assert on_invalid.required is False
        assert on_invalid.default is None, "a filled 'error' would stop the cross-field refusal ever firing"
        assert set(on_invalid.enum or ()) == {"error", "set_null"}, "'set_null' rather than 'null', as everywhere else"

        answers = set(values_type.enum or ()) | set(on_invalid.enum or ())
        assert not answers & YAML_UNSAFE_VALUES, "an answer a definition file rewrites can never arrive"

    def test_a_worked_example_states_a_type_so_the_catalog_runs_one(self) -> None:
        """Conformance executes every example, so the feature is documented by something checked.

        The example has to melt a set that genuinely disagrees. One that states a type where the
        columns already match demonstrates the field without demonstrating the reason for it, and
        would still pass if the tool ignored ``values_type`` entirely.
        """
        tool = Unpivot()

        check_tool(tool)

        stated = [example for example in tool.help().examples if "values_type" in example.configuration]
        assert stated, "the field a flow reaches for needs an example a flow can copy"

        example = stated[0]
        index = set(example.configuration["by"])
        melted = {dtype for name, dtype in example.inputs["rows"].schema.items() if name not in index}
        assert len(melted) > 1, "an example whose columns already agree shows nothing about this field"
        assert (
            example.outputs["rows"].schema[str(example.configuration["values_to"])]
            == DECLARED_DTYPE[str(example.configuration["values_type"])]
        ), "the documented output must carry the type the example says it states"


class TestAFrameWithNoRows:
    """No rows in means no rows out, and the schema a later step compiles against either way."""

    def test_a_frame_with_no_rows_gives_the_whole_schema_and_no_rows(self) -> None:
        """The columns to melt come from the schema, not from the data, so they are all still known.

        This is what the refusals being read off the schema buys. A tool that read its column list
        out of the rows would hand back an empty frame carrying only the index columns, and the
        step after it would fail naming a column nobody had deleted.
        """
        empty = _rows().head(0)

        result = run_tool(Unpivot(), Ports({"rows": empty}), BASE)

        assert result["rows"].height == 0
        assert dict(result["rows"].schema) == {
            "code": pl.String,
            "site": pl.Int64,
            "label": pl.String,
            "period": pl.String,
            "amount": pl.Float64,
        }

    def test_a_stated_melt_over_a_frame_with_no_rows_gives_the_stated_schema_and_no_rows(self) -> None:
        """Every refusal is doubled at no rows; the success this field exists for was not.

        A stated type is read off the schema like everything else here, so the day a filter upstream
        matches nothing the step still has to hand on a value column at the type the flow stated --
        not at the melted columns' own types, which is what the frame carries and is exactly what
        this configuration is there to override. ``comment`` is text in the input, and a melt that
        quietly kept that would give a later step a column of text where it was told to expect
        figures, on the one run where nothing looks wrong because nothing is there.

        Nothing is reported either: a column with no rows lost no values, and a warning on an empty
        frame is a warning about nothing.
        """
        empty = _mixed_block().head(0)

        result = run_tool(Unpivot(), Ports({"rows": empty}), TYPED)

        assert result["rows"].height == 0
        assert dict(result["rows"].schema) == {
            "code": pl.String,
            "period": pl.String,
            "amount": pl.Float64,
        }, "the value column carries the stated type, not the melted columns' own"
        assert result.notes == (), "no rows means no values lost"


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way an unpivot can be configured wrong."""
    return [
        pytest.param({**BASE, "by": []}, "config field 'by' must not be empty", id="an_index_of_nothing"),
        pytest.param({**BASE, "by": None}, "required config field 'by'", id="an_index_with_no_value"),
        pytest.param({**BASE, "names_to": None}, "names_to", id="an_into_with_no_value"),
        pytest.param({**BASE, "values_to": None}, "values_to", id="a_values_with_no_value"),
        pytest.param({**BASE, "names_to": ""}, "names_to", id="an_into_that_is_blank"),
        pytest.param({**BASE, "values_to": ""}, "values_to", id="a_values_that_is_blank"),
        pytest.param({**BASE, "by": ["branch"]}, "branch", id="an_index_column_that_is_not_there"),
        pytest.param(
            {**BASE, "by": ["code", "Site", "label"]},
            "Site",
            id="an_index_column_whose_case_is_wrong",
        ),
        pytest.param({**BASE, "names_to": "code"}, "code", id="an_into_that_takes_an_index_columns_name"),
        pytest.param({**BASE, "values_to": "site"}, "site", id="a_values_that_takes_an_index_columns_name"),
        pytest.param({**BASE, "values_to": "period"}, "period", id="an_into_and_a_values_with_the_same_name"),
        pytest.param({**BASE, "names_to": "p1"}, "p1", id="an_into_that_takes_a_melted_columns_name"),
        pytest.param({**BASE, "values_to": "p2"}, "p2", id="a_values_that_takes_a_melted_columns_name"),
        pytest.param(
            {**BASE, "by": ["code", "site", "label", "p2", "p1", "p10"]},
            "by",
            id="an_index_that_covers_every_column_so_nothing_is_left_to_melt",
        ),
        pytest.param({**BASE, "by": "code"}, "by", id="an_index_written_as_one_bare_name"),
        pytest.param({**BASE, "by": [7]}, "by[0]", id="an_index_entry_that_is_not_text"),
        pytest.param({**BASE, "names_to": ["period"]}, "names_to", id="an_into_written_as_a_list"),
        pytest.param({**BASE, "columns": ["p1"]}, "unknown config field", id="a_field_nobody_declared"),
    ]


class TestUnpivotRefusals:
    """What Unpivot refuses, and when: in its own error family, off the schema, before a row moves."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [7, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_configuration_the_schema_cannot_satisfy_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Every one of these has to arrive as a :class:`ConfigError`, never as an engine exception.

        A caller of this library catches one family. A column that is not there, a new column that
        takes the name of one already present, an index that leaves nothing to melt -- left to the
        query engine, each surfaces as some other library's exception type naming neither the field
        to open nor the file to fix, and one of them does not surface at all.

        The same refusal has to hold on a frame with no rows, which is what shows it was read off
        the column names rather than off the values.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(Unpivot(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_an_index_that_covers_every_column_is_refused_for_leaving_nothing_to_melt(self) -> None:
        """An unpivot that melts nothing is a configuration edited halfway, and saying so beats agreeing.

        ``cast_columns`` already reads a cast to the type a column already holds the same way, for
        the same reason: the step runs, the run reports success, and the frame downstream is the
        one that arrived. Here the cost is higher, because the frame that arrives is wide and every
        step after this one was written expecting it to be long.

        The refusal names the field the author has to change, since it is the only field that could
        have caused it -- there is no list of columns to melt to look at instead.
        """
        frame = _rows()

        with pytest.raises(ConfigError) as caught:
            run_tool(
                Unpivot(),
                Ports({"rows": frame}),
                {**BASE, "by": ["code", "site", "label", "p2", "p1", "p10"]},
            )

        message = str(caught.value)
        assert "by" in message, f"refusal should name the field to change: {message}"

    def test_the_template_names_three_columns_and_no_list_of_columns_to_melt(self) -> None:
        """Three fields name a column, and the absence of one naming the melted ones matters most.

        A field naming the columns to melt would be a one-line addition with a plausible reason
        behind it. It is redundant -- every column not in ``by`` is melted, so the list says
        nothing new and can contradict ``by`` -- and the obvious name for it is ``on``, which a
        definition file turns into the boolean ``True`` where a name should be. Both reasons live in
        prose that nobody reads while adding a field, so they are pinned here instead.

        ``names_to`` and ``values_to`` carry no default on purpose. A default publishes a column name
        nobody chose into the frame every later step reads.

        The exact set of fields is pinned in ``TestTheStatedMeltIsDeclaredAndDocumented``, alongside
        the two optional ones. What is pinned here is that none of them names the melted columns,
        under any of the names such a field would plausibly be given.
        """
        template = Unpivot().config_template()

        assert {"by", "names_to", "values_to"} <= set(template)
        assert not set(template) & {"on", "on_columns", "columns", "melt", "melt_columns", "value_columns"}, (
            "every column outside 'by' is melted, and a second list of them is a second truth"
        )
        assert expression_fields(template) == (), "every field here names a column; none computes one"
        assert not set(template) & YAML_REWRITES, "a field a definition file rewrites is not a field"

        assert template["by"].type == "array"
        assert template["by"].required is True
        assert template["by"].non_empty is True, "an empty list is a value, so 'required' alone accepts it"
        element = template["by"].items
        assert element is not None, "an array with no declared element type checks nothing"
        assert element.type == "string"

        for name in ("names_to", "values_to"):
            assert template[name].type == "string", f"{name} names one new column"
            assert template[name].required is True
            assert template[name].non_empty is True, f"a blank {name} is not a column name"
            assert template[name].default is None, f"there is no name to assume for {name}"


class TestUnpivotAndCrossTabAreInverses:
    """The clearest statement of what this tool is: run one and then the other and nothing moved.

    The library had long-to-wide and no way back, and that asymmetry is the whole justification for
    adding a tool at all. So the claim is not merely that Unpivot reshapes a frame -- it is that it
    reshapes it back. A round trip is the only test that says so, because every other test here
    compares the output against a frame written by the same hand that wrote the input.
    """

    def test_a_cross_tab_followed_by_an_unpivot_gives_back_the_long_frame(self) -> None:
        """Long, wide, long again. The content at the end is the content at the start.

        Sorted before comparing, and deliberately: CrossTab hands its rows back in the order each
        key first appeared, and Unpivot hands its rows back column-major, so the two orders have no
        reason to agree and neither is wrong. What must agree is every row and every type.

        Every cell of the wide frame is filled, which the fixture is built for. A key with no row in
        some period gives a null cell, and a null cell unpivots to a row holding nothing rather than
        to no row at all -- that is the right answer, and it is a different claim, pinned above.
        """
        original = pl.DataFrame(
            {
                "code": ["a", "a", "b", "b"],
                "period": ["p1", "p2", "p1", "p2"],
                "amount": [1.0, 2.0, 3.0, 4.0],
            },
            schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
        )

        wide = run_tool(
            CrossTab(),
            Ports({"rows": original}),
            {"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )
        assert wide["rows"].columns == ["code", "p1", "p2"], "the middle of the round trip has to be wide"

        back = run_tool(
            Unpivot(),
            Ports({"rows": wide["rows"]}),
            {"by": ["code"], "names_to": "period", "values_to": "amount"},
        )

        assert_frame_equal(back["rows"].sort("code", "period"), original.sort("code", "period"))


class TestUnpivotInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_unpivot_conforms_and_an_example_pins_the_row_order(self) -> None:
        """Conformance runs every worked example, so a documented unpivot is a checked one.

        The extra demand is that some example melt more than one column and opt into the row-order
        check. Column-major order is a published ruling, and an example that melts one column, or
        that leaves ``check_row_order`` at its default, demonstrates the reshape without saying
        anything at all about the decision a reader most needs checked before they trust it.
        """
        tool = Unpivot()

        check_tool(tool)

        assert tool.name == "unpivot"
        assert tool.category == "transform", "the frame changes shape, and what a row means changes with it"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert examples, "a tool nobody can copy from is a tool nobody uses"
        assert any(
            example.check_row_order and example.outputs["rows"][str(example.configuration["names_to"])].n_unique() > 1
            for example in examples
        ), "an example must melt more than one column and pin the order they come back in"

    def test_unpivot_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "unpivot" in REGISTRY
        assert isinstance(REGISTRY.get("unpivot"), Unpivot)


# ---------------------------------------------------------------------------
# exclude_nulls: leave out the cells that hold nothing, as SQL's UNPIVOT does
# ---------------------------------------------------------------------------

#: Melt every column outside ``record_id`` into a name and a value. This is the shape a flow uses
#: to find columns by their heading in a sheet whose heading row moves from tab to tab.
CELLS: dict[str, Any] = {"by": ["record_id"], "names_to": "column", "values_to": "value"}

#: The same melt with the option switched on.
CELLS_EXCLUDING_NULLS: dict[str, Any] = {**CELLS, "exclude_nulls": True}


def _sheet_as_text() -> pl.DataFrame:
    """A sheet read with every cell as text, the shape the workbook reader hands over.

    Each edge of the rule is in it, and each is a cell somebody might call "empty":

    - ``None``: a blank cell. The workbook reader reads a blank as null, never as ``""``. This is
      the only kind of cell the option leaves out.
    - ``""``: an empty string. It is a value, so it is kept.
    - ``"   "``: text that is only spaces. Somebody typed it, so it is kept.
    - ``"0"``: a zero written as text. Kept. It catches a tool that tests truthiness.

    ``record_id`` is null on one row. An index cell is not a melted cell, so that row's values must
    all survive. Otherwise the option would drop rows for a reason nobody asked for.

    The columns are not in alphabetical order (``c3``, ``c1``, ``c2``), so a tool that sorted its
    output would fail the order checks.
    """
    return pl.DataFrame(
        {
            "record_id": [1, 2, None, 4],
            "c3": ["Heading", None, "", "x"],
            "c1": [None, "   ", "0", None],
            "c2": [None, None, "Total", "7.5"],
        },
        schema={"record_id": pl.Int64, "c3": pl.String, "c1": pl.String, "c2": pl.String},
    )


def _figures() -> pl.DataFrame:
    """A block of figures holding a null, a not-a-number, a zero and a negative zero.

    Only the null is left out. Not-a-number is a float value, the result of ``0.0 / 0.0``, and
    SQL's UNPIVOT keeps it: ``EXCLUDE NULLS`` means NULL and nothing else. Dropping it would also
    hide a fault upstream. Zero is a measurement, and so is negative zero.
    """
    return pl.DataFrame(
        {
            "record_id": [1, 2, 3],
            "q2": [float("nan"), None, 0.0],
            "q1": [None, -0.0, 4.25],
        },
        schema={"record_id": pl.Int64, "q2": pl.Float64, "q1": pl.Float64},
    )


def _melt(frame: pl.DataFrame, config: Mapping[str, Any]) -> Ports:
    return run_tool(Unpivot(), Ports({"rows": frame}), config)


class TestExcludeNullsLeavesOutCellsHoldingNothing:
    """``exclude_nulls: true`` leaves out a melted cell that is null. Nothing else is left out.

    **Why the option exists.** A flow finds columns by their heading in a workbook whose heading row
    sits in a different place on each tab. It melts every cell into a row and keeps the rows that
    hold something. On a real report that is about 104 million rows. About 88 million of them are
    blank, and the run needs about 20 GB. The blank rows are built only to be thrown away.

    **The rule is one sentence: a null cell is left out, and every other cell comes through.** That
    is what SQL's ``UNPIVOT`` does by default (``EXCLUDE NULLS``). This tool follows the SQL
    operation it is named after, so:

    - null is left out;
    - an empty string ``""`` is kept. SQL keeps ``''``, and an empty string is a value. The
      workbook reader already reads a blank cell as null, so the blanks this option is for are
      nulls already;
    - text that is only spaces is kept. Somebody typed it;
    - not-a-number in a float column is kept. It is a value in SQL too. It usually means a
      division went wrong upstream, and a reader needs to see that, not lose it.

    A flow that wants ``""`` treated as nothing turns it into null in a step before this one, where
    a reader can see it. A second setting for "what counts as empty" would be a second rule to
    learn, and a place for two readers to disagree.

    **Off by default.** Without the option every cell becomes a row, nulls included, as before.
    ``test_a_cell_holding_nothing_becomes_a_row_holding_nothing`` still holds unchanged.
    """

    def test_the_option_leaves_out_null_cells_and_nothing_else(self) -> None:
        """The whole contract on one frame, with the expected output written out row by row.

        Column-major in the frame's own order (c3, then c1, then c2), with every null cell gone.
        The ``""``, the spaces and the ``"0"`` all survive. So does every value on the row whose
        ``record_id`` is null, because an index cell is not a melted cell.
        """
        result = _melt(_sheet_as_text(), CELLS_EXCLUDING_NULLS)

        expected = pl.DataFrame(
            {
                "record_id": [1, None, 4, 2, None, None, 4],
                "column": ["c3", "c3", "c3", "c1", "c1", "c2", "c2"],
                "value": ["Heading", "", "x", "   ", "0", "Total", "7.5"],
            },
            schema={"record_id": pl.Int64, "column": pl.String, "value": pl.String},
        )
        assert_frame_equal(result["rows"], expected, check_row_order=True, check_exact=True)

    def test_survivors_keep_the_order_the_full_melt_gives_them(self) -> None:
        """The output is today's output with the null rows taken out, in the same order.

        This is the property that matters downstream. A flow that takes the first row holding a
        heading gets the same heading with the option on as with it off. A tool that sorted the
        survivors, or grouped them by record instead of by column, would change which heading comes
        first, and no figure would look wrong.

        Checked on the text block and on the figures. The filter here is ``is_not_null``, which
        keeps not-a-number, so on the figures this also pins that not-a-number survives.
        """
        for frame in (_sheet_as_text(), _figures()):
            full = _melt(frame, CELLS)["rows"]
            excluded = _melt(frame, CELLS_EXCLUDING_NULLS)["rows"]

            expected = full.filter(pl.col("value").is_not_null())
            assert_frame_equal(excluded, expected, check_row_order=True, check_exact=True)
            assert excluded.height < full.height, "the fixture has null cells, so some rows must go"

    def test_not_a_number_and_zero_are_values_and_are_kept(self) -> None:
        """Not-a-number, zero and negative zero all come through. Only the null goes.

        Spelled out on its own because not-a-number is the case most likely to be "fixed" later.
        In the query engine ``is_null`` and ``is_nan`` are different questions, and SQL's
        ``EXCLUDE NULLS`` asks only the first.
        """
        result = _melt(_figures(), CELLS_EXCLUDING_NULLS)["rows"]

        assert result["column"].to_list() == ["q2", "q2", "q1", "q1"]
        assert result["record_id"].to_list() == [1, 3, 2, 3]
        values = result["value"].to_list()
        assert math.isnan(values[0]), "not-a-number is a value and is kept"
        assert values[1] == 0.0
        assert values[2] == 0.0 and math.copysign(1.0, values[2]) == -1.0, "negative zero arrives as itself"
        assert values[3] == 4.25
        assert result["value"].null_count() == 0

    def test_the_column_types_are_the_ones_the_full_melt_gives(self) -> None:
        """Leaving rows out changes no type: not the index, not the name column, not the values."""
        for frame in (_sheet_as_text(), _figures(), _rows()):
            index = ["record_id"] if "record_id" in frame.columns else BASE["by"]
            config = {**CELLS, "by": index}
            full = _melt(frame, config)["rows"]
            excluded = _melt(frame, {**config, "exclude_nulls": True})["rows"]

            assert excluded.schema == full.schema

    def test_a_column_with_no_type_at_all_contributes_no_rows(self) -> None:
        """A column declared ``Null`` holds nothing, so all of its rows are left out.

        The value column still takes the other columns' type. That is Union's ruling, and it is the
        same with the option on as off.
        """
        result = _melt(
            _one_null_column(),
            {"by": ["code"], "names_to": "measure", "values_to": "amount", "exclude_nulls": True},
        )

        expected = pl.DataFrame(
            {"code": ["a", "b"], "measure": ["opening", "opening"], "amount": [1.0, 2.0]},
            schema={"code": pl.String, "measure": pl.String, "amount": pl.Float64},
        )
        assert_frame_equal(result["rows"], expected, check_row_order=True)

    def test_a_frame_whose_every_melted_cell_is_null_gives_no_rows_and_the_full_schema(self) -> None:
        """No cell holds anything, so no row comes out, and the schema is the full melt's.

        A step after this one still has columns to compile against. Checked three ways: typed
        columns holding only nulls, columns declared ``Null``, and a frame with no rows.
        """
        typed_but_empty = pl.DataFrame(
            {"record_id": [1, 2], "c1": [None, None], "c2": [None, None]},
            schema={"record_id": pl.Int64, "c1": pl.String, "c2": pl.String},
        )
        no_type_at_all = pl.DataFrame(
            {"record_id": [1, 2], "c1": [None, None], "c2": [None, None]},
            schema={"record_id": pl.Int64, "c1": pl.Null, "c2": pl.Null},
        )
        no_rows = _sheet_as_text().head(0)

        for frame in (typed_but_empty, no_type_at_all, no_rows):
            full = _melt(frame, CELLS)["rows"]
            excluded = _melt(frame, CELLS_EXCLUDING_NULLS)["rows"]

            assert excluded.height == 0
            assert excluded.schema == full.schema

        assert dict(_melt(typed_but_empty, CELLS_EXCLUDING_NULLS)["rows"].schema) == {
            "record_id": pl.Int64,
            "column": pl.String,
            "value": pl.String,
        }

    def test_leaving_the_option_out_or_writing_false_gives_exactly_todays_output(self) -> None:
        """Off is the old behaviour: every value, every row, every type, every note.

        Checked on plain melts and on a melt with a stated type that empties values. That path also
        carries notes and a per-row status, and both must be unchanged too.
        """
        for frame, config in ((_rows(), BASE), (_sheet_as_text(), CELLS), (_mixed_block(), TYPED)):
            absent = _melt(frame, config)
            written_false = _melt(frame, {**config, "exclude_nulls": False})

            assert_frame_equal(written_false["rows"], absent["rows"], check_row_order=True, check_exact=True)
            assert written_false.notes == absent.notes
            assert set(written_false.status) == set(absent.status)
            for port, status in absent.status.items():
                assert_frame_equal(written_false.status[port], status, check_row_order=True)

        # And leaving it out still gives what the original tests wrote down: 21 rows, nulls included.
        assert _melt(_rows(), BASE)["rows"].height == 21

    def test_the_option_leaves_its_input_untouched(self) -> None:
        """Leaving rows out of the output must not leave rows out of the input."""
        frame = _sheet_as_text()
        before = frame.clone()

        _melt(frame, CELLS_EXCLUDING_NULLS)

        assert_frame_equal(frame, before)


class TestExcludeNullsWithAStatedType:
    """With ``values_type`` given, a cell is judged as it arrived, before it is converted.

    A cell that held nothing is left out. A cell that held text which did not read as the stated
    type is kept, because it held something. Under ``on_invalid: set_null`` it comes through as a row
    holding nothing, with its warning mark beside it, as it does today. If the option dropped it,
    the note would count values a reader can no longer find, and the per-row status would point at
    rows that are not there.
    """

    def test_a_value_that_failed_to_convert_is_kept_and_marked_and_a_blank_is_left_out(self) -> None:
        """On the mixed block, c's blank p2 and b's blank comment go. The three failed comments stay."""
        result = _melt(_mixed_block(), {**TYPED, "exclude_nulls": True})
        rows = result["rows"]

        expected = pl.DataFrame(
            {
                "code": ["a", "b", "d"] + ["a", "b", "c", "d"] + ["a", "c", "d"],
                "period": ["p2"] * 3 + ["p1"] * 4 + ["comment"] * 3,
                "amount": [5.0, 2.0, 7.0] + [10.0, 3.0, 8.0, A_HAIR_UNDER_HALF_A_CENT] + [None, None, None],
            },
            schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
        )
        assert_frame_equal(rows, expected, check_row_order=True, check_exact=True)

        # The marks line up with the rows that are left, and blame the column each row came from.
        status = result.status["rows"]
        assert status.height == rows.height
        assert status["level"].to_list() == ["ok"] * 7 + ["warning"] * 3
        reasons = status["reason"].to_list()
        for position in range(7, 10):
            assert [entry["column"] for entry in reasons[position]] == ["comment"]

        assert len(result.notes) == 1
        assert result.notes[0].startswith("3 of "), f"three values failed to read: {result.notes[0]}"

    def test_a_value_that_will_not_convert_still_stops_the_run(self) -> None:
        """The option leaves out what was empty. It does not hide what was wrong.

        Without ``on_invalid: set_null``, text that does not read as a figure stops the run, naming
        the column, exactly as it does with the option off.
        """
        strict = {key: value for key, value in TYPED.items() if key != "on_invalid"}

        with pytest.raises(DataError) as caught:
            _melt(_mixed_block(), {**strict, "exclude_nulls": True})

        assert "comment" in str(caught.value)


#: Run in a fresh interpreter by the memory test below, once per melt. It melts a sparse frame,
#: with ``exclude_nulls`` set to the argument it is given, and prints two numbers: the rows that
#: came out, and how far the melt raised the process's peak memory above where it started.
_PEAK_MEMORY_SCRIPT = textwrap.dedent(
    """
    import sys

    import polars as pl

    from dantron_etl.authoring import Ports, run_tool
    from dantron_etl.tools.unpivot import Unpivot


    def status_bytes(key: str) -> int:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith(key + ":"):
                    return int(line.split()[1]) * 1024
        raise SystemExit(f"/proc/self/status has no {key}")


    rows, columns = 20_000, 200
    # One cell in fifty holds something, about the share a sparse report sheet has.
    frame = pl.DataFrame({"record_id": pl.int_range(rows, eager=True)}).with_columns(
        pl.when((pl.int_range(pl.len()) + i) % 50 == 0).then(pl.lit(f"v{i}")).otherwise(None).alias(f"c{i}")
        for i in range(columns)
    )
    config = {"by": ["record_id"], "names_to": "column", "values_to": "value", "exclude_nulls": sys.argv[1] == "on"}

    # Warm up on a few rows, so loading code and starting threads is not counted.
    run_tool(Unpivot(), Ports({"rows": frame.head(50)}), config)

    # Building the frame set a high-water mark of its own. Writing 5 to clear_refs resets the
    # mark to the memory in use now, so the peak read afterwards is the melt's alone.
    with open("/proc/self/clear_refs", "w") as clear:
        clear.write("5")
    start = status_bytes("VmRSS")

    melted = run_tool(Unpivot(), Ports({"rows": frame}), config)["rows"]
    print(melted.height, status_bytes("VmHWM") - start)
    """
)


def _measure_melt(exclude_nulls: bool) -> tuple[int, int]:
    """Melt the sparse frame in a fresh interpreter; give back (rows out, peak memory growth in bytes)."""
    source = Path(dantron_etl.__file__).resolve().parent.parent
    path = os.pathsep.join(part for part in (str(source), os.environ.get("PYTHONPATH", "")) if part)

    finished = subprocess.run(
        [sys.executable, "-c", _PEAK_MEMORY_SCRIPT, "on" if exclude_nulls else "off"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": path},
        timeout=300,
        check=False,
    )
    assert finished.returncode == 0, f"the measuring interpreter failed:\n{finished.stderr}"
    height, growth = (int(word) for word in finished.stdout.split())
    return height, growth


class TestExcludeNullsDoesNotBuildTheRowsItLeavesOut:
    """The option exists to save memory, so it is measured on memory, not only on its output.

    A melt that builds every row and then filters gives the right output and saves nothing: the
    blank rows are built first and thrown away after. No output check can tell the two apart, so
    this test measures peak memory.

    How it stays reliable:

    - Each melt runs in its own fresh interpreter. Nothing an earlier test allocated hides the peak,
      and the second melt cannot reuse memory the first one freed. (In one interpreter it does: a
      second full melt measured about 10 MB against 70 MB for the first.)
    - The peak is reset after the frame is built, so the frame's own construction is not counted.
      Linux allows this by writing 5 to ``/proc/self/clear_refs``; the peak is then ``VmHWM``.
    - It compares against itself, not against a fixed number. The full melt of the same frame is the
      yardstick, so the result does not depend on the machine, the allocator or the polars version.
    - It checks the yardstick moved by a clear amount. If the full melt barely raised the peak, the
      measurement cannot see anything, and the test fails rather than passing by accident.

    Measured with a prototype on this frame: the full melt raised the peak by 65-75 MB. A melt that
    builds one column at a time and keeps only the cells holding something raised it by 0.5-1.5 MB.
    The bar is a quarter of the full melt, well clear of both.
    """

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform != "linux", reason="resetting and reading the peak needs Linux's /proc")
    def test_the_option_keeps_peak_memory_well_below_a_full_melt(self) -> None:
        full_height, full_growth = _measure_melt(exclude_nulls=False)
        kept, lean_growth = _measure_melt(exclude_nulls=True)

        assert full_height == 20_000 * 200
        assert kept == 20_000 * 200 // 50, "one cell in fifty holds something"
        assert full_growth > 20_000_000, (
            f"the full melt raised peak memory by only {full_growth:,} bytes, so this measurement "
            f"cannot tell a lean melt from a full one"
        )
        assert lean_growth < full_growth / 4, (
            f"with exclude_nulls the peak grew by {lean_growth:,} bytes, against {full_growth:,} for "
            f"the full melt, so the rows it leaves out were built first and thrown away after"
        )


class TestExcludeNullsIsDeclaredAndDocumented:
    """The option as a definition file meets it: a true-or-false switch, off unless written."""

    def test_the_template_declares_an_optional_switch_that_is_off_by_default(self) -> None:
        """A boolean, not required, defaulting to false, so a flow that says nothing gets today's melt.

        A switch rather than a word. Any pair of words for this (include/exclude, keep/drop) is a
        second spelling of true and false, and true and false are what a definition file already
        writes safely for a boolean field.
        """
        field = Unpivot().config_template()["exclude_nulls"]

        assert field.type == "boolean"
        assert field.required is False
        assert field.default is False, "a flow that says nothing must get every cell, as before"

    @pytest.mark.parametrize(
        "written",
        [
            pytest.param("true", id="the_word_true_in_quotes"),
            pytest.param("yes", id="the_word_yes_in_quotes"),
            pytest.param(1, id="the_number_one"),
            pytest.param(["true"], id="a_list"),
        ],
    )
    @pytest.mark.parametrize("height", [4, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_anything_but_true_or_false_is_refused_by_name(self, written: Any, height: int) -> None:
        """Refused before a row moves, naming the field, on a full frame and an empty one alike.

        ``"true"`` in quotes is text, and ``1`` is a number. Accepting either would make a switch
        that some spellings turn on and others quietly leave off.

        The refusal must be the type refusal, not "unknown config field": a field the tool does not
        declare is refused by name too, so the name alone would not show the field exists.
        """
        frame = _sheet_as_text().head(height)

        with pytest.raises(ConfigError) as caught:
            _melt(frame, {**CELLS, "exclude_nulls": written})

        message = str(caught.value)
        assert "'exclude_nulls'" in message, f"refusal should name the field: {message}"
        assert "boolean" in message, f"refusal should say a true-or-false is wanted: {message}"
        assert "unknown config field" not in message, f"the field must be declared: {message}"

    def test_a_worked_example_shows_where_the_rule_stops(self) -> None:
        """The catalog shows the option, and its example shows both sides of the rule.

        Conformance runs every example, so this one is checked, not just written. It has to show a
        null cell left out and an empty string kept. An example holding only nulls would teach a
        reader that "empty" cells go, which is the wrong half of the rule.
        """
        tool = Unpivot()

        check_tool(tool)

        excluding = [example for example in tool.help().examples if example.configuration.get("exclude_nulls") is True]
        assert excluding, "the option a flow reaches for needs an example a flow can copy"

        example = excluding[0]
        assert example.check_row_order, "the order of the survivors is part of the contract"

        index = set(example.configuration["by"])
        frame = example.inputs["rows"]
        melted = [name for name in frame.columns if name not in index]
        values = str(example.configuration["values_to"])
        output = example.outputs["rows"]

        nulls_in = sum(frame[name].null_count() for name in melted)
        assert nulls_in > 0, "an example with no null cell shows nothing being left out"
        assert output.height == frame.height * len(melted) - nulls_in, "exactly the null cells are left out"
        assert output[values].null_count() == 0

        empty_strings_in = sum(int((frame[name] == "").sum()) for name in melted if frame.schema[name] == pl.String)
        assert empty_strings_in > 0, "the example must hold an empty string, to show it is kept"
        assert int((output[values] == "").sum()) == empty_strings_in, "and every empty string comes through"
