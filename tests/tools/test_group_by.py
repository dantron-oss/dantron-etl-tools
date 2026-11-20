"""Tests for the GroupBy tool: one row per distinct key, and only the columns asked for.

GroupBy takes one input port, ``rows``, and produces one, also ``rows``: one row per distinct
combination of the ``by`` columns, carrying the aggregations the configuration names. Seven decisions
are pinned here beyond the happy path, each because it has a silent failure mode and each because a
different answer is defensible:

**An aggregate is configuration; the thing aggregated is an expression.** The expression language is
row-wise and its whitelist deliberately excludes aggregates, so ``aggregations[].function`` is an
enum drawn from a published whitelist of aggregate names and ``aggregations[].value`` is an ordinary
expression. Widening the expression language instead would make ``sum([amount])`` legal in exactly
one config field and meaningless in every other, which is a second dialect a reader has to learn
from the position a string sits in. The cost -- a second whitelist -- is paid once here; the enum is
derived from that whitelist rather than typed out beside it, because a whitelist that fails open has
already bitten this library once.

**Group order follows first appearance.** polars promises nothing about group order unless asked,
and a report whose row order moved for no reason anybody could name is the reason this has to be a
promise rather than an accident. First appearance, not sorted: it is the order the reader of the
input already has in front of them, it needs no total order on the key (a key spanning several
columns has none), and it is stable under a filter upstream.

**A null in a ``by`` column is a group.** This agrees with RemoveDuplicates and is the *opposite*
of Join's ruling for the same-looking question, and all three are right: this tool mirrors SQL's
``GROUP BY``, which groups nulls together, RemoveDuplicates mirrors ``DISTINCT``, which does the
same, and a join mirrors ``=``, under which a null pairs with nothing. The rule a reader can carry
between them is that each tool follows the operation it is named after, and the alternative here --
dropping null-keyed rows -- would silently delete money from a total.

**A sum of nothing is null, not zero.** polars sums an all-null group to ``0``. In a pipeline that
carries figures, zero is a claim ("this group is worth nothing") and null is the absence of one
("nothing here to add up"); manufacturing the first from the second puts a number nobody measured
into a report, and no downstream assertion can tell it from a real zero afterwards. Nulls are still
skipped within a group that has any value at all, so ``[10, null]`` sums to ``10`` -- which is what
makes "all null" the only case that produces null, and therefore meaningful.

**``count`` counts rows; ``count_non_null`` counts the values that are there.** Both are wanted and
they differ, so they get two names rather than one name and a flag: ``count`` takes no ``value`` at
all (a row count cannot depend on a column), and ``count_non_null`` requires one and ignores its
nulls. The trap this closes is a legacy one -- the dataframe library the earlier tools were written
against spells the non-null count ``count``, so an author porting a step gets the other number
without being told.

**Every column that is neither grouped nor aggregated disappears.** It is the most common surprise
with this tool, so it is pinned; the way to carry a column that is constant within a group is
``min`` or ``max`` over it, which is why those two ship.

**``join_text`` collapses a group's text into one string, and every question it raises is answered
here rather than left to the engine.** It is SQL's ``string_agg``: the values in the order the rows
arrived (so a flow that wants another order puts a Sort in front, as everywhere else here), joined
by a separator that defaults to ``", "``; missing values skipped, so ``["a", null, "b"]`` is
``"a, b"`` and not ``"a, , b"``; a group whose values are *all* missing giving the empty string,
which is the *opposite* of the ruling ``sum`` makes two paragraphs up and therefore has to be
written down where a reader looking for it will find it -- the two differ because an empty string is
what joining nothing produces, while a zero is a measured amount nobody measured. It takes a
``separator`` of its own, and that field is refused beside any other aggregate rather than ignored:
a separator that silently does nothing is the class of mistake this library refuses everywhere, and
it is refused the same way Join refuses ``keys`` beside ``left_keys``. Its value has to be text,
refused on the schema, because a number or a date joined into a string is an author error that
otherwise ships as a plausible-looking string of digits.

Vocabulary is deliberately generic -- code, region, amount, qty, term -- because the library carries
no caller's nouns; these are the shapes of three real aggregation steps, not their names.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import CATEGORIES, ConfigError, DataError, Ports, REGISTRY, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.cross_tab import CrossTab
from dantron_etl.tools.group_by import AGGREGATE_FUNCTIONS, GroupBy

# ---------------------------------------------------------------------------
# Test data -- the shape the real aggregation steps run against
# ---------------------------------------------------------------------------


def _rows() -> pl.DataFrame:
    """One frame carrying every shape the real aggregation steps need.

    Three groups under ``code``, in first-appearance order A, B, C. Group C holds a single row whose
    ``amount`` is missing, which is the all-null group; group B holds two different ``term`` values,
    which is the group a distinct count exists to find.
    """
    return pl.DataFrame(
        {
            "code": ["A", "B", "A", "C", "B"],
            "region": ["x", "y", "x", "x", "y"],
            "amount": [10.0, 20.0, 5.0, None, 2.5],
            "qty": [2, 1, 3, 4, 1],
            "term": ["12", "6", "12", None, "12"],
        }
    )


def _grouped(frame: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """``frame`` grouped through the boundary, with the single output port unwrapped."""
    result = run_tool(GroupBy(), Ports({"rows": frame}), config)
    assert result.names == ("rows",), "GroupBy declares one input and one output, both named 'rows'"
    return result["rows"]


def _aggregation(
    name: str,
    function: str,
    value: str | None = None,
    *,
    separator: str | None = None,
) -> dict[str, Any]:
    """One element of the ``aggregations`` array, with ``value`` and ``separator`` left out where there is none.

    Both optional fields are omitted rather than written as null, because "nobody configured this"
    is the state the tool reads, and a helper that always wrote the key could not express it. The
    empty string is a configured separator and survives that, which is what ``is not None`` buys
    over a plain truth test.
    """
    element: dict[str, Any] = {"name": name, "function": function}
    if value is not None:
        element["value"] = value
    if separator is not None:
        element["separator"] = separator
    return element


def _notes() -> pl.DataFrame:
    """The frame the text-joining tests run against: three groups, and every shape a join meets.

    Group A holds three notes whose arrival order (beta, alpha, gamma) differs from their sorted
    order, which is what makes "input order" a checkable claim rather than a coincidence. Group B
    holds one note, which is the group that must come back without a separator anywhere in it. Group
    C holds one row whose note is missing, which is the all-missing group. ``amount``, ``qty`` and
    ``posted`` are here to be refused: each is a value expression a text join must not accept.
    """
    return pl.DataFrame(
        {
            "code": ["A", "B", "A", "C", "A"],
            "note": ["beta", "solo", "alpha", None, "gamma"],
            "amount": [10.0, 20.0, 5.0, None, 2.5],
            "qty": [2, 1, 3, 4, 1],
            "posted": [
                date(2024, 1, 31),
                date(2024, 2, 29),
                date(2024, 1, 15),
                date(2024, 3, 1),
                date(2024, 1, 2),
            ],
        }
    )


def _grouping_cases() -> list[Any]:
    """(config, the frame it must produce) for one case per aggregate that ships."""
    return [
        pytest.param(
            {
                "by": ["code"],
                "aggregations": [
                    _aggregation("total", "sum", "[amount]"),
                    _aggregation("lines", "count"),
                ],
            },
            pl.DataFrame(
                {
                    "code": ["A", "B", "C"],
                    "total": [15.0, 22.5, None],
                    "lines": [2, 2, 1],
                }
            ),
            id="a_per_group_total_and_a_row_count",
        ),
        pytest.param(
            {
                "by": ["region", "code"],
                "aggregations": [_aggregation("total", "sum", "[amount]")],
            },
            pl.DataFrame(
                {
                    "region": ["x", "y", "x"],
                    "code": ["A", "B", "C"],
                    "total": [15.0, 22.5, None],
                }
            ),
            id="a_key_spanning_two_columns_keeps_them_in_config_order",
        ),
        pytest.param(
            {
                "by": ["code"],
                "aggregations": [_aggregation("total", "sum", "[amount] * [qty]")],
            },
            pl.DataFrame({"code": ["A", "B", "C"], "total": [35.0, 22.5, None]}),
            id="an_aggregate_over_an_expression_rather_than_a_bare_column",
        ),
        pytest.param(
            {
                "by": ["code"],
                "aggregations": [
                    _aggregation("terms", "count_distinct", "[term]"),
                    _aggregation("figures", "count_non_null", "[amount]"),
                ],
            },
            pl.DataFrame({"code": ["A", "B", "C"], "terms": [1, 2, 0], "figures": [2, 2, 0]}),
            id="a_distinct_count_finds_the_group_with_two_different_terms",
        ),
        pytest.param(
            {
                "by": ["code"],
                "aggregations": [
                    _aggregation("smallest", "min", "[amount]"),
                    _aggregation("largest", "max", "[qty]"),
                ],
            },
            pl.DataFrame(
                {
                    "code": ["A", "B", "C"],
                    "smallest": [5.0, 2.5, None],
                    "largest": [3, 1, 4],
                }
            ),
            id="min_and_max_are_how_a_column_constant_within_a_group_is_carried_forward",
        ),
    ]


# ---------------------------------------------------------------------------
# Grouping: one row per key, the columns asked for, in the order promised
# ---------------------------------------------------------------------------


class TestGroupByGrouping:
    """What each group's row holds, which columns survive, and the order the rows come back in."""

    @pytest.mark.parametrize(("config", "expected"), _grouping_cases())
    def test_a_group_is_one_row_per_distinct_key_with_the_aggregations_asked_for(
        self,
        config: Mapping[str, Any],
        expected: pl.DataFrame,
    ) -> None:
        """Values, dtypes, column order and row order, all pinned by one comparison.

        The dtype half is not incidental: a count that came back as an unsigned integer underflows
        the moment a later step subtracts one count from another, which is the same reason DateDiff
        counts into a signed column.
        """
        result = _grouped(_rows(), config)

        assert_frame_equal(result, expected, check_row_order=True)

    def test_group_order_follows_the_first_appearance_of_each_key(self) -> None:
        """The promise is first appearance, and the frame below sorts differently from that.

        polars makes no promise about group order unless it is asked for one, and a report whose row
        order moved for no reason anybody could name is what makes this a promise rather than an
        accident. Sorted order is deliberately not the promise: a key spanning several columns has no
        obvious total order, and first appearance is the order the reader of the input already holds.
        """
        frame = pl.DataFrame({"code": ["b", "a", "b", "c", "a"], "amount": [1.0, 2.0, 3.0, 4.0, 5.0]})
        config = {"by": ["code"], "aggregations": [_aggregation("total", "sum", "[amount]")]}

        result = _grouped(frame, config)

        assert result["code"].to_list() == ["b", "a", "c"], "first appearance, not sorted"
        assert result["total"].to_list() == [4.0, 7.0, 4.0]

    def test_only_grouped_and_aggregated_columns_survive_and_a_repeated_key_is_counted_once(self) -> None:
        """Everything else disappears -- the most common surprise with this tool.

        A repeated ``by`` column is counted once rather than refused, matching RemoveDuplicates,
        which says the same about a column named twice in its key.
        """
        config = {
            "by": ["code", "code"],
            "aggregations": [_aggregation("total", "sum", "[amount]")],
        }

        result = _grouped(_rows(), config)

        assert result.columns == ["code", "total"], "region, qty and term are gone; code appears once"
        assert result.height == 3

    def test_a_null_in_a_by_column_is_a_group_of_its_own(self) -> None:
        """Null-keyed rows group together, as GROUP BY does -- they are never dropped.

        Dropping them would delete their figures from every total downstream, with nothing left in
        the output to say a row had ever been there. Two null-keyed rows are one group, not two,
        which is the same ruling RemoveDuplicates makes and the opposite of Join's.
        """
        frame = pl.DataFrame({"code": ["A", None, "A", None], "amount": [1.0, 2.0, 3.0, 4.0]})
        aggregations = [_aggregation("total", "sum", "[amount]"), _aggregation("n", "count")]
        config = {"by": ["code"], "aggregations": aggregations}

        result = _grouped(frame, config)

        assert result["code"].to_list() == ["A", None]
        assert result["total"].to_list() == [4.0, 6.0], "the null group keeps its money"
        assert result["n"].to_list() == [2, 2], "two null keys are one group of two rows"
        assert result["total"].sum() == frame["amount"].sum(), "no figure may leave the pipeline"


# ---------------------------------------------------------------------------
# The aggregates themselves: what each one means when values are missing
# ---------------------------------------------------------------------------


class TestGroupByAggregates:
    """The published whitelist, and the null behaviour of each aggregate that ships."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param([10.0, 2.5], 12.5, id="every_value_present"),
            pytest.param([10.0, None], 10.0, id="a_missing_value_is_skipped"),
            pytest.param([None, None], None, id="nothing_to_add_up_is_null_not_zero"),
        ],
    )
    def test_a_sum_skips_missing_values_but_a_group_of_only_nulls_sums_to_null(
        self,
        values: Sequence[float | None],
        expected: float | None,
    ) -> None:
        """Zero and "no figure" are different claims, and only one of them is measured.

        polars sums an all-null group to 0. Passing that on would put a number nobody measured into
        a report, and no assertion downstream could tell it from a real zero -- while the mixed case
        must still skip its nulls, or a single missing line would void a whole group's total.
        """
        frame = pl.DataFrame({"code": ["A"] * len(values), "amount": values}, schema_overrides={"amount": pl.Float64})

        result = _grouped(frame, {"by": ["code"], "aggregations": [_aggregation("total", "sum", "[amount]")]})

        assert result["total"].to_list() == [expected]
        assert result["total"].dtype == pl.Float64, "a sum keeps the dtype of what it added"

    def test_count_counts_rows_and_count_non_null_counts_the_values_that_are_there(self) -> None:
        """Two names for two numbers, so neither can be mistaken for the other.

        The legacy dataframe library spells the non-null count ``count``, so an author porting a
        step from it gets the other number silently. Here ``count`` is the row count and takes no
        ``value``; ``count_non_null`` requires one and ignores its nulls.
        """
        frame = pl.DataFrame({"code": ["A", "A", "A"], "amount": [1.0, None, None]})
        config = {
            "by": ["code"],
            "aggregations": [
                _aggregation("rows_in_group", "count"),
                _aggregation("figures_present", "count_non_null", "[amount]"),
            ],
        }

        result = _grouped(frame, config)

        assert result["rows_in_group"].to_list() == [3], "every row, missing figures included"
        assert result["figures_present"].to_list() == [1], "only the rows carrying a figure"
        for name in ("rows_in_group", "figures_present"):
            assert result[name].dtype == pl.Int64, "a count is signed, so subtracting two cannot underflow"

    def test_count_distinct_skips_a_missing_value_like_sql(self) -> None:
        """``count_distinct`` follows SQL's ``COUNT(DISTINCT v)``: a missing value is not a value.

        The owner's ruling. A group whose values are all missing counts 0, a group holding one value
        and one blank counts 1, and a group with no blank is untouched. (It was ``n_unique``, which
        counted a blank as one more value: 1, 2 and 1 here.)
        """
        frame = pl.DataFrame(
            {
                "code": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "term": [None, None, None, "12", "12", "12", "12", "6"],
            }
        )

        result = _grouped(frame, {"by": ["code"], "aggregations": [_aggregation("terms", "count_distinct", "[term]")]})

        assert result["terms"].to_list() == [0, 1, 1, 2], "all missing is none; a blank beside a value is not a second"
        assert result["terms"].dtype == pl.Int64


class TestFirstAndFirstNonNull:
    """``first`` is the first row's value as the rows arrive, null included (the owner's ruling).

    ``first_non_null`` is the first value that is there. A group whose first row is blank tells the
    two apart; a group with no blank cannot.
    """

    frame = pl.DataFrame({"code": ["A", "A", "A", "B", "B", "C"], "term": [None, "12", "6", "3", None, None]})

    def test_first_keeps_a_blank_first_row_and_first_non_null_skips_it(self) -> None:
        config = {
            "by": ["code"],
            "aggregations": [
                _aggregation("first_term", "first", "[term]"),
                _aggregation("first_known", "first_non_null", "[term]"),
            ],
        }

        result = _grouped(self.frame, config)

        assert result["first_term"].to_list() == [None, "3", None]
        assert result["first_known"].to_list() == ["12", "3", None], "a group of blanks has no first value"


# ---------------------------------------------------------------------------
# join_text: a group's text as one string, and what it does with what is missing
# ---------------------------------------------------------------------------


class TestGroupByJoinsText:
    """SQL's ``string_agg``: the separator, the order, and what a missing value does to both."""

    def test_a_group_becomes_one_string_of_its_values_joined_by_the_separator(self) -> None:
        """Values, separator, dtype, column order and row order, pinned by one comparison.

        A group of one value is the half worth reading twice: it comes back as that value with no
        separator anywhere near it, because a separator goes *between* values and one value has no
        between. A count beside the join is what a reader uses to tell "one note" from "one note and
        four blanks".
        """
        config = {
            "by": ["code"],
            "aggregations": [
                _aggregation("notes", "join_text", "[note]", separator="; "),
                _aggregation("lines", "count"),
            ],
        }

        result = _grouped(_notes(), config)

        expected = pl.DataFrame(
            {
                "code": ["A", "B", "C"],
                "notes": ["beta; alpha; gamma", "solo", ""],
                "lines": [3, 1, 1],
            }
        )
        assert_frame_equal(result, expected, check_row_order=True)
        assert result["notes"].dtype == pl.String, "a join of text produces text"

    def test_the_values_are_joined_in_the_order_the_rows_arrived_in(self) -> None:
        """Input order, not sorted order -- and the frame below sorts differently from it.

        This is the same ruling ``first`` makes, for the same reason: the tool groups rows in the
        order they arrive and does not reorder them inside a group, so the string is a fact about
        the order as well as about the data. A flow that needs the notes alphabetical puts a Sort in
        front of this step, exactly as everywhere else in this library, rather than this step
        growing a second way to express an order.
        """
        result = _grouped(
            _notes(),
            {"by": ["code"], "aggregations": [_aggregation("notes", "join_text", "[note]")]},
        )

        assert result["notes"].to_list()[0] == "beta, alpha, gamma", "arrival order"
        assert result["notes"].to_list()[0] != "alpha, beta, gamma", "not sorted behind the author's back"

    def test_the_separator_defaults_to_a_comma_and_a_space(self) -> None:
        """Omitting the field is a configuration a reader has to be able to predict.

        ``", "`` because the overwhelming use is a list a human reads in a cell, and because the
        alternative defaults are worse in ways nobody notices: a bare comma runs the values
        together, and a space alone makes a two-word value indistinguishable from two values.
        """
        result = _grouped(
            _notes(),
            {"by": ["code"], "aggregations": [_aggregation("notes", "join_text", "[note]")]},
        )

        assert result["notes"].to_list() == ["beta, alpha, gamma", "solo", ""]

    @pytest.mark.parametrize(
        ("separator", "expected"),
        [
            pytest.param("; ", "beta; alpha; gamma", id="a_semicolon_and_a_space"),
            pytest.param(" | ", "beta | alpha | gamma", id="a_pipe_with_spaces_around_it"),
            pytest.param("/", "beta/alpha/gamma", id="a_single_character"),
            pytest.param("", "betaalphagamma", id="the_empty_string_joins_with_nothing_between"),
        ],
    )
    def test_a_configured_separator_is_used_exactly_as_written(self, separator: str, expected: str) -> None:
        """Whatever the author wrote goes between the values, the empty string included.

        The empty string is a separator like any other and is not read as "nothing was configured":
        a fixed-width key assembled from a group's parts is a real step, and refusing it would send
        that author looking for a second tool to do a thing this one already does.
        """
        config = {
            "by": ["code"],
            "aggregations": [_aggregation("notes", "join_text", "[note]", separator=separator)],
        }

        result = _grouped(_notes(), config)

        assert result["notes"].to_list()[0] == expected

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(["x", "y"], "x, y", id="every_value_present"),
            pytest.param(["x", None, "y"], "x, y", id="a_missing_value_leaves_no_gap_and_no_separator"),
            pytest.param([None, "y"], "y", id="a_missing_value_first_leaves_no_leading_separator"),
            pytest.param([None, None], "", id="nothing_to_join_is_the_empty_string"),
        ],
    )
    def test_missing_values_are_skipped_and_a_group_of_only_missing_values_joins_to_the_empty_string(
        self,
        values: Sequence[str | None],
        expected: str,
    ) -> None:
        """Skipping is what ``string_agg`` does, and the all-missing answer is the empty string.

        Joining a null in would put ``"x, , y"`` into a report, which reads as a value that is
        blank rather than as a value that is absent, and one null would otherwise void the whole
        string if it propagated the way arithmetic does.

        The all-missing group answers ``""`` and *not* null, which is deliberately the opposite of
        the ruling ``sum`` makes about the same group. The two differ because the claims differ:
        joining no strings together produces the empty string in any language that has both, while
        adding no numbers together produces a total nobody measured. A flow that needs to tell an
        empty join from a group that had nothing to join counts beside it with ``count_non_null``.
        """
        frame = pl.DataFrame({"code": ["A"] * len(values), "note": values}, schema_overrides={"note": pl.String})

        result = _grouped(frame, {"by": ["code"], "aggregations": [_aggregation("notes", "join_text", "[note]")]})

        assert result["notes"].to_list() == [expected]
        assert result["notes"].dtype == pl.String

    def test_join_text_reads_an_expression_like_every_other_aggregate(self) -> None:
        """What is joined is an ordinary expression, so the text can be shaped on the way in.

        The aggregate is configuration and the thing aggregated is an expression -- the rule the
        whole tool is built on -- so a new aggregate inherits it rather than restricting itself to
        bare columns.
        """
        config = {
            "by": ["code"],
            "aggregations": [_aggregation("notes", "join_text", "upper([note])", separator="+")],
        }

        result = _grouped(_notes(), config)

        assert result["notes"].to_list() == ["BETA+ALPHA+GAMMA", "SOLO", ""]


# ---------------------------------------------------------------------------
# Every aggregate that ships, computed and held against an answer written by hand
# ---------------------------------------------------------------------------


def _one_group_frame() -> pl.DataFrame:
    """One group, built so that no two published aggregates can agree about it.

    The amounts arrive 8.0, missing, 2.0, 13.0, 8.0, 8.0, 3.0. ``first`` is 8.0, ``min`` is 2.0,
    ``max`` is 13.0 and the last row holds 3.0, so every one of the four is a different number and
    an aggregate computed under the wrong name comes back visibly wrong rather than plausibly
    right. The repeats and the blank separate the three counts as well: seven rows, six values,
    five distinct values counting the blank.

    ``note`` is the same group in text, because an aggregate that joins text cannot be asked of a
    column of numbers -- that is refused on the schema, which is itself a ruling. Its values arrive
    in an order that is not their sorted order, and the row whose amount is missing has no note
    either, so a join that sorted behind the author's back or joined a blank in as a gap would come
    back visibly wrong too.
    """
    return pl.DataFrame(
        {
            "code": ["A"] * 7,
            "amount": [8.0, None, 2.0, 13.0, 8.0, 8.0, 3.0],
            "note": ["beta", None, "alpha", "gamma", "beta", "delta", "epsilon"],
            "late": [None, 6.5, 1.0, 20.0, None, None, None],
        },
        schema={"code": pl.String, "amount": pl.Float64, "note": pl.String, "late": pl.Float64},
    )


def _value_for(aggregate: str) -> str | None:
    """The ``value`` this aggregate is configured with over :func:`_one_group_frame`.

    ``count`` counts rows and takes no value at all, which is this tool's rule, so it gets none.
    Every other aggregate but one reads the amounts. ``join_text`` reads the notes, because joining
    a column of numbers is refused on the schema -- so asking it of ``amount`` would test the
    refusal, not the answer.
    """
    if aggregate == "count":
        return None
    if aggregate == "first_non_null":
        return "[late]"  # its first row is blank, so first_non_null and first disagree
    return "[note]" if aggregate == "join_text" else "[amount]"


def _group_answers() -> dict[str, Any]:
    """Every published aggregate, and what it makes of the single group in :func:`_one_group_frame`.

    Written down by hand rather than computed, so the test has an answer of its own to hold the
    tool against. No two of them are equal, which is what makes a misdispatched aggregate visible
    rather than merely possible.
    """
    return {
        "sum": 42.0,
        "count": 7,
        "count_non_null": 6,
        "count_distinct": 4,
        "min": 2.0,
        "max": 13.0,
        "first": 8.0,
        "first_non_null": 6.5,
        "join_text": "beta, alpha, gamma, beta, delta, epsilon",
    }


class TestEveryPublishedAggregateIsReallyComputed:
    """Each name in the published set computes the thing it is named after, and not something else.

    The dispatcher is a chain of branches ending in a raised assertion, and a branch that computed
    the wrong thing would be invisible everywhere else in this file: the cases above cover the
    aggregates one or two at a time, on a frame where ``first``, ``min`` and ``max`` often agree,
    and ``first`` is not computed at all -- it appears only as a configuration that is refused,
    which raises before ``execute`` ever runs. Today the only thing standing between
    ``expression.first()`` becoming ``expression.last()`` and a shipped release is a test in
    CrossTab's file, on CrossTab's fixture. That is the wrong tool answering for this one, and it
    is fourteen money columns of a real client flow that depend on the answer.

    So this class walks the published set rather than a list typed out beside it. An aggregate
    added to that set is accepted in configuration here the day it lands, and if nothing walks the
    set it is accepted and quietly computed as something else, in a finance column, with nothing
    red.
    """

    def test_an_answer_is_written_down_for_every_published_aggregate(self) -> None:
        """The set and the answers have to match exactly, in both directions.

        An aggregate in the set with no answer written down is the drift this class exists to
        catch: it makes somebody decide what the new aggregate does to a group before it ships. An
        answer written down for an aggregate nobody publishes is dead weight pointing at a name
        that is gone.
        """
        assert set(_group_answers()) == set(AGGREGATE_FUNCTIONS)

    def test_no_two_aggregates_agree_about_this_group(self) -> None:
        """Without this, the class below proves nothing.

        A fixture where ``first`` and ``max`` happen to be the same number cannot tell one computed
        as the other from one computed properly. Whole numbers and decimals are compared here as
        the same value on purpose, because a count of 5 and a first of 5.0 would not distinguish
        those two branches either.
        """
        answers = list(_group_answers().values())

        assert len(set(answers)) == len(answers), "an aggregate under the wrong name has to be a visible difference"

    @pytest.mark.parametrize("aggregate", sorted(AGGREGATE_FUNCTIONS))
    def test_the_group_holds_the_answer_written_down_for_this_aggregate(self, aggregate: str) -> None:
        """One group, one aggregate, against an answer written down by hand.

        ``first`` is the one this catches today. It is in the published set and in the template, it
        is what a real client flow carries a document's constant columns forward with, and nothing in
        this file computes it: swapped for ``last`` it answers 3.0 where the first row holds 8.0,
        and every test here stays green.
        """
        answers = _group_answers()
        assert aggregate in answers, f"no answer written down for {aggregate!r}"

        config = {"by": ["code"], "aggregations": [_aggregation("answer", aggregate, _value_for(aggregate))]}

        result = _grouped(_one_group_frame(), config)

        assert result["answer"].to_list() == [answers[aggregate]]


# ---------------------------------------------------------------------------
# Refusals: an author's mistake, in the author's words, before any row is read
# ---------------------------------------------------------------------------


def _refusal_cases() -> list[Any]:
    """(config, the words its refusal must carry) for each mistake this tool has to catch itself."""
    total = _aggregation("total", "sum", "[amount]")
    return [
        pytest.param({"by": ["cod"], "aggregations": [total]}, ("cod",), id="a_group_column_that_is_not_there"),
        pytest.param(
            {"by": ["code"], "aggregations": [_aggregation("total", "sum", "[amoun]")]},
            ("amoun",),
            id="an_expression_naming_a_column_that_is_not_there",
        ),
        pytest.param(
            {"by": ["code"], "aggregations": [_aggregation("code", "count")]},
            ("code",),
            id="an_output_name_colliding_with_a_group_column",
        ),
        pytest.param(
            {"by": ["code"], "aggregations": [total, _aggregation("total", "count")]},
            ("total",),
            id="two_aggregations_sharing_an_output_name",
        ),
        pytest.param(
            {"by": ["code"], "aggregations": [_aggregation("n", "count", "[amount]")]},
            ("count_non_null",),
            id="a_row_count_given_something_to_count",
        ),
        pytest.param(
            {"by": ["code"], "aggregations": [_aggregation("total", "sum")]},
            ("value", "sum"),
            id="an_aggregate_with_nothing_to_aggregate",
        ),
        pytest.param(
            {"by": ["code"], "aggregations": [_aggregation("total", "sum", "[term]")]},
            ("term",),
            id="a_sum_over_text",
        ),
        pytest.param(
            {"by": ["code"], "aggregations": [_aggregation("average", "mean", "[amount]")]},
            ("mean",),
            id="an_aggregate_that_does_not_ship",
        ),
    ]


class TestGroupByRefusals:
    """What GroupBy refuses, and when: on the schema, before a single row is touched."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [5, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_configuration_the_schema_refuses_is_refused_before_any_row_is_read(
        self,
        config: Mapping[str, Any],
        named: tuple[str, ...],
        height: int,
    ) -> None:
        """Every one of these is a configuration fault, never a data fault, at either height.

        Running each against a frame with no rows at all is what shows the refusal is read off the
        schema rather than off the data -- and a sum over text is the case that needs it, because a
        frame with no rows would otherwise sum nothing and raise nothing. ``mean`` is refused because
        no aggregate ships without a consumer, and the template's enum is what refuses it.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(GroupBy(), Ports({"rows": frame}), config)

        message = str(caught.value)
        for word in named:
            assert word in message, f"refusal should name {word!r}: {message}"

    def test_a_frame_with_no_rows_gives_no_groups_but_keeps_its_output_schema(self) -> None:
        """No rows means no groups -- and a typed, empty port, so the next step still compiles.

        A frame that arrives empty is not an error here: the step upstream is allowed to have found
        nothing. What is not allowed is losing the schema, because the assertion or the join below
        this step reads its columns.
        """
        config = {
            "by": ["code"],
            "aggregations": [_aggregation("total", "sum", "[amount]"), _aggregation("lines", "count")],
        }

        result = _grouped(_rows().head(0), config)

        assert result.height == 0
        assert result.schema == pl.Schema({"code": pl.String, "total": pl.Float64, "lines": pl.Int64})

    @pytest.mark.parametrize(
        ("value", "column", "dtype"),
        [
            pytest.param("[amount]", "amount", "Float64", id="a_decimal_number"),
            pytest.param("[qty]", "qty", "Int64", id="a_whole_number"),
            pytest.param("[posted]", "posted", "Date", id="a_date"),
        ],
    )
    @pytest.mark.parametrize("height", [5, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_text_join_over_something_that_is_not_text_is_refused_and_names_the_dtype(
        self,
        value: str,
        column: str,
        dtype: str,
        height: int,
    ) -> None:
        """Joining a number or a date into a string is an author's mistake, caught on the schema.

        The alternative is worse than a wrong answer: it is a plausible one. ``10.0, 5.0, 2.5``
        reads as a list of amounts and ``2024-01-31, 2024-01-15`` reads as a list of dates, so the
        step ships, the report fills, and the column is text from there on -- unsortable, unsummable,
        and nobody is told. ``min`` and ``max`` stay open to anything orderable because ordering a
        number is not a category error; joining one is.

        Read off the schema, so an empty frame is refused identically -- the case that matters,
        since a frame with no rows would otherwise join nothing and raise nothing. The refusal has
        to name the dtype the column actually holds, because an author looking at ``[posted]``
        already believes it is text.
        """
        frame = _notes().head(height)
        config = {"by": ["code"], "aggregations": [_aggregation("joined", "join_text", value)]}

        with pytest.raises(ConfigError) as caught:
            run_tool(GroupBy(), Ports({"rows": frame}), config)

        message = str(caught.value)
        for word in (column, dtype, "join_text"):
            assert word in message, f"refusal should name {word!r}: {message}"

    @pytest.mark.parametrize(
        ("function", "value"),
        [
            pytest.param("sum", "[amount]", id="beside_a_sum"),
            pytest.param("first", "[note]", id="beside_a_first"),
            pytest.param("min", "[note]", id="beside_a_min"),
            pytest.param("count", None, id="beside_a_row_count"),
        ],
    )
    @pytest.mark.parametrize(
        "separator",
        ["; ", ""],
        ids=["a_separator_somebody_typed", "a_separator_that_is_the_empty_string"],
    )
    def test_a_separator_beside_any_other_aggregate_is_refused_rather_than_ignored(
        self,
        function: str,
        value: str | None,
        separator: str,
    ) -> None:
        """A field that would do nothing is refused, naming where it sits and what it sits beside.

        This is Join's ``keys``-beside-``left_keys`` ruling applied to the same shape of mistake: a
        configuration whose fields are individually valid and jointly say two things. Accepting it
        would mean an author who wrote ``function: min`` where they meant ``join_text`` gets a
        separator that vanishes, and a column of single values where they expected a list, with
        nothing anywhere saying which field was ignored.

        The refusal carries three things, because a file may hold dozens of aggregations: which
        element it is, which function that element actually asks for, and what to do about it.

        The empty string is one of the separators run here, and it is not a spare case. Elsewhere
        in this tool ``""`` is a configured separator like any other -- the fixed-width key built
        out of a group's parts is a real step -- so a guard that asked whether a separator was
        truthy rather than whether one was written down would accept ``{function: min, separator:
        ""}`` in silence while the join half of the tool went on treating it as a real value. The
        two halves have to read the field the same way, and this is the half that says so.
        """
        aggregations = [
            _aggregation("total", "sum", "[amount]"),
            _aggregation("other", function, value, separator=separator),
        ]

        with pytest.raises(ConfigError) as caught:
            run_tool(GroupBy(), Ports({"rows": _notes()}), {"by": ["code"], "aggregations": aggregations})

        message = str(caught.value)
        for word in ("aggregations[1]", "separator", function, "join_text"):
            assert word in message, f"refusal should name {word!r}: {message}"

    def test_group_by_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the data must be unchanged too."""
        frame = _rows()
        before = frame.clone()

        _grouped(frame, {"by": ["code"], "aggregations": [_aggregation("total", "sum", "[amount]")]})

        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# The template: a published whitelist, and a nested shape a loader can read
# ---------------------------------------------------------------------------


class TestGroupByTemplate:
    """The aggregate whitelist is published, and the enum that enforces it is derived from it."""

    def test_the_aggregates_that_ship_are_published_and_each_has_a_consumer(self) -> None:
        """A whitelist a template author can read, and one that cannot fail open.

        The set is pinned exactly, in both directions. ``sum`` is a per-group total; ``count`` and
        ``count_non_null`` are the two counts; ``count_distinct`` decides whether a group covers a single
        coherent period; ``min``, ``max`` and ``first`` are the ways to carry forward a column that
        is constant within a group, since every other column disappears. ``mean`` is absent because
        no step wants it, and an aggregate with no consumer is an arm nobody has ever run.

        ``first`` is the newest and the only one that reads the order rows arrived in -- so it is the
        only one that can answer twice for the same data. It arrived with a consumer that needed
        exactly that: a rebuild reproducing an engine which takes the first row of each group, on
        data where the groups turned out not to be as constant as the engine believed. Faithfulness
        was the point; a conversion that quietly answered differently would have hidden the
        difference rather than shown it.

        ``join_text`` is the newest, and the only one that answers with something other than a
        number or a value that was already in the column: it makes one string out of a group's
        several, which is how a group's detail survives a step whose whole job is to throw detail
        away. Its consumer is the line-level note that has to stay readable beside a per-group
        total, where the alternatives are a second frame to join back or a ``first`` that quietly
        drops every line but one.
        """
        assert isinstance(AGGREGATE_FUNCTIONS, frozenset)
        assert AGGREGATE_FUNCTIONS == frozenset(
            {"sum", "count", "count_non_null", "count_distinct", "min", "max", "first", "first_non_null", "join_text"}
        )

        function_field = GroupBy().config_template()["aggregations"].items.fields["function"]
        assert set(function_field.enum or ()) == AGGREGATE_FUNCTIONS, (
            "the enum must be derived from the published set, or the two drift and the whitelist fails open"
        )

    def test_an_aggregate_added_to_the_published_set_reaches_cross_tab_the_same_day(self) -> None:
        """CrossTab imports this set rather than copying it, so publishing here publishes there.

        That is the point of one whitelist, and it is also the bill: the moment ``join_text`` joins
        the set, a CrossTab configuration naming it is accepted, and an aggregate a tool accepts
        without answering for is the failure that tool's own suite exists to catch. So this is only
        half the check -- the half that says the enum is still derived and the catalog still
        conforms. The other half, that a cell really is joined text, lives beside CrossTab.
        """
        assert "join_text" in AGGREGATE_FUNCTIONS

        aggregate_field = CrossTab().config_template()["function"]
        assert tuple(aggregate_field.enum or ()) == tuple(sorted(AGGREGATE_FUNCTIONS))
        check_tool(CrossTab())

    def test_the_template_declares_the_nested_shape_so_a_loader_can_pre_flight_the_expressions(self) -> None:
        """``by`` is column names, ``aggregations`` is a list of objects, and only ``value`` compiles.

        ``by`` holds column names and not expressions on purpose: a computed group key has to exist
        as a column, because every downstream reader of the result joins or reports on that key, and
        a key that lives only inside this step's configuration cannot be joined against. Computing
        one is a Formula step in front of this one.
        """
        template = GroupBy().config_template()

        assert set(template) == {"by", "aggregations"}
        assert (template["by"].required, template["by"].non_empty) == (True, True)
        assert template["by"].items is not None and template["by"].items.type == "string"
        assert (template["aggregations"].required, template["aggregations"].non_empty) == (True, True)

        element = template["aggregations"].items
        assert element is not None and element.type == "object"
        assert element.fields is not None and set(element.fields) == {"name", "function", "value", "separator"}
        assert element.fields["name"].required is True
        assert element.fields["function"].required is True
        assert element.fields["value"].required is False, "a row count has nothing to aggregate"
        assert expression_fields(template) == ("aggregations[].value",), "the loader seam is the declared type"

    def test_the_separator_is_an_optional_string_carrying_no_default_a_template_could_fill(self) -> None:
        """Declared so a loader type-checks it, and left empty so the tool's own default is the only one.

        A default written into a member of an element is never applied -- defaults are filled at the
        top level of a configuration only -- so declaring ``", "`` here would read to the next
        author as if it were filled while ``execute`` still had to supply it, which is two defaults
        that can disagree. The default lives in the tool's code, and this is the assertion that
        stops it from being written down twice.

        It is a plain string and not an expression: it is a piece of punctuation, not something
        computed per row, so the loader must not try to compile it -- which is what the unchanged
        ``expression_fields`` above says.
        """
        element = GroupBy().config_template()["aggregations"].items

        assert element is not None and element.fields is not None
        separator = element.fields["separator"]
        assert separator.type == "string"
        assert separator.required is False, "only 'join_text' uses it, and even there it is optional"
        assert separator.default is None, "a default inside an element is never filled in"


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestGroupByInTheCatalog:
    """GroupBy is described, documented with examples that actually run, and resolvable by name."""

    def test_group_by_conforms_and_an_example_pins_the_row_order_it_promises(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output."""
        tool = GroupBy()

        check_tool(tool)

        assert tool.name == "group_by"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert len(examples) >= 2, "one plain total, and one showing a second aggregate"
        assert any(example.check_row_order for example in examples), (
            "the tool promises first-appearance order, so an example must have that promise checked"
        )

    def test_a_worked_example_joins_text_so_conformance_runs_the_separator_and_the_blanks(self) -> None:
        """The catalog is where an author meets this aggregate, so the answer has to be shown there.

        Every question ``join_text`` raises -- what separator, which order, what a blank does -- is
        a question an author asks while reading the catalog, and a worked example answers all three
        at once in the only form this library trusts: one that is executed and compared on every
        conformance run, so the documented string cannot drift from the computed one.
        """
        tool = GroupBy()

        check_tool(tool)

        joining = [
            example
            for example in tool.help().examples
            if any(
                aggregation.get("function") == "join_text"
                for aggregation in example.configuration.get("aggregations", ())
            )
        ]
        assert joining, "an aggregate nobody demonstrates is one conformance never runs"

    def test_a_worked_example_shows_a_group_that_joins_to_the_empty_string(self) -> None:
        """The one ruling here that surprises a reader, executed rather than asserted in prose.

        A group whose values are all missing joins to ``""`` and not to null, which is deliberately
        the opposite of what ``sum`` says about the same group. The notes say so, and nothing runs
        the notes: ``check_tool`` executes every worked example and compares its output, and never
        reads a word of ``notes``. So that sentence could be changed to claim the opposite and the
        whole catalog would stay green.

        The example that demonstrates the join shows the skip rule -- a missing note left out
        rather than joined in as a gap -- but no group in it is empty, so the ruling a reader is
        most likely to want checked before trusting this aggregate with a report is the one thing
        the catalog does not execute. A group whose only note is missing closes that: conformance
        computes the empty string every run, and a reader meets it in the place they were already
        looking.
        """
        tool = GroupBy()

        check_tool(tool)

        joined = [
            (example, aggregation["name"])
            for example in tool.help().examples
            for aggregation in example.configuration.get("aggregations", ())
            if aggregation.get("function") == "join_text"
        ]
        assert joined, "an aggregate nobody demonstrates is one conformance never runs"
        assert any("" in example.outputs["rows"][name].to_list() for example, name in joined), (
            "no worked example joins a group to '', so the ruling a reader asks about is prose nothing executes"
        )

    def test_group_by_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "group_by" in REGISTRY
        assert isinstance(REGISTRY.get("group_by"), GroupBy)


class TestFirstWhenTheFirstValueIsNull:
    """What ``first`` gives when a group's first row holds no value.

    It gives that null; it does not skip ahead to the first real value. pandas'
    ``groupby().first()`` does the opposite, so a port or a "fix" would change figures quietly.
    The owner ruled on 2026-11-18 to keep this rule: the first row as it arrives, null or not.
    """

    # pins current behaviour, kept by the owner's ruling on finding F-02
    def test_first_returns_the_first_rows_null_rather_than_the_first_value(self) -> None:
        rows = pl.DataFrame(
            {"code": ["a", "a", "b", "b"], "amount": [None, 7.0, 3.0, None], "note": [None, "x", "y", None]},
            schema={"code": pl.String, "amount": pl.Float64, "note": pl.String},
        )
        config = {
            "by": ["code"],
            "aggregations": [
                {"name": "first_amount", "function": "first", "value": "[amount]"},
                {"name": "first_note", "function": "first", "value": "[note]"},
            ],
        }

        result = run_tool(GroupBy(), Ports({"rows": rows}), config)["rows"].sort("code")

        assert_frame_equal(
            result,
            pl.DataFrame(
                {"code": ["a", "b"], "first_amount": [None, 3.0], "first_note": [None, "y"]},
                schema={"code": pl.String, "first_amount": pl.Float64, "first_note": pl.String},
            ),
        )


class TestAnAggregateOfAFixedValueIsRefused:
    """An aggregation whose 'value' reads no column is refused before anything runs.

    polars aggregates a fixed value once per group, not once per row. So ``sum`` of ``1``, which
    is SQL's ``SUM(1)``, gave 1 for a group of five rows, and the run reported success.
    """

    @pytest.mark.parametrize(
        ("function", "value"),
        [
            ("sum", "1"),
            ("sum", "2 * 3"),
            ("count_non_null", "1"),
            ("count_distinct", "1"),
            ("min", "1"),
            ("max", "1"),
            ("first", "1"),
            ("first_non_null", "1"),
            ("join_text", "'x'"),
        ],
    )
    @pytest.mark.parametrize("height", [5, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_value_that_names_no_column_is_refused_by_name(self, function: str, value: str, height: int) -> None:
        aggregations = [_aggregation("lines", "count"), _aggregation("fixed", function, value)]

        with pytest.raises(ConfigError) as caught:
            run_tool(GroupBy(), Ports({"rows": _rows().head(height)}), {"by": ["code"], "aggregations": aggregations})

        message = str(caught.value)
        for word in ("aggregations[1].value", value, "no column", "'count'", "formula"):
            assert word in message, f"refusal should name {word!r}: {message}"

    def test_the_refusal_comes_from_the_configuration_alone(self) -> None:
        """No frame is needed, so a pipeline file is refused when it loads."""
        config = {"by": ["code"], "aggregations": [_aggregation("fixed", "sum", "1")]}

        with pytest.raises(ConfigError, match=r"aggregations\[0\]\.value"):
            GroupBy().validate_config(config)

    def test_a_value_that_reads_a_column_is_aggregated_once_per_row(self) -> None:
        """The mirror: the same fixed value, tied to a column, sums to the row count of each group."""
        config = {
            "by": ["code"],
            "aggregations": [_aggregation("ones", "sum", "[qty] * 0 + 1"), _aggregation("lines", "count")],
        }

        result = _grouped(_rows(), config)

        assert result["ones"].to_list() == result["lines"].to_list() == [2, 2, 1]


class TestATotalTooLargeForItsTypeStopsTheRun:
    """A sum of 32-bit whole numbers keeps its type, and a total that does not fit stops the run.

    polars adds up an Int32 column into an Int32 and wraps a total past about 2.1 billion round to a
    negative number. The type is kept, so a pipeline that runs today produces the same columns; a
    total that would have wrapped is refused instead, naming the aggregation.
    """

    @pytest.mark.parametrize(
        ("dtype", "values"),
        [
            (pl.Int32, [2_000_000_000, 2_000_000_000]),
            (pl.Int32, [-2_000_000_000, -2_000_000_000]),
            (pl.UInt32, [4_000_000_000, 4_000_000_000]),
        ],
        ids=["int32_above", "int32_below", "uint32_above"],
    )
    def test_a_total_past_the_types_limit_is_refused_by_name(self, dtype: pl.DataType, values: list[int]) -> None:
        rows = pl.DataFrame({"g": [1, 1, 2], "v": pl.Series([*values, 5], dtype=dtype)})
        config = {"by": ["g"], "aggregations": [_aggregation("total", "sum", "[v]")]}

        with pytest.raises(DataError) as caught:
            run_tool(GroupBy(), Ports({"rows": rows}), config)

        message = str(caught.value)
        for word in ("'total'", "aggregations[0]", str(dtype), "cast_columns", "Int64", "1 of 2"):
            assert word in message, f"refusal should name {word!r}: {message}"

    @pytest.mark.parametrize("dtype", [pl.Int32, pl.UInt32])
    def test_a_total_that_fits_keeps_the_columns_type_and_value(self, dtype: pl.DataType) -> None:
        """The mirror: a total just inside the limit, a null group and an empty frame keep the type."""
        limit = 2_147_483_647 if dtype == pl.Int32 else 4_294_967_295
        rows = pl.DataFrame(
            {"g": [1, 1, 2, 3], "v": pl.Series([limit - 1, 1, 7, None], dtype=dtype)},
        )
        config = {"by": ["g"], "aggregations": [_aggregation("total", "sum", "[v]")]}

        result = _grouped(rows, config)
        empty = _grouped(rows.head(0), config)

        assert result["total"].dtype == dtype
        assert result["total"].to_list() == [limit, 7, None]
        assert empty.schema["total"] == dtype


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_aggregation_name_is_refused_when_the_pipeline_loads(blank: str) -> None:
    """A column with no name cannot be referred to by any later step, so it is never written."""
    config = {"by": ["code"], "aggregations": [_aggregation("total", "sum", "[amount]"), _aggregation(blank, "count")]}

    with pytest.raises(ConfigError, match=r"'aggregations\[1\]\.name' must not be blank"):
        GroupBy().validate_config(config)
    with pytest.raises(ConfigError, match=r"'aggregations\[1\]\.name' must not be blank"):
        run_tool(GroupBy(), Ports({"rows": _rows()}), config)
