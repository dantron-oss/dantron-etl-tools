"""Tests for the CaseWhen tool: ordered rules, first match wins, one column out.

CaseWhen takes one input port, ``rows``, and produces one, ``rows``: the input with one added
column whose value is decided by the first rule a row satisfies. Its configuration is an ordered
array of ``{when, then}`` objects, an optional ``otherwise``, and the ``output`` column name -- so,
like Formula, it has no top-level expression field at all and every expression it holds sits inside
an element of an array.

This is the tool that decides whether a rebuild of a real pipeline ties out, because the pipeline's
central step is an eight-branch ladder that labels every candidate row with the reason it counts or
does not -- a status in a set of superseded values, a missing amount, a period that could not be
parsed, a backwards date range -- and nothing is silently dropped. Five rulings follow from that,
each with a silent failure mode:

**First match wins.** Pinned with a row that satisfies two rules, and with the same two rules in
the other order producing the other answer, so nobody can later "optimise" the ladder into
something order-independent: an eight-branch ladder whose branches overlap has one correct reading,
and it is the one the author wrote down.

**A condition that is null is not a match.** Three-valued logic, as the rest of the language has
it: a comparison against a missing value is null, not false, so such a row falls through to the
next rule and eventually to ``otherwise`` -- or to null when there is none. Folding null into false
would make the ladder disagree with the engine that stores the data, in the direction of silently
labelling rows.

**Every branch agrees on one family of value.** A ladder returning text in one arm and a number in
another is a configuration mistake, and it must be refused rather than coerced: the engine happily
renders ``2.0`` as the text ``"2.0"`` and ``true`` as ``"true"``, so the un-refused version of this
mistake is a column of plausible-looking strings that nobody notices. A branch whose value is
``null`` belongs to every family, because a null does.

**A ladder with no rules is refused.** ``required=True`` treats only a missing value as missing, so
``cases: []`` reaches the tool. It is either a mistake or a constant column; a constant column is
Formula's job, and it says what it means, whereas an empty ladder is exactly the shape a loader
produces when the rules were dropped from a file -- and it would quietly write a column of
``otherwise``, or of nulls.

**The output column is a new column unless the author typed otherwise.** Overwriting a column
another step depends on is a hazard, so a name the input already carries is refused -- unless the
configuration says ``replace: true``, exactly as a Formula entry does, and then the column is
overwritten in the position it already had. ``replace: true`` for a column that is *not* there is
refused too: that configuration claims something is there to repair, and a misspelling would
otherwise quietly write a new column while the one meant for repair kept its old values. The flag
is top-level rather than per-entry because this tool writes exactly one column, and the dtype a
replacement writes is whatever the branches produce -- replacing text with a number is legitimate,
and is what a blank-to-null reading or a code becoming a figure looks like.

Vocabulary is deliberately generic -- rows, status, amount, category, period, rate, disposition --
because the library carries no caller's nouns; this is the shape of a real ladder, not its names.
"""

import dataclasses
import datetime as dt
import re
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    REPLACE_FIELD,
    YAML_UNSAFE_VALUES,
    ConfigError,
    ConfigField,
    Ports,
    ToolError,
    apply_defaults,
    check_tool,
    expression_fields,
    expression_values,
    run_tool,
    validate_against_template,
)
from dantron_etl.tools.case_when import CaseWhen
from dantron_etl.tools.formula import Formula

# ---------------------------------------------------------------------------
# Test data -- the shape the real disposition ladder runs against
# ---------------------------------------------------------------------------

_LADDER: Mapping[str, Any] = {
    "cases": [
        {"when": "[status] in ('superseded', 'replaced')", "then": "'superseded'"},
        {"when": "[amount] = 0", "then": "'zero_total'"},
        {"when": "[category] is null", "then": "'no_category'"},
        {"when": "[period_start] is null", "then": "'no_period'"},
        {"when": "[period_end] < [period_start]", "then": "'backwards_period'"},
        {"when": "[amount] is null", "then": "'no_amount'"},
        {"when": "[rate] = 0", "then": "'bad_rate'"},
        {"when": "[months_in_period] > 60", "then": "'long_period'"},
    ],
    "otherwise": "'counted'",
    "output": "disposition",
}
"""The eight-branch ladder, in the order the real pipeline reads it, with a fall-through."""


_CANDIDATE_SCHEMA: Mapping[str, Any] = {
    "key": pl.Int64,
    "status": pl.String,
    "amount": pl.Float64,
    "category": pl.String,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "rate": pl.Float64,
    "months_in_period": pl.Int64,
}

_START = dt.date(2026, 1, 1)
_END = dt.date(2026, 12, 1)


def _candidates() -> pl.DataFrame:
    """One row per branch of the ladder, plus one row that reaches no branch at all.

    Written a row at a time, because a branch is a row: each line below differs from the ordinary
    row at the bottom in exactly the one field its branch asks about. The first row is the
    interesting one -- it satisfies the first branch *and* the sixth, having no amount -- which is
    what makes the ladder's order load-bearing rather than incidental.
    """
    ordinary = (0, "open", 10.0, "standard", _START, _END, 1.0, 12)

    def row(key: int, **changed: Any) -> tuple[Any, ...]:
        values = dict(zip(_CANDIDATE_SCHEMA, ordinary, strict=True)) | {"key": key} | changed
        return tuple(values[name] for name in _CANDIDATE_SCHEMA)

    return pl.DataFrame(
        [
            row(1, status="superseded", amount=None),
            row(2, status="replaced"),
            row(3, amount=0.0),
            row(4, category=None),
            row(5, period_start=None),
            row(6, period_start=dt.date(2026, 6, 1), period_end=dt.date(2026, 3, 1)),
            row(7, amount=None),
            row(8, rate=0.0),
            row(9, months_in_period=72),
            row(10),
        ],
        schema=_CANDIDATE_SCHEMA,
        orient="row",
    )


def _rows() -> pl.DataFrame:
    """A small frame for the rulings that do not need the whole ladder."""
    return pl.DataFrame({"key": [1, 2, 3], "code": ["A1", "B2", None], "amount": [10.0, -5.0, 0.0]})


def _existing() -> pl.DataFrame:
    """A frame whose replaceable columns sit in the middle, not at the end.

    Position is the whole point of these tests, so ``status``, ``company`` and ``amount`` are each
    surrounded by other columns: a replacement that appended instead of overwriting in place, or
    that dropped and re-added, would move them to the end and show up here rather than passing on a
    frame where the last column happened to be the one being written.
    """
    return pl.DataFrame(
        {
            "key": [1, 2, 3, 4, 5, 6, 7],
            "status": ["open", "superseded", "open", "void", None, "open", "replaced"],
            "company": ["  Acme  ", "", "Beta", None, "Delta", "   ", "Epsilon"],
            "amount": [10.0, 0.0, -5.0, 20.0, None, 7.5, 3.0],
            "code": ["A1", "B2", "C3", None, "E5", "F6", "G7"],
        }
    )


# ---------------------------------------------------------------------------
# Order: the first rule a row satisfies is the one that decides it
# ---------------------------------------------------------------------------


class TestCaseWhenOrder:
    """An ordered ladder, read in order, with a fall-through for the rows no rule reached."""

    def test_the_first_rule_a_row_satisfies_decides_it(self) -> None:
        """One row satisfies both rules; swapping the rules swaps its answer and nothing else.

        Running the same two rules in both orders is what makes this a test of the ladder rather
        than of one configuration: an implementation that evaluated the rules in any other order,
        or that combined them, would give the same answer twice.
        """
        frame = pl.DataFrame({"key": [1, 2, 3], "amount": [-5.0, 5.0, 0.0]})
        negative = {"when": "[amount] < 0", "then": "'negative'"}
        not_positive = {"when": "[amount] <= 0", "then": "'not_positive'"}

        first = run_tool(
            CaseWhen(),
            Ports({"rows": frame}),
            {"cases": [negative, not_positive], "otherwise": "'positive'", "output": "band"},
        )
        second = run_tool(
            CaseWhen(),
            Ports({"rows": frame}),
            {"cases": [not_positive, negative], "otherwise": "'positive'", "output": "band"},
        )

        assert first["rows"]["band"].to_list() == ["negative", "positive", "not_positive"]
        assert second["rows"]["band"].to_list() == ["not_positive", "positive", "not_positive"]

    def test_a_multi_branch_ladder_labels_every_row_and_the_rest_fall_through(self) -> None:
        """The eight-branch ladder: one row per branch, one falling through, nothing unlabelled.

        The first row is the one that matters most: it has no amount, so the sixth branch would
        claim it, and it must be labelled by the first branch instead. The last row reaches no
        branch at all and takes ``otherwise`` -- which is what "nothing is silently dropped" means
        in practice, because a row with no label is a row a reader has to explain later.

        Every branch is reached by some row, asserted rather than assumed: a later edit that left a
        branch unexercised would make this a seven-branch test wearing an eight-branch name.
        """
        frame = _candidates()

        result = run_tool(CaseWhen(), Ports({"rows": frame}), _LADDER)

        out = result["rows"]
        assert out["disposition"].to_list() == [
            "superseded",
            "superseded",
            "zero_total",
            "no_category",
            "no_period",
            "backwards_period",
            "no_amount",
            "bad_rate",
            "long_period",
            "counted",
        ]
        assert out["disposition"].null_count() == 0, "every row carries the reason it counts or does not"
        assert set(out["disposition"].to_list()) == {rule["then"].strip("'") for rule in _LADDER["cases"]} | {
            "counted"
        }, "every branch of the ladder is reached by some row, and the fall-through by one more"
        assert out.columns == [*frame.columns, "disposition"], "one column appended, the rest untouched"
        assert out.height == frame.height

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param(
                {
                    "cases": [
                        {"when": "[code] = 'A1'", "then": "'first'"},
                        {"when": "[code] is null", "then": "'missing'"},
                    ],
                    "otherwise": "'other'",
                    "output": "band",
                },
                ["first", "other", "missing"],
                id="the_null_row_falls_past_a_condition_it_cannot_answer",
            ),
            pytest.param(
                {"cases": [{"when": "[code] = 'A1'", "then": "'first'"}], "output": "band"},
                ["first", None, None],
                id="with_no_otherwise_the_rows_no_rule_matched_are_null",
            ),
            pytest.param(
                {"cases": [{"when": "[code] = 'A1'", "then": "'first'"}], "otherwise": "'other'", "output": "band"},
                ["first", "other", "other"],
                id="with_an_otherwise_a_null_row_takes_it_like_any_other_unmatched_row",
            ),
        ],
    )
    def test_a_condition_that_is_null_is_not_a_match(
        self,
        config: Mapping[str, Any],
        expected: Sequence[str | None],
    ) -> None:
        """A row whose condition cannot be answered is not matched -- it falls through.

        The first case is the one that distinguishes null from false: the row with no code is not
        claimed by ``[code] = 'A1'``, so it reaches the rule that asks about a null and is labelled
        by it. If a null condition were folded into a match, that row would be labelled 'first'.
        """
        result = run_tool(CaseWhen(), Ports({"rows": _rows()}), config)

        assert result["rows"]["band"].to_list() == list(expected)


# ---------------------------------------------------------------------------
# One family of value across every branch
# ---------------------------------------------------------------------------


class TestCaseWhenBranchTypes:
    """Every ``then`` and the ``otherwise`` must give the same family of value."""

    @pytest.mark.parametrize(
        ("config", "path", "quoted"),
        [
            pytest.param(
                {
                    "cases": [
                        {"when": "[code] = 'A1'", "then": "'first'"},
                        {"when": "[amount] > 0", "then": "[amount] * 2"},
                    ],
                    "output": "band",
                },
                "cases[1].then",
                "[amount] * 2",
                id="one_arm_gives_text_and_another_gives_a_number",
            ),
            pytest.param(
                {
                    "cases": [{"when": "[code] = 'A1'", "then": "'first'"}],
                    "otherwise": "[amount]",
                    "output": "band",
                },
                "otherwise",
                "[amount]",
                id="the_fallback_disagrees_with_the_arms",
            ),
            pytest.param(
                {
                    "cases": [
                        {"when": "[code] = 'A1'", "then": "[amount]"},
                        {"when": "[amount] > 0", "then": "starts_with([code], 'B')"},
                    ],
                    "output": "band",
                },
                "cases[1].then",
                "starts_with([code], 'B')",
                id="a_truth_value_beside_a_number",
            ),
        ],
    )
    def test_a_branch_that_disagrees_on_the_family_of_value_is_refused_naming_it(
        self,
        config: Mapping[str, Any],
        path: str,
        quoted: str,
    ) -> None:
        """Refused at compile time, naming the offending branch by its path and quoting it.

        Not coerced: handed to the engine as written, a number beside text is rendered as text --
        ``2.0`` becomes ``"2.0"``, ``true`` becomes ``"true"`` -- and a column of plausible-looking
        strings is the kind of wrong answer nobody notices. A file may hold a dozen branches, so
        the path is the only address that sends someone to the right line.
        """
        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _rows()}), config)

        message = str(caught.value)
        assert path in message, f"refusal should report the path {path!r}: {message}"
        assert quoted in message, f"refusal should quote the offending expression: {message}"

    @pytest.mark.parametrize(
        ("config", "expected", "dtype"),
        [
            pytest.param(
                {
                    "cases": [
                        {"when": "[code] = 'A1'", "then": "1"},
                        {"when": "[amount] < 0", "then": "[amount] / 2"},
                    ],
                    "otherwise": "0",
                    "output": "band",
                },
                [1.0, -2.5, 0.0],
                pl.Float64,
                id="a_whole_number_and_a_decimal_are_one_family",
            ),
            pytest.param(
                {
                    "cases": [
                        {"when": "[code] = 'A1'", "then": "'first'"},
                        {"when": "[amount] < 0", "then": "null"},
                    ],
                    "otherwise": "'other'",
                    "output": "band",
                },
                ["first", None, "other"],
                pl.String,
                id="a_branch_that_gives_null_belongs_to_every_family",
            ),
        ],
    )
    def test_families_are_coarse_and_a_null_branch_belongs_to_all_of_them(
        self,
        config: Mapping[str, Any],
        expected: Sequence[Any],
        dtype: pl.DataType,
    ) -> None:
        """The check has to be as coarse as the language's own families, or it costs capability.

        A whole number in one arm and a decimal in another is one family, exactly as a comparison
        between them is allowed. And a branch that deliberately yields nothing -- ``null`` -- is not
        a family disagreement: a null is the one value that belongs everywhere, so the column keeps
        the family of the branches that do give a value.
        """
        result = run_tool(CaseWhen(), Ports({"rows": _rows()}), config)

        band = result["rows"]["band"]
        assert band.dtype == dtype
        assert band.to_list() == list(expected)


# ---------------------------------------------------------------------------
# Refusals: an author's mistake, in the author's words, before any row is read
# ---------------------------------------------------------------------------


class TestCaseWhenRefusals:
    """What CaseWhen refuses, and when: on the schema, before a single row is touched."""

    @pytest.mark.parametrize(
        ("when", "named"),
        [
            pytest.param("[amount]", "amount", id="a_bare_number_column"),
            pytest.param("[amount] + 1", "amount", id="arithmetic_instead_of_a_comparison"),
            pytest.param("upper([code])", "code", id="text_instead_of_a_comparison"),
        ],
    )
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_condition_that_is_not_a_truth_value_is_refused_naming_the_column(
        self,
        when: str,
        named: str,
        height: int,
    ) -> None:
        """The same refusal Filter makes, for the same reason, on the same seam.

        ``[amount]`` compiles -- it is exactly how a column reference is written -- but it is not
        something a row can match or fail, and handed to the engine as a condition it raises about a
        dtype from a library the author has never heard of. Running against a frame with no rows is
        what shows the refusal is read off the schema rather than off the data.

        Filter implements this check itself; this is the second tool to need it, which is a signal
        it belongs beside ``compile_for`` in the contract rather than being written out per tool.
        """
        frame = _rows().head(height)
        config = {
            "cases": [{"when": "[code] = 'A1'", "then": "'first'"}, {"when": when, "then": "'second'"}],
            "output": "band",
        }

        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"
        assert when in message, f"refusal should quote the condition: {message}"
        assert "cases[1].when" in message, f"refusal should report the path: {message}"

    def test_a_ladder_with_no_rules_is_refused_by_the_tool(self) -> None:
        """``cases: []`` passes the template and must be refused by the tool, naming the field.

        ``required=True`` treats only a missing value as missing, so an empty list arrives at
        ``execute`` looking configured. It is the shape a loader produces when the rules were
        dropped from a file, and left alone it writes a column of ``otherwise`` -- or of nulls --
        that reads as if the ladder had run. A genuine constant column is Formula's job, where it
        says so.
        """
        config = {"cases": [], "otherwise": "'counted'", "output": "disposition"}
        validate_against_template(CaseWhen().config_template(), config)

        with pytest.raises(ConfigError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _rows()}), config)

        assert "cases" in str(caught.value)

    def test_the_output_column_must_not_silently_replace_an_input_column(self) -> None:
        """An ``output`` naming a column the input already holds is refused, naming the column.

        The hazard is a collision the author never intended, in a frame they did not write: silent
        replacement of a column a later step reads is a wrong answer with no failure attached to it.
        Saying ``replace: true`` is how an author says they meant it; saying nothing means they did
        not, and this is the configuration that said nothing.
        """
        config = {"cases": [{"when": "[amount] > 0", "then": "'positive'"}], "output": "code"}

        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _rows()}), config)

        assert "code" in str(caught.value)

    @pytest.mark.parametrize(
        ("config", "named", "path"),
        [
            pytest.param(
                {
                    "cases": [
                        {"when": "[amount] > 0", "then": "'first'"},
                        {"when": "[cdoe] = 'B2'", "then": "'second'"},
                    ],
                    "output": "band",
                },
                "cdoe",
                "cases[1].when",
                id="a_column_that_is_not_there_in_a_condition",
            ),
            pytest.param(
                {"cases": [{"when": "[amount] > 0", "then": "[amont]"}], "output": "band"},
                "amont",
                "cases[0].then",
                id="a_column_that_is_not_there_in_a_value",
            ),
            pytest.param(
                {
                    "cases": [{"when": "[amount] > 0", "then": "'first'"}],
                    "otherwise": "upper([amount])",
                    "output": "band",
                },
                "amount",
                "otherwise",
                id="a_text_function_on_a_number_in_the_fallback",
            ),
        ],
    )
    def test_a_mistake_in_a_nested_expression_is_refused_before_any_row_is_read(
        self,
        config: Mapping[str, Any],
        named: str,
        path: str,
    ) -> None:
        """Every branch is compiled against the frame's schema, and reported by its own path.

        The dtype case is the one a list of column names could not catch: it is caught only if the
        tool hands the compiler the frame's *schema*. Each of the three positions an expression can
        sit in -- a condition, a value, the fallback -- has to be reported by the path it sits at,
        because that is the address a loader and a person both use.
        """
        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _rows()}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"
        assert path in message, f"refusal should report the path {path!r}: {message}"

    def test_case_when_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the data must be unchanged too."""
        frame = _candidates()
        before = frame.clone()

        run_tool(CaseWhen(), Ports({"rows": frame}), _LADDER)

        assert_frame_equal(frame, before)


# ---------------------------------------------------------------------------
# Replacement: writing over a column that is already there, because the author said so
# ---------------------------------------------------------------------------


def _blanking_ladder(*, replace: bool | None = True) -> dict[str, Any]:
    """The ladder the replacement tests run: overlapping rules, a fallback, writing over 'company'.

    The rules read the very column they write, which is the shape of the real steps this flag is
    for -- a blank-to-null reading of a name, a status collapsed into a smaller set of statuses --
    and each of those needed a rename step first purely to move the name out of the way.
    """
    config: dict[str, Any] = {
        "cases": [
            {"when": "[status] = 'superseded'", "then": "'superseded_co'"},
            {"when": "trim([company]) = ''", "then": "null"},
        ],
        "otherwise": "trim([company])",
        "output": "company",
    }
    if replace is not None:
        config["replace"] = replace
    return config


class TestCaseWhenReplaces:
    """``replace: true`` overwrites a column already on the frame, where it already was."""

    def test_a_replaced_column_takes_the_new_value_in_the_position_it_already_had(self) -> None:
        """The whole frame, compared whole: the new values, in place, with nothing else moved.

        Position is asserted by comparing the entire frame rather than the one column, because a
        column that quietly moves to the end when a step rewrites it has changed in a way nobody
        wrote down -- every downstream step that reads by position, and every written table, sees a
        different shape for a reason the pipeline does not record. Formula already keeps a replaced
        column where it was; these two tools must not disagree about that.

        The ladder also reads the column it writes, and reads the values that were there before it
        ran: row 1 is trimmed from what ``company`` held, not from what this step is putting in it.
        Row 2 satisfies both rules and takes the first, so first-match-wins is unchanged by
        replacement, and row 4's condition is null rather than false and falls through to the
        fallback, which is unchanged too.
        """
        frame = _existing()

        result = run_tool(CaseWhen(), Ports({"rows": frame}), _blanking_ladder())

        expected = pl.DataFrame(
            {
                "key": [1, 2, 3, 4, 5, 6, 7],
                "status": ["open", "superseded", "open", "void", None, "open", "replaced"],
                "company": ["Acme", "superseded_co", "Beta", None, "Delta", None, "Epsilon"],
                "amount": [10.0, 0.0, -5.0, 20.0, None, 7.5, 3.0],
                "code": ["A1", "B2", "C3", None, "E5", "F6", "G7"],
            }
        )
        assert_frame_equal(result["rows"], expected)
        assert result["rows"].columns == frame.columns, "no column added, none moved"
        assert result["rows"].columns.index("company") == 2, "the replaced column is where it was"
        assert result["rows"].schema == frame.schema, "the other columns keep their types as well as their order"

    @pytest.mark.parametrize(
        "replace",
        [pytest.param(None, id="the_flag_left_out"), pytest.param(False, id="the_flag_written_as_false")],
    )
    @pytest.mark.parametrize("height", [7, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_without_the_flag_a_name_the_input_carries_is_still_refused_and_the_refusal_says_the_fix(
        self,
        replace: bool | None,
        height: int,
    ) -> None:
        """Adding the flag must not weaken the guard for every configuration written before it.

        This is the check that stops a step silently overwriting a column it did not mean to: the
        author of one step does not know every column of the frame they are writing into, and the
        value they overwrote is gone with no failure attached. So the default stays a refusal, and
        the refusal names the column and the one keystroke that would authorise it -- a refusal that
        does not say how to proceed sends the author to the source of a tool they did not write.
        """
        frame = _existing().head(height)

        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": frame}), _blanking_ladder(replace=replace))

        message = str(caught.value)
        assert "company" in message, f"refusal should name the column that already exists: {message}"
        assert "output" in message, f"refusal should name the field: {message}"
        assert "replace" in message, f"refusal should say how to authorise it: {message}"

    @pytest.mark.parametrize("height", [7, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_replacing_a_column_the_input_does_not_carry_is_refused_naming_it(self, height: int) -> None:
        """``replace: true`` on an absent column is refused -- it is the opposite of a no-op.

        The flag is a statement that a column is there to be written over, so if it is not, the
        author is wrong about the frame. Unchecked, this is the subtler of the two mistakes: a
        misspelling quietly creates a new column, the column meant for repair keeps its old values,
        and the run reports success having changed nothing. It is a question about the schema, so a
        frame with no rows answers it exactly as a populated one does.
        """
        frame = _existing().head(height)
        config = _blanking_ladder() | {"output": "copmany"}

        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert "copmany" in message, f"refusal should name the column asked for: {message}"
        assert "output" in message, f"refusal should name the field: {message}"

    def test_the_two_tools_that_write_a_named_column_answer_this_identically(self) -> None:
        """Word for word the same refusal as Formula's, save for the field it happened at.

        Two tools answering the same question similarly is how a difference nobody intended gets
        established: an author who has read one refusal must not have to re-read the other. The one
        thing that legitimately differs is the path, because the field is in a different place in
        each configuration -- so that is substituted out and the rest must match exactly.
        """
        frame = _existing()

        with pytest.raises(ToolError) as ladder:
            run_tool(CaseWhen(), Ports({"rows": frame}), _blanking_ladder() | {"output": "copmany"})
        with pytest.raises(ToolError) as formula:
            run_tool(
                Formula(),
                Ports({"rows": frame}),
                {"columns": [{"name": "copmany", "value": "trim([company])", "replace": True}]},
            )

        assert str(ladder.value).replace("'output'", "<field>") == str(formula.value).replace(
            "'columns[0].name'", "<field>"
        )

    @pytest.mark.parametrize(
        ("output", "config", "expected", "dtype"),
        [
            pytest.param(
                "status",
                {
                    "cases": [{"when": "[amount] > 0", "then": "[amount] * 2"}],
                    "otherwise": "[amount]",
                    "output": "status",
                    "replace": True,
                },
                [20.0, 0.0, -5.0, 40.0, None, 15.0, 6.0],
                pl.Float64,
                id="text_becomes_a_number",
            ),
            pytest.param(
                "amount",
                {
                    "cases": [
                        {"when": "[amount] > 0", "then": "'positive'"},
                        {"when": "[amount] = 0", "then": "'zero'"},
                    ],
                    "otherwise": "'other'",
                    "output": "amount",
                    "replace": True,
                },
                ["positive", "zero", "other", "positive", "other", "positive", "positive"],
                pl.String,
                id="a_number_becomes_text",
            ),
        ],
    )
    def test_a_replaced_column_takes_whatever_type_its_branches_produce(
        self,
        output: str,
        config: Mapping[str, Any],
        expected: Sequence[Any],
        dtype: pl.DataType,
    ) -> None:
        """The new type is the branches' type; it does not have to match the type it replaces.

        Both directions are real steps: a code column read as a figure, and a figure collapsed into
        a named band. Requiring the old type would refuse them for no reason the data supports --
        the rule that matters is the one already enforced, that the branches agree with *each
        other*, and it is doing its work here unchanged.

        Every other column keeps its own type, which is what says the frame was written through
        rather than rebuilt: a rebuild is where a date quietly arrives back as text.
        """
        frame = _existing()

        result = run_tool(CaseWhen(), Ports({"rows": frame}), config)

        out = result["rows"]
        assert out[output].dtype == dtype
        assert out[output].to_list() == list(expected)
        assert out.columns == frame.columns, "a replacement of any type stays in place"
        assert {name: kind for name, kind in out.schema.items() if name != output} == {
            name: kind for name, kind in frame.schema.items() if name != output
        }, "no other column changed type"

    @pytest.mark.parametrize(
        ("config", "expected"),
        [
            pytest.param(
                _blanking_ladder() | {"otherwise": "'unmatched'"},
                ["unmatched", "superseded_co", "unmatched", "unmatched", "unmatched", None, "unmatched"],
                id="a_row_no_rule_reached_takes_otherwise_over_the_old_value",
            ),
            pytest.param(
                {
                    "cases": [{"when": "[status] = 'superseded'", "then": "'superseded_co'"}],
                    "output": "company",
                    "replace": True,
                },
                [None, "superseded_co", None, None, None, None, None],
                id="with_no_otherwise_a_row_no_rule_reached_is_null_not_left_as_it_was",
            ),
        ],
    )
    def test_a_row_no_rule_reached_does_not_keep_the_value_that_was_there(
        self,
        config: Mapping[str, Any],
        expected: Sequence[str | None],
    ) -> None:
        """Replacement writes the whole column, not the rows that matched something.

        The tempting reading of ``replace`` is "write where a rule matched and leave the rest", and
        it is wrong: the column a ladder produces is the ladder's, all of it. A row that matched no
        rule takes ``otherwise``, or null when there is none -- the same answer it would get in a
        new column -- rather than silently keeping its old value, which would give a column that is
        part old and part new with nothing marking which rows are which.
        """
        result = run_tool(CaseWhen(), Ports({"rows": _existing()}), config)

        assert result["rows"]["company"].to_list() == list(expected)

    def test_replacing_a_column_leaves_the_input_frame_untouched(self) -> None:
        """The boundary compares input schemas; a replacement is where the data could change too.

        Replacement is the one operation that could plausibly be done in place on the caller's own
        frame, and ``Ports`` copies the mapping and not the frames -- so where one output fans into
        two branches, an in-place write would rewrite the other branch's input.
        """
        frame = _existing()
        before = frame.clone()

        result = run_tool(CaseWhen(), Ports({"rows": frame}), _blanking_ladder())

        assert_frame_equal(frame, before)
        assert result["rows"]["company"].to_list() != frame["company"].to_list(), "the output did change"

    @pytest.mark.parametrize(
        ("config", "named"),
        [
            pytest.param(
                {
                    "cases": [
                        {"when": "[status] = 'void'", "then": "'unknown'"},
                        {"when": "[amount] > 0", "then": "[amount] * 2"},
                    ],
                    "output": "company",
                    "replace": True,
                },
                "cases[1].then",
                id="branches_still_have_to_agree_with_each_other",
            ),
            pytest.param(
                {
                    "cases": [{"when": "[company]", "then": "'named'"}],
                    "output": "company",
                    "replace": True,
                },
                "cases[0].when",
                id="a_condition_still_has_to_be_a_truth_value",
            ),
            pytest.param(
                {"cases": [], "otherwise": "'anything'", "output": "company", "replace": True},
                "cases",
                id="an_empty_ladder_is_still_refused",
            ),
            pytest.param(
                {
                    "cases": [{"when": "[staus] = 'void'", "then": "'unknown'"}],
                    "output": "company",
                    "replace": True,
                },
                "staus",
                id="a_column_that_is_not_there_is_still_refused",
            ),
        ],
    )
    def test_every_rule_the_ladder_already_enforced_is_enforced_when_replacing(
        self,
        config: Mapping[str, Any],
        named: str,
    ) -> None:
        """The flag decides where the column goes, and nothing else about the ladder.

        A flag that also relaxed the checks would be the worst version of this change: the
        configurations that use it are exactly the ones writing over data another step already
        reads, so they need the checks more, not less.
        """
        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _existing()}), config)

        assert named in str(caught.value), f"refusal should report {named!r}: {caught.value}"

    @pytest.mark.parametrize("height", [7, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_the_refusal_for_an_absent_column_lists_the_columns_that_are_there(self, height: int) -> None:
        """The refusal has to carry the columns that *are* present, not only the one that is not.

        The mistake this refusal catches is a misspelling, and a misspelling is obvious only beside
        the real list: ``copmany`` reads as a column name until ``company`` sits next to it. Without
        the list the author is told their column is absent and sent to find out what is there
        instead -- from a frame some earlier step produced, which is exactly the frame they do not
        have in front of them. The list is the answer rather than merely the complaint.
        """
        frame = _existing().head(height)

        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": frame}), _blanking_ladder() | {"output": "copmany"})

        message = str(caught.value)
        for name in frame.columns:
            assert name in message, f"refusal should list the present column {name!r}: {message}"

    def test_a_rule_reads_the_values_that_were_there_before_this_step_ran(self) -> None:
        """The second rung's condition matches a literal the first rung writes, and never fires on it.

        This is the ordering ruling stated in values rather than in types: every branch is compiled
        against the frame as it arrived, so the ladder is one decision per row taken against the old
        column, not a sequence of writes each reading the last one's output. A partially-written
        frame would label the rows the first rung claimed 'void', then re-read them under the second
        rung, and every one of them would come back 'unknown' -- a column where no row kept the
        answer its own first match gave it, and nothing recording that.
        """
        frame = _existing()
        config = {
            "cases": [
                {"when": "[status] = 'open'", "then": "'void'"},
                {"when": "[status] = 'void'", "then": "'unknown'"},
            ],
            "output": "status",
            "replace": True,
        }

        result = run_tool(CaseWhen(), Ports({"rows": frame}), config)

        assert result["rows"]["status"].to_list() == ["void", None, "void", "unknown", None, "void", None], (
            "a rung reads the old values: a row this ladder writes 'void' into is not then read as 'void'"
        )
        assert result["rows"].columns == frame.columns, "no column added, none moved"

    def test_the_flag_written_out_as_nothing_is_refused_like_the_flag_left_out(self) -> None:
        """``replace:`` with nothing after it arrives as ``None``, which is a shape of its own.

        ``apply_defaults`` fills a field that is *absent*; a key written with no value after it is
        present and holds ``None``, so the declared default never reaches it and a tool reading
        ``config["replace"]`` sees ``None`` rather than ``False``. Both mean "not asked" and both
        must refuse a name the input already carries -- but they arrive by different routes, and the
        route that skips the default is the one nobody thinks to write down.
        """
        template = CaseWhen().config_template()
        config = _blanking_ladder(replace=None) | {"replace": None}
        assert "replace" in config, "this is the key written out as nothing, not the key left out"

        validate_against_template(template, config)
        assert apply_defaults(template, config)["replace"] is None, (
            "a key that is present is left alone; only an absent key takes the declared default"
        )

        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _existing()}), config)

        message = str(caught.value)
        assert "company" in message, f"refusal should name the column that already exists: {message}"
        assert "replace" in message, f"refusal should say how to authorise it: {message}"

    @pytest.mark.parametrize("written", [True, False], ids=["written_true", "written_false"])
    def test_the_flag_written_inside_a_rule_is_refused_naming_where_it_sits(self, written: bool) -> None:
        """The likeliest mis-write, because Formula's flag genuinely is per-entry.

        An author who has read Formula writes the flag beside the value it governs; here there is
        one output column, the flag is top level, and a flag inside a rung governs nothing. It is
        refused as an unknown field rather than ignored, and the refusal names the path it sits at,
        because a silently ignored flag is a configuration that reads as authorising an overwrite
        and does not. What was written after it makes no difference: the field is wrong, not its
        value.
        """
        config = {
            "cases": [
                {"when": "[status] = 'superseded'", "then": "'superseded_co'"},
                {"when": "trim([company]) = ''", "then": "null", "replace": written},
            ],
            "otherwise": "trim([company])",
            "output": "company",
            "replace": True,
        }

        with pytest.raises(ToolError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _existing()}), config)

        assert "cases[1].replace" in str(caught.value), (
            f"refusal should name the path the stray flag sits at: {caught.value}"
        )


# ---------------------------------------------------------------------------
# The template's nested shape, and the catalog
# ---------------------------------------------------------------------------


def _declared_words(template: Mapping[str, ConfigField]) -> set[str]:
    """Every word a definition file has to carry for this template: field names and enum values."""
    words: set[str] = set()
    for name, spec in template.items():
        words.add(name)
        words |= _spec_words(spec)
    return words


def _spec_words(spec: ConfigField) -> set[str]:
    """The words one field declares below itself -- its enum, its members, its element's."""
    words: set[str] = set(spec.enum or ())
    if spec.fields:
        words |= _declared_words(spec.fields)
    if spec.items is not None:
        words |= _spec_words(spec.items)
    return words


_WHOLE_COLUMN_PHRASINGS: tuple[str, ...] = ("whole column", "entire column", "column in full", "all of it")
"""Ways of saying that a replacement writes every row, not only the rows a rule matched."""


def _sentences(text: str) -> list[str]:
    """``text`` split into sentences, so a claim can be read against the claim beside it.

    A statement is pinned sentence by sentence rather than against the whole blob, because a blob
    holding the word 'replace' in one paragraph and 'whole column' in another has not said the two
    belong together, which is the only thing a reader takes from it.
    """
    return [sentence.strip() for sentence in re.split(r"(?<=[.;:])\s+", text) if sentence.strip()]


class TestCaseWhenTemplateAndCatalog:
    """The loader seam reaches every branch, and the tool is described and registered."""

    def test_the_template_declares_ordered_rules_and_the_loader_seam_reaches_every_branch(self) -> None:
        """Both halves of the seam, on the configuration the nested shape was designed against.

        CaseWhen has one top-level expression field, the fallback, and every other expression it
        holds sits inside an element of ``cases``. A seam that reached only the top level would
        report one expression out of seventeen for the real ladder.
        """
        template = CaseWhen().config_template()

        assert set(template) == {"cases", "otherwise", "output", "replace"}
        assert template["cases"].type == "array"
        assert template["cases"].required is True
        assert template["output"].required is True
        assert template["output"].type == "string", "a column name, not an expression"
        assert template["otherwise"].type == "expression"
        assert template["otherwise"].required is False, "a ladder may leave unmatched rows null"

        rule = template["cases"].items
        assert rule is not None, "an untyped array would leave every rule unchecked"
        assert rule.type == "object"
        assert set(rule.fields or {}) == {"when", "then"}
        assert all((rule.fields or {})[member].type == "expression" for member in ("when", "then"))
        assert all((rule.fields or {})[member].required is True for member in ("when", "then"))

        assert expression_fields(template) == ("cases[].when", "cases[].then", "otherwise")
        assert expression_values(template, _LADDER)[:3] == (
            ("cases[0].when", "[status] in ('superseded', 'replaced')"),
            ("cases[0].then", "'superseded'"),
            ("cases[1].when", "[amount] = 0"),
        )
        assert len(expression_values(template, _LADDER)) == 17, "eight rules, two expressions each, and the fallback"
        assert expression_values(template, _LADDER)[-1] == ("otherwise", "'counted'")

    def test_the_template_declares_replace_as_one_top_level_flag_a_definition_file_can_carry(self) -> None:
        """Top level, boolean, defaulting to no -- and a word YAML hands over as written.

        Top level because this tool writes exactly one column: Formula's flag sits inside an element
        of ``columns`` only because Formula writes several, and there is nothing here to put it
        inside. That also gives it a real default, which a field inside an array element cannot
        have -- nothing fills members of an element that does not exist -- so a filled configuration
        says ``False`` rather than ``None``, and a tool reads it by subscript.

        The default is the refusal, not the overwrite: the flag exists so that an author says they
        meant it, and a default of true would mean nobody ever says anything.

        And the name has to survive a definition file. ``on`` was a real field on another tool and
        the obvious way to write that step handed the loader the key ``True``; ``replace`` is safe,
        but the check belongs on the template rather than on this one name, because the template is
        least read where it is deepest.
        """
        template = CaseWhen().config_template()

        replace = template["replace"]
        assert replace.type == "boolean"
        assert replace.required is False, "every ladder written before this flag existed stays valid"
        assert replace.default is False, "left out, the flag means no, and a filled configuration says so"
        assert replace.enum is None, "a flag has two values and the type already says which"
        assert apply_defaults(template, dict(_LADDER))["replace"] is False
        assert expression_fields(template) == ("cases[].when", "cases[].then", "otherwise"), (
            "a flag is not an expression and must not appear on the seam a loader pre-flights"
        )

        assert not _declared_words(template) & YAML_UNSAFE_VALUES, (
            "a field name or enum value a definition file rewrites puts the fix on every author"
        )

    def test_case_when_conforms_and_its_examples_show_a_row_falling_through(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output."""
        tool = CaseWhen()

        check_tool(tool)

        assert tool.name == "case_when"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert len(examples) >= 2
        assert any(len(example.configuration["cases"]) >= 3 for example in examples), (
            "a two-branch example would not exercise first-match-wins, which is the whole ordering"
        )
        assert any("otherwise" in example.configuration for example in examples), (
            "a row falling through to the fallback is the behaviour a reader must see"
        )
        replacements = [example for example in examples if example.configuration.get("replace")]
        assert replacements, "replacement is the behaviour most likely to surprise, so an example must show it"
        assert all(example.configuration["output"] in example.inputs["rows"].columns for example in replacements), (
            "an example claiming a replacement must write over a column its own input already carries"
        )
        assert all(example.outputs["rows"].columns == example.inputs["rows"].columns for example in replacements), (
            "a replacement adds no column and moves none, which the example's own output has to show"
        )

    def test_the_replace_field_is_the_shared_one_with_a_default_and_nothing_else_changed(self) -> None:
        """The flag is derived from the one canonical declaration, not spelled out again here.

        The wording of this flag is shared so that a reader who has met it on one catalog page has
        met it on every page: a tool that re-typed the description would drift from the others one
        edit at a time, and no reader is ever looking at two pages at once to notice. A copy that
        happens to read the same today is not the same thing as a derivation, which is why the
        derivation is pinned rather than the words.

        The default is the one thing this tool changes, and it has to change it: the canonical field
        carries no default because it also sits inside an array element, where a default is inert --
        nothing fills the members of an element that does not exist. Here it is top level, so the
        default is real, and it is a refusal.
        """
        replace = CaseWhen().config_template()["replace"]

        assert replace.description == REPLACE_FIELD.description, (
            "one canonical wording of this flag across the catalog, not a copy that reads the same today"
        )
        differs = {
            spec.name
            for spec in dataclasses.fields(ConfigField)
            if getattr(replace, spec.name) != getattr(REPLACE_FIELD, spec.name)
        }
        assert differs == {"default"}, (
            f"the default is the only thing a top-level position changes about the shared field; also changed: "
            f"{sorted(differs - {'default'})}"
        )
        assert replace == dataclasses.replace(REPLACE_FIELD, default=False)

    def test_the_help_says_a_replacement_writes_the_whole_column(self) -> None:
        """The surprise has to be written where a reader looks, not only where a test asserts it.

        A row no rung reached takes ``otherwise``, or null when there is none -- including under
        ``replace``, where the tempting reading is that the rows no rule matched keep what they had.
        The cost of leaving that unsaid is not a wrong tool, it is a correct tool used wrongly: an
        author repairs three rows of a wide frame with a two-rung ladder and no fallback, and nulls
        every other row of the column, having typed exactly what the page told them to. So the help
        has to say it in the same breath as it says 'replace', because two paragraphs that each say
        half of it have not said it.
        """
        help_text = CaseWhen().help()
        prose = _sentences(help_text.detailed_description) + [
            sentence for note in help_text.notes for sentence in _sentences(note)
        ]

        about_replacing = [sentence for sentence in prose if "replac" in sentence.lower()]
        assert about_replacing, "the help never mentions replacement, which is the behaviour most likely to surprise"

        stated = [
            sentence
            for sentence in about_replacing
            if any(phrase in sentence.lower() for phrase in _WHOLE_COLUMN_PHRASINGS)
        ]
        assert stated, (
            "a replacement writes the whole column, and the help has to say so where it says 'replace': "
            f"{about_replacing}"
        )
        assert any("otherwise" in sentence.lower() or "null" in sentence.lower() for sentence in stated), (
            "saying 'the whole column' without saying what an unmatched row gets leaves the reader to guess "
            f"the half that costs them data: {stated}"
        )

    def test_case_when_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "case_when" in REGISTRY
        assert isinstance(REGISTRY.get("case_when"), CaseWhen)


# ---------------------------------------------------------------------------
# An output that is not a name
#
# CaseWhen was the tool with no answer here at all: no ``non_empty`` on the field and no rule of its
# own, so ``output: ""`` wrote a column called ``''`` and ``output: "   "`` wrote one called
# ``'   '`` -- invisible in every rendering, unreferenceable in an expression, and a real run that
# reported success. With ``replace: true`` the same configuration was refused, but as a column the
# input does not carry, which is a sentence about the frame and sends the author to read the wrong
# file.
#
# Both of those now land on the shared rule in the contract, which is deliberately checked before
# the flag is read, so the two spellings cannot answer differently.
# ---------------------------------------------------------------------------


def _blank_output_ladder(name: str, *, replace: bool | None = None) -> dict[str, Any]:
    """The smallest legitimate ladder, writing to ``name``."""
    config: dict[str, Any] = {
        "cases": [{"when": "[amount] > 0", "then": "'positive'"}],
        "otherwise": "'other'",
        "output": name,
    }
    if replace is not None:
        config["replace"] = replace
    return config


class TestCaseWhenRefusesAnOutputThatIsNotAName:
    """A blank ``output`` is a configuration fault, and it is reported as one."""

    @pytest.mark.parametrize("name", ["", " ", "   ", "\t"])
    def test_a_blank_output_is_refused_rather_than_written_as_a_nameless_column(self, name: str) -> None:
        """Empty and whitespace-only alike, because neither is a name a reader can point at."""
        with pytest.raises(ConfigError) as caught:
            run_tool(CaseWhen(), Ports({"rows": _rows()}), _blank_output_ladder(name))

        message = str(caught.value)
        assert "'output'" in message, f"the refusal must name the field: {message}"
        assert "blank" in message and "whitespace" in message, f"the refusal must say what a name has to be: {message}"
        assert "does not have" not in message, f"a blank name is not an absent column: {message}"

    @pytest.mark.parametrize("name", ["", "   "])
    def test_the_replace_flag_does_not_change_the_refusal(self, name: str) -> None:
        """The same words with the flag as without it.

        This is the defect the flag introduced: under ``replace: true`` a blank used to be refused
        as a column the input does not carry, so one mistake had two unrelated explanations
        depending on a field that has nothing to do with it.
        """
        refusals = []
        for replace in (None, False, True):
            with pytest.raises(ConfigError) as caught:
                run_tool(CaseWhen(), Ports({"rows": _rows()}), _blank_output_ladder(name, replace=replace))
            refusals.append(str(caught.value))

        assert len(set(refusals)) == 1, f"one mistake, one explanation; got: {refusals}"
