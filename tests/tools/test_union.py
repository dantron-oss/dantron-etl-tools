"""Tests for the Union tool: two frames carried on as one, with nothing invented between them.

Union is the reshaping every other tool leaves undone. Filter, RemoveDuplicates and Join each split
a frame into named ports, and none of them puts two frames back together -- so a flow that routes
rows apart to treat them differently cannot carry them on as one, and a flow reading the same thing
out of two sources has to run every later step twice.

The ruling these tests exist to protect is the refusal, not the stacking. Aligning two schemas by
inventing nulls is the quiet version of a join that lost half its rows: what comes back has the
shape somebody expected, with a column of nothing where the data was, and the step that reads it
cannot tell the difference. So a column present on one side and absent on the other stops the run,
and a flow that genuinely wants it says so with a Formula, where the invention is written down.

Column *order* is the deliberate exception. Two frames carrying the same columns in a different
order are the same data; refusing them would send an author to write a SelectColumns whose only
purpose is to satisfy this tool. The order one side happens to have is not a fact about the data,
where a missing column is.

Vocabulary is generic: the library carries no caller's nouns.
"""

from datetime import date

import polars as pl
import pytest

from dantron_etl.authoring import CATEGORIES, ConfigError, Ports, REGISTRY, check_tool, run_tool
from dantron_etl.tools.union import Union


def _top() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "record_id": [1, 2],
            "code": ["A1", "B2"],
            "amount": [10.0, -5.0],
            "seen": [date(2026, 1, 1), date(2026, 2, 1)],
        }
    )


def _bottom() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "record_id": [3],
            "code": [None],
            "amount": [7.5],
            "seen": [date(2026, 3, 1)],
        }
    )


def _stack(top: pl.DataFrame, bottom: pl.DataFrame) -> pl.DataFrame:
    """One run through the boundary, so the envelope is enforced on every case here."""
    return run_tool(Union(), Ports({"top": top, "bottom": bottom}), {})["rows"]


class TestStacking:
    """Every row of both sides, in one order, with nothing added and nothing taken away."""

    def test_every_row_of_top_then_every_row_of_bottom(self) -> None:
        rows = _stack(_top(), _bottom())

        assert rows.height == 3, "row conservation: neither side loses anything"
        assert rows["record_id"].to_list() == [1, 2, 3], "all of top, then all of bottom"
        assert rows.schema == _top().schema

    def test_a_null_is_carried_rather_than_treated_as_a_missing_column(self) -> None:
        """A column that is present and null is data; that is a different thing from absent."""
        rows = _stack(_top(), _bottom())

        assert rows["code"].to_list() == ["A1", "B2", None]

    def test_column_order_may_differ_and_is_taken_from_top(self) -> None:
        rows = _stack(_top(), _bottom().select("seen", "amount", "code", "record_id"))

        assert rows.columns == _top().columns
        assert rows["record_id"].to_list() == [1, 2, 3]

    def test_nothing_is_de_duplicated(self) -> None:
        """Stacking is not judging. A flow wanting one of each says so, where it is visible."""
        rows = _stack(_top(), _top())

        assert rows.height == 4
        assert rows["record_id"].to_list() == [1, 2, 1, 2]

    @pytest.mark.parametrize("side", ["top", "bottom"])
    def test_an_empty_side_leaves_the_other_exactly_as_it_was(self, side: str) -> None:
        """An empty branch is an ordinary day, not a reason to stop or to change the other side."""
        full, empty = _top(), _top().head(0)
        rows = _stack(empty, full) if side == "top" else _stack(full, empty)

        assert rows.height == 2
        assert rows.schema == full.schema
        assert rows["record_id"].to_list() == [1, 2]

    def test_two_empty_sides_give_an_empty_frame_that_still_carries_its_columns(self) -> None:
        rows = _stack(_top().head(0), _top().head(0))

        assert rows.height == 0
        assert rows.columns == _top().columns

    def test_union_leaves_its_inputs_untouched(self) -> None:
        top, bottom = _top(), _bottom()
        before_top, before_bottom = top.clone(), bottom.clone()

        _stack(top, bottom)

        assert top.equals(before_top)
        assert bottom.equals(before_bottom)


class TestRefusals:
    """A shape that would have to be invented stops the run, and the message says what and where."""

    def test_a_column_on_one_side_only_is_refused_rather_than_filled_with_nulls(self) -> None:
        """The whole argument for the tool: a filled column looks exactly like a real one."""
        with pytest.raises(ConfigError) as caught:
            _stack(_top(), _bottom().drop("amount"))

        message = str(caught.value)
        assert "amount" in message
        assert "Formula" in message, "the refusal has to say how to do it deliberately"

    def test_an_extra_column_on_bottom_is_refused_and_named(self) -> None:
        """Dropping it silently would lose data that somebody put there on purpose."""
        with pytest.raises(ConfigError) as caught:
            _stack(_top(), _bottom().with_columns(note=pl.lit("extra")))

        assert "note" in str(caught.value)

    def test_the_same_column_with_different_types_is_refused_naming_both(self) -> None:
        """Text stacked onto a number is a column of text that used to be money."""
        with pytest.raises(ConfigError) as caught:
            _stack(_top(), _bottom().with_columns(amount=pl.col("amount").cast(pl.String)))

        message = str(caught.value)
        assert "amount" in message
        assert "Float64" in message and "String" in message, "'they differ' sends a reader back to the data"

    @pytest.mark.parametrize("height", [0, 1])
    def test_a_shape_refusal_does_not_wait_for_rows(self, height: int) -> None:
        """Decided from the two schemas, so an empty frame earns the same refusal a full one does."""
        with pytest.raises(ConfigError):
            _stack(_top().head(height), _bottom().drop("code").head(height))


class TestUnionInTheCatalog:
    """Declared, described, and documented with examples that run."""

    def test_union_conforms_and_its_examples_stack_rather_than_merely_run(self) -> None:
        tool = Union()
        check_tool(tool)

        assert tool.name == "union"
        assert tool.category in CATEGORIES
        assert tool.input_ports == ("top", "bottom")
        assert tool.output_ports == ("rows",)
        assert tool.config_template() == {}, "there is nothing here to configure, and that is the design"

        for example in tool.help().examples:
            total = example.inputs["top"].height + example.inputs["bottom"].height
            assert example.outputs["rows"].height == total, "an example must demonstrate row conservation"
            assert example.check_row_order is True, "the order is the promise; an example has to show it"

    def test_union_is_in_the_shipped_catalog(self) -> None:
        assert REGISTRY.get("union") is not None


class TestAnAllNullColumnTakesTheOtherSidesType:
    """The one exception to "dtypes must agree", and it is the common case rather than a corner.

    A column no row on one side happened to populate comes back typed as nothing at all -- polars
    calls it Null. Refusing to stack it would refuse an ordinary frame: a branch a filter sent
    nowhere, a column only the other source fills. Null is the one value that belongs to every
    family, which is what this library's expression language already says about it, so the column
    takes the type the data actually has.
    """

    def test_a_null_column_on_bottom_takes_tops_type(self) -> None:
        top = pl.DataFrame({"code": ["A1"], "amount": [10.0]})
        bottom = pl.DataFrame({"code": [None], "amount": [7.5]}, schema={"code": pl.Null, "amount": pl.Float64})

        rows = _stack(top, bottom)

        assert rows.schema["code"] == pl.String, "the type the data has, not the absence of one"
        assert rows["code"].to_list() == ["A1", None]

    def test_a_null_column_on_top_takes_bottoms_type(self) -> None:
        """Both directions, because which side happens to be empty is an accident of the data."""
        top = pl.DataFrame({"code": [None], "amount": [10.0]}, schema={"code": pl.Null, "amount": pl.Float64})
        bottom = pl.DataFrame({"code": ["C3"], "amount": [7.5]})

        rows = _stack(top, bottom)

        assert rows.schema["code"] == pl.String
        assert rows["code"].to_list() == [None, "C3"]

    def test_two_null_columns_stay_null_rather_than_being_guessed_at(self) -> None:
        """Neither side says what it is, so nothing here invents an answer."""
        frame = pl.DataFrame({"code": [None], "amount": [1.0]}, schema={"code": pl.Null, "amount": pl.Float64})

        rows = _stack(frame, frame)

        assert rows.schema["code"] == pl.Null
        assert rows.height == 2

    def test_a_real_type_disagreement_is_still_refused(self) -> None:
        """The exception is about the absence of a type, never about two types that disagree."""
        top = pl.DataFrame({"code": ["A1"], "amount": [10.0]})
        bottom = pl.DataFrame({"code": ["C3"], "amount": [7]}, schema={"code": pl.String, "amount": pl.Int64})

        with pytest.raises(ConfigError) as caught:
            _stack(top, bottom)

        assert "amount" in str(caught.value)
