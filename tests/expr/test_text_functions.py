"""left, right, substr, split_part, pad_left and pad_right: PostgreSQL's answers, counted in characters.

Every count is a whole number written in the file, like round()'s places, so a count the function
cannot use is refused when the file loads. A character is a character however many bytes it takes:
an accented letter and an emoji each count one.
"""

from typing import Any

import polars as pl
import pytest

from dantron_etl.expr import ExpressionError, compile_expr

WORDS = pl.DataFrame({"t": ["héllo😀x", "ab", "", None]})


def _values(text: str, frame: pl.DataFrame = WORDS) -> list[Any]:
    out = frame.select(compile_expr(text, frame.schema).alias("out"))["out"]
    assert out.dtype == pl.String
    return out.to_list()


def _refusal(text: str) -> str:
    """The refusal of ``text``, against text in [t], [d] and [f] and a whole number in [n]."""
    with pytest.raises(ExpressionError) as raised:
        compile_expr(text, {"t": pl.String, "d": pl.String, "f": pl.String, "n": pl.Int64})
    return str(raised.value)


class TestLeftAndRight:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("left([t], 6)", ["héllo😀", "ab", "", None]),
            ("right([t], 2)", ["😀x", "ab", "", None]),
            ("left([t], 0)", ["", "", "", None]),
            ("right([t], 0)", ["", "", "", None]),
            ("left([t], 99)", ["héllo😀x", "ab", "", None]),
            # A negative count is PostgreSQL's "all but": all but the last two, all but the first two.
            ("left([t], -2)", ["héllo", "", "", None]),
            ("right([t], -2)", ["llo😀x", "", "", None]),
        ],
    )
    def test_characters_from_the_start_or_the_end(self, text: str, expected: list[Any]) -> None:
        assert _values(text) == expected

    @pytest.mark.parametrize("count", ["[n]", "1.5", "'2'", "null", "65536", "-65536"])
    def test_the_count_is_a_whole_number_written_in_the_file(self, count: str) -> None:
        assert "left()" in _refusal(f"left([t], {count})")

    def test_compactly_stored_text_gives_the_same(self) -> None:
        frame = WORDS.with_columns(pl.col("t").cast(pl.Categorical))
        assert _values("right([t], 2)", frame) == ["😀x", "ab", "", None]

    def test_the_text_must_be_text(self) -> None:
        frame = pl.DataFrame({"n": [12345]})
        with pytest.raises(ExpressionError) as raised:
            compile_expr("left([n], 2)", frame.schema)
        assert "left()" in str(raised.value)


class TestSubstr:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("substr([t], 2, 3)", ["éll", "b", "", None]),
            ("substr([t], 1, 2)", ["hé", "ab", "", None]),
            ("substr([t], 6)", ["😀x", "", "", None]),
            ("substr([t], 1)", ["héllo😀x", "ab", "", None]),
            ("substr([t], 3, 0)", ["", "", "", None]),
            ("substr([t], 7, 5)", ["x", "", "", None]),
            ("substr([t], 50, 5)", ["", "", "", None]),
        ],
    )
    def test_start_counts_from_one_and_length_runs_to_the_end_when_left_out(
        self, text: str, expected: list[Any]
    ) -> None:
        assert _values(text) == expected

    @pytest.mark.parametrize(
        ("text", "said"),
        [
            ("substr([t], 0, 2)", "between 1 and"),
            ("substr([t], -1)", "between 1 and"),
            ("substr([t], 1, -1)", "between 0 and"),
            ("substr([t], [n])", "whole number"),
            ("substr([t])", "2 or 3 arguments"),
            ("substr([t], 1, 2, 3)", "2 or 3 arguments"),
        ],
    )
    def test_a_start_before_1_a_negative_length_and_a_wrong_count_of_arguments_are_refused(
        self, text: str, said: str
    ) -> None:
        refusal = _refusal(text)
        assert "substr()" in refusal and said in refusal, refusal


class TestSplitPart:
    LINES = pl.DataFrame({"t": ["a,b,,c", "solo", "", None]})

    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (1, ["a", "solo", "", None]),
            (2, ["b", "", "", None]),
            (3, ["", "", "", None]),
            (4, ["c", "", "", None]),
            (5, ["", "", "", None]),
            # Negative counts from the end, as PostgreSQL 14 does.
            (-1, ["c", "solo", "", None]),
            (-4, ["a", "", "", None]),
            (-5, ["", "", "", None]),
        ],
    )
    def test_the_nth_part_counting_from_one_and_empty_text_past_the_last(self, n: int, expected: list[Any]) -> None:
        assert _values(f"split_part([t], ',', {n})", self.LINES) == expected

    def test_the_delimiter_is_matched_exactly_and_may_be_longer_than_one_character(self) -> None:
        frame = pl.DataFrame({"t": ["2026.01--x😀y", "a.b"]})
        assert _values("split_part([t], '.', 2)", frame) == ["01--x😀y", "b"]
        assert _values("split_part([t], '--', 2)", frame) == ["x😀y", ""]
        assert _values("split_part([t], '😀', 2)", frame) == ["y", ""]

    @pytest.mark.parametrize(
        ("text", "said"),
        [
            ("split_part([t], '', 1)", "empty"),
            ("split_part([t], [d], 1)", "quoted"),
            ("split_part([t], ',', 0)", "other than 0"),
            ("split_part([t], ',', [n])", "whole number"),
        ],
    )
    def test_an_empty_delimiter_a_part_number_of_0_and_a_column_are_refused(self, text: str, said: str) -> None:
        refusal = _refusal(text)
        assert "split_part()" in refusal and said in refusal, refusal


class TestPadding:
    CODES = pl.DataFrame({"t": ["7", "1234", "", None, "😀"]})

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("pad_left([t], 3, '0')", ["007", "1234", "000", None, "00😀"]),
            ("pad_right([t], 3, '.')", ["7..", "1234", "...", None, "😀.."]),
            ("pad_left([t], 3, 'é')", ["éé7", "1234", "ééé", None, "éé😀"]),
            ("pad_left([t], 0, ' ')", ["7", "1234", "", None, "😀"]),
        ],
    )
    def test_text_is_padded_to_the_length_and_never_cut(self, text: str, expected: list[Any]) -> None:
        assert _values(text, self.CODES) == expected

    @pytest.mark.parametrize(
        ("text", "said"),
        [
            ("pad_left([t], 3, '')", "one character"),
            ("pad_right([t], 3, 'ab')", "one character"),
            ("pad_left([t], 3, [f])", "one character"),
            ("pad_left([t], -1, '0')", "between 0 and"),
            ("pad_left([t], [n], '0')", "whole number"),
        ],
    )
    def test_a_fill_that_is_not_one_character_and_a_negative_length_are_refused(self, text: str, said: str) -> None:
        refusal = _refusal(text)
        assert ("pad_left()" in refusal or "pad_right()" in refusal) and said in refusal, refusal
