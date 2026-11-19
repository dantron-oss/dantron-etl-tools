"""Expressions that were accepted and wrong, or refused without saying where or why.

Each of these used to compile and quietly do something other than what was written, or failed with
"could not be understood" and nothing more. Each is now refused by name, or explained.
"""

from datetime import date, datetime

import polars as pl
import pytest

from dantron_etl.expr import ExpressionError, compile_expr


def _refusal(text: str, columns=None) -> str:
    with pytest.raises(ExpressionError) as raised:
        compile_expr(text, columns)
    return str(raised.value)


class TestAHashIsNotAComment:
    def test_a_note_after_a_hash_is_refused_instead_of_dropping_the_rest(self) -> None:
        message = _refusal("[status] = 'PAID'      # only settled invoices and [amount] > 0")
        assert "'#' is not a comment" in message and "why" in message

    def test_a_hash_inside_quotes_or_a_column_name_is_kept(self) -> None:
        frame = pl.DataFrame({"code#": ["a#b", "x"]})
        assert frame.filter(compile_expr("[code#] = 'a#b'"))["code#"].to_list() == ["a#b"]

    def test_a_backslash_outside_quotes_is_refused(self) -> None:
        assert "backslash" in _refusal("[a] = 1 \\\n + 2")

    def test_a_backslash_inside_quotes_is_text(self) -> None:
        frame = pl.DataFrame({"path": ["a\\b", "ab"]})
        assert frame.filter(compile_expr("[path] = 'a\\b'"))["path"].to_list() == ["a\\b"]


class TestOnlyPlainLettersOutsideQuotes:
    @pytest.mark.parametrize("text", ["'x' = ｌｉｔ(0)", "ｃｏａｌｅｓｃｅ([a], 1) > 0", "[a] = 1 and ｃｏｌ(0) = 2"])
    def test_a_full_width_spelling_is_refused(self, text: str) -> None:
        assert "not part of this language" in _refusal(text)

    def test_the_internal_names_cannot_be_reached_through_parentheses(self) -> None:
        assert "reserved" in _refusal("(lit)(0) = 'x'")
        assert "reserved" in _refusal("[a] = 1 and (col)(0) = 2")

    def test_non_ascii_text_and_column_names_are_still_fine(self) -> None:
        frame = pl.DataFrame({"café": ["crème", "x"]})
        assert frame.filter(compile_expr("[café] = 'crème'"))["café"].to_list() == ["crème"]

    def test_curly_quotes_are_told_to_be_straight(self) -> None:
        assert "straight single quotes" in _refusal("[status] = ‘PAID’")


class TestASyntaxErrorSaysWhereAndHints:
    @pytest.mark.parametrize(
        ("text", "hint"),
        [
            ("[status] = 'PAID' AND [amount] > 0", "write 'and' in lower case"),
            ("[a] = 1 OR [b] = 2", "write 'or' in lower case"),
            ("[a] IS NULL", "lower case"),
            ("[status] <> 'PAID'", "'!='"),
            ("[a] = 1 && [b] = 2", "'and'"),
            ("[a] = 1 || [b] = 2", "'or'"),
            ("[s] like 'x%'", "starts_with()"),
            ("if([a] > 1, 1, 0)", "case_when"),
        ],
    )
    def test_a_spelling_from_another_language_gets_a_hint(self, text: str, hint: str) -> None:
        assert hint in _refusal(text)

    def test_the_place_is_given_in_the_authors_text(self) -> None:
        message = _refusal("[status] = 'PAID' AND [amount] > 0")
        assert "near character 19, at `AND [amount] > 0`" in message

    def test_text_that_stops_short_says_so(self) -> None:
        assert "ends before the expression is complete" in _refusal("[amount] > 0 and")

    def test_an_unclosed_parenthesis_is_pointed_at(self) -> None:
        assert "the '(' at character 9 is never closed" in _refusal("coalesce([a], ")

    def test_an_upper_case_word_inside_quotes_or_brackets_gives_no_hint(self) -> None:
        message = _refusal("[AND] = 'OR' +")
        assert "lower case" not in message

    def test_a_list_that_is_too_long_is_explained_in_values(self) -> None:
        message = _refusal("[a] in (" + ", ".join(f"'C{i}'" for i in range(130)) + ")")
        assert "about 120 quoted values" in message and "join" in message
