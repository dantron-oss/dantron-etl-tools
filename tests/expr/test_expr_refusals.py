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


FRAME = pl.DataFrame(
    {
        "s": ["abc", "de", None],
        "t": [datetime(2025, 1, 31, 9, 30), datetime(2025, 1, 31), datetime(2025, 2, 1)],
        "d": [date(2025, 1, 31)] * 3,
        "u": pl.Series([1, 2, 3], dtype=pl.UInt32),
        "dec": pl.Series([100, 7, 1], dtype=pl.Decimal(12, 2)),
    }
)


class TestComparingWithNullIsRefused:
    @pytest.mark.parametrize(
        ("text", "fix"),
        [
            ("[s] = null", "is null"),
            ("null = [s]", "is null"),
            ("[s] != null", "is not null"),
            ("null != [s]", "is not null"),
            ("[u] > null", "is null"),
        ],
    )
    def test_with_a_pointer_to_is_null(self, text: str, fix: str) -> None:
        message = _refusal(text)
        assert "never true" in message and fix in message

    def test_is_null_still_works(self) -> None:
        assert FRAME.select(compile_expr("[s] is null", FRAME.schema))["s"].to_list() == [False, False, True]


class TestRegexExtractWithAGroupThePatternLacks:
    @pytest.mark.parametrize(
        ("text", "said"),
        [
            ("regex_extract([s], '[a-z]+', 1)", "has no groups"),
            ("regex_extract([s], '([a-z])', 2)", "has 1 group"),
            ("regex_extract([s], '(a)(?:b)(c)', 3)", "has 2 groups"),
        ],
    )
    def test_is_refused_naming_how_many_groups_there_are(self, text: str, said: str) -> None:
        assert said in _refusal(text)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("regex_extract([s], '[a-z]+', 0)", ["abc", "de", None]),
            ("regex_extract([s], '([a-z])', 1)", ["a", "d", None]),
            ("regex_extract([s], '(a)(?:b)(c)', 2)", ["c", None, None]),
        ],
    )
    def test_a_group_the_pattern_has_is_extracted(self, text: str, expected: list) -> None:
        assert FRAME.select(compile_expr(text, FRAME.schema).alias("x"))["x"].to_list() == expected


class TestATimestampAgainstADate:
    @pytest.mark.parametrize(
        "text",
        [
            "[t] = date'2025-01-31'",
            "[t] != date'2025-01-31'",
            "[t] <= date'2025-01-31'",
            "[t] > date'2025-01-31'",
            "date'2025-01-31' = [t]",
            "date'2025-01-31' >= [t]",
            "date'2025-01-31' < [t]",
            "[t] = [d]",
            "[d] >= [t]",
            "[t] in (date'2025-01-31', date'2025-02-01')",
        ],
    )
    def test_is_refused_where_reading_the_date_as_midnight_is_wrong(self, text: str) -> None:
        message = _refusal(text, FRAME.schema)
        assert "midnight" in message and "cast_columns" in message

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[t] >= date'2025-01-31'", [True, True, True]),
            ("[t] < date'2025-02-01'", [True, True, False]),
            ("date'2025-02-01' <= [t]", [False, False, True]),
            ("date'2025-02-01' > [t]", [True, True, False]),
        ],
    )
    def test_the_two_comparisons_that_midnight_answers_correctly_still_work(self, text: str, expected: list) -> None:
        assert FRAME.select(compile_expr(text, FRAME.schema).alias("x"))["x"].to_list() == expected

    def test_two_timestamps_and_two_dates_are_compared_as_before(self) -> None:
        assert FRAME.select(compile_expr("[t] = [t]", FRAME.schema).alias("x"))["x"].to_list() == [True] * 3
        assert FRAME.select(compile_expr("[d] = date'2025-01-31'", FRAME.schema).alias("x"))["x"].to_list() == [True] * 3

    def test_without_a_schema_nothing_is_known_and_nothing_is_refused(self) -> None:
        compile_expr("[t] = date'2025-01-31'")


class TestDividingADecimalColumn:
    @pytest.mark.parametrize("text", ["[dec] / 3", "3 / [dec]", "[dec] / [dec]"])
    def test_is_refused_with_a_pointer_to_a_cast(self, text: str) -> None:
        message = _refusal(text, FRAME.schema)
        assert "[dec] holds Decimal" in message and "Float64" in message

    def test_other_arithmetic_on_a_decimal_column_is_unchanged(self) -> None:
        out = FRAME.select(compile_expr("[dec] * 2 + 1", FRAME.schema).alias("x"))["x"]
        assert isinstance(out.dtype, pl.Decimal)


class TestUnsignedArithmeticDoesNotWrap:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("len([s]) - 5", [-2, -3, None]),
            ("5 - len([s])", [2, 3, None]),
            ("len([s]) - len([s]) - 1", [-1, -1, None]),
            ("-len([s])", [-3, -2, None]),
            ("[u] - 5", [-4, -3, -2]),
            ("-[u]", [-1, -2, -3]),
        ],
    )
    def test_the_result_is_a_signed_whole_number(self, text: str, expected: list) -> None:
        out = FRAME.select(compile_expr(text, FRAME.schema).alias("x"))["x"]
        assert out.to_list() == expected
        assert out.dtype == pl.Int64

    def test_len_on_its_own_still_gives_what_it_gave(self) -> None:
        assert FRAME.select(compile_expr("len([s])", FRAME.schema).alias("x"))["x"].dtype == pl.UInt32
