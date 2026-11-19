"""A rule value written into a longer piece of text, in an expression and in any other field.

A reference that is a whole field takes the rule's own value. A reference inside other text has to
become text, and that is only safe for a value whose text is not in doubt. A list, a mapping, a
truth value, a null and a decimal are each refused there, by name, in both kinds of field, rather
than written as Python would print them.
"""

from datetime import date
from typing import Any

import polars as pl
import pytest

from dantron_etl import PipelineError, load_pipeline
from dantron_etl.expr import compile_expr


def _document(predicate: str = "[code] = 'A'", path: str = "entries.csv") -> dict[str, Any]:
    return {
        "name": "rules",
        "steps": [
            {"id": "entries", "tool": "read_csv", "config": {"path": path}},
            {"id": "kept", "tool": "filter", "config": {"predicate": predicate}, "inputs": {"rows": "entries.rows"}},
            {
                "id": "written",
                "tool": "write_csv",
                "config": {"path": "out.csv", "overwrite": True},
                "inputs": {"rows": "kept.true"},
            },
        ],
    }


def _predicate(predicate: str, **rules: Any) -> str:
    return load_pipeline(_document(predicate=predicate), rules=rules).step("kept").config["predicate"]


def _path(path: str, **rules: Any) -> str:
    return load_pipeline(_document(path=path), rules=rules).step("entries").config["path"]


class TestInsideQuotesInAnExpression:
    @pytest.mark.parametrize(
        ("value", "named"),
        [
            (["A", "B"], "a list"),
            (("A", "B"), "a list"),
            ({"k": "A"}, "a mapping"),
            (1.5, "a decimal"),
        ],
    )
    def test_a_value_with_no_single_text_form_is_refused_by_name(self, value: Any, named: str) -> None:
        with pytest.raises(PipelineError) as raised:
            _predicate("[code] = '${rules.codes}'", codes=value)
        message = str(raised.value)
        assert "'kept'" in message and "'codes'" in message and named in message

    def test_a_list_inside_quotes_points_to_the_unquoted_spelling(self) -> None:
        with pytest.raises(PipelineError) as raised:
            _predicate("[code] = '${rules.codes}'", codes=["A", "B"])
        assert "[code] in ${rules.codes}" in str(raised.value) or "without quotes" in str(raised.value)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("O'Brian", "[code] = 'O''Brian'"), (7, "[code] = '7'"), (date(2026, 1, 31), "[code] = '2026-01-31'")],
    )
    def test_text_a_whole_number_and_a_date_are_written_as_their_text(self, value: Any, expected: str) -> None:
        assert _predicate("[code] = '${rules.code}'", code=value) == expected


class TestOutsideQuotesInAnExpression:
    def test_an_empty_list_is_refused_because_in_would_match_nothing(self) -> None:
        with pytest.raises(PipelineError) as raised:
            _predicate("[code] in ${rules.codes}", codes=[])
        assert "'codes'" in str(raised.value) and "empty" in str(raised.value)

    def test_a_list_of_one_is_still_a_list(self) -> None:
        resolved = _predicate("[code] in ${rules.codes}", codes=["A"])
        assert pl.DataFrame({"code": ["A", "B"]}).filter(compile_expr(resolved))["code"].to_list() == ["A"]

    def test_a_mapping_is_refused(self) -> None:
        with pytest.raises(PipelineError) as raised:
            _predicate("[code] = ${rules.codes}", codes={"k": "A"})
        assert "a mapping" in str(raised.value)

    def test_a_list_inside_a_list_is_refused(self) -> None:
        with pytest.raises(PipelineError) as raised:
            _predicate("[code] in ${rules.codes}", codes=[["A"], "B"])
        assert "'codes'" in str(raised.value)

    def test_a_date_is_written_as_a_date_literal(self) -> None:
        resolved = _predicate("[day] >= ${rules.start}", start=date(2026, 1, 1))
        assert resolved == "[day] >= date'2026-01-01'"
        days = pl.DataFrame({"day": [date(2025, 12, 31), date(2026, 1, 1)]})
        assert days.filter(compile_expr(resolved))["day"].to_list() == [date(2026, 1, 1)]


class TestInAFieldThatIsNotAnExpression:
    @pytest.mark.parametrize(
        ("value", "named"),
        [
            (False, "a truth value"),
            (True, "a truth value"),
            (None, "a null"),
            (["a", "b"], "a list"),
            ({"k": "a"}, "a mapping"),
            (1.5, "a decimal"),
        ],
    )
    def test_a_value_with_no_single_text_form_is_refused_by_name(self, value: Any, named: str) -> None:
        with pytest.raises(PipelineError) as raised:
            _path("out_${rules.flag}.csv", flag=value)
        message = str(raised.value)
        assert "'entries'" in message and "'flag'" in message and named in message
        assert "quote the value" in message.lower()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("north", "out_north.csv"), (3, "out_3.csv"), (date(2026, 1, 31), "out_2026-01-31.csv")],
    )
    def test_text_a_whole_number_and_a_date_are_written_as_their_text(self, value: Any, expected: str) -> None:
        assert _path("out_${rules.flag}.csv", flag=value) == expected

    def test_a_whole_field_reference_still_takes_the_rules_own_value(self) -> None:
        document = _document()
        document["steps"][0]["config"]["skip_rows"] = "${rules.skipped}"
        assert load_pipeline(document, rules={"skipped": 2}).step("entries").config["skip_rows"] == 2


class TestAnApostropheInsideAColumnName:
    def test_a_quote_inside_brackets_does_not_count_as_opening_text(self) -> None:
        resolved = _predicate("[client's name] = ${rules.who}", who="Acme")
        assert resolved == "[client's name] = 'Acme'"
        rows = pl.DataFrame({"client's name": ["Acme", "Other"]})
        assert rows.filter(compile_expr(resolved))["client's name"].to_list() == ["Acme"]

    def test_the_mirror_inside_quotes_after_such_a_column(self) -> None:
        resolved = _predicate("[client's name] = '${rules.who}'", who="O'Brian")
        assert resolved == "[client's name] = 'O''Brian'"
