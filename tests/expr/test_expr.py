"""Tests for the safe expression language that compiles author-written text into polars expressions.

A tool's configuration is a file on disk, edited by someone who is not a developer. So a predicate
arrives as text -- ``[status] = 'PAID' and [amount] > 0`` -- and this module turns it into a polars
expression. That makes the module an execution boundary, and the tests below are weighted
accordingly: roughly half of them are refusals.

Two rules govern the whole file:

1. Behaviour is asserted by **evaluating** against a frame, never by comparing expression reprs.
   A repr assertion would pin the implementation; the values are the contract.
2. Refusals are asserted **by AST node type** wherever the node type is deterministic, so the
   whitelist is pinned as a whitelist. A test that only tries a handful of known attacks would be
   satisfied by an implementation that blacklists that handful; the coverage test below instead
   walks every expression node type Python has and demands that everything not explicitly allowed
   is refused.

Vocabulary in this file is deliberately generic (code, label, amount, qty, flag, note, start_date)
because the module under test ships as a standalone library and must carry no caller's nouns.
"""

import ast
import dataclasses
import math
import re
import unicodedata
import uuid
from datetime import date
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any

import polars as pl
import pytest

from dantron_etl.expr import (
    ALLOWED_FUNCTIONS,
    ALLOWED_NODES,
    MAX_DEPTH,
    MAX_NODES,
    MAX_TEXT_LENGTH,
    ExpressionError,
    compile_expr,
    references,
)
from dantron_etl.expr.arguments import Value
from dantron_etl.expr.families import Family
from dantron_etl.expr.functions import DEFINITIONS, FUNCTIONS, Function

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def frame() -> pl.DataFrame:
    """A four-row frame whose last row is null in every column.

    Every column is nullable on purpose: null handling is the part of this language most likely to
    disagree silently with the engine it replaces, so every operator gets evaluated against a null.
    The ``note`` column holds text that looks like expression syntax (``a=b``, ``x [y]``, ``and``),
    which pins that a string literal is scanned as opaque text rather than rewritten.

    ``category`` and ``grade`` hold the *same values* as ``label`` in the two dtypes an engine uses
    to store text compactly -- a dictionary-encoded column, and one restricted to a fixed set of
    members. They are here because an author cannot see the difference in their data: a column of
    statuses read back from a store arrives in one of these dtypes without anybody choosing it, and
    every text operation must behave on it exactly as it behaves on ``label``. Holding the same
    values is what lets that be asserted as an equality rather than as a hand-written expectation.
    """
    labels = ["  alpha  ", "Beta", "gamma", None]
    return pl.DataFrame(
        {
            "code": ["A", "B", "C", None],
            "label": labels,
            "note": ["a=b", "x [y]", "and", None],
            "amount": [10.256, -2.4, 100.0, None],
            "qty": [1, 2, 3, None],
            "flag": [True, False, True, None],
            "start_date": [date(2026, 1, 1), date(2026, 6, 15), date(2025, 12, 31), None],
            "category": pl.Series(labels, dtype=pl.Categorical),
            # The members include the literals the sweep below compares against ('A', 'B', 'Z') as
            # well as the column's own values. A restricted column meeting a literal that is not one
            # of its members is a *separate* defect -- the engine refuses the literal rather than
            # answering false -- and putting it in here would make this sweep about two things.
            "grade": pl.Series(labels, dtype=pl.Enum(["  alpha  ", "Beta", "gamma", "A", "B", "Z"])),
        }
    )


def evaluate(frame: pl.DataFrame, text: str) -> list[Any]:
    """The column of values produced by compiling and evaluating ``text`` against ``frame``."""
    return frame.select(compile_expr(text).alias("out"))["out"].to_list()


def scalar(text: str) -> Any:
    """The single value produced by an expression that names no column."""
    return pl.DataFrame({"anchor": [1]}).select(compile_expr(text).alias("out"))["out"].to_list()[0]


def message(text: str, columns: list[str] | tuple[str, ...] | None = None) -> str:
    """The error text raised by compiling ``text``; fails the test if it compiles."""
    with pytest.raises(ExpressionError) as caught:
        compile_expr(text) if columns is None else compile_expr(text, columns=columns)
    return str(caught.value)


# ---------------------------------------------------------------------------
# The refusal corpus: one snippet per expression node type that is NOT whitelisted
# ---------------------------------------------------------------------------

# Keyed by ast node class name so the coverage test can compare against ast.expr.__subclasses__().
# Several of these snippets cannot even reach the whitelist walker, because '[' is reserved for a
# column reference and '=' is rewritten to a comparison -- a list display, a subscript and a walrus
# are all mangled on the way in. They are listed anyway: the contract is that each is *refused*,
# not that a particular check is the one that refuses it.
DISALLOWED_NODE_SNIPPETS: dict[str, str] = {
    "NamedExpr": "(x := 1)",
    "Lambda": "lambda: 1",
    "IfExp": "1 if true else 2",
    "Dict": "{'a': 1}",
    "Set": "{1, 2}",
    "List": "[1, 2]",
    "ListComp": "[x for x in y]",
    "SetComp": "{x for x in y}",
    "DictComp": "{k: v for k, v in y}",
    "GeneratorExp": "(x for x in y)",
    "Await": "await x",
    "Yield": "(yield 1)",
    "YieldFrom": "(yield from x)",
    "FormattedValue": "f'{code}'",
    "JoinedStr": "f'a{code}b'",
    "Attribute": "().__class__.__bases__",
    "Subscript": "coalesce([code], 'Z')[0]",
    "Starred": "round(*(1, 2))",
    "Slice": "coalesce([code], 'Z')[0:1]",
}


class TestOperators:
    """Every operator in the language compiles and evaluates to the right values."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[code] = 'A'", [True, False, False, None]),
            ("[code] != 'A'", [False, True, True, None]),
            ("[amount] > 0", [True, False, True, None]),
            ("[amount] < 0", [False, True, False, None]),
            ("[qty] >= 2", [False, True, True, None]),
            ("[qty] <= 2", [True, True, False, None]),
            ("[qty] + 1", [2, 3, 4, None]),
            ("[qty] - 1", [0, 1, 2, None]),
            ("[qty] * 3", [3, 6, 9, None]),
            ("[amount] / 2", [5.128, -1.2, 50.0, None]),
            ("-[qty]", [-1, -2, -3, None]),
            ("([qty] + 1) * 2", [4, 6, 8, None]),
            ("[flag] and [qty] > 1", [False, False, True, None]),
            ("[flag] or [qty] > 2", [True, False, True, None]),
            ("not [flag]", [False, True, False, None]),
            ("[code] in ('A', 'C')", [True, False, True, None]),
            ("[code] not in ('A', 'C')", [False, True, False, None]),
            ("[code] in ('A')", [True, False, False, None]),
            ("[code] is null", [False, False, False, True]),
            ("[code] is not null", [True, True, True, False]),
            ("[start_date] >= date'2026-01-01'", [True, True, False, None]),
            ("[start_date] = date'2026-06-15'", [False, True, False, None]),
        ],
    )
    def test_operator_matrix_evaluates_to_the_expected_values(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """Each operator, evaluated against the frame, produces the documented column of values."""
        assert evaluate(frame, text) == expected

    @pytest.mark.parametrize(
        ("text", "expected", "wrong"),
        [
            # '*' before '+': reading left to right instead gives (qty + 2) * 3.
            ("[qty] + 2 * 3", [7, 8, 9, None], [9, 12, 15, None]),
            ("[qty] + [qty] * 2", [3, 6, 9, None], [4, 8, 12, None]),
            # '/' before '-': reading left to right instead gives (12 - qty) * 6 / 3.
            ("12 - [qty] * 6 / 3", [10.0, 8.0, 6.0, None], [22.0, 20.0, 18.0, None]),
            # unary minus binds to the column, not to the sum: -(qty + 3) would give -4, -5, -6.
            ("-[qty] + 3", [2, 1, 0, None], [-4, -5, -6, None]),
            # 'not' before 'and': not (flag and flag) would just invert flag.
            ("not [flag] and [flag]", [False, False, False, None], [False, True, False, None]),
            # 'and' before 'or': ((code = A) or (code = B)) and false would be false everywhere.
            ("[code] = 'A' or [code] = 'B' and false", [True, False, False, None], [False] * 4),
            # ... and parentheses do exactly that.
            ("([code] = 'A' or [code] = 'B') and false", [False] * 4, [True, False, False, None]),
            ("(12 - [qty]) * 6 / 3", [22.0, 20.0, 18.0, None], [10.0, 8.0, 6.0, None]),
        ],
    )
    def test_precedence_and_associativity_are_not_left_to_right(
        self, frame: pl.DataFrame, text: str, expected: list[Any], wrong: list[Any]
    ) -> None:
        """not > and > or, and * / before + -, with parentheses overriding.

        Each case carries the plausible-but-different answer a naive left-to-right evaluation would
        produce, and the test asserts that answer is *not* what comes back -- so the case cannot
        pass by coincidence.
        """
        assert evaluate(frame, text) == expected
        assert evaluate(frame, text) != wrong

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("true", True),
            ("false", False),
            ("null", None),
            ("42", 42),
            ("-7", -7),
            ("1.5", 1.5),
            ("-0.25", -0.25),
            ("'hello'", "hello"),
            ("''", ""),
            ("'it is 5 > 3'", "it is 5 > 3"),
            ("date'2026-01-01'", date(2026, 1, 1)),
        ],
    )
    def test_literal_forms_compile_to_their_values(self, text: str, expected: Any) -> None:
        """The literal forms are: integers, floats, single-quoted text, true/false/null, date'...'.

        The date form is prefix-tagged (``date'2026-01-01'``) rather than a postfix cast
        (``'2026-01-01'::date``). Reasons: it keeps a date a *literal* instead of introducing a cast
        operator, whose obvious next step is ``::int`` and ``::text`` -- growing the grammar, which
        this design forbids; and it rewrites with one unambiguous scan rather than looking backwards
        at the token before ``::``.
        """
        assert scalar(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[note] = 'a=b'", [True, False, False, None]),
            ("[note] = 'x [y]'", [False, True, False, None]),
            ("[note] = 'and'", [False, False, True, None]),
            ("contains([note], '=')", [True, False, False, None]),
        ],
    )
    def test_text_literal_containing_syntax_characters_is_kept_verbatim(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """A literal may contain '=', '[' or a keyword: rewriting must skip over quoted text.

        This is the trap in a rewrite-then-parse design. Turning ``=`` into ``==`` or ``[x]`` into a
        column reference with a blind scan silently corrupts any literal that contains those
        characters, and the corruption looks like a data problem rather than a parser problem.
        """
        assert evaluate(frame, text) == expected


class TestFunctions:
    """Only whitelisted functions exist, and each one behaves as documented."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("coalesce([code], 'Z')", ["A", "B", "C", "Z"]),
            ("coalesce([code], [label], 'Z')", ["A", "B", "C", "Z"]),
            ("round([amount], 0)", [10.0, -2.0, 100.0, None]),
            ("round([amount] * 2, 0)", [21.0, -5.0, 200.0, None]),
            ("abs([amount])", [10.256, 2.4, 100.0, None]),
            ("lower([code])", ["a", "b", "c", None]),
            ("upper([code])", ["A", "B", "C", None]),
            ("strip([label])", ["alpha", "Beta", "gamma", None]),
            ("len([label])", [9, 4, 5, None]),
            ("starts_with([label], 'Be')", [False, True, False, None]),
            ("ends_with([label], 'ma')", [False, False, True, None]),
            ("contains([label], 'et')", [False, True, False, None]),
            ("regex_extract([label], '([a-z]+)', 1)", ["alpha", "eta", "gamma", None]),
            ("upper(strip([label]))", ["ALPHA", "BETA", "GAMMA", None]),
            ("coalesce([code], 'Z') in ('A', 'Z')", [True, False, False, True]),
        ],
    )
    def test_function_matrix_evaluates_to_the_expected_values(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """Each whitelisted function, evaluated against the frame, produces the documented values.

        ``len`` is the character count of text, not a row count. ``strip`` removes leading and
        trailing whitespace only. Functions nest.
        """
        assert evaluate(frame, text) == expected

    def test_contains_matches_a_literal_substring_and_not_a_pattern(self, frame: pl.DataFrame) -> None:
        """``contains`` is a plain substring test; ``regex_extract`` is the only regex door.

        ``'a.p'`` discriminates: as a regex it matches 'alpha' (a, any character, p), as literal
        text it matches nothing. Someone filtering on an account code that contains a dot must not
        have that dot silently mean 'any character'.
        """
        assert evaluate(frame, "contains([label], 'a.p')") == [False, False, False, None]
        assert evaluate(frame, "contains([label], 'lph')") == [True, False, False, None]
        assert evaluate(frame, "regex_extract([label], 'a(.)p', 1)") == ["l", None, None, None]


class TestTheFunctionTable:
    """Each function is defined once, in one entry that carries everything about it."""

    @staticmethod
    def incomplete(name: str, function: Function) -> list[str]:
        """What is missing or inconsistent in one entry, as plain sentences. Empty when it is complete."""
        problems: list[str] = []
        if function.name != name:
            problems.append(f"is filed under {name!r} but named {function.name!r}")
        if not function.params:
            problems.append("lists no arguments")
        if function.variadic and len(function.params) != 1:
            problems.append("is variadic, so it must list exactly one argument for all of them to meet")
        if function.result != "common" and not isinstance(function.result, Family):
            problems.append(f"gives {function.result!r}, which is neither a family nor 'common'")
        if function.result == "common" and not (function.variadic and function.params[0].family is None):
            problems.append("gives the family its arguments agree on, so it must take any number of any family")
        if function.result_dtype == "subject" and not isinstance(function.params[0], Value):
            problems.append("hands on its first argument's type, so the first argument must be a value")
        if not callable(function.to_polars):
            problems.append("has nothing to build its column with")
        if not function.summary.strip() or "\n" in function.summary or not function.summary.endswith("."):
            problems.append("has no one-line summary ending in a full stop")
        if not function.example.startswith(f"{name}("):
            problems.append(f"has no example calling {name}()")
        else:
            try:
                compile_expr(function.example)
            except ExpressionError as error:
                problems.append(f"has an example that does not compile: {error}")
        return problems

    def test_every_function_is_defined_completely_in_one_entry(self) -> None:
        """No entry is missing a part, no name is defined twice, and the whitelist is the table."""
        problems = {name: found for name, function in FUNCTIONS.items() if (found := self.incomplete(name, function))}
        assert not problems, problems
        assert len(DEFINITIONS) == len(FUNCTIONS), "a function is defined twice"
        assert ALLOWED_FUNCTIONS == frozenset(FUNCTIONS)

    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            ({"summary": ""}, "summary"),
            ({"summary": "Two lines.\nOf summary."}, "summary"),
            ({"example": "upper([code])"}, "example"),
            ({"example": "lower([code]"}, "does not compile"),
            ({"params": ()}, "no arguments"),
            ({"result": "text"}, "neither a family"),
            ({"result": "common"}, "agree on"),
            ({"to_polars": None}, "build"),
        ],
    )
    def test_the_check_refuses_an_entry_with_a_part_missing(self, change: dict[str, Any], expected: str) -> None:
        """The check above is itself checked: break one part of a real entry and it must say so."""
        broken = dataclasses.replace(FUNCTIONS["lower"], **change)
        problems = self.incomplete("lower", broken)
        assert any(expected in problem for problem in problems), problems


class TestNullSemantics:
    """Null follows polars, which follows SQL: a comparison against null is null, not false."""

    @pytest.mark.parametrize(
        ("text", "expected_on_the_null_row"),
        [
            ("[code] = 'X'", None),
            ("[code] != 'X'", None),
            ("not ([code] = 'X')", None),
            ("[amount] > 0", None),
            ("[qty] + 1", None),
            ("upper([code]) = 'A'", None),
            ("[code] in ('A', 'B')", None),
            ("[code] not in ('A', 'B')", None),
            # Kleene logic: unknown AND false is false, unknown OR true is true...
            ("[code] = 'X' and false", False),
            ("[code] = 'X' or true", True),
            # ... but unknown AND true, and unknown OR false, stay unknown.
            ("[code] = 'X' and true", None),
            ("[code] = 'X' or false", None),
            # The documented way to make a null row participate.
            ("coalesce([code], 'Z') = 'Z'", True),
            ("[code] is null", True),
            ("[code] is not null", False),
        ],
    )
    def test_null_row_follows_polars_three_valued_logic(
        self, frame: pl.DataFrame, text: str, expected_on_the_null_row: Any
    ) -> None:
        """On a row where the column is null, the result is null unless the rule says otherwise.

        This is the single most likely place a re-implementation disagrees with the engine it
        replaces, so it is pinned value by value rather than left to a filter test to imply.
        """
        assert evaluate(frame, text)[3] is expected_on_the_null_row

    def test_filter_drops_null_rows_and_a_null_test_is_never_null(self, frame: pl.DataFrame) -> None:
        """``filter`` keeps only true rows, so a null row is dropped -- and ``is null`` finds it.

        ``[code] != 'A'`` keeping two rows rather than three is the visible consequence of the rule
        above, and it is what SQL does. ``is null`` / ``is not null`` are the two expressions that
        are never null themselves, which is what makes them usable as an escape hatch.
        """
        assert frame.filter(compile_expr("[code] = 'A'")).height == 1
        assert frame.filter(compile_expr("[code] != 'A'")).height == 2
        assert frame.filter(compile_expr("[code] is null")).height == 1
        assert frame.filter(compile_expr("[code] is not null")).height == 3
        assert None not in evaluate(frame, "[code] is null")
        assert None not in evaluate(frame, "[code] is not null")


class TestColumnReferences:
    """The columns an expression names are reportable before the expression runs."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[code] = 'A' and [amount] > 0", {"code", "amount"}),
            ("upper([label])", {"label"}),
            ("[qty] + [qty] * 2", {"qty"}),
            ("1 + 2", set()),
            ("'no columns here'", set()),
            ("[order status] = 'open'", {"order status"}),
            ("[amount_2] > [amount_2]", {"amount_2"}),
            ("[gl.account] is null", {"gl.account"}),
            ("[part-number] = 'x'", {"part-number"}),
            # A bracket inside quoted text is text, not a column reference.
            ("[note] = '[label]'", {"note"}),
        ],
    )
    def test_references_reports_every_column_named_once(self, text: str, expected: set[str]) -> None:
        """``references`` returns the set of column names an expression reads.

        This is load-bearing rather than informational: the runner checks these names against the
        real input frame before a tool executes, so a typo fails by name up front instead of at row
        40,000. A column name may contain spaces, dots and hyphens. A column whose name contains
        ``]`` is out of scope -- there is no escape syntax, by design.
        """
        result = references(text)
        assert result == expected
        assert isinstance(result, frozenset)

    def test_references_refuses_an_invalid_expression_rather_than_reporting_partial_names(self) -> None:
        """``references`` validates the whole expression, so pre-flight and execution agree.

        If ``references`` were lenient, the runner would pre-flight an expression successfully and
        then fail on compile inside the tool -- the exact failure mode this function exists to
        prevent.
        """
        for text in ("bogus([code])", "[code = 'A'", "os", "[code].upper()", ""):
            with pytest.raises(ExpressionError):
                references(text)

    def test_unknown_column_error_names_the_column_and_lists_what_the_frame_has(self, frame: pl.DataFrame) -> None:
        """A misspelled column names itself and the available columns; a correct one compiles."""
        text = message("[amonut] > 0", columns=frame.columns)
        assert "amonut" in text
        for available in ("amount", "code", "start_date"):
            assert available in text
        assert compile_expr("[amount] > 0", columns=frame.columns) is not None
        assert evaluate(frame, "[amount] > 0") == [True, False, True, None]

    @pytest.mark.parametrize(
        ("text", "bracket_content"),
        [
            ("[x for x in y]", "x for x in y"),
            ("[1, 2]", "1, 2"),
            ("[code] + [x for x in y]", "x for x in y"),
        ],
    )
    def test_bracket_content_is_an_opaque_column_name_never_python_code(
        self, frame: pl.DataFrame, text: str, bracket_content: str
    ) -> None:
        """``[`` is reserved for a column reference, so a list or comprehension cannot be spelled.

        Whatever sits between the brackets is a name, so this text is refused as a bad or unknown
        column -- naming the offending content -- and never evaluated as Python.
        """
        assert bracket_content in message(text, columns=frame.columns)


class TestAuthorFacingErrors:
    """Errors are read by someone who is not a developer, so the wording is part of the contract."""

    @pytest.mark.parametrize(
        ("text", "fragments"),
        [
            ("", ("empty",)),
            ("   ", ("empty",)),
            ("[code = 'A'", ("bracket",)),
            ("[code] = 'A']", ("bracket",)),
            ("[]", ("empty column",)),
            ("[code] = 'A", ("quote",)),
            ("sqrt([amount])", ("sqrt", "coalesce")),
            ("SUM([amount])", ("SUM", "coalesce")),
            ("[code] == 'A'", ("==", "=")),
            ("[qty] ** 2", ("**",)),
            ("[qty] % 2", ("%",)),
            ("[code] ! 'A'", ("!",)),
            ('[code] = "A"', ("double", "single")),
            ("1 < [qty] < 3", ("chained", "and")),
            ("round([amount])", ("round", "2")),
            ("regex_extract([label], 'a')", ("regex_extract", "3")),
            ("[start_date] = date'2026-13-99'", ("2026-13-99", "date")),
            ("[code] in ([label])", ("in", "fixed values")),
            ("[code] is 'A'", ("is null", "is not null")),
        ],
    )
    def test_error_matrix_names_the_problem_in_words_an_author_can_act_on(
        self, text: str, fragments: tuple[str, ...]
    ) -> None:
        """Every fragment must appear in the message: the problem, and what to write instead.

        Note two deliberate strictnesses. ``==`` is refused rather than quietly accepted, because
        one spelling per operator keeps a pipeline file reviewable -- and the error teaches the
        convention. Functions have fixed arity with no defaulted arguments, for the same reason.
        """
        text_out = message(text)
        for fragment in fragments:
            assert fragment in text_out, f"{fragment!r} missing from {text_out!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "[code = 'A'",
            "sqrt([amount])",
            "[qty] ** 2",
            "[code] == 'A'",
            "os",
            "[code].upper()",
            "lambda: 1",
        ],
    )
    def test_error_message_quotes_the_offending_expression(self, text: str) -> None:
        """The message includes the expression itself, so a log line identifies the pipeline line."""
        assert text in message(text)


class TestSecurity:
    """Anything not explicitly allowed is refused. These are the tests that matter most."""

    @pytest.mark.parametrize(
        ("text", "alternatives"),
        [
            # Classic sandbox escapes.
            ("__import__('os').system('id')", ("__import__", "attribute")),
            ("().__class__.__bases__", ("Attribute", "attribute")),
            (
                "().__class__.__bases__[0].__subclasses__()",
                ("Attribute", "attribute", "syntax", "understood", "__subclasses__"),
            ),
            ("open('/etc/passwd')", ("open",)),
            ("globals()", ("globals",)),
            # Attribute access on a column, even though it is polars' own API.
            ("[code].str.to_uppercase()", ("to_uppercase", "attribute")),
            ("[code].upper()", ("upper", "attribute")),
            ("coalesce([code], 'Z').abs()", ("abs", "attribute")),
            # Callables, comprehensions, bindings, unpacking, formatting, generators.
            ("lambda: 1", ("Lambda",)),
            ("(lambda: 1)()", ("Lambda",)),
            ("(x for x in y)", ("GeneratorExp",)),
            ("{x for x in y}", ("SetComp",)),
            ("{k: v for k, v in y}", ("DictComp",)),
            ("(x := 1)", ("NamedExpr", "syntax", "understood")),
            ("round(*(1, 2))", ("Starred",)),
            ("round([amount], decimals=2)", ("keyword", "decimals")),
            ("f'{code}'", ("JoinedStr", "FormattedValue")),
            ("(yield 1)", ("Yield",)),
            ("(yield from x)", ("YieldFrom",)),
            ("await x", ("Await",)),
            # Names that are neither a whitelisted function nor a column.
            ("os", ("os",)),
            ("pl", ("pl",)),
            ("pl.col('code')", ("Attribute", "attribute", "pl")),
            ("True", ("True",)),
            ("None", ("None",)),
            # Containers, conditionals and literals that are not part of the language.
            ("{'a': 1}", ("Dict",)),
            ("{1, 2}", ("Set",)),
            ("1 if true else 2", ("IfExp",)),
            ("...", ("Ellipsis", "literal")),
            ("b'abc'", ("literal", "bytes", "Bytes")),
            # Operators outside the whitelist, named by node type.
            ("[qty] ** 2", ("Pow",)),
            ("[qty] % 2", ("Mod",)),
            ("[qty] // 2", ("FloorDiv",)),
            ("~[qty]", ("Invert",)),
            ("+[qty]", ("UAdd",)),
            ("[flag] & [flag]", ("BitAnd",)),
            ("[flag] | [flag]", ("BitOr",)),
            ("[qty] ^ 1", ("BitXor",)),
            ("[qty] @ [qty]", ("MatMult",)),
            ("[qty] << 1", ("LShift",)),
            ("[qty] >> 1", ("RShift",)),
        ],
    )
    def test_refusal_matrix_rejects_everything_outside_the_language(
        self, text: str, alternatives: tuple[str, ...]
    ) -> None:
        """Each case is refused, and the message names the reason.

        ``alternatives`` is an any-of list, not an all-of list: which check fires first on a
        compound attack depends on walk order, and pinning walk order would pin the implementation.
        What is pinned is that the message says something specific and true about the refusal.
        """
        text_out = message(text)
        assert any(fragment in text_out for fragment in alternatives), (
            f"none of {alternatives} appeared in {text_out!r}"
        )

    def test_every_expression_node_type_outside_the_whitelist_is_refused(self) -> None:
        """The whitelist is a whitelist: walk Python's own node types and demand refusal.

        ``DISALLOWED_NODE_SNIPPETS`` must cover exactly the expression node types that
        ``ALLOWED_NODES`` does not permit -- no more, no fewer. So a future Python that adds an
        expression node type breaks this test until someone decides, deliberately, which side of
        the whitelist it falls on. Each snippet is compiled against an empty column list, so a
        snippet that the bracket rule turns into a column reference is refused as an unknown column
        rather than quietly compiling.
        """
        allowed = {node.__name__ for node in ALLOWED_NODES}
        every_expression_node = {node.__name__ for node in ast.expr.__subclasses__()}
        must_be_refused = every_expression_node - allowed

        assert set(DISALLOWED_NODE_SNIPPETS) == must_be_refused, (
            "the refusal corpus and the whitelist disagree about which node types exist: "
            f"uncovered={sorted(must_be_refused - set(DISALLOWED_NODE_SNIPPETS))} "
            f"stale={sorted(set(DISALLOWED_NODE_SNIPPETS) - must_be_refused)}"
        )
        for node_name, snippet in sorted(DISALLOWED_NODE_SNIPPETS.items()):
            with pytest.raises(ExpressionError, match=r".+"):
                compile_expr(snippet, columns=())
            assert node_name not in {n.__name__ for n in ALLOWED_NODES}

    def test_allowed_node_whitelist_is_exactly_the_documented_set(self) -> None:
        """The AST whitelist is pinned member by member, so widening it is a deliberate act.

        Only the expression and operator members are pinned. Housekeeping nodes an implementation
        may or may not need to visit (``Expression``, ``Load``) are left free, because whether they
        are visited at all is an implementation choice.
        """
        assert all(isinstance(node, type) and issubclass(node, ast.AST) for node in ALLOWED_NODES)
        assert not any(issubclass(node, ast.stmt) for node in ALLOWED_NODES)

        expressions = {node for node in ALLOWED_NODES if issubclass(node, ast.expr)}
        assert expressions == {
            ast.BoolOp,
            ast.BinOp,
            ast.UnaryOp,
            ast.Compare,
            ast.Call,
            ast.Constant,
            ast.Name,
            ast.Tuple,
        }

        operator_bases = (ast.boolop, ast.operator, ast.unaryop, ast.cmpop)
        operators = {node for node in ALLOWED_NODES if issubclass(node, operator_bases)}
        assert operators == {
            ast.And,
            ast.Or,
            ast.Not,
            ast.USub,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.In,
            ast.NotIn,
            ast.Is,
            ast.IsNot,
        }
        assert ast.Attribute not in ALLOWED_NODES
        assert ast.Subscript not in ALLOWED_NODES

    def test_allowed_function_whitelist_is_exactly_the_documented_set(self) -> None:
        """Extending the language means adding a name here, never growing the grammar.

        ``regex_replace``, ``regex_match`` and ``concat`` are the three most recent additions, and
        they arrived the only way this language grows: as names on this list. A tool reshapes a
        frame; an expression function computes one column, so a rewrite, a pattern test and a join
        of two pieces of text belong here rather than in tools of their own. ``concat`` in
        particular is what makes it safe to refuse ``+`` between two text columns: the capability
        stays, under a name that says what it does. The set is still pinned member by member --
        widening it remains a deliberate act, and this assertion remains an equality.

        ``year`` is the newest, and the first that reads a *part* of a value rather than
        transforming the whole of it. It arrived with a consumer: repairing a term whose end year
        was mistyped means comparing against the year after the one it starts in, and no
        combination of the names above can produce that number. ``month`` and ``day`` are its
        obvious siblings and are deliberately absent until something needs them -- a function with
        no caller is checked only against whatever its author imagined.

        ``add_months`` is newer still. It moves a date forward by a whole number of months and
        hands back a date. It arrived with a caller too: spreading one billed amount across the
        months of a term needs the month ``n - 1`` after the start date, and no other name here can
        produce a date at all. Its second argument is a count and not a spelling, so it can be a
        column or a sum rather than a number typed into the file.

        ``format_number`` is the newest, and it arrived the same way: a tab of invariants has to
        read ``651,623.80``, and nothing on this list produces it. ``cast_columns`` to ``String``
        gives ``651623.8`` and loses the trailing zero; ``regex_replace`` cannot rearrange what it
        matched, because its replacement is verbatim text by design. One column in, one column out,
        so it is a function and not a tool -- and it takes four required arguments rather than
        Alteryx's two optional ones, because this language has no defaulted argument anywhere.

        ``uuid5`` is the newest, and it is the first name here that hashes anything. It arrived
        because a published table needs one narrow, fixed-width column to be joined on where its
        natural key is three columns of text -- and nothing else on this list can turn several
        values into one stable id. It is the standard library's version 5, not an invention, so
        the id a flow writes is the id anything else computing it the same way would write.
        """
        assert isinstance(ALLOWED_FUNCTIONS, frozenset)
        assert ALLOWED_FUNCTIONS == {
            "coalesce",
            "round",
            "year",
            "add_months",
            "abs",
            "lower",
            "upper",
            "strip",
            "len",
            "starts_with",
            "ends_with",
            "contains",
            "regex_extract",
            "regex_replace",
            "regex_match",
            "concat",
            "format_number",
            "uuid5",
        }
        # A column is written [name]; 'col' is an internal rewrite target, not a documented function.
        assert "col" not in ALLOWED_FUNCTIONS


class TestResourceLimits:
    """A pipeline file is untrusted input, so text length and nesting depth are bounded."""

    def test_limits_are_sane_constants(self) -> None:
        """The limits are published numbers, generous enough for real expressions.

        A depth or size limit *is* in scope, and deliberately so: ``ast.parse`` is itself the thing
        that fails on pathological input -- a long enough operator chain raises ``RecursionError``
        during tree construction, before any whitelist can be consulted -- so the limits have to be
        enforced by this module, not delegated to the parser.
        """
        assert isinstance(MAX_TEXT_LENGTH, int) and 500 <= MAX_TEXT_LENGTH <= 100_000
        assert isinstance(MAX_DEPTH, int) and 12 <= MAX_DEPTH <= 100

        long_but_reasonable = " or ".join(f"[code] = '{letter}'" for letter in "ABCDEFGHIJ")
        assert references(long_but_reasonable) == {"code"}
        compile_expr(long_but_reasonable, columns=["code"])

    @pytest.mark.parametrize(
        ("text", "fragment"),
        [
            ("1+" * 20_000 + "1", str(MAX_TEXT_LENGTH)),
            ("9" * (MAX_TEXT_LENGTH + 1), str(MAX_TEXT_LENGTH)),
            ("1+" * (MAX_DEPTH * 2) + "1", str(MAX_DEPTH)),
            ("not " * (MAX_DEPTH + 5) + "true", str(MAX_DEPTH)),
            ("abs(" * (MAX_DEPTH + 5) + "[amount]" + ")" * (MAX_DEPTH + 5), str(MAX_DEPTH)),
        ],
    )
    def test_oversize_or_deeply_nested_input_is_refused_not_parsed(self, text: str, fragment: str) -> None:
        """Oversize text is refused before parsing; deep nesting is refused after.

        The first case is the important one. It is 40,001 characters and would raise
        ``RecursionError`` inside ``ast.parse`` if the length check ran second, so an
        ``ExpressionError`` here is evidence the length gate sits in front of the parser.
        """
        with pytest.raises(ExpressionError) as caught:
            compile_expr(text)
        assert fragment in str(caught.value)


# ---------------------------------------------------------------------------
# Dtype checking, the number scanner, the node cap, and the regex pair
# ---------------------------------------------------------------------------
#
# Everything below hardens the language rather than widening it, and one idea runs through all of
# it: a bad expression must fail *here*, by name, in the author's vocabulary -- never later, inside
# the query engine, in the engine's vocabulary. Four consequences, in order:
#
# 1. ``compile_expr`` accepts a schema, not only names, so ``upper([qty])`` is refused at compile
#    time instead of raising a SchemaError from inside a select.
# 2. A number literal has to mean what it looks like, so ``1e400`` -- silently ``inf``, and
#    therefore a filter that matches every row -- is refused rather than compiled.
# 3. Breadth is capped by a published number, the way depth already is.
# 4. A pattern is validated against the engine that will run it, at compile time.


def checked(frame: pl.DataFrame, text: str) -> list[Any]:
    """Values from ``text`` compiled against the frame's *schema* -- so dtypes are checked -- then run."""
    return frame.select(compile_expr(text, columns=frame.schema).alias("out"))["out"].to_list()


def schema_message(frame: pl.DataFrame, text: str) -> str:
    """The error from compiling ``text`` against the frame's schema; fails the test if it compiles."""
    with pytest.raises(ExpressionError) as caught:
        compile_expr(text, columns=frame.schema)
    return str(caught.value)


# One template per operation the language has, to be applied to a column of every dtype the fixture
# frame carries. The product is swept in
# ``test_no_engine_error_escapes_when_a_schema_was_supplied``, which is the test that states the
# contract in general rather than case by case: for every one of these, either it is refused as an
# ExpressionError or it runs -- never "compiles, then raises the engine's own exception".
DTYPE_SWEEP_TEMPLATES: tuple[str, ...] = (
    "upper({c})",
    "lower({c})",
    "strip({c})",
    "len({c})",
    "abs({c})",
    "round({c}, 2)",
    "starts_with({c}, 'a')",
    "ends_with({c}, 'a')",
    "contains({c}, 'a')",
    "regex_extract({c}, '(a)', 1)",
    "regex_replace({c}, 'a', 'X')",
    "regex_match({c}, 'a')",
    "add_months({c}, 1)",
    # add_months() is the only two-place entry whose rule lives on the second argument: the count
    # has to be whole months. Varying the first argument alone leaves that rule unswept, so these
    # four vary the count instead. Each builds the count a different way -- arithmetic, and the
    # three functions that give back a number -- because the rule has to hold however the count
    # was arrived at, not only when it is a bare column.
    "add_months([start_date], {c} - 1)",
    "add_months([start_date], abs({c}))",
    "add_months([start_date], round({c}, 0))",
    "add_months([start_date], coalesce({c}, 0))",
    # format_number() is the one entry that takes a number and gives text, so it is the only way
    # this sweep asks "is a text column refused *before* it reaches a number function" for a
    # function whose result is text -- the shape that would otherwise compile and fail in a select.
    "format_number({c}, 2, ',', '.')",
    # uuid5() is the other direction through the same gate: text in, text out, by way of a
    # per-row call rather than an engine kernel -- so a dtype that reached it would not announce
    # itself by raising, and this is the only place that asks whether it gets that far. The
    # namespace is written out because the constant it comes from is defined further down.
    "uuid5('6ba7b810-9dad-11d1-80b4-00c04fd430c8', {c})",
    "coalesce({c}, 'Z')",
    "coalesce({c}, 0)",
    "coalesce({c}, [code])",
    "coalesce({c}, [qty])",
    "concat({c}, 'x')",
    "concat({c}, [code])",
    "{c} + 1",
    "{c} - 1",
    "{c} * 2",
    "{c} / 2",
    "-{c}",
    "{c} = 'A'",
    "{c} != 1",
    "{c} > 0",
    "{c} < 'A'",
    "{c} = true",
    "{c} = date'2026-01-01'",
    "{c} in ('A', 'B')",
    "{c} in (1, 2)",
    "{c} and true",
    "not {c}",
    "{c} or false",
    "{c} is null",
    "{c} is not null",
)

# One column per dtype the fixture carries: text, decimal, whole number, boolean, date, and the two
# dtypes that hold text compactly. The last two were outside this list, and outside the fixture, for
# as long as the sweep has existed -- which is the only reason it passed while every string function
# on such a column compiled and then failed inside the engine.
DTYPE_SWEEP_COLUMNS: tuple[str, ...] = (
    "[code]",
    "[amount]",
    "[qty]",
    "[flag]",
    "[start_date]",
    "[category]",
    "[grade]",
)


class TestSchemaDtypeChecking:
    """A schema turns a dtype mistake into a refusal, in the author's words, before any row is read.

    ``compile_expr`` takes column *names*, and names alone cannot catch ``upper([qty])``: it
    compiles, and then fails inside the query engine, with the engine's own exception type, naming
    the engine's problem rather than the author's mistake. So the second argument accepts a schema
    as well -- name to dtype -- and both callers are legitimate: one holding the upstream frame
    passes ``frame.schema`` and gets dtype checking, one that has no frame passes ``None`` and gets
    syntax-only checking.

    The rule the tests below pin is one sentence: **every operation needs its operands in the type
    family it works on, and bringing two families together is refused.** The families are text,
    number (whole and decimal together), date and boolean; ``null`` belongs to all of them.
    """

    @pytest.mark.parametrize(
        "text",
        [
            # A string function on a column that holds no text.
            "upper([qty])",
            "lower([amount])",
            "strip([flag])",
            "len([qty])",
            "starts_with([start_date], 'a')",
            "ends_with([qty], 'a')",
            "contains([qty], 'a')",
            "regex_extract([qty], '(a)', 1)",
            "regex_replace([amount], 'a', 'X')",
            "regex_match([flag], 'a')",
            # ... including the text argument such a function compares against.
            "starts_with([label], 1)",
            "contains([label], 1)",
            # A numeric function on a column that holds no number.
            "abs([code])",
            "abs([flag])",
            "abs([start_date])",
            "round([code], 2)",
            "round([flag], 0)",
            # Arithmetic, including the unary minus, on a column that cannot do arithmetic.
            "[code] + 1",
            "[code] + 'x'",
            # Two text columns are the case that costs money rather than merely reading badly.
            # polars concatenates them, so amounts imported as text add up to '10020' instead of
            # 120 -- and both operands being in the same family means no other rule in this
            # language would catch it. The capability moves to concat(), which says what it does.
            "[code] + [label]",
            "[label] + [code]",
            "[label] * 2",
            "[flag] + 1",
            "[start_date] + 1",
            "[start_date] - [start_date]",
            "-[code]",
            "-[flag]",
            # A logical operator on a column that is not a truth value.
            "[code] and [flag]",
            "[qty] and [flag]",
            "[amount] or [flag]",
            "not [code]",
            "not [qty]",
            # concat() joins text, and joins nothing else: polars would stringify a number, a date
            # or a truth value without a word, which is the same silent coercion coalesce() is
            # refused for two lists above.
            "concat([code], [qty])",
            "concat([qty], [code])",
            "concat([code], [start_date])",
            "concat([flag], 'x')",
            "concat([code], abs([amount]))",
            # Nesting: the dtype a function *returns* is checked too, or nesting is a hole.
            "upper(len([label]))",
            "abs(upper([label]))",
            "len(abs([amount]))",
        ],
    )
    def test_a_schema_refuses_an_operation_the_column_dtype_cannot_support(
        self, frame: pl.DataFrame, text: str
    ) -> None:
        """Each of these compiles happily against names and then fails in the engine; with a schema it is refused.

        The nested cases are the ones a partial implementation gets wrong: checking the dtype of a
        column reference but treating every *function result* as unknown leaves ``upper(len(...))``
        compiling, and the hole is invisible until a pipeline hits it.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text, columns=frame.schema)

    @pytest.mark.parametrize(
        "text",
        [
            # Text against a number, in both directions and for every comparison.
            "[code] = 1",
            "[code] > 0",
            "[code] != 2.5",
            "[qty] = 'a'",
            "[amount] < 'x'",
            "[qty] = [code]",
            "[code] != [qty]",
            # A date against anything that is not a date.
            "[start_date] = 'x'",
            "[start_date] > 1",
            "[code] = date'2026-01-01'",
            "[qty] = date'2026-01-01'",
            # A truth value against anything that is not one. polars answers both of these rather
            # than raising -- 'x' compares False everywhere, 1 compares equal to true -- so these
            # two are refusals this language adds, on the grounds that a filter which quietly
            # matches nothing is the worst answer available.
            "[flag] = 'x'",
            "[flag] = 1",
            "[flag] > [qty]",
            # Membership: every value in the list has to be in the column's family.
            "[qty] in ('A', 'B')",
            "[code] in (1, 2)",
            "[start_date] in ('2026-01-01')",
            "[flag] in (1, 2)",
            # coalesce: polars stringifies a mixed pair silently, so a number would arrive in a text
            # column looking like data somebody meant.
            "coalesce([code], [qty])",
            "coalesce([code], 1)",
            "coalesce([start_date], 'Z')",
            "coalesce([flag], 'Z')",
            "coalesce([qty], [start_date])",
            "coalesce([code], [qty], 'Z')",
        ],
    )
    def test_a_schema_refuses_two_type_families_brought_together(self, frame: pl.DataFrame, text: str) -> None:
        """A comparison, a membership test or a coalesce across two families is refused.

        Most of these already fail in the engine, so refusing them costs nothing and buys a message
        an author can act on. The rest -- a boolean against text or a number, a date against a
        number, a coalesce of text with a number -- are worse than an engine error: polars answers
        them, and the answer is meaningless. A predicate that matches no rows and raises nothing is
        indistinguishable, in a log, from a day on which nothing happened.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text, columns=frame.schema)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Whole numbers and decimals are one family: a rule that split them would refuse
            # expressions the engine runs perfectly well.
            ("[qty] > 2.5", [False, False, True, None]),
            ("[amount] = 100", [False, False, True, None]),
            ("[amount] / [qty] > 1", [True, False, True, None]),
            ("[amount] in (1, 100.0)", [False, False, True, None]),
            ("[qty] in (1, 2)", [True, True, False, None]),
            ("round([amount] * 2, 0)", [21.0, -5.0, 200.0, None]),
            ("abs([qty]) + 1 > 0", [True, True, True, None]),
            # Text, dates and truth values, each within their own family.
            ("upper(strip([label]))", ["ALPHA", "BETA", "GAMMA", None]),
            ("[note] = 'a=b'", [True, False, False, None]),
            ("len([label]) > 3", [True, True, True, None]),
            ("regex_match([label], 'a.p')", [True, False, False, None]),
            ("coalesce([code], 'Z') in ('A', 'Z')", [True, False, False, True]),
            ("[start_date] > date'2026-01-01'", [False, True, False, None]),
            ("[flag] = true", [True, False, True, None]),
            ("[flag] and not [flag]", [False, False, False, None]),
            ("not ([code] = 'X')", [True, True, True, None]),
            # null is in every family: it is the one literal that may meet any column at all.
            ("[qty] = null", [None, None, None, None]),
            ("coalesce([code], null) is null", [False, False, False, True]),
            ("[code] in ('A', null)", [True, False, False, None]),
        ],
    )
    def test_a_schema_accepts_every_expression_that_stays_inside_one_family(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """Dtype checking must not become stricter than the engine, so what is legal is pinned too.

        This is the half of dtype checking that can fail silently in the other direction. A rule
        stricter than polars rejects expressions that would have worked, and an author cannot argue
        with a refusal -- they just stop using the tool. So each case here both compiles under a
        schema *and* produces its documented values.
        """
        assert checked(frame, text) == expected

    @pytest.mark.parametrize(
        ("text", "column", "expectations"),
        [
            ("upper([qty])", "qty", ("text", "string")),
            ("len([start_date])", "start_date", ("text", "string")),
            ("abs([code])", "code", ("number", "numeric")),
            ("round([flag], 2)", "flag", ("number", "numeric")),
            ("[code] + 1", "code", ("number", "numeric", "arithmetic")),
            ("[flag] = 'x'", "flag", ("text", "string", "compar", "boolean", "truth")),
            ("not [qty]", "qty", ("boolean", "truth")),
            ("coalesce([code], [qty])", "qty", ("text", "string", "same", "famil")),
            ("concat([code], [qty])", "qty", ("text", "string")),
            ("[code] + [label]", "code", ("number", "numeric", "arithmetic", "concat")),
        ],
    )
    def test_a_dtype_refusal_names_the_column_its_dtype_and_what_it_expected(
        self, frame: pl.DataFrame, column: str, text: str, expectations: tuple[str, ...]
    ) -> None:
        """A dtype refusal is read by whoever wrote the line, so it must carry all three facts.

        Which column, what that column actually holds, and what the operation needed instead. Two
        of the three is not enough: "expected text" without a column name sends an author reading
        a line with four columns in it hunting, and a column name without its dtype sends them to
        the schema to find out why.
        """
        reported = schema_message(frame, text)
        # Twice, not once: every message quotes the offending expression, and the column name is
        # already in there, so a single occurrence would be satisfied by a message that says
        # nothing at all about which column is the problem.
        assert reported.count(column) >= 2, f"the problem itself does not name the column in {reported!r}"
        assert str(frame.schema[column]) in reported, f"the actual dtype is missing from {reported!r}"
        lowered = reported.lower()
        assert any(word in lowered for word in expectations), f"none of {expectations} appeared in {reported!r}"

    def test_no_engine_error_escapes_when_a_schema_was_supplied(self, frame: pl.DataFrame) -> None:
        """The contract, stated once over every operation crossed with every dtype.

        Case-by-case tests pin the rules an implementation was written against; this sweep pins the
        promise, which is what a caller actually relies on: with a schema in hand, a bad expression
        raises ``ExpressionError`` and a compiled expression runs. Nothing in between -- no
        SchemaError from a select three steps later, and no raw ``TypeError`` out of compilation
        either, which is how a mixed-type membership list currently fails.

        The two lower bounds at the end keep the sweep honest in both directions: an implementation
        that refused everything, or one that checked nothing, would otherwise satisfy it.

        The bounds are the counted answer, not a slack figure. There are 43 templates and 7
        columns, so 301 expressions, and no engine failure is allowed: every one of them is either
        compiled or refused. 104 compile and 197 are refused, and those two numbers add to 301, so
        the pair pins the split exactly. That is the point of stating them tightly. A fix that
        closes the add_months() gap by refusing whole-number counts as well would still leave the
        sweep with no engine failure, and a loose bound would call that a pass.
        """
        leaked: list[str] = []
        engine_failures: list[str] = []
        compiled = 0
        refused = 0

        for template in DTYPE_SWEEP_TEMPLATES:
            for column in DTYPE_SWEEP_COLUMNS:
                text = template.format(c=column)
                try:
                    expression = compile_expr(text, columns=frame.schema)
                except ExpressionError:
                    refused += 1
                    continue
                except Exception as error:  # noqa: BLE001 - the point of the test
                    leaked.append(f"{text!r} raised {type(error).__name__}: {error}")
                    continue
                try:
                    frame.select(expression.alias("out"))
                except Exception as error:  # noqa: BLE001 - the point of the test
                    engine_failures.append(f"{text!r} -> {type(error).__name__}: {error}")
                    continue
                compiled += 1

        assert not leaked, f"compilation raised something other than ExpressionError: {leaked}"
        assert not engine_failures, (
            f"these compiled against the schema and then failed in the engine: {engine_failures}"
        )
        assert compiled >= 104, f"only {compiled} of the swept expressions compiled; dtype checking is too strict"
        assert refused >= 197, f"only {refused} of the swept expressions were refused; dtype checking is not happening"

    def test_each_form_of_the_columns_argument_checks_what_it_can(self, frame: pl.DataFrame) -> None:
        """A schema checks existence and dtype, bare names check existence, ``None`` checks neither.

        Three forms because there are three callers. A loader holding the upstream frame passes
        ``frame.schema``. A caller validating a pipeline definition against a template has no frame
        at all and passes ``None``, so bad syntax is still caught at that boundary while a dtype
        stays the runner's problem. And the middle form is the one that must not regress: a caller
        already passing ``frame.columns`` keeps the behaviour it has today, because a list of names
        cannot say anything about a dtype and must not be guessed at.
        """
        reported = schema_message(frame, "[amonut] > 0")
        assert "amonut" in reported and "amount" in reported
        schema_message(frame, "upper([qty])")

        for names in (list(frame.columns), tuple(frame.columns), frozenset(frame.columns)):
            assert compile_expr("upper([qty])", columns=names) is not None
            with pytest.raises(ExpressionError):
                compile_expr("[amonut] > 0", columns=names)

        assert compile_expr("upper([qty])") is not None
        assert compile_expr("[amonut] > 0") is not None
        assert references("upper([qty])") == {"qty"}

        # An empty schema is a schema listing no columns, not the absence of one.
        with pytest.raises(ExpressionError):
            compile_expr("[code] is null", columns={})


# A column that stores its text compactly is text as far as an author is concerned. The templates
# below are applied to such a column and to the plain text column holding the *same values*, and the
# two results are compared, so the expectation is never hand-written: whatever the language does for
# text is what it must do here.
TEXT_OPERATION_TEMPLATES: tuple[str, ...] = (
    "upper({c})",
    "lower({c})",
    "strip({c})",
    "len({c})",
    "starts_with({c}, 'B')",
    "ends_with({c}, 'a')",
    "contains({c}, 'a')",
    "regex_extract({c}, '([a-z]+)', 1)",
    "regex_replace({c}, 'a', 'X')",
    "regex_match({c}, '^ +a')",
    "concat({c}, 'x')",
    "concat({c}, [code])",
    "uuid5('6ba7b810-9dad-11d1-80b4-00c04fd430c8', {c})",
    "{c} = 'Beta'",
    "{c} != 'Beta'",
    "{c} < 'Beta'",
    "{c} >= 'Beta'",
    "{c} in ('Beta', 'gamma')",
    "{c} is null",
    "{c} is not null",
    "upper({c}) = '  ALPHA  '",
    "not contains({c}, 'a')",
)


@pytest.fixture
def restricted_frame() -> pl.DataFrame:
    """A column restricted to two members, beside the same values as plain text.

    Separate from the main fixture on purpose. The sweep's fixture lists every literal the sweep
    compares against among its members, so that the sweep stays about one thing; this frame is the
    opposite case, and the one that arrives in real data: an author compares against a value the
    column does not hold -- a status that has since been retired, a code from the other side of a
    match -- and a fallback names a value that was never a member at all.
    """
    values = ["alpha", "beta", None]
    return pl.DataFrame(
        {
            "label": values,
            "grade": pl.Series(values, dtype=pl.Enum(["alpha", "beta"])),
        }
    )


class TestCompactlyStoredText:
    """A column that stores its text compactly is text: every text operation works on it.

    Two dtypes hold text without holding a string per row -- one dictionary-encodes the values it
    has seen, the other restricts them to a fixed set of members -- and both are already in this
    language's text family, which is right: ``[category] = 'Beta'`` answers correctly today. What
    did not work is every operation that reaches for the engine's string kernels. ``upper()`` on
    such a column compiled and then raised the engine's own SchemaError; ``contains()`` raised its
    InvalidOperationError. Both escaped ``ExpressionError`` entirely, which is the one promise this
    module makes, and both did so *after* compilation, so the caller that gets the error is a
    caller that had already been told the expression was fine.

    Refusing these would have been the smaller fix and the wrong one: an author cannot see which
    dtype their statuses arrived in, so a language that refuses ``upper()`` on one column of text
    and allows it on another draws a distinction they have no way to act on. So the pin is that
    they *work*, and work identically -- asserted against the same values held as plain text, so
    nothing here can be satisfied by an expectation somebody wrote to match the behaviour.
    """

    @pytest.mark.parametrize("template", TEXT_OPERATION_TEMPLATES, ids=lambda t: t.replace("{c}", "_"))
    @pytest.mark.parametrize("column", ["[category]", "[grade]"], ids=["dictionary_encoded", "fixed_members"])
    def test_every_text_operation_matches_what_it_gives_on_plain_text(
        self, frame: pl.DataFrame, column: str, template: str
    ) -> None:
        """The same operation, the same values, two ways of storing them: one answer.

        This is the whole contract for these dtypes in one assertion. It covers the four string
        functions, the three that compare text against text, the three pattern functions, joining,
        comparison, membership and the null tests -- because a fix applied to the function that
        happened to be reported is a fix that leaves the next one broken.
        """
        assert checked(frame, template.format(c=column)) == checked(frame, template.format(c="[label]"))

    @pytest.mark.parametrize(
        ("text", "equivalent"),
        [
            # A text function's *result* fed onward. The compactly-stored dtype is not confined to
            # the leaf: an operation that chooses between values hands one back, and the next
            # operation along then meets it. This is where a fix applied only where a column is read
            # is not enough, and where a fix applied only to the operation that reaches for the
            # engine is not enough either.
            ("upper(coalesce([category], 'Z'))", "upper(coalesce([label], 'Z'))"),
            ("upper(coalesce([grade], 'Z'))", "upper(coalesce([label], 'Z'))"),
            ("len(strip([category]))", "len(strip([label]))"),
            ("regex_replace(upper([grade]), 'A', 'X')", "regex_replace(upper([label]), 'A', 'X')"),
            ("contains(strip([category]), 'alpha')", "contains(strip([label]), 'alpha')"),
            # ... and the reverse direction: such a column as the argument an operation compares
            # *against*, rather than the subject it works on.
            ("starts_with([label], [category])", "starts_with([label], [label])"),
            ("contains([label], [grade])", "contains([label], [label])"),
            ("concat(upper([category]), [grade])", "concat(upper([label]), [label])"),
        ],
    )
    def test_such_a_value_survives_being_fed_into_and_out_of_a_text_operation(
        self, frame: pl.DataFrame, text: str, equivalent: str
    ) -> None:
        """Nesting is where the narrow version of this fix stops working.

        ``upper(coalesce([grade], 'Z'))`` is the case that separates the two: the choice hands back
        a value still stored compactly, and the engine will not run a string kernel over it. An
        implementation that converts only what it reads from a column, or only what it is about to
        hand to a kernel, leaves this expression compiling and then failing -- which is the same
        defect as before, one level in, and the level nobody tests by hand.
        """
        assert checked(frame, text) == checked(frame, equivalent)

    @pytest.mark.parametrize(
        ("text", "column"),
        [
            # Making these dtypes work with text must not be done by making them work with
            # everything: an implementation that moved them out of the text family, or into the
            # unchecked one, would satisfy every assertion above and quietly withdraw dtype
            # checking for exactly the columns this cycle is about.
            ("abs([category])", "category"),
            ("round([grade], 2)", "grade"),
            ("[category] + 1", "category"),
            ("[grade] * 2", "grade"),
            ("[category] = 1", "category"),
            ("[grade] > date'2026-01-01'", "grade"),
            ("not [category]", "category"),
            ("-[grade]", "grade"),
            ("coalesce([qty], [grade])", "grade"),
        ],
    )
    def test_such_a_column_is_still_text_for_the_purpose_of_a_refusal(
        self, frame: pl.DataFrame, text: str, column: str
    ) -> None:
        """A refusal that was right stays right, and still names the column and what it holds.

        ``abs([category])`` is refused today, correctly, because the column is in the text family.
        The dtype it reports is named as well, because an author reading "needs numbers" about a
        column of statuses needs to be told what the column actually is before that sentence means
        anything to them.
        """
        reported = schema_message(frame, text)
        assert reported.count(column) >= 2, f"the problem itself does not name the column in {reported!r}"
        assert str(frame.schema[column]) in reported, f"the actual dtype is missing from {reported!r}"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # A fallback that does not fall back is the worst outcome available here: no exception,
            # no log line, and a column of nulls where a value was asked for. It happens because
            # the engine coerces the fallback into the column's own dtype first, and a value that
            # is not one of the members coerces to null -- so the fallback is destroyed by the
            # thing it was written to survive.
            ("coalesce([grade], 'missing')", ["alpha", "beta", "missing"]),
            ("coalesce([grade], [label])", ["alpha", "beta", None]),
            ("upper(coalesce([grade], 'missing'))", ["ALPHA", "BETA", "MISSING"]),
            ("len(coalesce([grade], 'missing'))", [5, 4, 7]),
        ],
    )
    def test_a_fallback_falls_back_even_when_its_value_is_not_one_of_the_members(
        self, restricted_frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """The values are written out here, not compared against another column, deliberately.

        Every other assertion in this class is an equality against the same values held as plain
        text, which is the right shape for "behaves the same" -- but this case is the one where the
        wrong answer is a *quiet* one, so it is worth stating the three values a reader can check
        by eye. ``coalesce([grade], 'missing')`` gives ``[alpha, beta, null]`` today: the row the
        author wrote the fallback for is the row that still has nothing in it.
        """
        assert checked(restricted_frame, text) == expected

    @pytest.mark.parametrize(
        "template",
        [
            "{c} in ('missing', 'beta')",
            "{c} > 'missing'",
            "{c} < 'missing'",
            "{c} >= 'missing'",
            # These two answer correctly today, and are here so the rule is pinned as one rule:
            # an implementation that fixed the four above by special-casing each operator would
            # leave the next operator to be added exactly as broken as these were.
            "{c} = 'missing'",
            "{c} != 'missing'",
        ],
        ids=lambda t: t.replace("{c}", "_"),
    )
    def test_a_comparison_against_a_value_that_is_not_one_of_the_members_is_answered_not_refused(
        self, restricted_frame: pl.DataFrame, template: str
    ) -> None:
        """A value the column cannot hold is a value no row equals -- an answer, not an error.

        ``[grade] in ('missing', 'beta')`` and ``[grade] > 'missing'`` raise the engine's own
        InvalidOperationError, because the engine coerces the literal into the column's members
        before comparing and refuses the coercion. That refusal is the engine's, not this
        language's, and it arrives after compilation said the expression was fine. An author
        filtering on a value that has since been retired from a restricted column would see it,
        and there is nothing wrong with what they wrote: no row matches, which is the answer.
        """
        assert checked(restricted_frame, template.format(c="[grade]")) == checked(
            restricted_frame, template.format(c="[label]")
        )

    @pytest.mark.parametrize(
        ("text", "expected_dtype"),
        [
            # Text arrives as text and leaves as a plain string column, however it was stored.
            ("[category]", pl.String),
            ("[grade]", pl.String),
            ("coalesce([category], 'Z')", pl.String),
            ("coalesce([grade], 'Z')", pl.String),
            ("strip([category])", pl.String),
            ("concat([grade], 'x')", pl.String),
            ("[label]", pl.String),
            # ... and the rule is about text, not about casting everything: no other family's
            # dtype moves, so a whole number stays a whole number and a date stays a date.
            ("[qty]", pl.Int64),
            ("[amount]", pl.Float64),
            ("[flag]", pl.Boolean),
            ("[start_date]", pl.Date),
            ("round([amount], 1)", pl.Float64),
        ],
    )
    def test_an_expressions_result_is_a_plain_string_column_wherever_it_is_text(
        self, frame: pl.DataFrame, text: str, expected_dtype: pl.DataType
    ) -> None:
        """The published rule, and the reason it has to be published rather than merely true.

        Making these dtypes work means converting them, and a conversion has a consequence a step
        downstream can see: a compactly stored column that passes through an expression comes out
        as plain strings, so it no longer shares a dtype with the column it came from. That matters
        where the result is written -- the compact encoding is gone -- and where it is later used as
        a key against a column that never passed through an expression, which is a mismatch nobody
        would go looking for. So it is stated as a rule an author can rely on and design around,
        and pinned here as a promise rather than left as a side effect of how the fix was written.
        """
        produced = frame.select(compile_expr(text, columns=frame.schema).alias("out")).schema["out"]
        assert produced == expected_dtype, f"{text} produced {produced}, not {expected_dtype}"


class TestNumberLiterals:
    """A number has to mean what it looks like, because the parser underneath is more generous."""

    @pytest.mark.parametrize(
        "text",
        [
            "0x10",
            "0X10",
            "0x1f",
            "0b1010",
            "0o17",
            "0O17",
            "1_000",
            "1_0",
            "1_000.5",
            "1e1_0",
            "[qty] = 0x10",
            "[amount] > 1_000",
        ],
    )
    def test_a_number_in_another_base_or_carrying_separators_is_refused(self, text: str) -> None:
        """Python's tokenizer is inherited, so these all compile today -- to values nobody wrote.

        ``0x10`` is sixteen, ``1_000`` is a thousand, and both spellings arrive from a person who
        was thinking about something else: a column of codes that look like hex, or a thousands
        separator that this language does not have. A whole number is written in digits, so
        anything else is refused rather than reinterpreted.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text)

    @pytest.mark.parametrize(
        "text",
        [
            "1e400",
            "1E400",
            "-1e400",
            "1e999",
            "1.5e400",
            "1.7976931348623157e309",
            "[amount] < 1e400",
            "1e400 > [amount]",
            "coalesce([amount], 1e400)",
            # The mirror image: a decimal small enough to arrive as exactly zero.
            "1e-400",
            "-1e-400",
            "1e-999",
        ],
    )
    def test_a_decimal_is_refused_unless_it_survives_as_the_number_written(self, text: str) -> None:
        """The one on this list that costs money is ``1e400``.

        It compiles today, to ``inf``, silently -- so ``[amount] < 1e400`` is a filter that keeps
        every row, written by someone who typed one character too many and got no warning at all.
        Its mirror, ``1e-400``, arrives as exactly ``0.0``, which turns a tolerance into "anything
        at all". Neither is a number an author could have meant, so both are refused.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1e5", 100_000.0),
            ("1.5e3", 1_500.0),
            ("1e-9", 1e-9),
            ("2.5e-3", 0.0025),
            ("-1e5", -100_000.0),
            ("1e308", 1e308),
        ],
    )
    def test_exponent_notation_stays_legal_while_it_names_a_finite_number(self, text: str, expected: float) -> None:
        """Exponent notation survives, on one condition: the value it names is finite.

        Keeping it is a judgement, and the argument is a tolerance. ``1e-9`` is how a numeric
        author writes a tolerance, and the alternative spelling -- ``0.000000001`` -- is a row of
        zeros that a reviewer has to count, which is a likelier mistake than the notation itself.
        The overflow that made exponents dangerous is refused directly above, so what is left is
        the useful half.
        """
        assert scalar(text) == expected
        assert math.isfinite(scalar(text))


class TestNodeCountLimit:
    """Depth is capped; breadth needs its own published cap rather than a limit it happens to hit."""

    def test_the_node_cap_is_published_reachable_and_generous(self) -> None:
        """A limit has to be visible, hittable, and wide enough for a real expression.

        Visible, because a limit an author cannot see is a limit they cannot design around --
        the same reason the other two are published. Hittable is the interesting one: breadth is
        bounded today only by ``MAX_TEXT_LENGTH``, and roughly two characters buy one node, so a
        node cap set above half the character limit could never fire and would be decoration. And
        generous, because the widest honest expression in a pipeline file is a membership test
        against a list of codes, which is exactly the shape a node cap counts.
        """
        assert isinstance(MAX_NODES, int)
        assert MAX_NODES >= 200, "a real membership test against a list of codes runs to a few hundred nodes"
        assert 2 * MAX_NODES + 40 <= MAX_TEXT_LENGTH, (
            "a node cap above half the character limit can never be reached, so it caps nothing"
        )

        wide_but_honest = " or ".join(f"[code] = '{letter}'" for letter in "ABCDEFGHIJ")
        assert compile_expr(wide_but_honest, columns=["code"]) is not None
        membership = "[code] in (" + ", ".join(f"'{letter}'" for letter in "ABCDEFGHIJKLMNOPQRST") + ")"
        assert compile_expr(membership, columns=["code"]) is not None

        # And it fires, on a flat expression that is inside the character limit and nowhere near
        # the depth limit -- so before the cap existed it was refused by nothing at all, or at a
        # larger size by the character limit, which reports a number unrelated to the problem.
        too_wide = "coalesce(" + "1," * (MAX_NODES + 10) + "1)"
        assert len(too_wide) <= MAX_TEXT_LENGTH, "this case must be refused by the node cap, not by the length gate"
        with pytest.raises(ExpressionError) as caught:
            compile_expr(too_wide)
        assert str(MAX_NODES) in str(caught.value), "the refusal has to name the limit that actually fired"


class TestWholeNumberRange:
    """The bound on a whole number is the signed 64-bit range, with the sign taken into account."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("-9223372036854775808", -9223372036854775808),
            ("9223372036854775807", 9223372036854775807),
            ("-9223372036854775807", -9223372036854775807),
            ("-9223372036854775808 < 0", True),
        ],
    )
    def test_the_signed_64_bit_range_is_spellable_at_both_ends_and_no_further(self, text: str, expected: Any) -> None:
        """``-9223372036854775808`` is a valid whole number and has to be writable.

        It was refused because the bound was measured on the digits alone, beneath the unary minus,
        where the value reads as 2**63 and does not fit. Measuring the bound on the value the
        author wrote -- sign included -- is both correct and easier to state: a whole number in
        this language is one that fits in signed 64 bits. Nothing wider is allowed in by the fix,
        including the double negation, which is the hole a fix that merely folded one minus sign
        into the bound would leave open.
        """
        assert scalar(text) == expected

        for out_of_range in (
            "9223372036854775808",
            "-9223372036854775809",
            "18446744073709551616",
            "- -9223372036854775808",
            "[qty] > 9223372036854775808",
        ):
            with pytest.raises(ExpressionError):
                compile_expr(out_of_range)


class TestRegexFunctions:
    """Two additions, arriving the only way this language grows: as names on the whitelist.

    A tool reshapes a frame; an expression function computes one column. A rewrite and a pattern
    test each compute one column, so they belong here rather than in tools of their own.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # 'gamma' holds two a's, so this case discriminates: replacing only the first would
            # leave 'gXmma' and a column half-cleaned in a way nobody would notice.
            ("regex_replace([label], 'a', 'X')", ["  XlphX  ", "BetX", "gXmmX", None]),
            ("regex_replace([label], '[aeiou]', '')", ["  lph  ", "Bt", "gmm", None]),
            ("regex_replace([label], ' +', ' ')", [" alpha ", "Beta", "gamma", None]),
            # A capture group reference is *not* interpreted: the replacement is text, verbatim.
            ("regex_replace([label], '(a)', '$1')", ["  $1lph$1  ", "Bet$1", "g$1mm$1", None]),
            ("regex_replace([code], 'A', '$5.00')", ["$5.00", "B", "C", None]),
            ("regex_replace([code], 'A', '$')", ["$", "B", "C", None]),
            # Null in, null out -- as everywhere else in this language.
            ("regex_replace([code], 'A', 'Z') is null", [False, False, False, True]),
            ("upper(regex_replace([label], 'a', 'X'))", ["  XLPHX  ", "BETX", "GXMMX", None]),
        ],
    )
    def test_regex_replace_rewrites_every_occurrence_and_takes_its_replacement_literally(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """Every occurrence, and a replacement that is text rather than a second little language.

        Two decisions, both about what a reviewer of a pipeline file can see. *Every* occurrence,
        because a substitution rule read off a page applies to the whole value; leaving the second
        match in place is the surprise. And the replacement is taken verbatim, because the engine
        underneath reads ``$name`` in a replacement as a capture group -- which quietly deletes the
        text when the group does not exist, so a replacement of ``'$5.00'`` lands as ``'.00'``.
        Nobody reviewing a file would see that coming, and this is a language for amounts.
        """
        assert checked(frame, text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Unanchored, like the extract it sits beside: does the pattern occur anywhere.
            ("regex_match([label], 'a')", [True, True, True, None]),
            ("regex_match([label], '^a')", [False, False, False, None]),
            ("regex_match([label], '^ +a')", [True, False, False, None]),
            ("regex_match([label], 'a$')", [False, True, True, None]),
            ("regex_match([code], 'A|C')", [True, False, True, None]),
            ("regex_match([label], '[0-9]')", [False, False, False, None]),
            ("regex_match([code], 'A') is null", [False, False, False, True]),
            ("regex_match([label], 'a') and [flag]", [True, False, True, None]),
        ],
    )
    def test_regex_match_reports_whether_the_pattern_occurs_anywhere(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """The gap ``contains`` cannot fill, because ``contains`` is deliberately literal.

        ``contains`` matches a substring and nothing else, so that a dot in an account code stays a
        dot. That leaves no way at all to ask a pattern question and get a truth value back, which
        is what a filter needs; ``regex_extract`` returns text. Hence one boolean sibling, named
        with the same prefix, unanchored like the extract beside it, and null-preserving like
        everything else here.
        """
        assert checked(frame, text) == expected

    @pytest.mark.parametrize(
        "pattern",
        [
            "(",
            ")",
            "[",
            "*",
            "a{2,1}",
            "(?P<",
            # The engine underneath supports no look-around at all. A pattern checked against
            # Python's own regex module instead would accept this and fail at row time, which is
            # the whole failure this test exists to close.
            "(?<=a)b",
            # And a pattern that compiles to more than the engine will hold.
            "a{1000}{1000}",
        ],
    )
    @pytest.mark.parametrize(
        "template",
        [
            "regex_extract([label], {p}, 1)",
            "regex_replace([label], {p}, 'X')",
            "regex_match([label], {p})",
        ],
    )
    def test_a_malformed_pattern_is_refused_at_compile_time(self, template: str, pattern: str) -> None:
        """A pattern the engine cannot compile is an ``ExpressionError``, not a row-time surprise.

        ``regex_extract`` shipped without this: a malformed pattern compiled, and the engine raised
        its own exception from inside a select, escaping the promise that every bad expression
        raises one type. So the check lands on all three functions at once, and it has to be the
        engine's own judgement -- the look-around case is legal in Python's regex module and
        illegal in the engine, so validating against the wrong one is worse than not validating.
        """
        text = template.format(p=f"'{pattern}'")
        with pytest.raises(ExpressionError):
            compile_expr(text)

    @pytest.mark.parametrize(
        "pattern",
        [
            "[a-z]+",
            "^a$",
            "a|b",
            r"\d{2,4}",
            "(?i)abc",
            "(?s).",
            "(a)(b)?",
            # Legal for the engine, rejected by Python's regex module: the other direction of the
            # same point. Validation follows the engine that will run the pattern.
            r"\p{Greek}",
        ],
    )
    def test_a_pattern_the_engine_accepts_still_compiles(self, frame: pl.DataFrame, pattern: str) -> None:
        """Pattern validation must not be stricter than the engine, or it refuses working patterns."""
        for template in (
            "regex_extract([label], {p}, 0)",
            "regex_replace([label], {p}, 'X')",
            "regex_match([label], {p})",
        ):
            text = template.format(p=f"'{pattern}'")
            assert compile_expr(text, columns=frame.schema) is not None
            frame.select(compile_expr(text).alias("out"))

    @pytest.mark.parametrize(
        "text",
        [
            # A pattern has to be readable in the file, and checkable before the run: a column
            # holding a different pattern per row is neither.
            "regex_replace([label], [code], 'X')",
            "regex_match([label], [code])",
            "regex_extract([label], [code], 1)",
            "regex_replace([label], 'a', [code])",
            "regex_match([label], 1)",
            "regex_replace([label], 1, 'X')",
            "regex_replace([label], 'a', 1)",
            "regex_replace([label], 'a', null)",
            "regex_match([label], true)",
        ],
    )
    def test_a_pattern_and_a_replacement_must_be_quoted_text(self, text: str) -> None:
        """Both are fixed decisions recorded in the file, so both are literals or nothing.

        This is what makes compile-time validation possible at all: a pattern that varies row by
        row cannot be checked before the run, and a rewrite whose replacement comes out of the data
        would let the data decide what the data becomes.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text)

    @pytest.mark.parametrize(
        ("text", "fragments"),
        [
            ("regex_replace([label], 'a')", ("regex_replace", "3")),
            ("regex_replace([label], 'a', 'X', 'Y')", ("regex_replace", "3")),
            ("regex_match([label])", ("regex_match", "2")),
            ("regex_match([label], 'a', 'b')", ("regex_match", "2")),
            # concat is variadic, like coalesce, and needs two pieces to join.
            ("concat([code])", ("concat", "2")),
            ("concat('x')", ("concat", "2")),
        ],
    )
    def test_the_new_functions_have_fixed_arity_and_are_offered_by_name(
        self, text: str, fragments: tuple[str, ...]
    ) -> None:
        """Fixed arity with no defaults, like the eleven before them, and listed where a mistake is made."""
        assert {"regex_replace", "regex_match", "concat"} <= ALLOWED_FUNCTIONS
        reported = message(text)
        for fragment in fragments:
            assert fragment in reported, f"{fragment!r} missing from {reported!r}"

        offered = message("sqrt([amount])")
        for name in ("regex_replace", "regex_match", "concat"):
            assert name in offered, (
                f"{name!r} is not offered to an author who reached for a function that does not exist"
            )


class TestMembershipListTypes:
    """A membership list is built into a column of values, and that build can fail on its own."""

    @pytest.mark.parametrize(
        "text",
        [
            "[code] in ('A', 1)",
            "[code] in (1, 'A')",
            "[code] in ('A', 'B', 2.5)",
            "[qty] in (1, 'A')",
            "[code] in ('A', true)",
            "[code] in ('A', date'2026-01-01')",
        ],
    )
    def test_a_membership_list_mixing_type_families_is_refused_rather_than_leaking(self, text: str) -> None:
        """A mixed list raises a raw ``TypeError`` out of compilation today, past every refusal here.

        The engine builds the list into a typed column, takes its type from the first value, and
        rejects the rest -- with its own exception, from inside ``compile_expr``, before a schema is
        even involved. So this is refused with or without one: a list of values to match is a
        decision recorded in a file, and a list that mixes text with numbers is a decision nobody
        made on purpose.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text)

    def test_a_membership_list_may_still_mix_whole_numbers_with_decimals(self, frame: pl.DataFrame) -> None:
        """Whole numbers and decimals are one family, so one list may hold both.

        The engine refuses this list too, on a technicality about the column it builds -- but here
        the author has written something perfectly sensible, so it is this module's job to make it
        work rather than to pass the refusal on.
        """
        assert checked(frame, "[amount] in (1, 100.0)") == [False, False, True, None]
        assert checked(frame, "[qty] in (1.0, 3)") == [True, False, True, None]
        assert evaluate(frame, "[amount] in (100, 1.5)") == [False, False, True, None]


class TestConcat:
    """Joining two pieces of text is a named function, because ``+`` between two text columns is not.

    ``+`` is refused on text (see ``TestSchemaDtypeChecking``) for a reason that is about money
    rather than about style: polars concatenates strings with ``+``, so two amount columns that
    arrived from an import as text add up to ``'10020'`` instead of ``120``. Both operands are in
    the same type family, so no dtype rule in this language would ever catch it -- the only way to
    close it is to refuse the operator. That would remove a real capability, so the capability
    arrives here instead, under a name that says out loud what it does.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("concat([code], [label])", ["A  alpha  ", "BBeta", "Cgamma", None]),
            # No separator is inserted, so an author who wants one writes it.
            ("concat([code], '-', [label])", ["A-  alpha  ", "B-Beta", "C-gamma", None]),
            ("concat('(', [code], ')')", ["(A)", "(B)", "(C)", None]),
            ("concat([code], [code], [code])", ["AAA", "BBB", "CCC", None]),
            # It returns text, so it nests both ways like every other text function.
            ("upper(concat([code], [label]))", ["A  ALPHA  ", "BBETA", "CGAMMA", None]),
            ("concat(upper([code]), strip([label]))", ["Aalpha", "BBeta", "Cgamma", None]),
            ("len(concat([code], 'xy'))", [3, 3, 3, None]),
            ("concat([code], [label]) = 'BBeta'", [False, True, False, None]),
            ("concat([code], '-x') in ('A-x', 'C-x')", [True, False, True, None]),
        ],
    )
    def test_concat_joins_text_into_one_column_with_no_separator_of_its_own(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """Two or more pieces of text, joined in the order written, as one text column."""
        assert checked(frame, text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # A null part makes the whole result null -- it does not vanish.
            ("concat([code], 'x')", ["Ax", "Bx", "Cx", None]),
            ("concat([code], [label])", ["A  alpha  ", "BBeta", "Cgamma", None]),
            ("concat([code], null)", [None, None, None, None]),
            ("concat([code], 'x') is null", [False, False, False, True]),
            # ... and the escape hatch is the one this language already teaches for null.
            ("concat(coalesce([code], ''), '-x')", ["A-x", "B-x", "C-x", "-x"]),
            ("coalesce(concat([code], 'x'), 'missing')", ["Ax", "Bx", "Cx", "missing"]),
        ],
    )
    def test_a_null_part_makes_the_whole_joined_value_null(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """Null in, null out -- the same rule as every other operation here, and the safer one.

        The engine offers both behaviours: propagate the null, or treat a null part as empty text.
        Propagating is chosen for two reasons. It is what the rest of this language does, and one
        function quietly disagreeing is the one nobody remembers. And where a joined value is a key
        for matching, a row missing one part has an *incomplete* key: collapsing it to the parts
        that are present produces a shorter key that matches the wrong rows, while a null key
        matches nothing and shows up as a null. The author who wants the other behaviour writes it
        -- ``concat(coalesce([code], ''), '-x')`` -- exactly as they do to make a null row
        participate in a comparison.
        """
        assert checked(frame, text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            # A number, a date or a truth value is not text. polars would stringify each of them
            # without a word -- 'A1', 'A2026-01-01', 'Atrue' -- and a value that was a number when
            # somebody checked it is the hardest kind of defect to find later.
            "concat([code], 1)",
            "concat(1, [code])",
            "concat([code], 2.5)",
            "concat([code], true)",
            "concat([code], date'2026-01-01')",
            "concat(1, 2)",
            # And the operator it replaces stays refused, with or without a schema, wherever both
            # sides are knowably text.
            "'a' + 'b'",
            "'100' + '20'",
        ],
    )
    def test_concat_refuses_a_part_that_is_not_text_and_so_does_the_operator_it_replaces(self, text: str) -> None:
        """A number handed to ``concat`` is refused rather than quietly turned into text.

        ``concat([code], 1)`` is the exact shape of mistake this cycle exists to catch, so it is
        refused with no schema in hand at all: the type of a literal is knowable from the text, and
        the author who really wants the digit writes ``'1'``. The last two cases are the same rule
        seen from the operator's side -- ``'100' + '20'`` is ``'10020'`` in the engine, and needs no
        column and no schema for that to be wrong.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text)


# ---------------------------------------------------------------------------
# The rows a predicate keeps, on a frame built to tell rules apart
# ---------------------------------------------------------------------------
#
# These cases were first written to prove that a compile target for SQL kept exactly the rows the
# frame keeps. That target has been removed (see docs/design/sql-target.md). What the frame keeps
# is pinned here instead, because several cases are the only test of a rule: a null inside a
# membership list, an empty needle, a needle holding a wildcard character, case-sensitive matching,
# and text compared in byte order, where 'B' sorts before 'a'.


# The rows each predicate keeps, by key. Every case splits the table, except the few that pin
# "this matches nothing" or "this matches everything".
KEPT_ROWS: tuple[tuple[str, list[int]], ...] = (
    ("[code] = 'A'", [1, 7]),
    ("[code] != 'A'", [2, 3, 5, 6, 8, 9, 10]),
    ("[code] = 'a'", [6]),
    ("[code] = ''", [5]),
    ("[label] = 'é'", [5]),
    ('[amount] > 0', [1, 5, 6, 7, 8, 9, 10]),
    ('[amount] >= 0', [1, 3, 5, 6, 7, 8, 9, 10]),
    ('[amount] < 0', [2]),
    ('[amount] = 0', [3]),
    ('[amount] != 0', [1, 2, 5, 6, 7, 8, 9, 10]),
    ('[qty] <= 2', [1, 2]),
    ('[flag]', [1, 3, 6, 9]),
    ('not [flag]', [2, 5, 8, 10]),
    ('[flag] = true', [1, 3, 6, 9]),
    ('[flag] = false', [2, 5, 8, 10]),
    ('[flag] != true', [2, 5, 8, 10]),
    ('[qty] > 1.5', [2, 3, 5, 7, 8, 9, 10]),
    ('[qty] = 3.0', [3, 5, 7]),
    ('[amount] = 3', [5]),
    ('[qty] in (1, 3.0)', [1, 3, 5, 7]),
    ("[seen] >= date'2026-01-01'", [1, 2, 5, 7, 8, 9, 10]),
    ("[seen] < date'2026-01-01'", [3]),
    ("[seen] = date'2026-01-01'", [1, 5]),
    ('[seen] is null', [4, 6]),
    ('[code] is null', [4]),
    ('[code] is not null', [1, 2, 3, 5, 6, 7, 8, 9, 10]),
    ('[code] = null', []),
    ("[code] in ('A', 'B')", [1, 2, 7]),
    ("[code] not in ('A', 'B')", [3, 5, 6, 8, 9, 10]),
    ('[qty] in (1, 3)', [1, 3, 5, 7]),
    ('[qty] not in (1, 3)', [2, 8, 9, 10]),
    ("not ([code] in ('A', 'B'))", [3, 5, 6, 8, 9, 10]),
    ("[code] = 'X' and false", []),
    ("[code] = 'X' or true", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("[code] = 'X' and true", []),
    ("[code] = 'X' or false", []),
    ("not ([code] = 'A')", [2, 3, 5, 6, 8, 9, 10]),
    ('not (not [flag])', [1, 3, 6, 9]),
    ("[code] = 'A' and [qty] = 1", [1]),
    ("[code] = 'A' or [code] = 'B'", [1, 2, 7]),
    ('[qty] > 2 and not [flag]', [5, 8, 10]),
    ("[code] = 'A' and [qty] > 0 and [seen] is not null", [1, 7]),
    ("([code] = 'A' or [code] = 'B') and [flag] = false", [2]),
    ("[code] = 'A' or ([code] = 'B' and [flag] = false)", [1, 2, 7]),
    ("coalesce([code], 'Z') = 'Z'", [4]),
    ('coalesce([qty], 0) > 2', [3, 5, 7, 8, 9, 10]),
    ('coalesce([code], [label]) is null', [4]),
    ("contains([label], 'lph')", [1]),
    ("contains([label], 'ALPHA')", [6]),
    ("contains([label], 'a.p')", []),
    ("contains([label], '%')", []),
    ("contains([label], 'é')", [5]),
    ("contains([label], '')", [1, 2, 3, 5, 6, 7, 8, 9, 10]),
    ("not contains([label], 'a')", [5, 6, 8]),
    ('contains([label], [code])', [2, 5]),
    ("contains(coalesce([label], 'zz'), 'z')", [4]),
    ("starts_with([label], 'al')", [1]),
    ("starts_with([label], 'AL')", [6]),
    ("starts_with([label], 'é')", [5]),
    ("starts_with([code], '')", [1, 2, 3, 5, 6, 7, 8, 9, 10]),
    ('starts_with([label], [code])', [2, 5]),
    ("ends_with([label], 'ha')", [1]),
    ("ends_with([label], 'A')", [6]),
    ("ends_with([label], 'IR')", [8]),
    ("ends_with([label], '')", [1, 2, 3, 5, 6, 7, 8, 9, 10]),
    ("ends_with([label], 'zzzzzzzzzz')", []),
    ('ends_with([label], [code])', [5]),
    ("ends_with(coalesce([label], 'zz'), 'z')", [4]),
    ('len([label]) = 6', [8, 9]),
    ('len([label]) = 1', [5]),
    ('len([code]) = 0', [5]),
    ("concat([code], 'x') = 'Ax'", [1, 7]),
    ("concat([code], 'x') is null", [4]),
    ("concat([code], [label]) = 'ahaa'", []),
    ("[code] < 'B'", [1, 5, 7]),
    ("[code] >= 'B'", [2, 3, 6, 8, 9, 10]),
    ("[code] > 'A'", [2, 3, 6, 8, 9, 10]),
    ("[label] < 'z'", [1, 2, 3, 6, 7, 9]),
    ("[label] > 'ALPHA'", [1, 2, 3, 5, 8, 9, 10]),
    ("[label] < 'beta'", [1, 2, 6, 7]),
)

# What a predicate answers on the row that is null in every column, and therefore whether that row
# survives a filter. Only true keeps a row: null and false both drop it.
NULL_LOGIC_CASES: tuple[tuple[str, Any], ...] = (
    ("[code] = 'A'", None),
    ("[code] != 'A'", None),
    ("not ([code] = 'A')", None),
    ("[code] = null", None),
    ("[code] in ('A', 'B')", None),
    ("[code] not in ('A', 'B')", None),
    ("[code] = 'A' and false", False),
    ("[code] = 'A' or true", True),
    ("[code] = 'A' and true", None),
    ("[code] = 'A' or false", None),
    ("not [flag]", None),
    ("[flag] and [qty] > 0", None),
    ("[flag] or [qty] > 0", None),
    ("contains([label], 'a')", None),
    ("starts_with([label], 'a')", None),
    ("ends_with([label], 'a')", None),
    ("concat([code], 'x') = 'Ax'", None),
    ("[code] is null", True),
    ("[code] is not null", False),
    ("coalesce([code], 'Z') = 'Z'", True),
)


@pytest.fixture
def rows() -> pl.DataFrame:
    """Ten rows carrying every family, every null shape, and the values that tell rules apart.

    Key 4 is null in every column. Key 5 holds an empty string and a non-ASCII label. Key 6 holds
    the lowercase twin of key 1's code, so a case-insensitive comparison shows up as a row count.
    Key 7 holds tab-padded text. Key 8 holds an uppercase non-ASCII label and the largest whole
    number a column can hold. Keys 9 and 10 hold text whose case folding is unusual.
    """
    return pl.DataFrame(
        {
            "key": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "code": ["A", "B", "C", None, "", "a", "A", "ha", "D", "E"],
            "label": ["alpha", "Beta", "gamma", None, "é", "ALPHA", "\tspaced\t", "ÉCLAIR", "straße", "İstanbul"],
            "amount": [10.5, -2.0, 0.0, None, 3.0, 100.0, 7.25, 1.0, 2.0, 4.0],
            "qty": [1, 2, 3, None, 3, None, 3, 9223372036854775807, 4, 5],
            "flag": [True, False, True, None, False, True, None, False, True, False],
            "seen": [
                date(2026, 1, 1),
                date(2026, 6, 15),
                date(2025, 12, 31),
                None,
                date(2026, 1, 1),
                None,
                date(2026, 6, 15),
                date(2026, 3, 1),
                date(2026, 2, 1),
                date(2026, 4, 1),
            ],
        }
    )


def kept_by_frame(rows: pl.DataFrame, text: str) -> list[int]:
    """The keys a filter on ``text`` keeps."""
    return sorted(rows.filter(compile_expr(text, rows.schema))["key"].to_list())


class TestTheRowsAPredicateKeeps:
    """Which rows a predicate keeps, including the null row and the edge cases of text matching."""

    @pytest.mark.parametrize(("text", "kept"), KEPT_ROWS)
    def test_a_predicate_keeps_exactly_these_rows(self, rows: pl.DataFrame, text: str, kept: list[int]) -> None:
        """Each case keeps the rows listed. A wildcard or a dot in a needle is a plain character."""
        assert kept_by_frame(rows, text) == kept

    @pytest.mark.parametrize(("text", "on_the_null_row"), NULL_LOGIC_CASES)
    def test_the_row_that_is_null_everywhere_follows_three_valued_logic(
        self, rows: pl.DataFrame, text: str, on_the_null_row: Any
    ) -> None:
        """``and`` with false is false and ``or`` with true is true, even beside an unknown.

        Every other test against a null is null, and a filter drops a null row. The text functions
        are listed because a null subject has to make the whole test null rather than false.
        """
        values = rows.select(compile_expr(text, rows.schema).alias("out"))["out"].to_list()
        assert values[3] is on_the_null_row
        assert (4 in kept_by_frame(rows, text)) is (on_the_null_row is True)

    def test_a_null_inside_a_not_in_list_still_keeps_every_row_that_does_not_match(self, rows: pl.DataFrame) -> None:
        """``[code] not in ('A', null)`` keeps the seven rows whose code is neither 'A' nor null."""
        assert kept_by_frame(rows, "[code] not in ('A', null)") == [2, 3, 5, 6, 8, 9, 10]


class TestYearReadsThePartOfADate:
    """`year` is the first function here that reads a part of a value rather than transforming it.

    It exists for one thing and the thing is money-shaped: a term whose end year was mistyped runs
    backwards, and repairing it means trying the year the term starts in and the one after that. No
    combination of the other names can produce that number -- a year captured out of text is text,
    and neither text nor a date may be added to.
    """

    def test_the_year_of_a_date_is_a_number_that_can_be_added_to(self) -> None:
        frame = pl.DataFrame({"d": [date(2026, 3, 9), date(1999, 12, 31)]})

        assert frame.select(compile_expr("year([d])", frame.schema)).to_series().to_list() == [2026, 1999]
        assert frame.select(compile_expr("year([d]) + 1", frame.schema)).to_series().to_list() == [2027, 2000]

    def test_a_null_date_has_a_null_year_rather_than_a_zero(self) -> None:
        """A year nobody stated is not year nought, and a repair must not act on one."""
        frame = pl.DataFrame({"d": [None]}, schema={"d": pl.Date()})

        assert frame.select(compile_expr("year([d])", frame.schema)).to_series().to_list() == [None]

    @pytest.mark.parametrize("text", ["year([label])", "year([qty])", "year([flag])"])
    def test_year_refuses_anything_that_is_not_a_date(self, rows: pl.DataFrame, text: str) -> None:
        """Text that looks like a year is still text, and the refusal says which column and what it holds."""
        with pytest.raises(ExpressionError) as caught:
            compile_expr(text, rows.schema)

        assert "year()" in str(caught.value)

    def test_the_result_is_a_number_for_the_purpose_of_nesting(self, rows: pl.DataFrame) -> None:
        """So a misuse one level up is refused at compile time rather than inside the engine."""
        with pytest.raises(ExpressionError):
            compile_expr("upper(year([seen]))", rows.schema)

        assert compile_expr("abs(year([seen])) > 2000", rows.schema) is not None


class TestAddMonthsShiftsADateByWholeMonths:
    """`add_months` moves a date forward by a whole number of months and hands back a date.

    It is the first function here that takes a date and gives a date. Every other name either
    reads a part of a value or rewrites text, so before this there was no way to say "the month
    after this one" at all. The caller that asked for it spreads one billed amount across the
    months of a term: given a start date and a count of months, the last month of the term is the
    month ``count - 1`` after the start. That shape is settled. What these tests settle is
    everything around it -- what the second argument may be, what a missing day of the month does,
    and where the expression may and may not run.
    """

    def test_a_date_moves_forward_by_the_number_of_months_given(self) -> None:
        """The plain case, including a shift that crosses into the next year.

        Zero is here too. A term of one month asks for ``add_months(start, 0)``, so zero has to be
        an ordinary answer rather than an edge case, and it has to give the same date back.
        """
        frame = pl.DataFrame({"d": [date(2026, 1, 1), date(2026, 6, 15), date(2025, 12, 31)]}, schema={"d": pl.Date()})

        assert checked(frame, "add_months([d], 3)") == [date(2026, 4, 1), date(2026, 9, 15), date(2026, 3, 31)]
        assert checked(frame, "add_months([d], 0)") == [date(2026, 1, 1), date(2026, 6, 15), date(2025, 12, 31)]
        assert checked(frame, "add_months([d], 2)")[2] == date(2026, 2, 28)

    def test_a_negative_count_moves_the_date_backwards(self) -> None:
        """Backwards is the same function with a minus sign, not a second function.

        A term is sometimes counted back from its end rather than forward from its start, and an
        author who has to reach for a different name to do that will write the wrong one.
        """
        frame = pl.DataFrame({"d": [date(2026, 1, 15), date(2026, 5, 20)]}, schema={"d": pl.Date()})

        assert checked(frame, "add_months([d], -1)") == [date(2025, 12, 15), date(2026, 4, 20)]
        assert checked(frame, "add_months([d], -12)") == [date(2025, 1, 15), date(2025, 5, 20)]

    def test_a_day_the_target_month_does_not_have_lands_on_that_months_last_day(self) -> None:
        """The decision that matters. 31 January plus one month is 28 February, not 3 March.

        There is no 31 February, so something has to give, and there are only two honest answers:
        stop at the end of the target month, or spill over into the month after it. Spilling over
        is wrong for the thing this function is for. A term that starts on the 31st is a term whose
        months are January, February, March; if February spills into March then March is counted
        twice and the term grows a month that nobody sold. Clamping keeps one row per month, which
        is the only property the caller actually depends on.

        The same rule applies going backwards, and the last case shows why that has to be said out
        loud: shifting forward and then back does not always return the date you started on. 31
        March minus one month is 28 February, and 28 February plus one month is 28 March. The
        function is deliberately not reversible, because a calendar is not reversible.
        """
        frame = pl.DataFrame({"d": [date(2026, 1, 31), date(2026, 3, 31), date(2026, 1, 31)]}, schema={"d": pl.Date()})

        assert checked(frame, "add_months([d], 1)")[0] == date(2026, 2, 28)
        assert checked(frame, "add_months([d], -1)")[1] == date(2026, 2, 28)
        assert checked(frame, "add_months([d], 3)")[2] == date(2026, 4, 30)

    def test_a_leap_day_clamps_in_a_year_that_has_no_leap_day(self) -> None:
        """29 February plus twelve months is 28 February. Plus forty-eight it is 29 February again.

        The second half is the point. Clamping is not a habit the function falls into once a date
        has been near a month end; it happens only when the target month is short, so a leap day
        that lands in another leap year survives intact.
        """
        frame = pl.DataFrame({"d": [date(2024, 2, 29)]}, schema={"d": pl.Date()})

        assert checked(frame, "add_months([d], 12)") == [date(2025, 2, 28)]
        assert checked(frame, "add_months([d], 48)") == [date(2028, 2, 29)]

    def test_a_null_on_either_side_gives_null(self) -> None:
        """A date nobody stated has no month after it, and a count nobody stated is not zero.

        Treating a missing count as zero would quietly turn an unknown term into a one-month term,
        and that lands in a total as a real number rather than as a gap somebody can see.
        """
        frame = pl.DataFrame(
            {"d": [None, date(2026, 1, 1)], "months": [3, None]}, schema={"d": pl.Date(), "months": pl.Int64()}
        )

        assert checked(frame, "add_months([d], 3)") == [None, date(2026, 4, 1)]
        assert checked(frame, "add_months([d], [months])") == [None, None]

    def test_the_count_may_be_a_column_or_a_sum_rather_than_a_typed_number(self) -> None:
        """The real caller writes ``add_months([start], [months] - 1)``, so that has to compile.

        A count that has to be typed into the file would be useless here: the number of months is
        different on every row. It comes out of the data, and the ``- 1`` is arithmetic on it,
        because the last month of a three-month term is two months after the first.
        """
        frame = pl.DataFrame(
            {"start": [date(2026, 1, 1), date(2026, 1, 31), date(2025, 12, 1)], "months": [1, 2, 3]},
            schema={"start": pl.Date(), "months": pl.Int64()},
        )

        assert checked(frame, "add_months([start], [months])") == [
            date(2026, 2, 1),
            date(2026, 3, 31),
            date(2026, 3, 1),
        ]
        assert checked(frame, "add_months([start], [months] - 1)") == [
            date(2026, 1, 1),
            date(2026, 2, 28),
            date(2026, 2, 1),
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "add_months([label], 1)",
            "add_months([qty], 1)",
            "add_months([amount], 1)",
            "add_months([flag], 1)",
            "add_months('2026-01-01', 1)",
            "add_months(20260101, 1)",
        ],
    )
    def test_the_first_argument_has_to_be_a_date(self, rows: pl.DataFrame, text: str) -> None:
        """A date written as text is still text, and a date written as digits is still a number.

        Both of those arrive from real imports, and both would be answered by the engine in some
        way -- the wrong way. The refusal names the function so the message points at the call
        rather than at the column, spelled ``add_months()`` the way every other refusal here spells
        the function it is about.
        """
        with pytest.raises(ExpressionError) as caught:
            compile_expr(text, rows.schema)

        assert "add_months()" in str(caught.value)

    @pytest.mark.parametrize(
        "text",
        [
            "add_months([seen], 1.5)",
            "add_months([seen], [amount])",
            "add_months([seen], '1')",
            "add_months([seen], [label])",
            "add_months([seen], [seen])",
            "add_months([seen], date'2026-01-01')",
            "add_months([seen], true)",
            "add_months([seen], [flag])",
        ],
    )
    def test_the_second_argument_has_to_be_a_whole_number(self, rows: pl.DataFrame, text: str) -> None:
        """Half a month is not a length of time this function can move a date by.

        A decimal is the case worth refusing rather than rounding. Nobody writes 1.5 months on
        purpose; it arrives because a count was divided by something, and rounding it here would
        bury that division inside a date that looks perfectly ordinary. A decimal column is refused
        for the same reason a decimal literal is, because the schema says which one it is before a
        row is read.

        The message names ``add_months()``, so an author reading it knows which of the two
        arguments the complaint is about.
        """
        with pytest.raises(ExpressionError) as caught:
            compile_expr(text, rows.schema)

        assert "add_months()" in str(caught.value)

    @pytest.mark.parametrize("text", ["add_months([seen])", "add_months([seen], 1, 2)", "add_months([seen], 1, 2, 3)"])
    def test_add_months_takes_exactly_two_arguments(self, rows: pl.DataFrame, text: str) -> None:
        """Two arguments, always. There is no default count, because a default hides the count."""
        with pytest.raises(ExpressionError) as caught:
            compile_expr(text, rows.schema)

        assert "2 argument" in str(caught.value)

    def test_the_result_is_a_date_for_the_purpose_of_nesting(self, rows: pl.DataFrame) -> None:
        """What comes out is a date, so the check one level up happens here rather than in the engine.

        This is what makes the function composable. ``year(add_months(...))`` is the shape a term
        repair wants, and it only works if the compiler knows the inner call gives a date.
        """
        assert compile_expr("year(add_months([seen], 3))", rows.schema) is not None
        assert compile_expr("add_months([seen], 1) = date'2026-02-01'", rows.schema) is not None
        assert compile_expr("add_months(add_months([seen], 1), 1)", rows.schema) is not None

        with pytest.raises(ExpressionError):
            compile_expr("upper(add_months([seen], 1))", rows.schema)
        with pytest.raises(ExpressionError):
            compile_expr("abs(add_months([seen], 1))", rows.schema)

    def test_add_months_gives_the_same_answer_whether_the_date_is_a_column_or_written_out(self) -> None:
        """A date literal is a date, so the function takes one, and the clamping is the same.

        Worth pinning because the two reach the compiler by different paths: a literal is lifted
        out by the scan before parsing, a column is looked up in the schema.
        """
        assert scalar("add_months(date'2026-01-31', 1)") == date(2026, 2, 28)
        assert scalar("add_months(date'2026-01-31', -1)") == date(2025, 12, 31)


class TestAddMonthsCountsThatArriveThroughAnExpression:
    """The count of months is checked however it was built, not only when it is written out.

    The rule itself is settled: half a month is not a distance a date can move, so a decimal count
    is refused. What was not settled is where the check happens. A decimal count was caught when
    the author typed one, and when it came from a decimal column, and nowhere else. Every other
    route to a decimal -- arithmetic, abs(), round(), coalesce(), a division of two whole numbers
    -- compiled against the schema and then failed inside the engine.

    That is the thing this file's sweep exists to forbid. With a schema in hand a bad expression
    raises ExpressionError and a good one runs, and nothing sits in between. An engine error at
    select time is the in-between case: it names the engine's problem, arrives too late to point at
    the line of the file, and reaches whoever is running the pipeline rather than whoever wrote it.

    So these tests take the rule the other way round from the first round. Rather than listing the
    counts that are wrong, they pin that no route to the count can escape: each one is refused
    where it is written, or it runs.
    """

    @pytest.mark.parametrize(
        "text",
        [
            # Arithmetic on a decimal column. The first round left this open on purpose.
            "add_months([seen], [amount] - 1)",
            "add_months([seen], [amount] + 1)",
            "add_months([seen], [amount] * 2)",
            # The three functions that hand back a number. round(x, 0) is the one to look at
            # twice: it reads like it produces a whole number and it does not.
            "add_months([seen], round([amount], 0))",
            "add_months([seen], abs([amount]))",
            "add_months([seen], coalesce([amount], 0))",
            # Two whole numbers, a decimal result. Division here is true division, the way it is
            # everywhere else in this language, so 3 / 2 is 1.5 and not 1. Nothing in the operands
            # is a decimal, which is why a fix that only looks at the columns will miss it.
            "add_months([seen], [qty] / 2)",
            "add_months([seen], [qty] / [qty])",
            # A decimal reached by a longer route, to pin that the check is not one level deep.
            "add_months([seen], abs([qty] / 2))",
            "add_months([seen], coalesce([amount] - 1, 0))",
        ],
    )
    def test_a_decimal_count_is_refused_however_it_was_built(self, rows: pl.DataFrame, text: str) -> None:
        """Each of these raises ExpressionError at compile time. None reaches the engine.

        The message names ``add_months()``, because the count is the second of two arguments and an
        author has to be told which one the complaint is about. It is the same message a bare
        decimal column already gets, and it should stay that way: the author's mistake is the same
        mistake whether they divided in the expression or in the column.
        """
        with pytest.raises(ExpressionError) as caught:
            compile_expr(text, rows.schema)

        assert "add_months()" in str(caught.value)

    def test_a_whole_number_count_still_works_however_it_was_built(self) -> None:
        """A count built out of whole numbers runs, and gives the dates it should.

        This is the other half of the rule, and it is the half a fix can get wrong. Refusing every
        count that is not a bare column would close the gap and pass the refusal test above, and it
        would also break the caller that asked for this function. ``add_months([start], [months] -
        1)`` is the shape the production pipeline uses: the last month of a three-month term is two
        months after the first, so the ``- 1`` is not decoration.

        The three functions are here for the same reason. abs(), round() and coalesce() on a whole
        number give a whole number. The engine says so, and the rule is "this is not whole", not
        "we cannot tell".
        """
        frame = pl.DataFrame(
            {"start": [date(2026, 1, 1), date(2026, 1, 31), date(2025, 12, 1)], "months": [1, 2, 3]},
            schema={"start": pl.Date(), "months": pl.Int64()},
        )

        assert checked(frame, "add_months([start], [months] - 1)") == [
            date(2026, 1, 1),
            date(2026, 2, 28),
            date(2026, 2, 1),
        ]

        shifted = [date(2026, 2, 1), date(2026, 3, 31), date(2026, 3, 1)]
        assert checked(frame, "add_months([start], abs([months]))") == shifted
        assert checked(frame, "add_months([start], round([months], 0))") == shifted
        assert checked(frame, "add_months([start], coalesce([months], 0))") == shifted

    def test_a_count_that_is_the_difference_between_two_years_works(self) -> None:
        """``year([a]) - year([b])`` is a whole number, so it is allowed.

        This one is a decision rather than a reading of the old behaviour, so here is the reasoning.
        A year is always whole, and the difference of two whole numbers is whole, so nothing about
        this count can be half a month. Refusing it would mean the rule had quietly changed from
        "this is not whole" to "we cannot see whether this is whole", and those are different
        promises. The second one gets less true every time a function is added, because every new
        function is unknown until someone remembers to teach the check about it.

        It is also the shape the caller that asked for add_months() already has in hand. A term
        whose end year was mistyped is repaired by counting the years between two dates, and the
        count of months follows from that.
        """
        frame = pl.DataFrame(
            {
                "d": [date(2026, 1, 15), date(2026, 1, 15)],
                "a": [date(2028, 5, 1), date(2026, 5, 1)],
                "b": [date(2026, 5, 1), date(2026, 5, 1)],
            },
            schema={"d": pl.Date(), "a": pl.Date(), "b": pl.Date()},
        )

        assert checked(frame, "add_months([d], year([a]) - year([b]))") == [date(2026, 3, 15), date(2026, 1, 15)]

    def test_a_decimal_months_column_is_refused_inside_arithmetic_as_well(self) -> None:
        """``add_months([start], [months] - 1)`` is refused when ``months`` holds decimals.

        The same text, the same shape, one dtype different. The whole-number version above is
        pinned as working, and on its own it reads as a blanket allowance for arithmetic. It is
        not. What makes the count acceptable is that it is whole, and the schema says whether it is
        before a row is read.

        This is how a decimal count arrives in production. Nobody types 1.5 months. A column that
        was Int64 comes back Float64 after a load that saw a null, or after a join, or out of a
        source that reports every number as a decimal -- and the expression above it never changed.
        """
        frame = pl.DataFrame(
            {"start": [date(2026, 1, 1), date(2026, 1, 31)], "months": [1.0, 2.0]},
            schema={"start": pl.Date(), "months": pl.Float64()},
        )

        with pytest.raises(ExpressionError) as caught:
            compile_expr("add_months([start], [months] - 1)", frame.schema)

        assert "add_months()" in str(caught.value)

    def test_a_negated_null_count_behaves_like_a_null_count(self) -> None:
        """``add_months([d], -null)`` gives null, the same as ``add_months([d], null)``.

        A null count already gives a null date, and that is settled. A minus sign in front of a
        null changes nothing about it: there is no value to negate, so the answer is still the
        value nobody stated.

        Today the two disagree. The negation is refused and the bare null is not, which leaves the
        language saying that a missing count is fine but a missing count with a minus sign is a
        mistake. An author who wrote ``-[months]`` and hit a null row would have no way to read
        that message.
        """
        frame = pl.DataFrame({"d": [date(2026, 1, 31), date(2026, 6, 15)]}, schema={"d": pl.Date()})

        assert checked(frame, "add_months([d], null)") == [None, None]
        assert checked(frame, "add_months([d], -null)") == [None, None]

    def test_shifting_a_date_back_and_then_forward_does_not_always_return_it(self) -> None:
        """31 March, back one month and forward again, is 28 March.

        The clamping rule says this happens, and until now nothing evaluated it. March has 31 days
        and February has 28, so the step back lands on 28 February, and the step forward has no
        reason to remember where it came from. The same thing happens in the other direction: 31
        January forward and back is 28 January.

        It is worth an assertion rather than a sentence in a docstring because it is a real way to
        lose a day. An author who writes a round trip to normalise a date will get a different date
        out for exactly the rows that sit at the end of a long month, which is where billing dates
        live. The second row of each pair is the control: a date in a month the other month can
        hold comes back unchanged.
        """
        frame = pl.DataFrame({"d": [date(2026, 3, 31), date(2026, 1, 31)]}, schema={"d": pl.Date()})

        assert checked(frame, "add_months(add_months([d], -1), 1)") == [date(2026, 3, 28), date(2026, 1, 31)]
        assert checked(frame, "add_months(add_months([d], 1), -1)") == [date(2026, 3, 30), date(2026, 1, 28)]


# ---------------------------------------------------------------------------
# The escape: a doubled quote inside a literal
# ---------------------------------------------------------------------------


class TestADoubledQuoteIsAnEscape:
    """``'O''Brian'`` is the seven characters ``O'Brian``, and nothing else moves.

    SQL and Excel spell an escaped quote this way, which is the convention this audience already
    knows. Backslash-free deliberately: a backslash has to survive a YAML file on the way here, and
    the interaction between the two is not something a person editing a pipeline should have to
    hold in their head.

    The change is strictly additive. Before it, adjacent string literals were a clean syntax error
    -- a literal is lifted out into a table and replaced by a call, and two calls side by side is
    not something the parser underneath will read -- so there was no expression that meant one thing
    yesterday and another today. What there was instead was a value an author could not express: a
    name with an apostrophe in it, which is most of a page of any real contact list.

    The doubling has to be undone *before* the parser sees the text. Handing Python two adjacent
    literals would get implicit concatenation -- ``'O' 'Brian'`` is ``OBrian`` there -- which drops
    the apostrophe and returns a value that looks plausible, matches nothing, and reports success.
    """

    @pytest.fixture
    def names(self) -> pl.DataFrame:
        """Rows whose values are exactly the ones a quote makes hard to write.

        The decoys are the point. ``OBrian`` is what Python's implicit concatenation would produce,
        ``O Brian`` is what a scan that dropped the quote would leave, and ``O''Brian`` is a value
        that really does hold two quotes -- so an implementation that unescapes one time too many
        matches it by mistake. None of the three may be matched by a predicate naming ``O'Brian``.
        """
        return pl.DataFrame({"name": ["O'Brian", "OBrian", "O Brian", "O''Brian", "Midway"]})

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("'O''Brian'", "O'Brian"),
            ("'it''s the driver''s'", "it's the driver's"),
            ("'''leading'", "'leading"),
            ("'trailing'''", "trailing'"),
            ("''''", "'"),
            ("''", ""),
            ("'a''''b'", "a''b"),
        ],
    )
    def test_a_doubled_quote_inside_a_literal_is_one_quote(self, text: str, expected: str) -> None:
        """One quote, two quotes, a quote at either end, and a literal that is nothing but a quote.

        ``''`` is still the empty string rather than the start of an escape, which is the one place
        the two readings meet: a doubled quote is an escape only when the literal continues past it.
        """
        assert scalar(text) == expected

    def test_a_literal_carrying_a_quote_matches_exactly_the_value_it_spells(self, names: pl.DataFrame) -> None:
        """The point of the whole change: a name with an apostrophe is a value this language can say.

        Asserted against the decoys, because every plausible way to get this wrong produces a
        predicate that compiles and keeps the wrong rows.
        """
        assert evaluate(names, "[name] = 'O''Brian'") == [True, False, False, False, False]
        assert evaluate(names, "[name] = 'O''''Brian'") == [False, False, False, True, False]
        assert evaluate(names, "[name] in ('O''Brian', 'Midway')") == [True, False, False, False, True]
        assert evaluate(names, "contains([name], '''')") == [True, False, False, True, False]

    @pytest.mark.parametrize("text", ["'a' 'b'", "[note] = 'a' 'b'", "'a'  'b'", "concat('a' 'b', 'c')"])
    def test_adjacent_literals_separated_by_whitespace_are_still_refused(self, text: str) -> None:
        """Python's implicit concatenation is not part of this language, and did not become part of it.

        This is the refusal the escape rests on. If ``'a' 'b'`` were ever read as ``ab``, then
        ``'O' 'Brian'`` would be ``OBrian`` -- a value nobody wrote, arriving without a word.
        """
        text_out = message(text)
        assert any(fragment in text_out for fragment in ("syntax", "understood")), text_out

    @pytest.mark.parametrize("text", ["[code] = 'A", "'a''", "'''", "[code] = 'it''s"])
    def test_a_literal_that_is_never_closed_still_says_so(self, text: str) -> None:
        """A trailing escape leaves the literal open, and the refusal is the one it always was.

        ``'a''`` is an ``a``, an escaped quote, and then nothing: the quote that would have closed
        the literal was spent on the escape. "A quote is never closed" is exactly what happened and
        exactly what the author needs to hear -- a message about the escape would send them to fix
        the wrong half.
        """
        assert "quote" in message(text)

    def test_a_pattern_argument_may_carry_a_doubled_quote(self, names: pl.DataFrame) -> None:
        """A pattern is a string literal too, so it escapes the same way and reaches the engine whole.

        Worth its own test because a pattern is the one literal handed onward to another parser. The
        doubling is undone before that hand-off, so the engine compiles ``O'`` -- the pattern the
        author meant -- and never sees anything about the escape.
        """
        assert evaluate(names, "regex_match([name], 'O''B')") == [True, False, False, False, False]
        assert evaluate(names, "regex_extract([name], '(O'')', 1)") == ["O'", None, None, "O'", None]
        assert evaluate(names, "regex_replace([name], 'O''', 'X')") == [
            "XBrian",
            "OBrian",
            "O Brian",
            "X'Brian",
            "Midway",
        ]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("''", ""),
            ("'hello'", "hello"),
            ("'it is 5 > 3'", "it is 5 > 3"),
            ("'a=b'", "a=b"),
            ("'x [y]'", "x [y]"),
            ("'say \"hi\"'", 'say "hi"'),
            ("'a\\b'", "a\\b"),
            ("date'2026-01-01'", date(2026, 1, 1)),
            ("'A' = 'A'", True),
            ("'A' = 'B'", False),
        ],
    )
    def test_no_expression_that_was_valid_before_means_anything_else_now(self, text: str, expected: Any) -> None:
        """Every literal form that compiled before still compiles to the same value.

        A double quote *inside* a literal was always ordinary text -- only one reaching the scan
        outside a literal is refused -- and a backslash was always taken verbatim, because a literal
        is lifted out whole rather than read for escape sequences. Both stay that way.
        """
        assert scalar(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            *DISALLOWED_NODE_SNIPPETS.values(),
            "__import__('os').system('id')",
            "().__class__.__bases__",
            "open('/etc/passwd')",
            "[code].upper()",
            '[code] = "A"',
            "[code] == 'A'",
            "os",
            "True",
            "b'abc'",
            "f'{code}'",
            "r'abc'",
            "[code] = 'a''' or __import__",
        ],
    )
    def test_every_attack_this_module_already_refused_is_still_refused(self, text: str) -> None:
        """The escape changes how a literal is scanned, so the whitelist is re-run in full against it.

        A rewrite of the one function that decides where a literal ends is exactly the change that
        could let an author-supplied character through into the text that gets parsed, which is the
        single property this module exists to keep.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text, columns=[])


# ---------------------------------------------------------------------------
# The two places author text is spelled back: a refusal, and the scanner
# ---------------------------------------------------------------------------

#: The attacks from the sweep above that exercise the *scanner* -- the one function that decides
#: where a literal ends -- re-run with the column they name actually declared. Compiled with
#: ``columns=[]``, as the sweep does, every one of these is refused for an unknown column before a
#: parser ever sees it, so the sweep says nothing about the whitelist for exactly the cases the
#: escape syntax changed. Declaring the column is what makes the refusal the language's.
ESCAPE_SENSITIVE_ATTACKS: tuple[str, ...] = (
    "[code] = 'a''' or __import__",
    "[code] = 'a''' or open('/etc/passwd')",
    "[code] = '''' or [code].upper()",
    "[code] = 'a''' and lambda: 1",
    "[code] = 'a''' or f'{code}'",
    '[code] = "A"',
    "[code] == 'A'",
    "[code].upper()",
)


class TestARefusalQuotesAValueBackInThisLanguage:
    """A refusal tells the author what to write, so the value it quotes has to be writable.

    The message is read by someone holding the file, and the obvious next move on reading it is to
    copy the quoted value back into the expression. Python's ``repr`` switches to double quotes the
    moment a value holds an apostrophe -- and a double quote is refused by this language by design,
    so a refusal spelled that way tells the author to write something the language will refuse. The
    spelling of a literal is this module's own, in a refusal as much as anywhere else.
    """

    def _head(self, text: str, columns: dict[str, pl.DataType]) -> str:
        """The refusal's own words, without the tail that quotes the whole line back.

        The tail repeats the author's text verbatim, apostrophes and all, so an assertion over the
        whole message would be satisfied by the echo rather than by the rendering under test.
        """
        with pytest.raises(ExpressionError) as raised:
            compile_expr(text, columns=columns)
        return str(raised.value).split(", in expression:")[0]

    @pytest.mark.parametrize(
        "text",
        [
            "[amount] + 'O''Brian'",
            "abs('O''Brian')",
            "[amount] > 'O''Brian'",
        ],
    )
    def test_a_value_with_an_apostrophe_is_quoted_back_the_way_this_language_writes_one(self, text: str) -> None:
        """The doubled form appears, and no double quote does."""
        head = self._head(text, {"amount": pl.Int64, "code": pl.String})

        assert "'O''Brian'" in head, f"the value must be quoted back as this language spells it: {head!r}"
        assert '"' not in head, f"a double quote is refused by this language and cannot appear in advice: {head!r}"

    def test_the_value_a_refusal_quotes_back_is_text_this_language_accepts(self) -> None:
        """The whole point, asserted rather than implied: what the message offers, the parser takes."""
        head = self._head("abs('O''Brian')", {"amount": pl.Int64})
        quoted = re.search(r"'(?:[^']|'')*'", head)

        assert quoted is not None, f"the refusal names no literal at all: {head!r}"
        assert scalar(f"{quoted.group(0)} = 'O''Brian'") is True


class TestTheScannerIsSweptWithItsColumnsDeclared:
    """The escape-sensitive attacks, refused by the language rather than by the column list.

    ``test_every_attack_this_module_already_refused_is_still_refused`` compiles with ``columns=[]``,
    and an unknown column is refused before the text is parsed. Every attack in that list that
    names ``[code]`` therefore stops at the column check, which means the case written *for* the
    doubled-quote scanner -- ``[code] = 'a''' or __import__`` -- has never reached the node
    whitelist at all. Declaring the column removes the earlier refusal, so what is left is the one
    the attack is named for.
    """

    @pytest.mark.parametrize("text", ESCAPE_SENSITIVE_ATTACKS)
    def test_an_escape_sensitive_attack_is_refused_by_the_language_and_not_by_the_schema(self, text: str) -> None:
        with pytest.raises(ExpressionError) as raised:
            compile_expr(text, columns={"code": pl.String})

        message = str(raised.value)
        assert "not a column in the data" not in message, (
            f"this refusal came from the column list, so the attack never reached the parser: {message!r}"
        )


# ---------------------------------------------------------------------------
# format_number: a number written as text, for a report
# ---------------------------------------------------------------------------
#
# A validation pack has a tab of invariants whose figures must read ``651,623.80`` -- grouped in
# threes, and exactly two decimals. Nothing already in this language produces that. Casting to
# String gives ``651623.8`` and loses the trailing zero, and regex_replace cannot rearrange what it
# matched, because its replacement is verbatim text by design (``$1`` is two characters, not a
# capture group, and that ruling is deliberate: the engine underneath reads ``$name`` as a group
# and deletes it when none exists, so a replacement of ``$5.00`` would land as ``.00``).
#
# So the capability arrives as a function, by this library's own dividing line -- a tool reshapes
# the frame, an expression function computes one column -- and the shape is Alteryx's, which has
# exactly this call:
#
#     ToString(x, numDec, addThousandsSeparator, decimalSeparator)
#
# Their fourth argument carries the locale: the thousands separator is a comma unless the decimal
# separator is a comma, in which case it is a period. This language has no optional arguments at
# all, so the same four ideas arrive as four required arguments and the caller states both
# separators outright. The library then makes no locale decision of its own -- the same line
# normalize_text draws by taking its suffix list as configuration rather than assuming one.

#: Attacks written through the new call, so the whitelist is asked about it rather than around it.
#: Every column named here is declared when these are swept, so a refusal cannot come from the
#: column list -- it has to come from the language.
FORMATTING_ATTACKS: tuple[str, ...] = (
    "format_number([amount], 2, ',', '.').upper()",
    "format_number([amount].__class__, 2, ',', '.')",
    # Subscripted with a column that really exists, so the refusal cannot be the column list's:
    # '[0]' would be read as a column reference by this language's own syntax and stop there.
    "format_number([amount], 2, ',', '.')[code]",
    "format_number(__import__('os'), 2, ',', '.')",
    "format_number([amount], 2, ',', f'{code}')",
    "format_number([amount], 2, ',', '.') = 'a''' or __import__",
    "format_number([amount], 2, ',', lambda: 1)",
)


class TestFormatNumberWritesANumberAsText:
    """One number in, one piece of text out, with both separators stated by whoever wrote the line.

    The rulings this class pins, in the order they bite:

    * rounding follows the *double the column holds*, not the decimal somebody typed into the file;
    * exactly ``decimals`` places, always, so a trailing zero survives and ``0.0`` reads ``0.00``;
    * grouping is in threes from the right and touches the integer part only;
    * a negative number keeps its sign in front of the digits and does not confuse the grouping;
    * either separator may be empty, and either may be more than one character;
    * null in, null out, the same as everything else here;
    * ``decimals`` is a whole number that is not negative, and both separators are quoted text, all
      three refused in the file rather than at row time;
    * the value has to be a number, and the refusal names what the column actually holds;
    * the result is text, so it composes with every text function already here;
    * a negative that rounds away to zero still shows its sign;
    * a column holding an exact decimal and one holding a double disagree in the last place, and
      both are written faithfully to what they hold;
    * and the figures do not move when the process-wide decimal context does.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # The figure the tab of invariants has to show, and the whole reason this exists.
            ("format_number(651623.8, 2, ',', '.')", "651,623.80"),
            # An empty thousands separator means no grouping at all.
            ("format_number(651623.8, 2, '', '.')", "651623.80"),
            # Both separators are the caller's, so the other convention is spelled rather than
            # inferred from one of them. This is Alteryx's own worked example, stated outright.
            ("format_number(123456.789, 3, '.', ',')", "123.456,789"),
            # Exactly `decimals` places, always: a trailing zero is not dropped and a zero is not
            # written as '0'. This is what casting to String loses.
            ("format_number(0.0, 2, ',', '.')", "0.00"),
            ("format_number(123.4, 2, ',', '.')", "123.40"),
            ("format_number(1000, 4, ',', '.')", "1,000.0000"),
            # ... and 0 decimals is no decimal point either, rather than a bare one.
            ("format_number(5, 0, ',', '.')", "5"),
            ("format_number(1234567, 0, ',', '.')", "1,234,567"),
            # Grouping is in threes from the right, so the leading group is 1, 2 or 3 digits and
            # the fractional part is never grouped however long it is.
            ("format_number(1234.5, 2, ',', '.')", "1,234.50"),
            ("format_number(12345.5, 2, ',', '.')", "12,345.50"),
            ("format_number(123456.5, 2, ',', '.')", "123,456.50"),
            ("format_number(999.5, 2, ',', '.')", "999.50"),
            ("format_number(1.5, 8, ',', '.')", "1.50000000"),
        ],
    )
    def test_a_number_is_written_with_the_separators_the_line_states(self, text: str, expected: str) -> None:
        """The shape of the result, asserted as the text a reader of the report would see."""
        assert scalar(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("format_number(-1832.015, 2, ',', '.')", "-1,832.02"),
            ("format_number(-651623.8, 2, ',', '.')", "-651,623.80"),
            ("format_number(-1234567, 0, ',', '.')", "-1,234,567"),
            ("format_number(-100.0, 2, ',', '.')", "-100.00"),
            # The sign is not one of the digits being grouped, so it never lands inside a group.
            ("format_number(-1000.0, 2, ',', '.')", "-1,000.00"),
        ],
    )
    def test_a_negative_number_keeps_its_sign_in_front_of_the_grouped_digits(self, text: str, expected: str) -> None:
        """The minus goes before the digits, and the grouping is counted without it.

        Worth its own case because the obvious implementation groups the rendered string from the
        right and would produce ``-,1832.02`` on a four-digit negative -- a figure that still looks
        like a number in a report and is read by nobody as wrong.
        """
        assert scalar(text) == expected

    def test_rounding_follows_the_double_the_column_holds_not_the_decimal_that_was_typed(self) -> None:
        """``-15.215`` at two places is ``-15.21``, because that is what the double actually is.

        The two candidate rulings disagree, and they disagree on money. ``-15.215`` cannot be held
        by a double: the value in the column is ``-15.2149999999999998578...``, so rounding *that*
        to two places gives ``-15.21``. Rounding the decimal somebody typed gives ``-15.22``.

        The first is chosen, because the figure in the report has to be a faithful rendering of the
        value the pipeline computed -- a formatter that rounds the *literal* is inventing a value
        that was never in the frame, and it would disagree with every other reading of that column.
        The cost is stated rather than hidden: this language's ``round()`` takes the other view, so
        ``round([x], 2)`` and ``format_number([x], 2, ...)`` can differ in the last place, and the
        second assertion here exists so that nobody discovers that in a report.

        ``2.675`` and ``1.005`` are the two everybody meets first -- both are below their apparent
        half, so both round down.
        """
        assert scalar("format_number(-15.215, 2, ',', '.')") == "-15.21"
        assert scalar("round(-15.215, 2)") == -15.22

        assert scalar("format_number(2.675, 2, ',', '.')") == "2.67"
        assert scalar("format_number(1.005, 2, ',', '.')") == "1.00"
        # A tie the double really does hold ties to even, the way the engine and Python both do.
        assert scalar("format_number(1.125, 2, ',', '.')") == "1.12"
        assert scalar("format_number(1.135, 2, ',', '.')") == "1.14"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # A carry out of the last place has to travel through the grouping, not stop at it.
            ("format_number(999999.999, 2, ',', '.')", "1,000,000.00"),
            ("format_number(999.999, 2, ',', '.')", "1,000.00"),
            ("format_number(9.999, 2, ',', '.')", "10.00"),
            ("format_number(-999999.999, 2, ',', '.')", "-1,000,000.00"),
            # ... and 999999.995 is the one that looks like it should carry and does not, because
            # the double is 999999.99499999999534..., just under the half.
            ("format_number(999999.995, 2, ',', '.')", "999,999.99"),
            # Past 2^53 the spacing between doubles is wider than a cent, so the digits written are
            # the ones the double holds and the decimals are zeros. Written out rather than left to
            # be found in a report.
            ("format_number(1e16, 2, ',', '.')", "10,000,000,000,000,000.00"),
            ("format_number(12345678901234.567, 2, ',', '.')", "12,345,678,901,234.57"),
            # And very small: every digit asked for appears, including the ones that are zero.
            ("format_number(0.004, 2, ',', '.')", "0.00"),
            ("format_number(0.0000001, 3, ',', '.')", "0.000"),
        ],
    )
    def test_a_carry_and_a_value_past_the_precision_of_a_double_are_both_written_out(
        self, text: str, expected: str
    ) -> None:
        """The boundaries, pinned here rather than discovered in a validation pack."""
        assert scalar(text) == expected

    def test_the_same_digits_read_one_way_in_a_whole_number_and_another_in_a_decimal(self) -> None:
        """Past 2^53 the two dtypes hold different values, so they are written differently.

        "Faithful to the value" is the one rule, and these are two readings of it rather than two
        rules. A whole-number column holds 9007199254740993 exactly, so that is what it reads. A
        decimal column cannot: the nearest double is ...992, so ...992 is what it holds and what it
        reads. The literal forms are pinned beside the columns because the language types a bare
        9007199254740993 as a whole number and 9007199254740993.0 as a decimal, so the same digits
        in a file mean two different values before this function is reached at all.

        This is the general form of the rounding ruling above: -15.215 reads -15.21 for exactly the
        reason ...993.0 reads ...992 -- in both cases the decimal somebody typed is not the value
        the column ended up holding, and the column's value is the one written out.
        """
        whole = pl.DataFrame({"paid": pl.Series([9007199254740993], dtype=pl.Int64)})
        assert checked(whole, "format_number([paid], 2, ',', '.')") == ["9,007,199,254,740,993.00"]

        decimal = pl.DataFrame({"paid": pl.Series([9007199254740993.0], dtype=pl.Float64)})
        assert checked(decimal, "format_number([paid], 2, ',', '.')") == ["9,007,199,254,740,992.00"]

        assert scalar("format_number(9007199254740993, 2, ',', '.')") == "9,007,199,254,740,993.00"
        assert scalar("format_number(9007199254740993.0, 2, ',', '.')") == "9,007,199,254,740,992.00"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Nothing makes one character special, so a separator of any length is the caller's.
            ("format_number(1234567.891, 2, ' ', ',')", "1 234 567,89"),
            ("format_number(1234567.891, 2, ' -- ', '.')", "1 -- 234 -- 567.89"),
            ("format_number(1234567.891, 2, '', '.')", "1234567.89"),
            ("format_number(1234.5, 2, ',', ' point ')", "1,234 point 50"),
            # A doubled quote is how this language writes one, in this argument like any other.
            ("format_number(1234.5, 2, '''', '.')", "1'234.50"),
        ],
    )
    def test_either_separator_may_be_empty_or_longer_than_one_character(self, text: str, expected: str) -> None:
        """Both separators are text of the caller's choosing, and neither is inferred from the other."""
        assert scalar(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("format_number([amount], 2, ',', '.')", ["10.26", "-2.40", "100.00", None]),
            ("format_number([qty], 0, ',', '.')", ["1", "2", "3", None]),
            # A null part makes the whole value null, and the escape hatch is the one this language
            # already teaches for null everywhere else.
            ("format_number([amount], 2, ',', '.') is null", [False, False, False, True]),
            ("coalesce(format_number([amount], 2, ',', '.'), 'n/a')", ["10.26", "-2.40", "100.00", "n/a"]),
            ("format_number(coalesce([amount], 0.0), 2, ',', '.')", ["10.26", "-2.40", "100.00", "0.00"]),
        ],
    )
    def test_a_null_number_is_written_as_null_and_not_as_a_row_of_zeros(
        self, frame: pl.DataFrame, text: str, expected: list[Any]
    ) -> None:
        """Null in, null out -- the library's usual reading, and the safer one on a report.

        ``0.00`` where there was no figure at all is a measurement nobody made, and a reader adding
        a column of them cannot tell it from a real zero. A null shows up as a blank cell, and the
        author who wants the zero writes it.
        """
        assert checked(frame, text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("concat(format_number(651623.8, 2, ',', '.'), ' total')", "651,623.80 total"),
            ("concat('$', format_number(651623.8, 2, ',', '.'))", "$651,623.80"),
            ("len(format_number(651623.8, 2, ',', '.'))", 10),
            ("upper(format_number(651623.8, 2, ',', 'e'))", "651,623E80"),
            ("format_number(651623.8, 2, ',', '.') = '651,623.80'", True),
            ("starts_with(format_number(651623.8, 2, ',', '.'), '651,')", True),
            ("format_number(651623.8, 2, ',', '.') in ('651,623.80', '0.00')", True),
            # It takes a number, so it composes the other way round too.
            ("format_number(round(651623.789, 1), 2, ',', '.')", "651,623.80"),
            ("format_number(abs(-651623.8), 2, ',', '.')", "651,623.80"),
        ],
    )
    def test_the_result_is_text_for_the_purpose_of_nesting(self, text: str, expected: Any) -> None:
        """A function's result family is checked rather than assumed, so this has to be declared text.

        Without that, ``concat(format_number(...), ' total')`` is refused for a part that is not
        text while behaving perfectly well in the engine -- and ``upper(format_number(...))`` is the
        reverse mistake, compiling and failing in a select. Composition with ``concat`` is the one
        that matters: a report line is a figure with words around it.
        """
        assert scalar(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            # Not whole -- half a decimal place is not a width, and it arrives from a division
            # rather than from anybody typing it.
            "format_number([amount], 2.5, ',', '.')",
            "format_number([amount], 0.5, ',', '.')",
            # Not a count at all.
            "format_number([amount], -1, ',', '.')",
            "format_number([amount], -2, ',', '.')",
            "format_number([amount], true, ',', '.')",
            "format_number([amount], null, ',', '.')",
            "format_number([amount], '2', ',', '.')",
            # A width that varies row by row is not a width: how many places a figure is written to
            # is a decision recorded in the file, the way round()'s and regex_extract()'s are.
            "format_number([amount], [qty], ',', '.')",
            "format_number([amount], [qty] + 1, ',', '.')",
        ],
    )
    def test_the_number_of_decimal_places_is_a_whole_number_that_is_not_negative(
        self, frame: pl.DataFrame, text: str
    ) -> None:
        """Refused against the file, before a row is read, and the refusal names the function.

        The same stance ``add_months`` takes on a count of months: a value that cannot be shown to
        be whole is refused rather than rounded, because rounding it here buries the division that
        produced it inside a figure that looks perfectly ordinary.
        """
        reported = schema_message(frame, text)
        assert "format_number" in reported, f"the refusal does not name the function: {reported!r}"
        assert "second" in reported, f"the refusal does not say which argument is wrong: {reported!r}"

    @pytest.mark.parametrize(
        ("text", "position"),
        [
            ("format_number([amount], 2, [code], '.')", "third"),
            ("format_number([amount], 2, 1, '.')", "third"),
            ("format_number([amount], 2, true, '.')", "third"),
            ("format_number([amount], 2, ',', [code])", "fourth"),
            ("format_number([amount], 2, ',', 1)", "fourth"),
            ("format_number([amount], 2, ',', null)", "fourth"),
        ],
    )
    def test_both_separators_are_quoted_text_written_in_the_file(
        self, frame: pl.DataFrame, text: str, position: str
    ) -> None:
        """A separator that came out of the data would let the data decide how the data is spelled.

        The same rule ``regex_replace`` applies to its replacement, and for the same reason: these
        are decisions a reviewer reads in the file, not values a row supplies.
        """
        reported = schema_message(frame, text)
        assert "format_number" in reported, f"the refusal does not name the function: {reported!r}"
        assert position in reported, f"the refusal does not say which argument is wrong: {reported!r}"

    @pytest.mark.parametrize("column", ["code", "label", "note", "flag", "start_date", "category", "grade"])
    def test_the_value_has_to_be_a_number_and_the_refusal_names_what_the_column_holds(
        self, frame: pl.DataFrame, column: str
    ) -> None:
        """Text, a truth value and a date are refused here, in the author's words, naming the dtype.

        Text is the one that matters. An engine will happily write a string column out with a
        separator glued on, so an amount that arrived from an import as text would be "formatted"
        into something that is not a number and never was -- the same defect that made ``+`` on two
        text columns worth refusing.
        """
        reported = schema_message(frame, f"format_number([{column}], 2, ',', '.')")
        assert reported.count(column) >= 2, f"the problem itself does not name the column in {reported!r}"
        assert str(frame.schema[column]) in reported, f"the actual dtype is missing from {reported!r}"
        assert "number" in reported.lower(), f"the refusal does not say a number was needed: {reported!r}"

    @pytest.mark.parametrize(
        "text",
        [
            "format_number([amount])",
            "format_number([amount], 2)",
            "format_number([amount], 2, ',')",
            "format_number([amount], 2, ',', '.', 'x')",
        ],
    )
    def test_format_number_takes_exactly_four_arguments(self, frame: pl.DataFrame, text: str) -> None:
        """Four required arguments and no defaults, like every other call in this language.

        Alteryx spells this with two optional arguments, which means two spellings of one call and
        a reviewer who has to know a default to know what a line does. Arity is fixed here instead,
        so the caller states the width and both separators every time.
        """
        reported = schema_message(frame, text)
        assert "format_number" in reported, f"the refusal does not name the function: {reported!r}"
        assert "4" in reported, f"the refusal does not say how many arguments are wanted: {reported!r}"

    def test_format_number_is_offered_by_name_to_an_author_who_reached_for_something_else(self) -> None:
        """A function that is not on the list is refused with the list, so the new name is findable."""
        assert "format_number" in ALLOWED_FUNCTIONS
        assert "format_number" in message("to_string([amount], 2)")

    @pytest.mark.parametrize(
        "text",
        [
            *DISALLOWED_NODE_SNIPPETS.values(),
            *ESCAPE_SENSITIVE_ATTACKS,
            *FORMATTING_ATTACKS,
            "__import__('os').system('id')",
            "().__class__.__bases__",
            "open('/etc/passwd')",
            "[code].upper()",
            '[code] = "A"',
            "[code] == 'A'",
            "os",
            "True",
            "b'abc'",
            "f'{code}'",
            "r'abc'",
            "[code] = 'a''' or __import__",
        ],
    )
    def test_the_attack_corpus_is_still_refused_with_every_column_declared(
        self, frame: pl.DataFrame, text: str
    ) -> None:
        """The whole corpus re-run against a real schema, because a new call is a new way in.

        ``format_number`` reads three of its four arguments as literals out of the file, which is
        the shape that has to be checked rather than assumed: a function that pulled a separator or
        a width out of author text by any route other than the scan would be the one place a
        character could reach the parser. Declaring the columns is what makes these refusals the
        language's -- compiled against no schema, every attack naming ``[code]`` or ``[amount]``
        stops at the column check and says nothing about the whitelist at all.
        """
        with pytest.raises(ExpressionError):
            compile_expr(text, columns=frame.schema)

    @pytest.mark.parametrize("text", FORMATTING_ATTACKS)
    def test_an_attack_written_through_the_new_call_is_refused_by_the_language(
        self, frame: pl.DataFrame, text: str
    ) -> None:
        """... and refused by the whitelist rather than by the column list or by the engine."""
        with pytest.raises(ExpressionError) as raised:
            compile_expr(text, columns=frame.schema)

        reported = str(raised.value)
        assert "not a column in the data" not in reported, (
            f"this refusal came from the column list, so the attack never reached the parser: {reported!r}"
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Every digit asked for is below the half, so they all come out zero -- and the sign in
            # front of them is then the only thing left saying which side of zero the figure was on.
            ("format_number(-0.004, 2, ',', '.')", "-0.00"),
            ("format_number(-0.5, 0, ',', '.')", "-0"),
            ("format_number(-0.5, 2, ',', '.')", "-0.50"),
            # ... and a value that *is* zero has no side to be on, so it is written without a sign.
            ("format_number(-0.0, 2, ',', '.')", "0.00"),
        ],
    )
    def test_a_negative_that_rounds_away_to_zero_still_shows_its_sign(self, text: str, expected: str) -> None:
        """``-0.004`` at two places reads ``-0.00``, not ``0.00``.

        Python writes the same figure the same way -- ``f"{-0.004:.2f}"`` is ``-0.00`` -- and the
        agreement is the point rather than a coincidence worth matching. A column of small
        negatives written as ``0.00`` tells a reader the movement was neither up nor down, when it
        was down; the sign is the last thing the rounding has not eaten, so it stays.

        The other negatives pinned in this class are all of magnitude 100 or more, where the digits
        carry the sign's information anyway. Half is the boundary: it disappears at one width and
        survives at the next, and both readings are stated here.
        """
        assert scalar(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("format_number(1234.5, 2, ',', '')", "1,23450"),
            ("format_number(1234.5, 2, '', '')", "123450"),
            # Nothing is written after the whole number when no places were asked for, so an empty
            # decimal separator changes nothing and no separator is left dangling at the end.
            ("format_number(1234.5, 0, ',', '')", "1,234"),
        ],
    )
    def test_an_empty_decimal_separator_runs_the_places_onto_the_whole_number(self, text: str, expected: str) -> None:
        """Empty means empty in the fourth argument, the way it already does in the third.

        No default is supplied for either separator, and that is the whole shape of this call: the
        caller states both, so the library makes no locale decision of its own. A period quietly
        put back when the caller wrote ``''`` would be exactly such a decision, and it would be
        made on the one argument where the caller had said outright what they wanted.

        ``1,23450`` is a figure written in minor units with the grouping kept -- an odd-looking
        line, and the caller's to write.
        """
        assert scalar(text) == expected

    def test_a_column_of_no_type_at_all_is_written_as_null(self) -> None:
        """A ``Null`` column has no number in it to write, so every row is null and nothing raises.

        This shape has been a real defect in this module once already: negating a ``Null`` column
        raises in the engine rather than answering, which is what ``_FrameTarget.negative``'s
        special case and ``test_a_negated_null_count_behaves_like_a_null_count`` exist for. A
        column arrives typed this way without anybody choosing it -- an all-empty column read from
        a file, or one built from a literal null -- so the row-at-a-time writer here is asked the
        same question the negation was asked, and has to answer it the same way.
        """
        empty = pl.DataFrame({"empty": pl.Series([None, None], dtype=pl.Null)})

        assert checked(empty, "format_number([empty], 2, ',', '.')") == [None, None]
        assert scalar("format_number(null, 2, ',', '.')") is None

    def test_a_decimal_column_is_written_from_the_value_it_really_holds(self) -> None:
        """``-15.215`` held as a decimal reads ``-15.22``; held as a double it reads ``-15.21``.

        The two answers disagree, and the disagreement is the ruling rather than a bug in it. One
        rule is in force in both -- write the value the column holds -- and the columns hold
        different values. A ``Decimal`` column holds ``-15.215`` exactly, which is a true tie and
        ties to even, giving ``-15.22``. A ``Float64`` column holds ``-15.2149999999999998578...``,
        which is below the half, giving ``-15.21``.

        Worth stating side by side because ``Decimal`` is the dtype a database read produces for a
        money column -- ``write_excel`` and ``write_workbook`` both carry explicit refusals for it,
        which is evidence such columns really arrive -- and an author who reads the rounding ruling
        above and then formats one would otherwise expect the other answer.
        """
        held = pl.DataFrame({"paid": pl.Series([Decimal("-15.215"), Decimal("1234.5"), None], dtype=pl.Decimal(18, 3))})
        assert checked(held, "format_number([paid], 2, ',', '.')") == ["-15.22", "1,234.50", None]

        as_double = held.with_columns(pl.col("paid").cast(pl.Float64))
        assert checked(as_double, "format_number([paid], 2, ',', '.')") == ["-15.21", "1,234.50", None]

    def test_the_figures_do_not_move_when_the_process_wide_decimal_context_changes(self) -> None:
        """The same four answers inside a context of 5 digits rounding half up as outside it.

        Several ``Decimal`` operations are *context* operations: they round their result to
        ``context.prec`` digits using ``context.rounding``, where the context is a process-wide
        setting any library in the process may change. A formatter that reaches for one of them is
        carrying a dependency on a global it does not own, and the failure it produces is silent --
        ``651,623.80`` becomes ``651,620.00`` in a report that still looks perfectly ordinary.

        Nothing in this repository sets the context today, so this pins a property rather than
        catching a live fault: the figures a report shows must not be movable from outside the
        line that wrote them. ``prec = 5`` is far enough below the digits here to make any such
        dependency visible, and ``ROUND_HALF_UP`` disagrees with this function's ruling on the tie,
        so the third assertion tells the two sources of rounding apart.
        """
        with localcontext() as context:
            context.prec = 5
            context.rounding = ROUND_HALF_UP

            assert scalar("format_number(651623.8, 2, ',', '.')") == "651,623.80"
            assert scalar("format_number(-1832.015, 2, ',', '.')") == "-1,832.02"
            assert scalar("format_number(-15.215, 2, ',', '.')") == "-15.21"
            assert scalar("format_number(9007199254740993, 2, ',', '.')") == "9,007,199,254,740,993.00"

    def test_a_separator_and_a_rewrite_are_each_refused_in_their_own_words(self, frame: pl.DataFrame) -> None:
        """Two arguments refused by one rule, and each told the version of it that fits.

        A separator and a replacement are both text that has to be a decision in the file, but an
        author reading the refusal is holding one of them, not the rule. The separator's sentence
        is about how a figure is spelled; the rewrite's is about what the data becomes. Without an
        assertion on the wording the two can be swapped, or collapsed back into one, and the suite
        would not notice -- so each is pinned by a fragment the other does not contain.
        """
        third = schema_message(frame, "format_number([amount], 2, [code], '.')")
        fourth = schema_message(frame, "format_number([amount], 2, ',', [code])")
        rewrite = schema_message(frame, "regex_replace([code], 'a', [code])")

        for reported in (third, fourth):
            assert "a separator is a decision written in the file" in reported, (
                f"the separator refusal does not say why a separator belongs in the file: {reported!r}"
            )
            assert "what the data becomes" not in reported, (
                f"the separator refusal is wearing regex_replace()'s sentence: {reported!r}"
            )

        assert "a rewrite whose replacement came out of the data" in rewrite, (
            f"the rewrite refusal does not say why a replacement belongs in the file: {rewrite!r}"
        )
        assert "separator" not in rewrite, f"the rewrite refusal is wearing the separator's sentence: {rewrite!r}"


class TestAFigureThatIsNotANumberHasNoWrittenForm:
    """``inf``, ``-inf`` and ``nan`` are written as null -- never as a word, and never as a raise.

    These three are ordinary values here, not exotic ones. ``/`` is true division on doubles in
    this language, so ``[total] / [count]`` is ``inf`` on the row where the count is zero and
    ``0.0 / 0.0`` is ``nan``; and formatting a ratio at the end of a flow is the stated use case
    for this function. So the last step before a report is exactly where they arrive.

    The ruling is the one this library takes everywhere else: a figure that cannot be stated is
    stated as absent, never as a plausible-looking wrong one. That is the same answer a null
    already gets, and for the same reason -- the reader of the cell has to be able to tell "no
    figure" from a figure.

    The two things it rules out are what the code does today. Grouping ``Infinity`` in threes from
    the right puts ``In,fin,ity`` in a report cell, which reads as a spelling accident rather than
    as a division by zero. And ``Decimal('NaN') < 0`` signals ``InvalidOperation`` instead of
    answering False the way the float comparison would, so the sign test at the end raises a
    ``decimal`` exception out of a select -- precisely the thing this module's docstring promises
    cannot happen: a mistake has to fail in the author's words, and never later in the engine's.
    """

    def test_a_column_holding_infinity_or_not_a_number_is_written_as_null(self) -> None:
        """All three are null, the real value beside them is written, and nothing raises.

        The fourth row is the control: an answer of all nulls would satisfy the first assertion on
        its own, and would mean the function had stopped writing figures at all.
        """
        ratios = pl.DataFrame({"ratio": pl.Series([math.inf, -math.inf, math.nan, 2.0], dtype=pl.Float64)})

        assert checked(ratios, "format_number([ratio], 2, ',', '.')") == [None, None, None, "2.00"]
        assert checked(ratios, "format_number([ratio], 2, ',', '.') is null") == [True, True, True, False]
        assert checked(ratios, "coalesce(format_number([ratio], 2, ',', '.'), 'n/a')") == [
            "n/a",
            "n/a",
            "n/a",
            "2.00",
        ]

    def test_a_division_that_leaves_the_numbers_behind_is_written_as_null(self) -> None:
        """The route an author actually reaches this by: a division, written in the line itself.

        The first assertion of each pair is the route rather than the ruling -- it states what this
        language's ``/`` gives, so that the pins below are read as being about a real value a flow
        produces and not about a literal nobody can type.
        """
        assert scalar("1.0 / 0.0") == math.inf
        assert scalar("format_number(1.0 / 0.0, 2, ',', '.')") is None

        assert scalar("-1.0 / 0.0") == -math.inf
        assert scalar("format_number(-1.0 / 0.0, 2, ',', '.')") is None

        assert math.isnan(scalar("0.0 / 0.0"))
        assert scalar("format_number(0.0 / 0.0, 2, ',', '.')") is None


# ---------------------------------------------------------------------------
# A stable surrogate key: one fixed-width id derived from several columns
# ---------------------------------------------------------------------------
#
# A published table's natural key is often three or four columns of text, which is a poor thing to
# join on and a worse thing to quote in a ticket. ``uuid5`` turns that key into one 36-character
# column, and the whole value of it is that the derivation is *not* this library's: it is RFC 4122
# version 5, so the id a flow writes is the id anything else computing it the same way writes.
# Every assertion below is therefore made against ``uuid.uuid5`` computed in the test, and the one
# hardcoded id is a cross-check rather than the contract.

# The namespaces these tests use. The first is RFC 4122's own DNS namespace, which brings a
# published test vector with it; the second is arbitrary, so that no assertion here can pass by
# knowing only the well-known constant.
NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
OTHER_NAMESPACE = "1b4e28ba-2fa1-11d2-883f-0016d3cca427"

# The canonical written form: lowercase hex, hyphenated 8-4-4-4-12, the version nibble a '5' and
# the variant nibble one of 8/9/a/b. Matching this is what makes the id safe to put in a text
# column and compare as text.
CANONICAL_UUID: re.Pattern[str] = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")


def ids_for(namespace: str, values: list[str | None]) -> list[str | None]:
    """What the standard library makes of each value in ``namespace`` -- and null for a null.

    The expectation is computed rather than written down, because the contract is equality with
    ``uuid.uuid5`` and not equality with a string somebody once pasted into a test.
    """
    space = uuid.UUID(namespace)
    return [None if value is None else str(uuid.uuid5(space, value)) for value in values]


class TestUuid5:
    """``uuid5(namespace, value)`` derives a stable id, and agrees with the standard library.

    The namespace is a literal written in the file and the value is any text expression, which is
    the split that matters: a namespace is a decision a reviewer reads in the line -- it is what
    keeps two different flows' ids from colliding -- while the value is the row's own key, built
    out of columns with ``concat``.
    """

    @pytest.mark.parametrize("namespace", [NAMESPACE, OTHER_NAMESPACE])
    @pytest.mark.parametrize("column", ["code", "label", "note"])
    def test_every_row_gets_the_id_the_standard_library_makes_of_it(
        self, frame: pl.DataFrame, namespace: str, column: str
    ) -> None:
        """Equality with ``uuid.uuid5``, row for row, in two namespaces and over three columns.

        This is the binding contract and the reason the function is worth having: an id that is
        only stable within this library is a key nobody outside it can reproduce. ``note`` is in
        the sweep because it holds text that looks like expression syntax (``a=b``, ``x [y]``,
        ``and``), so the value hashed is pinned as the row's data rather than anything the scanner
        might have touched on the way past.
        """
        values = frame[column].to_list()
        text = f"uuid5('{namespace}', [{column}])"
        assert checked(frame, text) == ids_for(namespace, values)

    def test_the_id_is_text_in_the_canonical_lowercase_hyphenated_form(self, frame: pl.DataFrame) -> None:
        """A plain string column, 36 characters, hyphenated, lowercase, version 5.

        The written form is as much of the contract as the bytes are: an id that arrived as
        uppercase, or braced, or without hyphens, would compare unequal to the same id written by
        anything else -- and the comparison is the only thing anyone ever does with it. The last
        pair of assertions is the published RFC 4122 vector, kept as a cross-check beside the
        computed expectation rather than instead of it.
        """
        built = frame.select(compile_expr(f"uuid5('{NAMESPACE}', [code])", columns=frame.schema).alias("out"))["out"]
        assert built.dtype == pl.String

        for value in built.to_list()[:3]:
            assert CANONICAL_UUID.fullmatch(value) is not None, f"{value!r} is not canonically written"

        # It is text, so it nests into the text functions like any other text.
        assert checked(frame, f"len(uuid5('{NAMESPACE}', [code]))") == [36, 36, 36, None]
        assert checked(frame, f"uuid5('{NAMESPACE}', [code]) = '9759785f-95f6-554a-ba7a-c3e586ec5609'") == [
            True,
            False,
            False,
            None,
        ]

        vector = scalar(f"uuid5('{NAMESPACE}', 'www.example.org')")
        assert vector == str(uuid.uuid5(uuid.UUID(NAMESPACE), "www.example.org"))
        assert vector == "74738ff5-5367-5958-9aee-98fffdcd1876"

    def test_a_null_value_gives_a_null_id_and_never_an_id_for_nothing(self, frame: pl.DataFrame) -> None:
        """Null in, null out -- and specifically not the id of ``'null'``, ``''`` or anything else.

        This is the defect this function could most plausibly ship with, and the one that would be
        invisible: a surrogate key that turned a missing value into a real-looking id would put
        every row with a missing part under the *same* id, which reads downstream as a legitimate
        key with several rows behind it rather than as data that was never there. A null key
        matches nothing and shows up as a null, which is the rule the rest of this language
        already follows -- and the author who wants the other behaviour writes the ``coalesce``
        themselves, exactly as they do to make a null row participate in a comparison.
        """
        space = uuid.UUID(NAMESPACE)
        ids = checked(frame, f"uuid5('{NAMESPACE}', [code])")

        assert ids == ids_for(NAMESPACE, frame["code"].to_list())
        assert ids[-1] is None
        assert checked(frame, f"uuid5('{NAMESPACE}', [code]) is null") == [False, False, False, True]

        # No stand-in for a missing value was hashed anywhere in the column.
        for stand_in in ("null", "NULL", "None", "", "nan"):
            assert str(uuid.uuid5(space, stand_in)) not in ids, f"a missing value was hashed as {stand_in!r}"

        # A null part of a built key makes the whole key null, the way concat() already does.
        assert checked(frame, f"uuid5('{NAMESPACE}', concat([code], '|', [label]))")[-1] is None

        # ... and the escape hatch is the one this language already teaches.
        assert checked(frame, f"uuid5('{NAMESPACE}', coalesce([code], 'missing'))") == ids_for(
            NAMESPACE, ["A", "B", "C", "missing"]
        )

    def test_a_key_built_from_several_columns_is_the_id_of_the_joined_text(self, frame: pl.DataFrame) -> None:
        """The stated use: ``concat`` the natural key, hash it once, carry one column.

        The value argument is an expression and not a column, so the key a flow derives is written
        in the file where a reviewer can read which columns went into it and in what order.
        """
        text = f"uuid5('{NAMESPACE}', concat([code], '|', [label]))"
        joined = [
            None if code is None or label is None else f"{code}|{label}"
            for code, label in zip(frame["code"].to_list(), frame["label"].to_list(), strict=True)
        ]
        assert checked(frame, text) == ids_for(NAMESPACE, joined)

        # And it composes both ways, as every text function here does.
        assert checked(frame, f"upper({text})") == [
            None if one is None else one.upper() for one in ids_for(NAMESPACE, joined)
        ]
        assert checked(frame, f"uuid5('{NAMESPACE}', concat(upper([code]), '|', strip([label])))") == ids_for(
            NAMESPACE, ["A|alpha", "B|Beta", "C|gamma", None]
        )

    @pytest.mark.parametrize(
        "text",
        [
            # A column: the namespace would then be chosen by the data it is namespacing.
            "uuid5([code], [label])",
            "uuid5([note], [label])",
            # Computed, even out of literals: a namespace has to be readable as one line in a file.
            "uuid5(concat('6ba7b810-9dad-11d1-80b4', '-00c04fd430c8'), [label])",
            "uuid5(upper([code]), [label])",
            f"uuid5(coalesce([code], '{NAMESPACE}'), [label])",
            # And a literal that is not text at all.
            "uuid5(1, [label])",
            "uuid5(null, [label])",
            "uuid5(true, [label])",
            "uuid5(date'2026-01-01', [label])",
        ],
    )
    def test_the_namespace_must_be_quoted_text_written_in_the_file(self, text: str) -> None:
        """A namespace is a fixed decision recorded in the line, like a pattern or a separator.

        Two reasons, and the second is the one with teeth. It cannot be checked before the run
        otherwise -- a namespace that varied row by row could be malformed on row nine hundred,
        inside the engine, having already written nine hundred ids. And a namespace is what keeps
        one flow's ids from colliding with another's, so letting the data choose it means the data
        decides which flow's key space its rows land in.
        """
        reported = message(text)
        assert "uuid5" in reported, f"the refusal does not name the function: {reported!r}"
        assert "namespace" in reported.lower(), f"the refusal does not name the argument: {reported!r}"

    @pytest.mark.parametrize(
        "namespace",
        [
            "",
            "not-a-uuid",
            "6ba7b810-9dad-11d1-80b4",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8f",
            "zba7b810-9dad-11d1-80b4-00c04fd430c8",
            "6ba7b810_9dad_11d1_80b4_00c04fd430c8",
        ],
    )
    def test_a_namespace_that_is_not_a_uuid_is_refused_before_a_row_is_read(self, namespace: str) -> None:
        """Checked at compile time, in the author's words, and never inside a select.

        Each case is asserted to be one the standard library refuses too, so the test is about a
        malformed namespace and not about an opinion on spelling. Catching it here is the whole
        promise of this module: a mistake fails in the line the author wrote, not two hours into a
        load with a partial table behind it.
        """
        with pytest.raises(ValueError):
            uuid.UUID(namespace)

        reported = message(f"uuid5('{namespace}', [code])")
        assert "uuid5" in reported, f"the refusal does not name the function: {reported!r}"
        assert "namespace" in reported.lower(), f"the refusal does not name the argument: {reported!r}"

    def test_the_same_input_gives_the_same_id_every_time_it_is_evaluated(self, frame: pl.DataFrame) -> None:
        """Separate compilations, a different row order and a different frame all agree.

        A surrogate key is only a key if it is reproducible: a run that re-derives yesterday's id
        differently does not update a row, it inserts a second one.
        """
        text = f"uuid5('{NAMESPACE}', [code])"
        first = checked(frame, text)

        assert checked(frame, text) == first
        assert checked(frame.reverse(), text) == list(reversed(first))
        assert checked(pl.DataFrame({"code": ["C", "A"]}), text) == [first[2], first[0]]

    def test_rows_differing_in_any_one_component_get_different_ids(self) -> None:
        """Including the case a delimiter exists to prevent, and a change of namespace.

        ``('ab', 'c')`` and ``('a', 'bc')`` are the pair that collides when parts are run together
        without a separator -- two different natural keys arriving under one id, which is a silent
        merge of two rows. The last pair is the other direction: the same value in two namespaces
        is two ids, which is what makes a namespace worth writing at all.
        """
        parts = pl.DataFrame(
            {
                "code": ["ab", "a", "ab", "ac", "ab"],
                "label": ["c", "bc", "d", "c", "c"],
            }
        )
        ids = checked(parts, f"uuid5('{NAMESPACE}', concat([code], '|', [label]))")

        assert len(set(ids[:4])) == 4, f"two different keys landed on one id: {ids[:4]}"
        assert ids[0] != ids[1], "the delimiter did not separate the parts"
        assert ids[4] == ids[0], "two identical keys produced different ids"

        assert scalar(f"uuid5('{NAMESPACE}', 'ab|c')") != scalar(f"uuid5('{OTHER_NAMESPACE}', 'ab|c')")

    @pytest.mark.parametrize(
        "text",
        [
            f"uuid5('{NAMESPACE}')",
            "uuid5()",
            f"uuid5('{NAMESPACE}', [code], [label])",
            f"uuid5('{NAMESPACE}', [code], '|', [label])",
        ],
    )
    def test_uuid5_takes_exactly_two_arguments_and_is_offered_by_name(self, text: str) -> None:
        """Fixed arity with no defaults, like every name before it, and listed where a mistake is made.

        Three arguments is the shape an author reaches for when they mean to join two columns, so
        it has to be refused by name rather than quietly absorbed -- the answer they want is
        ``uuid5(namespace, concat([a], '|', [b]))``, where the separator is visible.
        """
        assert "uuid5" in ALLOWED_FUNCTIONS

        reported = message(text)
        assert "uuid5" in reported, f"the refusal does not name the function: {reported!r}"
        assert "2" in reported, f"the refusal does not say how many arguments it takes: {reported!r}"

        assert "uuid5" in message("sqrt([amount])"), (
            "uuid5 is not offered to an author who reached for a function that does not exist"
        )

    @pytest.mark.parametrize(
        "text",
        [
            f"uuid5('{NAMESPACE}', 1)",
            f"uuid5('{NAMESPACE}', [qty])",
            f"uuid5('{NAMESPACE}', [amount])",
            f"uuid5('{NAMESPACE}', [flag])",
            f"uuid5('{NAMESPACE}', [start_date])",
            f"uuid5('{NAMESPACE}', true)",
            f"uuid5('{NAMESPACE}', date'2026-01-01')",
        ],
    )
    def test_a_value_that_is_not_text_is_refused_rather_than_quietly_stringified(
        self, frame: pl.DataFrame, text: str
    ) -> None:
        """The same rule ``concat`` carries, and for the same reason, sharpened by what this builds.

        A number stringified on the way into a hash is a key whose stability depends on how the
        engine wrote it that day: ``1`` and ``1.0`` are one value and two ids. The author who
        really means to key on a number writes the conversion where a reviewer can see it.

        The refusal has to say *text*, not merely raise: "uuid5 is not a function this language
        has" would satisfy a bare ``pytest.raises`` while the function was still missing, and this
        case would then have been passing for a reason that has nothing to do with what it pins.
        """
        reported = schema_message(frame, text)
        assert "uuid5" in reported, f"the refusal does not name the function: {reported!r}"
        assert "text" in reported.lower(), f"the refusal does not say the value has to be text: {reported!r}"

    @pytest.mark.parametrize("namespace", [NAMESPACE, OTHER_NAMESPACE])
    def test_the_value_is_hashed_as_utf8_whatever_script_it_is_written_in(
        self, rows: pl.DataFrame, namespace: str
    ) -> None:
        """Equality with ``uuid.uuid5`` over text that is not ASCII, which is where encodings differ.

        ASCII is a fixed point of UTF-8, latin-1 and cp1252 alike, so a column of ``'A'``, ``'B'``,
        ``'C'`` cannot tell those three apart: a hand-rolled digest over the wrong encoding passes
        every assertion that only ever hashes ASCII. This fixture's labels are accented and
        dotted, so the bytes hashed are pinned as the ones the standard library hashes rather than
        as whichever encoding happened to be reached for.

        What is at stake is not a different id but an unreproducible one. Under latin-1 an accented
        name still yields a well-formed, stable id -- it simply is not the id anything else in the
        world derives from that value, which is only discovered when a second system tries to join
        on it.
        """
        values = rows["label"].to_list()
        assert any(value is not None and not value.isascii() for value in values), (
            "the fixture no longer carries a non-ASCII value, so this assertion proves nothing"
        )
        assert checked(rows, f"uuid5('{namespace}', [label])") == ids_for(namespace, values)

    def test_text_no_single_byte_encoding_could_hold_is_hashed_rather_than_failing_mid_select(self) -> None:
        """Scripts outside Latin-1 entirely, and a character outside the basic plane.

        The failure mode here is worse than a wrong id. A single-byte encoding cannot represent
        these values at all, so it raises from inside a select -- halfway through a load, with ids
        already written for every row that happened to be Latin-1 and nothing written for the rest.
        The emoji is the separate case: it lives outside the basic plane, so an implementation that
        reached its bytes by way of UTF-16 would split it into a surrogate pair and hash something
        no other system derives.
        """
        written = pl.DataFrame({"label": ["é", "ÉCLAIR", "straße", "İstanbul", "κόσμος", "文字", "😀", "a😀b"]})
        values = written["label"].to_list()

        for value in ("文字", "κόσμος", "😀"):
            with pytest.raises(UnicodeEncodeError):
                value.encode("latin-1")

        assert checked(written, f"uuid5('{NAMESPACE}', [label])") == ids_for(NAMESPACE, values)
        assert len(set(checked(written, f"uuid5('{NAMESPACE}', [label])"))) == len(values)

    def test_two_spellings_of_one_accent_are_two_values_and_are_never_normalised(self) -> None:
        """``'é'`` written as one character and as a letter plus a combining mark get different ids.

        The two read identically to a person and compare unequal to a machine, and the standard
        library normalises neither -- so neither does this. It is a real shape rather than a
        curiosity: text arriving from one operating system is decomposed and the same text typed
        elsewhere is composed, so a flow that mixed the two sources would derive two ids for what a
        reader sees as one value.

        Pinning it in this direction is deliberate. Normalising would be the *friendlier* thing to
        do and would still be wrong, because the id would then be reproducible only by code that
        normalised the same way -- which is the one promise this function exists to keep.
        """
        composed = "é"
        decomposed = "é"
        assert unicodedata.normalize("NFC", decomposed) == composed
        assert unicodedata.normalize("NFD", composed) == decomposed

        written = pl.DataFrame({"label": [composed, decomposed]})
        ids = checked(written, f"uuid5('{NAMESPACE}', [label])")

        assert ids == ids_for(NAMESPACE, [composed, decomposed])
        assert ids[0] != ids[1], "the two spellings were folded together before hashing"

    def test_an_empty_value_is_an_ordinary_value_and_never_a_missing_one(self, rows: pl.DataFrame) -> None:
        """``''`` has an id of its own: present, canonical, and unlike the id of anything else.

        This is the other side of the null rule, and the side nothing pinned. An implementation
        that read emptiness as absence -- the one-character difference between "is this value
        missing" and "is this value falsy" -- passes every assertion about nulls while writing a
        null key onto a row that has data on it. Downstream that is a key that joins to nothing and
        a primary key that will not take.

        So the three outcomes are asserted together, against one frame, as three distinct things:
        the id of a value, the id of the empty value, and nothing at all.
        """
        space = uuid.UUID(NAMESPACE)
        empty = scalar(f"uuid5('{NAMESPACE}', '')")

        assert empty is not None, "the empty value was read as a missing one"
        assert empty == str(uuid.uuid5(space, ""))
        assert CANONICAL_UUID.fullmatch(empty) is not None, f"{empty!r} is not canonically written"
        assert empty != scalar(f"uuid5('{NAMESPACE}', ' ')"), "an empty value and a blank one share an id"
        assert empty != scalar(f"uuid5('{NAMESPACE}', 'null')")

        written = pl.DataFrame({"code": ["A", "", None]})
        ids = checked(written, f"uuid5('{NAMESPACE}', [code])")

        assert ids == ids_for(NAMESPACE, ["A", "", None])
        assert ids[1] is not None, "the empty value was read as a missing one"
        assert ids[2] is None, "the missing value was given an id"
        assert len({ids[0], ids[1]}) == 2, "the empty value took another value's id"

        # ... and the same over the fixture that already carries an empty cell beside its nulls.
        assert checked(rows, f"uuid5('{NAMESPACE}', [code])") == ids_for(NAMESPACE, rows["code"].to_list())

    def test_a_built_key_with_an_empty_part_stays_a_key_of_its_own(self) -> None:
        """``('', 'a|b')`` and ``('', 'a')`` and ``('a', 'b')`` are three keys, so they are three ids.

        The delimiter case again, with an empty part this time: an empty first component is where a
        separator most easily gets dropped, and dropping it merges the first row into the third.
        """
        parts = pl.DataFrame({"code": ["", "", "a"], "label": ["a|b", "a", "b"]})
        ids = checked(parts, f"uuid5('{NAMESPACE}', concat([code], '|', [label]))")

        assert ids == ids_for(NAMESPACE, ["|a|b", "|a", "a|b"])
        assert len(set(ids)) == 3, f"two different keys landed on one id: {ids}"

    def test_a_compactly_stored_value_is_hashed_as_its_text_and_not_as_how_it_is_stored(
        self, frame: pl.DataFrame, restricted_frame: pl.DataFrame
    ) -> None:
        """The same values in the two compact dtypes give the ids the plain text column gives.

        Every other text function reaches for an engine string kernel, so a column stored
        compactly announced itself by raising. This one reaches its value a row at a time instead,
        which means it would raise nothing whatever it was handed: if what arrived were the
        dictionary's physical index rather than the text behind it, the ids would be well-formed,
        canonical, stable within the run, and keyed to the order the values were first seen -- so
        the next load, which saw them in another order, would derive different ids for the same
        rows. There is no symptom to notice before the duplicates appear.

        The last pair is the path where the cast is the only thing doing this work: compiled
        against no schema at all, where the leaf reference was never cast on the way in.
        """
        plain = checked(frame, f"uuid5('{NAMESPACE}', [label])")

        assert plain == ids_for(NAMESPACE, frame["label"].to_list())
        assert checked(frame, f"uuid5('{NAMESPACE}', [category])") == plain
        assert checked(frame, f"uuid5('{NAMESPACE}', [grade])") == plain

        # Nested, where the fallback hands back a value that is still stored compactly, and names a
        # value the column was never allowed to hold.
        assert checked(restricted_frame, f"uuid5('{NAMESPACE}', coalesce([grade], 'missing'))") == ids_for(
            NAMESPACE, ["alpha", "beta", "missing"]
        )
        assert checked(restricted_frame, f"uuid5('{NAMESPACE}', coalesce([grade], 'missing'))") == checked(
            restricted_frame, f"uuid5('{NAMESPACE}', coalesce([label], 'missing'))"
        )

        assert evaluate(frame, f"uuid5('{NAMESPACE}', [category])") == plain
        assert evaluate(frame, f"uuid5('{NAMESPACE}', [grade])") == plain

    @pytest.mark.parametrize(
        "spelling",
        [
            NAMESPACE.upper(),
            "{" + NAMESPACE + "}",
            f"urn:uuid:{NAMESPACE}",
            NAMESPACE.replace("-", ""),
        ],
        ids=["uppercase", "braced", "urn", "unhyphenated"],
    )
    def test_a_namespace_the_standard_library_reads_is_accepted_and_names_the_same_space(
        self, frame: pl.DataFrame, spelling: str
    ) -> None:
        """Four written forms of one namespace, all accepted, all deriving the same ids.

        The refusals beside this pin what is *not* a namespace; nothing pinned what is. These four
        are the forms the standard library reads, so they are the forms an author may already have
        pasted out of wherever the namespace was agreed. A later tightening to a hand-written
        pattern -- the obvious way to make the refusal message nicer -- would start refusing flows
        that run today, and the author would be told their namespace is not a UUID while holding
        one that is.
        """
        assert uuid.UUID(spelling) == uuid.UUID(NAMESPACE)
        assert checked(frame, f"uuid5('{spelling}', [code])") == checked(frame, f"uuid5('{NAMESPACE}', [code])")
