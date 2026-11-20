"""What every known expression does, recorded once and checked on every run.

The expression language was restructured with no change in behaviour intended. This test holds it
to that. ``tests/fixtures/expr_characterisation.json`` records, for about fifteen hundred
expressions, whether each one is accepted or refused, the refusal's exact wording, and for
accepted ones the type and values it gives on a small fixed frame. The expressions come from the
library's own tests, worked examples and manual, plus a list of edge cases below.

A deliberate change in behaviour shows here as a difference. Regenerate the file with
``PYTHONPATH=src python tests/expr/test_expr_characterisation.py`` and say in the commit what changed.
"""

import json
import sys
import warnings
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dantron_etl.expr import ExpressionError, compile_expr, references

SNAPSHOT_FILE = Path(__file__).resolve().parents[1] / "fixtures" / "expr_characterisation.json"
REGENERATE = "PYTHONPATH=src python tests/expr/test_expr_characterisation.py"

_NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

# Edge cases that no test spells as a plain string, because the tests build them in code or never
# needed them. Each one reaches a check the restructuring moved: a limit, a number spelling, an
# argument rule, the order two refusals are made in, or a dtype the frame below carries.
EDGE_CASES: tuple[str, ...] = (
    # limits
    "not " * 40 + "true",
    "not " * 41 + "true",
    "(" * 40 + "1" + ")" * 40,
    "(" * 41 + "1" + ")" * 41,
    "1+" * 249 + "1",
    "1+" * 250 + "1",
    "[qty] in (" + ", ".join(str(number) for number in range(400)) + ")",
    "'" + "x" * 1998 + "'",
    "'" + "x" * 1999 + "'",
    "1+" * 1_000 + "1",
    # number spellings
    "0x10",
    "1_000",
    "1e400",
    "1e-400",
    "1e-9",
    "1E5",
    "1e+5",
    "1.",
    "1.5.2",
    "1e",
    "007",
    "9223372036854775807",
    "-9223372036854775808",
    "9223372036854775808",
    "--9223372036854775808",
    "- -1",
    "-(1)",
    "-null",
    "-[nothing]",
    # dtypes beyond the common ones
    "upper([ts])",
    "year([ts])",
    "[ts] > date'2026-01-01'",
    "[nothing] is null",
    "[nothing] + 1",
    "coalesce([nothing], 1)",
    "coalesce([small], [qty])",
    "coalesce([small], [amount])",
    "coalesce([small], [f32])",
    "coalesce(null, null)",
    "coalesce([code], null) = 'A'",
    "round([f32], 1)",
    "abs([u])",
    "abs([small]) + [u]",
    "len([category]) + [small]",
    "[small] / 2",
    "[u] - [small]",
    "add_months([seen], [small])",
    "add_months([seen], [u])",
    "add_months([seen], [f32])",
    "add_months([ts], 1)",
    "format_number([f32], 2, ',', '.')",
    "format_number([qty], 0, ',', '.')",
    "format_number([small], 1, ' ', ',')",
    "format_number([amount], 3, '.', ',')",
    "format_number([amount] / 0, 2, ',', '.')",
    "format_number(0.0 / 0.0, 2, ',', '.')",
    f"uuid5('{_NAMESPACE}', [category])",
    f"uuid5('{_NAMESPACE}', [grade])",
    f"uuid5('{_NAMESPACE}', concat([code], '|', [label]))",
    "concat([category], '-', [grade])",
    "concat([code], [nothing])",
    "regex_replace([grade], 'a', '$1')",
    "regex_replace([category], 'a', '$$')",
    "regex_extract([grade], '(a)', 1)",
    "regex_match([category], '^g')",
    "contains([category], 'a')",
    "starts_with([grade], [category])",
    "[grade] = 'A'",
    "[grade] in ('A', 'B')",
    "[category] = [grade]",
    "lower([grade]) = 'beta'",
    "trim([category])",
    # arity
    "round([amount])",
    "round([amount], 1, 2)",
    "coalesce([code])",
    "coalesce()",
    "concat('a')",
    "upper()",
    "upper([code], [code])",
    "uuid5('x')",
    "format_number(1, 2, ',')",
    "year()",
    "regex_extract([label], 'a')",
    # fixed arguments
    "round([amount], [qty])",
    "round([amount], -1)",
    "round([amount], 65536)",
    "round([amount], 65535)",
    "round([amount], 1.0)",
    "round([amount], true)",
    "round([amount], null)",
    "regex_extract([label], '(a)', 99999)",
    "regex_extract([label], [code], 1)",
    "regex_extract([label], '(a)', [qty])",
    "regex_replace([label], 'a', [code])",
    "regex_replace([label], 'a', null)",
    "regex_match([label], '(')",
    "regex_replace([label], '(', 'x')",
    "regex_extract([label], '(?<=a)b', 1)",
    "regex_replace([label], 'a', '$5.00')",
    "regex_replace([label], 'l', '$')",
    "regex_match([label], null)",
    "uuid5([code], [code])",
    "uuid5('not-a-uuid', [code])",
    f"uuid5('{_NAMESPACE}', 1)",
    f"uuid5('{_NAMESPACE}', [qty])",
    f"uuid5('{_NAMESPACE.upper()}', [code])",
    "format_number([amount], 2, 1, '.')",
    "format_number([amount], 2, ',', [code])",
    "format_number([amount], 2.5, ',', '.')",
    "format_number([amount], true, ',', '.')",
    "format_number([code], 2, ',', '.')",
    # add_months counts, however they arrive
    "add_months([seen], 1.5)",
    "add_months([seen], [amount])",
    "add_months([seen], [qty] / 2)",
    "add_months([seen], [qty] * 2)",
    "add_months([seen], round([amount], 0))",
    "add_months([seen], round([qty], 0))",
    "add_months([seen], coalesce([qty], 1))",
    "add_months([seen], coalesce([qty], 1.5))",
    "add_months([seen], abs([qty]))",
    "add_months([seen], -[qty])",
    "add_months([seen], null)",
    "add_months(null, 1)",
    "add_months([seen], len([code]))",
    "add_months([seen], year([seen]))",
    "add_months([seen], [flag])",
    "add_months([seen], 'x')",
    # the parts of a date, moving by days, the start of a month, reading text as a date
    "month([seen])",
    "day([seen])",
    "quarter([seen])",
    "day_of_week([seen])",
    "month([ts])",
    "day_of_week([ts])",
    "month([seen]) * 1000",
    "month([label])",
    "quarter([qty])",
    "day()",
    "day_of_week([seen], 1)",
    "add_days([seen], [qty])",
    "add_days([seen], -[small])",
    "add_days([seen], [u])",
    "add_days([ts], 1)",
    "add_days([seen], 1.5)",
    "add_days([seen], [amount])",
    "add_days([seen], null)",
    "add_days(null, 1)",
    "add_days([label], 1)",
    "month_start([seen])",
    "month_start([ts])",
    "month_start([label])",
    "parse_date('03/02/2001', '%d/%m/%Y')",
    "parse_date([code], '%d/%m/%Y')",
    "parse_date([category], '%Y-%m-%d')",
    "parse_date(null, '%Y-%m-%d')",
    "parse_date([label], [code])",
    "parse_date([label], '%Q')",
    "parse_date([label], '%Y-%m')",
    "parse_date([qty], '%Y')",
    # floor, ceil, sign, mod, least, greatest, nullif
    "floor([amount])",
    "ceil([amount])",
    "floor([f32])",
    "ceil([qty])",
    "floor([u])",
    "sign([amount])",
    "sign([small])",
    "sign(-0.0)",
    "ceil(-0.5)",
    "floor(null)",
    "floor([nothing])",
    "floor([code])",
    "mod([qty], 2)",
    "mod(-[qty], 2)",
    "mod([qty], -2)",
    "mod([amount], 3)",
    "mod([u], [small])",
    "mod([qty], 0)",
    "mod([qty], 0.0)",
    "mod([qty], [qty])",
    "mod(null, 2)",
    "mod([nothing], 2)",
    "mod([code], 2)",
    "mod(-9223372036854775808, 3)",
    "mod(-9223372036854775807, 3)",
    "mod(-9223372036854775808, -1)",
    "mod(5, -9223372036854775808)",
    "least([qty], [amount])",
    "greatest([qty], [small], [u])",
    "least([code], [label])",
    "greatest([seen], [ts])",
    "least([qty], null)",
    "least(null, null)",
    "greatest([qty], [code])",
    "least([qty])",
    "nullif([qty], 0)",
    "nullif([code], '')",
    "nullif([code], null)",
    "nullif([qty], 'x')",
    "nullif([ts], [seen])",
    "nullif([category], 'A')",
    # left, right, substr, split_part, pad_left, pad_right
    "left([label], 3)",
    "left([label], -3)",
    "left([label], 0)",
    "right([label], 2)",
    "right([grade], -1)",
    "left([label], [qty])",
    "left([label], 65536)",
    "left([qty], 1)",
    "substr([label], 2, 3)",
    "substr([label], 3)",
    "substr([category], 1, 1)",
    "substr([label], 0, 1)",
    "substr([label], 1, -1)",
    "substr([label])",
    "substr([label], 1, 2, 3)",
    "split_part([note], '=', 2)",
    "split_part([note], ' ', -1)",
    "split_part([note], '=', 9)",
    "split_part([note], '', 1)",
    "split_part([note], '=', 0)",
    "split_part([note], [code], 1)",
    "pad_left([code], 3, '0')",
    "pad_right([label], 6, '.')",
    "pad_left([code], 3, '')",
    "pad_left([code], 3, '00')",
    "pad_left([code], -1, '0')",
    "pad_left([qty], 3, '0')",
    # which of two refusals comes first
    "concat(1, upper(2))",
    "concat(upper(1), 2)",
    "coalesce([code], 1, upper(2))",
    "coalesce(upper(1), 1, 'x')",
    "starts_with(1, upper(2))",
    "add_months(1, [code])",
    "format_number('x', -1, 1, 2)",
    "uuid5('bad', 1)",
    "regex_extract(1, '(', -1)",
    "regex_replace(1, '(', 2)",
    "regex_replace([label], '(', 2)",
    "[code] + 1 > upper(1)",
    "[code] in (1, 'a')",
    "upper(1) in ('a', 1)",
    # the scan's own names and other callees
    "reverse([code])",
    "col(0)",
    "lit(0)",
    "date('2026-01-01')",
    "date '2026-01-01'",
    "date'2026-02-30'",
    "Date'2026-01-01'",
    "x'41'",
    "upper ([code])",
)


def frame() -> pl.DataFrame:
    """Five rows; the fourth is null in every column that can hold a null.

    The column names are the ones the harvested expressions use most, so most of them evaluate
    rather than stopping at "not a column". ``ts``, ``nothing``, ``small``, ``u`` and ``f32`` carry
    the dtypes the edge cases above need.
    """
    labels = ["  alpha  ", "Beta", "gamma", None, "é"]
    days = [date(2026, 1, 31), date(2026, 6, 15), date(2025, 12, 31), None, date(2024, 2, 29)]
    text = ["A", "B", "a", None, ""]
    numbers = [10.256, -2.4, 100.0, None, -15.215]
    counts = [1, 2, 3, None, 0]
    columns: dict[str, Any] = {
        "code": text,
        "label": labels,
        "note": ["a=b", "x [y]", "and", None, ""],
        "amount": numbers,
        "qty": counts,
        "flag": [True, False, True, None, False],
        "category": pl.Series(labels, dtype=pl.Categorical),
        "grade": pl.Series(labels, dtype=pl.Enum(["  alpha  ", "Beta", "gamma", "é", "A", "B", "Z"])),
        "ts": [datetime(2026, 1, 31, 12), datetime(2026, 6, 15), None, None, datetime(2024, 2, 29, 23, 59)],
        "nothing": pl.Series([None] * 5, dtype=pl.Null),
        "small": pl.Series(counts, dtype=pl.Int32),
        "u": pl.Series([1, 2, 3, None, 0], dtype=pl.UInt32),
        "f32": pl.Series([1.25, -2.5, 3.0, None, 0.5], dtype=pl.Float32),
    }
    for name in ("seen", "start_date", "d", "start", "period_start", "period_end"):
        columns[name] = days
    for name in ("months", "months_in_period", "key", "record_id", "count"):
        columns[name] = counts
    for name in ("line_amount", "total", "ratio", "x"):
        columns[name] = numbers
    for name in ("name", "status", "company", "type"):
        columns[name] = text
    return pl.DataFrame(columns)


def _attempt(build: Callable[[], Any]) -> tuple[Any, str]:
    """``(result, "ok")``, or ``(None, "refused: <message>")`` for a refusal of this language."""
    try:
        return build(), "ok"
    except ExpressionError as error:
        return None, f"refused: {error}"


def _evaluated(data: pl.DataFrame, expression: pl.Expr) -> dict[str, str]:
    """The type and values an expression gives on ``data``, or the engine's error if it raises."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = data.select(expression.alias("out"))["out"]
    except Exception as error:  # noqa: BLE001 -- an engine error is behaviour too, and is recorded
        # Only the kind of error: its wording is the engine's own and changes between its versions.
        return {"error": type(error).__name__}
    return {"dtype": str(out.dtype), "values": repr(out.to_list())}


def record(text: str, data: pl.DataFrame) -> dict[str, Any]:
    """Everything this language says about ``text``, in each of the three ways a caller asks.

    A refusal is usually the same whichever way the text was compiled. Where it is, the later
    entry says "same" instead of repeating the message, which keeps the snapshot a readable size.
    """
    entry: dict[str, Any] = {"text": text}

    bare, entry["no_columns"] = _attempt(lambda: compile_expr(text))
    names, outcome = _attempt(lambda: sorted(references(text)))
    entry["references"] = names if names is not None else _same(outcome, entry["no_columns"])
    if bare is not None and set(names or ()) <= set(data.columns):
        entry["no_columns_result"] = _evaluated(data, bare)

    _, names_outcome = _attempt(lambda: compile_expr(text, data.columns))
    entry["names_only"] = _same(names_outcome, entry["no_columns"])

    typed, schema_outcome = _attempt(lambda: compile_expr(text, data.schema))
    entry["schema"] = _same(schema_outcome, names_outcome)
    if typed is not None:
        entry["schema_result"] = _evaluated(data, typed)
    return entry


def _same(outcome: str, earlier: str) -> str:
    return "same" if outcome == earlier and outcome != "ok" else outcome


def current(texts: list[str]) -> list[dict[str, Any]]:
    data = frame()
    return [record(text, data) for text in texts]


def _as_text(records: list[dict[str, Any]]) -> str:
    return "[\n" + ",\n".join(json.dumps(entry, ensure_ascii=True) for entry in records) + "\n]\n"


def test_every_recorded_expression_still_does_exactly_what_it_did() -> None:
    """Accepted or refused, the same wording, the same type and the same values, for every one."""
    expected = json.loads(SNAPSHOT_FILE.read_text(encoding="utf8"))
    actual = current([entry["text"] for entry in expected])
    changed = [(before, after) for before, after in zip(expected, actual, strict=True) if before != after]
    shown = "\n\n".join(f"before: {before}\nafter:  {after}" for before, after in changed[:5])
    assert not changed, (
        f"{len(changed)} of {len(expected)} expressions now behave differently. If that is deliberate, "
        f"regenerate with `{REGENERATE}` and say so in the commit.\n\n{shown}"
    )


def test_the_snapshot_covers_the_edge_cases_and_both_outcomes() -> None:
    """The file is not empty, holds every edge case above, and records refusals as well as results."""
    expected = json.loads(SNAPSHOT_FILE.read_text(encoding="utf8"))
    texts = {entry["text"] for entry in expected}
    assert set(EDGE_CASES) <= texts
    assert sum(1 for entry in expected if "schema_result" in entry) > 300
    assert sum(1 for entry in expected if "schema_result" not in entry) > 300


if __name__ == "__main__":
    # Run from the checkout's root with PYTHONPATH=src, so this checkout's own source is recorded.
    # With a file of texts as the argument, record those texts; without one, re-record the texts
    # the snapshot already holds.
    if len(sys.argv) > 1:
        texts = json.loads(Path(sys.argv[1]).read_text(encoding="utf8"))
    else:
        texts = [entry["text"] for entry in json.loads(SNAPSHOT_FILE.read_text(encoding="utf8"))]
    texts = list(dict.fromkeys([*texts, *EDGE_CASES]))
    SNAPSHOT_FILE.write_text(_as_text(current(texts)), encoding="utf8")
