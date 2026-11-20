"""What polars does with exact decimals, pinned so an engine upgrade cannot change it quietly.

The library's exact-decimal design leans on some of these behaviours and works around others. None
of these tests call the library. Each one asserts what the installed polars does, and its message
names the design decision to revisit if that changes.

The package supports a range of polars versions, and the two ends behave differently in a few
places. Where they do, the test states both answers. The library must not rely on either of them;
it makes its own answer, and the test says how.

Every exact column the library makes has 38 digits in all, so most cases here use ``Decimal(38, s)``.
``Decimal(18, s)`` appears where a source or the engine itself gives one.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import polars as pl
import pytest

#: The installed polars, as (major, minor).
ENGINE = tuple(int(part) for part in pl.__version__.split(".")[:2])

#: The two ends of the supported range, where every answer below was measured.
LOWEST = (1, 38)
LOCKED = (1, 44)


def exact(values: list[str | None], places: int, *, precision: int = 38, name: str = "x") -> pl.Series:
    """A series of exact decimals, written as text so nothing passes through a float."""
    return pl.Series(name, [None if v is None else Decimal(v) for v in values], dtype=pl.Decimal(precision, places))


def revisit(decision: str) -> str:
    """The message for a changed behaviour: which polars changed it, and what to look at again."""
    return f"polars {pl.__version__} no longer behaves as measured. Revisit: {decision}"


def by_version(on_lowest: Any, on_locked: Any) -> list[Any]:
    """The answers this polars may give where the two ends of the range differ.

    The lowest version must give the first, the locked version and anything newer the second. A
    version in between may give either, since the change was measured only at the ends.
    """
    if ENGINE <= LOWEST:
        return [on_lowest]
    if ENGINE >= LOCKED:
        return [on_locked]
    return [on_lowest, on_locked]


@contextmanager
def refused(error: type[BaseException], decision: str, match: str | None = None) -> Iterator[None]:
    """Expect polars to raise ``error``; if it does not, fail with the decision to revisit."""
    try:
        with pytest.raises(error, match=match):
            yield
    except (Exception, pytest.fail.Exception) as failure:
        pytest.fail(f"{revisit(decision)}\n{type(failure).__name__}: {failure}")


def column(frame: pl.DataFrame, expression: pl.Expr) -> pl.Series:
    """One computed column, named r."""
    return frame.select(expression.alias("r"))["r"]


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_add_and_subtract_are_exact_at_the_larger_places() -> None:
    frame = pl.DataFrame([exact(["812.40", "100.00"], 2, name="a"), exact(["0.0725", "7.0000"], 4, name="b")])
    total = column(frame, pl.col("a") + pl.col("b"))
    difference = column(frame, pl.col("a") - pl.col("b"))
    decision = "unit 10 lets the engine add and subtract exact columns, and gives the larger places"
    assert total.dtype == pl.Decimal(38, 4), revisit(decision)
    assert total.to_list() == [Decimal("812.4725"), Decimal("107.0000")], revisit(decision)
    assert difference.to_list() == [Decimal("812.3275"), Decimal("93.0000")], revisit(decision)


def test_multiply_rounds_the_product_to_the_larger_places_half_to_even() -> None:
    frame = pl.DataFrame(
        [exact(["812.40", "0.07", "0.15", "0.25"], 2, name="a"), exact(["0.07", "0.07", "0.10", "0.10"], 2, name="b")]
    )
    product = column(frame, pl.col("a") * pl.col("b"))
    decision = "unit 11 widens one side to the sum of both places before multiplying, so nothing rounds"
    assert product.dtype == pl.Decimal(38, 2), revisit(decision)
    # 56.868 to 56.87; 0.0049 to 0.00; the ties 0.0150 and 0.0250 both go to the even 0.02.
    assert product.to_list() == [Decimal("56.87"), Decimal("0.00"), Decimal("0.02"), Decimal("0.02")], revisit(decision)


def test_multiply_is_exact_once_one_side_carries_both_places() -> None:
    frame = pl.DataFrame([exact(["812.40", "0.07"], 2, name="a"), exact(["0.0725", "0.0700"], 4, name="b")])
    product = column(frame, pl.col("a").cast(pl.Decimal(38, 6)) * pl.col("b"))
    decision = "unit 11 multiplies exactly by casting one side to the sum of both places first"
    assert product.dtype == pl.Decimal(38, 6), revisit(decision)
    assert product.to_list() == [Decimal("58.899000"), Decimal("0.004900")], revisit(decision)
    # A product of 38 digits in all is still exact: the engine works it out wider and then fits it.
    wide = pl.DataFrame([exact(["999999999999999999.00"], 2, name="a"), exact(["999999999999999999.00"], 2, name="b")])
    assert column(wide, pl.col("a") * pl.col("b")).to_list() == [Decimal("999999999999999998000000000000000001.00")], (
        revisit(decision)
    )


def test_divide_keeps_the_larger_places_and_rounds_half_to_even() -> None:
    frame = pl.DataFrame([exact(["100.00", "1.00"], 2, name="a"), exact(["7.00", "8.00"], 2, name="b")])
    quotient = column(frame, pl.col("a") / pl.col("b"))
    decision = "unit 14 casts the dividend to 16 places itself, because the engine gives too few"
    assert quotient.dtype == pl.Decimal(38, 2), revisit(decision)
    # 14.2857... to 14.29; the tie 0.125 goes to the even 0.12.
    assert quotient.to_list() == [Decimal("14.29"), Decimal("0.12")], revisit(decision)
    ones = pl.DataFrame([exact(["1"], 0, name="a"), exact(["3"], 0, name="b")])
    assert column(ones, pl.col("a") / pl.col("b")).to_list() == [Decimal("0")], revisit(decision)


def test_divide_at_sixteen_places_rounds_the_sixteenth_half_to_even() -> None:
    frame = pl.DataFrame(
        [
            exact(["100.00", "0.0000000000000001", "0.0000000000000003"], 16, name="a"),
            exact(["7", "2", "2"], 2, name="b"),
        ]
    )
    quotient = column(frame, pl.col("a") / pl.col("b"))
    decision = (
        "unit 14 rounds the 16th place half away from zero itself, by dividing at 17 places to find the ties; "
        "the engine rounds them to even"
    )
    assert quotient.dtype == pl.Decimal(38, 16), revisit(decision)
    # The ties ...05 and ...15 at the 17th place go to the even ...0 and ...2.
    assert quotient.to_list() == [
        Decimal("14.2857142857142857"),
        Decimal("0E-16"),
        Decimal("0.0000000000000002"),
    ], revisit(decision)


def test_divide_by_a_zero_decimal_is_an_error_not_infinity() -> None:
    frame = pl.DataFrame([exact(["1.00"], 2, name="a"), exact(["0.00"], 2, name="b")])
    decision = "the library checks for zero divisors before the engine sees them"
    with refused(pl.exceptions.ComputeError, decision, match="division by zero"):
        column(frame, pl.col("a") / pl.col("b"))


def test_divide_by_zero_on_a_row_the_case_never_reaches() -> None:
    """The two ends of the range differ. The library nulls zero divisors so neither answer shows."""
    frame = pl.DataFrame([exact(["1.00", "2.00"], 2, name="a"), exact(["0.00", "4.00"], 2, name="b")])
    guarded = pl.when(pl.col("b") != 0).then(pl.col("a") / pl.col("b"))
    try:
        answer: Any = column(frame, guarded).to_list()
    except pl.exceptions.ComputeError:
        answer = "division by zero"
    decision = "unit 14 nulls zero divisors before dividing, so a guarded division works on every version"
    assert answer in by_version("division by zero", [None, Decimal("0.50")]), revisit(decision)


def test_exact_with_a_whole_number_column_stays_exact() -> None:
    frame = pl.DataFrame([exact(["100.00", "812.40"], 2, name="a"), pl.Series("i", [7, 3], dtype=pl.Int64)])
    decision = "units 10 to 12 let an exact column meet a whole number without converting either"
    for expression, expected in (
        (pl.col("a") + pl.col("i"), [Decimal("107.00"), Decimal("815.40")]),
        (pl.col("a") * pl.col("i").cast(pl.Int32), [Decimal("700.00"), Decimal("2437.20")]),
        (pl.col("a") - pl.lit(7), [Decimal("93.00"), Decimal("805.40")]),
        (pl.col("a") / pl.col("i"), [Decimal("14.29"), Decimal("270.80")]),
    ):
        result = column(frame, expression)
        assert result.dtype == pl.Decimal(38, 2), revisit(decision)
        assert result.to_list() == expected, revisit(decision)


def test_a_decimal_literal_carries_its_written_places() -> None:
    frame = pl.DataFrame([exact(["812.40"], 2, name="a")])
    decision = "unit 9 keeps a typed number's written text and makes it a literal of that many places"
    assert column(frame, pl.lit(Decimal("0.0725"))).dtype == pl.Decimal(38, 4), revisit(decision)
    assert column(frame, pl.lit(Decimal("1.10"))).dtype == pl.Decimal(38, 2), revisit(decision)
    total = column(frame, pl.col("a") + pl.lit(Decimal("0.0725")))
    assert (total.dtype, total.to_list()) == (pl.Decimal(38, 4), [Decimal("812.4725")]), revisit(decision)


def test_exact_meeting_a_float_slides_to_float_without_a_word() -> None:
    frame = pl.DataFrame([exact(["812.40", "0.30"], 2, name="a"), pl.Series("f", [0.0725, 0.1 + 0.2])])
    decision = "unit 12 stops the step when an exact column meets a float, because the engine never does"
    product = column(frame, pl.col("a") * pl.col("f"))
    assert product.dtype == pl.Float64, revisit(decision)
    assert product[0] == 58.898999999999994, revisit(decision)
    assert column(frame, pl.col("a") * pl.lit(0.0725)).dtype == pl.Float64, revisit(decision)
    # 0.30 against 0.1 + 0.2 is compared as floats, and is false.
    assert column(frame, pl.col("a") == pl.col("f")).to_list() == [False, False], revisit(decision)
    branch = pl.when(pl.col("a") > 1).then(pl.col("a")).otherwise(pl.col("f"))
    assert column(frame, branch).dtype == pl.Float64, revisit(decision)
    assert column(frame, pl.coalesce(pl.col("a"), pl.lit(0.0))).dtype == pl.Float64, revisit(decision)
    assert column(frame, pl.coalesce(pl.col("a"), pl.lit(0))).dtype == pl.Decimal(38, 2), revisit(decision)


def test_comparison_across_places_is_by_value() -> None:
    frame = pl.DataFrame([exact(["0.07", "1.00", None], 2, name="a"), exact(["0.0700", "0.9999", "1"], 4, name="b")])
    decision = "unit 13 compares exact columns of different places directly"
    assert column(frame, pl.col("a") == pl.col("b")).to_list() == [True, False, None], revisit(decision)
    assert column(frame, pl.col("a") > pl.col("b")).to_list() == [False, True, None], revisit(decision)


def test_membership_against_a_list_of_other_places_is_refused() -> None:
    decision = "unit 13 builds an 'in' list at the same places as the column, widening whichever has fewer"
    with refused(pl.exceptions.InvalidOperationError, decision):
        exact(["0.30", "0.10"], 2).is_in([Decimal("0.3")])
    assert exact(["0.30", "0.10"], 2).is_in([Decimal("0.30")]).to_list() == [True, False], revisit(decision)


def test_supertype_of_two_exact_branches_is_the_larger_places() -> None:
    frame = pl.DataFrame([exact(["1.5", "2.25"], 2, name="a"), exact(["1.5000", "2.2500"], 4, name="b")])
    branch = pl.when(pl.col("a") > 2).then(pl.col("a")).otherwise(pl.col("b"))
    decision = "unit 17 casts every branch to the larger places itself, since the engine's total digits vary"
    assert column(frame, branch).dtype.scale == 4, revisit(decision)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Converting to and from an exact column
# ---------------------------------------------------------------------------


def test_every_narrowing_conversion_rounds_half_to_even() -> None:
    decision = "unit 3 checks every value fits before converting to fewer places; the engine rounds instead"
    fewer = exact(["1.2350", "1.2250", "-1.2350", "1.2300"], 4).cast(pl.Decimal(38, 2))
    assert fewer.to_list() == [Decimal("1.24"), Decimal("1.22"), Decimal("-1.24"), Decimal("1.23")], revisit(decision)
    from_float = pl.Series([1.005, 2.675, 0.125, 0.1 + 0.2]).cast(pl.Decimal(38, 2))
    assert from_float.to_list() == [Decimal("1.00"), Decimal("2.68"), Decimal("0.12"), Decimal("0.30")], revisit(
        decision
    )
    from_text = pl.Series(["1.005", "1.015", "1.006"]).cast(pl.Decimal(38, 2))
    assert from_text.to_list() == [Decimal("1.00"), Decimal("1.02"), Decimal("1.01")], revisit(decision)


def test_text_conversion_takes_spellings_the_library_refuses() -> None:
    decision = "unit 3 reads text as plain digits itself, because the engine takes exponents and signs"
    read = pl.Series(["1e3", ".5", "+1.25", " 1.50", "1,000.00", "abc"]).cast(pl.Decimal(38, 2), strict=False)
    assert read.to_list() == [Decimal("1000.00"), Decimal("0.50"), Decimal("1.25"), None, None, None], revisit(decision)


def test_float_conversion_is_not_the_float_read_to_fifteen_digits() -> None:
    decision = "unit 3 reads a float to 15 significant digits itself, through text, before converting"
    wide = pl.Series([0.1, 1 / 3]).cast(pl.Decimal(38, 20))
    # 0.1 comes out clean, but 1/3 neither matches its binary value (0.33333333333333331483) nor
    # stops at 15 digits.
    assert wide.to_list() == [Decimal("0.10000000000000000000"), Decimal("0.33333333333333331968")], revisit(decision)
    special = pl.Series([float("nan"), float("inf"), 1.5])
    assert special.cast(pl.Decimal(38, 2), strict=False).to_list() == [None, None, Decimal("1.50")], revisit(decision)
    with refused(pl.exceptions.InvalidOperationError, decision):
        special.cast(pl.Decimal(38, 2))


def test_a_value_too_large_for_the_places_is_null_or_an_error() -> None:
    decision = "unit 3 counts whole digits itself so a value that does not fit is named, not emptied"
    whole = pl.Series([10**8 - 1, 10**8, None])
    assert whole.cast(pl.Decimal(38, 30), strict=False).to_list() == [
        Decimal("99999999.000000000000000000000000000000"),
        None,
        None,
    ], revisit(decision)
    with refused(pl.exceptions.InvalidOperationError, decision):
        whole.cast(pl.Decimal(38, 30))
    widened = exact(["1000000000000000000000000000000000.00"], 2).cast(pl.Decimal(38, 6), strict=False)
    assert widened.to_list() == [None], revisit(decision)


def test_exact_to_float_is_wrong_past_fifteen_significant_digits() -> None:
    decision = "unit 3's exact-to-float helper goes through text, which lands on the nearest double"
    values = exact(["99999999999999.99", "0.10", "1234567890.12"], 2)
    direct = values.cast(pl.Float64).to_list()
    through_text = values.cast(pl.String).cast(pl.Float64).to_list()
    assert direct[0] == 100000000000000.0, revisit(decision)
    assert through_text[0] == 99999999999999.98, revisit(decision)
    assert direct[1:] == through_text[1:] == [0.1, 1234567890.12], revisit(decision)


def test_exact_to_text_writes_every_place_held() -> None:
    decision = (
        "write_csv writes these digits, and unit 3's helpers read an exact or whole column through this text, "
        "so it must stay plain digits with no exponent"
    )
    text = exact(["12.00", "-0.50", "-0.00", None], 2).cast(pl.String)
    assert text.to_list() == ["12.00", "-0.50", "0.00", None], revisit(decision)
    assert exact(["58.899000"], 6).cast(pl.String).to_list() == ["58.899000"], revisit(decision)
    assert exact(["12"], 0).cast(pl.String).to_list() == ["12"], revisit(decision)
    assert exact(["9" * 38], 0).cast(pl.String).to_list() == ["9" * 38], revisit(decision)
    tiny = "-0." + "0" * 37 + "1"
    assert exact([tiny], 38).cast(pl.String).to_list() == [tiny], revisit(decision)
    whole = (
        pl.Series([2**64 - 1], dtype=pl.UInt64).cast(pl.String).to_list()
        + pl.Series([-(2**63)]).cast(pl.String).to_list()
    )
    assert whole == ["18446744073709551615", "-9223372036854775808"], revisit(decision)


def test_a_float_as_text_is_the_shortest_text_that_reads_back_the_same() -> None:
    decision = "unit 3 reads a float to 15 significant digits from this text, which may use an exponent"
    floats = pl.Series([0.1 + 0.2, 1.005, 1e-7, 0.00001, 1e16, 1.2345678901234568e20, -0.0, 5e-324])
    assert floats.cast(pl.String).to_list() == [
        "0.30000000000000004",
        "1.005",
        "1e-7",
        "0.00001",
        "1e+16",
        "1.2345678901234568e+20",
        "-0.0",
        "5e-324",
    ], revisit(decision)
    special = pl.Series([float("nan"), float("inf"), float("-inf")]).cast(pl.String).to_list()
    assert special == ["NaN", "inf", "-inf"], revisit(decision)
    narrow = pl.Series([0.1, 1.005, 16777217.0], dtype=pl.Float32).cast(pl.String).to_list()
    assert narrow == ["0.1", "1.005", "16777216.0"], revisit(decision)


def test_python_decimals_take_the_largest_places_and_stop_at_38_digits() -> None:
    decision = "unit 2 fixes total digits at 38, the most the engine holds"
    mixed = pl.Series([Decimal("1.5"), Decimal("2.255"), Decimal("3")])
    assert mixed.dtype == pl.Decimal(38, 3), revisit(decision)
    assert mixed.to_list() == [Decimal("1.500"), Decimal("2.255"), Decimal("3.000")], revisit(decision)
    assert exact(["9" * 38], 0).to_list() == [Decimal("9" * 38)], revisit(decision)
    with refused(Exception, decision):  # the error's class differs by version; any refusal will do
        exact(["1" + "0" * 38], 0)


def test_round_half_away_keeps_the_column_places() -> None:
    decision = "unit 15 casts the rounded value to exactly n places after rounding"
    rounded = exact(["1.005", "-2.345", "2.500"], 3).round(2, mode="half_away_from_zero")
    assert rounded.dtype == pl.Decimal(38, 3), revisit(decision)
    assert rounded.to_list() == [Decimal("1.010"), Decimal("-2.350"), Decimal("2.500")], revisit(decision)
    assert exact(["-2.500"], 3).round(0, mode="half_away_from_zero").to_list() == [Decimal("-3.000")], revisit(decision)


# ---------------------------------------------------------------------------
# Aggregates and overflow
# ---------------------------------------------------------------------------


def test_sum_total_digits_depend_on_the_version() -> None:
    """The two ends differ. Unit 7 makes every total 38 digits, so neither answer shows."""
    frame = pl.DataFrame({"k": ["a", "a"], "x": exact(["1.10", "2.20"], 2, precision=18)})
    decision = "unit 7 casts every exact total to 38 digits"
    expected = by_version(pl.Decimal(18, 2), pl.Decimal(38, 2))
    assert frame.select(pl.col("x").sum())["x"].dtype in expected, revisit(decision)
    assert frame.group_by("k").agg(pl.col("x").sum())["x"].dtype in expected, revisit(decision)
    pivoted = frame.with_columns(c=pl.lit("c")).pivot("c", index="k", values="x", aggregate_function="sum")
    assert pivoted["c"].dtype in expected, revisit(decision)
    assert frame.select(pl.col("x").min(), pl.col("x").max().alias("y")).dtypes == [pl.Decimal(18, 2)] * 2, revisit(
        decision
    )
    wide = pl.DataFrame({"x": exact(["1.10", "2.20"], 2)})
    assert wide.select(pl.col("x").sum())["x"].to_list() == [Decimal("3.30")], revisit(decision)


def test_sum_of_only_nulls_is_zero() -> None:
    decision = "group_by and cross_tab turn an all-null exact sum into null themselves, as SQL's SUM does"
    assert exact([None, None], 2).sum() == Decimal("0.00"), revisit(decision)


def test_statistics_of_an_exact_column_come_back_as_floats() -> None:
    decision = "mean and median work out their own exact answer; std and var refuse an exact column"
    frame = pl.DataFrame({"x": exact(["2.20", "3.35"], 2)})
    for statistic in (pl.col("x").mean(), pl.col("x").median(), pl.col("x").std(), pl.col("x").quantile(0.5)):
        assert frame.select(statistic)["x"].dtype == pl.Float64, revisit(decision)
    assert frame.select(pl.col("x").median())["x"][0] == 2.775, revisit(decision)


def test_arithmetic_overflow_is_an_error() -> None:
    decision = "unit 8 turns the engine's overflow error into a refusal that names the step and rows"
    biggest = "9" * 36 + ".00"  # 38 digits in all, the most there is
    frame = pl.DataFrame([exact([biggest], 2, name="a"), exact(["1.00"], 2, name="b")])
    for expression in (pl.col("a") + pl.col("b"), -pl.col("a") - pl.col("b"), pl.col("a") * (pl.col("b") * 10)):
        with refused(pl.exceptions.ComputeError, decision, match="overflow"):
            column(frame, expression)
    with refused(pl.exceptions.ComputeError, decision, match="overflow"):
        frame.select(pl.col("a").append(pl.col("a")).cum_sum())


def test_a_group_total_past_38_digits_is_silently_wrong() -> None:
    decision = "unit 8 checks each group total in Float64 before summing, because the engine does not"
    frame = pl.DataFrame({"k": ["a", "a"], "x": exact(["9" * 36 + ".00"] * 2, 2)})
    total = frame.group_by("k").agg(pl.col("x").sum()).rows()[0][1]
    assert total == Decimal("-1402823669209384634633746074317682116.6"), revisit(decision)


def test_a_plain_total_past_38_digits() -> None:
    """The two ends differ. The lowest gives a wrong value, and polars panics when it is later read;
    the locked version raises at once. Unit 8's check runs first, so neither is reached."""
    frame = pl.DataFrame({"x": exact(["9" * 36 + ".00"] * 2, 2)})
    decision = "unit 8 checks totals before summing; a 1.38 panic is not an Exception and escapes the boundary"
    try:
        total = frame.select(pl.col("x").sum())
    except pl.exceptions.ComputeError:
        answer: Any = "error"
    else:
        assert total.rows() == [(Decimal("-1402823669209384634633746074317682116.6"),)], revisit(decision)
        try:
            total["x"].to_list()
        except Exception:  # noqa: BLE001 - an ordinary error here would also be a change
            answer = "error when read"
        except BaseException:  # noqa: BLE001 - the panic is what this pins
            answer = "panic when read"
        else:
            answer = "wrong value, read quietly"
    assert answer in by_version("panic when read", "error"), revisit(decision)


# ---------------------------------------------------------------------------
# Joining, stacking and saving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("right", "label"),
    [
        (exact(["1.5000"], 4, name="k"), "4 places"),
        (exact(["1.50"], 2, precision=18, name="k"), "18 digits"),
        (pl.Series("k", [1], dtype=pl.Int64), "Int64"),
        (pl.Series("k", [1.5]), "Float64"),
    ],
)
def test_join_keys_must_have_the_same_type(right: pl.Series, label: str) -> None:
    decision = "unit 18 casts exact join keys to one type itself and refuses a float key by name"
    left = pl.DataFrame({"k": exact(["1.50"], 2, name="k"), "l": [1]})
    with refused(pl.exceptions.SchemaError, f"{decision} (right key: {label})"):
        left.join(pl.DataFrame({"k": right, "r": [2]}), on="k")


def test_stacking_fewer_places_over_more_rounds_without_a_word() -> None:
    decision = "unit 19 widens every exact column to the larger places before stacking"
    top = pl.DataFrame({"x": exact(["1.50"], 2)})
    below = pl.DataFrame({"x": exact(["1.2345"], 4)})
    stacked = pl.concat([top, below])
    assert (stacked["x"].dtype, stacked["x"].to_list()) == (
        pl.Decimal(38, 2),
        [Decimal("1.50"), Decimal("1.23")],
    ), revisit(decision)
    relaxed = pl.concat([top, below], how="vertical_relaxed")["x"]
    assert (relaxed.dtype, relaxed.to_list()) == (pl.Decimal(38, 4), [Decimal("1.5000"), Decimal("1.2345")]), revisit(
        decision
    )


def test_parquet_keeps_the_type_and_the_digits(tmp_path: Any) -> None:
    decision = "saved runs keep exact columns as Parquet"
    frame = pl.DataFrame({"x": exact(["58.899000", "-0.000001", None], 6)})
    frame.write_parquet(tmp_path / "x.parquet")
    back = pl.read_parquet(tmp_path / "x.parquet")
    assert back.schema == frame.schema, revisit(decision)
    assert back["x"].to_list() == frame["x"].to_list(), revisit(decision)
