"""Tests for the match_one_to_many tool: one whole, and the one set of parts that adds up to it.

The tool takes one input port, ``candidates``. Each row is one possible pairing of a whole (a bank
deposit, say) with a part (one payment it might contain), with the whole's amount and the part's
amount on the row. For each whole the tool looks at every set of 1 to ``max_parts`` of its parts and
keeps the sets whose total is within ``tolerance`` of the whole. Exactly one such set is a match.
Anything else is set aside for a person, with a reason. Nothing is guessed.

Two output ports. ``matched`` holds the rows of the chosen sets with the input columns only.
``rejected`` holds every other row with the reason added as the last column. Both keep input order,
and every input row leaves on exactly one of them.

What these tests pin:

- The worked example from the requirement, row for row.
- Each reason, next to the case that just avoids it: no set adds up, more than one set adds up, not
  in the matched set, a part claimed by more than one whole, missing key, missing amount, more
  candidates than the search allows.
- ``max_parts`` deciding the answer both ways, and ``tolerance`` included at its edge and not past it.
- Zero and negative amounts. A zero part makes a second set. A set has at least one part, so a whole
  of zero is not matched by "no parts".
- Exact decimals of different places compared by value, with nothing rounded; and whole numbers.
- Data errors (the same pair twice, a whole or a part with two amounts) next to the clean version.
- Settings refused by name, on a frame with rows and on one without.
- The answer does not depend on the order the rows arrive in, and agrees with a plain brute-force
  reading of the rules on random small inputs.

The tool is reached only through ``get_tool("match_one_to_many")`` and ``run_tool``.
"""

import random
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from itertools import combinations
from typing import Any, Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl import get_tool
from dantron_etl.authoring import CATEGORIES, ConfigError, DataError, Ports, Tool, check_tool, run_tool

TOOL_NAME: Final = "match_one_to_many"

# The verdict for a row, as these tests write it: "matched", or the reason on ``rejected``.
MATCHED: Final = "matched"
NO_SET: Final = "no set adds up"
MANY_SETS: Final = "more than one set adds up"
OUTSIDE: Final = "not in the matched set"
CLAIMED: Final = "a part is claimed by more than one whole"
MISSING_KEY: Final = "missing key"
MISSING_AMOUNT: Final = "missing amount"
TOO_MANY: Final = "more candidates than the search allows"

MONEY: Final = pl.Decimal(12, 2)
WHOLE_NUMBER: Final = pl.Int64

# (deposit id, deposit amount, payment id, payment amount, expected verdict)
Amount = str | int | None
Case = tuple[str | None, Amount, str | None, Amount, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool() -> Tool:
    return get_tool(TOOL_NAME)


def _amount(value: Amount, dtype: pl.DataType) -> Any:
    """A Decimal for an exact column, the value itself otherwise. Strings keep their places."""
    if value is None:
        return None
    return Decimal(str(value)) if isinstance(dtype, pl.Decimal) else value


def _frame(
    cases: Sequence[Case],
    *,
    whole_type: pl.DataType = MONEY,
    part_type: pl.DataType = MONEY,
) -> pl.DataFrame:
    """A candidate frame. ``row`` numbers the input rows so each can be followed to its port."""
    return pl.DataFrame(
        {
            "row": list(range(len(cases))),
            "deposit_id": [case[0] for case in cases],
            "deposit_amount": [_amount(case[1], whole_type) for case in cases],
            "payment_id": [case[2] for case in cases],
            "payment_amount": [_amount(case[3], part_type) for case in cases],
        },
        schema={
            "row": pl.Int64,
            "deposit_id": pl.String,
            "deposit_amount": whole_type,
            "payment_id": pl.String,
            "payment_amount": part_type,
        },
    )


def _config(max_parts: int = 3, **extra: Any) -> dict[str, Any]:
    return {
        "whole_key": "deposit_id",
        "part_key": "payment_id",
        "whole_amount": "deposit_amount",
        "part_amount": "payment_amount",
        "max_parts": max_parts,
        "reason_column": "reason",
        **extra,
    }


def _run(frame: pl.DataFrame, config: dict[str, Any]) -> Ports:
    """Run the tool and check the shape every run must have.

    Both ports exist, matched first. Every input row leaves once. ``matched`` has exactly the input
    columns, ``rejected`` the input columns then a String reason. Each port keeps input order.
    """
    result = run_tool(_tool(), Ports({"candidates": frame}), config)

    assert result.names == ("matched", "rejected"), "both ports are produced, matched first"
    matched, rejected = result["matched"], result["rejected"]
    matched_rows = matched["row"].to_list()
    rejected_rows = rejected["row"].to_list()
    assert not set(matched_rows) & set(rejected_rows), "a row may not leave on both ports"
    assert sorted(matched_rows + rejected_rows) == sorted(frame["row"].to_list()), "every row leaves once"

    assert matched.schema == frame.schema, "matched carries the input columns only, with their types"
    reason = config["reason_column"]
    assert list(rejected.schema.items()) == [*frame.schema.items(), (reason, pl.String)], (
        "rejected carries the input columns, then the reason as text"
    )

    position = {row: index for index, row in enumerate(frame["row"].to_list())}
    for port, rows in (("matched", matched_rows), ("rejected", rejected_rows)):
        places = [position[row] for row in rows]
        assert places == sorted(places), f"{port} keeps the input order"
    return result


def _verdicts(result: Ports, reason: str = "reason") -> dict[int, str]:
    """Each input ``row`` id, mapped to ``matched`` or to its reason."""
    verdicts = {row: MATCHED for row in result["matched"]["row"].to_list()}
    rejected = result["rejected"]
    verdicts.update(zip(rejected["row"].to_list(), rejected[reason].to_list(), strict=True))
    return verdicts


def _check(
    cases: Sequence[Case],
    *,
    whole_type: pl.DataType = MONEY,
    part_type: pl.DataType = MONEY,
    **config: Any,
) -> None:
    """Run ``cases`` and compare each row's verdict with the one written beside it."""
    frame = _frame(cases, whole_type=whole_type, part_type=part_type)
    result = _run(frame, _config(**config))
    got = _verdicts(result)
    assert [got[row] for row in range(len(cases))] == [case[4] for case in cases]


# The requirement's worked example, with max_parts 3.
WORKED_EXAMPLE: Final[list[Case]] = [
    ("W1", "250.00", "A", "100.00", MATCHED),
    ("W1", "250.00", "B", "150.00", MATCHED),
    ("W1", "250.00", "E", "40.00", OUTSIDE),
    ("W2", "300.00", "C", "200.00", MANY_SETS),
    ("W2", "300.00", "D", "100.00", MANY_SETS),
    ("W2", "300.00", "F", "150.00", MANY_SETS),
    ("W2", "300.00", "G", "150.00", MANY_SETS),
    ("W3", "90.00", "H", "45.00", NO_SET),
    ("W3", "90.00", "I", "40.00", NO_SET),
]


# ---------------------------------------------------------------------------
# Rule 1: one set adds up, none does, or more than one does
# ---------------------------------------------------------------------------


class TestChoosingTheSet:
    """For each whole, every set of 1 to max_parts parts is tried. Only a single set that adds up is a match."""

    def test_the_worked_example_gives_the_documented_ports_exactly(self) -> None:
        """W1 has one set (A+B). W2 has two (C+D, F+G). W3 has none. E is W1's leftover."""
        frame = _frame(WORKED_EXAMPLE)

        result = _run(frame, _config(max_parts=3))

        assert_frame_equal(result["matched"], frame[[0, 1]])
        expected_rejected = frame[2:].with_columns(
            reason=pl.Series([OUTSIDE, MANY_SETS, MANY_SETS, MANY_SETS, MANY_SETS, NO_SET, NO_SET], dtype=pl.String)
        )
        assert_frame_equal(result["rejected"], expected_rejected)

    @pytest.mark.parametrize(
        ("max_parts", "verdict"),
        [
            pytest.param(2, NO_SET, id="a_set_of_three_is_out_of_reach_when_max_parts_is_two"),
            pytest.param(3, MATCHED, id="the_same_set_of_three_matches_when_max_parts_is_three"),
        ],
    )
    def test_max_parts_decides_whether_a_large_set_counts(self, max_parts: int, verdict: str) -> None:
        """300 is made only of all three payments of 100."""
        _check(
            [
                ("W1", "300.00", "P", "100.00", verdict),
                ("W1", "300.00", "Q", "100.00", verdict),
                ("W1", "300.00", "R", "100.00", verdict),
            ],
            max_parts=max_parts,
        )

    @pytest.mark.parametrize(
        ("max_parts", "verdicts"),
        [
            pytest.param(1, [MATCHED, OUTSIDE, OUTSIDE, OUTSIDE], id="max_parts_one_sees_only_the_single_payment"),
            pytest.param(3, [MANY_SETS] * 4, id="max_parts_three_also_sees_the_three_hundreds"),
        ],
    )
    def test_max_parts_decides_whether_a_second_set_makes_the_whole_ambiguous(
        self, max_parts: int, verdicts: list[str]
    ) -> None:
        """300 is A alone, and also B+C+D. Only the second needs three parts."""
        amounts = [("A", "300.00"), ("B", "100.00"), ("C", "100.00"), ("D", "100.00")]
        _check(
            [
                ("W1", "300.00", part, amount, verdict)
                for (part, amount), verdict in zip(amounts, verdicts, strict=True)
            ],
            max_parts=max_parts,
        )


# ---------------------------------------------------------------------------
# Rule 2: a part claimed by two wholes
# ---------------------------------------------------------------------------


class TestClaimedParts:
    """A part in the chosen set of two wholes sinks both wholes entirely, with no second try."""

    def test_a_part_in_two_chosen_sets_rejects_every_row_of_both_wholes(self) -> None:
        """W1 chooses A+B; W2 chooses B. Both lose all their rows, E and F included. W3 is untouched."""
        _check(
            [
                ("W1", "250.00", "A", "100.00", CLAIMED),
                ("W1", "250.00", "B", "150.00", CLAIMED),
                ("W1", "250.00", "E", "40.00", CLAIMED),
                ("W2", "150.00", "B", "150.00", CLAIMED),
                ("W2", "150.00", "F", "70.00", CLAIMED),
                ("W3", "90.00", "H", "90.00", MATCHED),
            ]
        )

    def test_a_part_that_is_a_candidate_of_two_wholes_but_chosen_by_one_is_not_a_claim(self) -> None:
        """B is a candidate of both. W2's only set is F+G, so B is W1's alone and both wholes match."""
        _check(
            [
                ("W1", "250.00", "A", "100.00", MATCHED),
                ("W1", "250.00", "B", "150.00", MATCHED),
                ("W2", "200.00", "B", "150.00", OUTSIDE),
                ("W2", "200.00", "F", "70.00", MATCHED),
                ("W2", "200.00", "G", "130.00", MATCHED),
            ]
        )

    def test_an_ambiguous_whole_has_no_chosen_set_and_so_claims_nothing(self) -> None:
        """W2 adds up as B or as G, so it is set aside; that does not count as a claim on B."""
        _check(
            [
                ("W1", "250.00", "A", "100.00", MATCHED),
                ("W1", "250.00", "B", "150.00", MATCHED),
                ("W2", "150.00", "B", "150.00", MANY_SETS),
                ("W2", "150.00", "G", "150.00", MANY_SETS),
            ]
        )


# ---------------------------------------------------------------------------
# Rule 3: missing keys and amounts
# ---------------------------------------------------------------------------


class TestMissingValues:
    """A row with a missing key or a missing part amount is rejected and takes no part in any search."""

    @pytest.mark.parametrize(
        "cases",
        [
            pytest.param(
                [
                    ("W1", "250.00", "A", "100.00", MATCHED),
                    ("W1", "250.00", "B", "150.00", MATCHED),
                    ("W1", "250.00", None, "250.00", MISSING_KEY),
                ],
                id="a_missing_part_key_is_not_searched_so_it_cannot_make_a_second_set",
            ),
            pytest.param(
                [
                    (None, "250.00", "A", "100.00", MISSING_KEY),
                    (None, "250.00", "B", "150.00", MISSING_KEY),
                    ("W1", "100.00", "A", "100.00", MATCHED),
                ],
                id="rows_with_a_missing_whole_key_are_not_grouped_into_a_whole_of_their_own",
            ),
            pytest.param(
                [
                    ("W1", "250.00", "A", "100.00", MATCHED),
                    ("W1", "250.00", "B", "150.00", MATCHED),
                    ("W1", "250.00", "C", None, MISSING_AMOUNT),
                ],
                id="a_missing_part_amount_is_not_read_as_zero",
            ),
            pytest.param(
                [
                    ("W1", "250.00", "A", "100.00", MANY_SETS),
                    ("W1", "250.00", "B", "150.00", MANY_SETS),
                    ("W1", "250.00", "C", "0.00", MANY_SETS),
                ],
                id="a_part_amount_of_zero_is_a_value_and_makes_a_second_set",
            ),
            pytest.param(
                [
                    ("W1", None, "A", "100.00", MISSING_AMOUNT),
                    ("W1", None, "B", "150.00", MISSING_AMOUNT),
                    ("W2", "150.00", "C", "150.00", MATCHED),
                ],
                id="a_whole_with_no_amount_loses_all_its_rows_and_the_next_whole_still_matches",
            ),
        ],
    )
    def test_a_missing_value_is_rejected_by_name_and_never_searched(self, cases: list[Case]) -> None:
        _check(cases, max_parts=3)


# ---------------------------------------------------------------------------
# Rule 4: the search limit
# ---------------------------------------------------------------------------


class TestSearchLimit:
    """A whole with more candidates than max_candidates is not searched at all."""

    @pytest.mark.parametrize(
        ("max_candidates", "verdicts"),
        [
            pytest.param(2, [TOO_MANY] * 3, id="three_candidates_over_a_limit_of_two_are_not_searched"),
            pytest.param(3, [MATCHED, MATCHED, OUTSIDE], id="three_candidates_at_a_limit_of_three_are_searched"),
        ],
    )
    def test_a_whole_over_the_limit_is_set_aside_and_one_under_it_still_matches(
        self, max_candidates: int, verdicts: list[str]
    ) -> None:
        cases: list[Case] = [
            ("W1", "250.00", "A", "100.00", verdicts[0]),
            ("W1", "250.00", "B", "150.00", verdicts[1]),
            ("W1", "250.00", "E", "40.00", verdicts[2]),
            ("W2", "90.00", "H", "90.00", MATCHED),
        ]
        _check(cases, max_candidates=max_candidates)

    @pytest.mark.parametrize(
        ("count", "verdict_of_the_pair"),
        [
            pytest.param(25, MATCHED, id="twenty_five_candidates_are_searched_by_default"),
            pytest.param(26, TOO_MANY, id="twenty_six_candidates_are_over_the_default_limit"),
        ],
    )
    def test_the_default_limit_is_twenty_five_candidates(self, count: int, verdict_of_the_pair: str) -> None:
        """Only 1 + 2 adds up to 3; every other payment is far larger."""
        amounts = [1, 2, *range(1000, 1000 + count - 2)]
        outside = OUTSIDE if verdict_of_the_pair == MATCHED else TOO_MANY
        cases: list[Case] = [
            ("W1", 3, f"P{index:02d}", amount, verdict_of_the_pair if index < 2 else outside)
            for index, amount in enumerate(amounts)
        ]
        _check(cases, whole_type=WHOLE_NUMBER, part_type=WHOLE_NUMBER, max_parts=2)


# ---------------------------------------------------------------------------
# Rule 5: data errors
# ---------------------------------------------------------------------------


class TestDataErrors:
    """Rows that contradict each other stop the step, naming them. The clean version runs."""

    @pytest.mark.parametrize(
        ("broken", "clean", "named"),
        [
            pytest.param(
                [("W1", "250.00", "A", "100.00"), ("W1", "250.00", "A", "100.00")],
                [("W1", "250.00", "A", "100.00"), ("W1", "250.00", "B", "100.00")],
                "W1",
                id="the_same_whole_and_part_twice",
            ),
            pytest.param(
                [("W1", "250.00", "A", "100.00"), ("W1", "260.00", "B", "150.00")],
                [("W1", "250.00", "A", "100.00"), ("W1", "250.00", "B", "150.00")],
                "W1",
                id="one_whole_with_two_different_amounts",
            ),
            pytest.param(
                [("W1", "250.00", "A", "100.00"), ("W2", "100.00", "A", "90.00")],
                [("W1", "250.00", "A", "100.00"), ("W2", "100.00", "A", "100.00")],
                "A",
                id="one_part_with_two_different_amounts",
            ),
        ],
    )
    def test_contradictory_rows_stop_the_step_and_consistent_ones_run(
        self,
        broken: list[tuple[str, str, str, str]],
        clean: list[tuple[str, str, str, str]],
        named: str,
    ) -> None:
        with pytest.raises(DataError) as caught:
            run_tool(_tool(), Ports({"candidates": _frame([(*row, "") for row in broken])}), _config())
        assert named in str(caught.value), f"the error must name {named!r}: {caught.value}"

        _run(_frame([(*row, "") for row in clean]), _config())


# ---------------------------------------------------------------------------
# Rule 6: zero, negative and tolerance; exact and whole-number amounts
# ---------------------------------------------------------------------------


class TestAmounts:
    """Amounts are compared by value, exactly, in their own units."""

    @pytest.mark.parametrize(
        ("max_parts", "verdicts"),
        [
            pytest.param(3, [MANY_SETS] * 3, id="a_zero_payment_can_join_the_set_so_two_sets_add_up"),
            pytest.param(2, [MATCHED, MATCHED, OUTSIDE], id="with_room_for_two_parts_only_the_zero_cannot_join"),
        ],
    )
    def test_a_zero_part_makes_a_second_set_when_there_is_room_for_it(
        self, max_parts: int, verdicts: list[str]
    ) -> None:
        _check(
            [
                ("W1", "250.00", "A", "100.00", verdicts[0]),
                ("W1", "250.00", "B", "150.00", verdicts[1]),
                ("W1", "250.00", "Z", "0.00", verdicts[2]),
            ],
            max_parts=max_parts,
        )

    def test_negative_amounts_and_a_zero_whole_add_up_like_any_other(self) -> None:
        """A refund nets against a payment. A whole of zero needs at least one part: no parts is not a set."""
        _check(
            [
                ("W1", "100.00", "A", "150.00", MATCHED),
                ("W1", "100.00", "R", "-50.00", MATCHED),
                ("W1", "100.00", "B", "70.00", OUTSIDE),
                ("W2", "-50.00", "S", "-50.00", MATCHED),
                ("W2", "-50.00", "T", "30.00", OUTSIDE),
                ("W3", "0.00", "U", "25.00", MATCHED),
                ("W3", "0.00", "V", "-25.00", MATCHED),
                ("W4", "0.00", "X", "25.00", NO_SET),
            ]
        )

    @pytest.mark.parametrize(
        ("whole_type", "part_type", "second", "verdict"),
        [
            pytest.param(
                pl.Decimal(12, 2), pl.Decimal(12, 3), "149.875", MATCHED, id="three_places_adding_up_to_two_places"
            ),
            pytest.param(
                pl.Decimal(12, 2), pl.Decimal(12, 3), "149.874", NO_SET, id="a_thousandth_short_is_not_rounded_away"
            ),
            pytest.param(
                pl.Decimal(12, 4), pl.Decimal(12, 3), "149.875", MATCHED, id="four_places_against_three_places"
            ),
        ],
    )
    def test_exact_amounts_of_different_places_are_compared_by_value(
        self, whole_type: pl.DataType, part_type: pl.DataType, second: str, verdict: str
    ) -> None:
        """100.125 + 149.875 is 250 exactly. 100.125 + 149.874 is 249.999, which is not 250.00."""
        _check(
            [
                ("W1", "250.00", "A", "100.125", verdict),
                ("W1", "250.00", "B", second, verdict),
            ],
            whole_type=whole_type,
            part_type=part_type,
        )

    def test_whole_number_amounts_follow_the_same_rules(self) -> None:
        cases: list[Case] = [
            ("W1", 250, "A", 100, MATCHED),
            ("W1", 250, "B", 150, MATCHED),
            ("W1", 250, "E", 40, OUTSIDE),
            ("W2", 300, "C", 200, MANY_SETS),
            ("W2", 300, "D", 100, MANY_SETS),
            ("W2", 300, "F", 150, MANY_SETS),
            ("W2", 300, "G", 150, MANY_SETS),
            ("W3", 90, "H", 45, NO_SET),
            ("W3", 90, "I", 40, NO_SET),
        ]
        _check(cases, whole_type=WHOLE_NUMBER, part_type=WHOLE_NUMBER)

    @pytest.mark.parametrize(
        ("second", "tolerance", "verdict"),
        [
            pytest.param("39.99", 0.01, MATCHED, id="short_by_exactly_the_tolerance_adds_up"),
            pytest.param("40.01", 0.01, MATCHED, id="over_by_exactly_the_tolerance_adds_up"),
            pytest.param("39.98", 0.01, NO_SET, id="short_by_more_than_the_tolerance_does_not"),
            pytest.param("40.02", 0.01, NO_SET, id="over_by_more_than_the_tolerance_does_not"),
            pytest.param("39.99", None, NO_SET, id="the_default_tolerance_is_zero"),
            pytest.param("40.00", None, MATCHED, id="with_no_tolerance_an_exact_total_adds_up"),
        ],
    )
    def test_tolerance_is_inclusive_and_in_the_amounts_own_units(
        self, second: str, tolerance: float | None, verdict: str
    ) -> None:
        extra = {} if tolerance is None else {"tolerance": tolerance}
        _check(
            [
                ("W1", "100.00", "A", "60.00", verdict),
                ("W1", "100.00", "B", second, verdict),
            ],
            **extra,
        )

    @pytest.mark.parametrize(
        ("tolerance", "verdicts"),
        [
            pytest.param(0, [MATCHED, MATCHED, OUTSIDE], id="exact_only_one_set_adds_up"),
            pytest.param(1, [MANY_SETS] * 3, id="a_tolerance_of_one_lets_a_second_set_add_up"),
        ],
    )
    def test_a_tolerance_can_make_a_whole_ambiguous(self, tolerance: int, verdicts: list[str]) -> None:
        """60 + 40 is 100; 60 + 41 is 101, within a tolerance of 1."""
        _check(
            [
                ("W1", 100, "A", 60, verdicts[0]),
                ("W1", 100, "B", 40, verdicts[1]),
                ("W1", 100, "C", 41, verdicts[2]),
            ],
            whole_type=WHOLE_NUMBER,
            part_type=WHOLE_NUMBER,
            tolerance=tolerance,
        )


# ---------------------------------------------------------------------------
# Rule 7: empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    @pytest.mark.parametrize("amount_type", [MONEY, WHOLE_NUMBER], ids=["exact_amounts", "whole_number_amounts"])
    def test_no_candidates_give_two_empty_ports_with_the_right_columns(self, amount_type: pl.DataType) -> None:
        frame = _frame([], whole_type=amount_type, part_type=amount_type)

        result = _run(frame, _config())

        assert result["matched"].height == 0
        assert result["rejected"].height == 0


# ---------------------------------------------------------------------------
# Rule 8: input order does not change the answer
# ---------------------------------------------------------------------------


# One of everything, each whole on its own keys, with the verdict each row must get.
MIXED: Final[list[Case]] = [
    *WORKED_EXAMPLE,
    ("W4", "250.00", "A4", "100.00", CLAIMED),
    ("W4", "250.00", "B4", "150.00", CLAIMED),
    ("W5", "150.00", "B4", "150.00", CLAIMED),
    ("W5", "150.00", "F5", "70.00", CLAIMED),
    (None, "50.00", "X6", "50.00", MISSING_KEY),
    ("W6", "50.00", None, "50.00", MISSING_KEY),
    ("W6", "50.00", "Y6", "50.00", MATCHED),
    ("W7", "10.00", "Z7", None, MISSING_AMOUNT),
    ("W8", "250.00", "A8", "100.00", MANY_SETS),
    ("W8", "250.00", "B8", "150.00", MANY_SETS),
    ("W8", "250.00", "Z8", "0.00", MANY_SETS),
    ("W9", "100.00", "A9", "150.00", MATCHED),
    ("W9", "100.00", "R9", "-50.00", MATCHED),
    ("W9", "100.00", "B9", "70.00", OUTSIDE),
]


class TestInputOrder:
    """Shuffling the rows moves them on the ports but changes no row's verdict."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, None], ids=lambda s: "reversed" if s is None else f"seed_{s}")
    def test_every_row_gets_the_same_verdict_whatever_order_the_rows_arrive_in(self, seed: int | None) -> None:
        frame = _frame(MIXED)
        shuffled = frame.reverse() if seed is None else frame.sample(fraction=1.0, shuffle=True, seed=seed)

        result = _run(shuffled, _config(max_parts=3))

        expected = {row: case[4] for row, case in enumerate(MIXED)}
        assert _verdicts(result) == expected


# ---------------------------------------------------------------------------
# Against a plain brute-force reading of rules 1, 2 and 4
# ---------------------------------------------------------------------------


def _brute_force(
    pairs: Sequence[tuple[str, int, str, int]], *, max_parts: int, tolerance: int, max_candidates: int
) -> list[str]:
    """The rules written out the slow, obvious way, for rows with no missing values. One verdict per row."""
    by_whole: dict[str, list[int]] = {}
    for index, (whole, *_rest) in enumerate(pairs):
        by_whole.setdefault(whole, []).append(index)

    verdicts: list[str] = [""] * len(pairs)
    chosen: dict[str, tuple[int, ...]] = {}
    for whole, rows in by_whole.items():
        if len(rows) > max_candidates:
            for row in rows:
                verdicts[row] = TOO_MANY
            continue
        target = pairs[rows[0]][1]
        sets = [
            subset
            for size in range(1, max_parts + 1)
            for subset in combinations(rows, size)
            if abs(sum(pairs[row][3] for row in subset) - target) <= tolerance
        ]
        if not sets:
            for row in rows:
                verdicts[row] = NO_SET
        elif len(sets) > 1:
            for row in rows:
                verdicts[row] = MANY_SETS
        else:
            chosen[whole] = sets[0]

    claims = Counter(pairs[row][2] for subset in chosen.values() for row in subset)
    for whole, subset in chosen.items():
        claimed = any(claims[pairs[row][2]] > 1 for row in subset)
        for row in by_whole[whole]:
            verdicts[row] = CLAIMED if claimed else (MATCHED if row in subset else OUTSIDE)
    return verdicts


class TestAgainstBruteForce:
    def test_the_tool_agrees_with_a_brute_force_search_on_random_small_inputs(self) -> None:
        """Two hundred random frames: few wholes, shared parts, small amounts including zero and negatives.

        Each frame is crowded on purpose, so every reason turns up. A part has one amount and a whole
        one amount everywhere, and no pair repeats, so no frame is a data error.
        """
        generator = random.Random(20261126)
        for number in range(200):
            wholes = {f"W{w}": generator.randint(-2, 12) for w in range(generator.randint(1, 4))}
            parts = {f"P{p}": generator.randint(-3, 8) for p in range(generator.randint(1, 7))}
            pairs = [
                (whole, whole_amount, part, parts[part])
                for whole, whole_amount in wholes.items()
                for part in generator.sample(sorted(parts), generator.randint(1, len(parts)))
            ]
            settings = {
                "max_parts": generator.randint(1, 4),
                "tolerance": generator.choice([0, 0, 1]),
                "max_candidates": generator.choice([3, 25]),
            }
            frame = _frame([(*pair, "") for pair in pairs], whole_type=WHOLE_NUMBER, part_type=WHOLE_NUMBER).sample(
                fraction=1.0, shuffle=True, seed=number
            )

            result = _run(frame, _config(**settings))

            expected = _brute_force(pairs, **settings)
            got = _verdicts(result)
            assert [got[row] for row in range(len(pairs))] == expected, f"frame {number}, {settings}: {pairs}"


# ---------------------------------------------------------------------------
# Settings: refusals and the declared contract
# ---------------------------------------------------------------------------


class TestSettings:
    """Settings the frame cannot satisfy are refused by name before any row is read."""

    @pytest.mark.parametrize(
        ("change", "named"),
        [
            pytest.param(
                {"deposit_amount": pl.Float64}, ["deposit_amount", "cast_columns", "Decimal"], id="a_float_whole_amount"
            ),
            pytest.param(
                {"payment_amount": pl.Float64}, ["payment_amount", "cast_columns", "Decimal"], id="a_float_part_amount"
            ),
            pytest.param({"payment_amount": pl.String}, ["payment_amount"], id="a_text_part_amount"),
        ],
    )
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_an_amount_that_is_not_exact_is_refused_with_the_way_to_fix_it(
        self, change: dict[str, pl.DataType], named: list[str], height: int
    ) -> None:
        frame = (
            _frame(WORKED_EXAMPLE[:3])
            .head(height)
            .with_columns(pl.col(column).cast(dtype) for column, dtype in change.items())
        )

        with pytest.raises(ConfigError) as caught:
            run_tool(_tool(), Ports({"candidates": frame}), _config())

        for word in named:
            assert word in str(caught.value), f"the refusal must name {word!r}: {caught.value}"

    @pytest.mark.parametrize(
        ("change", "named"),
        [
            pytest.param({"tolerance": -1}, "tolerance", id="a_negative_tolerance"),
            pytest.param({"tolerance": -0.01}, "tolerance", id="a_tolerance_just_below_zero"),
            pytest.param({"max_parts": 0}, "max_parts", id="max_parts_below_one"),
            pytest.param({"max_candidates": 0}, "max_candidates", id="max_candidates_below_one"),
            pytest.param({"part_key": "deposit_id"}, "deposit_id", id="the_same_column_for_both_keys"),
            pytest.param({"reason_column": "payment_id"}, "payment_id", id="a_reason_column_the_input_already_has"),
            pytest.param({"whole_key": "deposit"}, "deposit", id="a_key_column_that_is_not_there"),
            pytest.param({"part_amount": "amount"}, "amount", id="an_amount_column_that_is_not_there"),
        ],
    )
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_setting_out_of_range_or_clashing_is_refused_by_name(
        self, change: dict[str, Any], named: str, height: int
    ) -> None:
        frame = _frame(WORKED_EXAMPLE[:3]).head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(_tool(), Ports({"candidates": frame}), {**_config(), **change})

        assert named in str(caught.value), f"the refusal must name {named!r}: {caught.value}"

    @pytest.mark.parametrize(
        "change",
        [
            pytest.param({"tolerance": 0}, id="a_tolerance_of_zero"),
            pytest.param({"max_parts": 1}, id="max_parts_of_one"),
            pytest.param({"max_candidates": 1}, id="max_candidates_of_one"),
        ],
    )
    def test_the_lowest_allowed_value_of_each_setting_is_accepted(self, change: dict[str, Any]) -> None:
        _run(_frame(WORKED_EXAMPLE[:3]), {**_config(), **change})

    def test_the_tool_declares_its_ports_and_settings_conforms_and_is_registered(self) -> None:
        """Conformance runs the tool's own worked examples, so what its help shows is checked."""
        tool = _tool()
        template = tool.config_template()

        assert tool.name == TOOL_NAME
        assert tool.category == "preparation"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("candidates",)
        assert tuple(tool.output_ports) == ("matched", "rejected")
        assert set(template) == {
            "whole_key",
            "part_key",
            "whole_amount",
            "part_amount",
            "max_parts",
            "tolerance",
            "max_candidates",
            "reason_column",
        }
        for name in ("whole_key", "part_key", "whole_amount", "part_amount", "reason_column"):
            assert template[name].type == "string", name
            assert template[name].required is True, name
        assert template["max_parts"].type == "integer"
        assert template["max_parts"].required is True
        assert template["tolerance"].type == "number"
        assert template["tolerance"].default == 0
        assert template["max_candidates"].type == "integer"
        assert template["max_candidates"].default == 25

        check_tool(tool)
