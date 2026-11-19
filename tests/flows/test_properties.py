"""Properties that must hold for any input, checked on many small random frames.

Each test builds a few hundred small frames from a fixed seed, runs a tool, and checks the result
against a plain-Python answer or a rule that must always hold: no row lost, no row invented, every
row on exactly one port. The frames are small and full of nulls and repeated keys, which is where
the rules are easiest to get wrong. A fixed seed keeps every run the same, so a failure can be
reproduced; the failing case is printed in the message.

No new dependency: plain ``random`` stands in for a property-testing library.
"""

import random
from collections import Counter
from typing import Any

import polars as pl
import pytest

from dantron_etl.tool import Ports, run_tool
from dantron_etl.tools.cross_tab import CrossTab
from dantron_etl.tools.filter import Filter
from dantron_etl.tools.group_by import GroupBy
from dantron_etl.tools.join import Join
from dantron_etl.tools.remove_duplicates import RemoveDuplicates
from dantron_etl.tools.sort import Sort
from dantron_etl.tools.union import Union
from dantron_etl.tools.unpivot import Unpivot

CASES = 100
KEYS = ["A", "B", "C", None]
AMOUNTS = [None, 1.0, 2.5, -4.0, 10.0]


def _frame(rng: random.Random, height: int, prefix: str = "x") -> pl.DataFrame:
    """``height`` rows: an id 0..height-1, a key from a small set with nulls, an amount with nulls."""
    return pl.DataFrame(
        {
            f"{prefix}_id": list(range(height)),
            "k": [rng.choice(KEYS) for _ in range(height)],
            f"{prefix}_amount": [rng.choice(AMOUNTS) for _ in range(height)],
        },
        schema={f"{prefix}_id": pl.Int64, "k": pl.String, f"{prefix}_amount": pl.Float64},
    )


def _show(**frames: pl.DataFrame) -> str:
    return "\n".join(f"{name} = {frame.to_dict(as_series=False)}" for name, frame in frames.items())


@pytest.mark.parametrize("how", ["inner", "left"])
def test_join_pairs_exactly_the_rows_whose_keys_are_equal_and_not_null(how: str) -> None:
    """Every (left row, right row) pair in ``joined`` is one the keys allow, and none is missing.

    A null key pairs with nothing, under either ``how``. Under ``left``, a left row that paired with
    nothing is kept once, with nothing from the right. The orphan ports hold exactly the rows that
    paired with nothing.
    """
    rng = random.Random(1)
    for _ in range(CASES):
        left, right = _frame(rng, rng.randint(0, 6), "l"), _frame(rng, rng.randint(0, 6), "r")
        result = run_tool(Join(), Ports({"left": left, "right": right}), {"keys": ["k"], "how": how})

        pairs = [
            (lid, rid)
            for lid, lk in zip(left["l_id"], left["k"], strict=True)
            for rid, rk in zip(right["r_id"], right["k"], strict=True)
            if lk is not None and lk == rk
        ]
        left_alone = [lid for lid in left["l_id"] if lid not in {pair[0] for pair in pairs}]
        right_alone = [rid for rid in right["r_id"] if rid not in {pair[1] for pair in pairs}]
        expected = Counter(pairs + ([(lid, None) for lid in left_alone] if how == "left" else []))

        got = Counter(zip(result["joined"]["l_id"].to_list(), result["joined"]["r_id"].to_list(), strict=True))
        context = _show(left=left, right=right)
        assert got == expected, f"how={how}: joined pairs {got} != {expected}\n{context}"
        assert sorted(result["left_only"]["l_id"].to_list()) == left_alone, context
        assert sorted(result["right_only"]["r_id"].to_list()) == right_alone, context


@pytest.mark.parametrize("nulls", ["exclude", "include"])
def test_filter_puts_every_row_on_exactly_one_port(nulls: str) -> None:
    """``true`` plus ``false`` is the input, row for row, and each row is on the side its value says."""
    rng = random.Random(2)
    for _ in range(CASES):
        rows = _frame(rng, rng.randint(0, 8))
        result = run_tool(Filter(), Ports({"rows": rows}), {"predicate": "[x_amount] > 0", "nulls": nulls})

        passes = [
            row_id
            for row_id, amount in zip(rows["x_id"], rows["x_amount"], strict=True)
            if (amount is None and nulls == "include") or (amount is not None and amount > 0)
        ]
        context = _show(rows=rows)
        assert result["true"]["x_id"].to_list() == passes, context
        assert sorted(result["true"]["x_id"].to_list() + result["false"]["x_id"].to_list()) == list(range(rows.height))
        assert result["true"].columns == result["false"].columns == rows.columns


def test_union_keeps_every_row_of_both_sides_in_order() -> None:
    """Height is top plus bottom, the columns are top's in top's order, and no row is changed."""
    rng = random.Random(3)
    for _ in range(CASES):
        top = _frame(rng, rng.randint(0, 5))
        bottom = _frame(rng, rng.randint(0, 5)).with_columns(pl.col("x_id") + 100)
        shuffled = bottom.select(rng.sample(bottom.columns, len(bottom.columns)))

        rows = run_tool(Union(), Ports({"top": top, "bottom": shuffled}), {})["rows"]

        assert rows.height == top.height + bottom.height
        assert rows.columns == top.columns
        assert rows.to_dicts() == top.to_dicts() + bottom.to_dicts(), _show(top=top, bottom=shuffled)


@pytest.mark.parametrize("keep", ["first", "last"])
def test_remove_duplicates_keeps_one_row_per_key_and_sets_the_rest_aside(keep: str) -> None:
    """``unique`` plus ``duplicate`` is the input; ``unique`` has one row per key, the first or last seen.

    Two nulls count as the same value here, as in ``DISTINCT``.
    """
    rng = random.Random(4)
    for _ in range(CASES):
        rows = _frame(rng, rng.randint(0, 8))
        by = rng.choice([["k"], ["k", "x_amount"]])
        result = run_tool(RemoveDuplicates(), Ports({"rows": rows}), {"by": by, "keep": keep})

        kept: dict[tuple[Any, ...], int] = {}
        for row in rows.iter_rows(named=True):
            key = tuple(row[column] for column in by)
            if keep == "last" or key not in kept:
                kept[key] = row["x_id"]
        context = _show(rows=rows) + f"\ncolumns={by}"
        assert sorted(result["unique"]["x_id"].to_list()) == sorted(kept.values()), context
        everything = result["unique"]["x_id"].to_list() + result["duplicate"]["x_id"].to_list()
        assert sorted(everything) == list(range(rows.height)), context


def test_group_by_sum_matches_a_hand_sum_per_group_and_in_total() -> None:
    """Each group's sum is the sum of its values (null when it has none); the sums add up to the total.

    Null keys form one group, as in ``GROUP BY``.
    """
    rng = random.Random(5)
    for _ in range(CASES):
        rows = _frame(rng, rng.randint(1, 8))
        config = {
            "by": ["k"],
            "aggregations": [
                {"name": "total", "function": "sum", "value": "[x_amount]"},
                {"name": "rows", "function": "count"},
            ],
        }
        grouped = run_tool(GroupBy(), Ports({"rows": rows}), config)["rows"]

        expected: dict[Any, list[float]] = {}
        for key, amount in zip(rows["k"], rows["x_amount"], strict=True):
            expected.setdefault(key, [])
            if amount is not None:
                expected[key].append(amount)
        got = {key: total for key, total in zip(grouped["k"], grouped["total"], strict=True)}
        context = _show(rows=rows)
        assert got == {key: (sum(values) if values else None) for key, values in expected.items()}, context
        assert grouped["rows"].sum() == rows.height, context
        assert (grouped["total"].sum() or 0.0) == (rows["x_amount"].sum() or 0.0), context


def test_unpivot_then_cross_tab_gives_back_the_wide_frame() -> None:
    """Long and back again: every cell, null cells included, returns to its row and column."""
    rng = random.Random(6)
    for _ in range(CASES // 2):
        height, width = rng.randint(1, 5), rng.randint(1, 4)
        periods = [f"p{n}" for n in range(1, width + 1)]
        wide = pl.DataFrame(
            {
                "k": rng.sample(["A", "B", "C", "D", "E"], height),
                **{period: [rng.choice(AMOUNTS) for _ in range(height)] for period in periods},
            },
            schema={"k": pl.String, **dict.fromkeys(periods, pl.Float64)},
        )

        long = run_tool(Unpivot(), Ports({"rows": wide}), {"by": ["k"], "names_to": "period", "values_to": "amount"})
        back = run_tool(
            CrossTab(),
            Ports({"rows": long["rows"]}),
            {"by": ["k"], "names_from": "period", "values_from": "amount", "function": "sum"},
        )["rows"]

        assert long["rows"].height == height * width, _show(wide=wide)
        assert back.select(wide.columns).sort("k").to_dicts() == wide.sort("k").to_dicts(), _show(wide=wide)


def test_sort_is_a_stable_permutation_in_the_order_asked_for() -> None:
    """Every row comes back once; rows with equal keys keep their order; nulls go where they are told."""
    rng = random.Random(7)
    for _ in range(CASES):
        rows = _frame(rng, rng.randint(0, 8))
        direction, nulls = rng.choice(["asc", "desc"]), rng.choice(["first", "last"])
        result = run_tool(
            Sort(), Ports({"rows": rows}), {"by": [{"column": "k", "direction": direction, "nulls": nulls}]}
        )["rows"]

        if direction == "desc" and rows.height > 1 and rows["k"].null_count() == rows.height:
            continue  # the one case where order is not kept; pinned in the test below

        present = sorted(
            (pair for pair in zip(rows["k"], rows["x_id"], strict=True) if pair[0] is not None),
            key=lambda pair: pair[0],
            reverse=direction == "desc",
        )
        missing = [pair for pair in zip(rows["k"], rows["x_id"], strict=True) if pair[0] is None]
        expected = [row_id for _, row_id in (missing + present if nulls == "first" else present + missing)]
        assert result["x_id"].to_list() == expected, _show(rows=rows) + f"\ndirection={direction} nulls={nulls}"


# pins current behaviour; see the stage-2 test report (new finding, not in report F): with a single
# sort key whose column is entirely null, a descending sort reverses the rows instead of keeping
# them in arrival order. polars 1.38 does this even with maintain_order=True. A second key, or one
# non-null value, restores the arrival order. record_id numbers rows after this sort.
@pytest.mark.parametrize("nulls", ["first", "last"])
def test_a_descending_sort_on_an_all_null_key_reverses_the_rows(nulls: str) -> None:
    rows = pl.DataFrame({"x_id": [0, 1, 2], "k": [None, None, None]}, schema={"x_id": pl.Int64, "k": pl.String})

    result = run_tool(Sort(), Ports({"rows": rows}), {"by": [{"column": "k", "direction": "desc", "nulls": nulls}]})

    assert result["rows"]["x_id"].to_list() == [2, 1, 0]
