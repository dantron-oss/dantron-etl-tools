"""Tests for the match_one_to_one tool: best match first, and nobody is matched twice.

The tool takes one input port, ``candidates``. Each row is one possible pairing of a left key with a
right key, plus any other columns (a date gap, an amount difference, an order number). It walks the
candidates in priority order and accepts a pair when neither its left key nor its right key has been
claimed yet. A join and a filter cannot say this. Whether a candidate is still available depends on
which candidates were accepted before it.

What these tests pin, and how they read the rules:

**The completed priority.** The configured priority is a list of columns, each with its own
direction, like ``sort``. The tool always adds the left key ascending and then the right key
ascending as the last two tie-breaks. So two candidates are never in an undecided order, and the
result cannot depend on the order the rows arrived in. Both output ports come out in this completed
priority order. That is what makes "shuffle the input, get identical output" true row for row.

**Two output ports.** ``accepted`` holds the pairs taken. ``rejected`` holds every other candidate.
Every input row leaves on exactly one of them, with all its columns, like ``filter``.

**The tie policy, ``ties``.**

- ``first``: the completed priority decides. Candidates that tie on every configured column fall to
  the smaller left key, then the smaller right key.
- ``refuse``: walking in the same order, the tool reaches a candidate whose left and right keys are
  both free. If another *available* candidate for the same left key has exactly the same values on
  every configured priority column, the left key is ambiguous. The same test is made for the right
  key. An ambiguous key is refused. It stays unmatched for the rest of the walk, and every candidate
  that names it is rejected, including worse ones further down. A refusal never claims the key on the
  other side of the pair, so that key stays free for other candidates.
- "Available" means neither of the rival's keys has been claimed or refused. A candidate whose other
  key was already taken is no rival. That is what lets ``refuse`` express "accept a left key, in
  order, when it has exactly one unused candidate left": the priority is the left side's order, and
  candidates already used up do not count.
- Only the configured columns count for a tie. The key tie-break never makes two candidates
  different for this purpose. A difference on any configured column is not a tie.

Vocabulary is generic on purpose: left and right keys, a gap, a difference, a sequence number.
"""

import random
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import (
    CATEGORIES,
    ConfigError,
    Ports,
    REGISTRY,
    ToolError,
    YAML_UNSAFE_VALUES,
    check_tool,
    run_tool,
)
from dantron_etl.config import expression_fields
from dantron_etl.tools.match_one_to_one import MatchOneToOne

Row = tuple[str, str, int]
Pair = tuple[str, str]

_GAP_ASC: list[dict[str, str]] = [{"column": "gap", "direction": "asc"}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidates(rows: Sequence[Row], *, value: str = "gap") -> pl.DataFrame:
    """A candidate frame: a row id, the two keys, and one priority column.

    ``row`` numbers the input rows so the conservation check can follow each one to its port.
    """
    return pl.DataFrame(
        {
            "row": list(range(len(rows))),
            "left_id": [left for left, _, _ in rows],
            "right_id": [right for _, right, _ in rows],
            value: [number for _, _, number in rows],
        },
        schema={"row": pl.Int64, "left_id": pl.String, "right_id": pl.String, value: pl.Int64},
    )


def _config(priority: Sequence[Mapping[str, str]], ties: str | None = None) -> dict[str, Any]:
    """A configuration over the ``left_id`` and ``right_id`` columns."""
    config: dict[str, Any] = {"left_key": "left_id", "right_key": "right_id", "priority": [dict(p) for p in priority]}
    if ties is not None:
        config["ties"] = ties
    return config


def _run(frame: pl.DataFrame, priority: Sequence[Mapping[str, str]], ties: str | None = None) -> Ports:
    """Run the tool, and check on every run that each input row left on exactly one port."""
    result = run_tool(MatchOneToOne(), Ports({"candidates": frame}), _config(priority, ties))

    assert result.names == ("accepted", "rejected"), "both ports are produced, accepted first"
    accepted_rows = result["accepted"]["row"].to_list()
    rejected_rows = result["rejected"]["row"].to_list()
    assert not set(accepted_rows) & set(rejected_rows), "a row may not leave on both ports"
    assert sorted(accepted_rows + rejected_rows) == sorted(frame["row"].to_list()), "every row leaves once"
    for port in ("accepted", "rejected"):
        assert result[port].schema == frame.schema, f"{port} keeps every candidate column and its type"
    return result


def _pairs(frame: pl.DataFrame) -> list[Pair]:
    """The (left, right) pairs on a port, in the port's order."""
    return list(zip(frame["left_id"].to_list(), frame["right_id"].to_list(), strict=True))


# ---------------------------------------------------------------------------
# Greedy claiming
# ---------------------------------------------------------------------------


class TestGreedyClaiming:
    """The best available candidate wins, and a key once claimed is gone."""

    def test_candidates_that_share_no_key_are_all_accepted_with_every_column_kept(self) -> None:
        """Nothing is contested, so everything is accepted, in priority order, untouched.

        The extra ``amount`` column rides along unchanged. The input frame is not modified.
        """
        frame = _candidates([("L1", "R1", 3), ("L2", "R2", 1), ("L3", "R3", 2)]).with_columns(
            amount=pl.Series([10.5, -2.0, 0.0])
        )
        before = frame.clone()

        result = _run(frame, _GAP_ASC)

        assert_frame_equal(result["accepted"], frame[[1, 2, 0]])
        assert result["rejected"].height == 0
        assert_frame_equal(frame, before)

    @pytest.mark.parametrize(
        ("rows", "accepted", "rejected"),
        [
            pytest.param(
                [("L1", "R1", 3), ("L2", "R1", 1)],
                [("L2", "R1")],
                [("L1", "R1")],
                id="a_right_key_wanted_by_two_lefts_goes_to_the_smaller_gap",
            ),
            pytest.param(
                [("L1", "R1", 1), ("L2", "R1", 3)],
                [("L1", "R1")],
                [("L2", "R1")],
                id="the_same_right_key_with_the_gaps_swapped_goes_the_other_way",
            ),
            pytest.param(
                [("L1", "R1", 3), ("L1", "R2", 1)],
                [("L1", "R2")],
                [("L1", "R1")],
                id="a_left_key_wanting_two_rights_takes_the_smaller_gap",
            ),
            pytest.param(
                [("L1", "R1", 1), ("L1", "R2", 3)],
                [("L1", "R1")],
                [("L1", "R2")],
                id="the_same_left_key_with_the_gaps_swapped_goes_the_other_way",
            ),
        ],
    )
    def test_a_contested_key_goes_to_its_higher_priority_candidate_on_either_side(
        self, rows: list[Row], accepted: list[Pair], rejected: list[Pair]
    ) -> None:
        """One key, two candidates: the better one wins. Both sides, both ways round.

        Swapping the gaps must swap the winner. A tool that kept the first row it saw would pass
        one case of each pair and fail the other.
        """
        result = _run(_candidates(rows), _GAP_ASC)

        assert _pairs(result["accepted"]) == accepted
        assert _pairs(result["rejected"]) == rejected

    def test_accepting_a_pair_pushes_its_rival_to_a_second_choice_down_a_chain(self) -> None:
        """L1 takes R1, so L2 falls to R2, so L3 falls to R3. Move L1 out and the chain unwinds.

        This is the case "keep the pairs that are unique on both sides" gets wrong: every key here
        has two candidates, yet the greedy walk matches all three lefts. In the mirror, L1's gap
        becomes the worst, so L2 keeps R1, L3 keeps R2, and L1 finds R1 taken.
        """
        chain = [("L1", "R1", 1), ("L2", "R1", 2), ("L2", "R2", 3), ("L3", "R2", 4), ("L3", "R3", 5)]

        result = _run(_candidates(chain), _GAP_ASC)

        assert _pairs(result["accepted"]) == [("L1", "R1"), ("L2", "R2"), ("L3", "R3")]
        assert _pairs(result["rejected"]) == [("L2", "R1"), ("L3", "R2")]

        unwound = [("L1", "R1", 9), *chain[1:]]

        result = _run(_candidates(unwound), _GAP_ASC)

        assert _pairs(result["accepted"]) == [("L2", "R1"), ("L3", "R2")]
        assert _pairs(result["rejected"]) == [("L2", "R2"), ("L3", "R3"), ("L1", "R1")]

    @pytest.mark.parametrize(
        ("priority", "winner"),
        [
            pytest.param([("gap", "asc")], "L3", id="smallest_gap"),
            pytest.param([("gap", "desc")], "L1", id="largest_gap_two_tie_and_the_left_key_decides"),
            pytest.param([("diff", "asc")], "L2", id="smallest_difference"),
            pytest.param([("diff", "desc")], "L3", id="largest_difference"),
            pytest.param([("gap", "asc"), ("diff", "asc")], "L3", id="the_first_column_decides_over_the_second"),
            pytest.param([("gap", "desc"), ("diff", "asc")], "L2", id="the_second_column_breaks_the_first_tie"),
            pytest.param([("gap", "desc"), ("diff", "desc")], "L1", id="the_second_column_reads_its_own_direction"),
        ],
    )
    def test_each_priority_column_reads_its_own_direction_and_later_columns_only_break_ties(
        self, priority: list[tuple[str, str]], winner: str
    ) -> None:
        """Three lefts want R1. Which one wins follows the columns, in order, each in its direction.

        L1 has gap 2 and difference 5. L2 has gap 2 and difference 1. L3 has gap 1 and difference 9.
        """
        frame = pl.DataFrame(
            {
                "row": [0, 1, 2],
                "left_id": ["L1", "L2", "L3"],
                "right_id": ["R1", "R1", "R1"],
                "gap": [2, 2, 1],
                "diff": [5, 1, 9],
            }
        )

        result = _run(frame, [{"column": column, "direction": direction} for column, direction in priority])

        assert _pairs(result["accepted"]) == [(winner, "R1")]


# ---------------------------------------------------------------------------
# Input order never matters
# ---------------------------------------------------------------------------


def _contested() -> pl.DataFrame:
    """Candidates with ties on the gap and contention on both sides.

    A tool that broke ties by arrival order would give a different answer for some shuffles. Under
    ``first`` the answer is L5-R5, L1-R1, L2-R2, L3-R3, L4-R4. Under ``refuse``:

    - L5-R5 (gap 0) is uncontested and accepted.
    - L1-R1 (gap 1) ties with L1-R2 for L1 and with L2-R1 for R1, so L1 and R1 are both refused.
    - L2-R2 (gap 2) ties with L3-R2 for R2, so R2 is refused.
    - L3-R3 (gap 3) ties with L4-R3 for R3, so R3 is refused.
    - L4-R4 (gap 4) would tie with L5-R4, but L5 is already claimed, so that is no rival.
      L4-R4 is accepted.
    """
    return _candidates(
        [
            ("L1", "R1", 1),
            ("L2", "R1", 1),
            ("L1", "R2", 1),
            ("L2", "R2", 2),
            ("L3", "R2", 2),
            ("L3", "R3", 3),
            ("L4", "R3", 3),
            ("L4", "R4", 4),
            ("L5", "R4", 4),
            ("L5", "R5", 0),
        ]
    )


class TestInputOrderIndependence:
    """The same candidates in any order give the same two ports, row for row."""

    @pytest.mark.parametrize(
        ("ties", "accepted"),
        [
            pytest.param(
                "first",
                [("L5", "R5"), ("L1", "R1"), ("L2", "R2"), ("L3", "R3"), ("L4", "R4")],
                id="first",
            ),
            pytest.param("refuse", [("L5", "R5"), ("L4", "R4")], id="refuse"),
        ],
    )
    def test_shuffling_the_input_changes_nothing_on_either_port(self, ties: str, accepted: list[Pair]) -> None:
        """Every shuffle and the reversed frame give frames identical to the original order.

        This is the property that matters most. A figure must not move because a database returned
        its rows in a different order this month.
        """
        frame = _contested()
        baseline = _run(frame, _GAP_ASC, ties)

        assert _pairs(baseline["accepted"]) == accepted

        orders = [frame.reverse()] + [frame.sample(fraction=1.0, shuffle=True, seed=seed) for seed in range(8)]
        for shuffled in orders:
            result = _run(shuffled, _GAP_ASC, ties)
            assert_frame_equal(result["accepted"], baseline["accepted"])
            assert_frame_equal(result["rejected"], baseline["rejected"])

    def test_full_ties_fall_to_the_left_key_then_the_right_key_and_both_ports_follow_that_order(self) -> None:
        """Everything at gap 0 ties. The left key decides first, then the right key, both ascending.

        The ``rejected`` port shows the key order: L1-R2 comes before L2-R1. If the right key were
        read first, L2-R1 would come first. L4-R4 (gap -1) leads and L3-R3 (gap 5) comes last,
        because the configured priority comes before the keys.
        """
        frame = _candidates(
            [("L2", "R1", 0), ("L1", "R2", 0), ("L3", "R3", 5), ("L2", "R2", 0), ("L1", "R1", 0), ("L4", "R4", -1)]
        )

        result = _run(frame, _GAP_ASC, "first")

        assert_frame_equal(result["accepted"], frame[[5, 4, 3, 2]])
        assert_frame_equal(result["rejected"], frame[[1, 0]])


# ---------------------------------------------------------------------------
# The tie policy
# ---------------------------------------------------------------------------


class TestTiePolicy:
    """``first`` resolves a tie by the keys. ``refuse`` leaves the ambiguous key unmatched."""

    @pytest.mark.parametrize(
        ("rows", "first", "refuse"),
        [
            pytest.param(
                [("L1", "R1", 1), ("L1", "R2", 1), ("L2", "R1", 5), ("L1", "R3", 9)],
                [("L1", "R1")],
                [("L2", "R1")],
                id="a_left_key_with_two_equally_good_rights",
            ),
            pytest.param(
                [("L1", "R1", 1), ("L2", "R1", 1), ("L1", "R2", 5), ("L3", "R1", 9)],
                [("L1", "R1")],
                [("L1", "R2")],
                id="a_right_key_with_two_equally_good_lefts",
            ),
        ],
    )
    def test_refuse_leaves_an_ambiguous_key_unmatched_where_first_resolves_it(
        self, rows: list[Row], first: list[Pair], refuse: list[Pair]
    ) -> None:
        """The same candidates under both policies, with the ambiguity on each side in turn.

        Left side: L1's best two (R1, R2) tie. ``first`` gives L1 the smaller key, R1. ``refuse``
        leaves L1 unmatched. L1 does not fall back to R3 either, though no one else wants R3. R1 is
        not blocked: it goes to L2, its next candidate.

        Right side, the mirror: R1's best two (L1, L2) tie. ``first`` gives R1 to L1. ``refuse``
        leaves R1 unmatched, and L3's worse candidate for R1 is rejected too. L1 is not blocked: it
        takes its second choice, R2.
        """
        frame = _candidates(rows)

        assert _pairs(_run(frame, _GAP_ASC, "first")["accepted"]) == first
        assert _pairs(_run(frame, _GAP_ASC)["accepted"]) == first, "first is the default"
        assert _pairs(_run(frame, _GAP_ASC, "refuse")["accepted"]) == refuse

    @pytest.mark.parametrize(
        ("rows", "accepted"),
        [
            pytest.param(
                [("L1", "R1", 1), ("L2", "R1", 2), ("L2", "R2", 2)],
                [("L1", "R1"), ("L2", "R2")],
                id="left_side_the_rival_right_was_claimed_first",
            ),
            pytest.param(
                [("L2", "R1", 2), ("L2", "R2", 2)],
                [],
                id="left_side_both_rights_still_free_so_refused",
            ),
            pytest.param(
                [("L1", "R1", 1), ("L1", "R2", 2), ("L2", "R2", 2)],
                [("L1", "R1"), ("L2", "R2")],
                id="right_side_the_rival_left_was_claimed_first",
            ),
            pytest.param(
                [("L1", "R2", 2), ("L2", "R2", 2)],
                [],
                id="right_side_both_lefts_still_free_so_refused",
            ),
        ],
    )
    def test_a_candidate_whose_other_key_is_already_claimed_is_no_rival(
        self, rows: list[Row], accepted: list[Pair]
    ) -> None:
        """Under ``refuse``, only candidates still available can make a tie.

        ``seq`` is the order keys are served in. L2 wants R1 and R2 equally. If L1 took R1 earlier,
        L2 has one unused candidate and takes R2. "Keep pairs unique on both sides" would refuse
        that link. If nobody took R1, L2 really is ambiguous and is refused. The same holds with
        left and right swapped.
        """
        result = _run(_candidates(rows, value="seq"), [{"column": "seq", "direction": "asc"}], "refuse")

        assert _pairs(result["accepted"]) == accepted

    def test_a_difference_on_any_priority_column_is_not_a_tie(self) -> None:
        """Two candidates for L1 share a gap. Only a shared difference as well makes them a tie.

        When the differences differ, ``refuse`` accepts the better one, as ``first`` would. When
        they match, L1 is refused. The key tie-break never counts as a difference.
        """
        priority = [{"column": "gap", "direction": "asc"}, {"column": "diff", "direction": "asc"}]
        distinct = pl.DataFrame(
            {"row": [0, 1], "left_id": ["L1", "L1"], "right_id": ["R1", "R2"], "gap": [1, 1], "diff": [2, 1]}
        )
        tied = distinct.with_columns(diff=pl.lit(1, dtype=pl.Int64))

        assert _pairs(_run(distinct, priority, "refuse")["accepted"]) == [("L1", "R2")]
        assert _pairs(_run(tied, priority, "refuse")["accepted"]) == []
        assert _pairs(_run(tied, priority, "first")["accepted"]) == [("L1", "R1")]


# ---------------------------------------------------------------------------
# Conservation and the one-to-one guarantee
# ---------------------------------------------------------------------------


def _many_candidates() -> pl.DataFrame:
    """Forty possible pairings over eight lefts and eight rights, with many ties on the gap."""
    generator = random.Random(20261110)
    seen: set[Pair] = set()
    rows: list[Row] = []
    while len(rows) < 40:
        pair = (f"L{generator.randint(1, 8)}", f"R{generator.randint(1, 8)}")
        if pair not in seen:
            seen.add(pair)
            rows.append((*pair, generator.randint(0, 3)))
    return _candidates(rows)


class TestConservation:
    """Every candidate leaves on exactly one port, and no key is accepted twice."""

    @pytest.mark.parametrize("ties", ["first", "refuse"])
    def test_every_candidate_leaves_once_and_no_key_is_accepted_twice(self, ties: str) -> None:
        """The partition is checked inside ``_run``. Here: each key at most once on ``accepted``.

        Under ``first``, nothing is left that could still have been taken. Every rejected candidate
        shares a key with an accepted pair.

        The mirror for ``refuse``: it only acts on a tie. Give every candidate its own gap, so no two
        candidates can tie, and ``refuse`` must return exactly what ``first`` returns.
        """
        frame = _many_candidates()

        result = _run(frame, _GAP_ASC, ties)

        accepted = result["accepted"]
        assert accepted["left_id"].n_unique() == accepted.height, "a left key is accepted at most once"
        assert accepted["right_id"].n_unique() == accepted.height, "a right key is accepted at most once"
        assert accepted.height > 0
        if ties == "first":
            lefts = set(accepted["left_id"].to_list())
            rights = set(accepted["right_id"].to_list())
            for left, right in _pairs(result["rejected"]):
                assert left in lefts or right in rights, f"{left}-{right} was free on both sides"
        else:
            untied = frame.with_columns(gap=pl.col("row") * 7 % 40)
            assert untied["gap"].n_unique() == untied.height
            refused = _run(untied, _GAP_ASC, "refuse")
            resolved = _run(untied, _GAP_ASC, "first")
            assert_frame_equal(refused["accepted"], resolved["accepted"])
            assert_frame_equal(refused["rejected"], resolved["rejected"])

    @pytest.mark.parametrize("ties", ["first", "refuse"])
    def test_no_candidates_give_two_empty_ports_with_the_input_columns(self, ties: str) -> None:
        """An empty frame is not an error. Both ports are empty and keep every column and type."""
        frame = _candidates([])

        result = _run(frame, _GAP_ASC, ties)

        assert result["accepted"].height == 0
        assert result["rejected"].height == 0


# ---------------------------------------------------------------------------
# The contract: configuration, refusals, ports, catalog
# ---------------------------------------------------------------------------


class TestContract:
    """Refused on the schema by name. Declared, documented and registered like every other tool."""

    @pytest.mark.parametrize(
        ("config", "error", "named"),
        [
            pytest.param({"left_key": "left_key"}, ToolError, "left_key", id="a_left_key_column_that_is_not_there"),
            pytest.param({"right_key": "right_key"}, ToolError, "right_key", id="a_right_key_column_that_is_not_there"),
            pytest.param(
                {"priority": [{"column": "gapp", "direction": "asc"}]},
                ToolError,
                "gapp",
                id="a_priority_column_that_is_not_there",
            ),
            pytest.param({"right_key": "left_id"}, ConfigError, "left_id", id="the_same_column_on_both_sides"),
            pytest.param({"ties": "maybe"}, ConfigError, "maybe", id="a_tie_policy_the_tool_does_not_offer"),
            pytest.param(
                {"priority": [{"column": "gap", "direction": "up"}]},
                ConfigError,
                "up",
                id="a_direction_the_tool_does_not_offer",
            ),
        ],
    )
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_configuration_the_schema_cannot_satisfy_is_refused_by_name(
        self, config: Mapping[str, Any], error: type[Exception], named: str, height: int
    ) -> None:
        """Each refusal quotes what was written. Rows or none, the refusal is the same."""
        frame = _candidates([("L1", "R1", 1), ("L2", "R2", 2), ("L3", "R3", 3)]).head(height)

        with pytest.raises(error) as caught:
            run_tool(MatchOneToOne(), Ports({"candidates": frame}), {**_config(_GAP_ASC), **config})

        assert named in str(caught.value), f"the refusal must name {named!r}: {caught.value}"

    def test_the_tool_declares_its_ports_and_fields_conforms_and_is_registered(self) -> None:
        """One input, two outputs. Two key columns, a priority list, and a YAML-safe tie policy.

        Conformance runs every worked example, so a documented output is a checked one.
        """
        tool = MatchOneToOne()
        template = tool.config_template()

        assert tool.name == "match_one_to_one"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("candidates",)
        assert tuple(tool.output_ports) == ("accepted", "rejected")
        assert set(template) == {"left_key", "right_key", "priority", "ties"}
        for name in ("left_key", "right_key"):
            assert template[name].type == "string", name
            assert template[name].required is True, name
        assert template["priority"].type == "array"
        assert template["priority"].required is True
        assert template["priority"].items is not None
        assert template["priority"].items.type == "object"
        assert set(template["priority"].items.fields or {}) >= {"column", "direction"}
        assert template["ties"].default == "first"
        assert set(template["ties"].enum or ()) == {"first", "refuse"}
        assert not set(template["ties"].enum or ()) & YAML_UNSAFE_VALUES
        assert expression_fields(template) == ()

        check_tool(tool)
        assert any(example.configuration.get("ties") == "refuse" for example in tool.help().examples), (
            "the refuse policy is the one most likely to surprise, so an example must show it"
        )
        assert isinstance(REGISTRY.get("match_one_to_one"), MatchOneToOne)


# ---------------------------------------------------------------------------
# Regressions, and a check against a plain row-by-row walk
# ---------------------------------------------------------------------------


Candidate = tuple[int, str | None, str | None, int | None, int | None]


def _walk_one_row_at_a_time(rows: Sequence[Candidate], ties: str) -> tuple[list[int], list[int]]:
    """The rule written out the slow, obvious way. Returns the accepted and rejected ``row`` ids.

    Each row is ``(row, left, right, gap, diff)``. The walk order is ``gap`` ascending, then
    ``diff`` descending, then the left key, then the right key. Missing values sort last. Both lists
    come back in walk order.
    """

    def last_if_missing(value: Any, flip: bool) -> tuple[bool, Any]:
        return (value is None, 0 if value is None else (-value if flip else value))

    def key_order(value: str | None) -> tuple[bool, str]:
        return (value is None, value or "")

    walk = sorted(
        rows,
        key=lambda t: (last_if_missing(t[3], False), last_if_missing(t[4], True), key_order(t[1]), key_order(t[2])),
    )
    used_left: set[str] = set()
    used_right: set[str] = set()
    accepted: list[int] = []
    for row, left, right, gap, diff in walk:
        if left is None or right is None or left in used_left or right in used_right:
            continue
        if ties == "refuse":
            open_rivals = [
                other
                for other in walk
                if other[0] != row
                and (other[3], other[4]) == (gap, diff)
                and other[1] is not None
                and other[2] is not None
                and other[1] not in used_left
                and other[2] not in used_right
            ]
            left_has_rival = any(other[1] == left for other in open_rivals)
            right_has_rival = any(other[2] == right for other in open_rivals)
            if left_has_rival:
                used_left.add(left)
            if right_has_rival:
                used_right.add(right)
            if left_has_rival or right_has_rival:
                continue
        used_left.add(left)
        used_right.add(right)
        accepted.append(row)
    return accepted, [row for row, *_ in walk if row not in accepted]


class TestAgainstAPlainWalk:
    """The tool decides many candidates at once. These check it against walking one row at a time."""

    def test_a_rival_whose_left_key_was_taken_earlier_does_not_block_a_later_pair(self) -> None:
        """Regression. L3-R2 goes first and takes L3. Then L1-R5 has no open rival, so it is accepted.

        L3-R5 ties with L1-R5 on the gap and shares R5. But L3 is already taken, so L3-R5 is no
        rival. An earlier version took one window over another inside a single expression. It got
        this wrong and refused R5, while every other test here still passed.
        """
        frame = _candidates([("L1", "R5", 2), ("L3", "R5", 2), ("L3", "R2", 0)])

        result = _run(frame, _GAP_ASC, "refuse")

        assert _pairs(result["accepted"]) == [("L3", "R2"), ("L1", "R5")]
        assert _pairs(result["rejected"]) == [("L3", "R5")]

    @pytest.mark.parametrize("ties", ["first", "refuse"])
    def test_the_tool_agrees_with_a_one_row_at_a_time_walk_on_random_candidates(self, ties: str) -> None:
        """Three thousand random candidate sets, each compared row for row with the slow walk above.

        The sets are small and crowded on purpose: few keys and few priority values, so ties are
        common. They include repeated pairs, missing keys and missing priority values. The two
        priority columns read in opposite directions, so a tie needs both to match.

        All the sets go through the tool in one frame, so the test takes seconds. Each set's keys
        carry the set's number, so no two sets share a key. Sets that share no key cannot affect
        each other, because pairs and rivals both need a shared key. The tool's answer for each set
        is then its rows on each port, in port order.
        """
        generator = random.Random(20261110)
        priority = [{"column": "gap", "direction": "asc"}, {"column": "diff", "direction": "desc"}]

        def maybe_missing(value: Any) -> Any:
            return None if generator.random() < 0.08 else value

        sets: list[list[Candidate]] = []
        next_row = 0
        for number in range(3000):
            keys = generator.randint(1, 6)
            spread = generator.randint(0, 3)
            rows: list[Candidate] = []
            for _ in range(generator.randint(0, 16)):
                left = maybe_missing(f"S{number}-L{generator.randint(1, keys)}")
                right = maybe_missing(f"S{number}-R{generator.randint(1, keys)}")
                rows.append(
                    (
                        next_row,
                        left,
                        right,
                        maybe_missing(generator.randint(0, spread)),
                        maybe_missing(generator.randint(0, 1)),
                    )
                )
                next_row += 1
            sets.append(rows)

        frame = pl.DataFrame(
            [row for rows in sets for row in rows],
            schema={"row": pl.Int64, "left_id": pl.String, "right_id": pl.String, "gap": pl.Int64, "diff": pl.Int64},
            orient="row",
        )
        result = _run(frame, priority, ties)

        set_of_row = {row: number for number, rows in enumerate(sets) for row, *_ in rows}
        answers: list[tuple[list[int], list[int]]] = [([], []) for _ in sets]
        for side, port in enumerate(("accepted", "rejected")):
            for row in result[port]["row"].to_list():
                answers[set_of_row[row]][side].append(row)
        for number, rows in enumerate(sets):
            assert answers[number] == _walk_one_row_at_a_time(rows, ties), f"set {number}, ties={ties}: {rows}"
