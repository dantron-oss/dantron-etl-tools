"""Tests for the Join tool: two inputs, three outputs, and no join-type option between them.

Join takes two input ports, ``left`` and ``right``, and produces three: ``matched``, carrying the
paired rows, and ``left_only`` / ``right_only``, carrying the rows of each side that failed to pair.

**There is no ``join_type``, and that is the design.** The tool this one is derived from required an
enum of inner/left/right/outer. With three named outputs those four stop being configuration and
become wiring: a left join is whoever reads ``matched`` and ``left_only``, an outer join reads all
three, an anti-join reads ``left_only`` alone, and an inner join reads ``matched`` and ignores the
rest. What that buys is the orphan check for free -- the rows a join failed to pair are a port a
flow can route to a report or an assertion, rather than a number a reader reconstructs afterwards
with a second join they have to remember to write.

Six decisions are pinned here beyond the happy path, because each has a plausible wrong answer:

**Fan-out is real.** A key appearing twice on the right fans one left row into two output rows, so
``matched.height`` is not ``left.height`` and never was. It is pinned with counts and with the
identity of the left row that was copied, so that nobody later "fixes" this into a first-match-only
join -- which would look tidier on a spot check and would silently drop money.

**A null key never pairs.** This is the *opposite* of the ruling RemoveDuplicates makes for the
same-looking question, and both are right: SQL's ``DISTINCT`` treats two nulls as one group, SQL's
``JOIN`` treats them as unpairable. A row whose key is missing is a row nothing is known about, and
pairing it with another such row would invent a relationship from two absences. Because it never
pairs, it leaves on its own side's orphan port, which is exactly where a reader wants to find it.
Two tools answering the same-looking question differently is a genuine hazard, so the ruling has to
be written down in the tool's own notes, not only here -- this file checks that it is.

**Row accounting.** A join has no conservation law as simple as Filter's, because fan-out breaks
addition: ``matched.height + left_only.height == left.height`` is false the moment a key repeats.
The law that does hold is about rows, not counts: every left row appears in ``matched`` *at least*
once or in ``left_only`` *exactly* once, and never in both; likewise for the right. Pinned with an
identity column per side, in the presence of fan-out, where the naive statement fails.

**Row order is deterministic, and that is all it is.** ``matched`` is in left-input order, and
within one left row its fan-out copies are in right-input order; each orphan port is in its own
side's input order. polars promises none of that on its own, so this costs the tool an explicit
order-maintaining join and the parallelism it gives up. It is worth it because a flow that writes a
file needs a diff that does not churn between runs. It is *not* a licence to depend on order for
correctness: a step that needs a meaningful order still needs an explicit Sort in front of it. The
catalog's own example checker ignores row order by default, so the promise is pinned in this file;
where the checker offers an opt-in, the fan-out example should take it, because the order of a
fanned-out copy is exactly what a reader of that example is trying to learn.

**Collisions and the suffix.** A right-hand column whose name collides with a left-hand one takes
the suffix; the key columns themselves do not double up, because the pair is one column after the
join; and a collision the suffix cannot resolve -- because the *suffixed* name is taken too -- is
refused in this library's own words, off the schema, rather than surfacing as an engine error about
a duplicate column. The orphan ports carry their own side's columns only: no suffix, no phantom
columns from the other side, because they are input rows that failed to pair, not join results.

**Which name survives a differently-named key pair.** ``left_on``/``right_on`` leave the left
spelling on ``matched`` and the right key column does not appear at all. A reader cannot guess that,
so it is pinned.

The cross-field rule -- give either ``keys``, or *both* of ``left_on`` and ``right_on``, never both
forms and never half of the second -- is the first rule in this catalog that a per-field template
cannot express, so its shape is the pattern the remaining tools copy. The trap it has to survive is
that ``validate_config`` receives the configuration with defaults already filled: every template key
is always present, ``"keys" in config`` is always true, and ``config.get("keys", fallback)`` never
returns the fallback. The only correct spelling is ``config["keys"] is None``, and the tests below
call ``validate_config`` on a filled configuration directly, as well as through ``run_tool``, so an
implementation reaching for ``in`` fails here rather than in a flow.

Vocabulary is deliberately generic -- entries and headers, credits and originals, movements and
nettings -- because the library carries no caller's nouns. These are the shapes of three real join
steps: detail rows paired to their header on one key (with the fan-out above), a reversing row
paired to the row it reverses on a normalised name plus an account code plus a quantity -- three
keys, and deliberately not the amount -- and a period-by-period netting whose two sides spell the
period column differently.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    YAML_UNSAFE_VALUES,
    ConfigError,
    Ports,
    apply_defaults,
    check_registry,
    check_tool,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.join import Join

# ---------------------------------------------------------------------------
# Test data -- the shapes the three real join steps run against
# ---------------------------------------------------------------------------


def _entries() -> pl.DataFrame:
    """Detail rows keyed to a header. ``entry_id`` is the identity; ``group_id`` is the join key.

    Entries 1 and 2 share group ``G1``, which the right side carries twice -- that is the fan-out.
    Entry 4's group is on no header at all, so it is the left orphan.
    """
    return pl.DataFrame(
        {
            "entry_id": [1, 2, 3, 4],
            "group_id": ["G1", "G1", "G2", "G4"],
            "amount": [10.0, 20.0, 30.0, 40.0],
        }
    )


def _headers() -> pl.DataFrame:
    """The header side. ``G1`` appears twice on purpose; ``G3`` pairs with nothing."""
    return pl.DataFrame(
        {
            "header_id": [10, 11, 12, 13],
            "group_id": ["G1", "G1", "G2", "G3"],
            "state": ["open", "closed", "open", "open"],
        }
    )


def _credits() -> pl.DataFrame:
    """A reversing row, keyed on a normalised name, an account code and a quantity -- not an amount.

    ``amount`` is deliberately present on both sides and deliberately *not* a key: the pair below
    matches on three columns whose amounts are nothing like each other, which is the whole point of
    leaving the amount out of the key. It is also the column that collides and takes the suffix.
    """
    return pl.DataFrame(
        {
            "credit_id": [501, 502, 503],
            "name_key": ["acme", "acme", "globex"],
            "account_code": ["4000", "4000", "5000"],
            "quantity": [2, 3, 1],
            "amount": [-100.0, -60.0, -25.0],
        }
    )


def _originals() -> pl.DataFrame:
    """The rows being reversed. ``original_id`` 903 pairs with nothing, so it is the right orphan."""
    return pl.DataFrame(
        {
            "original_id": [901, 902, 903],
            "name_key": ["acme", "globex", "initech"],
            "account_code": ["4000", "5000", "6000"],
            "quantity": [2, 1, 4],
            "amount": [120.0, 25.0, 80.0],
        }
    )


def _movements() -> pl.DataFrame:
    """The left side of a period netting; its period column is spelled ``period_start``."""
    return pl.DataFrame(
        {
            "movement_id": [71, 72],
            "period_start": ["2024-01", "2024-03"],
            "amount": [100.0, 300.0],
        }
    )


def _nettings() -> pl.DataFrame:
    """The right side of a period netting; the same period is spelled ``period`` here."""
    return pl.DataFrame(
        {
            "netting_id": [81, 82],
            "period": ["2024-01", "2024-02"],
            "offset": [-40.0, -20.0],
        }
    )


_THREE_KEYS = ["name_key", "account_code", "quantity"]


def _run(left: pl.DataFrame, right: pl.DataFrame, config: Mapping[str, Any]) -> Ports:
    """Join ``left`` to ``right`` through the boundary, which is the only way a tool is ever run."""
    return run_tool(Join(), Ports({"left": left, "right": right}), config)


def _take(frame: pl.DataFrame, positions: Sequence[int]) -> pl.DataFrame:
    """The rows of ``frame`` at ``positions``, in input order and keeping the schema when empty."""
    return frame[list(positions)] if positions else frame.head(0)


def _shape_cases() -> list[Any]:
    """(left, right, config, left identity, right identity) for each shape the law must hold over."""
    return [
        pytest.param(
            _entries(),
            _headers(),
            {"keys": ["group_id"]},
            "entry_id",
            "header_id",
            id="one_shared_key_with_fan_out_on_the_right",
        ),
        pytest.param(
            _credits(),
            _originals(),
            {"keys": _THREE_KEYS},
            "credit_id",
            "original_id",
            id="three_shared_keys_and_a_colliding_non_key_column",
        ),
        pytest.param(
            _movements(),
            _nettings(),
            {"left_on": ["period_start"], "right_on": ["period"]},
            "movement_id",
            "netting_id",
            id="key_columns_named_differently_on_each_side",
        ),
        pytest.param(
            pl.DataFrame({"entry_id": [1, 5], "group_id": ["G1", None], "amount": [10.0, 50.0]}),
            pl.DataFrame({"header_id": [10, 14], "group_id": ["G1", None], "state": ["open", "open"]}),
            {"keys": ["group_id"]},
            "entry_id",
            "header_id",
            id="a_missing_key_on_each_side",
        ),
        pytest.param(
            _entries(),
            pl.DataFrame({"header_id": [20], "group_id": ["G9"], "state": ["open"]}),
            {"keys": ["group_id"]},
            "entry_id",
            "header_id",
            id="nothing_pairs_at_all",
        ),
        pytest.param(
            _entries().head(0),
            _headers(),
            {"keys": ["group_id"]},
            "entry_id",
            "header_id",
            id="no_rows_on_the_left_at_all",
        ),
    ]


# ---------------------------------------------------------------------------
# Pairing: fan-out, the accounting law, and where a null key goes
# ---------------------------------------------------------------------------


class TestJoinPairing:
    """Which rows pair, how many times, and which port the rows that did not pair leave on."""

    def test_a_key_repeated_on_the_right_fans_one_left_row_into_a_copy_per_match(self) -> None:
        """One left row, two matching right rows, two output rows -- pinned with counts and identities.

        This is the behaviour a later reader is most likely to mistake for a defect and "fix" into a
        first-match-only join, which would satisfy a spot check on the first row and silently drop
        every second and subsequent match. Comparing the whole frame pins four things at once: which
        pairs, how many copies of each left row, the row order (left order, and right order within
        one left row's copies), and the column order and dtypes.
        """
        left, right = _entries(), _headers()

        result = _run(left, right, {"keys": ["group_id"]})

        assert result.names == ("matched", "left_only", "right_only"), "three ports, matched first"
        assert_frame_equal(
            result["matched"],
            pl.DataFrame(
                {
                    "entry_id": [1, 1, 2, 2, 3],
                    "group_id": ["G1", "G1", "G1", "G1", "G2"],
                    "amount": [10.0, 10.0, 20.0, 20.0, 30.0],
                    "header_id": [10, 11, 10, 11, 12],
                    "state": ["open", "closed", "open", "closed", "open"],
                }
            ),
        )
        assert result["matched"].height == 5, "three left rows paired, two of them twice over"
        assert result["matched"].height != left.height, "matched.height is not left.height, and never was"
        assert result["matched"]["entry_id"].to_list() == [1, 1, 2, 2, 3], "a copy per match, not one per row"
        assert result["matched"]["amount"].sum() == 90.0, "a fanned-out amount is repeated, not halved"

        repeat = _run(left, right, {"keys": ["group_id"]})
        assert_frame_equal(repeat["matched"], result["matched"]), "the row order is stable across runs"

    @pytest.mark.parametrize(("left", "right", "config", "left_id", "right_id"), _shape_cases())
    def test_every_row_is_paired_at_least_once_or_orphaned_exactly_once_and_never_both(
        self,
        left: pl.DataFrame,
        right: pl.DataFrame,
        config: Mapping[str, Any],
        left_id: str,
        right_id: str,
    ) -> None:
        """The law that survives fan-out, stated over rows rather than over counts.

        Addition does not hold: ``matched.height + left_only.height == left.height`` is false in the
        first case below, where a key repeats on the right. What holds is that every row of a side
        is accounted for exactly once as *paired or not* -- it is somewhere in ``matched``, possibly
        several times, or exactly once in its own orphan port, never both and never neither. An
        implementation that derived an orphan port from a left join and a null test would break this
        on the fourth case, where a key is genuinely missing, and one that dropped a fanned-out row's
        duplicates would break it on the first.
        """
        result = _run(left, right, config)

        for frame, identity, orphan_port in ((left, left_id, "left_only"), (right, right_id, "right_only")):
            everything = set(frame[identity].to_list())
            paired = result["matched"][identity].to_list()
            orphaned = result[orphan_port][identity].to_list()

            assert set(paired) | set(orphaned) == everything, f"{orphan_port}: no row may go missing"
            assert not (set(paired) & set(orphaned)), f"{orphan_port}: no row may be both paired and orphaned"
            assert len(orphaned) == len(set(orphaned)), f"{orphan_port}: an orphan appears exactly once"
            assert len(set(paired)) + len(orphaned) == frame.height, f"{orphan_port}: every row is accounted for"
            assert len(paired) >= len(set(paired)), "a paired row may appear more than once -- that is fan-out"

        assert result["matched"].columns[: left.width] == left.columns, "the left side's columns come first, in order"
        assert result["matched"].width >= left.width, "an empty matched port still carries its schema"

    def test_a_null_key_never_pairs_and_leaves_on_its_own_sides_orphan_port(self) -> None:
        """Sameness follows ``JOIN``, not ``DISTINCT``, and the tool's own notes have to say so.

        A missing key is a row nothing is known about; pairing two of them would invent a
        relationship out of two absences, and on dirty data it would fan every null-keyed row on one
        side across every null-keyed row on the other. Because such a row never pairs, it lands on
        its own orphan port, which is where a reader looking for what failed to pair will find it.
        This is deliberately the opposite of RemoveDuplicates' ruling on the same-looking question,
        so a reader of a configuration cannot infer it and the tool must write it down.
        """
        left = pl.DataFrame({"entry_id": [1, 5], "group_id": ["G1", None], "amount": [10.0, 50.0]})
        right = pl.DataFrame({"header_id": [10, 14], "group_id": ["G1", None], "state": ["open", "open"]})

        result = _run(left, right, {"keys": ["group_id"]})

        assert result["matched"]["entry_id"].to_list() == [1], "only the row with a key pairs"
        assert result["matched"]["header_id"].to_list() == [10]
        assert result["left_only"]["entry_id"].to_list() == [5], "a null key on the left is a left orphan"
        assert result["right_only"]["header_id"].to_list() == [14], "a null key on the right is a right orphan"

        help_ = Join().help()
        documented = " ".join((help_.detailed_description, *help_.notes, *help_.field_help.values())).lower()
        assert "null" in documented, "a reader of the config cannot guess this, so it must be written down"
        assert any("null" in note.lower() for note in help_.notes), (
            "the ruling contradicts another tool in the same catalog, so it belongs in the notes a reader scans"
        )

    def test_the_orphan_ports_carry_their_own_side_only_with_no_suffix_and_no_phantom_columns(self) -> None:
        """An orphan is an input row that failed to pair, not a join result with nulls in it.

        So each orphan port has exactly its own side's schema: no suffixed column, no column from the
        other side, and nothing renamed. An implementation that produced these from a left join and a
        null filter would carry the other side's columns as nulls -- and would also mistake a
        genuinely null value in a matched row for a failure to pair.
        """
        left, right = _credits(), _originals()

        result = _run(left, right, {"keys": _THREE_KEYS})

        assert_frame_equal(result["left_only"], _take(left, [1]))
        assert_frame_equal(result["right_only"], _take(right, [2]))
        assert result["left_only"].schema == left.schema, "an orphan port keeps its own side's schema"
        assert result["right_only"].schema == right.schema, "an orphan port keeps its own side's schema"
        assert "original_id" not in result["left_only"].columns, "no phantom column from the other side"
        assert "amount_right" not in result["left_only"].columns, "an unpaired row has nothing to suffix"
        assert "credit_id" not in result["right_only"].columns, "no phantom column from the other side"

        every_row_pairs = _run(left.head(1), right.head(1), {"keys": _THREE_KEYS})

        assert every_row_pairs["matched"].height == 1
        assert every_row_pairs["left_only"].height == 0
        assert every_row_pairs["right_only"].height == 0
        assert every_row_pairs["left_only"].schema == left.schema, "an empty orphan port is still typed"
        assert every_row_pairs["right_only"].schema == right.schema, "an empty orphan port is still typed"

        several_orphans = _run(
            pl.DataFrame({"entry_id": [9, 8, 7], "group_id": ["Z9", "G1", "Z7"], "amount": [90.0, 80.0, 70.0]}),
            pl.DataFrame({"header_id": [10, 11, 12, 13], "group_id": ["G1", "G3", "G1", "G2"], "state": ["o"] * 4}),
            {"keys": ["group_id"]},
        )

        assert several_orphans["left_only"]["entry_id"].to_list() == [9, 7], "left input order, not key order"
        assert several_orphans["right_only"]["header_id"].to_list() == [11, 13], "right input order, not key order"

    def test_join_leaves_both_of_its_inputs_untouched(self) -> None:
        """The boundary compares input schemas; the frames themselves must be unchanged too.

        Renaming a colliding column on the right in place would satisfy the boundary's schema check
        on the left port and corrupt the right port for every other branch reading it.
        """
        left, right = _credits(), _originals()
        before_left, before_right = left.clone(), right.clone()

        _run(left, right, {"keys": _THREE_KEYS})

        assert_frame_equal(left, before_left)
        assert_frame_equal(right, before_right)


# ---------------------------------------------------------------------------
# Columns: which name survives, what takes the suffix, and what cannot be resolved
# ---------------------------------------------------------------------------


class TestJoinColumns:
    """What ``matched`` is called, column by column, since a reader cannot guess any of it."""

    @pytest.mark.parametrize(
        ("config", "suffixed"),
        [
            pytest.param({"keys": _THREE_KEYS}, "amount_right", id="the_default_suffix"),
            pytest.param({"keys": _THREE_KEYS, "suffix": "_original"}, "amount_original", id="a_suffix_of_ones_own"),
        ],
    )
    def test_a_colliding_right_column_takes_the_suffix_and_the_key_columns_do_not_double_up(
        self,
        config: Mapping[str, Any],
        suffixed: str,
    ) -> None:
        """The key pair is one column after the join; a colliding non-key column becomes two.

        Both halves matter. Suffixing the keys as well would leave a ``matched`` frame carrying
        ``name_key`` and ``name_key_right`` holding identical values, which every downstream step
        would then have to choose between. Failing to suffix the non-key collision loses one side's
        value entirely -- here the amounts on the two sides are deliberately unrelated, because the
        amount is not part of the key, so silently keeping one of them would be a wrong number
        rather than a duplicated one.
        """
        result = _run(_credits(), _originals(), config)
        matched = result["matched"]

        assert matched.columns == [
            "credit_id",
            "name_key",
            "account_code",
            "quantity",
            "amount",
            "original_id",
            suffixed,
        ], "left columns in left order, then the right side's non-key columns in right order"
        assert matched["credit_id"].to_list() == [501, 503]
        assert matched["amount"].to_list() == [-100.0, -25.0], "the left side keeps its own name"
        assert matched[suffixed].to_list() == [120.0, 25.0], "the right side's value survives under the suffix"
        for key in _THREE_KEYS:
            assert matched.columns.count(key) == 1, f"the key {key!r} is one column, not two"
            assert f"{key}_right" not in matched.columns
            assert key + str(config.get("suffix", "_right")) not in matched.columns

    def test_only_the_left_spelling_of_a_differently_named_key_pair_survives(self) -> None:
        """``left_on``/``right_on`` leave the left name on ``matched``; the right key is gone.

        polars drops one of the two and a reader cannot guess which, so a flow written against the
        wrong guess fails one step later with a missing column. The orphan ports are unaffected:
        each keeps its own side's own spelling, because neither is a join result.
        """
        left, right = _movements(), _nettings()

        result = _run(left, right, {"left_on": ["period_start"], "right_on": ["period"]})

        assert result["matched"].columns == ["movement_id", "period_start", "amount", "netting_id", "offset"]
        assert "period" not in result["matched"].columns, "the right key column does not survive at all"
        assert "period_right" not in result["matched"].columns, "nor does it survive under the suffix"
        assert result["matched"]["period_start"].to_list() == ["2024-01"]
        assert result["matched"]["netting_id"].to_list() == [81]
        assert result["left_only"].columns == left.columns, "an orphan keeps its own side's spelling"
        assert result["right_only"].columns == right.columns, "including the spelling the join dropped"
        assert result["left_only"]["movement_id"].to_list() == [72]
        assert result["right_only"]["netting_id"].to_list() == [82]

    @pytest.mark.parametrize("height", [2, 0], ids=["frames_with_rows", "frames_with_no_rows"])
    def test_a_collision_the_suffix_cannot_resolve_is_refused_in_this_librarys_own_words(self, height: int) -> None:
        """When the suffixed name is taken too, the tool must refuse rather than let the engine.

        Left alone, this reaches the query engine as an error about a duplicate column, from a
        library the author has never heard of, naming neither the configuration field to change nor
        the fact that changing ``suffix`` is the fix. Running it against frames with no rows at all
        is what shows the refusal is read off the schema before any row is touched.
        """
        left = pl.DataFrame({"code": ["A", "B"], "amount": [1.0, 2.0], "amount_right": [9.0, 8.0]}).head(height)
        right = pl.DataFrame({"code": ["A", "B"], "amount": [3.0, 4.0]}).head(height)

        with pytest.raises(ConfigError) as caught:
            _run(left, right, {"keys": ["code"]})

        message = str(caught.value)
        assert "amount_right" in message, f"the refusal must name the column that cannot be renamed: {message}"
        assert "suffix" in message, f"the refusal must name the field that fixes it: {message}"


# ---------------------------------------------------------------------------
# The key specification: the first cross-field rule in this catalog
# ---------------------------------------------------------------------------


def _refused_key_specifications() -> list[Any]:
    """(config, the names its refusal must quote, whether the tool's own rule must be the refuser).

    Only a *combination* of fields is beyond what a template can say, so only those cases require
    the refusal to come from the tool's own ``validate_config``. An empty list is a single field's
    business, so where that refusal is written stays the implementation's choice -- what is not
    negotiable is that a configuration naming no key columns never reaches ``execute``.
    """
    return [
        pytest.param({}, ("keys", "left_on", "right_on"), True, id="no_key_specification_at_all"),
        pytest.param({"left_on": ["group_id"]}, ("right_on",), True, id="half_of_the_two_field_form_left_only"),
        pytest.param({"right_on": ["group_id"]}, ("left_on",), True, id="half_of_the_two_field_form_right_only"),
        pytest.param(
            {"keys": ["group_id"], "left_on": ["group_id"]}, ("keys", "left_on"), True, id="both_forms_with_left_on"
        ),
        pytest.param(
            {"keys": ["group_id"], "right_on": ["group_id"]}, ("keys", "right_on"), True, id="both_forms_with_right_on"
        ),
        pytest.param(
            {"keys": ["group_id"], "left_on": ["group_id"], "right_on": ["group_id"]},
            ("keys", "left_on", "right_on"),
            True,
            id="both_forms_in_full",
        ),
        pytest.param(
            {"left_on": ["group_id"], "right_on": ["group_id", "state"]},
            ("left_on", "right_on"),
            True,
            id="a_key_list_longer_on_one_side_than_the_other",
        ),
        pytest.param({"keys": []}, ("keys",), False, id="an_empty_shared_key_list"),
        pytest.param({"left_on": [], "right_on": []}, ("left_on",), False, id="two_empty_key_lists"),
    ]


def _accepted_key_specifications() -> list[Any]:
    """The only three shapes a key specification may take."""
    return [
        pytest.param({"keys": ["group_id"]}, id="one_shared_key"),
        pytest.param({"keys": ["group_id", "state"]}, id="several_shared_keys"),
        pytest.param({"left_on": ["group_id"], "right_on": ["group_id"]}, id="both_of_the_two_field_form"),
    ]


def _missing_key_cases() -> list[Any]:
    """(config, the column its refusal must name, the side it must name) for a key that is not there."""
    return [
        pytest.param({"keys": ["group_id", "state"]}, "state", "left", id="a_shared_key_absent_from_the_left"),
        pytest.param({"keys": ["group_id", "amount"]}, "amount", "right", id="a_shared_key_absent_from_the_right"),
        pytest.param({"keys": ["GROUP_ID"]}, "GROUP_ID", None, id="a_shared_key_whose_case_is_wrong"),
        pytest.param(
            {"left_on": ["entry"], "right_on": ["group_id"]},
            "entry",
            "left",
            id="a_left_key_that_is_not_there",
        ),
        pytest.param(
            {"left_on": ["group_id"], "right_on": ["group"]},
            "group",
            "right",
            id="a_right_key_that_is_not_there",
        ),
    ]


class TestJoinKeySpecification:
    """Either ``keys``, or both of ``left_on`` and ``right_on`` -- and nothing else."""

    @pytest.mark.parametrize(("config", "named", "own_rule"), _refused_key_specifications())
    def test_a_key_specification_that_is_neither_form_or_both_forms_is_refused(
        self,
        config: Mapping[str, Any],
        named: tuple[str, ...],
        own_rule: bool,
    ) -> None:
        """The cross-field rule, from both directions, on a configuration with its defaults filled.

        Each of the three key fields is individually optional and individually valid, so only the
        *combination* can be wrong and a per-field template cannot express it. The trap the rule has
        to survive is that it runs after defaults are applied: every template key is present, so
        ``"keys" in config`` is always true and ``config.get("keys", fallback)`` never returns the
        fallback. The ``left_on``-only case is the one that catches that spelling. The refusal is
        checked both through ``run_tool`` and, for the cases only a hand-written rule can express,
        by calling ``validate_config`` on the filled configuration directly -- the second is what
        every later tool in this catalog will copy. An empty key list is a single field's business
        rather than a combination, so it only has to be refused somewhere before ``execute``.
        """
        template = Join().config_template()
        filled = apply_defaults(template, config)
        assert set(filled) == {"keys", "left_on", "right_on", "suffix", "how"}, (
            "defaults are filled before validate_config runs, so membership tests are always true"
        )

        with pytest.raises(ConfigError) as through_boundary:
            _run(_entries(), _headers(), config)
        refusals = [through_boundary]

        if own_rule:
            with pytest.raises(ConfigError) as called_directly:
                Join().validate_config(filled)
            refusals.append(called_directly)

        for caught in refusals:
            message = str(caught.value)
            for name in named:
                assert name in message, f"the refusal must name {name!r}: {message}"

    @pytest.mark.parametrize("config", _accepted_key_specifications())
    def test_each_of_the_three_valid_key_specifications_is_accepted(self, config: Mapping[str, Any]) -> None:
        """A rule that refuses too much is as broken as one that refuses too little.

        All three shapes below name real columns on both sides, so nothing here may raise, and each
        must produce all three ports -- including the ``keys``-with-several-keys shape, whose second
        key (``state``) exists only on the right in the frames used elsewhere in this file and so is
        supplied on both sides here.
        """
        left = _entries().with_columns(pl.lit("open").alias("state"))
        right = _headers()

        result = _run(left, right, config)

        assert result.names == ("matched", "left_only", "right_only")
        assert Join().validate_config(apply_defaults(Join().config_template(), config)) is None

    @pytest.mark.parametrize(("config", "named", "side"), _missing_key_cases())
    @pytest.mark.parametrize("height", [4, 0], ids=["frames_with_rows", "frames_with_no_rows"])
    def test_a_key_column_that_is_not_on_its_own_side_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        side: str | None,
        height: int,
    ) -> None:
        """A key column is checked against the schema of the side it is meant to come from.

        ``state`` exists -- on the right -- and ``amount`` exists on the left, so a check that
        merely looked for the name somewhere would let both through, to fail inside the query
        engine. The refusal has to name the offending column, the field it was written in, and
        which side it is missing from, because a two-input tool gives an author two places to look.
        Running each case against frames with no rows at all shows the refusal is read off the
        schema rather than off the data.
        """
        left, right = _entries().head(height), _headers().head(height)

        with pytest.raises(ConfigError) as caught:
            _run(left, right, config)

        message = str(caught.value)
        assert named in message, f"the refusal must name {named!r}: {message}"
        assert any(field in message for field in ("keys", "left_on", "right_on")), (
            f"the refusal must name the config field to open: {message}"
        )
        if side is not None:
            assert side in message, f"the refusal must say which side {named!r} is missing from: {message}"

    def test_the_template_declares_three_optional_typed_key_lists_and_a_default_suffix(self) -> None:
        """No ``join_type``: four join types became three output ports, so the option is gone.

        The three key fields are individually optional because each is genuinely valid on its own
        terms -- which is precisely why the cross-field rule has to exist -- and each declares
        ``string`` elements, because an array with no declared element type checks nothing inside
        itself. A ``column_reference`` field type is planned and deliberately absent here, so a key
        list is an array of plain strings and there is no expression seam on this tool at all.
        """
        template = Join().config_template()

        assert set(template) == {"keys", "left_on", "right_on", "suffix", "how"}

        # This reverses an earlier ruling on purpose, and the measurement is the argument. The ports
        # were meant to make every join kind a wiring choice, and for an anti-join they do: read
        # `left_only` and there is nothing to configure. But a LEFT join is `matched` plus the
        # unmatched left rows with the right-hand columns nulled, which the ports alone cannot state,
        # and the first real pipeline converted to these tools wanted sixteen of them against one
        # anti-join. The design made the rare case free and the common case a two-step workaround
        # with a schema to align by hand. `how` is only about which rows reach `matched`; the orphan
        # ports keep answering their own question either way.
        assert set(template["how"].enum or ()) == {"inner", "left"}
        assert template["how"].default == "inner", "a join that drops unmatched rows is the safer default"
        assert not set(template["how"].enum or ()) & YAML_UNSAFE_VALUES, "an enum has to survive a definition file"
        assert "right" not in (template["how"].enum or ()), "a right join is the same join with the inputs swapped"
        for name in ("keys", "left_on", "right_on"):
            assert template[name].type == "array", name
            assert template[name].required is False, f"{name} alone is a valid thing to leave out"
            assert template[name].default is None, f"{name} must default to None so the rule can test for it"
            assert template[name].items is not None, f"{name}: an array with no element type checks nothing"
            assert template[name].items.type == "string", name
        assert template["suffix"].type == "string"
        assert template["suffix"].default == "_right"
        assert template["suffix"].enum is None, "any suffix an author needs is legitimate"
        assert expression_fields(template) == (), "a key column is not a condition; there is no seam here"


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestJoinInTheCatalog:
    """Described, documented with examples that actually run over all three ports, resolvable by name."""

    def test_join_conforms_and_its_examples_document_all_three_ports(self) -> None:
        """Conformance runs every example over every declared port, so an orphan port cannot drift.

        With three outputs, an example that documented only ``matched`` would have been *run* rather
        than *checked* -- and the orphan ports are both the least-read and the likeliest to be wrong.
        Two things an example must actually demonstrate rather than merely describe: fan-out, whose
        row count is the fact a reader will not believe until they see it, and a differently-named
        key pair, whose surviving column name cannot be guessed.
        """
        tool = Join()

        check_tool(tool)

        assert tool.name == "join"
        assert tool.category == "combine"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("left", "right")
        assert tuple(tool.output_ports) == ("matched", "left_only", "right_only")

        examples = tool.help().examples
        assert len(examples) >= 2, "one shared-key join, and one whose key columns are named differently"
        for example in examples:
            assert set(example.inputs.names) == {"left", "right"}
            assert set(example.outputs.names) == {"matched", "left_only", "right_only"}, (
                "three ports means three documented outputs, or the example has been run and not checked"
            )
        assert any(example.outputs["matched"].height > example.inputs["left"].height for example in examples), (
            "fan-out must be demonstrated by a worked example, not merely described"
        )
        assert any(example.configuration.get("left_on") is not None for example in examples), (
            "which name survives a differently-named key pair cannot be guessed, so it must be run"
        )
        assert any(
            example.outputs["left_only"].height > 0 and example.outputs["right_only"].height > 0 for example in examples
        ), "the orphan ports are the reason there is no join_type, so an example must fill both"

    def test_join_is_in_the_shipped_catalog_and_the_catalog_still_conforms(self) -> None:
        """Registered under its machine name, and adding it leaves the whole catalog conformant."""
        assert "join" in REGISTRY
        assert isinstance(REGISTRY.get("join"), Join)

        check_registry(REGISTRY)


class TestHowDecidesWhichRowsReachMatched:
    """`how` is the one thing the orphan ports cannot say: keep the rows that paired, or keep them all.

    The ports already answer "which rows found nothing" on either side, which is what makes an
    anti-join free. What they cannot express is a LEFT join -- every left row, with the right-hand
    columns null where nothing paired -- because that is one frame built from two of them, with a
    schema to align. Sixteen of those against one anti-join in the first real pipeline converted to
    these tools is why this field exists.
    """

    def test_left_keeps_a_row_that_paired_with_nothing_and_nulls_the_right_hand_columns(self) -> None:
        """Entry 4 sits on a group no header carries, and must survive with nulls rather than vanish."""
        result = run_tool(
            Join(),
            Ports({"left": _entries(), "right": _headers()}),
            {"keys": ["group_id"], "how": "left"},
        )
        matched = result["matched"]

        assert matched["entry_id"].to_list() == [1, 1, 2, 2, 3, 4], "every left row, fan-out included"
        orphan = matched.filter(pl.col("entry_id") == 4)
        assert orphan.height == 1
        assert orphan["header_id"].to_list() == [None], "nothing paired, so the right-hand column is null"
        assert orphan["state"].to_list() == [None]
        assert orphan["amount"].to_list() == [40.0], "the left row's own values are untouched"

    def test_inner_is_the_default_and_drops_the_row_that_paired_with_nothing(self) -> None:
        """A join that keeps only what paired is the safer thing to get by not saying anything."""
        stated = run_tool(
            Join(), Ports({"left": _entries(), "right": _headers()}), {"keys": ["group_id"], "how": "inner"}
        )
        silent = run_tool(Join(), Ports({"left": _entries(), "right": _headers()}), {"keys": ["group_id"]})

        assert 4 not in stated["matched"]["entry_id"].to_list()
        assert silent["matched"].equals(stated["matched"]), "leaving `how` out must mean 'inner', not something else"

    def test_both_modes_produce_the_same_columns_in_the_same_order(self) -> None:
        """So a step reading this port names the same columns whichever way the join was configured.

        The modes differ in which rows arrive and never in what a later step may refer to -- which is
        what lets `how` be changed without breaking anything downstream of it.
        """
        inner = run_tool(Join(), Ports({"left": _entries(), "right": _headers()}), {"keys": ["group_id"]})
        left = run_tool(Join(), Ports({"left": _entries(), "right": _headers()}), {"keys": ["group_id"], "how": "left"})

        assert left["matched"].columns == inner["matched"].columns
        assert left["matched"].schema == inner["matched"].schema

    def test_the_orphan_ports_answer_the_same_question_under_either_mode(self) -> None:
        """`left_only` is the diagnostic and keeps reporting, even when `matched` already carries them."""
        inner = run_tool(Join(), Ports({"left": _entries(), "right": _headers()}), {"keys": ["group_id"]})
        left = run_tool(Join(), Ports({"left": _entries(), "right": _headers()}), {"keys": ["group_id"], "how": "left"})

        assert left["left_only"].equals(inner["left_only"])
        assert left["left_only"]["entry_id"].to_list() == [4]
        assert left["right_only"].equals(inner["right_only"])
        assert left["right_only"]["header_id"].to_list() == [13], "G3 pairs with nothing on the left"

    def test_a_how_the_tool_does_not_offer_is_refused(self) -> None:
        """Including the ones a reader might reasonably expect: a right join is the inputs swapped."""
        for unsupported in ("right", "outer", "anti", "LEFT"):
            with pytest.raises(ConfigError):
                run_tool(
                    Join(),
                    Ports({"left": _entries(), "right": _headers()}),
                    {"keys": ["group_id"], "how": unsupported},
                )
