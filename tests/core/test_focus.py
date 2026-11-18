"""Tests for narrowing a finished run to one entity, port by port.

The runner leaves a trace: every frame the flow produced, keyed by the port that produced it. That
answers "what happened". It does not answer the question the tool exists for -- *show me this one
thing at every step* -- and a figure that is wrong is traced by following the one entity it belongs
to until the port where it stopped being right. That is what slicing is.

**A slice filters a finished run; it never re-runs one.** A reduction sums over siblings, a pairing
matches across the whole frame, and a flow re-run over one entity's rows answers a different
question -- with numbers that look entirely reasonable. So the trace is read and a parallel view is
handed back, which also means the same run can be sliced again for the next entity and gives the
same answer the second time.

**Identity is a set of candidate columns, and there is no per-port column override.** A detailed
frame and a reduced one name the same entity differently, so the declaration is a tuple and the
first one a port carries is the one matched on. What a focus deliberately *cannot* say is "at this
port, identity is some other column": a slice applies one value -- the entity's own -- so naming a
key column there would filter that column for a name it never holds and match nothing, silently. An
earlier attempt at this expressed exactly that override, and it could not be made to work. Another
grain is reached only by :class:`RelatedKey`, which learns the values it matches on at a port where
the entity does resolve, so the value applied is never the one the reader typed.

**Nothing here, no grain, and not retained are three different answers.** A port with none of this
entity's rows is a fact about the entity. A port carrying no column that names an entity at all is a
fact about the port, and reporting it as nought rows tells a reader their entity was dropped
somewhere it never was. A port whose frame a live run released is a fact about the run. Each gets
its own status, and only the first is a count.

**A count of entities is a count of distinct entities at a named port.** The previous
implementation summed an entity's matching rows across every frame it appeared in and presented the
total as a number of entities, which multiplied them by the length of the flow. Nothing here offers
a figure spanning two ports.

Vocabulary is deliberately neutral -- entries, regions, parties, tiers -- because the library ships
into other repositories and must carry no caller's nouns.
"""

from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.focus import (
    EntityOption,
    FocusedTrace,
    FocusError,
    FocusSpec,
    MissingEvidenceError,
    PortSlice,
    RelatedKey,
    RelatedKeyError,
    UnknownEntityError,
    available_entities,
    slice_trace,
)
from dantron_etl.pipeline import PipelineSpec, PortRef, load_pipeline
from dantron_etl.runner import PipelineTrace, StepFailedError, run_pipeline
from dantron_etl.tool import ToolError

# ---------------------------------------------------------------------------
# The fixture flow
#
# Three supplied sources, a split, a pairing with three output ports, a reduction and a derived
# column. Every tool is one the shipped registry already holds and the trace is a real one, because
# the shape of the trace is what is being sliced: a hand-built trace would only agree with itself.
#
# What it is built to contain, port by port:
#   * ports that name the entity directly, at two different grains;
#   * a port that names it under a different column name (`roster.rows`);
#   * ports at a grain that names no entity at all (`totals.rows`, `rated.rows`, `regions.rows`),
#     reachable only through a key;
#   * ports that name no entity and are left undeclared (`paired.right_only`, `checked.rows`);
#   * a port that is empty for one entity and not for another (`kept.false`);
#   * a port that is empty for everybody (`paired.left_only`);
#   * a row whose entity is null, which belongs to nobody.
# ---------------------------------------------------------------------------

#: Every port the flow produces, in the order the run produces them.
PORTS: tuple[PortRef, ...] = (
    PortRef("entries", "rows"),
    PortRef("regions", "rows"),
    PortRef("roster", "rows"),
    PortRef("vetted", "rows"),
    PortRef("kept", "true"),
    PortRef("kept", "false"),
    PortRef("paired", "joined"),
    PortRef("paired", "left_only"),
    PortRef("paired", "right_only"),
    PortRef("totals", "rows"),
    PortRef("rated", "rows"),
    PortRef("checked", "rows"),
)

ENTRIES = PortRef("entries", "rows")
REGIONS = PortRef("regions", "rows")
ROSTER = PortRef("roster", "rows")
VETTED = PortRef("vetted", "rows")
KEPT_TRUE = PortRef("kept", "true")
KEPT_FALSE = PortRef("kept", "false")
MATCHED = PortRef("paired", "joined")
LEFT_ONLY = PortRef("paired", "left_only")
RIGHT_ONLY = PortRef("paired", "right_only")
TOTALS = PortRef("totals", "rows")
RATED = PortRef("rated", "rows")
CHECKED = PortRef("checked", "rows")


def _document() -> dict[str, Any]:
    """A fresh, valid pipeline document; every trace below is a run of this flow."""
    return {
        "name": "a neutral fixture",
        "lanes": [
            {"id": "intake", "label": "Intake", "order": 1},
            {"id": "shaping", "label": "Shaping", "order": 2},
        ],
        "steps": [
            {
                "id": "entries",
                "tool": "read_csv",
                "lane": "intake",
                "stage": "collect",
                "config": {"path": "sources/entries.csv"},
            },
            {
                "id": "regions",
                "tool": "read_csv",
                "lane": "intake",
                "stage": "collect",
                "config": {"path": "sources/regions.csv"},
            },
            {
                "id": "roster",
                "tool": "read_csv",
                "lane": "intake",
                "stage": "collect",
                "config": {"path": "sources/roster.csv"},
            },
            {
                "id": "vetted",
                "tool": "assert_rows",
                "lane": "intake",
                "stage": "check",
                "config": {"predicate": "[tier] > 0"},
                "inputs": {"rows": "roster.rows"},
            },
            {
                "id": "kept",
                "tool": "filter",
                "lane": "shaping",
                "stage": "shape",
                "why": "only the entries carrying a positive amount travel on",
                "config": {"predicate": "[amount] > 0"},
                "inputs": {"rows": "entries.rows"},
            },
            {
                "id": "paired",
                "tool": "join",
                "lane": "shaping",
                "stage": "shape",
                "config": {"keys": ["region"]},
                "inputs": {"left": "kept.true", "right": "regions.rows"},
            },
            {
                "id": "totals",
                "tool": "group_by",
                "lane": "shaping",
                "stage": "shape",
                "config": {
                    "by": ["region"],
                    "aggregations": [{"name": "total", "function": "sum", "value": "[amount]"}],
                },
                "inputs": {"rows": "paired.joined"},
            },
            {
                "id": "rated",
                "tool": "formula",
                "lane": "shaping",
                "stage": "shape",
                "config": {"columns": [{"name": "share", "value": "[total] / 100"}]},
                "inputs": {"rows": "totals.rows"},
            },
            {
                "id": "checked",
                "tool": "assert_rows",
                "lane": "shaping",
                "stage": "check",
                "config": {"predicate": "[total] >= 0"},
                "inputs": {"rows": "rated.rows"},
            },
        ],
    }


def _entries() -> pl.DataFrame:
    """Line-grain rows: three parties, one row belonging to nobody, two amounts the split drops."""
    return pl.DataFrame(
        {
            "entry_id": [1, 2, 3, 4, 5, 6, 7],
            "party": ["hillcrest", "hillcrest", "brightwater", "oakfield", "hillcrest", "brightwater", None],
            "region": ["north", "north", "south", "south", "east", "north", "west"],
            "amount": [10.0, -5.0, 7.5, -2.5, 4.0, 3.0, -1.0],
        }
    )


def _regions() -> pl.DataFrame:
    """A shared reference table: no column here names a party, and one region is referenced by none."""
    return pl.DataFrame(
        {
            "region": ["north", "south", "east", "west"],
            "region_label": ["Northern", "Southern", "Eastern", "Western"],
        }
    )


def _roster(tiers: list[int] | None = None) -> pl.DataFrame:
    """A list of parties by name -- the same identity, under a different column name.

    ``seabright`` is on the roster and in no entry: an entity a run holds at some of its ports and
    not at others, which is a different fact from one it holds nowhere.
    """
    return pl.DataFrame(
        {
            "party_name": ["brightwater", "hillcrest", "oakfield", "seabright"],
            "tier": [1, 2, 3, 1] if tiers is None else tiers,
        }
    )


def _sources(roster: pl.DataFrame | None = None) -> dict[PortRef, pl.DataFrame]:
    """The frames for the flow's three source ports, which is what makes it run with no file present."""
    return {
        ENTRIES: _entries(),
        REGIONS: _regions(),
        ROSTER: _roster() if roster is None else roster,
    }


@pytest.fixture
def spec() -> PipelineSpec:
    return load_pipeline(_document())


@pytest.fixture
def trace(spec: PipelineSpec) -> PipelineTrace:
    """A finished run holding every frame, which is what a walkthrough is read from."""
    return run_pipeline(spec, sources=_sources(), retain="all")


#: The party a reader is following. ``party`` at line grain, ``party_name`` on the roster: one
#: identity, two column names, first-present at each port.
FOCUS = FocusSpec(
    label="party",
    identity=("party", "party_name"),
    via={
        # A reduction keyed by region names no party at all; its rows are reached through the
        # regions this party's paired rows are in. Learning comes from the port the reduction
        # actually consumed, so the values learned are the ones that fed the figures.
        TOTALS: RelatedKey(from_port=MATCHED, key="region"),
        RATED: RelatedKey(from_port=MATCHED, key="region"),
        # The reference table is narrowed the same way, by what referenced it.
        REGIONS: RelatedKey(from_port=MATCHED, key="region"),
    },
)


def _shape(one: PortSlice) -> tuple[Any, ...]:
    """Everything about a slice a second identical call must reproduce, frame aside."""
    return (
        one.port,
        one.status,
        one.rows_focused,
        one.rows_at_port,
        one.columns,
        one.matched_on,
        one.related,
        one.related_values,
        one.reason,
    )


class TestOneEntityAtEveryPort:
    """The thing the tool is for: this party's rows, wherever the run put them."""

    def test_a_port_that_names_the_entity_is_narrowed_to_that_entity_s_own_rows(self, trace: PipelineTrace) -> None:
        """A direct port hands back the entity's rows and says which column it matched on.

        Both counts are against this one port and neither spans two: how many rows the port produced
        and how many of them are the entity's is the pair that distinguishes a port that ignored an
        entity from one that dropped it.
        """
        focused = slice_trace(trace, FOCUS, "brightwater")
        assert isinstance(focused, FocusedTrace)

        at_entries = focused.slice(ENTRIES)
        assert at_entries.status == "matched"
        assert at_entries.matched_on == "party"
        assert at_entries.rows_focused == 2
        assert at_entries.rows_at_port == 7
        assert at_entries.columns == trace.shapes[ENTRIES].columns
        assert at_entries.related is None and at_entries.related_values == ()
        assert at_entries.reason == "", "a port that resolved by column has nothing to explain"
        assert_frame_equal(
            at_entries.frame,
            pl.DataFrame(
                {
                    "entry_id": [3, 6],
                    "party": ["brightwater", "brightwater"],
                    "region": ["south", "north"],
                    "amount": [7.5, 3.0],
                }
            ),
        )

        # The same identity column at a later grain, and the counts move with the flow.
        assert (focused.slice(KEPT_TRUE).rows_focused, focused.slice(KEPT_TRUE).rows_at_port) == (2, 4)
        at_pairing = focused.slice(MATCHED)
        assert (at_pairing.rows_focused, at_pairing.rows_at_port) == (2, 4)
        assert at_pairing.frame.get_column("party").unique().to_list() == ["brightwater"]
        assert at_pairing.frame is not trace.frame(MATCHED)

    def test_an_entity_is_found_under_whatever_each_port_calls_it(self, trace: PipelineTrace) -> None:
        """One value, several column names: the candidates are tried in declaration order, first present.

        Not the first column that happens to hold the value, because then two entities would be
        matched on different columns at one port and the label a panel prints would drift with the
        data; and not all of them at once, because one identity in two columns is one filter.
        """
        focused = slice_trace(trace, FOCUS, "brightwater")

        assert focused.slice(ENTRIES).matched_on == "party"
        assert focused.slice(ROSTER).matched_on == "party_name"
        assert (focused.slice(ROSTER).rows_focused, focused.slice(ROSTER).rows_at_port) == (1, 4)
        assert focused.slice(VETTED).matched_on == "party_name"

        # Drop the second spelling and the roster stops naming a party at all -- it does not quietly
        # match on something else, and it does not report nought rows.
        narrower = FocusSpec(label="party", identity=("party",))
        one_name_only = slice_trace(trace, narrower, "brightwater")
        assert one_name_only.slice(ENTRIES).rows_focused == 2
        assert one_name_only.slice(ROSTER).status == "no_grain"
        assert one_name_only.slice(ROSTER).matched_on is None

    def test_the_focused_view_covers_every_port_the_run_produced_in_its_own_order(self, trace: PipelineTrace) -> None:
        """A walkthrough is read front to back, and every port gets an answer of some kind.

        Including the ports nobody consumes: the rows a split discarded are where a missing figure is
        usually found, so a view that skipped them would omit the panel a reader needs most.
        """
        focused = slice_trace(trace, FOCUS, "brightwater")

        assert tuple(one.port for one in focused.slices) == PORTS
        assert isinstance(focused.slices, tuple)
        assert focused.pipeline == "a neutral fixture"
        assert focused.label == "party"
        assert focused.value == "brightwater"
        assert focused.slice(KEPT_FALSE).port == KEPT_FALSE

        with pytest.raises(FocusError) as unknown:
            focused.slice(PortRef("kept", "maybe"))
        assert "kept.maybe" in str(unknown.value)


class TestThreeWaysOfHavingNothingToShow:
    """Nothing of this entity's, no entity at all, and nothing kept: never one answer."""

    def test_a_port_with_none_of_the_entity_s_rows_is_not_a_port_with_no_grain_for_it(
        self, trace: PipelineTrace
    ) -> None:
        """Nought rows is a fact about the entity; no grain is a fact about the port.

        Read alike, they send a reader hunting a filter that never fired -- or worse, let them
        conclude a flow dropped an entity at a port that never carried one. So the two carry
        different statuses, and the count that cannot be taken is absent rather than nought.
        """
        focused = slice_trace(trace, FOCUS, "brightwater")

        empty = focused.slice(KEPT_FALSE)
        assert empty.status == "matched"
        assert empty.matched_on == "party"
        assert empty.rows_focused == 0, "the count is nought, and nought is an answer"
        assert empty.rows_at_port == 3
        assert empty.frame is not None and empty.frame.height == 0
        assert tuple(empty.frame.columns) == trace.shapes[KEPT_FALSE].columns
        assert empty.reason == ""

        no_grain = focused.slice(RIGHT_ONLY)
        assert no_grain.status == "no_grain"
        assert no_grain.status != empty.status
        assert no_grain.rows_focused is None, "a count nobody can take is not nought"
        assert no_grain.rows_at_port == 1, "how many rows the port produced is still reported honestly"
        assert no_grain.frame is None, "a shared row shown under one party's name would be a lie"
        assert no_grain.matched_on is None
        assert "right_only" in no_grain.reason
        assert "region_label" in no_grain.reason, "the columns it does carry are what a reader picks a key from"
        assert "brightwater" not in no_grain.reason, "the reason is about the port, not about this party"

        # A port at the same grain that was simply never declared reads the same way.
        assert focused.slice(CHECKED).status == "no_grain"

        # And a port that is empty for everybody is still a count, not a missing grain.
        for_everybody = focused.slice(LEFT_ONLY)
        assert (for_everybody.status, for_everybody.rows_focused, for_everybody.rows_at_port) == ("matched", 0, 0)

    def test_a_released_port_says_it_was_not_retained_rather_than_saying_nothing_is_here(
        self, spec: PipelineSpec
    ) -> None:
        """A live run frees a frame the moment nothing will read it again, and shapes survive.

        Every count in such a view is unknown rather than nought -- the one thing it must not do is
        report a run in which this party has nothing anywhere. What is knowable without the frame is
        still reported: how large the port was, what it carried, and whether it named a party at all.
        """
        live = run_pipeline(spec, sources=_sources(), retain="live")
        assert dict(live.frames) == {}, "precondition: a live run holds no frame once it is over"

        focused = slice_trace(live, FOCUS, "brightwater")
        assert tuple(one.port for one in focused.slices) == PORTS
        assert all(one.rows_focused is None for one in focused.slices), "nothing in a live view reads as nought"

        released = focused.slice(ENTRIES)
        assert released.status == "not_retained"
        assert released.rows_at_port == 7
        assert released.columns == ("entry_id", "party", "region", "amount")
        assert released.frame is None
        assert "not retained" in released.reason.lower()
        assert "live" in released.reason

        # A port reached through a key needs the frame the key is learned at, and says which one.
        through_a_key = focused.slice(TOTALS)
        assert through_a_key.status == "not_retained"
        assert "paired.joined" in through_a_key.reason
        assert "not retained" in through_a_key.reason.lower()

        # No grain is knowable from the shape alone, so it is still the more useful answer.
        assert focused.slice(RIGHT_ONLY).status == "no_grain"

        with pytest.raises(MissingEvidenceError) as nothing_to_pick:
            available_entities(live, FOCUS)
        assert "retain" in str(nothing_to_pick.value)
        assert "all" in str(nothing_to_pick.value), "the refusal must say how to get a list"

    def test_a_value_no_port_holds_is_refused_rather_than_shown_as_a_run_of_nought_rows(
        self, spec: PipelineSpec, trace: PipelineTrace
    ) -> None:
        """A view of nought everywhere reads as a flow that dropped an entity it never saw.

        That is the plausible wrong answer this module exists to prevent, and it is indistinguishable
        by eye from an entity dropped at the first step. So a value no port that names an entity
        holds is refused by name, and an entity that some ports hold and others do not is sliced
        exactly as it is.
        """
        with pytest.raises(UnknownEntityError) as raised:
            slice_trace(trace, FOCUS, "millbrook")
        message = str(raised.value)
        assert "millbrook" in message
        assert "party" in message
        assert "available_entities" in message, "the refusal must say where the values come from"
        assert isinstance(raised.value, FocusError) and isinstance(raised.value, ToolError)

        # On the roster and in no entry: held somewhere, so it is a view and not a refusal.
        on_the_roster = slice_trace(trace, FOCUS, "seabright")
        assert on_the_roster.slice(ROSTER).rows_focused == 1
        assert on_the_roster.slice(ENTRIES).rows_focused == 0, "dropped before the first port is a real answer"

        # Nothing was retained, so nothing can be refuted: a live run cannot say a value is unheld.
        live = run_pipeline(spec, sources=_sources(), retain="live")
        assert all(one.rows_focused is None for one in slice_trace(live, FOCUS, "millbrook").slices)


class TestReachingAGrainThatNamesNoEntity:
    """A reduction keyed by something else, reached through a key learned where the entity resolves."""

    def test_a_port_at_another_grain_is_reached_through_a_key_learned_where_the_entity_resolves(
        self, trace: PipelineTrace
    ) -> None:
        """The values matched on are learned, never typed: the reader's value is applied upstream.

        The figure a related port shows covers every row sharing those key values -- the northern
        total here is 13.0, of which 10.0 is another party's -- so the slice carries the key, the
        port it was learned at and the values themselves. A figure under an entity's name that was
        matched by something other than that name is otherwise unaccountable.
        """
        focused = slice_trace(trace, FOCUS, "brightwater")

        reduced = focused.slice(TOTALS)
        assert reduced.status == "related"
        assert reduced.matched_on == "region"
        assert reduced.related == RelatedKey(from_port=MATCHED, key="region")
        assert reduced.related_values == ("south", "north")
        assert (reduced.rows_focused, reduced.rows_at_port) == (2, 3)
        assert_frame_equal(reduced.frame, pl.DataFrame({"region": ["north", "south"], "total": [13.0, 7.5]}))
        assert "region" in reduced.reason
        assert "paired.joined" in reduced.reason, "where the values came from is part of the answer"
        assert "party" in reduced.reason, "and that the figure is not this party's alone"

        # The derived column downstream of it is followed the same way.
        assert (focused.slice(RATED).rows_focused, focused.slice(RATED).rows_at_port) == (2, 3)
        assert focused.slice(RATED).related_values == ("south", "north")

    def test_a_key_may_be_learned_at_a_port_the_run_produced_later(self, trace: PipelineTrace) -> None:
        """Slicing happens after the run, so a key has no ordering to obey.

        ``regions.rows`` is produced second and is narrowed by what referenced it at the seventh
        port. A rule that a key must be learned upstream would forbid the one thing a reference table
        is ever narrowed by -- and buys nothing, because every frame already exists when a trace is
        read.
        """
        focused = slice_trace(trace, FOCUS, "brightwater")

        reference = focused.slice(REGIONS)
        assert reference.status == "related"
        assert reference.related_values == ("south", "north")
        assert (reference.rows_focused, reference.rows_at_port) == (2, 4)
        assert_frame_equal(
            reference.frame,
            pl.DataFrame({"region": ["north", "south"], "region_label": ["Northern", "Southern"]}),
        )
        assert PORTS.index(REGIONS) < PORTS.index(MATCHED), "the key is learned at a later port than it is used"

    def test_an_entity_the_key_learns_nothing_for_reaches_no_rows_rather_than_raising(
        self, trace: PipelineTrace
    ) -> None:
        """Every one of this party's entries was dropped before the pairing, so there is nothing to learn.

        Nought learned values means nought rows, which is the honest answer: attributing the regional
        figure to a party whose rows never reached the reduction would put another party's money
        under its name.
        """
        focused = slice_trace(trace, FOCUS, "oakfield")

        assert focused.slice(ENTRIES).rows_focused == 1
        assert focused.slice(KEPT_TRUE).rows_focused == 0
        reduced = focused.slice(TOTALS)
        assert reduced.status == "related", "it is still resolved, and still reported as resolved"
        assert reduced.related_values == ()
        assert (reduced.rows_focused, reduced.rows_at_port) == (0, 3)
        assert reduced.frame is not None and reduced.frame.height == 0


class TestWhatAReaderMayPick:
    """The list a picker offers, and the count that goes with it."""

    def test_the_entities_a_run_offers_are_counted_once_each_and_against_a_named_port(
        self, trace: PipelineTrace
    ) -> None:
        """An entity is counted once, however many rows and however many ports carry it.

        The previous implementation summed matching rows across every frame, so an entity surviving
        eight ports was counted eight times and the reported number of entities grew with the length
        of the flow rather than with the data. Every count here is against one port, and which port
        it is is part of the answer.
        """
        options = available_entities(trace, FOCUS)
        assert all(isinstance(option, EntityOption) for option in options)
        assert [option.value for option in options] == ["brightwater", "hillcrest", "oakfield", "seabright"]

        # The figure the old implementation would have offered, for contrast: it is a length of
        # pipeline, not a number of parties, and nothing here reports it.
        across_ports = sum(option.rows_at(port) for option in options for port in option.ports)
        assert across_ports == 24
        assert len(options) == 4, "four parties, whatever the rows across ports add up to"

        hillcrest = next(option for option in options if option.value == "hillcrest")
        assert hillcrest.ports == (ENTRIES, ROSTER, VETTED, KEPT_TRUE, KEPT_FALSE, MATCHED)
        assert hillcrest.rows_at(ENTRIES) == 3
        assert hillcrest.rows_at(KEPT_TRUE) == 2
        assert hillcrest.rows_at(KEPT_FALSE) == 1
        assert hillcrest.rows_at(ROSTER) == 1, "counted under whatever that port calls a party"
        assert hillcrest.rows_at(RIGHT_ONLY) == 0
        assert hillcrest.rows_at(TOTALS) == 0, "a port reached through a key offers nobody to pick"

        for banned in ("rows", "row_count", "total_rows"):
            assert not hasattr(hillcrest, banned), f"{banned!r} would be a figure spanning two ports"

        # Which port is part of the question as well as of the answer.
        at_the_split = available_entities(trace, FOCUS, port=KEPT_FALSE)
        assert [option.value for option in at_the_split] == ["hillcrest", "oakfield"]
        assert trace.shapes[KEPT_FALSE].rows == 3, "three rows, two parties: a row whose party is null is nobody's"
        assert at_the_split[0].ports == (KEPT_FALSE,)
        assert [option.value for option in available_entities(trace, FOCUS, port=ENTRIES)] == [
            "brightwater",
            "hillcrest",
            "oakfield",
        ]
        assert next(option for option in options if option.value == "seabright").ports == (ROSTER, VETTED)

        with pytest.raises(FocusError) as unknown_port:
            available_entities(trace, FOCUS, port=PortRef("kept", "maybe"))
        assert "kept.maybe" in str(unknown_port.value)


class TestWhatAFocusMayNotSay:
    """The declarations a focus refuses, and the one it does not have at all."""

    def test_a_focus_cannot_name_a_column_at_a_port_that_does_not_hold_the_entity_s_value(self) -> None:
        """There is no per-port column override, and that is the point rather than an omission.

        A slice applies one value, the entity's own. An override naming a key column at a reduced
        port would filter that column for a party's name, match nothing, and report it as a party
        with no rows there. An earlier attempt expressed exactly that and could not be made to work,
        so the only per-port declaration a focus has is a related key -- which cannot apply the
        reader's value, because it learns the values it applies.
        """
        with pytest.raises(FocusError) as override:
            FocusSpec(label="party", identity=("party",), via={TOTALS: ("region",)})
        message = str(override.value)
        assert "RelatedKey" in message, "the refusal must name the thing that does work"
        assert "totals.rows" in message

        with pytest.raises(FocusError) as itself:
            FocusSpec(label="party", identity=("party",), via={TOTALS: RelatedKey(from_port=TOTALS, key="region")})
        assert "totals.rows" in str(itself.value)

        with pytest.raises(FocusError):
            FocusSpec(label="party", identity=())

        # A port is named the way a trace names one. A mistyped key that was merely ignored would
        # turn its port into one with no grain, which reads as an answer.
        with pytest.raises(FocusError) as spelled:
            FocusSpec(
                label="party", identity=("party",), via={"totals.rows": RelatedKey(from_port=MATCHED, key="region")}
            )
        assert "PortRef" in str(spelled.value)

        frozen = FocusSpec(
            label="party", identity=("party",), via={TOTALS: RelatedKey(from_port=MATCHED, key="region")}
        )
        with pytest.raises(AttributeError):
            frozen.identity = ("something else",)  # type: ignore[misc]
        with pytest.raises(TypeError):
            frozen.via[RATED] = RelatedKey(from_port=MATCHED, key="region")  # type: ignore[index]

    def test_a_key_that_is_not_a_column_of_both_ends_is_refused_naming_the_end_that_lacks_it(
        self, trace: PipelineTrace
    ) -> None:
        """Either end may be the misspelling, so the refusal names which one came up short.

        This is a mis-declaration, not missing evidence: the fix is to correct the key, and a message
        that did not distinguish the two would send a reader to re-run a flow that is fine.
        """
        wrong_at_the_far_end = FocusSpec(
            label="party",
            identity=("party", "party_name"),
            via={TOTALS: RelatedKey(from_port=MATCHED, key="entry_id")},
        )
        with pytest.raises(RelatedKeyError) as raised:
            slice_trace(trace, wrong_at_the_far_end, "brightwater")
        assert "totals.rows" in str(raised.value)
        assert "entry_id" in str(raised.value)
        assert not isinstance(raised.value, MissingEvidenceError)

        wrong_at_the_origin = FocusSpec(
            label="party",
            identity=("party", "party_name"),
            via={TOTALS: RelatedKey(from_port=ROSTER, key="region")},
        )
        with pytest.raises(RelatedKeyError) as at_origin:
            slice_trace(trace, wrong_at_the_origin, "brightwater")
        assert "roster.rows" in str(at_origin.value)


class TestATraceIsEvidence:
    """A partial run is exactly when somebody slices, and slicing must leave the run untouched."""

    def test_a_run_that_stopped_early_refuses_the_key_it_cannot_learn(self, spec: PipelineSpec) -> None:
        """A trace that stopped mid-flow is the one most likely to be sliced, so its refusal matters most.

        The key can no longer be learned, because the run never reached the port it is learned at.
        Reaching into the frames for it raises a bare mapping error naming a port and nothing else,
        which tells a reader neither what was being attempted nor what to change. It is refused in
        the library's own family instead, naming the port that needs the key, the port it would have
        come from, and the ports the run did produce.
        """
        with pytest.raises(StepFailedError) as failure:
            run_pipeline(spec, sources=_sources(roster=_roster(tiers=[1, 0, 3, 1])), retain="all")
        partial = failure.value.trace
        assert tuple(partial.shapes) == (ENTRIES, REGIONS, ROSTER), "precondition: the run stopped at the fourth step"

        with pytest.raises(MissingEvidenceError) as raised:
            slice_trace(partial, FOCUS, "brightwater")
        message = str(raised.value)
        assert "regions.rows" in message, "the port that cannot be shown"
        assert "paired.joined" in message, "the port its key would have come from"
        assert "entries.rows" in message, "and what the run did produce, so the focus can be pointed at it"
        assert isinstance(raised.value, FocusError) and isinstance(raised.value, ToolError)
        assert not isinstance(raised.value, KeyError), "a bare lookup failure is not an account of anything"

        # Pointed only at ports the run reached, the same partial trace slices to what it managed.
        reachable = FocusSpec(label="party", identity=("party", "party_name"))
        salvaged = slice_trace(partial, reachable, "brightwater")
        assert tuple(one.port for one in salvaged.slices) == (ENTRIES, REGIONS, ROSTER)
        assert salvaged.slice(ENTRIES).rows_focused == 2
        assert salvaged.slice(REGIONS).status == "no_grain"

    def test_slicing_reads_the_trace_without_changing_it_and_answers_the_same_twice(self, trace: PipelineTrace) -> None:
        """A picker slices the same finished run once per entity, so a slice cannot consume one.

        Read twice for two entities and then a third time for the first, every answer is the same and
        every frame of the run is the frame it was.
        """
        before = {port: trace.frame(port) for port in PORTS}

        first = slice_trace(trace, FOCUS, "brightwater")
        between = slice_trace(trace, FOCUS, "hillcrest")
        again = slice_trace(trace, FOCUS, "brightwater")

        assert between.value == "hillcrest"
        assert between.slice(ENTRIES).rows_focused == 3, "the second read is the second entity's, not the first's"

        assert [_shape(one) for one in first.slices] == [_shape(one) for one in again.slices]
        for one, other in zip(first.slices, again.slices):
            if one.frame is None:
                assert other.frame is None
            else:
                assert_frame_equal(one.frame, other.frame)

        for port in PORTS:
            assert trace.frame(port) is before[port], "the run's own frames are the objects they were"
            assert_frame_equal(trace.frame(port), before[port])

        assert available_entities(trace, FOCUS) == available_entities(trace, FOCUS)
