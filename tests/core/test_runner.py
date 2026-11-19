"""Tests for running a loaded pipeline and leaving a trace of what happened.

The loader says what a flow *is*: steps, port-to-port edges, an order to walk them in, and the
column references a runner will have to check once frames exist. Nothing in it moves a row. This
module specifies the thing that does -- and the evidence it leaves behind, which is the whole point
of it. These flows run inside an orchestrator, and when one fails in the middle of the night the
trace is the only account anybody has of what happened.

**A trace is keyed by port, not by step.** A step has several output ports -- a split's two sides, a
pairing's three -- so one frame per step cannot say which frame. Every frame the run produced is
reachable as :class:`PortRef`, every shape is reported against the port that produced it, and an
event's row counts are per output port, because a filter's two numbers are the fact somebody needs.

**Two readers, one artefact.** A walkthrough wants every intermediate frame so a person can see the
data at each step; a production run holding every intermediate is the obvious memory failure on a
real dataset. The loader publishes ``consumers()`` for every port, so last use is computable and the
choice is a retention mode rather than a second runner. The property that matters is not which
frames survive but that *the answers do not change*: the same events, the same counts, the same
shapes, whichever mode ran.

**Refusals come before side effects.** A configuration still holding ``${...}`` stops the run before
step one, because a refusal that arrives after eleven steps have written to a database is not a
refusal. A column an expression names and a frame does not carry stops that step before its tool is
entered, the instant the upstream frame exists -- not up front for the whole graph, because no tool
declares its output columns and a whole-graph pass could only ever reach the steps fed by frames
that were supplied.

**Frames come in, rather than being read.** Every source tool in the catalog opens a file or a
database; a run is therefore handed the frames for its source ports, which is also what an
orchestrator has (an upstream asset, already materialised) and what a walkthrough has (a frame in a
notebook). A supplied source is not executed. A *computed* port may not be supplied, because then
the flow that ran would not be the flow in the file.

Vocabulary here is deliberately neutral -- entries, headers, regions, codes -- because the library
ships into other repositories and must carry no caller's nouns.
"""

import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl import runner
from dantron_etl import Pipeline, PortRef, load_pipeline
from dantron_etl import (
    CachedFrame,
    ColumnCheckError,
    PipelineTrace,
    RunError,
    StepEvent,
    StepFailedError,
    run_pipeline,
)
from dantron_etl.runner import logging_sink
from dantron_etl.trace import FrameShape
from dantron_etl.authoring import DataError, ExecutionError, ToolError

# ---------------------------------------------------------------------------
# The fixture flow
#
# Two supplied sources; a split with two outputs; a pairing with two inputs and three outputs; a
# fan-out from one source port into two branches; a reduction, a derived column, and two assertions
# that end their branches. Every tool it names is one the shipped registry already holds, so the
# runner is exercised against real templates, real port declarations and a real refusal rather than
# against a stub that agrees with it. No file and no database is touched, because the source ports
# are handed their frames.
# ---------------------------------------------------------------------------

#: The order the loader's dependency sort produces for the document below: declaration order among
#: the steps that are ready, which makes every position here deterministic and worth asserting. It
#: is deliberately *not* the order the document declares its steps in -- see the note there.
ORDER: tuple[str, ...] = (
    "entries",
    "headers",
    "trimmed",
    "audited",
    "kept",
    "paired",
    "totals",
    "rated",
    "checked",
)

#: Every port the flow produces, in the order the run produces them: step order, and within a step
#: the order its tool declares its outputs in.
PORTS: tuple[PortRef, ...] = (
    PortRef("entries", "rows"),
    PortRef("headers", "rows"),
    PortRef("trimmed", "rows"),
    PortRef("audited", "rows"),
    PortRef("kept", "true"),
    PortRef("kept", "false"),
    PortRef("paired", "joined"),
    PortRef("paired", "left_only"),
    PortRef("paired", "right_only"),
    PortRef("totals", "rows"),
    PortRef("rated", "rows"),
    PortRef("checked", "rows"),
)


def _document() -> dict[str, Any]:
    """A fresh, valid pipeline document. Every test below loads this one, or a mutation of it."""
    return {
        "name": "a neutral fixture",
        "lanes": [
            {"id": "intake", "title": "Intake", "order": 1},
            {"id": "shaping", "title": "Shaping", "order": 2},
        ],
        "steps": [
            # Written first, run last, deliberately -- do not tidy it back into place.
            #
            # Everywhere else the order a step is written in and the order it runs in would be the
            # same sequence, and while they are the same sequence nothing in this file can tell
            # spec.order() from spec.steps. A runner that walked the declaration list, or a release
            # schedule that took its positions from it, would satisfy every assertion below and
            # still report a different flow on any real document -- where nobody writes six hundred
            # steps in topological order. 'checked' reads 'rated.rows', so no ordering can put it
            # anywhere but last; writing it first costs nothing, leaves ORDER exactly as it was,
            # and makes the two sequences disagree.
            {
                "id": "checked",
                "tool": "assert_rows",
                "lane": "shaping",
                "stage": "check",
                "config": {"predicate": "[total] >= 0", "allow_empty": True},
                "inputs": {"rows": "rated.rows"},
            },
            {
                "id": "entries",
                "tool": "read_csv",
                "lane": "intake",
                "stage": "collect",
                "config": {"path": "sources/entries.csv"},
            },
            {
                "id": "headers",
                "tool": "read_csv",
                "lane": "intake",
                "stage": "collect",
                "config": {"path": "sources/headers.csv"},
            },
            {
                "id": "trimmed",
                "tool": "select_columns",
                "lane": "shaping",
                "stage": "shape",
                "config": {"columns": ["entry_id", "region"]},
                "inputs": {"rows": "entries.rows"},
            },
            {
                "id": "audited",
                "tool": "assert_rows",
                "lane": "shaping",
                "stage": "check",
                "config": {"predicate": "[entry_id] > 0"},
                "inputs": {"rows": "trimmed.rows"},
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
                "config": {"keys": ["entry_id"]},
                "inputs": {"left": "kept.true", "right": "headers.rows"},
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
        ],
    }


def _entries() -> pl.DataFrame:
    """The entries frame the flow is handed: two of its five rows fail the split's condition."""
    return pl.DataFrame(
        {
            "entry_id": [1, 2, 3, 4, 5],
            "region": ["north", "north", "south", "south", "east"],
            "amount": [10.0, -5.0, 7.5, -2.5, 4.0],
        }
    )


def _headers() -> pl.DataFrame:
    """The headers frame: three of its four rows pair, so the pairing's side ports are not all empty."""
    return pl.DataFrame({"entry_id": [1, 3, 5, 9], "header_code": ["A", "B", "C", "D"]})


def _nothing_positive() -> pl.DataFrame:
    """An entries frame the split routes entirely to 'false', so 'kept.true' comes out empty.

    Every row still satisfies the assertion on 'entry_id', so the whole flow runs on this frame to
    completion -- which is what lets a truncated run on it be compared against a complete one.
    """
    return pl.DataFrame({"entry_id": [1, 2], "region": ["north", "south"], "amount": [-1.0, -2.0]})


def _sources(
    entries: pl.DataFrame | None = None,
    headers: pl.DataFrame | None = None,
) -> dict[PortRef, pl.DataFrame]:
    """The frames for the flow's two source ports, which is what makes it run with no file present."""
    return {
        PortRef("entries", "rows"): _entries() if entries is None else entries,
        PortRef("headers", "rows"): _headers() if headers is None else headers,
    }


def _spec(**changes: Any) -> Pipeline:
    """The fixture flow, optionally with one step's configuration replaced by ``step_id=config``."""
    document = _document()
    for step in document["steps"]:
        if step["id"] in changes:
            step["config"] = changes[step["id"]]
    return load_pipeline(document)


def _comparable(event: StepEvent) -> tuple[Any, ...]:
    """Everything about an event that a second run of the same flow must reproduce exactly.

    Duration and timestamp are excluded because they are measurements, and ``released`` because it
    is the one field the retention mode is *allowed* to change.

    ``cached`` is in here for the same reason ``supplied`` is. Every comparison below claims two
    runs reported the same thing, and the strongest of them claims a run given no cache is the run
    there was before caches existed -- a claim this tuple has to be able to see being broken, or it
    says less than its name does.
    """
    return (
        event.pipeline,
        event.step_id,
        event.tool,
        event.lane,
        event.stage,
        event.level,
        event.supplied,
        event.cached,
        event.inputs,
        event.outputs,
        event.notes,
        event.error,
    )


class TestARunWalksTheFlowAndKeepsWhatItProduced:
    """The whole surface of a run: every port's frame, one event per step, in dependency order."""

    def test_a_run_walks_every_step_and_keeps_one_frame_per_port(self) -> None:
        """A frame per port, not per step -- a split's two sides are both the answer."""
        spec = _spec()
        assert spec.order() == ORDER

        trace = run_pipeline(spec, sources=_sources(), retain="all")

        assert isinstance(trace, PipelineTrace)
        assert trace.pipeline == "a neutral fixture"
        assert tuple(event.step_id for event in trace.events) == ORDER

        # Keyed by port, in production order, and complete: the ports nobody consumes are there too,
        # because the rows a split discarded are a result and not a leftover.
        assert tuple(trace.shapes) == PORTS
        assert tuple(trace.frames) == PORTS

        assert_frame_equal(
            trace.frame(PortRef("kept", "true")),
            pl.DataFrame({"entry_id": [1, 3, 5], "region": ["north", "south", "east"], "amount": [10.0, 7.5, 4.0]}),
        )
        assert_frame_equal(
            trace.frame(PortRef("kept", "false")),
            pl.DataFrame({"entry_id": [2, 4], "region": ["north", "south"], "amount": [-5.0, -2.5]}),
        )
        assert_frame_equal(
            trace.frame(PortRef("paired", "joined")),
            pl.DataFrame(
                {
                    "entry_id": [1, 3, 5],
                    "region": ["north", "south", "east"],
                    "amount": [10.0, 7.5, 4.0],
                    "header_code": ["A", "B", "C"],
                }
            ),
            check_row_order=False,
        )
        assert trace.frame(PortRef("paired", "left_only")).height == 0
        assert_frame_equal(
            trace.frame(PortRef("paired", "right_only")),
            pl.DataFrame({"entry_id": [9], "header_code": ["D"]}),
        )
        assert_frame_equal(
            trace.frame(PortRef("rated", "rows")),
            pl.DataFrame(
                {"region": ["north", "south", "east"], "total": [10.0, 7.5, 4.0], "share": [0.1, 0.075, 0.04]}
            ),
            check_row_order=False,
        )

        # A shape is reported against the port that produced the frame, and it carries the column
        # names rather than a width, because a released port's schema is still worth drawing.
        assert trace.shapes[PortRef("trimmed", "rows")] == FrameShape(
            port=PortRef("trimmed", "rows"), rows=5, columns=("entry_id", "region")
        )

    def test_a_trace_is_keyed_by_port_and_can_be_read_again(self) -> None:
        """A trace is evidence, so reading it never consumes it and never changes the answer.

        The next cycle slices a trace to one entity, and a slicer reads the same run once per entity
        in a picker. It also has a counting defect to fix rather than reproduce: an entity that
        appears in eight frames was reported eight times over, because the count was summed across
        frames. Nothing here offers a cross-port total to sum -- every count is against one port, and
        which port it is is part of the answer.
        """
        spec = _spec()
        first = run_pipeline(spec, sources=_sources(), retain="all")
        second = run_pipeline(spec, sources=_sources(), retain="all")

        assert tuple(first.shapes.values()) == tuple(second.shapes.values())
        for port in PORTS:
            assert_frame_equal(first.frame(port), second.frame(port))

        # Read twice, same answer, same object: nothing is recomputed behind the accessor.
        assert first.frame(PortRef("kept", "true")) is first.frame(PortRef("kept", "true"))

        # A reader cannot write into the evidence.
        with pytest.raises(TypeError):
            first.frames[PortRef("kept", "true")] = _entries()  # type: ignore[index]

        # There is deliberately no single row count for a run: the same entity is one row at a
        # reduced grain and several at a detailed one, so a summed figure is wrong at every grain.
        assert not hasattr(first, "rows")
        assert not hasattr(first, "total_rows")

        unknown = PortRef("kept", "maybe")
        with pytest.raises(RunError) as raised:
            first.frame(unknown)
        assert "kept.maybe" in str(raised.value)

        with pytest.raises(RunError) as no_event:
            first.event("nobody")
        assert "nobody" in str(no_event.value)

    def test_each_step_goes_through_the_boundary_with_the_ports_its_edges_name(self, monkeypatch) -> None:
        """One ``run_tool`` call per executed step, with the frames its edges name and the stored config.

        The configuration is passed exactly as it was authored -- ``run_tool`` fills the defaults, at
        the one moment they are needed. A runner that filled them first would hand the boundary a
        configuration the file does not contain, and the trace would stop being an account of the
        file somebody is about to edit.

        A frame produced once is handed to every consumer of it, unchanged and uncopied: recomputing
        per consumer pays for the work as many times as a port has consumers and returns a
        plausible-looking trace either way.
        """
        calls: list[tuple[str, tuple[str, ...], list[pl.DataFrame], Mapping[str, Any]]] = []
        boundary = runner.run_tool

        def recording(tool, inputs, config):
            calls.append((tool.name, inputs.names, [frame for _, frame in inputs.items()], config))
            return boundary(tool, inputs, config)

        monkeypatch.setattr(runner, "run_tool", recording)

        spec = _spec()
        sources = _sources()
        run_pipeline(spec, sources=sources, retain="all")

        # The two supplied sources are never run: their tools open files that are not there.
        assert [name for name, _, _, _ in calls] == [
            "select_columns",
            "assert_rows",
            "filter",
            "join",
            "group_by",
            "formula",
            "assert_rows",
        ]

        by_tool = {name: (ports, frames, config) for name, ports, frames, config in calls}

        # As authored: 'nulls' and 'action' carry defaults the template would fill, and neither is
        # here, because neither was written.
        assert dict(by_tool["filter"][2]) == {"predicate": "[amount] > 0"}
        assert dict(by_tool["select_columns"][2]) == {"columns": ["entry_id", "region"]}
        assert dict(spec.step("kept").config) == {"predicate": "[amount] > 0"}, "the spec is read, never filled in"

        # A pairing is handed both of its declared ports, each carrying the frame its own edge names.
        assert by_tool["join"][0] == ("left", "right")
        assert by_tool["join"][1][1] is sources[PortRef("headers", "rows")]

        # Fan-out: one port, two consumers, the very same frame handed to each.
        assert by_tool["select_columns"][1][0] is sources[PortRef("entries", "rows")]
        assert by_tool["filter"][1][0] is sources[PortRef("entries", "rows")]


class TestFramesComeInRatherThanBeingRead:
    """A source port is supplied; a computed one may not be."""

    def test_a_source_is_supplied_rather_than_read_and_a_computed_port_may_not_be(self) -> None:
        """Both key spellings work, a supplied source is not executed, and an override is refused."""
        spec = _spec()

        # A port may be named by reference or by the text a file would carry; they are the same port.
        trace = run_pipeline(
            spec,
            sources={PortRef("entries", "rows"): _entries(), "headers.rows": _headers()},
            retain="all",
        )
        assert_frame_equal(trace.frame(PortRef("entries", "rows")), _entries())
        assert trace.event("entries").supplied is True
        assert trace.event("kept").supplied is False
        # Nothing opened 'sources/entries.csv', which does not exist: the step was not run.
        assert "suppl" in trace.event("entries").message.lower()

        with pytest.raises(RunError) as computed:
            run_pipeline(spec, sources={**_sources(), PortRef("kept", "true"): _entries()}, retain="all")
        assert "kept.true" in str(computed.value)
        assert "kept" in str(computed.value)

        with pytest.raises(RunError) as unknown_port:
            run_pipeline(spec, sources={**_sources(), PortRef("entries", "lines"): _entries()}, retain="all")
        assert "entries.lines" in str(unknown_port.value)
        assert "rows" in str(unknown_port.value), "the refusal must list the ports that step has"

        with pytest.raises(RunError) as unknown_step:
            run_pipeline(spec, sources={**_sources(), PortRef("entires", "rows"): _entries()}, retain="all")
        assert "entires" in str(unknown_step.value)

        # A source nobody supplies is read, exactly as it would be in production -- and its failure
        # reaches the caller inside the one error family, as a step that failed.
        with pytest.raises(StepFailedError) as unread:
            run_pipeline(spec, sources={PortRef("headers", "rows"): _headers()}, retain="all")
        assert unread.value.step_id == "entries"
        assert isinstance(unread.value.__cause__, ToolError)


class TestAnEventSaysWhatOneStepDid:
    """What the log carries, per step, for somebody who was not there when it ran."""

    def test_an_event_reports_a_row_count_for_every_output_port_not_one_number(self) -> None:
        """A split's two ports are the point; one figure for the step would hide half the answer."""
        trace = run_pipeline(_spec(), sources=_sources(), retain="all", run_id="run-7")

        event = trace.event("kept")
        assert event.pipeline == "a neutral fixture"
        assert event.run_id == "run-7"
        assert event.tool == "filter"
        assert event.lane == "shaping"
        assert event.stage == "shape"
        assert event.level == "info"
        assert event.error is None
        assert event.duration_ms >= 0.0
        assert event.at.tzinfo is not None, "a stamp with no zone cannot be correlated against a run log"

        # Every shape names the port that produced the frame, on both sides of the step -- so an
        # input shape is the same fact the trace already reports for its upstream port.
        assert event.inputs == (trace.shapes[PortRef("entries", "rows")],)
        assert event.inputs[0].port == PortRef("entries", "rows")
        assert event.outputs == (
            FrameShape(port=PortRef("kept", "true"), rows=3, columns=("entry_id", "region", "amount")),
            FrameShape(port=PortRef("kept", "false"), rows=2, columns=("entry_id", "region", "amount")),
        )

        # The one line a person reads has to carry both counts, or it has not said what the step did.
        assert "kept" in event.message
        assert "true" in event.message and "3" in event.message
        assert "false" in event.message and "2" in event.message

        # A pairing reports one input shape per declared input port, in the tool's own order.
        paired = trace.event("paired")
        assert tuple(shape.port for shape in paired.inputs) == (PortRef("kept", "true"), PortRef("headers", "rows"))
        assert tuple(shape.port for shape in paired.outputs) == (
            PortRef("paired", "joined"),
            PortRef("paired", "left_only"),
            PortRef("paired", "right_only"),
        )

        # A supplied source has nothing feeding it and one port coming out of it.
        assert trace.event("entries").inputs == ()
        assert trace.event("entries").outputs[0].rows == 5

    def test_an_empty_frame_something_reads_is_a_warning_and_one_nobody_reads_is_not(self) -> None:
        """An empty frame flowing onward is the quiet failure; an empty side port is ordinary.

        Every downstream step of an empty port computes nothing and reports success, which in a log
        is indistinguishable from a day when there was nothing to do. So it is said out loud, once,
        at the step that produced it -- and only where a step will actually read it, because a split
        that routed every row to one side has an empty other side by design.
        """
        trace = run_pipeline(_spec(), sources=_sources(entries=_nothing_positive()), retain="all")

        kept = trace.event("kept")
        assert kept.level == "warning"
        assert any("true" in note for note in kept.notes), "the note must name the port that came out empty"
        assert kept.outputs[1].rows == 2, "the port that is not empty is reported as usual"

        # It travels: every step that read an empty frame and produced one says so too.
        assert [trace.event(step).level for step in ("paired", "totals", "rated")] == [
            "warning",
            "warning",
            "warning",
        ]

        # And it stops at the end of the flow: nothing reads 'checked', so its empty output is not a
        # warning, and neither is the empty side of a split.
        assert trace.event("checked").level == "info"
        assert trace.frame(PortRef("checked", "rows")).height == 0

        everything_positive = pl.DataFrame({"entry_id": [1], "region": ["north"], "amount": [3.0]})
        quiet = run_pipeline(_spec(), sources=_sources(entries=everything_positive), retain="all")
        assert quiet.frame(PortRef("kept", "false")).height == 0
        assert quiet.event("kept").level == "info", "an empty port nobody consumes is not news"


class TestAFailureKeepsTheEvidence:
    """A step that raises stops the run, and everything that had already happened survives."""

    def test_a_failing_step_stops_the_run_and_leaves_the_trace_behind(self) -> None:
        """The one thing nobody can debug is a failure that discarded what led up to it.

        The boundary already guarantees that whatever a tool does arrives as a ``ToolError``, and
        which one it is is the reader's next move: a ``DataError`` says go and look at the rows, an
        ``ExecutionError`` says go and look at the run. The runner passes that distinction through
        untouched rather than flattening it into one 'step failed'.
        """
        bad_entry = pl.DataFrame({"entry_id": [1, 0], "region": ["north", "south"], "amount": [10.0, 5.0]})

        with pytest.raises(StepFailedError) as raised:
            run_pipeline(_spec(), sources=_sources(entries=bad_entry), retain="all")

        failure = raised.value
        assert isinstance(failure, ToolError), "a runner refusal stays inside the one error family"
        assert failure.step_id == "audited"

        cause = failure.__cause__
        assert isinstance(cause, DataError), "the tool's own diagnosis about the rows reaches the caller"
        assert not isinstance(cause, ExecutionError)
        assert cause.offending_rows == 1
        assert str(failure) == failure.event.message
        assert str(cause) in failure.event.message, "one failure, one wording, wherever it is read"

        trace = failure.trace
        assert trace.event("audited").level == "error"
        assert trace.event("audited").error is not None
        assert trace.event("audited").outputs == (), "a step that raised produced nothing"

        # Everything that got that far is still reachable, by port.
        assert tuple(trace.frames) == (
            PortRef("entries", "rows"),
            PortRef("headers", "rows"),
            PortRef("trimmed", "rows"),
        )
        assert_frame_equal(trace.frame(PortRef("trimmed", "rows")), bad_entry.select("entry_id", "region"))

        # And the run stopped: every step after the failure is absent from the trace entirely.
        assert tuple(event.step_id for event in trace.events) == ("entries", "headers", "trimmed", "audited")
        for step_id in ORDER[ORDER.index("audited") + 1 :]:
            with pytest.raises(RunError):
                trace.event(step_id)

    def test_a_column_a_frame_does_not_carry_is_caught_before_the_step_that_names_it_runs(self, monkeypatch) -> None:
        """The loader publishes what each expression names; the runner checks it the moment it can.

        Per step, as it is reached -- not once up front for the whole graph. No tool declares the
        columns it produces, so an up-front pass could only reach the steps whose inputs are supplied
        frames: one layer, and then a claim that the flow had been checked which would be false for
        every step after it. Checking each step against the frames that actually arrived is the
        version that is true everywhere, and it is still before the step runs, which is the property
        that matters.
        """
        entered: list[str] = []
        boundary = runner.run_tool

        def recording(tool, inputs, config):
            entered.append(tool.name)
            return boundary(tool, inputs, config)

        monkeypatch.setattr(runner, "run_tool", recording)

        spec = _spec(rated={"columns": [{"name": "share", "value": "[totl] / 100"}]})
        assert spec.column_references()["rated"]["columns[0].value"] == frozenset({"totl"})

        with pytest.raises(ColumnCheckError) as raised:
            run_pipeline(spec, sources=_sources(), retain="all")

        failure = raised.value
        assert failure.step_id == "rated"
        message = str(failure)
        assert "totl" in message
        assert "rated" in message
        assert "columns[0].value" in message, "the field to open, not merely the step"
        assert "total" in message, "the refusal must list the columns the frame does carry"

        # The tool was never entered: the step is refused before its own code runs at all.
        assert "formula" not in entered
        assert entered == ["select_columns", "assert_rows", "filter", "join", "group_by"]

        # Everything upstream still ran and is still reachable, which is what makes this debuggable.
        trace = failure.trace
        assert PortRef("totals", "rows") in trace.frames
        assert PortRef("rated", "rows") not in trace.frames
        assert [event.step_id for event in trace.events if event.level == "info"] == [
            "entries",
            "headers",
            "trimmed",
            "audited",
            "kept",
            "paired",
            "totals",
        ], "a late reference does not stop the steps before it from running: the check is per step"

    @pytest.mark.parametrize(
        ("step_id", "config", "placeholder", "until"),
        [
            pytest.param(
                "rated",
                {"columns": [{"name": "share", "value": "[total] / ${rules.divisor}"}]},
                "${rules.divisor}",
                None,
                id="a whole run, poisoned on a step it would have reached",
            ),
            pytest.param(
                "headers",
                {"path": "sources/${rules.folder}/headers.csv"},
                "${rules.folder}",
                "audited",
                id="a truncated run, poisoned on a step its closure never reaches",
            ),
        ],
    )
    def test_a_placeholder_that_survived_into_a_configuration_stops_the_run_before_it_starts(
        self,
        step_id: str,
        config: dict[str, Any],
        placeholder: str,
        until: str | None,
    ) -> None:
        """The loader should have caught it; a runner that trusts its input is a runner that ships one.

        An unresolved ``${...}`` compiles as ordinary text, matches nothing and leaves the run
        reporting success -- which is exactly why the tool boundary refuses one. But the boundary
        only sees it when the run reaches that step, and by then the steps before it have written
        wherever they write. It needs no frames, so it is swept for across the whole flow first.

        Across the whole flow, and not narrowed to the closure ``until`` will walk -- which is the
        second case. Narrowing would be the obvious economy and the wrong one: a placeholder that
        survived means the loader never finished resolving the document, which is a fact about the
        file rather than about this run. A scan that followed the closure would call the same file
        clean while the author was running the first ten steps and unresolved once they ran the
        next ten, for a reason nobody had touched -- and would hand back a trace of a run that
        succeeded over a configuration nobody can claim was the one in the file.

        The poisoned field there is a path rather than an expression, and deliberately so: an
        expression carrying a ``${...}`` is refused a second time over by the column pre-flight,
        which reads every step's expression fields whatever the closure, so poisoning one would
        prove only that *something* refused. A path is read by nobody until the step runs, and that
        step never runs -- under a narrowed scan the run would simply succeed.
        """
        spec = _spec()
        poisoned = replace(
            spec,
            steps=tuple(replace(step, config=config) if step.id == step_id else step for step in spec.steps),
        )

        seen: list[StepEvent] = []
        with pytest.raises(RunError) as raised:
            run_pipeline(poisoned, sources=_sources(), retain="all", until=until, sink=seen.append)

        message = str(raised.value)
        assert step_id in message
        assert placeholder in message
        assert not isinstance(raised.value, StepFailedError), "no step failed; the run never began"
        assert seen == [], "nothing ran, so nothing was reported: the refusal arrives before any effect"


class TestTwoModesOneAnswer:
    """Every intermediate frame for a walkthrough, or none that nobody will read, for a real run."""

    def test_a_live_run_releases_a_port_nobody_will_read_again_and_a_full_run_keeps_it(self) -> None:
        """Last use is computable from ``consumers()``, so it is computed rather than guessed at.

        A mode rather than a callback: the release schedule is a property of the wiring, and a
        caller handing one in could only get it wrong. It is behaviour, not a knob -- a port is
        released when no step will read it again, including a port no step ever would, because a
        terminal frame kept 'just in case' is the largest one in the run. A flow's results leave
        through its output tools; the trace is evidence, not a return value.
        """
        spec = _spec()
        live = run_pipeline(spec, sources=_sources(), retain="live")

        # Every shape survives -- how big a released frame was is exactly what a failure needs.
        assert tuple(live.shapes) == PORTS
        assert dict(live.frames) == {}

        released = {event.step_id: set(event.released) for event in live.events}
        assert released["entries"] == set(), "a port with a consumer still to run is not released"
        assert released["trimmed"] == set(), "the source still has a second consumer waiting"
        assert released["audited"] == {PortRef("trimmed", "rows"), PortRef("audited", "rows")}
        assert released["kept"] == {PortRef("entries", "rows"), PortRef("kept", "false")}
        assert released["paired"] == {
            PortRef("headers", "rows"),
            PortRef("kept", "true"),
            PortRef("paired", "left_only"),
            PortRef("paired", "right_only"),
        }
        assert released["checked"] == {PortRef("rated", "rows"), PortRef("checked", "rows")}

        # A released frame is gone, and says so by name rather than as an absence.
        with pytest.raises(RunError) as gone:
            live.frame(PortRef("kept", "true"))
        assert "kept.true" in str(gone.value)
        assert "live" in str(gone.value), "the refusal must name the mode that released it"

        kept_everything = run_pipeline(spec, sources=_sources(), retain="all")
        assert all(event.released == () for event in kept_everything.events)
        assert tuple(kept_everything.frames) == PORTS

        with pytest.raises(RunError):
            run_pipeline(spec, sources=_sources(), retain="every-other-one")

    def test_the_ports_released_at_one_step_are_listed_in_run_order_not_declaration_order(self) -> None:
        """``released`` is a tuple, and a tuple on a piece of evidence has to mean its order.

        Every other assertion about ``released`` in this file wraps it in ``set()``, and
        ``_comparable`` leaves it out, because *which* ports fall due at a step is the fact those
        tests are about. That leaves the sequence unpinned, and a sequence taken from ``spec.steps``
        rather than from the walk would render the same run differently for no reason but how the
        document was typed -- the very property this codebase refused for ``trace.until``, and a
        rendering of a run that changes when nothing about the run did.

        'checked' is the step that can tell the two apart. It frees the port it read and the port
        it produced, and their producers sit in opposite relative positions under the two orderings,
        because 'checked' is written first in the document and run last. 'audited' is the ordinary
        case, where the two orderings agree, and is pinned beside it so the tuple is not asserted
        only where it is strange.
        """
        spec = _spec()
        assert spec.order() != tuple(step.id for step in spec.steps), (
            "the fixture must declare its steps out of dependency order, or this test discriminates nothing"
        )

        live = run_pipeline(spec, sources=_sources(), retain="live")

        assert live.event("checked").released == (PortRef("rated", "rows"), PortRef("checked", "rows"))
        assert live.event("audited").released == (PortRef("trimmed", "rows"), PortRef("audited", "rows"))

    def test_both_retention_modes_answer_identically(self) -> None:
        """The property that matters is not which frames survive but that the answers do not change.

        A release schedule that freed a frame one step too early would not fail loudly; it would
        change a downstream number, or empty one. So the two modes are run over the same flow and
        every reported fact is compared -- every event, every count, every shape.
        """
        spec = _spec()
        every_frame = run_pipeline(spec, sources=_sources(), retain="all", run_id="run-9")
        live = run_pipeline(spec, sources=_sources(), retain="live", run_id="run-9")

        assert [_comparable(event) for event in live.events] == [_comparable(event) for event in every_frame.events]
        assert tuple(live.shapes.values()) == tuple(every_frame.shapes.values())


class TestTheLogIsWhatSurvivesTheRun:
    """An orchestrator's own logger is the lowest common denominator every consumer already has."""

    def test_a_sink_sees_every_event_as_it_happens_and_one_that_raises_does_not_stop_the_run(self, caplog) -> None:
        """Telemetry reports the run; it does not get to end it."""
        seen: list[StepEvent] = []
        trace = run_pipeline(_spec(), sources=_sources(), retain="all", sink=seen.append)

        assert [event.step_id for event in seen] == list(ORDER), "each event goes out as it is produced"
        assert tuple(seen) == trace.events, "the returned trace carries the same sequence"

        def hostile(event: StepEvent) -> None:
            raise RuntimeError(f"the sink is broken at {event.step_id}")

        with caplog.at_level(logging.WARNING, logger="dantron_etl.runner"):
            survived = run_pipeline(_spec(), sources=_sources(), retain="all", sink=hostile)

        assert tuple(event.step_id for event in survived.events) == ORDER
        assert tuple(survived.frames) == PORTS
        complaints = [record for record in caplog.records if record.name == "dantron_etl.runner"]
        assert complaints, "a sink that fails is reported, never silently dropped"
        assert "the sink is broken at entries" in complaints[0].getMessage()

    def test_the_logging_sink_routes_a_level_to_a_log_level_and_logs_the_message_verbatim(self, caplog) -> None:
        """An adapter that reworded an event would give a reader a second account to square with."""
        logger = logging.getLogger("neutral.fixture.run")
        nothing_positive = pl.DataFrame({"entry_id": [1, 2], "region": ["north", "south"], "amount": [-1.0, -2.0]})

        with caplog.at_level(logging.INFO, logger="neutral.fixture.run"):
            trace = run_pipeline(
                _spec(),
                sources=_sources(entries=nothing_positive),
                retain="all",
                sink=logging_sink(logger),
            )

        written = [record for record in caplog.records if record.name == "neutral.fixture.run"]
        assert [record.getMessage() for record in written] == [event.message for event in trace.events]
        assert {record.levelno for record in written} == {logging.INFO, logging.WARNING}
        assert written[ORDER.index("kept")].levelno == logging.WARNING
        assert written[ORDER.index("entries")].levelno == logging.INFO

        caplog.clear()
        bad_entry = pl.DataFrame({"entry_id": [1, 0], "region": ["north", "south"], "amount": [10.0, 5.0]})
        with caplog.at_level(logging.INFO, logger="neutral.fixture.run"):
            with pytest.raises(StepFailedError) as raised:
                run_pipeline(
                    _spec(),
                    sources=_sources(entries=bad_entry),
                    retain="all",
                    sink=logging_sink(logger),
                )

        last = [record for record in caplog.records if record.name == "neutral.fixture.run"][-1]
        assert last.levelno == logging.ERROR
        assert last.getMessage() == raised.value.event.message, "the log line and the exception are one text"


# ---------------------------------------------------------------------------
# Running part of a flow
#
# A pipeline is built by growing it, and a flow that can only be run whole cannot be grown: the
# steps that do not exist yet are wired to the ones that do, and the only way to look at step ten
# is to survive step six hundred. ``until`` names where to stop, and the run executes the ancestor
# closure of what is named -- the steps that feed it, and nothing else. Everything below is about
# that word "nothing": a step outside the closure must leave no event, no shape and no frame, or a
# reader of the trace cannot tell a step that was skipped from one that ran and produced nothing.
# ---------------------------------------------------------------------------


class TestUntilRunsPartOfAFlow:
    """The ancestor closure of the steps named, and nothing outside it."""

    @pytest.mark.parametrize(
        ("until", "expected"),
        [
            pytest.param(
                "paired",
                ("entries", "headers", "kept", "paired"),
                id="one step pulls in both of its own branches and neither of the others",
            ),
            pytest.param(
                "audited",
                ("entries", "trimmed", "audited"),
                id="a step on one branch leaves the other branch entirely unrun",
            ),
            pytest.param(
                ["trimmed", "audited"],
                ("entries", "trimmed", "audited"),
                id="naming a step and its own descendant runs each step once",
            ),
            pytest.param(
                ("audited", "kept"),
                ("entries", "trimmed", "audited", "kept"),
                id="two unrelated steps run the union of their ancestors",
            ),
            pytest.param(
                "checked",
                ("entries", "headers", "kept", "paired", "totals", "rated", "checked"),
                id="the last step in the file still leaves a side branch nothing feeds it unrun",
            ),
        ],
    )
    def test_until_runs_the_ancestor_closure_of_the_steps_named(
        self,
        until: str | list[str] | tuple[str, ...],
        expected: tuple[str, ...],
    ) -> None:
        """What runs is decided by the wiring, not by position in the file or in the walk order.

        Truncating at a position would be the easy implementation and the wrong one: the order is a
        topological sort with ties broken by declaration order, so a step that happens to be written
        early sits early in the walk whether or not the named step needs it. The last case is the
        one that separates the two readings -- 'checked' is the final step of the flow, and the
        assertion branch that hangs off 'trimmed' still does not run, because nothing on the way to
        'checked' reads it.

        A step named twice over, once directly and once as somebody else's ancestor, is still one
        step: the closure is a union, and a step that ran twice would write twice.
        """
        spec = _spec()

        trace = run_pipeline(spec, sources=_sources(), retain="all", until=until)

        assert tuple(event.step_id for event in trace.events) == expected
        assert tuple(dict.fromkeys(shape.port.step for shape in trace.shapes.values())) == expected

    def test_a_step_outside_the_closure_leaves_no_trace_of_having_run(self) -> None:
        """Absent, not empty. The distinction is the whole feature.

        A trace that carried a step with no rows against one that never ran would be read the same
        way by every reader downstream of it -- the walkthrough page, a log, a person at three in the
        morning -- and 'it produced nothing' and 'it was not asked to' are opposite facts. So the
        refusals the trace already makes by name are the assertion: a step outside the closure is
        refused by ``event`` exactly as a step that does not exist is, and its ports are refused by
        ``frame`` exactly as a misspelling is.
        """
        trace = run_pipeline(_spec(), sources=_sources(), retain="all", until="paired")

        for step_id in ("trimmed", "audited", "totals", "rated", "checked"):
            with pytest.raises(RunError) as no_event:
                trace.event(step_id)
            assert step_id in str(no_event.value)
            assert "paired" in str(no_event.value), "the refusal lists the steps that did run"

        for port in (PortRef("trimmed", "rows"), PortRef("totals", "rows"), PortRef("rated", "rows")):
            assert port not in trace.shapes
            assert port not in trace.frames
            with pytest.raises(RunError) as no_frame:
                trace.frame(port)
            assert str(port) in str(no_frame.value)

        # And what did run is whole: the closure's own ports are all there, frames included.
        assert tuple(trace.frames) == (
            PortRef("entries", "rows"),
            PortRef("headers", "rows"),
            PortRef("kept", "true"),
            PortRef("kept", "false"),
            PortRef("paired", "joined"),
            PortRef("paired", "left_only"),
            PortRef("paired", "right_only"),
        )

    def test_no_until_and_until_none_both_run_the_whole_flow(self) -> None:
        """The default has to stay exactly what it was, or every existing caller is a rewrite.

        ``None`` is spelled out as well as omitted, because a caller that computes its stopping
        point -- an asset that stops early in development and runs whole in production -- passes the
        value rather than choosing between two call sites.
        """
        spec = _spec()
        omitted = run_pipeline(spec, sources=_sources(), retain="all")
        explicit = run_pipeline(spec, sources=_sources(), retain="all", until=None)

        assert tuple(event.step_id for event in omitted.events) == ORDER
        assert [_comparable(event) for event in explicit.events] == [_comparable(event) for event in omitted.events]
        assert tuple(explicit.shapes) == PORTS
        assert tuple(explicit.frames) == PORTS

    def test_the_steps_that_ran_report_what_they_would_have_reported_in_a_whole_run(self) -> None:
        """A truncated run is the same run, stopped -- not a second, differently behaved runner.

        This is the property that would catch a closure computed over the wrong edges, or a step fed
        a frame from the wrong port because the walk was reindexed to the closure. Every reported
        fact about a step that ran is compared against the same step in a run of the whole flow:
        same tool, same level, same input shapes, same output shapes, same notes.
        """
        spec = _spec()
        whole = run_pipeline(spec, sources=_sources(), retain="all", run_id="run-11")
        part = run_pipeline(spec, sources=_sources(), retain="all", run_id="run-11", until="paired")

        for event in part.events:
            assert _comparable(event) == _comparable(whole.event(event.step_id))

        for port, shape in part.shapes.items():
            assert shape == whole.shapes[port]
            assert_frame_equal(part.frame(port), whole.frame(port))

    @pytest.mark.parametrize(
        ("supplied", "until", "step_id", "port_name"),
        [
            pytest.param(
                {"entries": _nothing_positive()},
                "kept",
                "kept",
                "true",
                id="a split whose onward side came out empty, and the split is the stopping point",
            ),
            pytest.param(
                {"headers": _headers().clear()},
                "headers",
                "headers",
                "rows",
                id="a source that arrived empty, in a closure of one supplied step",
            ),
        ],
    )
    def test_an_empty_port_is_still_a_warning_when_the_step_that_reads_it_is_outside_the_closure(
        self,
        supplied: dict[str, pl.DataFrame],
        until: str,
        step_id: str,
        port_name: str,
    ) -> None:
        """The quiet-failure detector has to keep working in the mode a pipeline is built in.

        An empty frame on a port a later step reads is the one thing a run says out loud that no
        row count gives away, and ``until`` is precisely the mode in which the reader of that port
        has not been written yet, or has been written and is not being run. Asking whether the
        empty port has a consumer *in this closure* rather than in the flow would silence the note
        exactly there: the growing edge of a pipeline, where an author runs the first ten steps,
        looks at them, and writes the next ten. The step would report ``info`` with no notes, and a
        logging sink would then write it at INFO, so neither the trace nor the log would carry it.

        This is the sibling of the release schedule's ruling and deliberately the opposite of it.
        Releasing a frame is a fact about this run -- a consumer that will not run cannot keep a
        frame alive, so the schedule asks the closure. A note is a fact about the flow: the wiring
        says something will read this port, and that it is empty is worth saying whether or not
        this particular run gets as far as the reader.

        The two cases are the two shapes the situation comes in: a computed port whose consumer is
        in the file but not in the run, and a supplied source in a one-step closure, where nothing
        downstream has run at all. Both frames keep the whole flow viable, so each part-run event is
        compared against the same step's event in a complete run -- which is the assertion, since
        the whole run and the truncated one must say the identical thing.
        """
        spec = _spec()
        sources = _sources(**supplied)
        whole = run_pipeline(spec, sources=sources, retain="all", run_id="run-15")
        part = run_pipeline(spec, sources=sources, retain="all", run_id="run-15", until=until)

        event = part.event(step_id)
        assert event.level == "warning", "a step at the growing edge produced nothing onward, and must say so"
        assert any(repr(port_name) in note for note in event.notes), "the note names the port that came out empty"
        assert port_name in event.message, "and the one line a log carries says it too"
        assert _comparable(event) == _comparable(whole.event(step_id)), "identical to the same step in a whole run"

        # Without this the case proves nothing: if the port's reader were inside the closure, the
        # note would be owed under every reading of the wiring and the test could not tell them
        # apart. The reader exists in the file, and it is not among the steps that ran.
        readers = spec.consumers(PortRef(step_id, port_name))
        assert readers, "a port nothing anywhere reads is not owed a note at all"
        ran = {other.step_id for other in part.events}
        assert not any(reader.step in ran for reader in readers), "every reader of it is outside this run"


class TestUntilIsRefusedWhereItCouldNotMeanAnything:
    """Both refusals arrive before a single step runs, because a run is not undoable."""

    @pytest.mark.parametrize(
        "until",
        [
            pytest.param("pared", id="a misspelt step id on its own"),
            pytest.param(["kept", "pared"], id="a misspelt step id beside a real one"),
        ],
    )
    def test_a_step_the_pipeline_does_not_have_is_refused_by_name(self, until: str | list[str]) -> None:
        """A typo in a stopping point would otherwise be the quietest failure this feature could have.

        Silently running nothing, or silently running everything, are both defensible readings of a
        name that matches no step, and both leave an author looking at a result that does not answer
        the question they asked. So it is refused the way the loader already refuses one -- naming
        the id that is wrong and listing the ids there are -- and refused even when a second, real
        step would have given the run something to do.
        """
        seen: list[StepEvent] = []

        with pytest.raises(RunError) as raised:
            run_pipeline(_spec(), sources=_sources(), retain="all", until=until, sink=seen.append)

        message = str(raised.value)
        assert "pared" in message
        for step_id in ORDER:
            assert step_id in message, "the refusal lists the steps that exist, so the typo is visible"
        assert seen == [], "nothing ran: the refusal arrives before any effect"

    @pytest.mark.parametrize(
        "until",
        [
            pytest.param([], id="an empty list"),
            pytest.param((), id="an empty tuple"),
            pytest.param(set(), id="an empty set"),
        ],
    )
    def test_an_empty_collection_is_refused_rather_than_read_as_run_nothing(self, until: object) -> None:
        """No author ever means 'run nothing', and a run that produced nothing cannot be read.

        An empty closure returns a trace with no events, no shapes and no frames -- which is
        indistinguishable from a run that failed on its first step, from a pipeline with no steps in
        it, and from a bug in whatever computed the collection. The value almost always arrives from
        a comprehension over something that turned out empty, and saying so is the only outcome that
        sends the author to the line that produced it.
        """
        seen: list[StepEvent] = []

        with pytest.raises(RunError) as raised:
            run_pipeline(_spec(), sources=_sources(), retain="all", until=until, sink=seen.append)

        message = str(raised.value)
        assert "until" in message
        assert "empty" in message.lower()
        assert not isinstance(raised.value, StepFailedError), "no step failed; the run never began"
        assert seen == []


class TestATruncatedRunSaysSoOnItsTrace:
    """A partial run drawn as a complete flow is a lie the trace itself has to prevent."""

    def test_a_trace_carries_the_steps_the_run_was_asked_to_stop_at(self) -> None:
        """The reader of a trace is not the caller of ``run_pipeline``, and cannot ask it anything.

        The walkthrough page renders a trace it was handed. Rendering a seven-step closure of a
        nine-step flow as the flow would quietly tell somebody that two steps do not exist -- the
        exact failure mode this codebase refuses everywhere else. So the trace states what it was
        asked for, and a whole run states the absence of any such request.

        The order is the order the run reached them, whatever order the caller wrote them in and
        whatever kind of collection carried them: a set has no order of its own, and a field on a
        piece of evidence that renders differently run to run is not evidence.
        """
        whole = run_pipeline(_spec(), sources=_sources(), retain="all")
        assert whole.until == (), "a whole run was asked for nothing, and says so"

        one = run_pipeline(_spec(), sources=_sources(), retain="all", until="paired")
        assert one.until == ("paired",), "a single id is still reported as the collection it stands for"

        listed = run_pipeline(_spec(), sources=_sources(), retain="all", until=["kept", "audited"])
        unordered = run_pipeline(_spec(), sources=_sources(), retain="all", until={"audited", "kept"})
        assert listed.until == ("audited", "kept"), "in the order the run reached them, not as written"
        assert unordered.until == listed.until

        # 'audited' and 'kept' are the pair on which dependency order and alphabetical order agree,
        # so between them they cannot say which of the two this field is in -- a field sorted by
        # name would satisfy every assertion above. 'trimmed' and 'audited' are the pair on which
        # they disagree: 'trimmed' feeds 'audited' and so is reached first, while 'a' sorts before
        # 't'. Written in each order, because a caller's own sequence is not the answer either.
        dependent = run_pipeline(_spec(), sources=_sources(), retain="all", until=["trimmed", "audited"])
        reversed_ = run_pipeline(_spec(), sources=_sources(), retain="all", until=["audited", "trimmed"])
        assert dependent.until == ("trimmed", "audited"), "the order the run reaches them, not alphabetical"
        assert reversed_.until == dependent.until, "and not the order the caller happened to write them in"


class TestASourceTheClosureNeverReaches:
    """Supply everything you have, run part of it -- the natural way to use this."""

    def test_a_source_outside_the_closure_is_accepted_and_simply_unused(self) -> None:
        """Refusing it would hand the caller back the very work this feature exists to do for them.

        Sources are assembled once, for the flow, and a caller who then truncates the run would have
        to compute the ancestor closure themselves just to know which of their own frames to stop
        passing -- and would have to recompute it every time the wiring changed. So an unused source
        is accepted in silence. What is still refused is a source naming a step that does not exist
        at all: that is a typo whatever the run was asked to reach, and ``until`` is no reason to
        stop saying so.
        """
        spec = _spec()

        trace = run_pipeline(spec, sources=_sources(), retain="all", until="audited")

        assert tuple(event.step_id for event in trace.events) == ("entries", "trimmed", "audited")
        assert PortRef("headers", "rows") not in trace.shapes, "the frame was offered; the step never ran"
        assert PortRef("headers", "rows") not in trace.frames

        with pytest.raises(RunError) as unknown_step:
            run_pipeline(
                spec,
                sources={**_sources(), PortRef("entires", "rows"): _entries()},
                retain="all",
                until="audited",
            )
        assert "entires" in str(unknown_step.value)


class TestReleasingFramesUnderUntil:
    """``retain="live"`` frees a port at its last use, and under ``until`` the last use moves."""

    def test_a_port_whose_only_consumer_is_outside_the_closure_is_released_not_held(self) -> None:
        """The schedule has to be computed over what will run, or the biggest frames are never freed.

        Last use is read from ``consumers()``, which answers for the whole wiring. Under a truncated
        run that answer is wrong in the one direction that matters: 'entries.rows' is last read by
        'kept', which is not in this closure and will never run, so a schedule built from the whole
        spec has that port falling due at a step the walk never reaches -- and holds the source frame,
        the largest one in the run, until the process exits. Computed over the closure, its last use
        is 'trimmed', and it goes there.

        Nothing downstream may be released early to buy that: 'trimmed.rows' is still read by
        'audited', so it survives 'trimmed' and goes at 'audited'.
        """
        spec = _spec()
        live = run_pipeline(spec, sources=_sources(), retain="live", until="audited")

        released = {event.step_id: set(event.released) for event in live.events}
        assert released["entries"] == set(), "'trimmed' has still to read it"
        assert released["trimmed"] == {PortRef("entries", "rows")}, "its only remaining consumer is outside the run"
        assert released["audited"] == {PortRef("trimmed", "rows"), PortRef("audited", "rows")}

        assert dict(live.frames) == {}, "a truncated live run holds nothing when it ends"
        assert tuple(live.shapes) == (
            PortRef("entries", "rows"),
            PortRef("trimmed", "rows"),
            PortRef("audited", "rows"),
        ), "every shape survives, exactly as in a whole live run"

        with pytest.raises(RunError) as gone:
            live.frame(PortRef("entries", "rows"))
        assert "entries.rows" in str(gone.value)
        assert "live" in str(gone.value)

    @pytest.mark.parametrize(
        "until",
        [
            pytest.param("paired", id="a closure in which truncation moves no release at all"),
            pytest.param("audited", id="a closure in which truncation genuinely relocates one"),
        ],
    )
    def test_both_retention_modes_answer_identically_under_until(self, until: str) -> None:
        """A release moved one step too early changes a number rather than raising, so it is compared.

        The same argument as for a whole run, and a sharper one here: the schedule is now computed
        from a different set of steps, which is exactly the kind of change that frees a frame a later
        step in the closure still needed.

        Both closures, because they are not the same test. Under 'paired' every port falls due
        exactly where it would have in a whole run, so the comparison holds however the schedule was
        computed. Under 'audited' a release actually moves -- 'entries.rows' is last read by 'kept'
        in the flow and by 'trimmed' in this closure -- which is the case where a schedule computed
        over the wrong set of steps frees a frame a step still to come was going to read.
        """
        spec = _spec()
        every_frame = run_pipeline(spec, sources=_sources(), retain="all", until=until, run_id="run-13")
        live = run_pipeline(spec, sources=_sources(), retain="live", until=until, run_id="run-13")

        assert [_comparable(event) for event in live.events] == [_comparable(event) for event in every_frame.events]
        assert tuple(live.shapes.values()) == tuple(every_frame.shapes.values())
        assert live.until == every_frame.until == (until,)


# ---------------------------------------------------------------------------
# Resuming a run from a cached port
#
# The other half of cache-and-run. ``until`` stops a run at a step; ``cached`` starts one there, by
# handing in the frame a previous run left on a computed port so that the steps which produced it do
# not run again. The saving is the ancestors: caching one port near the end of a fifty-step flow buys
# nothing if the forty-nine steps behind it still run.
#
# Every entry carries the fingerprint recorded when the frame was written, because the danger is not
# a cache that fails -- it is a cache that works against a pipeline somebody has since edited, and
# quietly reports a number produced by a configuration that no longer exists.
#
# In memory only, this cycle. Writing a cache to a file and reading it back is a separate question
# and nothing here touches a disk.
# ---------------------------------------------------------------------------


def _totals_frame() -> pl.DataFrame:
    """A frame for 'totals.rows' that no run of this flow could have produced from its sources.

    Deliberately unrecognisable: the regions and totals below appear in no source frame in this
    module, so a step that reads them can only have read them from the cache, and an implementation
    that quietly recomputed the step would produce the real numbers and fail loudly.
    """
    return pl.DataFrame({"region": ["west", "midlands"], "total": [200.0, 50.0]})


def _trimmed_frame() -> pl.DataFrame:
    """A frame for 'trimmed.rows', carrying the column the assertion downstream of it names."""
    return pl.DataFrame({"entry_id": [7, 8], "region": ["west", "midlands"]})


def _cache(spec: Pipeline, port: PortRef, frame: pl.DataFrame) -> dict[PortRef, CachedFrame]:
    """One cached port, carrying the fingerprint a previous run of ``spec`` would have written beside it."""
    return {port: CachedFrame(frame, spec.fingerprint(port))}


def _cache_every_port(spec: Pipeline, frames: Mapping[PortRef, pl.DataFrame]) -> dict[PortRef, CachedFrame]:
    """Every port of one step, each carrying its own fingerprint.

    A step may only be cached on all of its ports or none, so this is the shape a cache of any step
    that produces more than one frame actually has.
    """
    return {port: CachedFrame(frame, spec.fingerprint(port)) for port, frame in frames.items()}


def _kept_frames() -> dict[PortRef, pl.DataFrame]:
    """Both sides of the split, in the order its tool declares them, and deliberately not alike.

    Different heights as well as different rows, so a port handed the other one's frame is caught by
    a shape on the trace rather than only by somebody recognising the numbers. The entry ids appear
    in no source frame in this module, so either frame can only have come from the cache.
    """
    return {
        PortRef("kept", "true"): pl.DataFrame(
            {"entry_id": [7, 8], "region": ["west", "west"], "amount": [200.0, 50.0]}
        ),
        PortRef("kept", "false"): pl.DataFrame({"entry_id": [99], "region": ["midlands"], "amount": [-7.0]}),
    }


def _paired_frames() -> dict[PortRef, pl.DataFrame]:
    """All three branches of the pairing: one row, two rows and three rows, in declared order.

    Three distinct heights rather than three distinct sets of rows, because with three ports the
    question is which frame landed on which port, and a height answers it in one number.
    """
    return {
        PortRef("paired", "joined"): pl.DataFrame(
            {"entry_id": [7], "region": ["west"], "amount": [200.0], "header_code": ["A"]}
        ),
        PortRef("paired", "left_only"): pl.DataFrame(
            {"entry_id": [8, 9], "region": ["west", "east"], "amount": [50.0, 1.0]}
        ),
        PortRef("paired", "right_only"): pl.DataFrame({"entry_id": [11, 12, 13], "header_code": ["X", "Y", "Z"]}),
    }


def _pairable_headers() -> pl.DataFrame:
    """Headers that pair with the cached split above, so neither side of the join comes out empty.

    Both cached sides of the split have a partner here, which is what makes the pairing's answer a
    discriminator: whichever frame reaches its left port produces rows, and they are different rows.
    """
    return pl.DataFrame({"entry_id": [7, 8, 99], "header_code": ["A", "B", "Z"]})


def _empty_totals_frame() -> pl.DataFrame:
    """'totals.rows' with its real schema and no rows at all -- a previous run that found nothing."""
    return pl.DataFrame(schema={"region": pl.String, "total": pl.Float64})


class TestACachedPortStandsInForTheStepThatProducedIt:
    """A frame handed in for a computed port, and the steps behind it that therefore do not run."""

    @pytest.mark.parametrize(
        ("port", "frame", "expected"),
        [
            pytest.param(
                PortRef("totals", "rows"),
                _totals_frame(),
                ("entries", "trimmed", "audited", "totals", "rated", "checked"),
                id="the pairing and both of its branches fall away, and the source one of them shared does not",
            ),
            pytest.param(
                PortRef("trimmed", "rows"),
                _trimmed_frame(),
                ORDER,
                id="an ancestor two other steps still read is not saved by caching its only consumer",
            ),
        ],
    )
    def test_a_cached_step_does_not_run_and_nor_do_the_ancestors_nothing_else_needs(
        self,
        port: PortRef,
        frame: pl.DataFrame,
        expected: tuple[str, ...],
    ) -> None:
        """The point of the feature is the ancestors, not the one step whose answer was kept.

        'Run the first ten tools, look at them, then run the next ten' is worth doing because the
        first ten do not run again. An implementation that skipped only the cached step itself would
        save one step out of ten, still open every file the flow reads, and take just as long -- while
        reporting that it had resumed.

        The two cases are the two halves of the ruling, and the second is the one that stops the
        first from being implemented as 'drop every ancestor'. Caching 'totals' makes the pairing,
        the split and the headers source pointless, so none of them runs -- but 'entries' feeds the
        other branch as well, so it runs, and a run that dropped it would fail on a step that never
        asked to be resumed. Caching 'trimmed' saves nothing at all: its one ancestor is read by two
        further steps that are still going to run. A cache that buys nothing is not an error; it is
        just a cache that buys nothing, and refusing it would send an author hunting for a mistake
        they have not made.
        """
        spec = _spec()

        trace = run_pipeline(spec, sources=_sources(), retain="all", cached=_cache(spec, port, frame))

        assert tuple(event.step_id for event in trace.events) == expected

        # A step that did not run leaves nothing behind at all -- the same ruling ``until`` already
        # makes. 'Absent' and 'produced nothing' are opposite facts, and the trace refuses by name
        # rather than offering an empty frame that reads like a result.
        skipped = tuple(step_id for step_id in ORDER if step_id not in expected)
        for step_id in skipped:
            with pytest.raises(RunError) as no_event:
                trace.event(step_id)
            assert step_id in str(no_event.value)
        for absent in (p for p in PORTS if p.step in skipped):
            assert absent not in trace.shapes
            assert absent not in trace.frames

    def test_the_cached_frame_is_the_frame_the_next_step_reads(self) -> None:
        """Everything downstream must read it exactly as if the step had run, or this proves nothing.

        The frame handed in carries regions and totals that no source in this module contains, so the
        figures the last steps report are traceable to the cache and to nothing else. A runner that
        recorded the cached frame on the trace but recomputed the step for its consumers would satisfy
        a test that only read ``trace.frame``, and would quietly do all the work the cache was
        supposed to save.
        """
        spec = _spec()
        frame = _totals_frame()
        port = PortRef("totals", "rows")

        trace = run_pipeline(spec, sources=_sources(), retain="all", cached=_cache(spec, port, frame))

        assert trace.frame(port) is frame, "handed on unchanged and uncopied, the way a supplied source is"
        assert trace.shapes[port] == FrameShape(port=port, rows=2, columns=("region", "total"))
        assert_frame_equal(
            trace.frame(PortRef("rated", "rows")),
            pl.DataFrame({"region": ["west", "midlands"], "total": [200.0, 50.0], "share": [2.0, 0.5]}),
        )
        assert trace.event("rated").inputs == (trace.shapes[port],), "the shape it read is the cached port's"

    def test_a_resumed_run_reports_what_the_whole_run_reported_from_the_cached_port_onward(self) -> None:
        """A resumed run is the same run, resumed -- not a second, differently behaved runner.

        The frame cached here is the one a complete run actually produced, so every step after it is
        entitled to report exactly what it reported before: same level, same input shapes, same output
        shapes, same notes, same message. That is the property that catches a step fed from the wrong
        port because the walk was reindexed around the cache, or a shape recorded against the step
        that was skipped rather than the port it produced.
        """
        spec = _spec()
        whole = run_pipeline(spec, sources=_sources(), retain="all", run_id="run-21")
        port = PortRef("totals", "rows")

        resumed = run_pipeline(
            spec,
            sources=_sources(),
            retain="all",
            run_id="run-21",
            cached={port: CachedFrame(frame=whole.frame(port), fingerprint=spec.fingerprint("totals"))},
        )

        for step_id in ("rated", "checked"):
            assert _comparable(resumed.event(step_id)) == _comparable(whole.event(step_id))
        for reported, shape in resumed.shapes.items():
            assert shape == whole.shapes[reported]
            assert_frame_equal(resumed.frame(reported), whole.frame(reported), check_row_order=False)

        # The sources for the branch that no longer runs were still offered, and were still accepted
        # in silence: a caller assembles the frames a flow needs once, and works out neither the
        # closure nor the cache's effect on it in order to know which of their own frames to withhold.
        assert PortRef("headers", "rows") not in resumed.shapes

    def test_an_empty_cached_frame_a_later_step_reads_is_a_warning_the_way_a_computed_one_is(self) -> None:
        """A resumed run that carries no rows must read in a log exactly like a run that found none.

        The note on an empty port exists because every downstream step of one computes nothing and
        reports success, which in a log is indistinguishable from a day when there was nothing to do.
        A cached port makes that worse, not better: an author resuming from a frame they wrote out
        last week has no chance at all of noticing it came back empty, because they are not looking
        at the step that produced it -- that step did not run. So the note is about the frame on the
        port and not about how the frame got there.

        Compared against the run that reaches the same emptiness by computing it, level for level
        and note for note, because 'a warning is raised' is the weak half of the claim. The whole
        claim is that the warning travels: 'rated' reads an empty frame and says so, and 'checked'
        produces one nobody reads and does not.
        """
        spec = _spec()
        port = PortRef("totals", "rows")

        resumed = run_pipeline(spec, sources=_sources(), retain="all", cached=_cache(spec, port, _empty_totals_frame()))
        computed = run_pipeline(spec, sources=_sources(entries=_nothing_positive()), retain="all")

        totals = resumed.event("totals")
        assert totals.level == "warning", "a cached port that carries nothing is not a quiet day"
        assert any("rows" in note for note in totals.notes), "the note names the port that came out empty"
        assert "cache" in totals.message.lower(), "the one line a log carries still says where the answer came from"
        assert "no rows" in totals.message.lower(), "and says, in the same sentence, that it carried none"

        downstream = ("totals", "rated", "checked")
        assert [(resumed.event(step).level, resumed.event(step).notes) for step in downstream] == [
            (computed.event(step).level, computed.event(step).notes) for step in downstream
        ], "an empty frame is an empty frame, whether this run produced it or a previous one did"

    def test_a_cache_may_be_keyed_by_the_text_a_cache_index_would_hold(self) -> None:
        """The stated use is an index written to a file, and a file has no ``PortRef`` in it.

        A caller who writes down what they cached -- a port and a fingerprint per line -- reads it
        back as text, and having to reassemble a ``PortRef`` before a run would accept it is the
        kind of friction that ends with people passing the frames around by hand instead. ``sources``
        already takes either spelling; so does this, and for the same reason.
        """
        spec = _spec()
        port = PortRef("totals", "rows")

        trace = run_pipeline(
            spec,
            sources=_sources(),
            retain="all",
            cached={"totals.rows": CachedFrame(_totals_frame(), spec.fingerprint("totals"))},
        )

        assert trace.cached == (port,), "read as the port it names, and reported as one"
        assert trace.event("totals").cached is True
        assert_frame_equal(trace.frame(port), _totals_frame())
        assert trace.frame(PortRef("rated", "rows"))["region"].to_list() == ["west", "midlands"], (
            "the text key stood in for the same computation a PortRef key stands in for"
        )


class TestAMultiOutputStepIsCachedPortByPort:
    """Which frame landed on which port -- the question a step with one output cannot ask.

    Every other cache in this module is a single-output step, where 'the cached frame' and 'the
    frame for this port' are the same sentence and a runner that confused them could not be caught.
    A split and a pairing are where they come apart, and where the damage is: hand a join's
    unmatched rows on as its matched ones and the run reports success, the fingerprint matches, and
    the trace reads like an ordinary resumed run while every figure downstream is wrong.

    This is also the only place the *positive* half of the all-ports-or-none rule is exercised. A
    partial cache is refused further down; here every port of a step arrives and the run proceeds.
    """

    @pytest.mark.parametrize(
        ("frames", "sources", "expected"),
        [
            pytest.param(
                _kept_frames(),
                _sources(headers=_pairable_headers()),
                ORDER,
                id="a split, one side of which a pairing reads",
            ),
            pytest.param(
                _paired_frames(),
                _sources(),
                ("entries", "trimmed", "audited", "paired", "totals", "rated", "checked"),
                id="a pairing, whose three branches are three different answers",
            ),
        ],
    )
    def test_every_port_of_a_cached_step_keeps_its_own_frame(
        self,
        frames: Mapping[PortRef, pl.DataFrame],
        sources: dict[PortRef, pl.DataFrame],
        expected: tuple[str, ...],
    ) -> None:
        """One frame per port, each the one it was cached with, in the tool's declared order.

        The failure this is written against is a runner that takes the cached frames as a bag and
        hands the same one to every port -- which is indistinguishable from correct on every
        single-output step in this file, and on a two-output step is a join's unmatched rows
        travelling on as its matched ones.

        The three heights in the second case are the assertion: one, two and three rows, so the
        declared order of the ports and the contents of each are both readable off the shapes in one
        go, and no permutation of them passes.
        """
        spec = _spec()
        heights = [frame.height for frame in frames.values()]
        assert len(set(heights)) == len(heights), "the fixture proves nothing unless no two ports look alike"

        trace = run_pipeline(spec, sources=sources, retain="all", cached=_cache_every_port(spec, frames))

        assert tuple(event.step_id for event in trace.events) == expected

        step_id = next(iter(frames)).step
        event = trace.event(step_id)
        assert event.cached is True
        assert event.inputs == (), "it read nothing this run, whatever its tool declares it reads"
        assert event.outputs == tuple(
            FrameShape(port=port, rows=frame.height, columns=tuple(frame.columns)) for port, frame in frames.items()
        ), "one shape per declared port, against the port, in the order the tool declares them"

        for port, frame in frames.items():
            assert trace.frame(port) is frame, "handed on unchanged and uncopied"
            assert_frame_equal(trace.frame(port), frame)

        # Within a step as well as between steps: the trace names the ports in declared order, which
        # is the ordering a reader of a resumed walkthrough page is entitled to rely on.
        assert trace.cached == tuple(frames)

    def test_each_branch_of_a_cached_split_is_read_by_whatever_that_branch_feeds(self) -> None:
        """The frames have to reach the consumers, not merely be recorded against the right ports.

        A runner could store the cached frames port by port on the trace and still hand its
        consumers the wrong one, and every assertion in the test above would pass. So this follows
        the rows: the pairing reads 'kept.true', and the entries that come out of it -- 7 and 8, and
        not 99 -- are the ones cached on that port and on no other. The headers were chosen so that
        both sides have a partner, which means the wrong frame would produce rows too, and a
        different set of them, rather than an empty result that could have come from anywhere.
        """
        spec = _spec()
        # Handed over with the side nothing reads first, so that a runner reaching for whichever
        # frame the mapping happens to yield first reaches for the wrong one. Which port a frame
        # belongs to is written on the key; the order the keys arrived in says nothing at all.
        frames = _kept_frames()
        backwards = {port: frames[port] for port in reversed(list(frames))}

        trace = run_pipeline(
            spec,
            sources=_sources(headers=_pairable_headers()),
            retain="all",
            cached=_cache_every_port(spec, backwards),
        )

        assert trace.event("paired").inputs == (
            trace.shapes[PortRef("kept", "true")],
            trace.shapes[PortRef("headers", "rows")],
        ), "the left side reads the port it is wired to, not whichever frame the cache held first"

        assert trace.frame(PortRef("paired", "joined"))["entry_id"].to_list() == [7, 8]
        assert trace.frame(PortRef("paired", "left_only")).height == 0
        assert trace.frame(PortRef("paired", "right_only"))["entry_id"].to_list() == [99], (
            "99 was cached on the side nothing reads, so here it can only be the header that found no entry"
        )

        # And the figures at the end follow from that frame and from no other: 200 and 50 are the
        # amounts cached on 'true', and -7 is the one cached on 'false'.
        assert_frame_equal(
            trace.frame(PortRef("rated", "rows")),
            pl.DataFrame({"region": ["west"], "total": [250.0], "share": [2.5]}),
        )


class TestATraceSaysWhichPortsCameOutOfACache:
    """A step that ran and a step whose answer was kept are different facts, and read differently."""

    def test_a_cached_step_is_marked_as_such_and_is_not_the_same_fact_as_a_supplied_source(self) -> None:
        """A supplied source is ordinary; a cached step is a claim about a previous run.

        ``supplied`` already says 'this source was handed its frame rather than opening a file', which
        is how every run in this module works and is nobody's cause for concern. A cached computed
        port is a different statement: that a run which happened earlier, under a configuration this
        run has merely checked the fingerprint of, produced these rows. Folding the two into one flag
        would leave a reader of a walkthrough page unable to tell the routine case from the one that
        needs questioning, so the trace carries them separately.

        The event for a cached step reports no input shapes, because it read nothing: its inputs were
        not gathered and the steps that would have filled them did not run. A runner that reported the
        input shapes of a step that never read them would be inventing evidence.
        """
        spec = _spec()
        port = PortRef("totals", "rows")

        trace = run_pipeline(spec, sources=_sources(), retain="all", cached=_cache(spec, port, _totals_frame()))

        resumed = trace.event("totals")
        assert resumed.cached is True
        assert resumed.supplied is False, "it was not supplied as a source; it stood in for a step that ran once"
        assert resumed.inputs == (), "it read nothing this run"
        assert resumed.outputs == (trace.shapes[port],)
        assert resumed.level == "info"
        assert "cache" in resumed.message.lower(), "the one line a log carries says where the answer came from"
        assert "suppl" not in resumed.message.lower(), "and does not read like a supplied source"

        supplied = trace.event("entries")
        assert supplied.supplied is True
        assert supplied.cached is False, "a source handed a frame is not a claim about a previous run"

        ran = trace.event("rated")
        assert ran.cached is False and ran.supplied is False

        # The trace as a whole names the ports it took from a cache, in the order the run reached
        # them, so a reader holding only the trace can ask the question without walking the events.
        assert trace.cached == (port,)
        assert run_pipeline(spec, sources=_sources(), retain="all").cached == (), "a whole run rests on no cache"

        # Evidence about a previous run, not a value this run may edit on its way past.
        with pytest.raises(AttributeError):
            _cache(spec, port, _totals_frame())[port].frame = _entries()  # type: ignore[misc]

    def test_a_cached_run_is_identifiable_as_one_because_the_fingerprint_cannot_see_the_data(self) -> None:
        """The limit of the whole mechanism, pinned rather than papered over.

        A fingerprint covers configuration, not data. It cannot cover data: the expensive reads are
        exactly the ones an author wants to skip, so hashing what they returned would mean performing
        them, which is the work the cache exists to avoid. So a cache built from yesterday's database
        pull and resumed against today's is a wrong number this mechanism cannot see -- and the run
        below is that wrong number. Every configuration matches, nothing is refused, and the flow
        reports totals for two regions that appear in none of its sources.

        Which is why a cache is a development convenience for building a pipeline without re-running
        it, and not a production mechanism: a run that used one is not evidence about today's figures.
        The trace is what lets a reader enforce that, since it says on its own face that it rested on
        a cache -- so a report, a walkthrough page or a scheduled job can refuse it without having to
        ask the caller anything.
        """
        spec = _spec()
        port = PortRef("totals", "rows")

        resumed = run_pipeline(spec, sources=_sources(), retain="all", cached=_cache(spec, port, _totals_frame()))
        honest = run_pipeline(spec, sources=_sources(), retain="all")

        # Nothing refused it, and the figures are fiction: 'west' is in no source frame this module has.
        assert resumed.frame(PortRef("rated", "rows"))["region"].to_list() == ["west", "midlands"]
        assert honest.frame(PortRef("rated", "rows"))["region"].to_list() == ["north", "south", "east"]

        # And the one thing that tells them apart is on the trace, available to a reader who was not
        # there and cannot ask. This is the check a report is expected to make.
        assert resumed.cached, "a run that rested on a cache says so, and must not be read as evidence"
        assert not honest.cached, "a run that computed everything it reports is evidence"

    def test_two_cached_steps_are_reported_in_run_order_and_not_in_the_order_they_arrived(self) -> None:
        """``cached`` is a mapping, and the order a mapping was typed in is not a fact about a run.

        A trace reports the ports it resumed from in the order the run reached them, because that is
        the order everything else on the trace is in and a reader following a walkthrough page from
        top to bottom has no second ordering to match up. With one cached step the claim is
        untestable -- run order, declaration order and sorted order are one sequence -- so the two
        here are handed over backwards.
        """
        spec = _spec()
        trimmed = PortRef("trimmed", "rows")
        totals = PortRef("totals", "rows")

        trace = run_pipeline(
            spec,
            sources=_sources(),
            retain="all",
            cached={
                totals: CachedFrame(_totals_frame(), spec.fingerprint("totals")),
                trimmed: CachedFrame(_trimmed_frame(), spec.fingerprint("trimmed")),
            },
        )

        assert tuple(event.step_id for event in trace.events) == ("trimmed", "audited", "totals", "rated", "checked")
        assert trace.cached == (trimmed, totals), "the order the run reached them, not the order they were typed"

    def test_a_cache_behind_another_cache_is_never_reached_and_is_not_on_the_trace(self) -> None:
        """Caching A and B where A feeds B leaves A out of the run, which leaves it off the trace.

        It is the feature working, not a gap in it: the whole point of caching B is that nothing
        behind B runs, and A is behind B. But ``cached`` names both, so a reader could reasonably
        expect both on the trace, and the ruling has to be written down somewhere. What the trace
        says is what this run rested on, and it rested on B alone -- A was offered and never
        consulted, exactly like a cache for a step outside the closure.
        """
        spec = _spec()
        trimmed = PortRef("trimmed", "rows")
        audited = PortRef("audited", "rows")

        trace = run_pipeline(
            spec,
            sources=_sources(),
            retain="all",
            cached={
                trimmed: CachedFrame(_trimmed_frame(), spec.fingerprint("trimmed")),
                audited: CachedFrame(_trimmed_frame(), spec.fingerprint("audited")),
            },
        )

        ran = tuple(event.step_id for event in trace.events)
        assert "trimmed" not in ran, "its only consumer was itself cached, so nothing needed it"
        assert trace.cached == (audited,), "a cache the run never consulted is not something the run rested on"
        assert trimmed not in trace.shapes and trimmed not in trace.frames


class TestACacheIsRefusedWhereItCouldStandInForSomethingElse:
    """Every refusal arrives before the first step runs, because a run is not undoable."""

    @pytest.mark.parametrize(
        ("spec_changes", "fingerprint_step"),
        [
            pytest.param(
                # `max` rather than `mean`: the catalog has no `mean` and refuses one by name, so a
                # document naming it never loads and this case would have been measuring the
                # aggregate whitelist rather than the fingerprint. Any real edit serves -- what the
                # case needs is a configuration that differs, not a particular way of differing.
                {
                    "totals": {
                        "by": ["region"],
                        "aggregations": [{"name": "total", "function": "max", "value": "[amount]"}],
                    }
                },
                "totals",
                id="the cached step's own configuration was edited after the frame was written",
            ),
            pytest.param(
                {"kept": {"predicate": "[amount] > 5"}},
                "totals",
                id="an ancestor was edited -- one the resumed run would not even have run",
            ),
            pytest.param(
                {},
                "rated",
                id="a fingerprint copied from the wrong step of an otherwise untouched flow",
            ),
        ],
    )
    def test_a_fingerprint_that_no_longer_matches_is_refused_by_name_before_anything_runs(
        self,
        spec_changes: dict[str, Any],
        fingerprint_step: str,
    ) -> None:
        """This is the guard the whole feature rests on.

        A cached port standing in for a step that has since been edited is a wrong number nobody can
        see: the run succeeds, the trace is complete, every row looks plausible, and the figures were
        produced by a configuration that is no longer in the file. So the fingerprint is compared
        before the walk begins and the refusal names the port and says what changed -- an author who
        is told 'totals.rows' and 'the configuration that produced it has changed' knows both that
        their cache is stale and why, and can take it again.

        The second case is the one that separates this from a cheaper check. The edited step is
        'kept', which a run resuming from 'totals' does not run at all, so nothing in the run would
        ever touch the changed configuration -- and the frame it is standing in for was nonetheless
        produced by it. Only a fingerprint over the closure catches that.
        """
        recorded = _spec().fingerprint(fingerprint_step)
        edited = _spec(**spec_changes)
        port = PortRef("totals", "rows")
        seen: list[StepEvent] = []

        with pytest.raises(RunError) as raised:
            run_pipeline(
                edited,
                sources=_sources(),
                retain="all",
                sink=seen.append,
                cached={port: CachedFrame(_totals_frame(), recorded)},
            )

        message = str(raised.value)
        assert "totals.rows" in message, "the refusal names the port whose cache cannot be trusted"
        assert "configuration" in message and "changed" in message.lower(), "and says what went stale"
        assert seen == [], "refused before a single step ran, because a run that has written cannot be undone"

    @pytest.mark.parametrize(
        ("port", "fingerprint_step", "named"),
        [
            pytest.param(PortRef("totls", "rows"), "totals", ("totls", "totals"), id="a misspelt step id"),
            pytest.param(PortRef("totals", "row"), "totals", ("totals.row", "rows"), id="a port that step has not got"),
            pytest.param(
                PortRef("paired", "joined"),
                "paired",
                ("paired.left_only", "paired.right_only"),
                id="one port of a step that produces three",
            ),
        ],
    )
    def test_a_port_a_cache_cannot_stand_in_for_is_refused_the_way_a_source_already_is(
        self,
        port: PortRef,
        fingerprint_step: str,
        named: tuple[str, ...],
    ) -> None:
        """Two arguments hand frames to a run, and a typo must read the same way in either.

        The third case is the one this feature adds. A step with three output ports produced all
        three in one pass, and a cache holding one of them cannot stand in for it: the run would have
        to execute the step -- and its ancestors, and the reads they open -- to obtain the other two,
        which is the entire cost the cache was taken to avoid, and would then hold two answers for one
        step, one of them from a previous run. So a partial cache is refused, naming the ports still
        missing, exactly as a partially supplied source already is.
        """
        spec = _spec()
        seen: list[StepEvent] = []

        with pytest.raises(RunError) as raised:
            run_pipeline(
                spec,
                sources=_sources(),
                retain="all",
                sink=seen.append,
                cached={port: CachedFrame(_totals_frame(), spec.fingerprint(fingerprint_step))},
            )

        for fragment in named:
            assert fragment in str(raised.value)

        # Before the walk, not at the cached step's turn in it. The third case is where that stops
        # being pedantry: 'paired' sits six steps into this flow and six hundred into a real one, so
        # a partial cache noticed when the walk reached it would be noticed after every step in
        # front of it had run -- and after any write step among them had written. There is no
        # undoing that, which is why the check cannot live in the loop.
        assert seen == [], "refused before a single step ran"

    @pytest.mark.parametrize(
        ("cached_port", "supplied_port"),
        [
            pytest.param(
                PortRef("entries", "rows"),
                PortRef("headers", "rows"),
                id="the source the walk reaches first",
            ),
            pytest.param(
                PortRef("headers", "rows"),
                PortRef("entries", "rows"),
                id="a source the walk reaches second, with a step already behind it",
            ),
        ],
    )
    def test_caching_a_source_port_is_refused_and_points_at_sources_instead(
        self,
        cached_port: PortRef,
        supplied_port: PortRef,
    ) -> None:
        """One thing, one way to do it -- because two ways with different staleness rules rot.

        A source port is what ``sources`` is for, and the two arguments do not mean the same thing. A
        supplied source is the caller stating the input to this run, checked against nothing because
        there is nothing upstream of it to check. A cached port is a claim that a computation which
        already happened may be skipped. Letting a source arrive either way would give one frame two
        doors with two different guards on them, and every question afterwards -- was this run
        resumed, is this trace evidence -- would have to be asked twice and could be answered both
        ways at once. So this is refused, and the refusal names the argument that does want it.

        Both sources, because they sit at different places in the walk and only one of them can show
        *when* the refusal happens. 'entries' is the first step of this flow, so a check made at its
        turn in the loop and a check made before the loop are indistinguishable; 'headers' is the
        second, and a run that got as far as it has already emitted an event for a step that ran.
        """
        spec = _spec()
        seen: list[StepEvent] = []

        with pytest.raises(RunError) as raised:
            run_pipeline(
                spec,
                sources={supplied_port: _sources()[supplied_port]},
                retain="all",
                sink=seen.append,
                cached={cached_port: CachedFrame(_sources()[cached_port], spec.fingerprint(cached_port.step))},
            )

        assert str(cached_port) in str(raised.value)
        assert "sources" in str(raised.value), "the refusal names the argument that does take a source frame"
        assert seen == [], "refused before a single step ran, whichever end of the walk the port sits at"

    def test_a_key_that_names_no_port_at_all_is_refused_naming_cached_and_not_sources(self) -> None:
        """A key read out of a cache index is text somebody typed, and a step id is not a port.

        The wording is shared with ``sources`` because the three mistakes are the same three
        mistakes whichever argument they were made in. What must not be shared is which argument the
        refusal names: an author who wrote a bad key in ``cached`` and is sent to look at ``sources``
        goes and reads a correct line for a while.
        """
        spec = _spec()

        with pytest.raises(RunError) as raised:
            run_pipeline(
                spec,
                sources=_sources(),
                retain="all",
                cached={"totals": CachedFrame(_totals_frame(), spec.fingerprint("totals"))},
            )

        message = str(raised.value)
        assert "totals" in message and "step.port" in message, "it says what a key looks like"
        assert "cached" in message, "and names the argument the key was written in"
        assert "sources" not in message, "and not the other one, which has nothing wrong with it"


class TestCachedAndUntilCompose:
    """Cache one end, name the other, and what runs is the leg in between -- which is the feature."""

    def test_naming_a_stopping_point_and_caching_upstream_of_it_runs_only_what_is_between_them(self) -> None:
        """The Alteryx sentence, made literal: cache here, run to there, look at it, write the next leg.

        ``until`` alone still runs everything behind the stopping point; a cache alone still runs
        everything in front of it. Together they cut a flow at both ends, and the two steps that run
        below are the only two a person building this pipeline is looking at.
        """
        spec = _spec()
        port = PortRef("totals", "rows")

        trace = run_pipeline(
            spec,
            sources=_sources(),
            retain="all",
            until="rated",
            cached=_cache(spec, port, _totals_frame()),
        )

        assert tuple(event.step_id for event in trace.events) == ("totals", "rated")
        assert trace.until == ("rated",), "both facts are on the trace, because its reader cannot ask for either"
        assert trace.cached == (port,)

    def test_a_cache_the_closure_never_reaches_is_accepted_and_leaves_no_mark(self) -> None:
        """The same ruling ``sources`` already makes, for the same reason, plus one of its own.

        A caller assembles what they have -- frames, caches -- and then chooses how far to run. Making
        them work out which of their own cache entries the closure reaches would hand them back the
        computation this feature exists to do for them, and would have to be redone every time the
        wiring changed.

        The mark is the part that matters. This run computed every figure it reports from its sources,
        so it is evidence, and a trace that claimed otherwise merely because a cache was in the room
        would train every reader to ignore the flag.
        """
        spec = _spec()

        trace = run_pipeline(
            spec,
            sources=_sources(),
            retain="all",
            until="audited",
            cached=_cache(spec, PortRef("totals", "rows"), _totals_frame()),
        )

        assert tuple(event.step_id for event in trace.events) == ("entries", "trimmed", "audited")
        assert trace.cached == (), "offered, never reached: this run rested on nothing"

    @pytest.mark.parametrize(
        "cached",
        [
            pytest.param(None, id="stated as None by a caller that computes whether to resume"),
            pytest.param({}, id="an empty mapping, which is what a comprehension over no cache produces"),
        ],
    )
    def test_no_cache_is_exactly_the_run_there_was_before_this_feature(
        self,
        cached: dict[PortRef, object] | None,
    ) -> None:
        """Every existing caller must be untouched by this, event for event.

        An empty mapping is accepted rather than refused, unlike an empty ``until``. The two read
        differently: an empty ``until`` would mean 'run nothing', which is never what anybody meant,
        while an empty cache means 'resume from nowhere' -- which is a complete run, and is exactly
        what a caller who assembles a cache from whatever they happen to have on hand should get when
        they have none.
        """
        spec = _spec()
        before = run_pipeline(spec, sources=_sources(), retain="all", run_id="run-22")
        after = run_pipeline(spec, sources=_sources(), retain="all", run_id="run-22", cached=cached)

        assert [_comparable(event) for event in after.events] == [_comparable(event) for event in before.events]
        assert tuple(after.shapes) == PORTS
        assert tuple(after.frames) == PORTS
        assert after.cached == ()


class TestReleasingFramesUnderACache:
    """``until`` moves a port's last use by cutting the walk short; a cache moves it by removing steps."""

    def test_a_port_whose_last_consumer_the_cache_dropped_is_freed_and_both_modes_still_agree(self) -> None:
        """The release schedule has to be computed over the walk the cache leaves, not the one before it.

        ``until`` already has this invariant pinned, and a cache breaks it in the same direction for
        a different reason: 'entries.rows' is last read by 'kept' in the wiring, and caching 'totals'
        removes 'kept' from the run entirely. A schedule built before the cache was taken into
        account has the largest frame in the run falling due at a step the walk never reaches, so it
        is never freed and ``retain="live"`` quietly holds everything it promised to let go of --
        which on a real flow is the difference between a run that fits in memory and one that does
        not, and shows up as nothing at all on the trace.

        And the two modes are compared as they are everywhere else, because a release moved one step
        too early in the other direction frees a frame a later step still reads, which changes a
        number rather than raising.
        """
        spec = _spec()
        port = PortRef("totals", "rows")

        every_frame = run_pipeline(
            spec, sources=_sources(), retain="all", run_id="run-31", cached=_cache(spec, port, _totals_frame())
        )
        live = run_pipeline(
            spec, sources=_sources(), retain="live", run_id="run-31", cached=_cache(spec, port, _totals_frame())
        )

        assert [_comparable(event) for event in live.events] == [_comparable(event) for event in every_frame.events]
        assert live.cached == every_frame.cached == (port,)

        released = {event.step_id: set(event.released) for event in live.events}
        assert released["entries"] == set(), "'trimmed' has still to read it"
        assert released["trimmed"] == {PortRef("entries", "rows")}, "'kept' would have read it, and no longer runs"
        assert released["rated"] == {port}, "a cached frame is released like any other once nothing reads it"
        assert dict(live.frames) == {}, "a resumed live run holds nothing when it ends"
