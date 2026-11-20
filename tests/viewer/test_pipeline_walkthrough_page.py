"""Structural guards on the generated walkthrough page.

Every defect this page has shipped was invisible to the suite and obvious to a person looking at it:
an entity escaped as data, an octal escape eaten by Python, bytes read as the wrong encoding, two
steps drawn at one coordinate, and -- the one these tests exist for -- a section and a table wearing
the same class name, so selecting a step hid every table on the page.

None of those are testable by rendering and squinting. All of them are testable as *structure*: the
page either says what it claims to, or it does not. So these are assertions about the document rather
than about how it looks.

The page script is not tested here. It reads the data the page and the server hand it and builds
markup from it; it computes nothing a figure depends on. Reading its source with regular expressions
pinned how it was spelled, not what it did, so those tests were removed. The data it is handed is
pinned instead: here, in tests/viewer/test_server.py, and in tests/flows/test_command_end_to_end.py.

Built over the neutral fixture pipeline, which needs no database and no client data.
"""

import hashlib
import itertools
import random
import re
from collections import Counter
from contextlib import chdir
from pathlib import Path

import pytest

from dantron_etl.focus import FocusSpec
from dantron_etl.paths import project_root
from dantron_etl import REGISTRY, Pipeline, PortRef, Step, read_pipeline
from dantron_etl.spec import Container, Lane
from dantron_etl import PipelineTrace, run_pipeline

PROJECT_ROOT = project_root(__file__)

# `_canvas` is the layout itself, and the cases at the foot of this file are about the layout. The
# whole page cannot stand in for it on the real flow: `build` wants the frames a run produced, and
# the real flow reads a database this suite has no access to. Where the layout *is* reachable
# through the page -- the fixture flow -- the cases above go through `build`, and this is the seam
# for the one flow that cannot.
from dantron_etl.viewer import build  # noqa: E402
from dantron_etl.viewer.layout import canvas as _canvas  # noqa: E402


@pytest.fixture(scope="module")
def page(fixture_flow_sandbox: Path) -> str:
    """One rendered page, shared: building it is pure and costs nothing to reuse."""
    spec = read_pipeline(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    with chdir(fixture_flow_sandbox):  # the flow's sinks write under the working folder
        trace = run_pipeline(spec)
    return build(spec, trace, FocusSpec(label="party", identity=("party",)))


@pytest.fixture(scope="module")
def panels(fixture_flow_sandbox: Path) -> dict[str, str]:
    """Every step's data panel, keyed by the step it answers to, as the service will hand them over.

    Imported here rather than at the top of the file so that a suite run before the panels exist
    fails on the cases that are about them, rather than collecting nothing at all.
    """
    from dantron_etl.viewer import build_panels

    spec = read_pipeline(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    with chdir(fixture_flow_sandbox):
        trace = run_pipeline(spec)
    return build_panels(spec, trace, FocusSpec(label="party", identity=("party",)))


@pytest.fixture(scope="module")
def shown(page: str, panels: dict[str, str]) -> str:
    """Everything a reader can be put in front of: the page, and every panel it will fetch.

    The panels used to be inside the document, so the guarantees about what a reader sees -- a
    search box under every column, a flip only where there is something to flip to, a port table
    that carries no rows -- were assertions about the page. They are now assertions about the page
    *and* what arrives into it, which is the same set of facts about the same markup; the only thing
    that changed is which side of the fetch each fact lives on.
    """
    return page + "".join(panels.values())


@pytest.fixture(scope="module")
def steps() -> tuple[str, ...]:
    spec = read_pipeline(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    return tuple(spec.order())


@pytest.fixture(scope="module")
def wiring() -> Counter[tuple[str, str, str]]:
    """Every connection the pipeline file declares, as (upstream step, downstream step, port).

    A Counter rather than a set: one step may take the same upstream port on two of its own inputs,
    which is two lines on the canvas and has to stay two lines in the markup.
    """
    spec = read_pipeline(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    return Counter((ref.step, step.id, ref.port) for step in spec.steps for ref in step.inputs.values())


class TestOneStepAtATime:
    """The panes rest on one rule: exactly one section of each kind answers to each step."""

    def test_every_step_has_one_detail_section_in_the_page_itself(self, page: str, steps: tuple[str, ...]) -> None:
        """The configuration stays in the document: it is what the page can still answer offline."""
        details = re.findall(r'<section class="detail[^"]*" data-step="([a-z_]+)"', page)

        assert sorted(details) == sorted(steps)

    def test_every_step_has_exactly_one_data_panel_to_be_shown(
        self, panels: dict[str, str], steps: tuple[str, ...]
    ) -> None:
        """One panel per step, wherever it is built: two would race each other into the same pane."""
        data = re.findall(r'<section class="stepdata" data-step="([a-z_]+)"', "".join(panels.values()))

        assert sorted(data) == sorted(steps)

    def test_no_table_answers_to_a_section_name(self, shown: str) -> None:
        """The defect this file exists for.

        `<section class="data">` and `<table class="data">` were two different things wearing one
        name, so the selector that shows one step's section silently matched every table in the
        page and hid them all. The note sits outside the table, so a port reported "1-100 of 11,834
        rows" over nothing at all -- the failure looked like a data problem and was a naming one.
        """
        for name in ("detail", "stepdata"):
            tagged = re.findall(rf'<(\w+)[^>]*class="{name}[ "]', shown)
            assert set(tagged) <= {"section"}, f"{name!r} is worn by {set(tagged)}, not by sections alone"

    def test_the_page_names_the_step_it_opens_on(self, page: str, steps: tuple[str, ...]) -> None:
        """Opening on nothing would show a reader an empty right-hand pane and no reason why."""
        found = re.search(r'data-first="([a-z_]+)"', page)

        assert found is not None
        assert found.group(1) == steps[0]


class TestRowsComeFromTheService:
    """A port the service can serve carries no rows, so the page stays the size of the flow's shape."""

    def test_a_port_table_is_empty_and_names_the_port_to_fetch(self, shown: str) -> None:
        tables = re.findall(r'<table class="data"[^>]*data-port="([^"]+)"[^>]*>(.*?)</table>', shown, re.DOTALL)

        assert tables, "no port table was drawn at all"
        for port, body in tables:
            assert "." in port, f"{port!r} is not a step.port reference"
            assert "<tbody></tbody>" in body, f"{port} carries rows the service was going to serve"

    def test_every_column_offers_a_search_box(self, shown: str) -> None:
        """The search is per column, so a header without one is a column nobody can narrow."""
        tables = re.findall(r'<table class="data"[^>]*>(.*?)</table>', shown, re.DOTALL)

        for body in tables:
            headings = len(re.findall(r"<th>(?!<input)", body))
            boxes = len(re.findall(r'<input class="flt"', body))
            assert headings == boxes, "a column heading without a search box"


class TestTheFlowIsTheIndex:
    """Clicking a node is how a step is chosen, so every node has to be addressable."""

    def test_every_step_has_a_node_on_the_canvas(self, page: str, steps: tuple[str, ...]) -> None:
        nodes = set(re.findall(r'<a class="node[^"]*" id="node-([a-z_]+)"', page))

        assert nodes == set(steps)

    def test_two_steps_are_never_drawn_at_one_coordinate(self, page: str) -> None:
        """A canvas showing fewer tools than the flow has is a canvas that lies by omission."""
        svg = page[page.index("<svg") : page.index("</svg>")]
        boxes = re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="176"', svg)

        assert len(set(boxes)) == len(boxes), "two nodes share a position; one is hidden under the other"

    def test_only_a_step_with_inputs_offers_an_in_toggle(self, shown: str) -> None:
        """A source has no `in` to turn to, and a control that does nothing is worse than none."""
        sections = re.findall(r'<section class="stepdata" data-step="([a-z_]+)"(.*?)</section>', shown, re.DOTALL)
        spec = read_pipeline(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})

        for step_id, body in sections:
            has_inputs = bool(spec.step(step_id).inputs)
            has_flip = 'class="flip"' in body
            assert has_flip == has_inputs, f"{step_id}: flip={has_flip} but inputs={has_inputs}"


def test_the_page_declares_its_encoding(page: str) -> None:
    """Undeclared, a browser guesses, and a client's accented name arrives as three letters."""
    # A browser only honours the declaration within the first 1,024 bytes.
    assert '<meta charset="utf-8">' in page[:1024]


def test_no_control_character_reaches_the_page(page: str) -> None:
    """An escape eaten by one layer before the next could read it comes out as one of these."""
    assert not {c for c in page if ord(c) < 32 and c not in "\n\t\r"}


def test_no_entity_is_escaped_as_though_it_were_data(page: str) -> None:
    """`&amp;middot;` is an entity that was run through the escaper: markup escaped as data."""
    assert not re.search(r"&amp;[a-z]+;", page)


def test_the_generated_file_can_be_written_and_read_back(tmp_path: Path, page: str) -> None:
    """utf-8 end to end, which is the other half of declaring the encoding."""
    out = tmp_path / "walkthrough.html"
    out.write_text(page, encoding="utf-8")

    assert out.read_text(encoding="utf-8") == page


class TestTheFlowCanBeZoomed:
    """A fifty-three step canvas has to be seeable whole and readable close up, which are two sizes."""

    def test_the_drawing_states_its_own_size_so_zooming_is_a_multiplication(self, page: str) -> None:
        """Kept on the element rather than measured, so scaling never has to re-render anything."""
        found = re.search(r'<svg viewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*data-w="([\d.]+)" data-h="([\d.]+)"', page)

        assert found is not None, "the canvas does not carry its own dimensions"
        view_w, view_h, stated_w, stated_h = found.groups()
        assert (view_w, view_h) == (stated_w, stated_h), "the stated size must be the coordinate system's"

    def test_the_zoom_control_offers_both_directions_and_a_way_back(self, page: str) -> None:
        """Zoom with no way back is a control a reader gets lost in."""
        offered = set(re.findall(r'<button data-zoom="(\w+)"', page))

        assert offered == {"in", "out", "fit", "reset"}


class TestAnyPaneCanBeFoldedAway:
    """So any one of the three can have the whole window when it is the one being read."""

    def test_each_pane_has_a_control_that_folds_it(self, page: str) -> None:
        offered = set(re.findall(r'<button data-fold="(\w+)"', page))

        assert offered == {"flow", "detail", "data"}


# ---------------------------------------------------------------------------
# Following one connection across a canvas nobody can read by eye
# ---------------------------------------------------------------------------
#
# The flow this page is really for draws 672 steps and some 900 connections. At that size the canvas
# answers "what shape is this" and nothing else: a reader who wants to know where one port goes has
# to follow a 1.4px line behind six other tools, and cannot. So a connection has to be something the
# page can talk about -- selecting a step lights everything that reaches it and everything it
# reaches, and clicking one line lights that line alone.
#
# The markup contract these tests pin:
#
#     <g class="edge" data-from="entries" data-to="receivable" data-port="rows">
#       <path class="wire" d="M..."/>      the line a reader sees
#       <path class="hit"  d="M..."/>      the same curve, wide and transparent, to click
#       <text class="elab" ...>rows &middot; 6</text>
#     </g>
#
# One group, so the line and its label take a state class together and never light apart; the ends
# named as data, so a script can ask which edges touch a step without re-deriving the flow; the same
# `d` on the hit path, so the target a hand aims at is where the eye sees the line.
#
# Four states, carried on the group and on the node: `direct` (touching the chosen step), `upstream`
# (somewhere in what feeds it), `downstream` (somewhere in what it feeds), `dim` (everything else).


def _svg(page: str) -> str:
    return page[page.index("<svg") : page.index("</svg>")]


def _edges(page: str) -> list[tuple[dict[str, str], str]]:
    """Every drawn connection as (its attributes, its contents)."""
    found = []
    for tag, body in re.findall(r'<g class="edge[^"]*"([^>]*)>(.*?)</g>', _svg(page), re.DOTALL):
        found.append((dict(re.findall(r'([a-z-]+)="([^"]*)"', tag)), body))
    return found


class TestAConnectionIsSomethingThePageCanName:
    """Today a line is two anonymous siblings, so no script can ask which lines touch a step."""

    def test_every_declared_connection_is_drawn_and_names_its_two_ends(
        self, page: str, wiring: Counter[tuple[str, str, str]]
    ) -> None:
        """The canvas graph must be the pipeline's graph, edge for edge.

        Reading the ends off the markup is what makes tracing possible at all, so a missing or
        mis-attributed edge is not a cosmetic fault: it is a chain that stops halfway, with nothing
        to tell the reader that anything was left out.
        """
        drawn = Counter(
            (attrs.get("data-from", ""), attrs.get("data-to", ""), attrs.get("data-port", ""))
            for attrs, _ in _edges(page)
        )

        assert drawn == wiring

    def test_an_edges_named_ends_are_the_ids_the_nodes_answer_to(self, page: str) -> None:
        """The script looks an endpoint up as '#node-' + the attribute, so the two have to be equal.

        A slug, a label, or a `step.port` reference in `data-from` all look perfectly reasonable in
        the markup and find nothing at all when the highlight goes to light the endpoint.
        """
        nodes = set(re.findall(r'<a class="node[^"]*" id="node-([a-z_]+)"', page))
        edges = _edges(page)

        assert edges, "no connection names its ends, so this would pass by having nothing to check"
        for attrs, _ in edges:
            for end in ("data-from", "data-to"):
                assert attrs.get(end) in nodes, f"{attrs.get(end)!r} on {end} is not a node on the canvas"

    def test_the_line_and_its_label_are_one_element(self, page: str) -> None:
        """A path and a text that light separately are how a lit chain reads as a broken one.

        Grouping is also what lets the state live in one place: a class on the group restyles the
        curve, its hit target and its caption at once, instead of three elements to keep in step.
        """
        edges = _edges(page)

        assert edges, "no connection is drawn as an element with an identity"
        for attrs, body in edges:
            assert len(re.findall(r'<path class="wire"', body)) == 1, f"{attrs}: not exactly one visible line"
            assert len(re.findall(r'<text class="elab"', body)) == 1, f"{attrs}: not exactly one label"
        loose = re.findall(r'<text class="elab"', _svg(page))
        assert len(loose) == len(edges), "a label is drawn outside the group whose state it should share"

    def test_a_line_offers_a_target_a_hand_can_actually_hit(self, page: str) -> None:
        """`fill:none` means a path takes clicks on its stroke alone -- here, 1.4 pixels of it.

        A reader picking one line out of a crowded area is aiming at a hair. The wide transparent
        twin is the usual answer, and it has to follow the same curve: a hit target that is not
        where the line is drawn is worse than none, because it lights a neighbour instead.
        """
        edges = _edges(page)

        assert edges, "no connection is drawn as a group, so there is nothing to give a target to"
        for attrs, body in edges:
            wire = re.search(r'<path class="wire" d="([^"]+)"', body)
            hit = re.search(r'<path class="hit" d="([^"]+)"', body)
            assert hit is not None, f"{attrs}: no hit target, so the line is a 1.4px target"
            assert wire is not None and hit.group(1) == wire.group(1), f"{attrs}: the target is not on the line"


#: The states the page's script puts on a node or a connection when a reader traces a step.
TRACE_STATES = frozenset({"direct", "upstream", "downstream", "dim"})


class TestThePageOpensShowingTheWholeFlow:
    """The trace answers a question, and on arrival the reader has not put one.

    Measured on the flow this page is for, choosing a late step lights 427 nodes of 672. A page that
    opened traced would show the canvas two-thirds lit and one-third dark, which reads as a drawing
    that failed to finish.
    """

    def test_no_element_arrives_already_wearing_a_trace_state(self, page: str) -> None:
        """The other way to open dimmed, and it does not go through the script at all.

        A state baked into the markup survives every assertion about what the script does, and the
        canvas opens with part of itself faded and no question behind it. The states start on the
        page the moment a reader asks for one, and not before.
        """
        worn = set()
        for value in re.findall(r'class="([^"]*)"', page):
            worn |= set(value.split()) & TRACE_STATES

        assert not worn, f"the page is rendered with {sorted(worn)} already on it, so it opens part-dimmed"


class TestThereIsAWayBackToTheWholeCanvas:
    """Six hundred dimmed nodes with no undo is a worse page than the one this replaces."""

    def test_a_control_offers_to_clear_the_trace(self, page: str) -> None:
        found = re.search(r'<button[^>]*data-trace="clear"[^>]*>', page)

        assert found is not None, "nothing on the page says how to get the whole canvas back"
        assert "title=" in found.group(0), "an unlabelled glyph is not an obvious way back"

    def test_the_clear_control_sits_outside_the_zoom_bar(self, page: str) -> None:
        """The zoom handler takes every button inside `.zoom` and reads `data-zoom` off it.

        A clear control placed there with no `data-zoom` would zoom the canvas out on its way to
        clearing.
        """
        bar = re.search(r'<div class="zoom".*?</div>', page, re.DOTALL)

        assert 'data-trace="clear"' in page, "there is no clear control, so this invariant has nothing to hold"
        assert bar is not None, "the page draws no zoom bar"
        assert 'data-trace="clear"' not in bar.group(0), (
            "the clear control sits inside .zoom, where the zoom handler claims every button it finds"
        )


# ---------------------------------------------------------------------------
# The panel is built for the step being opened, not for all six hundred and seventy-two
# ---------------------------------------------------------------------------
#
# Measured on a real client's page of 672 steps, the data panels were
# 7.76 MB of an 8.98 MB document: 24,937 search boxes and 52,654 header cells, every one of them
# laid out by the browser before it could paint a single pixel, for the 671 steps the reader was not
# looking at. The table BODIES were already fetched on demand; the headers and the per-column search
# box wrapped around them were not, and they are the weight.
#
# So a panel is written beside the run's ports, and fetched when its step is opened. The reader must
# see exactly what they saw before once it arrives -- same tables, same headers, same search boxes,
# same in/out flip, same lazy bodies -- which is why the guarantees about that markup moved onto the
# `panels` fixture above rather than being weakened. This is a change to WHEN the panel is built.
#
# The tests below hold the other half: that the page carries almost none of it, and that every step
# the flow can select has a panel to fetch. That the server answers each of those fetches with the
# right panel is pinned against a real server in tests/flows/test_command_end_to_end.py.


def _weight(markup: str) -> dict[str, int]:
    """The three counts that made the real page slow, over whatever markup is handed in.

    Header cells and search boxes are counted inside the data tables alone: a step's configuration
    is drawn with `<th>` too, and it stays in the page, so counting every `<th>` in the document
    would mix the thing that moved with the thing that did not.
    """
    tables = re.findall(r'<table class="data".*?</table>', markup, re.DOTALL)
    return {
        "tables": len(tables),
        "headers": sum(len(re.findall(r"<th\b", table)) for table in tables),
        "boxes": sum(len(re.findall(r'<input class="flt"', table)) for table in tables),
    }


def _shipped(page: str) -> list[str]:
    """The steps whose panels the page carries in the document, rather than fetching."""
    return re.findall(r'<section class="stepdata" data-step="([a-z_]+)"', page)


class TestThePageNoLongerCarriesEveryPanel:
    """The whole point, stated as the numbers that made the real page unusable."""

    def test_the_page_ships_no_panel_beyond_the_one_it_opens_on(self, page: str, panels: dict[str, str]) -> None:
        """Six hundred and seventy-one of the 672 were for steps the reader had not asked about.

        Exactly one, and it is the step named by ``data-first``: the step the page opens on is
        chosen before any fetch can have answered, so shipping its panel is the difference between
        opening on data and opening on a pane that fills in a moment later. Any other panel in the
        document is weight for a step nobody has clicked, which is the defect.

        Stated as an equality rather than as "no more than one", because every assertion in the
        cheaper spelling -- at most one, a subset of the opening step, and a loop over what was
        carried -- is satisfied by a page carrying nothing at all. That page is the other defect,
        and it is the one a reader meets first: opened as a plain file it shows the notice saying
        the service is not answering where the first step's tables used to be.

        The byte-identity check below is the only place in this file where markup from ``build``'s
        pass over the views meets markup from ``build_panels``' separate pass, so it has to run on
        every page rather than only on a page that happened to ship something. Two passes that keyed
        their views differently would put the picker's buttons and the panels' ``data-for`` keys out
        of step, and the reader would see a panel open with every view of it hidden.
        """
        carried = _shipped(page)
        first = re.search(r'data-first="([a-z_]+)"', page)

        assert first is not None, "the page does not say which step it opens on"
        assert carried == [first.group(1)], (
            f"the page ships panels for {carried} and opens on {first.group(1)!r}; it must ship that one and "
            "only that one -- shipping more is weight for steps nobody clicked, shipping none is a page that "
            "opens on an empty pane"
        )
        assert panels[first.group(1)] in page, (
            f"the panel shipped for {first.group(1)} is not the one the fetch would bring, so a reader sees one "
            "thing on arrival and another after they click away and back"
        )

    def test_the_search_boxes_and_header_cells_leave_the_page_with_the_panels(
        self, page: str, panels: dict[str, str]
    ) -> None:
        """24,937 form controls is what Chrome was laying out before it could paint.

        Counted rather than looked for: a page that still built every header and every search box
        but hid them would satisfy every assertion about markup in this file and be exactly as slow,
        because the cost is the layout and not the pixels. So the page's data tables must be
        accounted for, one for one, by the panels it admits to shipping -- nothing else may be in
        there at all.
        """
        carried = _weight(page)
        accounted = {
            key: sum(_weight(panels[step])[key] for step in _shipped(page)) for key in ("tables", "headers", "boxes")
        }
        everything = {
            key: sum(_weight(panel)[key] for panel in panels.values()) for key in ("tables", "headers", "boxes")
        }

        assert carried == accounted, (
            f"the page carries {carried} but its shipped panels account for {accounted}; the difference is "
            "markup for steps the reader has not opened"
        )
        assert everything["boxes"] > 0, "no panel carries a search box, so this would pass by having nothing to count"
        assert carried["boxes"] * 4 <= everything["boxes"], (
            f"the page still carries {carried['boxes']} of the {everything['boxes']} search boxes; on the real "
            "flow that ratio is the 24,937 controls the browser laid out before it could paint"
        )

    def test_every_step_the_reader_can_pick_has_a_panel_to_fetch(
        self, page: str, panels: dict[str, str], steps: tuple[str, ...]
    ) -> None:
        """A step the list offers but whose panel cannot be had is a dead end found by clicking.

        Both ways of choosing a step are checked against the same set, because they are separate
        markup built from separate passes: a panel dictionary keyed off the trace's events rather
        than the spec's order would leave a node on the canvas that answers with nothing.
        """
        offered = set(re.findall(r'<button data-goto="([a-z_]+)"', page))
        drawn = set(re.findall(r'<a class="node[^"]*" id="node-([a-z_]+)"', page))

        assert set(panels) == set(steps)
        assert offered <= set(panels), f"the step list offers {sorted(offered - set(panels))} with no panel behind it"
        assert drawn <= set(panels), f"the canvas draws {sorted(drawn - set(panels))} with no panel behind it"

    def test_a_panel_is_one_section_that_can_be_put_straight_into_the_page(
        self, panels: dict[str, str], steps: tuple[str, ...]
    ) -> None:
        """What arrives is inserted as markup, so it has to be markup and nothing around it.

        A panel written as a whole document -- a doctype, a `<meta>`, the stylesheet again -- looks
        perfectly fine in a file and inserts a second page into a pane. And it has to arrive hidden,
        the way it was rendered when it lived in the document, because the selection rule that shows
        one step's section works by clearing that attribute on the one it wants.
        """
        for step in steps:
            panel = panels[step]
            assert re.fullmatch(rf'<section class="stepdata" data-step="{step}"[^>]*>.*</section>', panel, re.DOTALL), (
                f"{step}'s panel is not a single section answering to that step"
            )
            assert _shipped(panel) == [step], f"{step}'s panel carries panels for {_shipped(panel)}"
            assert re.match(rf'<section class="stepdata" data-step="{step}"[^>]*\bhidden\b', panel), (
                f"{step}'s panel does not arrive hidden, so inserting it shows two steps' data at once"
            )
            for stray in ("<html", "<!doctype", "<meta", "<script", "<style"):
                assert stray not in panel.lower(), f"{step}'s panel carries {stray}: that is a document, not a panel"

    def test_nothing_the_reader_needs_without_the_service_moved_into_a_panel(
        self, page: str, panels: dict[str, str], steps: tuple[str, ...]
    ) -> None:
        """The flow, the step list and the configuration have to survive with nothing answering.

        The page is written to disk and opened as a file. Before this change it was self-contained;
        after it, the rows are the only part that needs the service -- and the temptation, while
        moving one thing out of the document, is to move the step's configuration with it, since it
        sits in the same pane. That would leave a file-opened page that cannot say what any step
        does, which is most of what the page is for.
        """
        for step in steps:
            assert f'data-step="{step}"' in page, f"{step} has nothing in the page at all"
            assert re.search(rf'<section class="detail[^"]*" data-step="{step}"', page), (
                f"{step}'s configuration is no longer in the document, so a page opened as a file cannot say "
                "what that step does"
            )
        for step, panel in panels.items():
            for carried in ('class="node', '<g class="edge', "data-goto=", '<section class="detail'):
                assert carried not in panel, (
                    f"{step}'s panel carries {carried!r}; the flow, the step list and the configuration are "
                    "what the page keeps when nothing is serving it, and a copy inside a panel means one of "
                    "them arrives twice or not at all"
                )


# ---------------------------------------------------------------------------
# Containers: a few dozen titled boxes instead of six hundred and seventy-two
#
# The flow this page is really for draws 672 steps. Nobody follows that by reading it. What a reader
# can follow is a few dozen boxes with sentences on them -- "fetch the account list and put
# the header row back" over the six tools that do it -- opened only when the sentence is not enough.
#
# The markup contract these tests pin:
#
#     <g class="group" id="group-classify" data-group="classify" data-rows-in="7" data-rows-out="7">
#       <rect class="gbox" .../>                 the box, drawn round exactly its own steps
#       <rect class="gpill" .../>                the same container closed, one node's worth
#       <text ...>Split the entries ...</text>   the title, readable in both states
#       <text ...>7 in &middot; 7 out</text>     what entered and what left
#       <g ... data-collapse="classify">...</g>  the control that closes it and opens it again
#       <a class="node" id="node-receivable">...</a>    its steps, and nobody else's
#       <a class="node" id="node-set_aside">...</a>
#     </g>
#
# The steps are drawn *inside* the container element rather than beside it, and the rest follows from
# that: enclosure becomes something the document states rather than something the coordinates imply,
# and closing a container becomes one class on one element instead of a pass over six hundred nodes
# asking each whether it belongs to the box being closed.
#
# A connection stays where it is -- at the top of the drawing, beside its siblings -- because the
# trace lifts a lit connection above the steps, and a connection living inside a container would be
# lifted out of the box it belongs to and never come back. Instead every connection names the
# container at each of its ends, which is what lets a closed container hide the connections that were
# internal to it and re-point the ones that crossed its edge.
#
# The counts. *In* is every port produced outside a container and read inside it, named once however
# many members read it. *Out* is every port of a member read by a non-member or by nobody at all --
# the second half matters, because a container ending in a write would otherwise report that nothing
# came out of it. A port used up inside the box is the container's own plumbing and is counted at
# neither end, which is the difference between a reader checking a section without opening it and a
# reader being handed a number that agrees with nothing.
# ---------------------------------------------------------------------------

#: The containers the cases below declare over the fixture flow. Two of its six steps are left in no
#: container at all, which is the state every step starts in and has to keep working: containers are
#: adopted a stage at a time, not all at once.
#:
#: The ampersand in the first title is deliberate. A title is a person's sentence, and one carrying
#: an `&` that reaches the drawing unescaped ends the text there.
GROUP_TITLES: dict[str, str] = {
    "classify": "Split the entries into revenue & what was set aside",
    "totalling": "Spread each amount over its months and total it by party",
}
GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    "classify": ("receivable", "set_aside"),  # lane 'shaping'
    "totalling": ("monthly", "totals"),  # lane 'figures'
}


def _grouped_spec() -> Pipeline:
    """The fixture flow with two of its stages boxed, read from the same file the page renders.

    Built by reading the shipped pipeline and adding the containers in memory rather than by editing
    the file on disk: every other test in this repository runs against that file, and a fixture that
    quietly grew a feature underneath them would be a change nobody reviewed.
    """
    import yaml

    from dantron_etl import load_pipeline

    document = yaml.safe_load((PROJECT_ROOT / "pipelines" / "fixture.yaml").read_text(encoding="utf8"))
    document["containers"] = [{"id": group_id, "title": title} for group_id, title in GROUP_TITLES.items()]
    for group_id, members in GROUP_MEMBERS.items():
        for step in document["steps"]:
            if step["id"] in members:
                step["container"] = group_id
    return load_pipeline(document, rules={"revenue_type": "REVENUE"})


@pytest.fixture(scope="module")
def boxed(fixture_flow_sandbox: Path) -> tuple[Pipeline, PipelineTrace]:
    """The grouped pipeline and one run of it, which the counts on the boxes are read off."""
    spec = _grouped_spec()
    with chdir(fixture_flow_sandbox):
        return spec, run_pipeline(spec)


@pytest.fixture(scope="module")
def boxed_page(boxed: tuple[Pipeline, PipelineTrace]) -> str:
    spec, trace = boxed
    return build(spec, trace, FocusSpec(label="party", identity=("party",)))


def _elements(markup: str, opener: str) -> list[tuple[dict[str, str], str]]:
    """Every element whose start tag begins with ``opener``, as (its attributes, its contents).

    Written as a scan rather than as a non-greedy regex because a container holds the very elements
    it draws. A non-greedy match stops at the first closing tag it meets, which is the end of
    something nested inside, and every enclosure assertion below would then be made about a fragment
    of the container -- passing or failing for reasons that have nothing to do with the container.
    """
    tag = opener[1:].split()[0]
    found: list[tuple[dict[str, str], str]] = []
    for start in re.finditer(re.escape(opener), markup):
        head = markup.index(">", start.start())
        attrs = dict(re.findall(r'([a-z-]+)="([^"]*)"', markup[start.start() : head]))
        depth, at = 1, head + 1
        while depth:
            nxt = re.search(rf"<{tag}\b|</{tag}>", markup[at:])
            assert nxt is not None, f"{opener} is never closed"
            depth += -1 if nxt.group(0).startswith("</") else 1
            at += nxt.end()
        found.append((attrs, markup[head + 1 : at - len(f"</{tag}>")]))
    return found


def _containers(page: str) -> dict[str, tuple[dict[str, str], str]]:
    """Every drawn container, keyed by the group it answers to."""
    return {attrs.get("data-group", ""): (attrs, body) for attrs, body in _elements(_svg(page), '<g class="group"')}


def _rect(markup: str) -> tuple[float, float, float, float]:
    """The first rectangle in ``markup`` as (x, y, width, height)."""
    found = re.search(
        r'<rect[^>]*\bx="([\d.-]+)"[^>]*\by="([\d.-]+)"[^>]*\bwidth="([\d.]+)"[^>]*\bheight="([\d.]+)"', markup
    )
    assert found is not None, f"no rectangle in {markup[:120]!r}"
    x, y, width, height = (float(value) for value in found.groups())
    return x, y, width, height


def _node_boxes(page: str) -> dict[str, tuple[float, float, float, float]]:
    """Where every step's box is drawn, by step id, read off the drawing itself."""
    return {
        attrs.get("id", "").removeprefix("node-"): _rect(body)
        for attrs, body in _elements(_svg(page), '<a class="node')
    }


def _own_rects(body: str) -> list[tuple[float, float, float, float]]:
    """The rectangles a container drew for itself -- its steps' boxes and its control's taken out.

    A container draws its state: the outline round its steps, and the one box it becomes when it is
    closed. Those are what the title has to sit on. A step's own box, and whatever the control that
    closes it is drawn with, are not the container saying anything about itself.
    """
    bare = re.sub(r'<a class="node.*?</a>', "", body, flags=re.DOTALL)
    bare = re.sub(r"<g[^>]*data-collapse.*?</g>", "", bare, flags=re.DOTALL)
    return [_rect(f"<rect{tag}>") for tag in re.findall(r"<rect([^>]*)>", bare) if "data-collapse" not in tag]


def _encloses(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[0] + outer[2] >= inner[0] + inner[2]
        and outer[1] + outer[3] >= inner[1] + inner[3]
    )


def _overlaps(one: tuple[float, float, float, float], other: tuple[float, float, float, float]) -> bool:
    return (
        one[0] < other[0] + other[2]
        and other[0] < one[0] + one[2]
        and one[1] < other[1] + other[3]
        and other[1] < one[1] + one[3]
    )


def _boundary(spec: Pipeline, group_id: str) -> tuple[tuple[PortRef, ...], tuple[PortRef, ...]]:
    """What entered a container and what left it, worked out here from the wiring alone.

    Derived in the test rather than taken from the graph's own answer to the same question: a page
    that faithfully renders whatever the library hands it and a library that was asked the wrong
    question agree with each other perfectly, and this is the one figure on the box a reader is
    invited to trust without opening it.
    """
    members = tuple(step.id for step in spec.steps if step.container == group_id)
    entered: list[PortRef] = []
    for step_id in members:
        for source in spec.step(step_id).inputs.values():
            if source.step not in members and source not in entered:
                entered.append(source)
    left: list[PortRef] = []
    for step_id in members:
        for port in spec.registry.get(spec.step(step_id).tool).output_ports:
            read_by = spec.consumers(PortRef(step_id, port))
            if not read_by or any(ref.step not in members for ref in read_by):
                left.append(PortRef(step_id, port))
    return tuple(entered), tuple(left)


class TestAContainerIsOneBoxWithASentenceOnIt:
    """Six boxes a reader has to follow, replaced by one they can read."""

    def test_every_declared_container_is_drawn_once_and_carries_its_title(
        self, boxed_page: str, boxed: tuple[Pipeline, PipelineTrace]
    ) -> None:
        """A title is the feature. An id over six boxes is the same canvas with a line drawn on it.

        Compared against the declared title escaped the way this page escapes every other piece of a
        person's text, because that is what a title is: written with an `&` in it and put into the
        drawing raw, the text node ends there and the reader is shown half a sentence.
        """
        import html

        spec, _ = boxed
        drawn = _containers(boxed_page)

        assert set(drawn) == set(GROUP_TITLES), f"the containers drawn are {sorted(drawn)}"
        assert {group.id: group.title for group in spec.containers} == GROUP_TITLES
        for group_id, title in GROUP_TITLES.items():
            attrs, body = drawn[group_id]
            said = re.findall(r"<text[^>]*>(.*?)</text>", body)
            assert said.count(html.escape(title, quote=True)) == 1, (
                f"{group_id}: the container's own text is {said}, which is not its title written once"
            )
            assert attrs.get("id") == f"group-{group_id}", (
                f"{group_id}: a container with no id of its own is one nothing can open, close or point at"
            )

    def test_a_container_encloses_its_own_steps_and_no_others(self, boxed_page: str) -> None:
        """Both senses of enclosing, because the box makes its claim in both.

        In the document, because that is what makes closing a container one class on one element
        rather than a pass over six hundred nodes asking each whether it belongs to the box.

        And on the canvas, because a box is a visual claim about what is inside it. A rectangle drawn
        round five of a container's six steps says nothing is missing; one drawn over a stranger says
        that stranger is part of the job. Neither announces itself -- the drawing looks finished
        either way, and a reader believes the line.
        """
        boxes = _node_boxes(boxed_page)
        containers = _containers(boxed_page)

        assert set(containers) == set(GROUP_TITLES), "nothing is drawn to enclose anything"
        inside_any: set[str] = set()
        for group_id, (_, body) in containers.items():
            held = {attrs.get("id", "").removeprefix("node-") for attrs, _ in _elements(body, '<a class="node')}
            assert held == set(GROUP_MEMBERS[group_id]), f"{group_id} holds {sorted(held)}"
            inside_any |= held

            around = [box for box in _own_rects(body) if all(_encloses(box, boxes[member]) for member in held)]
            assert around, f"{group_id}: no rectangle of its own is drawn round all of its steps"
            box = max(around, key=lambda drawn: drawn[2] * drawn[3])
            strangers = sorted(at for at, drawn in boxes.items() if at not in held and _overlaps(box, drawn))
            assert not strangers, f"{group_id}: the box is drawn over {strangers}, which are not in it"

        assert inside_any == {member for members in GROUP_MEMBERS.values() for member in members}
        assert set(boxes) - inside_any, "every step is in a container, so nothing here proves the ungrouped case"

    def test_a_step_in_no_container_is_drawn_as_it_always_was(
        self, page: str, boxed_page: str, steps: tuple[str, ...]
    ) -> None:
        """The property that lets containers be adopted a stage at a time rather than all at once.

        Most of a 672-step flow will be in no container for a long while, and those steps have to be
        drawn exactly as they are drawn today -- no box, no indent, no reserved space. A canvas that
        put a container round every ungrouped step, or one box round the lot of them, would make the
        first person who tries this feature undo it.

        The second half is the risk that comes with drawing a step inside a container: a node
        rendered inside the box and again outside it draws twice, and the copy nobody sees is found
        by clicking the one that turns out not to be wired to anything.
        """
        assert not _containers(page), "a pipeline that declares no groups still drew a container"
        assert '<g class="group"' not in page, "an empty container element is drawn for a flow that has none"
        assert set(_containers(boxed_page)) == set(GROUP_TITLES), (
            "a container is drawn for something other than a declared group, or one of them is missing"
        )

        drawn = re.findall(r'<a class="node[^"]*" id="node-([a-z_]+)"', boxed_page)
        assert sorted(drawn) == sorted(steps), "boxing some steps drew one of them twice, or dropped one"

    def test_a_container_says_what_entered_it_and_what_left_it(
        self, boxed_page: str, boxed: tuple[Pipeline, PipelineTrace]
    ) -> None:
        """The figure that lets a reader check a section without opening it.

        Which is what makes a wrong one worse than none: a container reporting more rows out than a
        reader can find anywhere downstream sends them hunting for rows that never existed. Both
        wrong answers are plausible. Summing every member's ports counts the container's own plumbing
        -- `receivable.false` goes to `set_aside` inside the same box -- and counting only the ports
        some later step reads loses the rows a container wrote to a file and stopped.

        So the figures are derived from the wiring and the trace, and then checked to be *different*
        from what the first of those mistakes would produce. An assertion that agrees with the
        mistake it exists to catch is worse than no assertion, because it is also a claim that the
        rule is covered.
        """
        spec, trace = boxed
        containers = _containers(boxed_page)

        assert set(containers) == set(GROUP_TITLES), "no container reports anything"
        for group_id, (attrs, body) in containers.items():
            entered, left = _boundary(spec, group_id)
            rows_in = sum(trace.shapes[port].rows for port in entered)
            rows_out = sum(trace.shapes[port].rows for port in left)

            assert attrs.get("data-rows-in") == str(rows_in), (
                f"{group_id}: {attrs.get('data-rows-in')} in, not {rows_in}"
            )
            assert attrs.get("data-rows-out") == str(rows_out), (
                f"{group_id}: {attrs.get('data-rows-out')} out, not {rows_out}"
            )
            said = " ".join(re.findall(r"<text[^>]*>(.*?)</text>", body))
            assert re.search(rf"\b{rows_in}\b", said) and re.search(rf"\b{rows_out}\b", said), (
                f"{group_id}: the counts are in the markup and in nothing a reader can see: {said!r}"
            )

        naive = {
            group_id: sum(shape.rows for port, shape in trace.shapes.items() if spec.step(port.step).container == group_id)
            for group_id in GROUP_TITLES
        }
        honest = {
            group_id: sum(trace.shapes[port].rows for port in _boundary(spec, group_id)[1]) for group_id in GROUP_TITLES
        }
        assert naive != honest, "no container has any plumbing of its own, so this case proves nothing about the rule"


class TestAContainerOpensAndCloses:
    """Closed, it is one box with a sentence on it. Open, it is the six tools that do the work."""

    def test_each_container_offers_a_control_that_closes_it(self, boxed_page: str) -> None:
        """A container with no control of its own is a box the reader cannot close or open."""
        containers = _containers(boxed_page)

        assert set(containers) == set(GROUP_TITLES), "no container is drawn, so no control can be offered"
        for group_id, (_, body) in containers.items():
            assert re.findall(r'data-collapse="([^"]*)"', body) == [group_id], (
                f"{group_id} is not drawn with exactly one control, naming itself, that closes it"
            )

    def test_the_title_sits_on_every_box_the_container_draws(self, boxed_page: str) -> None:
        """A box with nothing written on it is a section of the flow that has become unreadable.

        A container draws two boxes: its outline when open and the one box it becomes when closed.
        The title has to sit on both, or one of the two states shows a rectangle with no sentence.
        """
        import html

        containers = _containers(boxed_page)
        assert set(containers) == set(GROUP_TITLES), "no container is drawn, so no title can be lost"

        for group_id, (_, body) in containers.items():
            title = html.escape(GROUP_TITLES[group_id], quote=True)
            found = re.search(rf"<text([^>]*)>{re.escape(title)}</text>", body)
            assert found is not None, f"{group_id}: no text element carries the title"
            at = (
                float(re.search(r'\bx="([\d.-]+)"', found.group(1)).group(1)),
                float(re.search(r'\by="([\d.-]+)"', found.group(1)).group(1)),
            )

            drawn = _own_rects(body)
            assert drawn, f"{group_id}: the container draws no box of its own"
            outside = [
                box for box in drawn if not (box[0] <= at[0] <= box[0] + box[2] and box[1] <= at[1] <= box[1] + box[3])
            ]
            assert not outside, (
                f"{group_id}: the title does not sit on {outside}, so whichever state draws that box shows a "
                f"rectangle with nothing written on it"
            )


class TestAClosedContainerStillJoinsUpToTheFlow:
    """A connection into a step that is no longer drawn is a flow that appears to stop dead at a box."""

    def test_every_connection_names_the_container_at_each_of_its_ends(
        self, boxed_page: str, boxed: tuple[Pipeline, PipelineTrace]
    ) -> None:
        """The fact a closed container re-wires itself from, so it is read off the drawing.

        An end in no container names none, and has to name it as nothing rather than by leaving the
        attribute off altogether: `undefined === undefined` is true in a browser, so two ungrouped
        ends would compare equal, every connection between two ordinary steps would be read as the
        internals of a container that does not exist, and all of them would go out the moment any
        container was closed.
        """
        spec, _ = boxed
        expected = Counter(
            (source.step, step.id, source.port, spec.step(source.step).container or "", step.container or "")
            for step in spec.steps
            for source in step.inputs.values()
        )
        drawn = Counter(
            (
                attrs.get("data-from", ""),
                attrs.get("data-to", ""),
                attrs.get("data-port", ""),
                attrs.get("data-from-group", "∅"),
                attrs.get("data-to-group", "∅"),
            )
            for attrs, _ in _edges(boxed_page)
        )

        assert drawn == expected


# ---------------------------------------------------------------------------
# Clicking into a container brings its title forward
# ---------------------------------------------------------------------------
#
# A container's title is written at the top left of its own band, and a lane writes its name at x=6
# in the gap above its stripe. On the fixture flow these cases are drawn over, those two baselines
# land a pixel apart -- the lane names at y=21, 153 and 285, the two container titles at y=154 and
# y=286 -- and on a dense flow a container whose topmost step is in the first column starts its
# title at x=28, across the first twenty-odd characters of the lane's name. The lane name wins, on
# purpose: it is painted last of everything, over a halo, because it is the one label on the canvas
# a reader cannot recover from anything else on screen.
#
# So the title loses, and the reader who wants to read it has nothing to do about it. This is what
# they do about it: clicking into a container brings THAT container's title forward over a backing
# for as long as they are reading it, and clicking it again, or clicking blank canvas, puts it back.
#
# Two facts make this harder than a class. SVG has no z-index -- paint order is document order --
# so a rule alone cannot put a title over a name drawn after it; something has to change what is
# drawn after what. And a title routinely runs past the right edge of its own box: a one-column
# container is 200px wide and a sixty-five character title at 12px is far wider than that, so a
# backing that is the box, or anything that clips the title to the box, covers the wrong half of
# the sentence.
#
# What the click does is the page script's business. What it depends on is in the drawing, and that
# is what is pinned here.


class TestTheDrawingLetsATitleBeBroughtForward:
    """The drawing has to leave a title somewhere it can be brought to without moving."""

    def test_the_lane_names_are_painted_after_every_container(self, boxed_page: str) -> None:
        """The lane name is kept legible on purpose by being drawn last of everything."""
        svg = _svg(boxed_page)

        assert '<text class="lane"' in svg and '<g class="group"' in svg
        assert svg.rindex('<text class="lane"') > svg.rindex('<g class="group"'), (
            "the lane names are no longer painted after the containers, so the name that was kept legible on purpose "
            "is now the one underneath"
        )

    def test_no_container_is_drawn_under_a_transform(self, boxed_page: str) -> None:
        """A title taken out of a transformed container lands somewhere other than where it was drawn."""
        groups = _elements(_svg(boxed_page), '<g class="group"')

        assert groups, "no container is drawn, so nothing here is checked"
        for attrs, _ in groups:
            assert "transform" not in attrs, (
                f"a container is drawn under a transform ({attrs.get('transform')}), so a title taken out of it is "
                "drawn somewhere other than where it was"
            )


# ---------------------------------------------------------------------------
# The layout, at the density a real flow has
#
# The cases above box two stages of a six-step fixture, and every one of them passes. The same rule
# applied to a flow of the size this page was built for -- 672 steps, three lanes, 152 columns --
# breaks: most boxes enclose steps that are not theirs, a third write their title across the step
# stacked above, and outlines cross each other in their hundreds. None of that is reachable at six
# steps in two lanes, because a lane that stacks one deep puts a container's members next to each
# other whatever the layout does. Density is the whole test.
#
# So the flow is generated rather than written out: `_dense_flow` builds a spec in memory with the
# shape a real one has and neutral names throughout. Generated rather than copied from a real
# pipeline for a plain reason -- this library is extracted and shipped, and one client's vocabulary
# has no business in it -- and the value that was wanted from a real flow is its shape, which is
# what is reproduced:
#
#   672 steps, 3 lanes, 152 columns, lanes stacking 4, 16 and 17 deep, 222 of the 672 at slot 0
#
# and, deliberately, the awkward parts of that shape rather than a tidy grid:
#
#   * cell sizes that are skewed rather than even -- a few columns where sixteen steps sit at one
#     depth, and many holding one or two. A container inside a deep cell is one whose header has
#     nothing above it but another step.
#   * a declaration order that only loosely tracks depth. A person writes a file in the order the
#     work makes sense in, not in the order the dependency graph sorts it, so steps declared next to
#     each other land columns apart -- which is what makes a container's members scatter, and a
#     scattered container is what encloses a stranger.
#   * connections that skip columns and branches that rejoin, through the two-input tools. A flow
#     wired strictly column to column has no long edges, and a layout can satisfy every rule below
#     on one.
#
# A tidy synthetic grid would pass every case here while the real thing failed, which is exactly the
# trap the fixture already fell into once. The measured figures on the generated flow are in the
# docstrings below; they are of the same order as the ones measured on the flow this page is for.
# ---------------------------------------------------------------------------

#: The shape the generated flow is built to, which is a real flow's shape: lanes carrying 148, 450
#: and 74 steps, occupying 70, 120 and 32 of the 152 columns, and stacking at most 16, 17 and 4 deep
#: where they are busiest. The names are the flow's own and mean nothing.
LANE_SHAPE: tuple[tuple[str, int, int, int], ...] = (
    ("alpha", 148, 70, 16),
    ("beta", 450, 120, 17),
    ("gamma", 74, 32, 4),
)

#: Which lane owns each column before the spare cells are handed out, as a pattern of 19 that
#: divides the 152 columns exactly. Every column gets one lane this way, which is what lets a step
#: at column c always find something to be wired to at column c-1.
COLUMN_OWNER: tuple[str, ...] = (
    "beta",
    "beta",
    "alpha",
    "beta",
    "beta",
    "beta",
    "gamma",
    "beta",
    "beta",
    "alpha",
    "beta",
    "beta",
    "beta",
    "beta",
    "alpha",
    "beta",
    "beta",
    "beta",
    "beta",
)

COLUMNS = 152

#: Fixed, because a layout test that draws a different flow on every run reports a different defect
#: on every run, and the golden below would be meaningless.
SEED = 20260916

#: How far a step's place in the file may drift from its place in the flow. Just over half of them
#: do not drift at all, most of the rest by up to a dozen columns, and a few by up to sixty -- which
#: is a person writing the steps of one job together even though the flow reaches them at very
#: different depths. This is the single knob that decides whether a container's members are adjacent
#: for free, so it is the one thing a synthetic flow must not be tidy about.
NEARBY, FAR = 14, 60
STAYS_PUT, DRIFTS = 0.55, 0.94

#: How many steps take two inputs, which is where a connection that skips columns and a branch that
#: rejoins both come from. At this rate the generated flow carries 824 connections over its 672
#: steps, against the 765 over 672 a real one of this size carries.
REJOINING = 0.26

#: The longest run of steps boxed together, and how many consecutive steps share a stage. Six is the
#: size the feature was asked for -- one sentence over the handful of tools that carry out one job --
#: the two together yield 150 containers over the 672 steps, against 143 on a real flow of this size.
RUN_LENGTH, STAGE_LENGTH = 6, 9

#: Tools by how many inputs they take, drawn from the shipped catalog so the flow is wired out of
#: real ports -- `filter` branching to `true` and `false`, `join` rejoining two of them. No
#: configuration is generated with them: nothing in the layout reads a step's configuration, and a
#: generator that had to produce 672 valid configurations would be testing the loader instead.
SOURCE_TOOLS = ("read_table", "read_csv", "read_excel")
ONE_INPUT_TOOLS = (
    "filter",
    "sort",
    "select_columns",
    "record_id",
    "limit_rows",
    "group_by",
    "remove_duplicates",
    "normalize_text",
    "cast_columns",
    "formula",
    "rename_columns",
)
TWO_INPUT_TOOLS = ("join", "union")

#: The canvas the generated flow draws today with no containers declared over it, and a digest of
#: where every one of its 672 steps sits. Recorded rather than derived because it is the thing that
#: must not move: a layout that learns about containers and shifts a flow that has none is a silent
#: regression on every page shipped so far, and nothing but yesterday's numbers can catch that.
#:
#: The digest is over the coordinates rather than over the markup. A page whose node markup changes
#: for a reason that has nothing to do with the layout -- another attribute, another anchor -- has
#: not moved a step, and a golden that failed on that would be re-blessed without being read, which
#: is a golden that has stopped guarding anything.
UNBOXED_CANVAS = (35292.0, 3146.0)
UNBOXED_PLACES = "a55d4909b0c581ae37759edcc4ab9637ee73b89577bd4cac53ede41d383401a0"

#: How much taller the same flow may get once it is boxed. Members have to become adjacent and every
#: container needs a band for its header, so the canvas cannot stay the height it is. Packing the
#: containers into their lanes as tightly as their column spans allow needs 2.49x on this flow, and
#: a greedy pass over the same intervals lands at 2.69x -- measured on a real flow of the same size
#: the two figures are 2.07x and 2.10x -- so anything under 2.7 is a bound no correct layout can
#: meet. 3.5 leaves a working margin over the harder of the two and still refuses the layout that
#: satisfies every rule below by giving each container a row of its own, which is 11.6x here.
#:
#: Width has no allowance at all. A column is a step's distance along the flow, and containers do
#: not change how far along the flow anything is; a layout that widens a 35,292px canvas has moved a
#: step into a column it does not belong in, which is the drawing lying about the order of the work.
MAX_TALLER = 3.5


def _cell_sizes(cells: int, total: int, cap: int, rng: random.Random) -> list[int]:
    """How many steps sit in each of one lane's cells: skewed, not spread evenly.

    Preferential attachment -- a cell already holding many is the likeliest to take the next -- so
    the lane ends up with a few columns where a dozen steps sit at one depth and many holding one or
    two, which is how a real flow looks and is not how `total // cells` looks. It matters here
    because a container drawn inside a deep cell is one whose header has another step directly above
    it, and an evenly spread lane has almost none of those.
    """
    sizes = [1] * cells
    sizes[0] = cap
    left = total - sum(sizes)
    assert left >= 0, f"{cells} cells cannot hold {total} steps"
    while left:
        room = [n for n in range(1, cells) if sizes[n] < cap]
        sizes[rng.choices(room, weights=[sizes[n] ** 2 for n in room])[0]] += 1
        left -= 1
    return sizes


def _drift(rng: random.Random) -> int:
    """How far this step's place in the file sits from its place in the flow. See :data:`NEARBY`."""
    roll = rng.random()
    if roll < STAYS_PUT:
        return 0
    if roll < DRIFTS:
        return rng.randrange(-NEARBY, NEARBY + 1)
    return rng.randrange(-FAR, FAR + 1)


def _dense_flow(*, boxed: bool) -> Pipeline:
    """A flow with a real one's shape and no client's vocabulary, optionally boxed into containers.

    Built as a spec rather than as a document, because the loader is not what is under test: the
    layout is handed a :class:`Pipeline` and reads its wiring, its lanes and its groups. The
    registry is the real one, so every tool named here has the ports it is wired through.

    A step's column is its longest path from a source, which the layout works out from the wiring
    rather than being told. So the wiring is built to produce the intended columns: every step at
    column c takes an input from some step at column c-1, which fixes its depth at c, and a step
    with a second input takes it from further back still, which is a connection that skips columns
    and leaves the depth alone.
    """
    rng = random.Random(SEED)

    # Which (lane, column) cells hold steps: one lane owns each column outright, then the lanes that
    # are owed more cells than the pattern gave them take spare columns elsewhere.
    cells: dict[str, list[int]] = {lane: [] for lane, *_ in LANE_SHAPE}
    for column in range(COLUMNS):
        cells[COLUMN_OWNER[column % len(COLUMN_OWNER)]].append(column)
    for lane, _, wanted, _ in LANE_SHAPE:
        spare = [column for column in range(COLUMNS) if column not in cells[lane]]
        rng.shuffle(spare)
        cells[lane].extend(spare[: wanted - len(cells[lane])])
        cells[lane].sort()

    held: dict[tuple[str, int], int] = {}
    for lane, total, wanted, cap in LANE_SHAPE:
        assert len(cells[lane]) == wanted, f"lane {lane} took {len(cells[lane])} cells, not {wanted}"
        for column, size in zip(cells[lane], _cell_sizes(wanted, total, cap, rng), strict=True):
            held[(lane, column)] = size

    # The order the steps are written down in: by lane, and within a lane by where the flow reaches
    # them plus however far this one drifted.
    written: list[tuple[str, int]] = []
    for lane, *_ in LANE_SHAPE:
        mine = [(column, column + _drift(rng)) for column in cells[lane] for _ in range(held[(lane, column)])]
        written.extend((lane, column) for column, _ in sorted(mine, key=lambda place: (place[1], place[0])))

    ids = [f"step_{n:03d}" for n in range(len(written))]
    lane_of = {step_id: written[n][0] for n, step_id in enumerate(ids)}
    column_of = {step_id: written[n][1] for n, step_id in enumerate(ids)}
    at_column: dict[int, list[str]] = {column: [] for column in range(COLUMNS)}
    for step_id in ids:
        at_column[column_of[step_id]].append(step_id)

    # Every tool is settled before anything is wired, because a step is wired to steps written down
    # after it as well as before it, and what it can be wired to is a question about their ports.
    tool_of: dict[str, str] = {}
    for step_id in ids:
        column = column_of[step_id]
        if column == 0:
            tool_of[step_id] = rng.choice(SOURCE_TOOLS)
        elif column >= 3 and rng.random() < REJOINING:
            tool_of[step_id] = rng.choice(TWO_INPUT_TOOLS)
        else:
            tool_of[step_id] = rng.choice(ONE_INPUT_TOOLS)

    def some_port(step_id: str) -> PortRef:
        return PortRef(step_id, rng.choice(REGISTRY.get(tool_of[step_id]).output_ports))

    steps: list[Step] = []
    written_in_lane: Counter[str] = Counter()
    for step_id in ids:
        lane, column = lane_of[step_id], column_of[step_id]
        stage = f"{lane}_{written_in_lane[lane] // STAGE_LENGTH:02d}"
        written_in_lane[lane] += 1
        ports = REGISTRY.get(tool_of[step_id]).input_ports
        inputs: dict[str, PortRef] = {}
        if ports:
            inputs[ports[0]] = some_port(rng.choice(at_column[column - 1]))
        if len(ports) > 1:
            inputs[ports[1]] = some_port(rng.choice(at_column[rng.randrange(0, column - 1)]))
        steps.append(Step(id=step_id, tool=tool_of[step_id], config={}, inputs=inputs, lane=lane, stage=stage))

    groups: tuple[Container, ...] = ()
    if boxed:
        runs = _runs(steps)
        owner = {step_id: f"box_{n:03d}" for n, run in enumerate(runs) for step_id in run}
        groups = tuple(
            Container(id=f"box_{n:03d}", title=f"Step {n}: what {len(run)} tools do between them")
            for n, run in enumerate(runs)
        )
        steps = [
            Step(
                id=step.id,
                tool=step.tool,
                config=step.config,
                inputs=step.inputs,
                lane=step.lane,
                stage=step.stage,
                container=owner[step.id],
            )
            for step in steps
        ]
    lanes = tuple(Lane(id=lane, title=f"Lane {lane}", order=n) for n, (lane, *_) in enumerate(LANE_SHAPE))
    return Pipeline(name="dense", steps=tuple(steps), lanes=lanes, containers=groups, registry=REGISTRY)


def _runs(steps: list[Step]) -> list[list[str]]:
    """Consecutive steps sharing a lane and a stage, cut into runs of at most :data:`RUN_LENGTH`."""
    runs: list[list[str]] = []
    key: tuple[str | None, str | None] | None = None
    for step in steps:
        here = (step.lane, step.stage)
        if runs and here == key and len(runs[-1]) < RUN_LENGTH:
            runs[-1].append(step.id)
        else:
            runs.append([step.id])
        key = here
    return runs


def _no_run(spec: Pipeline) -> PipelineTrace:
    """A trace of nothing, which is all the layout needs.

    Nothing in the placing of a step consults the trace -- it supplies the figures written onto the
    boxes and the labels on the connections, and a container with no rows is drawn in exactly the
    place a container with rows is. Asserting geometry against an empty run is asserting it against
    the wiring, which is what decides the geometry. It is also what makes 672 steps cost a test
    fraction of a second rather than a run of a flow this size.
    """
    return PipelineTrace(pipeline=spec.name, retain="live", events=(), shapes={}, frames={})


@pytest.fixture(scope="module")
def dense_unboxed() -> str:
    """The dense flow drawn with no containers declared over it."""
    spec = _dense_flow(boxed=False)
    return _canvas(spec, _no_run(spec))


@pytest.fixture(scope="module")
def dense_boxed() -> tuple[Pipeline, str]:
    """The dense flow with a container round every run of steps doing one job, and its canvas."""
    spec = _dense_flow(boxed=True)
    return spec, _canvas(spec, _no_run(spec))


def _canvas_size(canvas: str) -> tuple[float, float]:
    found = re.search(r'data-w="([\d.]+)"\s+data-h="([\d.]+)"', canvas)
    assert found is not None, "the drawing does not state its own size"
    return float(found.group(1)), float(found.group(2))


def _places(canvas: str) -> str:
    """Every step's top-left corner, as one sorted text, so one layout can be compared to another."""
    return "\n".join(sorted(f"{step} {at[0]:g},{at[1]:g}" for step, at in _node_boxes(canvas).items()))


def _columns(spec: Pipeline) -> dict[str, int]:
    """How far along the flow each step sits, worked out here from the wiring rather than taken
    from the page, so the shape case is checking the flow and not the drawing of it."""
    column: dict[str, int] = {}
    for step_id in spec.order():
        column[step_id] = 1 + max((column[ref.step] for ref in spec.step(step_id).inputs.values()), default=-1)
    return column


def _outline_and_header(body: str) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """A container's two rectangles: the one it draws round its steps, and the band it writes in.

    Told apart by size rather than by class name, so this says nothing about how the markup spells
    them. A container draws the outline it keeps while it is open and the bar it shrinks to when it
    is closed, and that bar *is* the header: the two share a corner and a width, and the bar is the
    shorter. A step's own box and the collapse control are not the container drawing itself, and
    `_own_rects` has already taken those out.
    """
    own = _own_rects(body)
    assert own, "the container draws no rectangle of its own"
    return max(own, key=lambda drawn: drawn[2] * drawn[3]), min(own, key=lambda drawn: drawn[2] * drawn[3])


class TestTheLayoutHoldsAtTheDensityARealFlowHas:
    """The same containers the cases above draw over six steps, drawn over six hundred and seventy-two."""

    def test_the_generated_flow_is_the_shape_these_cases_were_measured_against(
        self, dense_boxed: tuple[Pipeline, str]
    ) -> None:
        """The figures the rest of the class argues from, stated once and checked.

        Every bound below -- 3.5x the height, no extra width -- was worked out against a flow of
        this shape, and the whole reason the flow is generated is that the shape is the point. If
        the generator is ever tuned and the shape goes with it, those bounds stop meaning what they
        were chosen to mean and this class goes quiet in exactly the way the six-step fixture went
        quiet.

        The stacking is the load-bearing figure and the slot-0 count is beside it: a container's
        members are adjacent whatever the layout does in a lane one step deep, and a flow where
        every step is the top of its cell is a flow where every header has room for free.
        """
        spec, _ = dense_boxed
        column = _columns(spec)
        deepest: Counter[str] = Counter()
        cell: Counter[tuple[str, int]] = Counter()
        for step in spec.steps:
            cell[(step.lane or "", column[step.id])] += 1
            deepest[step.lane or ""] = max(deepest[step.lane or ""], cell[(step.lane or "", column[step.id])])

        assert len(spec.steps) == 672, f"the flow now has {len(spec.steps)} steps"
        assert len(spec.lanes) == 3 and max(column.values()) + 1 == COLUMNS
        assert sorted(deepest.values()) == [4, 16, 17], (
            f"the lanes now stack {sorted(deepest.values())} deep; a lane that stacks one deep cannot tell a "
            "layout that places containers from one that draws a box round wherever they landed"
        )
        assert len(cell) == 222, f"{len(cell)} of the 672 steps are at slot 0, not 222"
        assert len(spec.containers) == 150, f"boxing runs of {RUN_LENGTH} now yields {len(spec.containers)} containers"
        assert sum(len(step.inputs) for step in spec.steps) == 824, "the flow is not wired as it was measured"
        skipping = [
            (step.id, ref.step)
            for step in spec.steps
            for ref in step.inputs.values()
            if column[step.id] - column[ref.step] > 1
        ]
        assert len(skipping) > 150, (
            f"only {len(skipping)} connections skip a column; a flow wired strictly column to column has no long "
            "edges, and a layout can satisfy every case below on one"
        )

    def test_every_container_on_a_dense_flow_encloses_its_own_steps_and_no_others(
        self, dense_boxed: tuple[Pipeline, str]
    ) -> None:
        """The fatal one: a box is a claim about what is inside it, and today 145 of 150 are false.

        Two halves, and both have to hold at once. A box that does not reach round one of its own
        steps says that step is no part of the job it is captioned with. A box drawn over a step
        that is not its own says that step *is* part of it -- and at this density that is not a
        pixel of untidiness, it is a box titled with one job drawn round as many as 218 steps doing
        others. Neither announces itself: the drawing looks finished either way and a reader
        believes the line.

        Held against every step on the canvas rather than against the container's neighbours,
        because a stranger arrives from anywhere -- a step in the same lane forty columns along, a
        step in the next lane whose slot the box grew past.
        """
        spec, canvas = dense_boxed
        boxes = _node_boxes(canvas)
        containers = _containers(canvas)

        assert set(containers) == {group.id for group in spec.containers}, "not every declared container is drawn"
        assert len(boxes) == len(spec.steps), f"{len(boxes)} steps are drawn, not {len(spec.steps)}"

        missing: dict[str, list[str]] = {}
        swallowed: dict[str, list[str]] = {}
        for group_id, (_, body) in containers.items():
            held = set(spec.container_members(group_id))
            drawn = {attrs.get("id", "").removeprefix("node-") for attrs, _ in _elements(body, '<a class="node')}
            assert drawn == held, f"{group_id} draws {sorted(drawn)}, which is not {sorted(held)}"

            outline, _ = _outline_and_header(body)
            loose = sorted(member for member in held if not _encloses(outline, boxes[member]))
            if loose:
                missing[group_id] = loose
            strangers = sorted(at for at, box in boxes.items() if at not in held and _overlaps(outline, box))
            if strangers:
                swallowed[group_id] = strangers

        worst = max(swallowed.values(), key=len, default=[])
        assert not missing, (
            f"{len(missing)} of {len(containers)} containers do not reach round their own steps, "
            f"for example {sorted(missing)[0]}: {missing[sorted(missing)[0]][:6]}"
        )
        assert not swallowed, (
            f"{len(swallowed)} of {len(containers)} containers are drawn over a step that is not theirs; "
            f"the worst swallows {len(worst)} of them, for example {worst[:6]}"
        )

    def test_every_containers_header_has_room_of_its_own_on_a_dense_flow(
        self, dense_boxed: tuple[Pipeline, str]
    ) -> None:
        """The title and the counts are the feature; drawn over a step, neither can be read.

        The header is the band a closed container shrinks to, so it is the one part of a container
        that has to be legible in both states, and it is where the sentence a reader came for is
        written. Today it is drawn in space nothing reserved for it: it fits above a container whose
        topmost step is the top of its lane, and on 47 of 150 it is written straight across the box
        of the step stacked above -- two sentences and a row count over one another.

        Asked of every step on the canvas including the container's own. A header overlapping a
        member is the same unreadable band, and is what a layout that reserves room everywhere
        except inside the box produces.
        """
        spec, canvas = dense_boxed
        boxes = _node_boxes(canvas)
        containers = _containers(canvas)

        assert set(containers) == {group.id for group in spec.containers}, "not every declared container is drawn"
        written_over: dict[str, list[str]] = {}
        for group_id, (_, body) in containers.items():
            outline, header = _outline_and_header(body)
            assert header[3] < outline[3], (
                f"{group_id}: the container draws one rectangle at one size, so nothing here is a header band"
            )
            over = sorted(at for at, box in boxes.items() if _overlaps(header, box))
            if over:
                written_over[group_id] = over

        assert not written_over, (
            f"{len(written_over)} of {len(containers)} containers write their title and counts across a step: "
            f"for example {sorted(written_over)[0]} over {written_over[sorted(written_over)[0]]}"
        )

    def test_two_containers_are_never_drawn_over_each_other_on_a_dense_flow(
        self, dense_boxed: tuple[Pipeline, str]
    ) -> None:
        """A step belongs to at most one container, so two outlines crossing is a drawing of a lie.

        Distinct from the stranger case and not implied by it. Two boxes can cross in space that
        holds no step at all -- one reaching up for its header through the width of another -- and
        the reader is then shown two captions over one region with nothing saying which of them owns
        it. Today 1,089 of the 11,175 pairs cross.
        """
        spec, canvas = dense_boxed
        containers = _containers(canvas)

        assert set(containers) == {group.id for group in spec.containers}, "not every declared container is drawn"
        outlines = {group_id: _outline_and_header(body)[0] for group_id, (_, body) in containers.items()}
        crossing = [
            (one, other)
            for one, other in itertools.combinations(sorted(outlines), 2)
            if _overlaps(outlines[one], outlines[other])
        ]

        assert not crossing, (
            f"{len(crossing)} pairs of containers are drawn over each other, for example "
            f"{crossing[0]} at {outlines[crossing[0][0]]} and {outlines[crossing[0][1]]}"
        )

    def test_a_flow_with_no_containers_is_laid_out_exactly_as_it_is_today(self, dense_unboxed: str) -> None:
        """What stops a layout that learns about containers being a regression for every other page.

        Every page shipped so far declares no containers, and most of a 672-step flow will be in
        none for a long while. Those steps have to land where they land now -- not near it, on it.
        A layout rewritten to make members adjacent has every opportunity to shift a flow that has
        no members to make adjacent, and the shift would be invisible: the canvas is still a canvas,
        every step is still drawn, and nothing is wrong except that the page a reader knows has
        moved under them.

        Compared against yesterday's numbers because nothing else can catch it, and the flow is
        generated from a fixed seed so that yesterday's numbers are a fact about the layout rather
        than about which flow happened to be drawn.
        """
        places = _places(dense_unboxed)

        assert len(places.splitlines()) == 672, f"{len(places.splitlines())} steps are drawn, not 672"
        assert _canvas_size(dense_unboxed) == UNBOXED_CANVAS, (
            f"the canvas is {_canvas_size(dense_unboxed)}, not the {UNBOXED_CANVAS} it is today"
        )
        assert hashlib.sha256(places.encode("utf8")).hexdigest() == UNBOXED_PLACES, (
            "a flow with no containers is no longer laid out where it is laid out today, so every page that "
            "declares no group has moved"
        )

    def test_boxing_a_dense_flow_does_not_make_the_canvas_absurd(
        self, dense_unboxed: str, dense_boxed: tuple[Pipeline, str]
    ) -> None:
        """Every rule above is satisfied by a layout nobody could use, so the cost is pinned too.

        Give each of the 150 containers a row of its own and no box encloses a stranger, no header
        is written over anything and no two boxes cross -- on a canvas 11.6 times taller than the
        one it replaced, which is a page a reader scrolls past rather than reads. The feature exists
        to make a 672-step flow followable; a layout that makes it less followable has failed at the
        thing it was for.

        Width carries no allowance at all: a column is how far along the flow a step is, containers
        change nothing about that, and a wider canvas means a step was moved into a column it does
        not belong in.

        The last assertion catches the cheap way to keep a stranger out -- stop drawing it. A box
        with nothing else in it encloses nothing else.

        This is a ceiling rather than a defect report, and it holds today for the reason the three
        cases above fail: nothing about the layout moves when a container is declared, so the boxed
        canvas is the unboxed one to the pixel. It starts meaning something the moment members are
        actually made adjacent, which is the change that has to pay for itself.
        """
        _, canvas = dense_boxed
        was, now = _canvas_size(dense_unboxed), _canvas_size(canvas)

        assert now[0] <= was[0], (
            f"boxing widened the canvas from {was[0]:,.0f} to {now[0]:,.0f}px, so a step has been moved along the flow"
        )
        assert now[1] <= was[1] * MAX_TALLER, (
            f"boxing made the canvas {now[1] / was[1]:.2f}x taller ({was[1]:,.0f} -> {now[1]:,.0f}px), over the "
            f"{MAX_TALLER}x a layout that packs its containers needs"
        )
        assert len(_node_boxes(canvas)) == 672, (
            f"{len(_node_boxes(canvas))} steps are drawn once the flow is boxed, not 672"
        )


def test_each_setting_that_names_a_file_is_marked_for_the_page_and_no_other(page: str) -> None:
    """A file setting carries ``data-file`` so the page can offer to show the file; nothing else does."""
    from dantron_etl.viewer.files import file_settings

    spec = read_pipeline(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    sections = dict(re.findall(r'<section class="detail[^"]*" data-step="([^"]+)"[^>]*>(.*?)</section>', page, re.S))
    assert set(sections) == {step.id for step in spec.steps}
    marked_somewhere = False
    for step in spec.steps:
        wanted = [name for name in file_settings(REGISTRY.get(step.tool).config_template()) if name in step.config]
        marked = re.findall(r'<div class="cfg" data-file="([^"]+)"><span class="ck">([^<]+)</span>', sections[step.id])
        assert [setting for setting, _ in marked] == wanted, step.id
        assert all(setting == key for setting, key in marked)
        assert sections[step.id].count("data-file=") == len(wanted)
        marked_somewhere = marked_somewhere or bool(wanted)
    assert marked_somewhere
