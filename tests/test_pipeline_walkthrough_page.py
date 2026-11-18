"""Structural guards on the generated walkthrough page.

Every defect this page has shipped was invisible to the suite and obvious to a person looking at it:
an entity escaped as data, an octal escape eaten by Python, bytes read as the wrong encoding, two
steps drawn at one coordinate, and -- the one these tests exist for -- a section and a table wearing
the same class name, so selecting a step hid every table on the page.

None of those are testable by rendering and squinting. All of them are testable as *structure*: the
page either says what it claims to, or it does not. So these are assertions about the document rather
than about how it looks.

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
from dantron_etl.pipeline import REGISTRY, Group, Lane, PipelineSpec, PortRef, StepSpec, read_pipeline_file
from dantron_etl.runner import PipelineTrace, run_pipeline

PROJECT_ROOT = project_root(__file__)

# `_canvas` is the layout itself, and the cases at the foot of this file are about the layout. The
# whole page cannot stand in for it on the real flow: `build` wants the frames a run produced, and
# the real flow reads a database this suite has no access to. Where the layout *is* reachable
# through the page -- the fixture flow -- the cases above go through `build`, and this is the seam
# for the one flow that cannot.
from scripts.pipeline_walkthrough_page import _canvas, build  # noqa: E402 -- needs PROJECT_ROOT first


@pytest.fixture(scope="module")
def page(fixture_flow_sandbox: Path) -> str:
    """One rendered page, shared: building it is pure and costs nothing to reuse."""
    spec = read_pipeline_file(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    with chdir(fixture_flow_sandbox):  # the flow's sinks write under the working folder
        trace = run_pipeline(spec)
    return build(spec, trace, FocusSpec(label="party", identity=("party",)))


@pytest.fixture(scope="module")
def panels(fixture_flow_sandbox: Path) -> dict[str, str]:
    """Every step's data panel, keyed by the step it answers to, as the service will hand them over.

    Imported here rather than at the top of the file so that a suite run before the panels exist
    fails on the cases that are about them, rather than collecting nothing at all.
    """
    from scripts.pipeline_walkthrough_page import build_panels

    spec = read_pipeline_file(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
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
    spec = read_pipeline_file(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    return tuple(spec.order())


@pytest.fixture(scope="module")
def wiring() -> Counter[tuple[str, str, str]]:
    """Every connection the pipeline file declares, as (upstream step, downstream step, port).

    A Counter rather than a set: one step may take the same upstream port on two of its own inputs,
    which is two lines on the canvas and has to stay two lines in the markup.
    """
    spec = read_pipeline_file(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
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


class TestTheScriptSurvivesBeingAFormatString:
    """The template is passed through str.format, so every literal brace has to be doubled."""

    def test_braces_balance_in_the_rendered_script(self, page: str) -> None:
        script = page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]

        assert script.count("{") == script.count("}")

    def test_no_doubled_brace_survives_into_the_page(self, page: str) -> None:
        """A doubled brace reaching the browser is a template that was escaped twice."""
        script = page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]

        assert "{{" not in script
        assert "}}" not in script

    def test_every_selector_the_script_uses_matches_something_in_the_page(self, page: str, shown: str) -> None:
        """A selector matching nothing is a control that silently does nothing when clicked.

        That is exactly how the in/out toggle broke: it reached for a class the pane refactor had
        removed, threw, and left a button that looked fine and did nothing.

        Searched over the panels as well as the page, because a class the script reaches for may now
        arrive with a panel rather than be shipped in the document -- and a class that is in neither
        is the same dead control it always was.
        """
        script = page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]
        classes = set(re.findall(r"closest\('\.([a-z-]+)'\)", script))
        classes |= set(re.findall(r"querySelectorAll\('\.([a-z-]+)", script))

        for name in sorted(classes):
            assert f'class="{name}' in shown or f" {name}" in shown, (
                f"the script reaches for .{name}, which is neither drawn in the page nor in any panel"
            )


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
        spec = read_pipeline_file(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})

        for step_id, body in sections:
            has_inputs = bool(spec.step(step_id).inputs)
            has_flip = 'class="flip"' in body
            assert has_flip == has_inputs, f"{step_id}: flip={has_flip} but inputs={has_inputs}"


def test_the_page_declares_its_encoding(page: str) -> None:
    """Undeclared, a browser guesses, and a client's accented name arrives as three letters."""
    assert page.lstrip().startswith('<meta charset="utf-8">')


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

    def test_zoom_is_bounded_at_both_ends(self, page: str) -> None:
        """A flow zoomed to nothing cannot be zoomed back, which is a control that traps a reader."""
        script = page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]

        assert "Math.min" in script and "Math.max" in script, "applyZoom does not bound what it is given"

    def test_the_wheel_is_only_taken_with_a_modifier(self, page: str) -> None:
        """Taking the bare wheel over the canvas would break scrolling for everyone who expects it."""
        script = page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]
        handler = script[script.index("addEventListener('wheel'") :]

        assert "ctrlKey" in handler and "metaKey" in handler
        assert "passive: false" in handler, "preventDefault in a passive listener does nothing at all"


class TestAnyPaneCanBeFoldedAway:
    """So any one of the three can have the whole window when it is the one being read."""

    def test_each_pane_has_a_control_that_folds_it(self, page: str) -> None:
        offered = set(re.findall(r'<button data-fold="(\w+)"', page))

        assert offered == {"flow", "detail", "data"}

    def test_a_folded_pane_is_not_drawn_rather_than_drawn_at_no_size(self, page: str) -> None:
        """A pane at zero height still lays out its contents; one that is not drawn costs nothing."""
        for pane in ("flow", "detail", "data"):
            assert f".wrap.no-{pane} .{pane}pane" in page
            assert "display:none" in page

    def test_folding_a_pane_takes_its_grip_with_it(self, page: str) -> None:
        """A divider between a pane and nothing is a control that drags one edge of empty space."""
        assert ".wrap.no-flow .flowpane, .wrap.no-flow .grip-v" in page
        assert ".wrap.no-data .datapane, .wrap.no-data .grip-h" in page

    def test_the_grid_gives_the_space_back_rather_than_leaving_a_gap(self, page: str) -> None:
        for rule in (
            ".wrap.no-flow .panes",
            ".wrap.no-detail .panes",
            ".wrap.no-data .panes",
        ):
            assert rule in page, f"{rule} does not re-size the grid"

    def test_the_last_visible_pane_refuses_to_fold(self, page: str) -> None:
        """A window showing nothing has no control left saying which pane to bring back."""
        script = page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]
        handler = script[script.index("closest('.folds button')") :]

        assert "showing.length === 1" in handler, "nothing stops the reader folding the last pane away"


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


def _script(page: str) -> str:
    return page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]


def _edges(page: str) -> list[tuple[dict[str, str], str]]:
    """Every drawn connection as (its attributes, its contents)."""
    found = []
    for tag, body in re.findall(r'<g class="edge[^"]*"([^>]*)>(.*?)</g>', _svg(page), re.DOTALL):
        found.append((dict(re.findall(r'([a-z-]+)="([^"]*)"', tag)), body))
    return found


def _block(text: str, opener: str) -> str:
    """The braced block that ``opener`` begins, balanced rather than guessed at by line count."""
    start = text.index(opener)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _with_callees(script: str, snippet: str) -> str:
    """A snippet plus the bodies of the functions it calls.

    Whether the tracing lives inside `choose` or in a helper it calls is the author's business; what
    these tests are entitled to is that choosing a step reaches it. Following one level of calls
    keeps the assertions about behaviour rather than about how the script is cut up.
    """
    reached = snippet
    for name in sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", snippet))):
        opener = f"function {name}("
        if opener in script:
            reached += _block(script, opener)
    return reached


def _code(script: str) -> str:
    """The script with its comments taken out.

    Every assertion below asks what the script *does*, and prose about a call is not a call. The
    comment inside `choose` argues "do not collapse this back to an unconditional traceFrom()" --
    which is exactly the string a count of call sites would read as a second call, so a guard could
    be deleted and the count would not move. Strings are left alone: the script has no `//` inside
    one, and the state names these tests follow are string literals.
    """
    script = re.sub(r"/\*.*?\*/", "", script, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", script)


def _js(page: str) -> str:
    return _code(_script(page))


def _args(call: str) -> list[str]:
    """The arguments of a call, split at the commas that belong to it and not to a nested one.

    `choose(node.id.replace(/^node-/, ''))` is one argument containing a comma, and reading it as
    two would make the load-time call look no different from a reader's click.
    """
    parts, depth, at = [], 0, 0
    for i, ch in enumerate(call):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(call[at:i].strip())
            at = i + 1
    tail = call[at:].strip()
    if tail:
        parts.append(tail)
    return parts


def _calls(page: str, name: str) -> list[tuple[int, list[str]]]:
    """Every call to ``name`` as (how deeply nested it sits, its arguments).

    Depth 0 is the top level of the script -- a statement that runs because the page loaded, not
    because a reader clicked something. That distinction is the whole opening-undimmed ruling, and
    it is invisible to any test that only asks whether an argument appears somewhere.
    """
    found: list[tuple[int, list[str]]] = []
    depth = 0
    for hit in re.finditer(r"[{}]|(?<![\w.])(?<!function )" + name + r"\(([^;]*)\)", _js(page)):
        if hit.group(0) == "{":
            depth += 1
        elif hit.group(0) == "}":
            depth -= 1
        else:
            found.append((depth, _args(hit.group(1))))
    return found


def _css_rules(page: str) -> list[tuple[str, str]]:
    """Every style rule as (selector, declarations), in the order the sheet writes them.

    Order matters as much as content here: two rules of equal specificity are settled by which one
    is written last, so a sheet that is sorted rather than argued changes what the page shows.
    """
    sheet = page[page.index("<style>") + len("<style>") : page.index("</style>")]
    sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.DOTALL)
    return [(sel.strip(), decl.strip()) for sel, decl in re.findall(r"([^{}]+)\{([^{}]*)\}", sheet)]


def _state_decl(page: str, kind: str, state: str) -> str:
    """Everything the sheet declares for one state of one kind of thing, rules joined together.

    `.node.direct rect` and `.node.dim` are the same fact written at different depths, so the
    comparison has to be over what the sheet says about the state, not over one rule of it.
    """
    said = []
    for sel, decl in _css_rules(page):
        for one in sel.split(","):
            head = one.strip().split()[0] if one.strip() else ""
            if head == f".{kind}.{state}":
                said.append(decl)
    return " ".join(said)


def _opacity(decl: str) -> float | None:
    found = re.search(r"(?:^|;)\s*opacity:\s*([\d.]+)", decl)
    return float(found.group(1)) if found else None


def _painter(page: str) -> tuple[str, set[str]]:
    """The function that puts a state onto an element, and the states it knows about.

    Found by what it does rather than by its name: it is the one that clears the vocabulary and
    then adds one of it. Derived so that the clearing and the painting cannot be compared against a
    list written down in this file, which would go stale in exactly the same edit that breaks them.
    """
    code = _js(page)
    for name in re.findall(r"function\s+(\w+)\(", code):
        body = _block(code, f"function {name}(")
        if "classList.add(" not in body:
            continue
        removes = re.search(r"classList\.remove\(([^)]*)\)", body)
        if removes:
            return name, set(re.findall(r"'(\w+)'", removes.group(1)))
    raise AssertionError("nothing in the script assigns a state class to an element")


def _index_build(page: str) -> dict[str, tuple[str, str]]:
    """Each map of the flow as {its name: (the end it is keyed on, the end it reports as `other`)}.

    Read off the script rather than named in the test: the direction a reader is told is the pairing
    of these two facts, and a test that spelled the map names out would be defeated by renaming them.
    """
    found = re.findall(
        r"(\w+)\.get\(\s*edge\.dataset\.(\w+)\s*\)\.push\(\s*\{[^}]*other:\s*edge\.dataset\.(\w+)",
        _js(page),
    )
    return {name: (key, other) for name, key, other in found}


def _sides(page: str) -> dict[str, str]:
    """Which map the walk pairs with `upstream` and which with `downstream`, by the map's own name."""
    paired = re.findall(r"\[\s*'(upstream|downstream)'\s*,\s*([\w.]+)\s*\]", _js(page))
    return {state: dotted.rsplit(".", 1)[-1] for state, dotted in paired}


def _walk(page: str) -> tuple[str, str, str]:
    """The worklist's name, the neighbour's name, and the body of the loop that drains it."""
    reached = _with_callees(_js(page), _block(_js(page), "function choose("))
    loop = re.search(r"while\s*\(\s*(\w+)\.length\s*\)", reached)
    assert loop is not None, "no worklist: a trace that cannot loop can only reach the neighbours"
    body = _block(reached[loop.start() :], "{")
    link = re.search(r"for\s*\(\s*const\s+(\w+)\s+of\s+", body)
    assert link is not None, "the loop reaches no connections at all"
    return loop.group(1), link.group(1), body


def _edge_click(page: str) -> tuple[str, str]:
    """The name the click handler gives the clicked connection, and the branch that answers it."""
    code = _js(page)
    found = re.search(r"const\s+(\w+)\s*=\s*e\.target\.closest\('\.edge'\)", code)
    assert found is not None, "no handler ever picks a connection out of a click"
    return found.group(1), _block(code[found.start() :], "{")


def _clear_handler(page: str) -> str:
    """The branch that answers the way-back control, with the body of whatever it calls."""
    code = _js(page)
    found = re.search(r"data(?:set\.trace|-trace)", code)
    assert found is not None, "the clear control is drawn but never handled"
    return _with_callees(code, _block(code[found.start() :], "{"))


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

    def test_the_hit_target_is_wide_and_invisible(self, page: str) -> None:
        rule = re.search(r"\.hit \{([^}]*)\}", page)

        assert rule is not None, ".hit is drawn but never styled, so it is a second visible line"
        width = re.search(r"stroke-width:\s*(\d+)", rule.group(1))
        assert width is not None and int(width.group(1)) >= 10, "the target is not appreciably wider than the line"
        assert "transparent" in rule.group(1) or "stroke:none" in rule.group(1), "the target is drawn over the line"
        assert "pointer-events:stroke" in rule.group(1), "a path with fill:none takes clicks on its stroke"

    def test_the_visible_line_declares_no_stroke_of_its_own(self, page: str) -> None:
        """`.wire` declaring nothing is what makes lighting a connection possible at all.

        The group carries the colour and the weight and the path inherits them, so one state class
        on the group restyles the curve, its hit target and its caption together. Give `.wire` its
        own `stroke` -- a tidy-up that reads as an improvement, since it puts the line's look on the
        line -- and a declaration on the element beats anything inherited: `.edge.direct` then
        changes nothing a reader can see. It is worse than a dead rule, because `.edge.dim` sets
        opacity on the group, which still composites over its children: the un-chosen lines fade and
        the chain does not brighten, so a trace reports a thinner flow than the one that is wired.
        """
        for sel, decl in _css_rules(page):
            for one in sel.split(","):
                last = one.strip().split()[-1] if one.strip() else ""
                if not re.fullmatch(r"\.wire(?::[\w-]+)?", last):
                    continue
                said = [p for p in ("stroke", "stroke-width", "opacity") if re.search(rf"(?:^|;)\s*{p}\s*:", decl)]
                assert not said, (
                    f"{one.strip()} declares {said}, which beats what it should inherit from .edge -- "
                    "the state classes on the group will then restyle nothing on the visible curve"
                )


class TestChoosingAStepTracesItsWholeChain:
    """Two lit neighbours do not answer the question at 672 steps; the chain does."""

    def test_the_tracing_hangs_off_choose_rather_than_off_the_node(self, page: str) -> None:
        """`choose` is called from the node and from the step list, and both must do the same thing.

        A highlight bolted onto the node's click handler leaves the list -- which is how a reader
        finds a step they could not spot on the canvas in the first place -- silently doing less.
        """
        script = _script(page)
        reached = _with_callees(script, _block(script, "function choose("))

        assert "'.edge'" in reached or '".edge"' in reached, "choosing a step never looks at the connections"

    @pytest.mark.parametrize("state", ["direct", "upstream", "downstream", "dim"])
    def test_each_state_is_both_assigned_and_drawn(self, page: str, state: str) -> None:
        """Parametrised so the script and the stylesheet cannot drift apart one name at a time.

        A class the script sets and the CSS never styles is a highlight that does nothing; a class
        the CSS styles and the script never sets is a rule nobody dares delete because it looks used.
        """
        script = _script(page)
        reached = _with_callees(script, _block(script, "function choose("))

        assert f"'{state}'" in reached, f"nothing ever puts a connection into the {state!r} state"
        assert f".edge.{state}" in page, f"the {state!r} state is never drawn on a connection"
        # The node wears the same four states as the connection, and the sheet has to draw both: the
        # lines say which way the data runs, and the nodes are what a reader is looking for when they
        # follow one. A state drawn on the edge and not on the node lights the wires around a step
        # that itself looks no different from the six hundred that are not in the chain.
        assert f".node.{state}" in page, f"the {state!r} state is never drawn on a node"

    def test_the_rest_of_the_canvas_goes_quiet(self, page: str) -> None:
        """Lighting four lines among nine hundred is not visible; the other eight hundred fading is.

        Asserted as a number rather than as a selector, and on both kinds of thing. `.node.dim`
        existing says nothing: `.node.dim {{ opacity:1 }}` is a rule that satisfies every spelling
        test in this file and leaves the whole canvas at full strength, which is the state the
        feature exists to get away from.
        """
        for kind in ("node", "edge"):
            faded = _opacity(_state_decl(page, kind, "dim"))
            assert faded is not None, f"a dimmed {kind} is not actually faded, so nothing reads as lit"
            assert faded < 1, f"a dimmed {kind} is drawn at opacity {faded}, which dims nothing"

    def test_upstream_and_downstream_do_not_look_alike(self, page: str) -> None:
        """Which way the data runs is the question a reader traces a line to answer.

        Three states drawn identically light the same pixels as one state would, and tell a reader
        only that a connection is involved -- not whether it feeds the step or is fed by it.
        """
        rules = {}
        for state in ("direct", "upstream", "downstream"):
            found = re.search(rf"\.edge\.{state} \{{([^}}]*)\}}", page)
            assert found is not None, f".edge.{state} is not styled"
            rules[state] = found.group(1).strip()

        assert len(set(rules.values())) == 3, f"these states are drawn the same: {rules}"

    def test_a_node_in_the_two_directions_does_not_look_alike_either(self, page: str) -> None:
        """The twin of the rule above, on the thing a reader is actually hunting for.

        A reader follows a lit line to a node. If every node in the chain is drawn identically, the
        node they arrive at cannot tell them whether they walked forwards or backwards, and the
        direction has to be re-derived by eye from the lines -- which is the work this feature was
        built to take off them.
        """
        drawn = {state: _state_decl(page, "node", state) for state in ("direct", "upstream", "downstream")}

        for state, decl in drawn.items():
            assert decl, f".node.{state} is never drawn, so a node in that state looks untouched"
        assert len(set(drawn.values())) == 3, f"these node states are drawn the same: {drawn}"

    def test_a_warning_outranks_a_highlight_rather_than_the_other_way_round(self, page: str) -> None:
        """`.node.warning rect` and `.node.direct rect` have equal specificity, so the sheet's order decides.

        What the run had to say about a step is a fact about the data; a highlight is a question the
        reader put a second ago. A warning painted over by a highlight is the swap in the wrong
        direction and leaves nothing on the page to say anything was covered up. The ordering is the
        whole mechanism, and alphabetising the sheet -- or moving one rule up beside its neighbours
        -- reverses it in an edit that looks like tidying.
        """
        rules = _css_rules(page)
        traced = [i for i, (sel, _) in enumerate(rules) if re.match(r"\.node\.(direct|upstream|downstream)\b", sel)]
        run_said = [i for i, (sel, _) in enumerate(rules) if re.match(r"\.node\.(warning|error)\b", sel)]

        assert traced and run_said, "one of the two sets of rules is missing, so there is no contest to settle"
        assert min(run_said) > max(traced), (
            "a trace state is written after what the run said about the step, so at equal specificity "
            "the highlight wins and a warned step is indistinguishable from a highlighted one"
        )

    def test_the_chain_is_walked_rather_than_stepped_once(self, page: str) -> None:
        """One hop needs no worklist; the whole chain does, and the whole chain is the feature.

        The fixture flow is the smallest thing that tells them apart: choosing `monthly` has to
        reach back past `receivable` to `entries` and forward past `totals` to `published`, while
        `set_aside` -- a sibling branch off the same upstream step -- stays dim. A neighbours-only
        implementation passes every other test in this class and still fails the reader.
        """
        script = _script(page)
        reached = _with_callees(script, _block(script, "function choose("))

        assert "querySelectorAll('.edge')" in reached, "the trace is not built from the edges the page drew"
        assert "while (" in reached, "no worklist: a trace that cannot loop can only reach the neighbours"

    def test_what_the_walk_reaches_goes_back_onto_the_worklist(self, page: str) -> None:
        """A loop that never grows its own worklist drains after one pass, which is one hop.

        This is the half the loop existing does not prove. Delete the push and keep everything else
        -- the `while`, the visited set, the two directions, all four state names -- and the trace
        lights the chosen step's neighbours and stops. Nothing on the page says it stopped: the
        reader sees two lit lines and a dim canvas beyond them, and reads that as "nothing else
        feeds this", which is a wrong answer wearing the look of a complete one.
        """
        work, link, loop = _walk(page)

        assert re.search(rf"\b{work}\.push\(\s*{link}\.other\s*\)", loop), (
            f"the loop never pushes {link}.other back onto {work}, so it drains after the first pass "
            "and the trace reaches the neighbours and no further"
        )

    def test_the_walk_only_takes_a_step_it_has_not_already_taken(self, page: str) -> None:
        """The visited set is not tidiness; without it the worklist is what stops the page.

        Every flow this page is for splits and rejoins. Pushing a step that has already been walked
        means every path back to a fork is walked again, and a diamond a few steps deep turns one
        click into work the canvas is not worth. The guard and the push are one mechanism, so they
        are pinned against the same value rather than as two facts that happen to both be present.
        """
        work, link, loop = _walk(page)
        guarded = re.search(
            r"if\s*\(\s*!\s*(\w+)\.has\(\s*" + link + r"\.other\s*\)\s*\)\s*\{?[^{}]*"
            r"\b" + work + r"\.push\(\s*" + link + r"\.other\s*\)",
            loop,
        )

        assert guarded is not None, (
            f"{work}.push is not guarded by a membership test on {link}.other, so a flow that splits "
            "and rejoins is walked once per path back to the fork"
        )

    def test_a_node_is_matched_by_its_step_id_rather_than_its_element_id(self, page: str) -> None:
        """`id="node-totals"` is the element; `totals` is what an edge names and what `choose` takes.

        Dropping the round trip is a one-token edit with no visible fault anywhere near it: every
        lookup still runs, every state is still assigned, and every name is still spelled. It is the
        comparison that fails -- `node-totals` matches no step in the trace -- so every node falls
        through to `dim` and choosing any step blacks out the whole canvas.
        """
        code = _js(page)
        prefix = re.search(r'<a class="node[^"]*" id="([a-z]+-)', page)
        assert prefix is not None, "no node carries a prefixed id, so there is no round trip to make"
        worn = prefix.group(1)
        strip = r"(?:\.replace\(\s*/\^" + worn + r"/[^)]*\)|\.slice\(\s*" + str(len(worn)) + r"\s*\))"

        reached = _with_callees(code, _block(code, "function choose("))
        assert re.search(r"\.get\(\s*\w+\.id" + strip + r"\s*\)", reached), (
            "the trace looks a node up by its raw element id, which matches no step and dims everything"
        )
        assert re.search(r"choose\(\s*\w+\.id" + strip + r"\s*\)", code), (
            "the node's click hands choose() a prefixed element id instead of the step it stands for"
        )


class TestThePageOpensShowingTheWholeFlow:
    """The trace answers a question, and on arrival the reader has not put one.

    Measured on the flow this page is for, choosing a late step lights 427 nodes of 672. Tracing on
    the call the page makes for itself at load would therefore open the canvas two-thirds lit and
    one-third dark -- which reads as a drawing that failed to finish, not as an answer, and it is an
    answer to a question nobody asked. So the opening call suppresses the trace and every selection
    a reader makes takes the default. All three parts of that are one-token edits away from gone,
    and none of them changes a name, a selector or a state that any other test in this file looks at.
    """

    def test_choose_can_be_told_not_to_trace_and_traces_unless_it_is(self, page: str) -> None:
        """The default is the whole safety of the arrangement.

        Flip it to `false` and tracing is silently dead from both the node and the step list, while
        every test that follows `choose` into its callees still passes -- the body of `traceFrom` is
        reachable from the text of the call whether or not the call is ever taken.
        """
        declared = re.search(r"function\s+choose\s*\(\s*\w+\s*,\s*(\w+)\s*=\s*(\w+)\s*\)", _js(page))

        assert declared is not None, (
            "choose takes no second parameter, so the page cannot open a step without also answering "
            "a question about it"
        )
        assert declared.group(2) == "true", (
            f"choose defaults {declared.group(1)} to {declared.group(2)!r}, so a reader clicking a node "
            "or a step in the list gets no trace at all and nothing on the page says why"
        )

    def test_the_trace_inside_choose_is_taken_only_when_it_was_asked_for(self, page: str) -> None:
        """The guard is the parameter's only reason to exist.

        Unguarded, the parameter still exists, every call site still compiles, and the page opens
        with two-thirds of the canvas dimmed against a question nobody put to it.
        """
        code = _js(page)
        declared = re.search(r"function\s+choose\s*\(\s*\w+\s*,\s*(\w+)\s*=", code)
        assert declared is not None, "choose takes no second parameter to guard anything with"
        asked = declared.group(1)
        body = _block(code, "function choose(")

        calls = re.findall(r"\btraceFrom\(", body)
        assert len(calls) == 1, f"choose calls traceFrom {len(calls)} times, so one of them is not guarded by anything"
        assert re.search(r"if\s*\(\s*" + asked + r"\s*\)\s*\{?\s*traceFrom\(", body), (
            f"traceFrom is called without asking {asked}, so the page's own opening selection dims the "
            "canvas before the reader has looked at it"
        )

    def test_only_the_call_made_at_load_suppresses_the_trace(self, page: str) -> None:
        """Which call passes `false` is the distinction, not that the word appears somewhere.

        Two mistakes are the same edit away and neither shows up as a missing name: drop the
        argument from the load-time call and the page opens dimmed, or pass it from a reader's
        handler and the control the feature is for does nothing. So this reads every call site --
        with its nesting, because a statement at the top level of the script is one that runs
        because the page loaded rather than because someone clicked.
        """
        calls = _calls(page, "choose")

        assert calls, "nothing in the script ever chooses a step"
        quiet = [(depth, args) for depth, args in calls if len(args) > 1]
        assert len(quiet) == 1, (
            f"{len(quiet)} calls pass a second argument to choose; exactly one -- the one made at load -- should"
        )
        depth, args = quiet[0]
        assert args[1] == "false", f"the load-time call passes {args[1]!r} rather than a literal false"
        assert depth == 0, (
            "the call that suppresses the trace sits inside a handler, so it is a reader's selection that "
            "silently does not trace"
        )
        assert [a for d, a in calls if d == 0] == [args], (
            "a second statement at the top level chooses a step, and one of them will trace on load"
        )
        assert re.search(r"dataset\.first|'data-first'", args[0]), (
            f"the load-time call opens on {args[0]}, not on the step the page declares it opens with"
        )

    def test_no_element_arrives_already_wearing_a_trace_state(self, page: str) -> None:
        """The other way to open dimmed, and it does not go through the script at all.

        A state baked into the markup survives every assertion about what the script does, and the
        canvas opens with part of itself faded and no question behind it. The states start on the
        page the moment a reader asks for one, and not before.
        """
        _, states = _painter(page)
        worn = set()
        for value in re.findall(r'class="([^"]*)"', page):
            worn |= set(value.split()) & states

        assert not worn, f"the page is rendered with {sorted(worn)} already on it, so it opens part-dimmed"


class TestTheTraceRunsTheWayItSaysItDoes:
    """A trace reported backwards is the worst thing this feature can do.

    An absent highlight is visible -- the reader sees nothing happen and clicks something else. A
    highlight that names the wrong direction is believed: the reader follows it, reads "this is what
    feeds the step", and takes a decision about a figure on a flow that runs the other way. The two
    maps are interchangeable at a glance, both are `Map<string, link[]>`, and swapping them (or
    swapping the two ends as the index is built) leaves every name, literal and selector in place.
    """

    @pytest.mark.parametrize(
        ("direction", "keyed_on", "reports"),
        [("upstream", "to", "from"), ("downstream", "from", "to")],
    )
    def test_each_direction_walks_the_map_built_for_it(
        self, page: str, direction: str, keyed_on: str, reports: str
    ) -> None:
        """What feeds a step is found by looking that step up as an edge's `to` and taking its `from`.

        Both facts are derived from the script rather than named here: which map the walk pairs with
        each direction, and what that map was built out of. Rename the maps and this still holds;
        swap them, or swap the ends they are built from, and it does not.
        """
        sides = _sides(page)
        built = _index_build(page)

        assert set(sides) == {"upstream", "downstream"}, f"the walk does not run both ways: {sides}"
        assert direction in sides, f"nothing is walked for {direction}"
        used = sides[direction]
        assert used in built, f"{direction} walks {used}, which is not built from the drawn connections at all"
        assert built[used] == (keyed_on, reports), (
            f"{direction} walks a map keyed on data-{built[used][0]} reporting data-{built[used][1]}; "
            f"following it tells a reader the flow runs the other way, and it looks exactly as convincing"
        )


class TestClickingOneLineAsksAboutThatLine:
    """A reader pointing at a line in a crowded area is asking where it goes, not to move the page."""

    def test_clicking_a_line_lights_it_and_both_its_ends(self, page: str) -> None:
        script = _script(page)
        assert "closest('.edge')" in script, "a line cannot be clicked at all"
        branch = _block(script[script.index("closest('.edge')") :], "{")

        assert re.search(r"data(?:set\.from|-from)", branch), "the clicked line never names its upstream end"
        assert re.search(r"data(?:set\.to|-to)", branch), "the clicked line never names its downstream end"

    def test_clicking_a_line_does_not_change_the_step_being_read(self, page: str) -> None:
        """Detail and data panes swapping under a reader who clicked a line loses their place.

        The question was about the line. Answering a different one -- and scrolling the node into
        view, and re-fetching a port -- is the page taking over a reader's attention uninvited.
        """
        script = _script(page)
        assert "closest('.edge')" in script, "a line cannot be clicked at all"
        branch = _block(script[script.index("closest('.edge')") :], "{")

        assert "choose(" not in branch, "clicking a line also selects a step, which moves both panes"

    def test_the_clicked_line_is_the_one_that_lights_and_the_rest_go_out(self, page: str) -> None:
        """Naming both ends proves the line was read, not that anything was lit with them.

        The interaction inverts on one swapped ternary: the clicked line goes dark and the other
        seven hundred and sixty-four light up. Nothing about that is a silent failure to the code --
        every name is used, every element is painted -- and to the reader it is a canvas that lit
        everything except the thing they pointed at.
        """
        var, branch = _edge_click(page)

        assert re.search(r"querySelectorAll\('\.edge'\)", branch), "the click never looks at the other connections"
        lit = re.search(
            r"paint\(\s*(\w+)\s*,\s*(?:\1\s*===\s*" + var + r"|" + var + r"\s*===\s*\1)"
            r"\s*\?\s*'direct'\s*:\s*'dim'\s*\)",
            branch,
        )
        assert lit is not None, (
            "the clicked connection is not the one put into 'direct' with every other put into 'dim' -- "
            "check the states have not been swapped, which lights the whole canvas except the line clicked"
        )

    def test_clicking_a_line_puts_the_nodes_out_before_lighting_its_two_ends(self, page: str) -> None:
        """The lines are answered and the nodes are left as the last question left them.

        Without this the previous trace's `upstream` and `downstream` nodes stay lit around a single
        lit line, and the two halves of the canvas answer two different questions at once -- the
        older one looking, to a reader who has clicked since, like part of the answer to the new one.
        """
        _, branch = _edge_click(page)

        assert re.search(r"querySelectorAll\('\.node'\)\)\s*(?:\{\s*)?paint\(\s*\w+\s*,\s*'dim'\s*\)", branch), (
            "clicking a line never puts the nodes back into 'dim', so whatever a previous selection lit "
            "stays lit beside an answer to a different question"
        )

    def test_both_ends_of_the_clicked_line_are_the_ones_lit(self, page: str) -> None:
        """A line lit between two dim nodes says where it runs and not what it joins.

        The ends have to be looked up and painted `direct`, and painting them `dim` -- the state
        every other node was just given -- is a one-word edit that leaves both lookups, both
        attributes and the whole loop in place.
        """
        var, branch = _edge_click(page)
        ends = re.search(r"\[\s*" + var + r"\.dataset\.(\w+)\s*,\s*" + var + r"\.dataset\.(\w+)\s*\]", branch)

        assert ends is not None, "the clicked line's two ends are never gathered to be looked up"
        looked_up = set(ends.groups())
        assert looked_up == {"from", "to"}, f"the ends looked up are {sorted(looked_up)}, not both of them"
        found = re.search(r"const\s+(\w+)\s*=\s*wrap\.querySelector\('#node-'\s*\+\s*(\w+)\)", branch)
        assert found is not None, "the ends are named but no node is ever found for them"
        assert re.search(r"paint\(\s*" + found.group(1) + r"\s*,\s*'direct'\s*\)", branch), (
            "the node found for each end is not painted 'direct', so the line lights between two nodes "
            "that look no different from the six hundred it does not touch"
        )


class TestThereIsAWayBackToTheWholeCanvas:
    """Six hundred dimmed nodes with no undo is a worse page than the one this replaces."""

    def test_a_control_offers_to_clear_the_trace(self, page: str) -> None:
        found = re.search(r'<button[^>]*data-trace="clear"[^>]*>', page)

        assert found is not None, "nothing on the page says how to get the whole canvas back"
        assert "title=" in found.group(0), "an unlabelled glyph is not an obvious way back"

    def test_the_zoom_bar_cannot_swallow_the_clear_click(self, page: str) -> None:
        """The zoom handler takes every button inside `.zoom` and reads `data-zoom` off it.

        A clear control placed there with no `data-zoom` falls through that handler's else and zooms
        the canvas out on its way to clearing -- so either it sits outside the bar, or the handler
        learns to let go of buttons that are not its own.

        The block read has to be the handler's own. Taken from the first brace after the lookup, the
        extraction lands past the end of a brace-free if/else chain and into the wheel listener two
        statements later, where it can say nothing at all about either half of this.
        """
        code = _js(page)
        at = code.index("closest('.zoom button')")
        handler = _block(code[code.rindex("addEventListener('click'", 0, at) :], "{")
        button = re.search(r"const\s+(\w+)\s*=\s*e\.target\.closest\('\.zoom button'\)", handler)
        bar = re.search(r'<div class="zoom".*?</div>', page, re.DOTALL)

        assert 'data-trace="clear"' in page, "there is no clear control yet, so this invariant has nothing to hold"
        assert bar is not None and button is not None
        assert 'data-trace="clear"' not in bar.group(0), (
            "the clear control sits inside .zoom, where the zoom handler claims every button it finds"
        )
        assert re.search(r"if\s*\(\s*!\s*" + button.group(1) + r"\s*\)\s*return", handler), (
            "the zoom handler never lets go of a click that found no zoom button, so any control added to "
            "the bar falls through its else and zooms the canvas out on the way to doing its own job"
        )

    def test_clearing_puts_every_line_and_node_back(self, page: str) -> None:
        """A control that clears three of the four states leaves a canvas still half in the dark."""
        script = _script(page)
        found = re.search(r"data(?:set\.trace|-trace)", script)

        assert found is not None, "the clear control is drawn but never handled"
        reached = _with_callees(script, _block(script[found.start() :], "{"))
        for state in ("direct", "upstream", "downstream", "dim"):
            assert f"'{state}'" in reached, f"clearing leaves the {state!r} state on the canvas"

    def test_the_way_back_reaches_the_nodes_as_well_as_the_lines(self, page: str) -> None:
        """Naming all four states says what is removed, not what it is removed from.

        Narrow the selector to the connections and the four names are all still there, so nothing
        above fails -- and the lines come back while six hundred nodes stay at opacity .22 for the
        rest of the session, with the one control that could have fixed it already spent. That is
        the worst shape a fault in this feature can take: the way out of a dimmed canvas leaves the
        canvas dimmed, and the reader has no reason to press it twice.
        """
        looked_at = re.findall(r"querySelectorAll\('([^']*)'\)", _clear_handler(page))
        named = {part.strip() for selector in looked_at for part in selector.split(",")}

        assert {".edge", ".node"} <= named, (
            f"clearing looks at {sorted(named)}; whichever of the lines and the nodes it misses stays "
            "in the state the last trace left, with nothing on the page left to say why"
        )

    def test_clearing_undoes_exactly_the_states_the_painting_assigns(self, page: str) -> None:
        """The two vocabularies are compared with each other rather than with a list kept here.

        A fifth state, or a renamed one, is added in the painting and forgotten in the clearing --
        and a list written into this test would be stale in the same edit that broke them, so it
        would keep passing. Reading both off the script means the two cannot drift by one name.
        """
        painter, assigns = _painter(page)
        cleared = re.search(r"classList\.remove\(([^)]*)\)", _clear_handler(page))

        assert cleared is not None, "the way back removes no state class at all"
        removes = set(re.findall(r"'(\w+)'", cleared.group(1)))
        assert removes == assigns, (
            f"{painter}() assigns {sorted(assigns)} and clearing removes {sorted(removes)}; the difference "
            "is the part of the canvas that never comes back"
        )


# ---------------------------------------------------------------------------
# The panel is built for the step being opened, not for all six hundred and seventy-two
# ---------------------------------------------------------------------------
#
# Measured on the real page -- 672 steps, output_files/runs/mrr/page.html -- the data panels were
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
# The tests below hold the other half: that the page carries almost none of it, that every step the
# flow can select has a panel to fetch, that the fetch names the step and happens once, and -- the
# one that turns a fast page into a broken one -- that a reader whose panel cannot be had is told so
# rather than left in front of an empty pane.


def _strings(code: str) -> list[str]:
    """Every string literal in a piece of script, in source order.

    What a reader is told when something fails is a literal, so the words themselves have to be
    reachable from a test; there is nowhere else for a message to live.
    """
    return [hit.group(2) for hit in re.finditer(r"(['\"`])((?:\\.|(?!\1).)*)\1", code, re.DOTALL)]


def _invocations(code: str, name: str) -> list[str]:
    """Every call to ``name``, as the text of its arguments, balanced rather than read to the comma.

    A URL built by concatenation or a template is one argument full of punctuation, and reading it
    to the first bracket would cut it in half.
    """
    found: list[str] = []
    for hit in re.finditer(r"(?<![\w.])(?<!function )" + re.escape(name) + r"\s*\(", code):
        depth, start = 0, hit.end() - 1
        for i in range(start, len(code)):
            if code[i] in "([{":
                depth += 1
            elif code[i] in ")]}":
                depth -= 1
                if depth == 0:
                    found.append(code[start + 1 : i])
                    break
    return found


def _function(code: str, name: str) -> tuple[list[str], str]:
    """One named function's parameters and its body, whether or not it is declared async."""
    found = re.search(r"(?:async\s+)?function\s+" + re.escape(name) + r"\s*\(([^)]*)\)", code)
    assert found is not None, f"the script declares no function {name}()"
    params = [part.split("=")[0].strip() for part in found.group(1).split(",") if part.strip()]
    return params, _block(code[found.start() :], "{")


def _panel_fn(page: str) -> tuple[str, list[str], str]:
    """The code that goes and gets a step's panel: its name, its parameters and its body.

    Found by what it does rather than by what it is called -- it is reached from `choose`, it
    fetches, and it is about the step panels rather than about a window of rows, which is what tells
    it apart from `live()`. Naming it here would make every assertion below a statement about the
    author's choice of identifier instead of about the page.
    """
    code = _js(page)
    params, chosen = _function(code, "choose")
    if "fetch(" in chosen and "stepdata" in chosen:
        return "choose", params, chosen
    for name in sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", chosen))):
        if not re.search(r"(?:async\s+)?function\s+" + re.escape(name) + r"\s*\(", code):
            continue
        reached, body = _function(code, name)
        if "fetch(" in body and "stepdata" in body:
            return name, reached, body
    raise AssertionError(
        "choosing a step reaches nothing that fetches a panel for it, so the panels either never "
        "arrive or arrive by some route the selection does not control"
    )


def _panel_request(page: str) -> str:
    """The URL the page asks a panel for, up to the point where it appends the step it wants.

    Derived from the script rather than written down here, so that the case which puts this request
    to the real service is asking the question the page actually asks -- a test that spelled the
    route out would keep passing while the page called something the service has never heard of.
    """
    _, _, body = _panel_fn(page)
    calls = _invocations(body, "fetch")
    assert len(calls) == 1, f"the panel is fetched {len(calls)} times in one function, so which one arrives is a race"
    literal = re.search(r"(['\"`])((?:\\.|(?!\1).)*)\1", calls[0], re.DOTALL)
    assert literal is not None, "the panel's URL is not written down anywhere in the call that fetches it"
    return literal.group(2).split("${")[0]


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


class TestTheFetchNamesTheStep:
    """Showing one step's rows under another step's name is silent, and it is believed."""

    def test_choosing_a_step_is_what_goes_and_gets_its_panel(self, page: str) -> None:
        """`choose` is the selection seam: the node, the step list and the opening call all pass here.

        Hung off the node's click handler instead, the step list would silently show a reader the
        panel of whatever step they looked at last -- and the page's own opening selection, which
        takes neither path, would show nothing.
        """
        name, params, _ = _panel_fn(page)
        chosen_params, chosen = _function(_js(page), "choose")

        if name != "choose":
            # Which step is fetched has to be the one being chosen, not one read back off the page
            # or remembered from last time: `panelFor(wrap.dataset.first)` compiles, runs, and shows
            # the opening step's rows under every other step's name.
            assert re.search(r"\b" + re.escape(name) + r"\s*\(\s*" + chosen_params[0] + r"\b", chosen), (
                f"{name}() is not called with the step {chosen_params[0]} that was just chosen"
            )
        assert params, f"{name}() takes no step, so it cannot be asking for one step's panel rather than another's"

    def test_the_request_asks_for_the_step_being_chosen_and_asks_the_run_that_is_serving(self, page: str) -> None:
        """Two mistakes here are invisible and both mis-attribute rows.

        A URL that does not carry the step fetches whatever the route defaults to and draws it under
        the chosen step's name. An absolute URL -- `/api/panel` -- resolves against the service root
        rather than against the run, so a service holding two flows answers from whichever one it
        picks, and the rows are another flow's entirely.
        """
        _, params, body = _panel_fn(page)
        asked = _invocations(body, "fetch")[0]
        url = _panel_request(page)

        assert re.search(r"\b" + params[0] + r"\b", asked), (
            f"the panel request does not mention {params[0]}, so every step asks for the same panel"
        )
        assert not url.startswith(("/", "http:", "https:", "//")), (
            f"the panel is fetched from {url!r}, which is resolved against the service root rather than "
            "against the run the page belongs to"
        )
        assert "?" in url and url.endswith("="), (
            f"the panel is fetched from {url!r}; the step is named as a query parameter, the way a window of "
            "rows is, so that a step id can never be read as a path"
        )

    def test_the_request_the_page_makes_is_the_one_the_service_answers(
        self, tmp_path: Path, page: str, panels: dict[str, str], steps: tuple[str, ...]
    ) -> None:
        """The two halves are written in different files and nothing else compares them.

        A renamed route or a renamed parameter leaves both sides looking entirely correct: the page
        asks, the service refuses, and what the reader gets is every panel replaced by the notice
        that says the service is not answering -- which reads as "the service is down", not as "the
        page and the service disagree about a word".
        """
        import json
        import urllib.request
        from urllib.parse import quote

        import polars as pl

        from dantron_etl.preview import PAGE_NAME, persist_panels, persist_ports, serve

        root = tmp_path / "runs"
        persist_ports({"entries.rows": pl.DataFrame({"party": ["a"]})}, root / "fixture")
        persist_panels(panels, root / "fixture")
        (root / "fixture" / PAGE_NAME).write_text(page, encoding="utf-8")

        server = serve(root, host="127.0.0.1", port=0)
        try:
            served = f"http://127.0.0.1:{server.server_address[1]}/fixture/"
            for step in steps:
                with urllib.request.urlopen(served + _panel_request(page) + quote(step), timeout=5) as answer:
                    assert answer.status == 200
                    assert answer.read().decode("utf-8") == panels[step], f"{step} was answered with another panel"
            with urllib.request.urlopen(served + "api/ports", timeout=5) as answer:
                assert json.loads(answer.read())["ports"], "the run the panels were asked of is not serving its ports"
        finally:
            server.shutdown()
            server.server_close()


class TestAPanelIsFetchedOnceAndThenShown:
    """Asking twice is slow; arriving and staying hidden is worse, because it looks like no data."""

    def test_a_panel_already_in_the_page_is_not_asked_for_again(self, page: str) -> None:
        """The panel in the document is the cache, and the guard is the only thing that reads it.

        Without it a reader clicking between two steps re-fetches both every time -- and the second
        copy is inserted beside the first, so the pane then holds two sections answering to one step
        and shows whichever the selection rule reaches first.
        """
        _, params, body = _panel_fn(page)
        step = params[0]
        guard = None
        for hit in re.finditer(r"querySelector\(([^;]*)", body):
            if "stepdata" in hit.group(1) and re.search(r"\b" + step + r"\b", hit.group(1)):
                guard = hit
                break

        assert guard is not None, (
            f"nothing looks for a panel already answering to {step} before fetching one, so every selection "
            "asks the service again"
        )
        fetched = body.index("fetch(")
        assert guard.start() < fetched, "the page fetches the panel first and looks for the one it has afterwards"
        assert re.search(r"\breturn\b", body[guard.start() : fetched]), (
            "the lookup for a panel already in the page is not followed by a return, so the fetch happens anyway "
            "and the lookup is a statement with no effect"
        )

    def test_what_the_service_answers_with_is_what_goes_into_the_page(self, page: str) -> None:
        """The other side of the cache: a fetch whose answer is dropped is asked again forever.

        Inserting a placeholder and discarding the response leaves a reader with a pane that never
        fills, a guard that is satisfied by the placeholder, and a service that is asked once and
        then never again -- so no amount of waiting or re-clicking produces the rows.
        """
        _, _, body = _panel_fn(page)
        read = re.search(r"(?:(\w+)\s*=\s*)?await\s+[\w.]+\.text\(\)", body)

        assert read is not None, "the response is never read as markup, so nothing the service sent is used"
        carrier = read.group(1)
        writes = re.findall(r"(?:insertAdjacentHTML|innerHTML|outerHTML|replaceChildren|append)\s*[=(][^;]*", body)
        assert any(".text()" in write or (carrier and re.search(r"\b" + carrier + r"\b", write)) for write in writes), (
            "the markup the service answered with never reaches the page; the panel is fetched and thrown away"
        )

    def test_a_panel_that_arrives_after_the_selection_is_actually_revealed(self, page: str) -> None:
        """A panel arrives hidden, which is how it was rendered, and the fetch outlives the click.

        The selection hides every section that is not the chosen step's and shows the one that is --
        and when it runs, the panel is not there yet. Insert it afterwards and nothing clears the
        attribute: the reader gets an empty pane, with the data sitting in the document beside it.
        That is indistinguishable, from the outside, from a step that produced nothing.
        """
        name, _, body = _panel_fn(page)
        placed = max(
            (body.rfind(verb) for verb in ("insertAdjacentHTML", "innerHTML", "replaceChildren", "append")),
            default=-1,
        )
        assert placed >= 0, "the panel is never put into the page at all"

        after = body[placed:]
        if name != "choose":
            # Or the selection does it, once the panel has landed -- which is the same fact written
            # on the other side of the call, and only if the call is waited for. An unawaited fetch
            # followed by the show-and-hide pass is the bug itself: the pass runs, finds no panel,
            # and the panel lands hidden a moment later.
            chosen = _function(_js(page), "choose")[1]
            call = re.search(r"await\s+" + re.escape(name) + r"\s*\(", chosen)
            after += chosen[call.end() :] if call else ""

        assert re.search(r"\bhidden\b|\bchoose\s*\(", after), (
            "nothing shows the panel once it has arrived, so a reader who clicked a step is left in front of an "
            "empty pane with that step's data hidden in the document beside it"
        )


class TestWhenThePanelCannotBeHad:
    """The page is written to disk and opened as a file, and it is no longer self-contained.

    This is the failure that turns a fast page into a broken one: a reader opens the file, clicks a
    step, and gets nothing -- no rows, no error, no reason. The rows have always been served, and
    the page said so where they would have been; the panel around them now needs the same sentence,
    because there is no longer any markup left to put the sentence in.
    """

    def test_a_panel_that_cannot_be_fetched_leaves_the_reader_a_sentence_rather_than_a_blank_pane(
        self, page: str
    ) -> None:
        """A refusal, a dead socket and a page opened from the filesystem all land here.

        Caught and written where the panel would have been -- not logged, not thrown, not left as a
        spinner. A message in the console is a message to whoever wrote the page, and the reader is
        somebody looking at a figure.
        """
        _, _, body = _panel_fn(page)

        assert "catch" in body, "a fetch that fails is not caught, so the page opened as a file does nothing at all"
        writes = re.findall(
            r"(?:textContent|innerHTML|outerHTML|insertAdjacentHTML|append|replaceChildren)\s*[=(][^;]*", body
        )
        said = [text for write in writes for text in _strings(write)]
        for name in {word for write in writes for word in re.findall(r"\b([A-Za-z_]\w*)\b", write)}:
            assigned = re.search(r"\b" + name + r"\s*=\s*(['\"`][^'\"`]*['\"`])", body)
            if assigned:
                said += _strings(assigned.group(1))

        assert any(re.search(r"service", text, re.IGNORECASE) for text in said), (
            f"nothing the panel puts in front of a reader names the service: it says {said}. A pane that stays "
            "empty when the page is opened as a file is read as a step that produced no rows"
        )
        assert "datapane" in body, (
            "the notice is not put where the panel would have been, which is where the reader is looking"
        )

    def test_a_panel_that_never_arrived_is_not_remembered_as_though_it_had(self, page: str) -> None:
        """The notice must not be written as the panel it stands in for.

        The cache is "a section answering to this step is in the page". Write the notice as one of
        those and the reader is locked out: the service comes back, they click the step again, the
        guard finds the notice, and the page never asks again. Every other assertion in this file
        still passes -- the fetch exists, the guard exists, the message exists -- and the only way
        out is a reload the reader has no reason to try.
        """
        _, _, body = _panel_fn(page)
        written = [text for text in _strings(body) if "<" in text]

        for text in written:
            assert not re.search(r'class\s*=\s*"?[^"]*\bstepdata\b', text), (
                f"the panel function writes its own stepdata section ({text!r}); the only markup wearing that "
                "class should be the panel the service answered with, or a failure is cached as a success"
            )


# The half of the fetch that lives in the script. Five mutations were green against the suite as it
# stood: the awaited call unawaited, the whole post-arrival block deleted, the stale guard deleted,
# the pre-fetch hide narrowed to `.detail`, and the shipped panel emptied.
#
# They were green for one reason, and it is why these helpers exist. An assertion of the form
# "somewhere after the insert, the word `hidden` appears" reads its region off the last
# insert-looking verb in the function -- which is `pane.append(notice)` in the FAILURE branch, later
# in the source than the success path. So the region examined was the failure path's tail, `hidden`
# matched `notice.hidden`, and the assertion held however the success path was mangled.
#
# Everything below therefore derives its region from the write that carries the FETCHED markup, and
# proves the region is the right one by refusing to reach the failure branch at all.


def _hides(region: str) -> list[tuple[int, set[str], str]]:
    """Every pass that shows and hides what it selected, as (where it sits, which classes, the statement).

    The classes are read out of the whole selector rather than off its first name: a pass over
    `.detail, .stepdata` swaps two kinds of section, and reading only the first is how narrowing it
    to `.detail` -- which leaves the previous step's rows on screen under the new step's name --
    went unnoticed.
    """
    found: list[tuple[int, set[str], str]] = []
    for hit in re.finditer(r"querySelectorAll\(\s*(['\"`])(.*?)\1\s*\)", region):
        statement = region[hit.end() : hit.end() + 240].split(";")[0]
        if re.search(r"\bhidden\b", statement):
            found.append((hit.start(), set(re.findall(r"\.([a-z][\w-]*)", hit.group(2))), statement))
    return found


def _notice(page: str) -> str | None:
    """The element the failure branch makes to stand in for a panel that could not be had."""
    _, _, body = _panel_fn(page)
    made = re.search(r"(\w+)\s*=\s*document\.createElement\(", body)
    return made.group(1) if made else None


def _panel_insert(body: str) -> tuple[int, str]:
    """Where the fetched markup is written into the page, and the element it is written into.

    Found by following what the response was read into -- the same carrier
    `test_what_the_service_answers_with_is_what_goes_into_the_page` identifies -- rather than by
    looking for the last verb that could insert something, because that one is in the catch block.
    """
    read = re.search(r"(?:(\w+)\s*=\s*)?await\s+[\w.]+\.text\(\)", body)
    assert read is not None, "the response is never read as markup, so nothing the service sent is used"
    carrier = read.group(1)
    verbs = r"insertAdjacentHTML|innerHTML|outerHTML|replaceChildren|append"
    for hit in re.finditer(r"([\w.]+)\.(?:" + verbs + r")\s*[=(]([^;]*)", body):
        written = hit.group(2)
        if ".text()" in written or (carrier and re.search(r"\b" + carrier + r"\b", written)):
            return hit.start(), hit.group(1)
    raise AssertionError("the markup the service answered with never reaches the page; the panel is fetched and lost")


def _around_the_fetch(page: str) -> tuple[str, str, str, str]:
    """`choose`, cut at the point its step's panel is gone for.

    Returns the step it was handed, what runs before the panel is asked for, the call that asks, and
    what runs once it has landed -- with the failure branch taken out of that last region, so a case
    about "after the panel arrives" cannot be satisfied by what the page does when it does not.

    The cut is derived rather than written down: when the fetching lives in a helper the seam is the
    call `choose` makes to it, and when it is inline the seam is the write that carries the fetched
    markup into the page.
    """
    name, _, body = _panel_fn(page)
    params, chosen = _function(_js(page), "choose")
    assert params, "choose() is handed no step, so nothing it does can be about one step rather than another"
    if name == "choose":
        at, _ = _panel_insert(chosen)
        read = re.search(r"(?:\w+\s*=\s*)?(?:await\s+)?[\w.]+\.text\(\)", chosen)
        assert read is not None
        before, call, after = chosen[:at], read.group(0), chosen[at:]
    else:
        site = re.search(r"(?:await\s+)?(?<![\w.])(?<!function )" + re.escape(name) + r"\s*\([^;]*", chosen)
        assert site is not None, f"choose() never calls {name}(), so choosing a step does not go and get its panel"
        before, call, after = chosen[: site.start()], site.group(0), chosen[site.end() :]
    if "catch" in body:
        after = after.replace(_block(body[body.index("catch") :], "{"), "")
    return params[0], before, call, after


def _view_switch(page: str) -> tuple[str, str]:
    """The function that decides which entity's views are shown, and the variable it remembers it in.

    Found by what it does -- it is the one thing that hides `.view` against the key it was given --
    rather than by its name, so these cases stay about the page and not about an identifier.
    """
    code = _js(page)
    for hit in re.finditer(r"(?:async\s+)?function\s+(\w+)\s*\(", code):
        params, body = _function(code, hit.group(1))
        if not params or not any("view" in classes for _, classes, _ in _hides(body)):
            continue
        kept = re.search(r"(\w+)\s*=\s*" + params[0] + r"\s*;", body)
        assert kept is not None, (
            f"{hit.group(1)}() shows one entity's views and remembers nothing, so a panel arriving afterwards has "
            "nothing to be put back onto"
        )
        return hit.group(1), kept.group(1)
    raise AssertionError("nothing in the script decides which of a panel's views is the visible one")


def _rows_fn(page: str) -> str:
    """The function that goes and gets a window of rows for one open port."""
    code = _js(page)
    panel, _, _ = _panel_fn(page)
    for name in sorted(set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", code))):
        if name == panel:
            continue
        _, body = _function(code, name)
        if "fetch(" in body and "offset" in body:
            return name
    raise AssertionError("nothing in the script fetches a window of rows, so an open port can never fill")


def _top_level(page: str, pattern: str) -> bool:
    """Whether ``pattern`` occurs at the top level of the script rather than inside something.

    A choice remembered between one click and the next has to outlive both, and a `let` that moved
    inside a handler would reset on every selection -- which reads as the picker forgetting what the
    reader chose the moment a panel arrives.
    """
    depth = 0
    for hit in re.finditer(r"[{}]|" + pattern, _js(page)):
        if hit.group(0) == "{":
            depth += 1
        elif hit.group(0) == "}":
            depth -= 1
        elif depth == 0:
            return True
    return False


class TestThePanelIsWaitedForAndThenActuallyShown:
    """A panel that lands and stays hidden is worse than one that never lands: it reads as no data."""

    def test_the_panel_is_waited_for_rather_than_set_going_and_left_to_land(self, page: str) -> None:
        """The unawaited fetch, by name. It is one keyword and it breaks every step but the first.

        The pass that reveals a step's section runs, finds no section -- the fetch has not answered
        yet -- and leaves the reader in front of an empty pane; the panel lands a moment later still
        wearing the `hidden` it was rendered with. The page still opens correctly, because the step
        it opens on ships its panel in the document, so the failure begins at the reader's first
        click and never stops.
        """
        name, _, body = _panel_fn(page)
        _, chosen = _function(_js(page), "choose")
        if name == "choose":
            region = body
            sites = list(re.finditer(r"(?<![\w.])fetch\s*\(|[\w.]+\.text\s*\(", region))
        else:
            region = chosen
            sites = list(re.finditer(r"(?<![\w.])(?<!function )" + re.escape(name) + r"\s*\(", region))

        assert sites, f"choosing a step reaches {name}() nowhere, so no panel is ever asked for"
        for site in sites:
            assert re.search(r"\bawait\s+$", region[: site.start()]), (
                f"the panel is set going at offset {site.start()} and not waited for, so the selection finishes "
                "before the panel exists: the reader gets an empty pane, with that step's data landing hidden in "
                "the document beside it a moment later"
            )

    def test_the_panel_that_arrives_is_revealed_by_the_step_that_was_chosen(self, page: str) -> None:
        """Once it has landed, the section answering to the chosen step has its `hidden` cleared.

        Pinned as a pass over `.stepdata` keyed on the step the function was handed, rather than as
        the word `hidden` turning up somewhere later in the source. That weaker spelling is what let
        the whole post-arrival block be deleted with the suite still green: the region it read ran
        from the failure branch's own `append`, and `notice.hidden` satisfied it.
        """
        step, _, _, after = _around_the_fetch(page)
        notice = _notice(page)

        if notice:
            assert not re.search(r"\b" + notice + r"\b", after), (
                f"the region this case examines reaches {notice}, the element the failure branch makes; an "
                "assertion about what happens once the panel arrives would then be satisfied by what happens "
                "when it never does"
            )
        revealed = [
            classes
            for _, classes, statement in _hides(after)
            if "stepdata" in classes and re.search(r"\b" + step + r"\b", statement)
        ]
        assert revealed, (
            f"nothing after the panel lands shows the .stepdata answering to {step}; the panel is in the document "
            "wearing the hidden it was rendered with, and the reader is looking at an empty pane"
        )

    def test_a_panel_landing_after_the_reader_moved_on_is_dropped_rather_than_drawn(self, page: str) -> None:
        """The worst thing this page can do, and it is silent.

        A reader clicks a slow step, then a cached one. The first fetch resolves, inserts, and --
        with nothing to stop it -- runs its show-and-hide pass anyway: the second step's section is
        hidden and the first step's is shown, while the configuration pane, the chosen node and the
        step list all still name the second. One step's rows under another step's name, persistent,
        and believed.

        Derived structurally rather than by the name of the variable: what has to exist is a record
        of the selection taken before the panel is gone for, compared against the step this call was
        handed, returning early when they differ -- and the comparison has to come before anything
        is shown or hidden.
        """
        step, before, _, after = _around_the_fetch(page)
        recorded = re.findall(r"(\w+)\s*=\s*" + step + r"\s*;", before)

        assert recorded, (
            "nothing records which step was chosen before the panel is gone for, so a fetch that resolves late has "
            f"nothing to compare {step} against"
        )
        guard = None
        for mark in recorded:
            found = re.search(rf"(?:{mark}\s*!==?\s*{step}|{step}\s*!==?\s*{mark})", after)
            if found and re.search(r"\breturn\b", after[found.end() : found.end() + 80]):
                guard = found
                break

        assert guard is not None, (
            f"the panel is inserted and shown without asking whether {step} is still the step the reader is on; a "
            f"slow fetch then draws its own step's rows under the name of whatever they clicked next (what the "
            f"selection is recorded in: {recorded})"
        )
        swaps = [at for at, _, _ in _hides(after)]
        assert swaps and guard.start() < min(swaps), (
            "the stale selection is checked only after the page has been swapped over to the late panel, which is "
            "the wrong step on the screen for as long as the check takes to run"
        )

    def test_the_step_being_left_takes_its_rows_with_it_before_the_fetch_begins(self, page: str) -> None:
        """What is swapped before the await is both kinds of section, not just the configuration.

        The panel of the step being left is in the document; the panel of the step being opened is
        not here yet. Narrow this pass to `.detail` and for the whole duration of every fetch the
        reader is looking at the previous step's tables, under the new step's name, beside the new
        step's configuration -- which is the mis-attribution the rest of this file exists to
        prevent, held for as long as the service takes to answer.
        """
        step, before, _, _ = _around_the_fetch(page)
        passes = _hides(before)

        assert passes, "nothing is shown or hidden before the panel is asked for, so the old step stays on screen"
        swapped: set[str] = set()
        for _, classes, statement in passes:
            assert re.search(r"\b" + step + r"\b", statement), (
                f"a pass over {sorted(classes)} before the fetch hides sections without asking which step they "
                f"belong to, so it is not keyed on {step} at all"
            )
            swapped |= classes
        assert {"detail", "stepdata"} <= swapped, (
            f"only {sorted(swapped)} is swapped over before the fetch; the section carrying the previous step's "
            "rows stays visible under the new step's name until the service answers"
        )

    def test_a_port_left_open_is_filled_again_when_its_panel_comes_back(self, page: str) -> None:
        """Rows are fetched when a port is opened, and opening is the only thing that asks for them.

        A panel the reader has already opened a port in stays in the page, so coming back to that
        step shows an open port whose `toggle` will never fire again. Without the tail that clears
        the lazy marker and asks for its rows, the port sits open and empty -- which is the sentence
        this page spends everywhere else avoiding: a port that looks as though it carried nothing.
        """
        step, _, _, after = _around_the_fetch(page)
        rows = _rows_fn(page)
        looked = [
            hit.group(1)
            for hit in re.finditer(r"querySelector\(([^;]*)", after)
            if "stepdata" in hit.group(1) and "lazy" in hit.group(1)
        ]

        assert looked, f"nothing looks for an already-open port in {step}'s panel once it is back in the page"
        assert re.search(r"\b" + step + r"\b", looked[0]), (
            f"the search for an open port is not scoped to {step}, so it can fill a table in another step's panel"
        )
        assert re.search(r"delete\s+\w+\.dataset\.lazy", after), (
            "the port is filled without its lazy marker being cleared, so the toggle handler fetches it again"
        )
        assert _invocations(after, rows), (
            f"a port already open when its panel is shown again is never handed to {rows}(), so it stays open and "
            "empty with nothing the reader can do to fill it"
        )


class TestThePanelOpensOnTheEntityTheReaderChose:
    """Every view in a panel is rendered hidden. Something has to choose one, or the pane is blank."""

    def test_the_entity_the_reader_chose_is_put_back_onto_the_panel_that_arrives(self, page: str) -> None:
        """The buttons are in the page and the views are in the panel, so only a remembered key joins them.

        This one hides behind the opening step: that panel is in the document while the load-time
        selection runs, so the page arrives looking entirely correct and then every step the reader
        clicks opens with no tables in it at all -- which reads as a step that produced nothing,
        which is precisely what this page exists to be able to deny.
        """
        _, _, _, after = _around_the_fetch(page)
        switch, remembered = _view_switch(page)

        assert _top_level(page, rf"\b(?:let|var|const)\s+{remembered}\b"), (
            f"{remembered} is declared inside something, so the entity the reader chose does not outlive the click "
            "that chose it and cannot be put back onto a panel that arrives later"
        )
        assert any("dataset.pick" in argument for argument in _invocations(_js(page), switch)), (
            f"nothing hands {switch}() what a picker button says it stands for, so the remembered entity is never "
            "the one the reader clicked"
        )
        assert remembered in [argument.strip() for argument in _invocations(after, switch)], (
            f"the panel goes into the page and {switch}({remembered}) is never run on it; every view in it was "
            "rendered hidden, so the reader opens the step and finds no tables at all"
        )

    def test_the_pickers_buttons_and_a_panels_views_name_the_same_entities(
        self, page: str, fixture_flow_sandbox: Path
    ) -> None:
        """The two halves are built by separate passes over the trace, and nothing else compares them.

        The picker is rendered with the page and the views are rendered into the panels, so a pass
        that offered entities the panels were not keyed on would put the reader one click away from
        a pane with every view hidden -- and the page, the panel and the picker would each look
        perfectly correct on their own.

        Built here rather than taken from the shared fixture because two builds of one run is the
        case: the page is written once and the panels are written beside it, and it is the second
        pass drifting from the first that does this.
        """
        from scripts.pipeline_walkthrough_page import build_panels

        spec = read_pipeline_file(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
        with chdir(fixture_flow_sandbox):
            trace = run_pipeline(spec)
        fetched = build_panels(spec, trace, FocusSpec(label="party", identity=("party",)))
        offered = re.findall(r'<button data-pick="([^"]*)"', page)

        assert offered, "the picker offers nothing, so there is no entity a panel could be opened on"
        assert len(set(offered)) == len(offered), f"the picker offers {offered}, which names one entity twice"
        seen = 0
        for step, panel in fetched.items():
            keys = re.findall(r'<div class="view" data-for="([^"]*)"', panel)
            if not keys:
                continue
            seen += 1
            assert set(keys) == set(offered), (
                f"{step}'s panel carries views for {sorted(set(keys))} while the picker offers "
                f"{sorted(set(offered))}; the difference is a button that hides every table in the panel, or a "
                "table no button can reach"
            )
        assert seen, "no panel carries a view at all, so this would pass by having nothing to compare"


class TestTheNoticeStandsWhereThePanelWouldHaveBeen:
    """A page opened as a plain file is an ordinary case, so the failure path is a shipped feature."""

    def test_the_sentence_is_put_where_the_panel_would_have_gone(self, page: str) -> None:
        """Where the reader is looking, which is the pane a panel is inserted into and nowhere else.

        Asserting that the word `datapane` appears somewhere in the function does not say this: the
        function already looks that pane up for its own reasons, so appending the sentence to the
        document body instead satisfies it while the sentence lands at the bottom of the page, past
        the canvas and the step list -- out of sight, and read as nothing having happened at all.
        """
        _, _, body = _panel_fn(page)
        _, into = _panel_insert(body)
        notice = _notice(page)

        assert notice is not None, "nothing is made to stand in for a panel that could not be had"
        placed = re.search(
            r"([\w.]+)\.(?:append|appendChild|insertAdjacentElement|replaceChildren)\s*\([^)]*\b" + notice + r"\b",
            body,
        )
        assert placed is not None, f"{notice} is made and never put into the page, so the reader is told nothing"
        assert placed.group(1) == into, (
            f"the notice is put into {placed.group(1)} while a panel is put into {into}; a sentence explaining an "
            "empty pane has to be in the pane that is empty"
        )

    def test_a_failure_after_a_success_says_so_again_rather_than_staying_down(self, page: str) -> None:
        """The notice is made once and moved, so it is also hidden once and has to come back up.

        A successful fetch takes the sentence down -- otherwise it sits above a panel that did
        arrive. Which means the second failure, and every failure after it, is silent unless the
        failure branch puts it back: the reader clicks a step the service cannot answer for and gets
        a blank pane with no reason in it, having been told once, about a different step, some time
        ago.
        """
        _, _, body = _panel_fn(page)
        notice = _notice(page)

        assert notice is not None, "nothing is made to stand in for a panel that could not be had"
        taken_down = re.search(rf"\b{notice}\.hidden\s*=\s*true", body)
        assert taken_down is not None, (
            f"{notice} is never hidden, so the sentence about a service that did not answer stays on the screen "
            "above the panels that did arrive"
        )
        assert taken_down.start() < body.index("fetch("), (
            "the notice is taken down after the fetch rather than before it, so it stays on the screen while the "
            "panel it stands in for is being fetched"
        )
        failure = _block(body[body.index("catch") :], "{")
        assert re.search(rf"\b{notice}\.hidden\s*=\s*false", failure), (
            f"the failure branch writes into {notice} without showing it again; once any fetch has succeeded the "
            "notice is down for good, and every later failure is a blank pane with nothing to explain it"
        )

    def test_the_step_is_encoded_into_the_request_rather_than_pasted_into_it(self, page: str) -> None:
        """A step id is a person's text, and the page puts it into a query string.

        Pasted in raw, a step named with an `&` ends the parameter and the service is asked for the
        part before it; a `#` ends the URL and the service is asked for nothing at all; a `+` comes
        back out as a space. Each of those fetches the wrong panel or none, and the reader is shown
        either another step's rows or the notice saying the service is not answering.
        """
        _, params, body = _panel_fn(page)
        asked = _invocations(body, "fetch")[0]
        bare = re.sub(r"(?:encodeURI|encodeURIComponent)\s*\([^)]*\)", "", asked)
        bare = re.sub(r"new\s+URLSearchParams\s*\([^)]*\)", "", bare)

        assert re.search(r"\b" + params[0] + r"\b", asked), (
            f"the panel request does not mention {params[0]}, so every step asks for the same panel"
        )
        assert not re.search(r"\b" + params[0] + r"\b", bare), (
            f"{params[0]} is pasted into the URL unencoded ({asked.strip()}); a step id carrying &, # or + then "
            "asks for a different panel, or for none"
        )


# ---------------------------------------------------------------------------
# Containers: a few dozen titled boxes instead of six hundred and seventy-two
#
# The flow this page is really for draws 672 steps. Nobody follows that by reading it. What a reader
# can follow is a few dozen boxes with sentences on them -- "fetch the crm accounts and put
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
# reader being handed a number that reconciles with nothing.
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


def _grouped_spec() -> PipelineSpec:
    """The fixture flow with two of its stages boxed, read from the same file the page renders.

    Built by reading the shipped pipeline and adding the containers in memory rather than by editing
    the file on disk: every other test in this repository runs against that file, and a fixture that
    quietly grew a feature underneath them would be a change nobody reviewed.
    """
    import yaml

    from dantron_etl.pipeline import load_pipeline

    document = yaml.safe_load((PROJECT_ROOT / "pipelines" / "fixture.yaml").read_text(encoding="utf8"))
    document["groups"] = [{"id": group_id, "title": title} for group_id, title in GROUP_TITLES.items()]
    for group_id, members in GROUP_MEMBERS.items():
        for step in document["steps"]:
            if step["id"] in members:
                step["group"] = group_id
    return load_pipeline(document, rules={"revenue_type": "REVENUE"})


@pytest.fixture(scope="module")
def boxed(fixture_flow_sandbox: Path) -> tuple[PipelineSpec, PipelineTrace]:
    """The grouped pipeline and one run of it, which the counts on the boxes are read off."""
    spec = _grouped_spec()
    with chdir(fixture_flow_sandbox):
        return spec, run_pipeline(spec)


@pytest.fixture(scope="module")
def boxed_page(boxed: tuple[PipelineSpec, PipelineTrace]) -> str:
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


def _boundary(spec: PipelineSpec, group_id: str) -> tuple[tuple[PortRef, ...], tuple[PortRef, ...]]:
    """What entered a container and what left it, worked out here from the wiring alone.

    Derived in the test rather than taken from the graph's own answer to the same question: a page
    that faithfully renders whatever the library hands it and a library that was asked the wrong
    question agree with each other perfectly, and this is the one figure on the box a reader is
    invited to trust without opening it.
    """
    members = tuple(step.id for step in spec.steps if step.group == group_id)
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
        self, boxed_page: str, boxed: tuple[PipelineSpec, PipelineTrace]
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
        assert {group.id: group.title for group in spec.groups} == GROUP_TITLES
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
        self, boxed_page: str, boxed: tuple[PipelineSpec, PipelineTrace]
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
            group_id: sum(shape.rows for port, shape in trace.shapes.items() if spec.step(port.step).group == group_id)
            for group_id in GROUP_TITLES
        }
        honest = {
            group_id: sum(trace.shapes[port].rows for port in _boundary(spec, group_id)[1]) for group_id in GROUP_TITLES
        }
        assert naive != honest, "no container has any plumbing of its own, so this case proves nothing about the rule"


class TestAContainerOpensAndCloses:
    """Closed, it is one box with a sentence on it. Open, it is the six tools that do the work."""

    def test_each_container_offers_a_control_that_closes_it_and_opens_it_again(self, boxed_page: str) -> None:
        """Both states have to be reachable, which a control that only ever adds a class is not.

        A container that closes and cannot be opened is a section of the flow the reader has
        permanently lost, with the page's own control the thing that lost it for them.
        """
        for group_id, (_, body) in _containers(boxed_page).items():
            assert re.search(rf'data-collapse="{group_id}"', body), (
                f"{group_id} is drawn with no control that closes it"
            )

        script = _js(boxed_page)
        assert "closest('[data-collapse]')" in script, "nothing in the script answers the control"
        handler = _block(script[script.index("closest('[data-collapse]')") :], "{")
        assert re.search(r"classList\.toggle\(\s*'collapsed'\s*\)", handler), (
            "the control does not toggle the state, so a container closes once and never opens again"
        )

    def test_a_closed_container_draws_one_box_with_its_steps_put_away(self, boxed_page: str) -> None:
        """What closing means, said in the sheet rather than by moving anything on the canvas.

        The steps are children of the container, so closing is one class on one element and the
        drawing does the rest. A closed container that still drew its steps would be a box with a
        title over the six boxes it was meant to replace, which is the canvas this exists to escape.

        And it has to become one box rather than keep its full outline with nothing inside it: a
        container that keeps its expanded footprint when closed frees no space at all, and space is
        the only reason a reader closes one.
        """
        hides_steps, when_closed, when_open = [], [], []
        for selector, declarations in _css_rules(boxed_page):
            for one in selector.split(","):
                head = one.strip()
                if not head.startswith(".group"):
                    continue
                if ".collapsed" in head and head.endswith(".node") and "display:none" in declarations:
                    hides_steps.append(head)
                if ".collapsed" in head and "display:none" not in declarations:
                    when_closed.append(head)
                if ".collapsed" not in head and "display:none" in declarations:
                    when_open.append(head)

        assert hides_steps, "nothing in the sheet puts a closed container's steps away, so closing one changes nothing"
        assert when_closed, "the sheet says nothing about what a closed container draws instead of its steps"
        assert when_open, "whatever stands in for a closed container is drawn while it is open as well"

    def test_a_closed_container_still_says_what_it_is(self, boxed_page: str) -> None:
        """A box with nothing written on it is a section of the flow that has become unreadable.

        Two ways to lose the sentence, and both read as tidying. The sheet can hide the title along
        with everything else a closed container stops drawing; or the box a closed container becomes
        can be drawn somewhere the title is not, leaving the sentence floating over the space where
        the steps used to be.
        """
        import html

        containers = _containers(boxed_page)
        assert set(containers) == set(GROUP_TITLES), "no container is drawn, so no title can be lost"
        hidden_when_closed = [
            one.strip()
            for selector, declarations in _css_rules(boxed_page)
            for one in selector.split(",")
            if ".collapsed" in one and "display:none" in declarations
        ]

        for group_id, (_, body) in containers.items():
            title = html.escape(GROUP_TITLES[group_id], quote=True)
            found = re.search(rf"<text([^>]*)>{re.escape(title)}</text>", body)
            assert found is not None, f"{group_id}: no text element carries the title"
            worn = re.findall(r'class="([^"]*)"', found.group(1))
            at = (
                float(re.search(r'\bx="([\d.-]+)"', found.group(1)).group(1)),
                float(re.search(r'\by="([\d.-]+)"', found.group(1)).group(1)),
            )

            for name in " ".join(worn).split():
                for rule in hidden_when_closed:
                    assert f".{name}" not in rule, f"{group_id}: closing the container hides its own title ({rule})"

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
        self, boxed_page: str, boxed: tuple[PipelineSpec, PipelineTrace]
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
            (source.step, step.id, source.port, spec.step(source.step).group or "", step.group or "")
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

    def test_closing_a_container_puts_away_its_own_connections_and_re_points_the_rest(self, boxed_page: str) -> None:
        """The failure this prevents is a flow that reads as finishing at a box.

        A closed container stops drawing its steps, so every connection that ran into one of them now
        runs into the middle of nothing. Two different answers are owed. A connection with both ends
        inside belongs to the closed box and goes away with it; a connection with one end outside is
        how the rest of the flow reaches this section, and has to land on the box instead.

        Telling those apart needs both ends in one test. Ask about one end alone and every crossing
        connection is hidden too -- the section a reader closed takes its inputs and its outputs with
        it, and the flow appears to stop.

        The re-pointed curve goes onto the visible line and onto its wide transparent twin together.
        Move one and not the other and the target a hand aims at is no longer where the line is
        drawn, which lights a neighbouring connection instead -- the very defect the twin was added
        to fix.
        """
        script = _js(boxed_page)
        assert "closest('[data-collapse]')" in script, "nothing in the script answers the control"
        handler = _with_callees(script, _block(script[script.index("closest('[data-collapse]')") :], "{"))

        assert re.search(r"from-group|fromGroup", handler), "closing a container never asks where a connection starts"
        assert re.search(r"to-group|toGroup", handler), "closing a container never asks where a connection lands"
        assert re.search(
            r"(?:from-group|fromGroup)[^;]{0,200}(?:to-group|toGroup)"
            r"|(?:to-group|toGroup)[^;]{0,200}(?:from-group|fromGroup)",
            handler,
        ), (
            "the two ends of a connection are consulted in two separate places, so nothing here tells a connection "
            "inside the container from one that crosses its edge"
        )
        assert re.search(r"getBBox\(\)|getAttribute\('x'\)|\.x\.baseVal", handler), (
            "the handler never reads the container's own position, so a re-pointed connection lands wherever the "
            "hidden step used to be rather than on the box that replaced it"
        )
        curves = {value.strip() for value in re.findall(r"setAttribute\(\s*'d'\s*,\s*([^)]+)\)", handler)}
        assert curves, "no connection's curve is ever rewritten, so a crossing line still runs into a step nobody sees"
        assert len(curves) == 1, (
            f"the handler writes {len(curves)} different curves ({sorted(curves)}); the visible line and its wide "
            "transparent twin have to follow the same one, or the target a hand aims at is not on the line"
        )
        assert re.search(
            r"'\.wire'[^;]{0,120}'\.hit'|'\.hit'[^;]{0,120}'\.wire'|querySelectorAll\('path'\)", handler
        ), "only one of a connection's two paths is reached, so the line moves and its click target stays behind"


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


def _click_listener(code: str, at: int) -> tuple[int, int]:
    """Where the click listener containing ``at`` begins and ends.

    Two branches of one listener are settled by which is written first, because the first to answer
    returns. Two separate listeners are not: both run, whatever order they were registered in. So
    the ordering arguments below are only worth making inside one listener, and this is what says
    whether they are in one.
    """
    start = code.rindex("addEventListener('click'", 0, at)
    return start, start + len(_block(code[start:], "{"))


def _container_click(page: str) -> tuple[int, str, str]:
    """A click on a container itself, as (where it is answered, what picks it out, what it reaches).

    Found by `e.target.closest(...)` naming a container, because that is the idiom every other
    branch of this handler is already written in -- a step, a connection and the collapse control
    are each picked out of a click that way -- and a container has to be told apart from all three
    by the same means.

    The negated form is passed over: the canvas clause asks `!e.target.closest('.group')` to tell
    blank canvas from a container, and that is not a branch answering a container.
    """
    code = _js(page)
    for hit in re.finditer(r"e\.target\.closest\(\s*'(?:g\.group|\.group|\[data-group\])'\s*\)", code):
        if code[: hit.start()].rstrip().endswith("!"):
            continue
        body = _block(code[hit.start() :], "{")
        opened = code.index(body, hit.start())
        bound = max(code.rfind(mark, 0, hit.start()) for mark in (";", "{", "}"))
        return hit.start(), code[bound + 1 : opened], _with_callees(code, body)
    raise AssertionError(
        "nothing in the script answers a click on a container itself, so clicking into one cannot bring anything "
        "forward"
    )


def _lift_names(page: str, branch: str) -> set[str]:
    """Every name the lift works in: classes it sets, selectors it looks up, and the selectors
    behind whatever element it was handed at the top of the script."""
    code = _js(page)
    names = {"." + name for name in re.findall(r"classList\.(?:add|remove|toggle|contains)\(\s*'([\w-]+)'", branch)}
    names |= set(re.findall(r"querySelector(?:All)?\(\s*'([^']+)'\s*\)", branch))
    names |= {"#" + name for name in re.findall(r"getElementById\(\s*'([^']+)'\s*\)", branch)}
    for word in set(re.findall(r"\b[A-Za-z_]\w*\b", branch)):
        handed = re.search(
            rf"\b(?:const|let|var)\s+{word}\s*=\s*[\w.]*\.?querySelector(?:All)?\(\s*'([^']+)'\s*\)", code
        )
        if handed:
            names.add(handed.group(1))
    return names


def _drawn_at(svg: str, selector: str) -> list[int]:
    """Where every element the selector's last part names begins in the drawing."""
    last = re.split(r"[\s>+~,]+", selector.strip())[-1]
    places: list[int] = []
    for name in re.findall(r"\.([\w-]+)", last):
        places += [hit.start() for hit in re.finditer(rf'<[^>]*class="[^"]*\b{re.escape(name)}\b[^"]*"', svg)]
    for name in re.findall(r"#([\w-]+)", last):
        places += [hit.start() for hit in re.finditer(rf'<[^>]*id="{re.escape(name)}"', svg)]
    return places


def _title_classes(page: str) -> set[str]:
    """What a container's title wears as the page ships it -- the state a resting title is in."""
    import html

    worn: set[str] = set()
    for group_id, (_, body) in _containers(page).items():
        title = html.escape(GROUP_TITLES[group_id], quote=True)
        found = re.search(rf"<text([^>]*)>{re.escape(title)}</text>", body)
        assert found is not None, f"{group_id}: no text element carries the title"
        worn |= set(" ".join(re.findall(r'class="([^"]*)"', found.group(1))).split())
    assert worn, "no container's title is drawn at all"
    return worn


def _lift_style(page: str, branch: str, resting: set[str]) -> str:
    """What the sheet draws for a lifted title that it does not draw for a resting one.

    Only the names the lift *puts a title into* count: the state it sets on one, and whatever it
    draws into that is painted after the lane names. A name it merely reads on the way past -- the
    row counts it inserts a title back above, say -- describes a resting container and has nothing
    to say about a lifted title.

    A rule whose names are all worn at rest is what a title looks like anyway, so it is not the
    lift saying anything -- counting it would let a page that styles every title identically claim
    to have brought one forward.
    """
    svg = _svg(page)
    last_name = svg.rindex('<text class="lane"')
    wanted = set(re.findall(r"classList\.(?:add|toggle)\(\s*'([\w-]+)'", branch))
    for selector in _lift_names(page, branch):
        if any(place > last_name for place in _drawn_at(svg, selector)):
            wanted |= set(re.findall(r"[.#]([\w-]+)", selector))
    said = []
    for selector, declarations in _css_rules(page):
        for one in selector.split(","):
            worn = set(re.findall(r"[.#]([\w-]+)", one))
            if _reaches_a_title(one) and worn & wanted and not worn <= resting:
                said.append(declarations)
    return " ".join(said)


def _reaches_a_title(selector: str) -> bool:
    """Whether a rule could be about a drawn title at all.

    A class name is not owned by the canvas -- the page already styles `a.up` in the prose beside
    it -- so a rule naming a tag the drawing does not write a title as is somebody else's.
    """
    last = re.split(r"[\s>+~]+", selector.strip())[-1]
    tag = re.match(r"^([a-zA-Z][\w-]*)", last)
    return not tag or tag.group(1) in {"text", "tspan", "g", "svg", "rect"}


def _resting_style(page: str, resting: set[str]) -> str:
    """Everything the sheet says about a title that has not been clicked into."""
    return " ".join(
        declarations
        for selector, declarations in _css_rules(page)
        for one in selector.split(",")
        if _reaches_a_title(one)
        and re.findall(r"[.#]([\w-]+)", one)
        and set(re.findall(r"[.#]([\w-]+)", one)) <= resting
    )


def _remembered(code: str, branch: str) -> str | None:
    """The variable the page keeps the lifted container in, if it keeps one.

    Read off the script rather than named here, the way the traced step is: which container is
    forward is a fact about what the reader last asked for, and a class on an element answers a
    different question.
    """
    traced = set(re.findall(r"(\w+)\s*=", _block(code, "function traceFrom(")))
    for name in re.findall(r"let\s+(\w+)\s*=\s*null", code):
        if name not in traced and re.search(rf"\b{name}\b\s*=[^=]", branch):
            return name
    return None


def _aliases(branch: str, name: str) -> set[str]:
    """``name`` and whatever the branch copied it into.

    The state has to be read before it is overwritten, so a branch that keeps the old value in a
    local and compares that is doing the same thing by another name.
    """
    found = {name}
    for other in re.findall(rf"\b(?:const|let|var)\s+(\w+)\s*=\s*{name}\s*[;\n]", branch):
        found.add(other)
    return found


def _puts_back(branch: str, added: set[str], remembered: str | None) -> bool:
    """Whether a piece of script takes a lifted title back down again."""
    if any(re.search(rf"classList\.(?:remove|toggle)\(\s*'{name}'", branch) for name in added):
        return True
    if re.search(r"replaceChildren\(\s*\)|innerHTML\s*=\s*''|\.remove\(\)|removeChild\(", branch):
        return True
    return bool(remembered and re.search(rf"\b{remembered}\b", branch))


def _blank_canvas(page: str) -> tuple[str, str]:
    """The clause that answers a click on the canvas between the steps, as (its test, its body)."""
    code = _js(page)
    found = re.search(r"if\s*\(([^{]*closest\('\.canvas'\)[^{]*)\)\s*\{", code)
    assert found is not None, "nothing answers a click on the canvas itself, which is where a reader's hand goes first"
    return found.group(1), _with_callees(code, _block(code[found.end() - 1 :], "{"))


class TestClickingIntoAContainerBringsItsTitleForward:
    """The title is the feature, and on the flow this page is for it is drawn under a lane's name."""

    def test_a_lifted_title_ends_up_drawn_after_every_lane_name(self, boxed_page: str) -> None:
        """The whole point, and the one an implementation can look like it has and not have.

        SVG paints in document order. The lane names are written after every container and every
        step on purpose -- that ordering is what keeps them legible where a container's header
        reaches the first column -- so a class that puts a halo behind a title sitting where it
        sits changes nothing about which of the two the reader can read. The markup of that page
        and the markup of a page that works are all but identical, and only this says which is
        which: whatever the click produces has to END UP after the lane names, by being moved
        there, or by being drawn into something that already is.
        """
        svg = _svg(boxed_page)
        last_name = svg.rindex('<text class="lane"')

        assert last_name > svg.rindex('<g class="group"'), (
            "the lane names are no longer painted after the containers, so the fact this case exists for has been "
            "changed rather than answered -- and the name that was kept legible on purpose is now the one underneath"
        )

        at, head, branch = _container_click(boxed_page)
        moved = re.findall(r"([\w.$()'\[\]-]+)\.(?:appendChild|append|prepend|insertBefore|replaceChildren)\(", branch)
        to_the_root = [where for where in moved if re.search(r"\b(svg|flow|canvas|root|drawing)\b", where)]
        after = sorted(
            name for name in _lift_names(boxed_page, branch) if any(place > last_name for place in _drawn_at(svg, name))
        )

        assert to_the_root or after, (
            "clicking a container reaches nothing that is painted after the lane names: it moves nothing to the root "
            f"of the drawing and every name it works in ({sorted(_lift_names(boxed_page, branch))}) belongs to an "
            "element drawn before them. A halo on a title left where it is is drawn under the lane name exactly as "
            "the title was, so the reader sees the same collision with a heavier smudge behind it"
        )

    def test_the_backing_covers_the_whole_title_rather_than_the_box_it_belongs_to(self, boxed_page: str) -> None:
        """A one-column container is 200px wide and its title is routinely twice that.

        So the two ways to get this wrong are a backing drawn at the width of the box, which leaves
        the back half of the sentence over whatever it was already over, and anything that clips
        the title to the box, which cuts the sentence off instead of lifting it. What is left is a
        backing that comes from the text: the glyphs' own halo, as the lane names already earn
        theirs, or a rectangle measured off the text in the browser -- the only place the width of
        a run of text is known.
        """
        at, head, branch = _container_click(boxed_page)
        resting = _title_classes(boxed_page)
        style = _lift_style(boxed_page, branch, resting)

        haloed = "paint-order" in style and "stroke" in style
        widths = re.findall(r"setAttribute\(\s*'width'[^;]*|\.width\s*=[^;]*", branch)
        for said in widths:
            assert re.search(r"getBBox|getComputedTextLength|getExtentOfChar|getBoundingClientRect", said), (
                f"the backing is sized by {said.strip()!r}, which does not measure the title. A width the page "
                "worked out from anything but the text ends where the text does not"
            )
        assert haloed or widths, (
            f"nothing draws a backing over the whole title: the sheet says {style.strip()!r} for a lifted one, which "
            "carries no halo of its own, and the script measures no text to size one"
        )
        assert not re.search(r"(gbox|gpill)[^;]{0,160}width|width[^;]{0,160}(gbox|gpill)", branch), (
            "the backing takes its width from the container's own rectangle, so it stops at the right edge of the box "
            "and the rest of the title is left exactly where it was"
        )
        assert "clip" not in branch and "clip-path" not in _svg(boxed_page) and "clipPath" not in boxed_page, (
            "something clips the lifted title, which is the same defect the other way round: the part of the sentence "
            "that runs past the box is cut off rather than brought forward"
        )

    def test_a_title_at_rest_carries_no_backing(self, boxed_page: str) -> None:
        """A page where every title wears the halo has a hundred and twenty-nine lifted titles.

        Which is a page that says nothing: the reader clicks a container and cannot tell that
        anything answered. The lift is only a lift if a title that has not been clicked into looks
        different from one that has -- so nothing arrives already lifted, and the sheet has
        something to say about the lifted state that it does not say about the resting one.
        """
        at, head, branch = _container_click(boxed_page)
        resting = _title_classes(boxed_page)
        added = set(re.findall(r"classList\.(?:add|toggle)\(\s*'([\w-]+)'", branch))
        lifted = _lift_style(boxed_page, branch, resting)

        assert not (added & resting), (
            f"a container's title is drawn already wearing {sorted(added & resting)}, so every one of them is lifted "
            "before anything is clicked"
        )
        said_at_rest = _resting_style(boxed_page, resting)
        assert "paint-order" not in said_at_rest and "stroke" not in said_at_rest, (
            f"a resting title already carries the backing ({said_at_rest.strip()!r}), so clicking into a container "
            "changes nothing a reader can see"
        )
        assert lifted.strip(), (
            "the sheet says nothing about a lifted title that it does not say about every other one, so the two are "
            "drawn identically"
        )

    def test_lifting_one_container_puts_any_other_back(self, boxed_page: str) -> None:
        """Two titles forward is the collision this exists to settle, drawn twice.

        A lift that only adds to the container clicked leaves the last one where it was, and after
        a few clicks the canvas carries a row of haloed sentences over each other with nothing
        saying which the reader asked for. So the click has to put back whatever is forward before
        it brings anything else forward -- either by sweeping the state off the drawing, or by
        holding the one that is up and taking it down.
        """
        code = _js(boxed_page)
        at, head, branch = _container_click(boxed_page)
        added = set(re.findall(r"classList\.(?:add|toggle)\(\s*'([\w-]+)'", branch))
        remembered = _remembered(code, branch)

        swept = [
            selector
            for selector in re.findall(r"querySelectorAll\(\s*'([^']+)'\s*\)", branch)
            if set(re.findall(r"[.#]([\w-]+)", selector)) & added
        ]
        emptied = re.search(r"replaceChildren\(\s*\)|innerHTML\s*=\s*''|removeChild\(|\.remove\(\)", branch)
        held = remembered is not None and bool(re.search(rf"\b{remembered}\b(?!\s*=[^=])", branch))

        assert swept or emptied or held, (
            "clicking a container never reaches the one already forward: nothing sweeps the lifted state off the "
            "drawing, nothing empties whatever the lifted title is drawn into, and nothing is remembered from the "
            "last click -- so a second click lifts a second title and leaves the first one up"
        )

    def test_clicking_the_lifted_container_again_or_the_canvas_puts_it_back(self, boxed_page: str) -> None:
        """The way back, which has to be the one already on this canvas.

        A reader who has finished with a title puts it away by clicking it again or by clicking the
        canvas beside it -- the same two gestures that already put a traced chain away, and the
        hand goes to them before it goes looking for a control. The canvas clause is the one that
        can quietly fail: it is written today behind a test for a lit chain, and a title lifted
        over nothing else would be left up by a clause that returns before it looks.
        """
        code = _js(boxed_page)
        at, head, branch = _container_click(boxed_page)
        added = set(re.findall(r"classList\.(?:add|toggle)\(\s*'([\w-]+)'", branch))
        remembered = _remembered(code, branch)

        compared = "|".join(sorted(_aliases(branch, remembered))) if remembered else ""
        asked = re.search(r"classList\.(?:contains|toggle)\(", branch) or (
            compared and re.search(rf"\b(?:{compared})\b\s*[!=]==|[!=]==\s*(?:{compared})\b", branch)
        )
        assert asked, (
            "the branch never asks whether the container clicked is the one already forward, so clicking it a second "
            "time lifts it again and the reader has no way back through the thing they clicked"
        )

        test, body = _blank_canvas(boxed_page)
        assert _puts_back(body, added, remembered), (
            "a click on blank canvas does not take a lifted title back down, so the one gesture a reader makes "
            "without looking leaves the halo on the canvas"
        )
        blocking = [
            one.strip()
            for one in test.split("&&")
            if re.fullmatch(r"[(\s]*\w+\s*!==\s*null[)\s]*", one)
            and not any(name in one for name in {remembered} | added if name)
        ]
        assert not blocking, (
            f"the canvas clause is reached only when {blocking} says something else is lit, so a title lifted on its "
            "own is not put back by clicking beside it"
        )

    def test_a_lifted_title_is_drawn_where_the_resting_one_was(self, boxed_page: str) -> None:
        """A title that jumps as it comes forward is a defect a reader sees every time.

        Two ways for it to move. A title drawn out of its container and into something else lands
        somewhere else the moment anything between it and the root carries a transform, so nothing
        may. And the lift may not restate where or how the text is drawn: a place or a size written
        a second time is the geometry of this page said in two languages, and the two drift.
        """
        svg = _svg(boxed_page)
        at, head, branch = _container_click(boxed_page)
        resting = _title_classes(boxed_page)

        for attrs, _ in _elements(svg, '<g class="group"'):
            assert "transform" not in attrs, (
                f"a container is drawn under a transform ({attrs.get('transform')}), so a title taken out of it is "
                "drawn somewhere other than where it was"
            )
        for name in _lift_names(boxed_page, branch):
            for place in _drawn_at(svg, name):
                tag = svg[place : svg.index(">", place)]
                assert "transform" not in tag, f"a lifted title is drawn into {tag!r}, which moves whatever is in it"

        for said in re.findall(r"setAttribute\(\s*'(?:x|y|dx|dy|transform)'[^;]*", branch):
            assert re.search(r"getAttribute\(|getBBox", said), (
                f"the lift writes the title's place itself ({said.strip()!r}) rather than keeping the one it was "
                "drawn at, so the sentence moves as it comes forward"
            )
        style = _lift_style(boxed_page, branch, resting)
        for property_ in ("font:", "font-size", "font-weight", "letter-spacing", "transform"):
            assert property_ not in style, (
                f"a lifted title is drawn with a different {property_.rstrip(':')} ({style.strip()!r}), so bringing "
                "it forward re-sets the text and it moves under the reader"
            )

    def test_clicking_a_container_does_not_take_a_click_meant_for_a_step_or_the_control(self, boxed_page: str) -> None:
        """A step is drawn INSIDE its container, and so is the control that closes it.

        `closest('.group')` is therefore true of a click on either of them. A branch that answers a
        container and is reached before the step's is a canvas where no step can be chosen any
        more, and one reached before the collapse control's is a container that lifts its title
        every time a reader closes it. Either is settled by being written after the branch that
        returns, or by saying out loud that it is not those.
        """
        code = _js(boxed_page)
        at, head, branch = _container_click(boxed_page)
        flat = head.replace(" ", "")
        assert "addEventListener('click'" in code[:at], (
            "a container is answered outside any click listener, so nothing about it can be settled by what a click "
            "reaches first"
        )
        start, end = _click_listener(code, at)

        for what, selector in (("a step", ".node"), ("the collapse control", "[data-collapse]")):
            assert re.search(rf"closest\(\s*'{re.escape(selector)}'\s*\)", code), (
                f"nothing picks {what} out of a click any more"
            )
            # Looked for inside the listener the container is answered in, because that is the only
            # place being written first settles anything: a branch in another listener runs whether
            # this one returned or not.
            hit = re.search(rf"closest\(\s*'{re.escape(selector)}'\s*\)", code[start:end])
            says_not = re.search(rf"!\s*[\w.]*closest\(\s*'({re.escape(selector)}|\.gctl)'\s*\)", flat)
            ordered = bool(hit and start + hit.start() < at and "return" in _block(code[start + hit.start() :], "{"))
            assert ordered or says_not, (
                f"a click on {what} is inside a container, so it reaches the container branch as well: that branch is "
                f"written before the one answering {what} (or in another listener, where returning settles nothing) "
                f"and does not exclude it. Clicking {what} would lift a title instead of, or as well as, doing what "
                "it has always done"
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


def _dense_flow(*, boxed: bool) -> PipelineSpec:
    """A flow with a real one's shape and no client's vocabulary, optionally boxed into containers.

    Built as a spec rather than as a document, because the loader is not what is under test: the
    layout is handed a :class:`PipelineSpec` and reads its wiring, its lanes and its groups. The
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

    steps: list[StepSpec] = []
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
        steps.append(StepSpec(id=step_id, tool=tool_of[step_id], config={}, inputs=inputs, lane=lane, stage=stage))

    groups: tuple[Group, ...] = ()
    if boxed:
        runs = _runs(steps)
        owner = {step_id: f"box_{n:03d}" for n, run in enumerate(runs) for step_id in run}
        groups = tuple(
            Group(id=f"box_{n:03d}", title=f"Step {n}: what {len(run)} tools do between them")
            for n, run in enumerate(runs)
        )
        steps = [
            StepSpec(
                id=step.id,
                tool=step.tool,
                config=step.config,
                inputs=step.inputs,
                lane=step.lane,
                stage=step.stage,
                group=owner[step.id],
            )
            for step in steps
        ]
    lanes = tuple(Lane(id=lane, label=f"Lane {lane}", order=n) for n, (lane, *_) in enumerate(LANE_SHAPE))
    return PipelineSpec(name="dense", steps=tuple(steps), lanes=lanes, groups=groups, registry=REGISTRY)


def _runs(steps: list[StepSpec]) -> list[list[str]]:
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


def _no_run(spec: PipelineSpec) -> PipelineTrace:
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
def dense_boxed() -> tuple[PipelineSpec, str]:
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


def _columns(spec: PipelineSpec) -> dict[str, int]:
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
        self, dense_boxed: tuple[PipelineSpec, str]
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
        assert len(spec.groups) == 150, f"boxing runs of {RUN_LENGTH} now yields {len(spec.groups)} containers"
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
        self, dense_boxed: tuple[PipelineSpec, str]
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

        assert set(containers) == {group.id for group in spec.groups}, "not every declared container is drawn"
        assert len(boxes) == len(spec.steps), f"{len(boxes)} steps are drawn, not {len(spec.steps)}"

        missing: dict[str, list[str]] = {}
        swallowed: dict[str, list[str]] = {}
        for group_id, (_, body) in containers.items():
            held = set(spec.group_members(group_id))
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
        self, dense_boxed: tuple[PipelineSpec, str]
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

        assert set(containers) == {group.id for group in spec.groups}, "not every declared container is drawn"
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
        self, dense_boxed: tuple[PipelineSpec, str]
    ) -> None:
        """A step belongs to at most one container, so two outlines crossing is a drawing of a lie.

        Distinct from the stranger case and not implied by it. Two boxes can cross in space that
        holds no step at all -- one reaching up for its header through the width of another -- and
        the reader is then shown two captions over one region with nothing saying which of them owns
        it. Today 1,089 of the 11,175 pairs cross.
        """
        spec, canvas = dense_boxed
        containers = _containers(canvas)

        assert set(containers) == {group.id for group in spec.groups}, "not every declared container is drawn"
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
        self, dense_unboxed: str, dense_boxed: tuple[PipelineSpec, str]
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
