"""The HTML of a run's page: the tables, each step's settings and data, and the page around them.

The page itself is a template file with the stylesheet and the script beside it, in ``templates/``
and ``static/``. Python only builds the fragments that carry data and puts them into the template.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import polars as pl

from ..focus import FocusSpec
from ..spec import Pipeline
from ..status import STATUS_LEVELS
from ..trace import PipelineTrace
from .assets import escape as _e
from .assets import page
from .layout import canvas

# Rows drawn into the page for a slice of one entity, which the service cannot serve.
MAX_ROWS = 25

# Rows fetched when a reader opens a port. Bigger than MAX_ROWS because it costs the page nothing:
# it arrives only for the port actually being looked at.
WINDOW = 100

# The only view a panel carries. One entity's rows are asked of the service instead.
EVERYONE = {"all": None}

def _table(frame: pl.DataFrame, port: str | None = None, marked: bool = False) -> str:
    """One port's rows, with a box under each column heading that narrows to matching rows.

    The search is per column rather than one box over the whole table, because the question a
    reader actually arrives with is about a column -- which rows carry this account code, which
    carry this status -- and a single box would match that text anywhere in the row and answer a
    different question. Filtering is over the rows drawn, which the note beneath already says is a
    window onto a larger frame: this narrows what is on the page, it does not re-run the step.

    ``marked`` draws the status column: one cell per row, first, saying what the step that made
    these rows said about that row. First because that is where an eye picks a row up and because it
    is what the reader asked for, and set apart in the markup rather than by its name -- a column
    headed ``level`` sitting among a client's own columns reads as something the step computed, gets
    searched with the box under it, and travels into whatever the reader copies out. It carries no
    dtype, which every real column's header does, and it is drawn only for a port some step actually
    marked: almost every port in a flow degraded nothing, and a column of ticks on all of them is a
    column a reader stops looking at.

    Its control is a choice rather than a box, because three words are a vocabulary and not a
    search: typed, ``warn`` would narrow to warnings on the page and be refused by the service. The
    options come from :data:`STATUS_LEVELS` itself, so a level added to the library cannot end up
    with no way to filter to it.
    """
    if not frame.columns:
        return '<p class="more">no columns</p>'

    levels = "".join(f'<option value="{_e(v)}">{_e(v or "any level")}</option>' for v in ("", *STATUS_LEVELS))
    head = ('<th class="lvl">status</th>' if marked else "") + "".join(
        f"<th>{_e(c)}<em>{_e(frame.schema[c])}</em></th>" for c in frame.columns
    )
    # Every box says which cell of a row it narrows and the script reads that number straight off
    # it, so inserting a column at the front moves all of them along: left alone, each box would
    # narrow the column to its left and the reader would watch rows vanish for no reason they can
    # see.
    first = 1 if marked else 0
    boxes = (
        f'<th class="lvl"><select class="lvl" aria-label="narrow to a level">{levels}</select></th>' if marked else ""
    ) + "".join(
        f'<th><input class="flt" data-c="{i + first}" data-col="{_e(c)}" type="search" placeholder="{_e(c)}" '
        f'aria-label="filter {_e(c)}"></th>'
        for i, c in enumerate(frame.columns)
    )
    # A port the service can serve carries no rows in the page at all: the header and the search
    # boxes are the whole of it until a reader opens it, and then a window is fetched. The page has
    # to stay the same size as the flow grows, and a flow grows by steps -- every table embedded
    # here is multiplied by however many steps the pipeline ends up having.
    #
    # A slice of one entity is not a port the service knows, so those still carry their rows.
    lazy = port is not None
    shown = frame.head(0) if lazy else frame.head(MAX_ROWS)
    body = "".join(
        "<tr>" + "".join(f"<td>{'<span class=nul>null</span>' if v is None else _e(v)}</td>" for v in row) + "</tr>"
        for row in shown.iter_rows()
    )
    drawn = 0 if lazy else min(frame.height, MAX_ROWS)
    more = (
        ""
        if lazy
        else (f'<p class="more">showing {drawn:,} of {frame.height:,} rows</p>' if frame.height > MAX_ROWS else "")
    )
    # A port names itself so the page can ask the service for another window of it. The rows written
    # here are the first paint and the fallback: opened as a file with no service answering, the page
    # still shows what it always showed.
    names = f' data-port="{_e(port)}" data-offset="0" data-limit="{WINDOW}" data-lazy="1"' if port else ""
    pager = (
        '<p class="more pager" hidden><button data-page="prev">previous</button>'
        '<button data-page="next">next</button><span class="at"></span></p>'
        if port
        else ""
    )
    return (
        f'<div class="tbl"><div class="scroll"><table class="data" data-drawn="{drawn}"{names}>'
        f'<thead><tr>{head}</tr><tr class="flts">{boxes}</tr></thead>'
        f'<tbody>{body}</tbody></table></div>{more}<p class="more match" hidden></p>{pager}</div>'
    )


def _scalar(value: Any) -> str:
    """One configured value, drawn so that what it *is* survives being read.

    An empty string and an absent value are one character apart in JSON and look identical on a
    page, so each says its own name; a boolean stays lower case, the way a definition file spells
    it, rather than becoming Python's True.
    """
    if value is None:
        return '<span class="nul">not set</span>'
    if isinstance(value, bool):
        return f"<code>{'true' if value else 'false'}</code>"
    if value == "":
        return '<span class="nul">empty</span>'
    return f"<code>{_e(value)}</code>"


def _value(value: Any) -> str:
    """A configured value at any shape: a list of like objects becomes the table it already is."""
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        columns: list[str] = []
        for item in value:
            columns.extend(key for key in item if key not in columns)
        head = "".join(f"<th>{_e(key)}</th>" for key in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{_value(item.get(key))}</td>" for key in columns) + "</tr>" for item in value
        )
        return (
            f'<div class="scroll"><table class="cfgtable"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>"
        )
    if isinstance(value, list):
        return " ".join(_scalar(item) for item in value) if value else '<span class="nul">none</span>'
    if isinstance(value, dict):
        return "".join(
            f'<div class="cfg"><span class="ck">{_e(key)}</span><span>{_value(item)}</span></div>'
            for key, item in value.items()
        )
    return _scalar(value)


def _config(config: Mapping[str, Any]) -> str:
    """A step's configuration as the fields it sets, rather than as the JSON it is stored in.

    The JSON was honest and unreadable: a reader after "which column becomes which name" had to
    parse punctuation to find two strings. Drawn from the value's own shape rather than from the
    tool's template, so a tool this renderer has never heard of still draws, and a field the
    template forgot is still shown rather than silently dropped.
    """
    if not config:
        return '<p class="more">nothing to configure</p>'
    return "".join(
        f'<div class="cfg"><span class="ck">{_e(key)}</span><span>{_value(value)}</span></div>'
        for key, value in config.items()
    )


def _event_if_ran(trace: PipelineTrace, step_id: str) -> Any:
    """The step's event, or None when the run stopped before reaching it.

    A run stops at the first step that fails, and that is when its page is most wanted: the rows
    that went into the failing step are the first thing to look at. So every part of the page draws a
    step that never ran, as one that did not run, rather than refusing to draw the run at all.
    """
    return trace.event(step_id) if any(event.step_id == step_id for event in trace.events) else None


def _index(spec: Pipeline, trace: PipelineTrace) -> str:
    """Every step as a line, grouped by lane, in the order the run took them.

    Fifty-three nodes is more than a canvas can be scanned for, and a reader looking for one by name
    should not have to pan a drawing to find it. Ordered by the run rather than by the file, because
    that is the order a reader is thinking in when they are following a figure.
    """
    lanes: dict[str, list[str]] = {}
    for step_id in spec.order():
        step = spec.step(step_id)
        event = _event_if_ran(trace, step_id)
        mark = ' class="notrun"' if event is None else ' class="flag"' if event.level == "warning" else ""
        lanes.setdefault(step.lane or "", []).append(
            f'<li><button data-goto="{_e(step_id)}"{mark}>'
            f"<span>{_e(step_id)}</span><em>{_e(step.tool)}</em></button></li>"
        )

    labels = {lane.id: lane.title for lane in spec.lanes}
    groups = "".join(
        f'<li class="lanehead">{_e(labels.get(lane, lane) or "elsewhere")}</li>{"".join(items)}'
        for lane, items in lanes.items()
    )
    return (
        '<nav class="index" aria-label="every step"><header><b>steps</b>'
        f"<span>{sum(len(items) for items in lanes.values())}</span></header>"
        f"<ul>{groups}</ul></nav>"
    )


def _step_detail(spec: Pipeline, trace: PipelineTrace, step_id: str) -> str:
    """What this step is: why it exists, how it is configured, what feeds it, what it cost."""
    step = spec.step(step_id)
    event = _event_if_ran(trace, step_id)
    notes = (
        "<ul class=notes>" + "".join(f"<li>{_e(note)}</li>" for note in event.notes) + "</ul>"
        if event is not None and event.notes
        else ""
    )
    level = event.level if event is not None else "notrun"
    cost = f"{event.duration_ms:.1f} ms" if event is not None else "did not run: the run stopped before this step"
    feeds = "".join(
        f"<li><code>{_e(name)}</code> &larr; <code>{_e(ref)}</code></li>" for name, ref in step.inputs.items()
    )
    return (
        f'<section class="detail {level}" data-step="{_e(step_id)}" hidden>'
        f'<header><h3><code>{_e(step_id)}</code><span class="tool">{_e(step.tool)}</span></h3>'
        f'<p class="where">{_e(step.lane or "")} &middot; {_e(step.stage or "")} &middot; '
        f"{cost}</p>"
        f"{f'<p class=why>{_e(step.why)}</p>' if step.why else ''}{notes}</header>"
        f"<h5>configuration, as written</h5>{_config(step.config)}"
        f"{f'<h5>fed by</h5><ul class=feeds>{feeds}</ul>' if feeds else ''}"
        f"</section>"
    )


def _step_data(
    spec: Pipeline,
    trace: PipelineTrace,
    step_id: str,
    views: dict[str, Any],
) -> str:
    step = spec.step(step_id)
    event = _event_if_ran(trace, step_id)
    # What left, and what arrived. A step's inputs are upstream output ports, so both sides are
    # frames this run already produced and the service already serves by name -- which is what makes
    # searching and paging work on an input exactly as it does on an output.
    #
    # Reading a step used to mean holding its output in your head and scrolling to the step before it
    # to see what went in. The comparison a reader actually wants -- what did this step DO -- is
    # between two frames that were never on the screen together.
    leaving = [(shape.port, shape.rows, _e(shape.port.port)) for shape in event.outputs] if event is not None else []
    arriving = [
        (ref, trace.shapes[ref].rows if ref in trace.shapes else None, f"{_e(name)} &larr; {_e(ref.port)}")
        for name, ref in step.inputs.items()
    ]

    sides = []
    for side, entries in (("out", leaving), ("in", arriving)):
        if not entries:
            continue
        ports = []
        for ref, rows, label in entries:
            panes = []
            for key, view in views.items():
                if ref not in trace.shapes:
                    # An input whose step never ran: the run stopped before it.
                    body = '<p class="more">not produced &mdash; the step that makes it did not run</p>'
                    count = "&mdash;"
                else:
                    frame = trace.frames.get(ref)
                    body = (
                        # Keyed off the port rather than off the step being read, so a marked port
                        # drawn as the next step's input carries the same marks it carries as an
                        # output: it is the same port and the same rows.
                        _table(frame, port=f"{ref.step}.{ref.port}", marked=ref in trace.status)
                        if frame is not None
                        else '<p class="more">released &mdash; not retained</p>'
                    )
                    count = "&mdash;" if rows is None else f"{rows:,} rows"
                # Shut by default, and `details` rather than a script: a closed one is not laid out,
                # so a page of wide tables costs the browser almost nothing until a port is opened,
                # and what a reader came to see -- which port carries how many rows -- stays readable
                # without scrolling past the rows themselves.
                panes.append(
                    f'<div class="view" data-for="{_e(key)}" hidden><details>'
                    f'<summary><span class="pill">{label}</span><span>{count}</span></summary>'
                    f"{body}</details></div>"
                )
            ports.append(f'<div class="port">{"".join(panes)}</div>')
        sides.append(f'<div class="side" data-side="{side}"{"" if side == "out" else " hidden"}>{"".join(ports)}</div>')

    # No toggle on a source: there is no "in" to flip to, and a control that does nothing is worse
    # than none.
    flip = (
        '<div class="flip"><button data-side="out" aria-pressed="true">out</button>'
        '<button data-side="in" aria-pressed="false">in</button></div>'
        if arriving
        else ""
    )
    return f'<section class="stepdata" data-step="{_e(step_id)}" hidden>{flip}{"".join(sides)}</section>'


def build_panels(spec: Pipeline, trace: PipelineTrace, focus: FocusSpec | None) -> dict[str, str]:
    """Every step's data panel, keyed by the step, ready to be written down and fetched one at a time.

    The markup is exactly what the page used to carry: same tables, same headers, same per-column
    search boxes, same in/out flip, same bodies left for the service. What changed is *when* it is
    built -- on the real flow, 672 of these came to 7.76 MB of an 8.98 MB page, with 24,937 form
    controls the browser laid out before it could paint, for the 671 steps nobody had opened.

    Keyed off ``spec.order()`` rather than off the trace's events, so every step a reader can pick
    -- a node on the canvas, a line in the step list -- has a panel behind it. A step the flow can
    select and the service cannot answer for is a dead end found by clicking.

    Only the everyone view is drawn. One entity's rows are fetched from the service when asked for.
    """
    return {step_id: _step_data(spec, trace, step_id, EVERYONE) for step_id in spec.order()}


def _page_data(spec: Pipeline, focus: FocusSpec | None, run: Mapping[str, Any] | None) -> str:
    """The page's facts as JSON for its script, safe inside a script element."""
    data = {"pipeline": spec.name, "focus": focus.label if focus is not None else None, **(run or {})}
    text = json.dumps(data, default=str).replace("</", "<\\/")
    return f'<script type="application/json" id="page-data">{text}</script>'


def _banner(run: Mapping[str, Any] | None) -> str:
    """What a reader must know about this run before reading it: a preview, or where it stopped."""
    if not run:
        return ""
    lines = []
    if run.get("kind") == "preview":
        held = list(run.get("held_back") or ())
        text = "This was a preview. Nothing was written."
        if held:
            text += f" Held back: {', '.join(held)}"
        lines.append(f'<p class="banner preview">{_e(text)}</p>')
    if run.get("outcome") == "failed":
        text = f"The run stopped at step {run.get('failed_step')}: {run.get('message')}"
        lines.append(f'<p class="banner failed">{_e(text)}</p>')
    return "".join(lines)


def build(
    spec: Pipeline,
    trace: PipelineTrace,
    focus: FocusSpec | None,
    run: Mapping[str, Any] | None = None,
) -> str:
    """The run as a page: the flow, every step's configuration, and the data of the step being read.

    The panels are no longer in here -- they are written beside the run's ports and fetched as they
    are opened, which is what :func:`build_panels` is for. One exception, and it is the reason the
    page opens on anything at all: the step named by ``data-first`` is chosen before any fetch can
    have answered, so its panel ships in the document. Every other panel in here would be weight for
    a step nobody has clicked.

    What stays is everything a reader has without a service answering: the canvas, the step list and
    each step's configuration as it was authored. Opened as a plain file the page still says what
    every step does; it is the rows, and now the tables around them, that need something serving.

    ``run`` is the saved run's record. It goes into the page data and decides the banner.
    """
    total_in = sum(s.rows for r, s in trace.shapes.items() if not spec.step(r.step).inputs)
    published = [r for r in trace.shapes if not spec.consumers(r)]
    order = spec.order()
    details = "".join(_step_detail(spec, trace, step_id) for step_id in order)
    # One name for the step the page opens on and the one panel it ships, because they have to be
    # the same step: named without shipping, the page opens on an empty pane until a fetch answers,
    # and shipped without naming, it carries a panel nothing ever shows.
    opening = order[0] if order else ""
    panel = _step_data(spec, trace, opening, EVERYONE) if opening else ""
    summary = (
        f"<b>{len(trace.events)}</b> steps &middot; <b>{len(trace.shapes)}</b> ports &middot; "
        f"<b>{total_in}</b> rows in &middot; <b>{len(published)}</b> ports nothing reads further"
    )
    picker = (
        f'<div class="picker"><label><span class="plabel">{_e(focus.label)}</span>'
        f'<input type="search" class="pick" placeholder="find a {_e(focus.label)}" '
        f'aria-label="follow one {_e(focus.label)}"></label></div>'
        if focus is not None
        else ""
    )
    return page(
        "walkthrough",
        scripts=("runbar", "walkthrough"),
        name=_e(spec.name),
        summary=summary,
        canvas=canvas(spec, trace),
        picker=picker,
        banner=_banner(run),
        pagedata=_page_data(spec, focus, run),
        details=details,
        panel=panel,
        first=_e(opening),
        index=_index(spec, trace),
    )
