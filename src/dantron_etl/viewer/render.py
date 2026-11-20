"""The HTML of a run's page: the tables, each step's settings and data, and the page around them.

The page itself is a template file with the stylesheet and the script beside it, in ``templates/``
and ``static/``. Python only builds the fragments that carry data and puts them into the template.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ..focus import FocusSpec
from ..spec import Pipeline
from ..status import STATUS_LEVELS
from ..trace import PipelineTrace
from .assets import escape as _e
from .assets import page, page_data
from .files import file_settings
from .layout import canvas
from .project import ROWS_NOT_SAVED

# Rows drawn into the page for a slice of one entity, which the service cannot serve.
MAX_ROWS = 25

# Rows fetched when a reader opens a port. Bigger than MAX_ROWS because it costs the page nothing:
# it arrives only for the port actually being looked at.
WINDOW = 100

# The only view a panel carries. One entity's rows are asked of the service instead.
EVERYONE = {"all": None}

def _table(frame: pl.DataFrame, port: str | None = None, marked: bool = False) -> str:
    """One port's rows, with a search box under each column heading.

    Search is per column, since a reader's question is about a column. `marked` adds the status column
    first: a mark per row saying what the step said about it, with a choice of levels as its filter, drawn
    only for ports a step marked. Why: docs/design/viewer.md#keeping-the-page-small
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
    # A served port carries no rows in the page; a window is fetched when it is opened. A slice of
    # one entity is not a served port, so it still carries its rows.
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


def _columns_only(frame: pl.DataFrame) -> str:
    """A port's column names and types, and why its rows are not here, for a pipeline that keeps none."""
    head = "".join(f"<th>{_e(c)}<em>{_e(frame.schema[c])}</em></th>" for c in frame.columns)
    table = f'<div class="scroll"><table class="data"><thead><tr>{head}</tr></thead></table></div>' if head else ""
    return f'<div class="tbl norows">{table}<p class="more">{_e(ROWS_NOT_SAVED)}</p></div>'


def _scalar(value: Any) -> str:
    """One configured value, drawn so an empty string, a missing value and a boolean each stay recognisable."""
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


def _config(config: Mapping[str, Any], files: tuple[str, ...] = ()) -> str:
    """A step's configuration as the fields it sets, drawn from the value's own shape so any tool draws.

    A setting in ``files`` holds a file path. It is marked, so the page can offer to show the file.
    """
    if not config:
        return '<p class="more">nothing to configure</p>'
    rows = []
    for key, value in config.items():
        mark = f' data-file="{_e(key)}"' if key in files else ""
        rows.append(f'<div class="cfg"{mark}><span class="ck">{_e(key)}</span><span>{_value(value)}</span></div>')
    return "".join(rows)


def _event_if_ran(trace: PipelineTrace, step_id: str) -> Any:
    """The step's event, or None when the run stopped before reaching it.

    A failed run's page is drawn too, since that is when it is most wanted; a step that never ran is shown as not run.
    """
    return trace.event(step_id) if any(event.step_id == step_id for event in trace.events) else None


def _index(spec: Pipeline, trace: PipelineTrace) -> str:
    """Every step as a line, grouped by lane, in the order the run took them, for finding a step by name."""
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


def _files(spec: Pipeline, step_id: str) -> tuple[str, ...]:
    """The settings of a step's tool that hold a file path."""
    registry = spec.registry
    assert registry is not None  # a Pipeline always fills it in
    return file_settings(registry.get(spec.step(step_id).tool).config_template())


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
        f"<h5>configuration, as written</h5>{_config(step.config, _files(spec, step_id))}"
        f"{f'<h5>fed by</h5><ul class=feeds>{feeds}</ul>' if feeds else ''}"
        f"</section>"
    )


def _step_data(
    spec: Pipeline,
    trace: PipelineTrace,
    step_id: str,
    views: dict[str, Any],
    rows_saved: bool = True,
) -> str:
    step = spec.step(step_id)
    event = _event_if_ran(trace, step_id)
    # What went in and what came out, side by side, both served by name so search and paging work on
    # each.
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
                    if frame is None:
                        body = '<p class="more">released &mdash; not retained</p>'
                    elif not rows_saved:
                        body = _columns_only(frame)
                    else:
                        # Keyed off the port rather than off the step being read, so a marked port
                        # drawn as the next step's input carries the same marks it carries as an
                        # output: it is the same port and the same rows.
                        body = _table(frame, port=f"{ref.step}.{ref.port}", marked=ref in trace.status)
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


def build_panels(
    spec: Pipeline, trace: PipelineTrace, focus: FocusSpec | None, *, rows_saved: bool = True
) -> dict[str, str]:
    """Every step's data panel, keyed by step, saved beside the run and fetched one at a time.

    One panel per step in `spec.order()`, including steps that never ran. Only the all-rows view is drawn;
    one entity's rows are fetched when asked for. `focus` is not used. With `rows_saved` false, each
    port shows its columns and says its rows were not kept.
    """
    return {step_id: _step_data(spec, trace, step_id, EVERYONE, rows_saved) for step_id in spec.order()}


def _page_data(spec: Pipeline, focus: FocusSpec | None, run: Mapping[str, Any] | None) -> str:
    """The page's facts for its script: the pipeline, the focus, and the run record."""
    return page_data({"pipeline": spec.name, "focus": focus.label if focus is not None else None, **(run or {})})


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
    """The run as a page: the flow, every step's configuration, and the first step's data.

    Opened as a plain file it still shows every step and its settings; the rows need the server. Other
    steps' panels are fetched when opened (see `build_panels`). `run` is the saved run's record; it goes
    into the page data and decides the banner.
    """
    total_in = sum(s.rows for r, s in trace.shapes.items() if not spec.step(r.step).inputs)
    published = [r for r in trace.shapes if not spec.consumers(r)]
    order = spec.order()
    details = "".join(_step_detail(spec, trace, step_id) for step_id in order)
    # One name for the step the page opens on and the one panel it ships, because they have to be
    # the same step: named without shipping, the page opens on an empty pane until a fetch answers,
    # and shipped without naming, it carries a panel nothing ever shows.
    opening = order[0] if order else ""
    rows_saved = (run or {}).get("rows_saved", True) is not False
    panel = _step_data(spec, trace, opening, EVERYONE, rows_saved) if opening else ""
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
    if focus is not None and not rows_saved:
        # Following one entity reads its rows, and there are none to read.
        picker = (
            f'<div class="picker"><span class="plabel">{_e(focus.label)}</span>'
            f'<span class="picknote">Following one {_e(focus.label)} needs the rows. {_e(ROWS_NOT_SAVED)}</span></div>'
        )
    return page(
        "walkthrough",
        scripts=("files", "runbar", "walkthrough"),
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
