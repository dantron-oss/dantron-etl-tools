"""A chain's page: its pipelines as boxes, the order as lines, and what passes along each line.

The page shows no rows. Each box opens that pipeline's own page, where its steps and rows are. When
the chain has run, each box says how its pipeline did in the chain's latest run. Drawing the page
loads the chain's pipelines, the way checking it does, and runs nothing.

Why: docs/design/chains.md#the-chains-page-is-a-graph-and-nothing-more
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..chains import ChainError, ChainPlan, ChainRun, chain_runs, load_chain
from ..cli import COMMAND
from ..chains.chain_file import read_chain_file, run_order
from .assets import escape as _e
from .assets import page, page_data
from .project import Project
from .runs import last_runs, latest

__all__ = ["chain_page"]

BOX_W, BOX_H, COL_W, ROW_H, PAD = 210, 66, 400, 96, 24
#: How many files a line's label names before it gives a count instead.
LABEL_NAMES = 1


@dataclass(frozen=True)
class _Box:
    name: str
    title: str
    after: tuple[str, ...]


def chain_page(project: Project, name: str) -> str:
    """The page for one chain of ``project``: its graph, its values, its hand-offs and its last run."""
    entry = project.chains[name]
    plan: ChainPlan | None = None
    refusal: str | None = None
    try:
        plan = load_chain(project, name)
    except ChainError as error:
        refusal = str(error)
    boxes, order = _boxes(project, name, plan)
    runs = chain_runs(project, name)
    last = runs[0] if runs else None
    title = plan.title if plan else entry.title
    return page(
        "chain",
        scripts=(),
        title=_e(title),
        name=_e(name),
        path=_e(_shown(project, entry.path)),
        refusal=f'<pre class="message">{_e(refusal)}</pre>' if refusal else "",
        graph=_graph(boxes, order, plan, last) if boxes else "",
        values=_values(plan),
        last=_last(project, last),
        handoffs=_handoffs(plan),
        command=_e(f"{COMMAND} run {name} --write"),
        pagedata=page_data({"chain": name, "pipelines": list(order)}),
    )


def _shown(project: Project, path: Any) -> str:
    try:
        return path.relative_to(project.root).as_posix()
    except ValueError:
        return str(path)


def _boxes(project: Project, name: str, plan: ChainPlan | None) -> tuple[dict[str, _Box], tuple[str, ...]]:
    """The boxes to draw: from the checked plan, or from the chain file alone when it was refused."""
    if plan is not None:
        boxes = {one: _Box(one, member.title, member.after) for one, member in plan.members.items()}
        return boxes, plan.order
    try:
        chain = read_chain_file(project.chains[name])
        order = run_order(chain.members, name)
    except ChainError:
        return {}, ()
    titles = {one: entry.title for one, entry in project.pipelines.items()}
    boxes = {member.name: _Box(member.name, titles.get(member.name, member.name), member.after) for member in chain.members}
    return boxes, order


def _depths(boxes: Mapping[str, _Box], order: Sequence[str]) -> dict[str, int]:
    """Each pipeline's column: one more than the furthest column of what it runs after."""
    depth: dict[str, int] = {}
    for one in order:
        depth[one] = 1 + max((depth[before] for before in boxes[one].after), default=-1)
    return depth


def _graph(boxes: Mapping[str, _Box], order: Sequence[str], plan: ChainPlan | None, last: ChainRun | None) -> str:
    depth = _depths(boxes, order)
    rows: dict[int, int] = {}
    place: dict[str, tuple[int, int]] = {}
    for one in order:
        row = rows.get(depth[one], 0)
        rows[depth[one]] = row + 1
        place[one] = (PAD + depth[one] * COL_W, PAD + row * ROW_H)
    width = PAD * 2 + (max(depth.values()) * COL_W) + BOX_W
    height = PAD * 2 + (max(rows.values()) - 1) * ROW_H + BOX_H
    edges = [_edge(before, one, place, plan) for one in order for before in boxes[one].after]
    nodes = [_node(boxes[one], place[one], last) for one in order]
    return (
        f'<svg class="chain-graph" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="The pipelines of this chain and the order they run in">'
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z"/></marker></defs>'
        f"{''.join(edges)}{''.join(nodes)}</svg>"
    )


def _edge(before: str, one: str, place: Mapping[str, tuple[int, int]], plan: ChainPlan | None) -> str:
    (x1, y1), (x2, y2) = place[before], place[one]
    start_x, start_y, end_x, end_y = x1 + BOX_W, y1 + BOX_H / 2, x2, y2 + BOX_H / 2
    bend = (end_x - start_x) / 2
    path = f"M{start_x},{start_y} C{start_x + bend},{start_y} {end_x - bend},{end_y} {end_x - 2},{end_y}"
    passed = [handoff.target.shown for handoff in plan.handoffs_between(before, one)] if plan else []
    passed = list(dict.fromkeys(passed))
    kind = "passes" if passed else "order"
    if not passed:
        label, full = "order only", f"{one} runs after {before}; no file or table passes between them"
    elif len(passed) <= LABEL_NAMES:
        label, full = passed[0], f"{before} writes {passed[0]}, which {one} reads"
    else:
        label, full = f"{len(passed)} files and tables", f"{before} writes, and {one} reads: " + ", ".join(passed)
    mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2 - 6
    return (
        f'<g class="edge {kind}" data-from="{_e(before)}" data-to="{_e(one)}"><title>{_e(full)}</title>'
        f'<path d="{path}" marker-end="url(#arrow)"/>'
        f'<text class="elab" x="{mid_x}" y="{mid_y}" text-anchor="middle">{_e(_short(label, 30))}</text></g>'
    )


def _short(text: str, most: int = 34) -> str:
    return text if len(text) <= most else "…" + text[-(most - 1) :]


def _node(box: _Box, at: tuple[int, int], last: ChainRun | None) -> str:
    x, y = at
    outcome, said = "never", "not run in a chain yet"
    if last is not None:
        result = last.member(box.name) if box.name in {m.pipeline for m in last.members} else None
        if result is not None:
            outcome = result.outcome.replace(" ", "-")
            said = result.outcome if result.outcome != "finished" else f"finished {_when(result.finished)}"
    return (
        f'<a href="/{_e(box.name)}/" class="node {_e(outcome)}" data-pipeline="{_e(box.name)}" '
        f'data-outcome="{_e(outcome)}"><title>{_e(box.title)}: {_e(said)}</title>'
        f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8"/>'
        f'<text class="ntitle" x="{x + 12}" y="{y + 24}">{_e(_short(box.title, 26))}</text>'
        f'<text class="nname" x="{x + 12}" y="{y + 41}">{_e(_short(box.name, 30) if box.title != box.name else "")}</text>'
        f'<text class="nstate" x="{x + 12}" y="{y + 57}">{_e(said)}</text></a>'
    )


def _when(text: str | None) -> str:
    return text[:16].replace("T", " ") if text else ""


def _values(plan: ChainPlan | None) -> str:
    if plan is None or not plan.values:
        return ""
    rows = []
    for key, value in plan.values.items():
        users = [one for one, member in plan.members.items() if key in member.given]
        replaced = [
            f"{one}'s own setting says {member.replaced[key]}"
            for one, member in plan.members.items()
            if key in member.replaced
        ]
        note = f'<p class="note">Replaces: {_e("; ".join(replaced))}</p>' if replaced else ""
        rows.append(f"<tr><td><code>{_e(key)}</code></td><td>{_e(value)}{note}</td><td>{_e(', '.join(users))}</td></tr>")
    return (
        "<section><h2>Parameters</h2><p class=\"hint\">The default values in the chain file. Change one for a run "
        "with <code>--set name=value</code>.</p><table><thead><tr><th>Parameter</th><th>Value</th>"
        f"<th>Used by</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"
    )


def _last(project: Project, last: ChainRun | None) -> str:
    if last is None:
        return '<section><h2>Last run</h2><p class="hint">This chain has not been run yet.</p></section>'
    rows = []
    for result in last.members:
        note = ""
        if result.started:
            good, failed = last_runs(project.runs_folder, result.pipeline)
            now = latest(good, failed)
            if now is None or now.started != result.started:
                note = '<p class="note">Run again since, on its own or in another chain: its page shows the later run.</p>'
        message = f'<p class="note">{_e(result.message)}</p>' if result.message and result.outcome != "finished" else ""
        rows.append(
            f'<tr><td><a href="/{_e(result.pipeline)}/">{_e(result.pipeline)}</a>{note}</td>'
            f'<td class="outcome {_e(result.outcome.replace(" ", "-"))}">{_e(result.outcome)}{message}</td>'
            f"<td>{_e(_when(result.started))}</td><td>{_e(_when(result.finished))}</td></tr>"
        )
    values = ", ".join(f"{key} = {value}" for key, value in last.values.items())
    told = f" with {values}" if values else ""
    return (
        f'<section><h2>Last run</h2><p>Started {_e(_when(last.started))}{_e(told)}: '
        f'<span class="outcome {_e(last.outcome)}">{_e(last.outcome)}</span>.</p>'
        "<table><thead><tr><th>Pipeline</th><th>Outcome</th><th>Started</th><th>Finished</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _handoffs(plan: ChainPlan | None) -> str:
    if plan is None:
        return ""
    parts = []
    if plan.handoffs:
        rows = "".join(
            f"<tr><td>{_e(h.writer)}</td><td>{_e(h.target.shown)}</td><td>{_e(h.reader)}</td></tr>" for h in plan.handoffs
        )
        parts.append(
            "<section><h2>What passes between them</h2><table><thead><tr><th>Written by</th><th>File or table</th>"
            f"<th>Read by</th></tr></thead><tbody>{rows}</tbody></table></section>"
        )
    if plan.outside:
        items = "".join(f"<li><code>{_e(t.shown)}</code>, read by {_e(', '.join(r))}</li>" for t, r in plan.outside)
        parts.append(f"<section><h2>Read from outside the chain</h2><ul>{items}</ul></section>")
    if plan.results:
        items = "".join(f"<li><code>{_e(t.shown)}</code>, written by {_e(w)}</li>" for t, w in plan.results)
        parts.append(f"<section><h2>Written and not read again inside the chain</h2><ul>{items}</ul></section>")
    return "".join(parts)
