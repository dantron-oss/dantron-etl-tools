"""Where each step of a flow is drawn: the canvas, its lanes, its containers and its connections.

Everything here is read from the pipeline and the trace of a run. The shape of the canvas comes from
the edges; nothing is placed by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ..spec import Container, Pipeline, Step
from ..trace import PipelineTrace
from .assets import escape as _e

NODE_W, NODE_H, COL_W, PAD = 176, 62, 232, 28

# Vertical breathing room: between two steps stacked in one lane cell, inside a lane band, and
# between one lane band and the next.
STACK_GAP, LANE_PAD, LANE_GAP = 18, 22, 26

# A container's own margins: the room its outline keeps round the steps it holds, and the band above
# them its sentence and its counts are written in. The band is what a closed container shrinks to,
# so it has to be tall enough to be read on its own -- and the outline has to stay clear of the
# steps of neighbouring lanes, which a box drawn tight to its members does not.
GROUP_PAD, GROUP_HEAD = 12, 46

# The margins above keep a container's outline inside its own columns and lane, which `_slots`
# relies on. Why: docs/design/viewer.md#the-canvas


# Containers cost canvas height; the measurements are in docs/design/viewer.md#the-canvas.


def _depths(spec: Pipeline) -> dict[str, int]:
    """How far along the flow each step sits -- its longest path from a source."""
    depth: dict[str, int] = {}
    for step_id in spec.order():
        feeders = [ref.step for ref in spec.step(step_id).inputs.values()]
        depth[step_id] = 1 + max((depth[f] for f in feeders), default=-1)
    return depth


def _lane_index(spec: Pipeline) -> dict[str, int]:
    ordered = sorted(spec.lanes, key=lambda lane: lane.order)
    return {lane.id: i for i, lane in enumerate(ordered)}


def _lane_row(step: Step, lanes: dict[str, int]) -> int:
    """Which band a step is drawn in. A step naming no lane, or a lane the page never heard of,
    goes in the first band rather than nowhere -- the layout is not the place a bad lane is reported,
    and `_refuse_undeclared_lanes` has already refused one that came out of a file."""
    return lanes.get(step.lane or "", 0)


def _group_parts(spec: Pipeline, lanes: dict[str, int]) -> list[tuple[int, tuple[str, ...]]]:
    """Each container's members, split by lane, as (lane row, members).

    A container over two lanes is refused at load, but a Pipeline built in Python can still have one; each
    half is then placed in its own lane, so two steps are never drawn at one spot.
    """
    split: list[tuple[int, tuple[str, ...]]] = []
    for group in spec.containers:
        mine: dict[int, list[str]] = {}
        for member in spec.container_members(group.id):
            mine.setdefault(_lane_row(spec.step(member), lanes), []).append(member)
        split.extend((row, tuple(members)) for row, members in mine.items())
    return split


def _claimed_slots(base: int, deep: int) -> int:
    """The slots a container claims, as a bit per slot: its `deep` rows from `base`, and the row above for its header.

    A header (46px) reaches into the slot above (80px pitch), so it needs that slot, except at slot 0,
    where the gap above the lane already has room. Why: docs/design/viewer.md#the-canvas
    """
    return (1 << (base + deep)) - (1 << (base - 1 if base else 0))


def _slots(spec: Pipeline, depth: dict[str, int], lanes: dict[str, int]) -> dict[str, int]:
    """Where each step sits within its lane-and-depth cell, so no two steps share a spot.

    Containers are placed first, widest first in a fixed order, each claiming a rectangle of its lane's grid
    that nothing else may use; that rectangle is the outline drawn. Then each loose step takes the lowest
    free slot in its cell. With no containers this is the plain sequential fill.
    Why: docs/design/viewer.md#the-canvas
    """
    slot: dict[str, int] = {}
    # Which slots of each (lane, column) cell are taken, as one integer per cell with a bit per
    # slot. Both loops below read it and the first one reads whole rectangles of it at a time.
    used: dict[tuple[int, int], int] = {}

    widest_first = sorted(_group_parts(spec, lanes), key=lambda part: (-_width(part[1], depth), part[1]))
    for row, members in widest_first:
        # The members of one container, gathered into the columns they sit in. One pass, because the
        # claim is measured off these columns and the members are then placed down them.
        held: dict[int, list[str]] = {}
        for member in members:
            held.setdefault(depth[member], []).append(member)
        first, last, deep = min(held), max(held), max(len(at_column) for at_column in held.values())

        # Asking whether a whole rectangle is free is then one AND against the columns it spans,
        # rather than a sweep over its cells. It is the difference between a page that renders and
        # one that does not: 150 containers over 152 columns is a few million cell questions.
        wall = 0
        for column in range(first, last + 1):
            wall |= used.get((row, column), 0)
        base = 0
        while wall & _claimed_slots(base, deep):
            base += 1
        for column in range(first, last + 1):
            used[(row, column)] = used.get((row, column), 0) | _claimed_slots(base, deep)

        # Members fill their own column downwards from the top of the claim, so every column that
        # holds one puts it at `base`. That is what makes the claim and the outline the same
        # rectangle: a column whose members started lower would leave the top row of the claim empty
        # in every column, and the outline would sit a slot below the room reserved for its header.
        for at_column in held.values():
            for n, member in enumerate(at_column):
                slot[member] = base + n

    for step in spec.steps:
        if step.id in slot:
            continue
        cell = (_lane_row(step, lanes), depth[step.id])
        free = 0
        while used.get(cell, 0) >> free & 1:
            free += 1
        used[cell] = used.get(cell, 0) | 1 << free
        slot[step.id] = free
    return slot


def _width(members: tuple[str, ...], depth: dict[str, int]) -> int:
    """How many columns a container's members span, which is how hard it is to place."""
    columns = [depth[member] for member in members]
    return max(columns) - min(columns) + 1


def canvas(spec: Pipeline, trace: PipelineTrace) -> str:
    """The flow as it is wired: a node per step, an edge per port-to-port connection."""
    depth, lanes = _depths(spec), _lane_index(spec)
    slot = _slots(spec, depth, lanes)

    # A lane is as tall as the most crowded cell in it, so a lane holding one step stays slim and
    # one holding four grows to fit rather than overflowing into its neighbour.
    stacked: dict[int, int] = {}
    for step in spec.steps:
        row = _lane_row(step, lanes)
        stacked[row] = max(stacked.get(row, 1), slot[step.id] + 1)

    top: dict[int, float] = {}
    tall: dict[int, float] = {}
    y = PAD
    for row in sorted(set(lanes.values()) | set(stacked)):
        count = stacked.get(row, 1)
        tall[row] = count * NODE_H + (count - 1) * STACK_GAP + 2 * LANE_PAD
        top[row] = y
        y += tall[row] + LANE_GAP

    width = (max(depth.values()) + 1) * COL_W + PAD
    height = y - LANE_GAP + PAD

    def box(step_id: str) -> tuple[float, float]:
        step = spec.step(step_id)
        row = _lane_row(step, lanes)
        x = PAD + depth[step_id] * COL_W
        return x, top[row] + LANE_PAD + slot[step_id] * (NODE_H + STACK_GAP)

    bands = "".join(
        f'<rect class="band" x="0" y="{top[lanes[lane.id]]}" width="{width}" height="{tall[lanes[lane.id]]}"/>'
        for lane in spec.lanes
    )
    # The lane's name is drawn last, over a halo, so a container header in the same gap cannot hide it.
    lane_names = "".join(
        f'<text class="lane" x="6" y="{top[lanes[lane.id]] - 7}">{_e(lane.title)}</text>' for lane in spec.lanes
    )

    edges = []
    for step in spec.steps:
        tx, ty = box(step.id)
        for ref in step.inputs.values():
            sx, sy = box(ref.step)
            x1, y1 = sx + NODE_W, sy + NODE_H / 2
            x2, y2 = tx, ty + NODE_H / 2
            mid = (x1 + x2) / 2
            rows = trace.shapes[ref].rows if ref in trace.shapes else None
            # Escape the data, never the markup: `_e` on a string that already holds an entity
            # turns its `&` into `&amp;` and the reader is shown the entity's source text.
            label = f"{_e(ref.port)} &middot; {rows:,}" if rows is not None else _e(ref.port)
            curve = f"M{x1},{y1} C{mid},{y1} {mid},{y2} {x2},{y2}"
            # One group per connection, naming its two end steps, their containers ("" for none) and their
            # coordinates, with a 14px transparent twin path to click on. Why: docs/design/viewer.md#the-canvas
            edges.append(
                f'<g class="edge" data-from="{_e(ref.step)}" data-to="{_e(step.id)}" data-port="{_e(ref.port)}"'
                f' data-from-group="{_e(spec.step(ref.step).container or "")}"'
                f' data-to-group="{_e(step.container or "")}"'
                f' data-sx="{x1}" data-sy="{y1}" data-tx="{x2}" data-ty="{y2}">'
                f'<path class="wire" d="{curve}"/><path class="hit" d="{curve}"/>'
                f'<text class="elab" x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 6}">{label}</text>'
                f"</g>"
            )

    nodes: dict[str, str] = {}
    for step in spec.steps:
        x, y_node = box(step.id)
        event = trace.event(step.id) if any(e.step_id == step.id for e in trace.events) else None
        level = event.level if event else "info"
        outs = "".join(
            f'<circle class="anchor out" cx="{x + NODE_W}" cy="{y_node + 14 + n * 16}" r="4"/>'
            for n in range(len(event.outputs) if event else 1)
        )
        ins = "".join(
            f'<circle class="anchor in" cx="{x}" cy="{y_node + 14 + n * 16}" r="4"/>' for n in range(len(step.inputs))
        )
        nodes[step.id] = (
            f'<a class="node {level}" id="node-{_e(step.id)}" href="#step-{_e(step.id)}">'
            f'<rect x="{x}" y="{y_node}" width="{NODE_W}" height="{NODE_H}" rx="7"/>'
            f'<text class="nid" x="{x + 12}" y="{y_node + 24}">{_e(step.id)}</text>'
            f'<text class="ntool" x="{x + 12}" y="{y_node + 43}">{_e(step.tool)}</text>'
            # The port dots sit inside the node rather than beside it, so a faded step fades whole:
            # six hundred bright anchors over dimmed boxes is the noise the trace exists to remove.
            f"{ins}{outs}</a>"
        )

    # The containers, each holding the steps it was declared over. A step in no container is written
    # out exactly as it always was: most of a 672-step flow will be in none for a long while, and a
    # canvas that indented those, or reserved space for a box round them, would make the first person
    # who tries this feature undo it.
    boxed = {member for group in spec.containers for member in spec.container_members(group.id)}
    containers = "".join(_group_box(spec, trace, group, box, nodes) for group in spec.containers)
    loose = "".join(nodes[step.id] for step in spec.steps if step.id not in boxed)
    return (
        f'<div class="scroll canvas" id="flow"><svg viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" data-w="{width}" data-h="{height}">'
        # A layer with nothing in it, last of everything, for the one title a reader has clicked
        # into. It has to be a place in the document rather than a class on the title, because SVG
        # has no z-index: what is written last is drawn on top, and nothing a stylesheet says will
        # move a title written above the lane names to below them on the screen.
        f"{bands}{''.join(edges)}{containers}{loose}{lane_names}"
        f'<g class="front"></g></svg></div>'
    )


def _group_box(
    spec: Pipeline,
    trace: PipelineTrace,
    group: Container,
    box: Callable[[str], tuple[float, float]],
    nodes: Mapping[str, str],
) -> str:
    """One container: the outline round its steps, the bar it becomes when closed, and the steps inside it.

    The steps are inside the element, so closing is one class. The bar keeps the full width, so the title
    fits. Connections stay at the root of the drawing and name their containers at both ends.
    Why: docs/design/viewer.md#the-canvas
    """
    members = spec.container_members(group.id)
    corners = [box(member) for member in members]
    left = min(x for x, _ in corners) - GROUP_PAD
    top = min(y for _, y in corners) - GROUP_HEAD
    width = max(x for x, _ in corners) + NODE_W + GROUP_PAD - left
    height = max(y for _, y in corners) + NODE_H + GROUP_PAD - top

    rows_in = sum(trace.shapes[port].rows for port in spec.container_inputs(group.id) if port in trace.shapes)
    rows_out = sum(trace.shapes[port].rows for port in spec.container_outputs(group.id) if port in trace.shapes)
    return (
        f'<g class="group" id="group-{_e(group.id)}" data-group="{_e(group.id)}" '
        f'data-rows-in="{rows_in}" data-rows-out="{rows_out}">'
        f'<rect class="gbox" x="{left}" y="{top}" width="{width}" height="{height}" rx="10"/>'
        f'<rect class="gpill" x="{left}" y="{top}" width="{width}" height="{GROUP_HEAD}" rx="10"/>'
        # A click target over the header strip, sized from the title's length; not the whole box, which
        # would take clicks from the connections crossing it.
        f'<rect class="ghit" x="{left}" y="{top}" width="{12 + len(group.title) * 7.0 + 8}"'
        f' height="{GROUP_HEAD}"/>'
        # The plate the raised title sits on, sized by the same character-count estimate.
        f'<rect class="gplate" x="{left + 5}" y="{top + 4}"'
        f' width="{len(group.title) * 7.0 + 14}" height="22" rx="4"/>'
        f'<text class="gtitle" x="{left + 12}" y="{top + 18}">{_e(group.title)}</text>'
        # Grouped digits like every count on the page; the tooltip and data attributes carry the plain figures.
        f'<text class="gcount" x="{left + 12}" y="{top + 34}">'
        f"<title>{rows_in} rows came in from outside this container, {rows_out} rows left it</title>"
        f"{rows_in:,} in &middot; {rows_out:,} out</text>"
        f'<g class="gctl" data-collapse="{_e(group.id)}">'
        f'<rect x="{left + width - 26}" y="{top + 8}" width="18" height="18" rx="4"/>'
        f'<text class="gopen" x="{left + width - 17}" y="{top + 21}">&minus;</text>'
        f'<text class="gshut" x="{left + width - 17}" y="{top + 21}">+</text>'
        f"</g>"
        f"{''.join(nodes[member] for member in members)}"
        f"</g>"
    )

