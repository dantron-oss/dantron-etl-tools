"""Render one run of a pipeline as a page: the canvas, and the data at every port.

The page exists because a figure that is wrong has to be traceable to the step that made it
wrong. So it draws the flow as it is wired, and then shows, for every port, the rows that
actually left it -- including the ports a step set aside, which is where a missing figure
usually went.

Everything here is read from the pipeline file and the trace of a run. Nothing is written down
twice: the shape of the canvas comes from the edges, the panel under a step comes from that
step's own configuration as authored, and the tables come from the frames the run produced.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import polars as pl

import dantron_etl
from dantron_etl.focus import FocusSpec, PortSlice, available_entities, slice_trace
from dantron_etl.paths import project_root
from dantron_etl.pipeline import Group, PipelineSpec, StepSpec, read_pipeline_file
from dantron_etl.runner import PipelineTrace, run_pipeline
from dantron_etl.tool import STATUS_LEVELS

# Resolved from this script's own location rather than from the library's -- see `dantron_etl.paths`.
PROJECT_ROOT = project_root(__file__)
OUT_PATH = PROJECT_ROOT / "output_files" / "pipeline_walkthrough.html"
# Rows drawn into the page for a slice of one entity, which the service cannot serve.
MAX_ROWS = 25

# Rows fetched when a reader opens a port. Bigger than MAX_ROWS because it costs the page nothing:
# it arrives only for the port actually being looked at.
WINDOW = 100

# How many entities the picker may offer. Every one of them embeds a slice of every port, so this
# multiplies the page; pagination and search are the real answer and are not built yet.
MAX_ENTITIES = 3

NODE_W, NODE_H, COL_W, PAD = 176, 62, 232, 28

# Vertical breathing room: between two steps stacked in one lane cell, inside a lane band, and
# between one lane band and the next.
STACK_GAP, LANE_PAD, LANE_GAP = 18, 22, 26

# A container's own margins: the room its outline keeps round the steps it holds, and the band above
# them its sentence and its counts are written in. The band is what a closed container shrinks to,
# so it has to be tall enough to be read on its own -- and the outline has to stay clear of the
# steps of neighbouring lanes, which a box drawn tight to its members does not.
GROUP_PAD, GROUP_HEAD = 12, 46

# The three margins above are what make a container's rectangle derivable from its members' slots
# rather than measured off the drawing, which is what `_slots` relies on. Two of them are load
# bearing beyond their own box:
#
#   GROUP_PAD 12 < (COL_W - NODE_W) / 2 = 28, so an outline never reaches out of its own columns
#   into the one beside it. Column-disjoint containers therefore cannot touch, whatever their rows.
#
#   GROUP_PAD + GROUP_HEAD = 58 < 2 * LANE_PAD + LANE_GAP = 70, so a container at the bottom of one
#   lane and one at the top of the next cannot touch either, and the header of a container in the
#   first slot of a lane fits in the gap above that lane without reaching the lane before it.
#
# Change either margin past those bounds and containers start colliding across a boundary `_slots`
# never looks at, because `_slots` reasons in whole grid cells and these are what make a cell's
# worth of separation enough.


# WHERE THE CONTAINERS COST SOMETHING. A container is drawn round its members, so the layout has to
# put the members somewhere a box can be drawn round without the box also reaching round a step that
# is not its own. `_slots` does that by handing each container a rectangle of its lane's grid and
# letting nothing else into it. What that costs is height: on the flow this page was built for --
# 672 steps, 150 containers, 152 columns -- the canvas goes from 3,146px unboxed to 8,346px boxed,
# 2.65x. A flow that declares no container reserves nothing and is laid out to the pixel where it
# always was.
#
# Two things spend that height, and anyone tuning it should know which is which before touching
# `_slots`. Measured by re-running the same flow with one of them turned off at a time:
#
#   The header row costs the most: 2,240px of the 5,200px added. Stop reserving a slot for a header
#   and the canvas is 6,106px. A header needs GROUP_HEAD 46 of a slot's 80 and takes the whole slot
#   because slots are what the grid is made of, so the 34px left over is what a finer grid would win
#   back -- most of it, not all.
#
#   The claims' column spans cost 1,840px. Let a claim reserve only the columns its members are
#   actually in and the canvas is 6,506px -- but the outline is the bounding box of the members, so
#   the columns in between are inside the box whether they were reserved or not, and a step there is
#   a step inside a container that does not own it. That height is not paid for the layout's
#   convenience; it is paid for the rectangle. Winning it back means a container that is not drawn
#   as one rectangle.
#
# The order the containers are placed in is not where the height is. Widest-first lands at 8,346px
# against 8,826 left to right and 8,666 in declaration order, and the busiest column of the busiest
# lane needs 52 rows on its own, so the shipped layout is one row off what any order could do. There
# is nothing left in the packing; the two figures above are the whole bill.


def _e(v: object) -> str:
    return html.escape(str(v), quote=True)


def _depths(spec: PipelineSpec) -> dict[str, int]:
    """How far along the flow each step sits -- its longest path from a source."""
    depth: dict[str, int] = {}
    for step_id in spec.order():
        feeders = [ref.step for ref in spec.step(step_id).inputs.values()]
        depth[step_id] = 1 + max((depth[f] for f in feeders), default=-1)
    return depth


def _lane_index(spec: PipelineSpec) -> dict[str, int]:
    ordered = sorted(spec.lanes, key=lambda lane: lane.order)
    return {lane.id: i for i, lane in enumerate(ordered)}


def _lane_row(step: StepSpec, lanes: dict[str, int]) -> int:
    """Which band a step is drawn in. A step naming no lane, or a lane the page never heard of,
    goes in the first band rather than nowhere -- the layout is not the place a bad lane is reported,
    and `_refuse_undeclared_lanes` has already refused one that came out of a file."""
    return lanes.get(step.lane or "", 0)


def _group_parts(spec: PipelineSpec, lanes: dict[str, int]) -> list[tuple[int, tuple[str, ...]]]:
    """Each container's members, split by the lane they are in, as (lane row, members).

    Split rather than taken whole because `_slots` reserves cells of one lane's grid, and a lane is
    the only thing that grid is indexed by. A container declared over two lanes is not a shape this
    page has a drawing for -- its outline would have to reach across the band between them, over
    every step of both lanes in its columns -- so the honest thing is to place each half properly in
    its own lane and let the outline be the bounding box of the two, which is what it is today. The
    guarantees below are per lane and so is the claim they support.

    No flow read from a file can get here with one, though: `_refuse_bad_groups` refuses a group
    whose steps sit in two lanes when the document is loaded, and says to move the step or split the
    container. What reaches this split is a `PipelineSpec` built in Python, which is how the layout's
    own cases are written. It stays because it is what makes "a claim is a rectangle of one lane's
    grid" true of every input rather than of the ones the loader happens to allow -- take it out and
    a two-lane container silently has half its members placed against the wrong lane's occupancy,
    which is two steps drawn at one coordinate, the defect this whole function exists under.
    """
    split: list[tuple[int, tuple[str, ...]]] = []
    for group in spec.groups:
        mine: dict[int, list[str]] = {}
        for member in spec.group_members(group.id):
            mine.setdefault(_lane_row(spec.step(member), lanes), []).append(member)
        split.extend((row, tuple(members)) for row, members in mine.items())
    return split


def _claimed_slots(base: int, deep: int) -> int:
    """The slots a container claims outright, as a bit per slot: its own ``deep`` rows from ``base``,
    and the empty row above them that its header is written in.

    Named for the slots rather than for the band it is drawn as, because a *band* on this page is
    already the horizontal stripe a lane is drawn as -- `<rect class="band">` -- and these are slots
    of one column inside one of those, the same slots `_slots` hands out.

    The header row is the whole reason a container claims more than its members. `GROUP_HEAD` is 46
    and a slot's pitch is `NODE_H + STACK_GAP` = 80, so the band the title is written in reaches 46px
    up into a slot 80px tall -- past the bottom 18px of gap and 28px into the box of whatever sits in
    the slot above. There is no arrangement in which a header shares a slot with a step; it either
    has the slot or is written across a step's own box.

    Except at the top of a lane, where the room is already there and reserving a slot for it would
    waste a row on every container that starts at slot 0. Above slot 0 sits LANE_PAD and then either
    the page's own PAD (28 + 22 = 50) or LANE_GAP and the lane before's LANE_PAD (22 + 26 + 22 = 70),
    and both are more than GROUP_HEAD. That space holds no step by construction, so a header at slot
    0 is drawn in a gap that was already empty.

    Empty of *steps*, which is not the same as empty. The lane writes its own label in that gap, at
    x=6 above the band, so a container that both starts at slot 0 and reaches column 0 writes its
    title across the lane's name. That is an older and smaller thing than the three this layout was
    built for: on the dense flow it happens to 1 container of 150, where the layout this replaced --
    which drew a header GROUP_HEAD above whatever its topmost member was -- did it to 4. It is also
    not a layout fault and has no layout fix: the header is where it has to be, and what is wrong is
    that the label shares the space. Moving the label, or drawing it over the containers with
    something behind it, is a change to the drawing order and belongs with whoever next touches the
    lane bands.
    """
    return (1 << (base + deep)) - (1 << (base - 1 if base else 0))


def _slots(spec: PipelineSpec, depth: dict[str, int], lanes: dict[str, int]) -> dict[str, int]:
    """Where each step sits *within* its lane-and-depth cell, so two steps never share a spot.

    Depth and lane alone do not place a step uniquely: two sources read at the same moment into the
    same lane, which is the ordinary shape of a flow rather than an edge case. Drawing both at one
    coordinate puts one exactly on top of the other, and the canvas then shows fewer tools than the
    pipeline has -- with the hidden one's data still appearing below, which reads as data arriving
    from nowhere.

    Containers are placed first, and they are placed by claiming grid cells rather than by being
    drawn round wherever their members landed. A container spanning columns c0..c1 whose busiest
    column holds ``deep`` of its members takes the rectangle c0..c1 x base..base+deep-1, plus the
    slot above for its header, and nothing else is ever put in it. That rectangle *is* the outline
    `_group_box` draws: the topmost member is at ``base`` in every column that has one, the lowest is
    at ``base + deep - 1``, so the bounding box of the members is the bounding box of the claim. So
    "no stranger inside the outline" and "no two outlines crossing" both reduce to "no two claims
    share a cell", which is the one thing the loop below has to get right.

    That the outline stops at the claim, rather than spilling into the cells beside it, is what the
    two margin bounds above pay for: GROUP_PAD keeps it inside its own columns and GROUP_PAD +
    GROUP_HEAD keeps it inside its own lane. The one direction with no margin at all is downwards
    into the next slot -- an outline ends GROUP_PAD = 12 below its last member, and the next slot
    starts 18px below that -- so a step directly under a container is fine and needs no reservation.

    Widest first, because a container's claim is a horizontal interval and the wide ones are what
    every later interval has to fit around. Left to right instead costs six rows on the flow this
    page is for: a container spanning seventy columns placed last has to clear seventy columns'
    worth of whatever has already been put down, and it ends up alone at the top of the lane. The
    order is a tie-broken total order rather than a sort by width alone, so that two renders of one
    file draw the same page.

    Then the ungrouped steps, each into the lowest slot of its own cell that no container claimed.
    On a flow with no containers nothing is claimed and this is exactly the sequential fill it has
    always been -- same steps, same declaration order, same slots, which is why a page that declares
    no group does not move.
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


def _canvas(spec: PipelineSpec, trace: PipelineTrace) -> str:
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
    # The lane's name is drawn separately from its band, and last of everything, because the strip
    # of canvas it sits in -- the gap above the band -- is also where a container starting at the
    # first column puts its header. One container in a hundred and fifty lands there. Painted with
    # the bands, the name goes under that header and cannot be read at all; painted last, over a
    # halo, both read, and the name wins the few characters they share. The name is the one label
    # on the canvas a reader cannot recover from anything else on screen, so it is the one to keep.
    lane_names = "".join(
        f'<text class="lane" x="6" y="{top[lanes[lane.id]] - 7}">{_e(lane.label)}</text>'
        for lane in spec.lanes
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
            # One group per connection, naming its two ends as the ids the nodes answer to. A line
            # drawn as two anonymous siblings is a line no script can ask a question about: to light
            # what feeds a step you have to know which lines land on it, and the only place that can
            # be read off is the drawing itself. Naming the ends here is what makes the canvas a
            # graph rather than a picture of one.
            #
            # The second path is the same curve, fourteen pixels wide and transparent. A stroked
            # path with no fill takes clicks on its stroke alone -- 1.4 pixels of it -- and picking
            # one line out of six that cross is not something a hand can do at that width. The twin
            # has to follow the same `d`: a target beside the line lights its neighbour, which is
            # worse than no target at all.
            #
            # Each end also names the container it sits in, as the empty string where there is
            # none. Stated at both ends rather than worked out when a container closes, because that
            # is the fact a closed container re-wires itself from: one end inside and one outside is
            # a connection that has to land on the box, both ends inside is the box's own plumbing.
            # Written as "" rather than left off, because `undefined === undefined` is true in a
            # browser -- two ungrouped ends would compare equal, every connection between two
            # ordinary steps would read as the internals of a container that does not exist, and all
            # of them would go out the moment any container was closed.
            #
            # And the two ends' own coordinates, so that re-opening a container puts its connections
            # back where they were drawn. Re-deriving them in the browser means going two ways, and
            # both are worse than four numbers. Measuring the node is out on its face: while a
            # container is shut its steps are display:none, and a display:none node measures as
            # nothing. Reading the node's rectangle off its attributes does survive being hidden --
            # but the point a connection leaves from is not the rectangle, it is `x + NODE_W` and
            # `y + NODE_H / 2`, so that route restates this file's geometry in a second language and
            # every future change to a node's shape has to be made in both. Four numbers per
            # connection is 765 of them on the real flow and a few tens of KB on a page that already
            # embeds its own data; the arithmetic staying in one language is worth that.
            edges.append(
                f'<g class="edge" data-from="{_e(ref.step)}" data-to="{_e(step.id)}" data-port="{_e(ref.port)}"'
                f' data-from-group="{_e(spec.step(ref.step).group or "")}"'
                f' data-to-group="{_e(step.group or "")}"'
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
    boxed = {member for group in spec.groups for member in spec.group_members(group.id)}
    containers = "".join(_group_box(spec, trace, group, box, nodes) for group in spec.groups)
    loose = "".join(nodes[step.id] for step in spec.steps if step.id not in boxed)
    return (
        f'<div class="scroll canvas" id="flow"><svg viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" data-w="{width}" data-h="{height}">'
        f"{bands}{''.join(edges)}{containers}{loose}{lane_names}</svg></div>"
    )


def _group_box(
    spec: PipelineSpec,
    trace: PipelineTrace,
    group: Group,
    box: Callable[[str], tuple[float, float]],
    nodes: Mapping[str, str],
) -> str:
    """One container: the outline round its steps, the bar it becomes when closed, and its steps.

    The steps are written *inside* the container element rather than beside it, and the rest follows
    from that. Enclosure becomes something the document states rather than something the coordinates
    imply, and closing a container becomes one class on one element instead of a pass over six
    hundred nodes asking each whether it belongs to the box being closed.

    Two rectangles, because a container makes the same claim in two states and they are different
    shapes. ``gbox`` is the outline round exactly its own steps, drawn while it is open. ``gpill`` is
    the bar it shrinks to when it is closed -- the header alone, which is the whole point: a
    container that kept its expanded footprint when shut would free no space, and space is the only
    reason a reader shuts one. Both start at the same corner, so the sentence written in the header
    sits on whichever of the two is being drawn.

    It shrinks in one direction only. The bar keeps the container's full width, because the width is
    what the sentence is written across -- a bar one node wide would truncate every title, which is
    to say it would put away the only thing a closed container has to say. So what closing actually
    gives back, measured on the real flow with its steps boxed six at a time, is 154px of height at
    the median against a width of 1,360px that does not move. The reader's real gain is the steps and
    their internal connections going away in a section they have finished with -- 483 of that flow's
    765 connections, 63%, are internal to a six-step container and go out with it -- and the height is
    a bonus; a closed container is not a way to make a 35,292px canvas narrower.

    The connections are *not* in here. They stay at the root of the drawing, because a lit connection
    is lifted above the steps with ``appendChild`` -- a connection living inside a container would be
    lifted out of its box the first time a reader followed it, and never come back. Each names its
    container at each end instead, which is what the collapse handler re-points from.
    """
    members = spec.group_members(group.id)
    corners = [box(member) for member in members]
    left = min(x for x, _ in corners) - GROUP_PAD
    top = min(y for _, y in corners) - GROUP_HEAD
    width = max(x for x, _ in corners) + NODE_W + GROUP_PAD - left
    height = max(y for _, y in corners) + NODE_H + GROUP_PAD - top

    rows_in = sum(trace.shapes[port].rows for port in spec.group_inputs(group.id) if port in trace.shapes)
    rows_out = sum(trace.shapes[port].rows for port in spec.group_outputs(group.id) if port in trace.shapes)
    return (
        f'<g class="group" id="group-{_e(group.id)}" data-group="{_e(group.id)}" '
        f'data-rows-in="{rows_in}" data-rows-out="{rows_out}">'
        f'<rect class="gbox" x="{left}" y="{top}" width="{width}" height="{height}" rx="10"/>'
        f'<rect class="gpill" x="{left}" y="{top}" width="{width}" height="{GROUP_HEAD}" rx="10"/>'
        f'<text class="gtitle" x="{left + 12}" y="{top + 18}">{_e(group.title)}</text>'
        # Grouped digits, because the figure is read at a glance and `11834 in` is not read at a
        # glance -- it is counted, a digit at a time, which is exactly the work this feature exists
        # to spare a reader. Grouped is also what the rest of the page already does: every edge
        # label, every "N rows" under a port and every "showing N of M" is written `{n:,}`, and a
        # container reporting its rows in the one unpunctuated style on the page would read as a
        # different kind of number rather than as the same rows counted at a different place.
        #
        # The plain figure is not thrown away -- it is put where the reader who wants it goes. The
        # tooltip is the container's accessible name and the string a reader copies, so it carries
        # the two figures unpunctuated and says which of them is which, which `in &middot; out` in a
        # 10px header has no room to. The two are the same rows said twice, once to be read and once
        # to be taken, and `data-rows-in`/`data-rows-out` above are the same figures again for the
        # script -- three spellings, one sum each, all computed in one place a line above.
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


def _slice_body(slice_: PortSlice, frame: pl.DataFrame | None) -> str:
    """One port, for one reader's chosen entity -- or the reason there is nothing to draw."""
    if slice_.status in {"no_grain", "not_retained"}:
        return f'<p class="more">{_e(slice_.reason)}</p>'
    if frame is None or frame.height == 0:
        return '<p class="more">no rows here for this one &mdash; the port does carry the entity</p>'
    return _table(frame)


def _index(spec: PipelineSpec, trace: PipelineTrace) -> str:
    """Every step as a line, grouped by lane, in the order the run took them.

    Fifty-three nodes is more than a canvas can be scanned for, and a reader looking for one by name
    should not have to pan a drawing to find it. Ordered by the run rather than by the file, because
    that is the order a reader is thinking in when they are following a figure.
    """
    lanes: dict[str, list[str]] = {}
    for step_id in spec.order():
        step = spec.step(step_id)
        event = trace.event(step_id)
        mark = ' class="flag"' if event.level == "warning" else ""
        lanes.setdefault(step.lane or "", []).append(
            f'<li><button data-goto="{_e(step_id)}"{mark}>'
            f"<span>{_e(step_id)}</span><em>{_e(step.tool)}</em></button></li>"
        )

    labels = {lane.id: lane.label for lane in spec.lanes}
    groups = "".join(
        f'<li class="lanehead">{_e(labels.get(lane, lane) or "elsewhere")}</li>{"".join(items)}'
        for lane, items in lanes.items()
    )
    return (
        '<nav class="index" aria-label="every step"><header><b>steps</b>'
        f"<span>{sum(len(items) for items in lanes.values())}</span></header>"
        f"<ul>{groups}</ul></nav>"
    )


def _step_detail(spec: PipelineSpec, trace: PipelineTrace, step_id: str) -> str:
    """What this step is: why it exists, how it is configured, what feeds it, what it cost."""
    step = spec.step(step_id)
    event = trace.event(step_id)
    notes = (
        "<ul class=notes>" + "".join(f"<li>{_e(note)}</li>" for note in event.notes) + "</ul>" if event.notes else ""
    )
    feeds = "".join(
        f"<li><code>{_e(name)}</code> &larr; <code>{_e(ref)}</code></li>" for name, ref in step.inputs.items()
    )
    return (
        f'<section class="detail {event.level}" data-step="{_e(step_id)}" hidden>'
        f'<header><h3><code>{_e(step_id)}</code><span class="tool">{_e(step.tool)}</span></h3>'
        f'<p class="where">{_e(step.lane or "")} &middot; {_e(step.stage or "")} &middot; '
        f"{event.duration_ms:.1f} ms</p>"
        f"{f'<p class=why>{_e(step.why)}</p>' if step.why else ''}{notes}</header>"
        f"<h5>configuration, as written</h5>{_config(step.config)}"
        f"{f'<h5>fed by</h5><ul class=feeds>{feeds}</ul>' if feeds else ''}"
        f"</section>"
    )


def _step_data(
    spec: PipelineSpec,
    trace: PipelineTrace,
    step_id: str,
    views: dict[str, Any],
) -> str:
    step = spec.step(step_id)
    event = trace.event(step_id)
    # What left, and what arrived. A step's inputs are upstream output ports, so both sides are
    # frames this run already produced and the service already serves by name -- which is what makes
    # searching and paging work on an input exactly as it does on an output.
    #
    # Reading a step used to mean holding its output in your head and scrolling to the step before it
    # to see what went in. The comparison a reader actually wants -- what did this step DO -- is
    # between two frames that were never on the screen together.
    leaving = [(shape.port, shape.rows, _e(shape.port.port)) for shape in event.outputs]
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
                if view is None:
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
                else:
                    sliced = view.slice(ref)
                    body = _slice_body(sliced, sliced.frame)
                    shown = "&mdash;" if sliced.rows_focused is None else f"{sliced.rows_focused:,}"
                    count = f"{shown} of {rows:,} rows" if rows is not None else shown
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


def _views(trace: PipelineTrace, focus: FocusSpec) -> tuple[dict[str, Any], int, int]:
    """The picker's entries -- everybody, then one per offered entity -- with how many of each there are.

    Every entity carries a full slice of every port, so the page grows with the number of entities
    multiplied by the number of ports. Three of them is nothing; 1,557 is a page no browser will
    open. So the picker takes the first ``MAX_ENTITIES`` and says how many there were.

    First, deliberately, rather than largest or busiest. Any ranking would need a port to rank at --
    an entity's count means nothing except against one named port, since the same entity is one row
    at a parent grain and several at a child -- and choosing that port is a judgement about one
    flow's shape, which is not something a renderer shared by every flow should be making. Paging
    and search are the real answer; this is the honest placeholder until they exist.

    Derived here rather than inside the page, because the panels are now built separately and a
    panel whose views were keyed differently from the picker's buttons would arrive with every one
    of them hidden: the reader clicks a step and the pane fills with nothing.

    Which means both :func:`build` and :func:`build_panels` run this, so writing one page slices the
    trace twice. Left as it is, with no memo: at ``MAX_ENTITIES`` of three the work is three passes
    over the ports a run retained, done once when the run is written, beside writing every one of
    those ports to Parquet -- and a memo keyed on a trace, which is mutable and unhashable, is a
    stale page waiting to happen. The day paging replaces the cap, this stops being small, and the
    answer then is a caller that slices once and hands the views to both.
    """
    options = available_entities(trace, focus)
    shown = options[:MAX_ENTITIES]
    views: dict[str, Any] = {"all": None}
    for option in shown:
        views[str(option.value)] = slice_trace(trace, focus, option.value)
    return views, len(shown), len(options)


def build_panels(spec: PipelineSpec, trace: PipelineTrace, focus: FocusSpec) -> dict[str, str]:
    """Every step's data panel, keyed by the step, ready to be written down and fetched one at a time.

    The markup is exactly what the page used to carry: same tables, same headers, same per-column
    search boxes, same in/out flip, same bodies left for the service. What changed is *when* it is
    built -- on the real flow, 672 of these came to 7.76 MB of an 8.98 MB page, with 24,937 form
    controls the browser laid out before it could paint, for the 671 steps nobody had opened.

    Keyed off ``spec.order()`` rather than off the trace's events, so every step a reader can pick
    -- a node on the canvas, a line in the step list -- has a panel behind it. A step the flow can
    select and the service cannot answer for is a dead end found by clicking.
    """
    views, _, _ = _views(trace, focus)
    return {step_id: _step_data(spec, trace, step_id, views) for step_id in spec.order()}


def build(spec: PipelineSpec, trace: PipelineTrace, focus: FocusSpec) -> str:
    """The run as a page: the flow, every step's configuration, and the data of the step being read.

    The panels are no longer in here -- they are written beside the run's ports and fetched as they
    are opened, which is what :func:`build_panels` is for. One exception, and it is the reason the
    page opens on anything at all: the step named by ``data-first`` is chosen before any fetch can
    have answered, so its panel ships in the document. Every other panel in here would be weight for
    a step nobody has clicked.

    What stays is everything a reader has without a service answering: the canvas, the step list and
    each step's configuration as it was authored. Opened as a plain file the page still says what
    every step does; it is the rows, and now the tables around them, that need something serving.
    """
    views, shown, offered = _views(trace, focus)

    total_in = sum(s.rows for r, s in trace.shapes.items() if not spec.step(r.step).inputs)
    published = [r for r in trace.shapes if not spec.consumers(r)]
    order = spec.order()
    details = "".join(_step_detail(spec, trace, step_id) for step_id in order)
    # One name for the step the page opens on and the one panel it ships, because they have to be
    # the same step: named without shipping, the page opens on an empty pane until a fetch answers,
    # and shipped without naming, it carries a panel nothing ever shows.
    opening = order[0] if order else ""
    panel = _step_data(spec, trace, opening, views) if opening else ""
    summary = (
        f"<b>{len(trace.events)}</b> steps &middot; <b>{len(trace.shapes)}</b> ports &middot; "
        f"<b>{total_in}</b> rows in &middot; <b>{len(published)}</b> ports nothing reads further"
    )
    buttons = "".join(f'<button data-pick="{_e(k)}">{_e("everyone" if k == "all" else k)}</button>' for k in views)
    capped = "" if shown == offered else f'<span class="pmore">showing the first {shown} of {offered:,}</span>'
    picker = f'<div class="picker"><span class="plabel">{_e(focus.label)}</span>{buttons}{capped}</div>'
    return TEMPLATE.format(
        name=_e(spec.name),
        summary=summary,
        canvas=_canvas(spec, trace),
        picker=picker,
        details=details,
        panel=panel,
        first=_e(opening),
        index=_index(spec, trace),
    )


TEMPLATE = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{name}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#f4f4f1; --surface:#fff; --ink:#16181b; --muted:#5c6470; --line:#dfdfd9;
  --accent:#0d6e70; --accent-soft:#e5f0f0; --warn:#8a4b1f; --warn-soft:#f6ece3;
  --sans:'Archivo','Segoe UI',system-ui,sans-serif; --mono:'IBM Plex Mono',ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
  --ground:#131618; --surface:#191d20; --ink:#e8eae6; --muted:#98a0a8; --line:#2a2f34;
  --accent:#46b3b0; --accent-soft:#12312f; --warn:#c98a57; --warn-soft:#2a1f16; }} }}
:root[data-theme="dark"] {{
  --ground:#131618; --surface:#191d20; --ink:#e8eae6; --muted:#98a0a8; --line:#2a2f34;
  --accent:#46b3b0; --accent-soft:#12312f; --warn:#c98a57; --warn-soft:#2a1f16; }}
* {{ box-sizing:border-box; }}

/* Scrollbars, in both spellings: the standard properties, which Firefox honours, and the WebKit
   pseudo-elements, which Chromium and Safari honour. Neither alone covers a browser a reader might
   actually be using, and the two cannot express quite the same thing -- so the colours are shared
   through the same tokens as everything else, and the sizes are stated twice on purpose.

   Deliberately quiet rather than invisible: a pane whose content runs past its edge has to say so,
   and on a laptop screen with four scrollable regions on it at once, four loud bars are the loudest
   thing on the page. They darken on hover, which is when a reader is looking for one. */
* {{ scrollbar-width:thin; scrollbar-color:var(--line) transparent; }}
*:hover {{ scrollbar-color:var(--muted) transparent; }}
::-webkit-scrollbar {{ width:7px; height:7px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:var(--line); border-radius:4px; }}
::-webkit-scrollbar-thumb:hover {{ background:var(--muted); }}
::-webkit-scrollbar-corner {{ background:transparent; }}
body {{ background:var(--ground); color:var(--ink); font-family:var(--sans); line-height:1.55; }}
code, pre, .data {{ font-family:var(--mono); }}
html, body {{ height:100%; }}
.wrap {{ height:100vh; display:flex; flex-direction:column; padding:0 .6rem .6rem;
  --split-x:52%; --split-y:52%; }}

.body {{ flex:1 1 auto; min-height:0; display:flex; gap:.6rem; }}
.body > .panes {{ flex:1 1 auto; }}

.index {{ flex:0 0 13rem; min-height:0; overflow:auto; background:var(--surface);
  border:1px solid var(--line); border-radius:9px; }}
.wrap.railed .index {{ display:none; }}
.index header {{ position:sticky; top:0; background:var(--surface); display:flex;
  justify-content:space-between; align-items:baseline; padding:.55rem .7rem;
  border-bottom:1px solid var(--line); font:600 .68rem var(--sans);
  text-transform:uppercase; letter-spacing:.11em; color:var(--muted); }}
.index ul {{ list-style:none; margin:0; padding:.3rem 0 .6rem; }}
.index .lanehead {{ font:600 .6rem var(--sans); text-transform:uppercase; letter-spacing:.1em;
  color:var(--muted); padding:.7rem .7rem .25rem; }}
.index button {{ width:100%; text-align:left; background:none; border:0; cursor:pointer;
  padding:.24rem .7rem; display:flex; flex-direction:column; gap:.05rem; border-left:2px solid transparent; }}
.index button span {{ font:500 .76rem var(--mono); color:var(--ink); }}
.index button em {{ font:400 .66rem var(--sans); font-style:normal; color:var(--muted); }}
.index button:hover {{ background:var(--accent-soft); }}
.index button[aria-current="true"] {{ background:var(--accent-soft); border-left-color:var(--accent); }}
.index button[aria-current="true"] span {{ color:var(--accent); font-weight:600; }}
.index button.flag span::after {{ content:' !'; color:var(--warn); font-weight:700; }}
.rail {{ background:none; border:1px solid var(--line); border-radius:6px; cursor:pointer;
  font-size:.9rem; line-height:1; padding:.3rem .45rem; color:var(--muted); }}
.rail:hover {{ border-color:var(--accent); color:var(--accent); }}

.wrap.no-flow .flowpane, .wrap.no-flow .grip-v {{ display:none; }}
.wrap.no-detail .detailpane, .wrap.no-detail .grip-v {{ display:none; }}
.wrap.no-data .datapane, .wrap.no-data .grip-h {{ display:none; }}
.wrap.no-flow .panes {{ grid-template-columns:0 0 minmax(0,1fr); }}
.wrap.no-detail .panes {{ grid-template-columns:minmax(0,1fr) 0 0; }}
.wrap.no-data .panes {{ grid-template-rows:minmax(0,1fr) 0 0; }}

.folds {{ display:flex; gap:0; margin-left:auto; border:1px solid var(--line);
  border-radius:6px; overflow:hidden; }}
.folds button {{ font:600 .68rem var(--sans); color:var(--muted); background:var(--surface);
  border:0; padding:.22rem .6rem; cursor:pointer; letter-spacing:.04em; }}
.folds button + button {{ border-left:1px solid var(--line); }}
.folds button[aria-pressed="true"] {{ background:var(--accent); color:var(--surface); }}
.folds button:hover:not([aria-pressed="true"]) {{ background:var(--accent-soft); color:var(--accent); }}

.panes {{ min-height:0; display:grid; gap:0;
  grid-template-columns:var(--split-x) 7px minmax(0,1fr);
  grid-template-rows:var(--split-y) 7px minmax(0,1fr); }}
.pane {{ min-width:0; min-height:0; overflow:auto; background:var(--surface);
  border:1px solid var(--line); border-radius:9px; }}
.flowpane {{ grid-area:1 / 1; padding:.4rem; display:flex; flex-direction:column; overflow:hidden; }}
.flowpane .canvas {{ flex:1 1 auto; min-height:0; overflow:auto; }}
.bar {{ display:flex; align-items:center; gap:.25rem; padding:0 .2rem .35rem; flex:0 0 auto; }}
.zoom {{ display:flex; align-items:center; gap:.25rem; }}
.bar .tclear {{ font:600 .7rem var(--sans); color:var(--muted); background:var(--surface);
  border:1px solid var(--line); border-radius:5px; padding:.1rem .42rem; cursor:pointer;
  min-width:1.7rem; margin-left:.4rem; }}
.bar .tclear:hover {{ border-color:var(--accent); color:var(--accent); }}
.zoom button {{ font:600 .7rem var(--sans); color:var(--muted); background:var(--surface);
  border:1px solid var(--line); border-radius:5px; padding:.1rem .42rem; cursor:pointer;
  min-width:1.7rem; }}
.zoom button:hover {{ border-color:var(--accent); color:var(--accent); }}
.zoom .level {{ font:500 .68rem var(--mono); color:var(--muted); min-width:2.9rem;
  text-align:center; font-variant-numeric:tabular-nums; }}
.detailpane {{ grid-area:1 / 3; padding:.8rem 1rem; }}
.datapane {{ grid-area:3 / 1 / 4 / 4; padding:.7rem .9rem; }}

/* The grips. Wider than they look, because a 1px target is a target nobody hits. */
.grip {{ position:relative; background:transparent; }}
.grip::after {{ content:''; position:absolute; inset:0; margin:auto; background:var(--line);
  border-radius:2px; }}
.grip:hover::after, .grip.dragging::after {{ background:var(--accent); }}
.grip-v {{ grid-area:1 / 2; cursor:col-resize; }}
.grip-v::after {{ width:3px; height:38px; }}
.grip-h {{ grid-area:2 / 1 / 3 / 4; cursor:row-resize; }}
.grip-h::after {{ height:3px; width:38px; }}
body.resizing {{ cursor:inherit; user-select:none; }}
header.top {{ display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap;
  padding:.45rem .2rem .45rem; flex:0 0 auto; }}
header.top h1 {{ font-size:1.15rem; font-weight:700; letter-spacing:-.01em; margin:0; }}
header.top p {{ margin:0; color:var(--muted); }}
header.top b {{ color:var(--accent); }}

.canvas {{ background:transparent; border:0; padding:0; margin:0; }}
.band {{ fill:var(--accent-soft); opacity:.4; }}
/* Drawn last, so the same halo the edge captions use keeps it legible over whatever it lands on
   -- a container header reaching the first column is the case that made this necessary. */
.lane {{ fill:var(--muted); font:600 9px var(--sans); letter-spacing:.11em; text-transform:uppercase;
  paint-order:stroke; stroke:var(--surface); stroke-width:3px; stroke-linejoin:round; }}
/* A connection is a group, and the group carries the line's colour and weight rather than the paths
   doing it themselves: a state class on the group then restyles the curve, its hit target and its
   caption in one place, and they can never light apart. `.wire` declares nothing of its own so that
   stroke and width inherit -- a declaration on the path would beat anything inherited from the
   group, and the highlight would silently do nothing. */
.edge {{ fill:none; stroke:var(--muted); stroke-width:1.4; opacity:.75; }}
/* Hidden until its connection is lit. 765 captions at rest is a canvas nobody can read, and
   most of them answer a question nobody is asking; the row count matters at the moment a
   reader is following one line, and then it matters a lot. */
.elab {{ fill:var(--muted); font:500 9px var(--mono); text-anchor:middle; visibility:hidden; }}
.edge.direct .elab, .edge.upstream .elab, .edge.downstream .elab {{ visibility:visible; }}
.hit {{ fill:none; stroke:transparent; stroke-width:14; pointer-events:stroke; cursor:pointer; }}

/* Four states, not two. At 672 steps, lighting the lines that touch the chosen tool answers "what
   is it wired to" and leaves the actual question -- where does this run, three tools further on --
   as unreadable as before. And the two directions are drawn differently because which way the data
   runs is what a reader follows a line to find out; three states in one colour light the same
   pixels as one state would. */
.edge.direct {{ stroke:var(--ink); stroke-width:3.2; opacity:1; }}
.edge.upstream {{ stroke:var(--accent); stroke-width:2.2; opacity:.95; }}
.edge.downstream {{ stroke:var(--warn); stroke-width:2.2; opacity:.95; }}
.edge.dim {{ opacity:.1; }}
/* A lit line is drawn thick and in the same ink the caption is written in, so a caption sitting
   on top of it disappears into it -- reported from the real canvas as numbers that could not be
   made out. The fix is contrast, not weight: the caption keeps its ordinary weight and is given
   a halo of the surface colour, painted UNDER the glyphs (paint-order), so it punches out of
   whatever runs beneath it. Bolding it made it heavier against a line of the same colour, which
   is the one change that could not help. */
.edge.direct .elab, .edge.upstream .elab, .edge.downstream .elab {{
  fill:var(--ink); paint-order:stroke; stroke:var(--surface); stroke-width:3.5px;
  stroke-linejoin:round; }}
.node rect {{ fill:var(--surface); stroke:var(--accent); stroke-width:1.5; }}
/* Faded, not hidden: the shape of the flow is the reader's map of where the lit chain sits in it. */
.node.dim {{ opacity:.22; }}
.node.direct rect {{ stroke:var(--ink); stroke-width:3; }}
.node.upstream rect {{ stroke:var(--accent); stroke-width:2.4; }}
.node.downstream rect {{ stroke:var(--warn); stroke-width:2.4; }}
/* Warning and error are written last on purpose, so that at equal specificity they beat a trace
   state: what the run had to say about a step is a fact about the data, and a highlight is a
   question the reader asked a second ago. A warning hidden under a highlight is the swap in the
   wrong direction, and the reader has no way of knowing anything was covered up.
   The cost is that a warning node in the upstream chain wears the downstream colour, both being
   --warn. That is a collision worth carrying rather than solving with a third hue: direction is
   carried by the CONNECTIONS, which are what a reader follows to ask the question in the first
   place -- every lit line around a node in the upstream chain is teal, every one around a node in
   the downstream chain is rust -- and the node's own stroke only confirms what they already say.
   The stroke width, which a warning leaves alone, still says the node is in the chain at all. */
.node.warning rect {{ stroke:var(--warn); }}
.node.error rect {{ stroke:var(--warn); stroke-width:2.5; }}
.node:hover rect {{ fill:var(--accent-soft); }}
.node.chosen rect {{ fill:var(--accent-soft); stroke-width:3; }}
.node {{ cursor:pointer; }}

/* Containers. Open, the outline is a dashed hint round steps a reader is still reading one at a
   time -- a solid box at the weight of a node would read as another tool. It draws no fill at all,
   because the connections are written before it and a filled rectangle would paint over every line
   that crosses the section.

   Closed, the same container is the header bar alone: its steps are put away, the outline goes with
   them, and the bar takes a fill because it now stands for what is underneath it rather than round
   it. The sentence and the counts are drawn in the header, which is inside both rectangles, so they
   survive whichever state the container is in -- a box with nothing written on it is a section of
   the flow that has become unreadable, and that is the failure this arrangement exists to avoid. */
.gbox {{ fill:none; stroke:var(--accent); stroke-width:1.2; stroke-dasharray:5 4; opacity:.7; }}
.gpill {{ fill:var(--accent-soft); stroke:var(--accent); stroke-width:1.2; }}
.gtitle {{ fill:var(--ink); font:600 12px var(--sans); }}
.gcount {{ fill:var(--muted); font:500 10px var(--mono); }}
.gctl {{ cursor:pointer; }}
.gctl rect {{ fill:var(--surface); stroke:var(--accent); stroke-width:1.2; }}
.gctl text {{ fill:var(--accent); font:700 13px var(--sans); text-anchor:middle; }}
/* Which of the two states is drawn, said in the sheet rather than by moving anything on the canvas:
   the steps are children of the container, so closing one is a single class on a single element. */
.group .gpill, .group .gshut {{ display:none; }}
.group.collapsed .gbox, .group.collapsed .node, .group.collapsed .gopen {{ display:none; }}
.group.collapsed .gpill, .group.collapsed .gshut {{ display:block; }}
/* A connection with both ends inside a closed container is that container's own plumbing, and goes
   away with it. One with an end outside is how the rest of the flow reaches the section: it stays,
   re-pointed at the bar. */
.edge.gone {{ display:none; }}
.nid {{ fill:var(--ink); font:600 12px var(--mono); }}
.ntool {{ fill:var(--muted); font:500 10px var(--sans); }}
.anchor {{ stroke:var(--surface); stroke-width:1.5; }}
.anchor.in {{ fill:var(--surface); stroke:var(--accent); }}
.anchor.out {{ fill:var(--accent); }}

.step {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:1.2rem 1.4rem; margin-bottom:1.1rem; scroll-margin-top:1rem; }}
.step.warning {{ border-left:3px solid var(--warn); }}
.pmore {{ color:var(--muted); font-size:.8rem; margin-left:.5rem; }}
.step h3 {{ margin:0; font-size:1.05rem; display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap; }}
.tool {{ color:var(--accent); font:600 .72rem var(--mono); }}
.where {{ color:var(--muted); font-size:.66rem; text-transform:uppercase; letter-spacing:.09em; }}
.why {{ margin:.5rem 0 0; color:var(--muted); max-width:70ch; }}
.cols {{ display:grid; grid-template-columns:minmax(220px,300px) minmax(0,1fr); gap:1.4rem; margin-top:1rem; }}
@media (max-width:820px) {{ .cols {{ grid-template-columns:1fr; }} }}
.cols h5 {{ font-size:.62rem; text-transform:uppercase; letter-spacing:.11em; color:var(--muted);
  margin:0 0 .35rem; font-weight:600; }}
pre {{ background:var(--ground); border:1px solid var(--line); border-radius:6px; padding:.6rem .7rem;
  font-size:.76rem; overflow-x:auto; margin:0 0 .9rem; }}
ul.feeds {{ list-style:none; margin:0 0 .9rem; padding:0; font-size:.78rem; color:var(--muted); }}
.timing {{ margin:0; color:var(--muted); font-size:.72rem; }}
.picker {{ display:flex; gap:.4rem; flex-wrap:wrap; align-items:center; margin:0 0 1.5rem; }}
.plabel {{ font-size:.62rem; text-transform:uppercase; letter-spacing:.11em; color:var(--muted);
  font-weight:600; margin-right:.3rem; }}
.picker button {{ font:inherit; font-size:.8rem; padding:.3rem .7rem; border:1px solid var(--line);
  background:var(--surface); color:var(--ink); border-radius:999px; cursor:pointer; }}
.picker button:hover {{ border-color:var(--accent); }}
.picker button[aria-pressed="true"] {{ background:var(--accent); border-color:var(--accent);
  color:var(--surface); font-weight:600; }}
.port {{ margin-bottom:1rem; }}
.pager {{ display:flex; gap:.4rem; align-items:center; }}
.pager button {{ font:500 .72rem var(--sans); color:var(--ink); background:var(--surface);
  border:1px solid var(--line); border-radius:5px; padding:.14rem .55rem; cursor:pointer; }}
.pager button:hover:not(:disabled) {{ border-color:var(--accent); color:var(--accent); }}
.pager button:disabled {{ opacity:.4; cursor:default; }}
table.data input.flt {{ width:100%; min-width:5.5rem; box-sizing:border-box; font:400 .72rem var(--mono);
  color:var(--ink); background:var(--ground); border:1px solid var(--line); border-radius:4px;
  padding:.12rem .3rem; }}
table.data input.flt:focus {{ outline:2px solid var(--accent); outline-offset:-1px; }}
table.data tr.flts th {{ padding:.2rem .5rem .4rem; border-bottom:1px solid var(--line); }}
table.data tbody tr.hide {{ display:none; }}
/* The status column. Narrow, because it is one mark per row and the data is what the reader came
   for; centred, because a column of single marks read down the edge of a table is how a reader
   finds the flagged rows without reading any of them. */
table.data th.lvl, table.data td.lvl {{ width:1%; text-align:center; white-space:nowrap; }}
table.data th.lvl select {{ font:400 .72rem var(--mono); color:var(--ink); background:var(--ground);
  border:1px solid var(--line); border-radius:4px; padding:.12rem .3rem; }}
/* Three levels, drawn three ways, and never by the glyph alone: a tick and a cross at 12px on a
   dense table are two marks of the same weight in the same colour, and the row that matters is the
   one a reader has to be able to pick out while scrolling. The cell also carries the level and its
   reason as text a browser can announce, which is the same distinction made for a reader who is not
   looking at the screen at all. */
table.data td.lvl.ok {{ color:var(--muted); }}
table.data td.lvl.warning {{ color:var(--warn); font-weight:700; background:var(--warn-soft); }}
table.data td.lvl.error {{ color:var(--surface); font-weight:700; background:var(--warn); }}
a.up {{ font:600 1.05rem var(--mono); color:var(--ink); text-decoration:none;
  border-bottom:2px solid var(--accent-soft); }}
a.up::before {{ content:'↑'; color:var(--accent); font-size:.8rem; margin-right:.3rem;
  vertical-align:.06em; }}
a.up:hover {{ border-bottom-color:var(--accent); }}
a.up:hover::before {{ color:var(--accent); }}
.canvas {{ scroll-margin-top:1rem; }}
.node {{ scroll-margin-top:6rem; }}
.flip {{ display:inline-flex; gap:0; margin-bottom:.6rem; border:1px solid var(--line);
  border-radius:6px; overflow:hidden; }}
.flip button {{ font:600 .72rem var(--sans); color:var(--muted); background:var(--surface);
  border:0; padding:.2rem .7rem; cursor:pointer; letter-spacing:.04em; }}
.flip button + button {{ border-left:1px solid var(--line); }}
.flip button[aria-pressed="true"] {{ background:var(--accent); color:var(--surface); }}
.flip button:hover:not([aria-pressed="true"]) {{ background:var(--accent-soft); color:var(--accent); }}
ul.notes {{ list-style:none; margin:.55rem 0 0; padding:0; }}
ul.notes li {{ background:var(--warn-soft); color:var(--warn); border-left:3px solid var(--warn);
  padding:.3rem .6rem; border-radius:0 5px 5px 0; font-size:.82rem; margin-bottom:.3rem; }}
ul.notes li::before {{ content:'! '; font-weight:700; }}
.cols > * {{ min-width:0; }}
.cfg > span:last-child {{ min-width:0; flex:1 1 auto; }}
.cfg code, table.cfgtable code {{ overflow-wrap:anywhere; white-space:pre-wrap; }}
.side {{ margin-top:.2rem; }}
.cfg {{ display:flex; gap:.6rem; align-items:baseline; padding:.22rem 0;
  border-bottom:1px dotted var(--line); }}
.cfg:last-child {{ border-bottom:none; }}
.ck {{ color:var(--muted); font:500 .72rem var(--sans); min-width:7.5rem; flex:0 0 auto;
  letter-spacing:.03em; }}
.cfg code {{ font-size:.78rem; color:var(--ink); background:var(--accent-soft);
  padding:.06rem .3rem; border-radius:4px; }}
table.cfgtable {{ border-collapse:collapse; font-size:.76rem; }}
table.cfgtable th {{ text-align:left; font:600 .68rem var(--sans); color:var(--muted);
  text-transform:uppercase; letter-spacing:.07em; padding:.2rem .6rem .2rem 0; }}
table.cfgtable td {{ padding:.16rem .6rem .16rem 0; vertical-align:top; }}
.port summary {{ margin:0; font-size:.8rem; font-weight:500; display:flex; gap:.5rem;
  align-items:center; cursor:pointer; padding:.3rem .1rem; border-radius:5px; list-style:none; }}
.port summary::-webkit-details-marker {{ display:none; }}
.port summary::before {{ content:'▸'; color:var(--muted); font-size:.7rem;
  transition:transform .12s ease; display:inline-block; }}
.port details[open] > summary::before {{ transform:rotate(90deg); }}
.port summary:hover {{ background:var(--accent-soft); }}
.port details > *:not(summary) {{ margin-top:.4rem; }}
.port h4 {{ margin:0 0 .35rem; font-size:.8rem; font-weight:500; display:flex; gap:.5rem;
  align-items:center; color:var(--muted); }}
.pill {{ background:var(--accent); color:var(--surface); font:600 .64rem var(--mono);
  padding:.1rem .45rem; border-radius:3px; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; }}
table.data {{ border-collapse:collapse; font-size:.76rem; width:100%; font-variant-numeric:tabular-nums; }}
table.data th {{ text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--line);
  background:var(--ground); font-weight:600; white-space:nowrap; }}
table.data th em {{ display:block; font-style:normal; font-weight:400; font-size:.85em; color:var(--muted); }}
table.data td {{ padding:.26rem .5rem; border-bottom:1px solid var(--line); white-space:nowrap; }}
table.data tr:last-child td {{ border-bottom:none; }}
.nul {{ color:var(--muted); font-style:italic; }}
.more {{ margin:.3rem 0 0; color:var(--muted); font-size:.74rem; font-style:italic; }}
</style>
<div class="wrap" data-first="{first}">
  <header class="top"><button class="rail" title="show or hide the list of steps">&#9776;</button>
    <h1>{name}</h1><p>{summary}</p>{picker}
    <div class="folds"><button data-fold="flow" aria-pressed="true">flow</button>
      <button data-fold="detail" aria-pressed="true">config</button>
      <button data-fold="data" aria-pressed="true">data</button></div></header>
  <div class="body">
  {index}
  <div class="panes">
    <div class="pane flowpane">
      <div class="bar"><div class="zoom"><button data-zoom="out" title="zoom out">&minus;</button>
        <span class="level">100%</span>
        <button data-zoom="in" title="zoom in">+</button>
        <button data-zoom="fit" title="fit the whole flow">fit</button>
        <button data-zoom="reset" title="back to actual size">1:1</button></div>
        <button class="tclear" data-trace="clear"
          title="clear the trace and light the whole flow again">&#9678;</button></div>
      {canvas}</div>
    <div class="grip grip-v" title="drag to widen either side"></div>
    <div class="pane detailpane">{details}</div>
    <div class="grip grip-h" title="drag to raise or lower the data"></div>
    <div class="pane datapane">{panel}</div>
  </div>
  </div>
</div>
<script>
const wrap = document.querySelector('.wrap');

// The step being read, kept so that a panel landing after the reader has clicked elsewhere is not
// drawn over the step they moved on to: a fetch outlives the click that started it, and one step's
// rows under another step's name is the one failure here nobody can see.
let chosen = null;

// One step at a time. The flow is the index: click a node and that step's configuration and its
// data arrive beside it, with nothing moving and nothing to scroll past. A page of fifty-three
// cards is a page nobody reads.
//
// Asynchronous because only half of it is in the document. The configuration, the canvas and the
// step list are here and move at once; the step's data panel is fetched, and what it needs doing to
// it -- showing it, and putting it on the entity the picker is on -- can only be done once it has
// landed, which is what the tail of this function is.
async function choose(stepId, readerAsked = true) {{
  chosen = stepId;
  // .stepdata as well as .detail, and it is not the incoming panel this reaches -- that one is not
  // here yet. It is the panel of the step the reader has just left, which goes at the same moment
  // the rest of the selection moves. Narrow this to .detail alone and the previous step's rows stay
  // on the screen under the new step's name for as long as the fetch takes.
  for (const section of wrap.querySelectorAll('.detail, .stepdata'))
    section.hidden = section.dataset.step !== stepId;
  for (const node of wrap.querySelectorAll('.node'))
    node.classList.toggle('chosen', node.id === 'node-' + stepId);
  for (const line of wrap.querySelectorAll('.index button'))
    line.setAttribute('aria-current', String(line.dataset.goto === stepId));
  // The trace hangs off choosing rather than off clicking a node, because a node is only one of the
  // two ways a step gets chosen -- the step list is the other, and is how a reader finds a step they
  // could not pick out of six hundred in the first place. Bolted onto the node's handler, the list
  // would quietly do less than the canvas.
  //
  // But only when a reader asked for it, which is what the parameter is for and the only reason it
  // exists. The page opens by choosing its first step, and tracing on that call would fade most of
  // the canvas before the reader has put any question to it. Measured on the real flow, choosing a
  // late step lights 427 nodes of 672: the opening view would be two thirds lit and one third dark,
  // which reads as a drawing that failed to finish rather than as an answer. A reader arriving at
  // 672 tools needs the shape of the thing first; dimming it is an answer to a question nobody put.
  //
  // So the one call made at load passes false and every selection a reader makes -- a node, a line
  // in the step list -- takes the default. Do not collapse this back to an unconditional
  // traceFrom(): that parameter is the whole distinction between a selection the page made for the
  // reader and one the reader made for themselves, and only the second is a question worth
  // answering by putting six hundred tools in the dark.
  if (readerAsked) traceFrom(stepId);

  await ensurePanel(stepId);
  // The panel arrives hidden, which is how it was rendered, and the pass above ran before it
  // existed. So the showing is done again now that it is here -- awaited, because the same pass run
  // before the panel lands finds nothing to show and leaves the reader in front of an empty pane
  // with the data sitting in the document beside it.
  //
  // Unless the reader has chosen something else while this was in flight, in which case the newer
  // selection is the one on the screen and re-running this one would put this step's rows under
  // that step's name.
  if (chosen !== stepId) return;
  for (const section of wrap.querySelectorAll('.stepdata'))
    section.hidden = section.dataset.step !== stepId;
  // A panel is rendered with every one of its views hidden, and the picker's buttons are in the
  // page while the views are in the panel: re-applying the entity the reader is on is the only
  // thing that brings the two together, and without it the panel opens on nothing at all.
  //
  // By calling pick() rather than by showing this panel's view here, so that one function decides
  // what a chosen entity looks like. A second copy of that rule is one that drifts, and the drift
  // shows as an open panel with no rows in it, which reads as a step that produced none.
  pick(picked);
  const open = wrap.querySelector(`.stepdata[data-step="${{stepId}}"] details[open] table.data[data-lazy]`);
  if (open) {{ delete open.dataset.lazy; live(open); }}
}}

// The one notice standing in for a panel that could not be had, made once and then moved: a reader
// clicking through five steps with nothing serving them is told once, where they are looking,
// rather than handed five copies of the same sentence down the pane.
let notice = null;

// The panel of the step being opened, fetched as it is opened and left in the page afterwards.
// Measured on the real flow, carrying all 672 of them was 7.76 MB of an 8.98 MB page and 24,937
// search boxes for the browser to lay out before it could paint -- for the 671 steps the reader was
// not looking at. Named for what it guarantees rather than for what it hands back, which is
// nothing at all: the caller waits for it and then finds the panel in the page for itself.
async function ensurePanel(stepId) {{
  const pane = wrap.querySelector('.datapane');
  if (notice) notice.hidden = true;
  // The page in the document is the cache: a section already answering to this step is the one the
  // service would send back. Without the return, a reader clicking between two steps re-fetches
  // both every time and the second copy lands beside the first, so the pane holds two panels for
  // one step and shows whichever the selection above reaches first.
  if (wrap.querySelector(`.stepdata[data-step="${{stepId}}"]`)) return;
  try {{
    const answer = await fetch('api/panel?step=' + encodeURIComponent(stepId));
    if (!answer.ok) throw new Error('the service answered ' + answer.status);
    const markup = await answer.text();
    pane.insertAdjacentHTML('beforeend', markup);
  }} catch (err) {{
    // A refusal, a dead socket, or the page opened as a plain file with nothing serving it. Said
    // once, where the panel would have been, because that is where the reader is looking -- a line
    // in the console is a message to whoever wrote the page, and the reader is somebody looking at
    // a figure.
    //
    // Deliberately NOT written as a .stepdata section: the cache above would then find it, this
    // step would never be asked for again, and a reader coming back once the service is answering
    // would be locked out of it with nothing on the page to say why.
    if (!notice) {{
      notice = document.createElement('p');
      notice.className = 'more';
      pane.append(notice);
    }}
    notice.hidden = false;
    notice.textContent = 'this step is drawn by the preview service, which is not answering';
  }}
}}

// Following one connection across a canvas nobody can read by eye.
//
// The flow this page is really for draws 672 steps and some 900 connections, and at that size the
// canvas answers "what shape is this" and nothing else: a 1.4px line runs behind six other tools
// and a reader cannot say where it comes out. Lighting only what touches the chosen step does not
// fix that -- it shows what a reader could already see. So the chain is walked to its ends in both
// directions, and everything else is faded, which is the half that makes the lit part visible.
let WIRING = null;

// Where the drawing puts a connection back when it stops being lit: before the first step box,
// which is where every connection starts out. Found on first use rather than at load, because
// `flow` is declared further down and reading it here would be reading it before it exists.
//
// A container counts as a step box here, and has to: its steps are drawn inside it, so the first
// `.node` on a canvas with containers is a grandchild of the drawing rather than a child of it, and
// insertBefore against a node that is not its own child throws. The containers are written before
// the loose steps, so the first of these is a child of the drawing either way.
let FIRST_NODE;
function firstNode() {{
  if (FIRST_NODE === undefined) FIRST_NODE = flow ? flow.querySelector('.group, .node') : null;
  return FIRST_NODE;
}}

function paint(el, state) {{
  el.classList.remove('direct', 'upstream', 'downstream', 'dim');
  el.classList.add(state);
  // A lit connection is lifted above the steps, because its caption sits at the midpoint and a long
  // connection's midpoint lands on whatever step happens to sit between its two ends -- which draws
  // after it and covers it. Lifting the whole group keeps the line and its caption together, and
  // only ever applies to the handful a reader is actually following.
  if (el.classList.contains('edge') && flow) {{
    if (state === 'dim') {{
      const at = firstNode();
      if (at && el.nextElementSibling !== at) flow.insertBefore(el, at);
    }} else {{
      flow.appendChild(el);
    }}
  }}
}}

// Where every connection runs, once some container has been closed.
//
// Run over the whole drawing rather than over the container that just moved, because the two ends of
// a connection can sit in two different containers: answering for one of them alone would leave a
// line attached to a box a reader shut a moment ago and has since re-opened. Every connection is
// asked the same two questions -- is the container it starts in closed, is the one it lands in
// closed -- and is drawn from the answers, so the state of the canvas is the state of the canvas
// whatever order the containers were clicked in.
function repointGroups() {{
  // Where each closed container's own bar stands, read off the drawing. Read rather than remembered:
  // the position is already stated in the markup, and a second copy of it drifts.
  const shut = new Map();
  for (const box of wrap.querySelectorAll('[data-group].collapsed')) {{
    const bar = box.querySelector('rect.gpill');
    shut.set(box.dataset.group, {{
      x: Number(bar.getAttribute('x')), y: Number(bar.getAttribute('y')),
      w: Number(bar.getAttribute('width')), h: Number(bar.getAttribute('height')),
    }});
  }}
  for (const edge of wrap.querySelectorAll('.edge')) {{
    const fromGroup = edge.dataset.fromGroup, toGroup = edge.dataset.toGroup;
    const a = shut.get(fromGroup), b = shut.get(toGroup);
    // Both ends inside one closed container: the connection is that container's own plumbing and is
    // put away with it. Asked of both ends together, because asking about one end alone hides every
    // crossing connection too -- the section takes its inputs and its outputs with it and the flow
    // reads as stopping dead at a box.
    edge.classList.toggle('gone', Boolean(a && b && fromGroup === toGroup));
    // An end in a closed container attaches to that container's bar; an end that is still drawn goes
    // back to the coordinate it was drawn at -- the one the page wrote down for it, which is why it
    // was written down. Re-measuring the node instead would measure something that is display:none,
    // which measures as nothing, and working it out from the node's rectangle would mean knowing
    // here what a node's width and height are.
    const x1 = a ? a.x + a.w : Number(edge.dataset.sx);
    const y1 = a ? a.y + a.h / 2 : Number(edge.dataset.sy);
    const x2 = b ? b.x : Number(edge.dataset.tx);
    const y2 = b ? b.y + b.h / 2 : Number(edge.dataset.ty);
    const mid = (x1 + x2) / 2;
    const d = 'M' + x1 + ',' + y1 + ' C' + mid + ',' + y1 + ' ' + mid + ',' + y2 + ' ' + x2 + ',' + y2;
    // The visible line and its wide transparent twin, in one pass and off one string. Move one and
    // not the other and the target a hand aims at is no longer on the line, which lights a
    // neighbouring connection instead -- the very defect the twin was added to fix.
    for (const path of edge.querySelectorAll('path')) path.setAttribute('d', d);
    const label = edge.querySelector('text');
    if (label) {{ label.setAttribute('x', mid); label.setAttribute('y', (y1 + y2) / 2 - 6); }}
  }}
}}

// Which step's chain is lit, or null. Held here rather than read back off the drawing, because
// "is this step the one already traced" is a question about what the reader last asked for, and a
// class on a node answers a different one -- `chosen` stays on through a clear.
let TRACED = null;

function traceFrom(stepId) {{
  if (!WIRING) {{
    // Who feeds whom, read off the drawing once and then kept. A click touches every connection on
    // the canvas; re-deriving the graph from the DOM each time would make a question about one step
    // cost a pass over the whole flow, twice.
    const groups = [...wrap.querySelectorAll('.edge')];
    const feeds = new Map(), fedBy = new Map();
    for (const edge of groups) {{
      if (!feeds.has(edge.dataset.from)) feeds.set(edge.dataset.from, []);
      if (!fedBy.has(edge.dataset.to)) fedBy.set(edge.dataset.to, []);
      feeds.get(edge.dataset.from).push({{ edge: edge, other: edge.dataset.to }});
      fedBy.get(edge.dataset.to).push({{ edge: edge, other: edge.dataset.from }});
    }}
    WIRING = {{ groups: groups, feeds: feeds, fedBy: fedBy }};
  }}
  const lit = new Map();
  const marked = new Map([[stepId, 'direct']]);
  for (const side of [['upstream', WIRING.fedBy], ['downstream', WIRING.feeds]]) {{
    // A worklist rather than one hop. The visited set is not tidiness: a flow that splits and
    // rejoins -- which every one of these does -- would otherwise be walked once per path back to
    // the fork, and a diamond a few steps deep costs more than the canvas is worth.
    const seen = new Set([stepId]);
    const work = [stepId];
    while (work.length) {{
      const at = work.pop();
      for (const link of side[1].get(at) || []) {{
        // Touching the chosen step is a different fact from being three tools away in the same
        // direction, so it gets its own state: `direct` says what this step is wired to, the two
        // side states say which way the data runs through everything beyond it.
        const state = at === stepId ? 'direct' : side[0];
        lit.set(link.edge, state);
        if (marked.get(link.other) !== 'direct') marked.set(link.other, state);
        if (!seen.has(link.other)) {{ seen.add(link.other); work.push(link.other); }}
      }}
    }}
  }}
  TRACED = stepId;
  for (const edge of WIRING.groups) paint(edge, lit.get(edge) || 'dim');
  for (const node of wrap.querySelectorAll('.node'))
    paint(node, marked.get(node.id.replace(/^node-/, '')) || 'dim');
}}

function clearTrace() {{
  TRACED = null;
  // Every state named out, rather than whichever ones this trace happened to set: a clear that
  // misses one leaves part of the canvas dark with nothing on the page left to say why.
  for (const el of wrap.querySelectorAll('.edge, .node'))
    el.classList.remove('direct', 'upstream', 'downstream', 'dim');
  // And back under the steps, in one pass rather than per connection: a lifted connection left
  // above them would draw over a step box for the rest of the session with nothing lit to explain
  // why. Order among the connections themselves does not matter -- they never overlap meaningfully.
  const at = firstNode();
  if (flow && at)
    for (const edge of wrap.querySelectorAll('.edge')) flow.insertBefore(edge, at);
}}

wrap.addEventListener('click', e => {{
  const rail = e.target.closest('.rail');
  if (rail) {{ wrap.classList.toggle('railed'); return; }}

  const fold = e.target.closest('.folds button');
  if (fold) {{
    const hidden = 'no-' + fold.dataset.fold;
    // The last visible pane will not fold: a window showing nothing has no control left on it that
    // says which pane to bring back, so the reader is stuck with a blank screen and three buttons.
    const showing = ['flow', 'detail', 'data'].filter(n => !wrap.classList.contains('no-' + n));
    if (showing.length === 1 && !wrap.classList.contains(hidden)) return;
    wrap.classList.toggle(hidden);
    fold.setAttribute('aria-pressed', String(!wrap.classList.contains(hidden)));
    // The drawing sizes itself against its pane, so a fold changes what "fit" means.
    if (fold.dataset.fold !== 'flow' && Math.abs(zoom - 1) > 0.001) fitZoom();
    return;
  }}

  // The way back. It sits beside the zoom buttons but outside `.zoom`, because that handler takes
  // every button inside the bar and reads a `data-zoom` off it -- a clear control placed there
  // would fall through to the else and zoom the canvas out on its way to clearing.
  const back = e.target.closest('[data-trace]');
  if (back) {{ clearTrace(); return; }}

  // A container opening and closing. Asked for by attribute rather than by class, so the control can
  // be drawn however it needs to be drawn -- what the page answers is the thing that says which
  // container it belongs to.
  const shutter = e.target.closest('[data-collapse]');
  if (shutter) {{
    const box = wrap.querySelector('#group-' + shutter.dataset.collapse);
    // Toggled, never added. A container that closes and cannot be opened again is a section of the
    // flow the reader has permanently lost, with the page's own control the thing that lost it.
    box.classList.toggle('collapsed');
    repointGroups();
    return;
  }}

  const edge = e.target.closest('.edge');
  if (edge) {{
    // A reader picking one line out of a crowded area is asking where that line goes. Answering a
    // different question -- swapping both panes, scrolling a node into view, re-fetching a port --
    // takes their place away over a click they made to keep it. So this lights the line and its two
    // ends, and the step being read does not move.
    e.preventDefault();
    for (const other of wrap.querySelectorAll('.edge')) paint(other, other === edge ? 'direct' : 'dim');
    for (const node of wrap.querySelectorAll('.node')) paint(node, 'dim');
    for (const end of [edge.dataset.from, edge.dataset.to]) {{
      const at = wrap.querySelector('#node-' + end);
      if (at) paint(at, 'direct');
    }}
    return;
  }}

  // A step chosen from the list is also brought into view on the flow: a reader who could not find
  // it by scanning still needs to know where it sits, or the list answers only half the question.
  const line = e.target.closest('.index button');
  if (line) {{
    choose(line.dataset.goto);
    const found = wrap.querySelector('#node-' + line.dataset.goto);
    if (found) found.scrollIntoView({{ behavior: 'smooth', block: 'center', inline: 'center' }});
    return;
  }}

  const node = e.target.closest('.node');
  if (node) {{
    e.preventDefault();
    const step = node.id.replace(/^node-/, '');
    // Clicking the step whose chain is already lit puts the chain away and leaves the step chosen:
    // a reader who has followed a connection and finished with it wants the canvas back, not a
    // different step open. Clicking a DIFFERENT step traces that one, which is the ordinary case.
    if (TRACED === step) {{ clearTrace(); TRACED = null; }}
    // Written out again rather than passing `step`: a case pins that what reaches choose() is the
    // step's own id and not the element's, and it reads the call to know. The guard above is the
    // same expression for the same reason.
    else {{ choose(node.id.replace(/^node-/, '')); }}
    return;
  }}
  // The canvas itself, between the steps. Nothing there is a thing to select, so a click on it can
  // only mean "put this away" -- which is where a reader's hand goes first, ahead of finding a
  // control. Only when something is lit, so an idle click does not fight the pan.
  if (TRACED !== null && e.target.closest('.canvas') && !e.target.closest('.group')) {{
    clearTrace();
    TRACED = null;
  }}
}});

// The two grips. Sizes live in CSS variables on the wrapper, so the layout is one grid and the
// drag is two numbers -- nothing is measured, positioned or re-flowed by hand.
for (const grip of wrap.querySelectorAll('.grip')) {{
  grip.addEventListener('pointerdown', down => {{
    const vertical = grip.classList.contains('grip-v');
    grip.setPointerCapture(down.pointerId);
    grip.classList.add('dragging');
    document.body.classList.add('resizing');

    const move = at => {{
      const box = wrap.querySelector('.panes').getBoundingClientRect();
      const share = vertical
        ? (at.clientX - box.left) / box.width
        : (at.clientY - box.top) / box.height;
      // Kept away from both edges: a pane dragged to nothing is a pane nobody can drag back.
      const bounded = Math.min(0.85, Math.max(0.15, share));
      wrap.style.setProperty(vertical ? '--split-x' : '--split-y', (bounded * 100).toFixed(1) + '%');
    }};
    const up = () => {{
      grip.classList.remove('dragging');
      document.body.classList.remove('resizing');
      grip.removeEventListener('pointermove', move);
      grip.removeEventListener('pointerup', up);
    }};
    grip.addEventListener('pointermove', move);
    grip.addEventListener('pointerup', up);
  }});
}}
// Which entity the picker is on, kept because a panel that arrives after it was chosen has to be
// put on the same one: the buttons are in the page and the views are in the panel, so the two only
// meet if the choice is remembered.
let picked = 'all';

function pick(key) {{
  picked = key;
  for (const b of wrap.querySelectorAll('.picker button'))
    b.setAttribute('aria-pressed', String(b.dataset.pick === key));
  for (const v of wrap.querySelectorAll('.view')) v.hidden = v.dataset.for !== key;
}}
// Two modes. With the service answering, a filter runs over the WHOLE port and a window of rows
// is fetched; without it -- the page opened as a plain file -- the same boxes narrow the rows that
// were written into the page. The note under each table says which of the two just happened, so a
// count is never read as covering more than it did.
let LIVE = false;

function boxesOf(table) {{ return [...table.querySelectorAll('tr.flts input.flt')]; }}

// Each box is asked which cell it narrows rather than being counted off in order, because on a
// table with a status column the two differ by one: the boxes still stand one per data column, and
// the cells they read have a mark in front of them. Counting held while the two were the same and
// would have been the same off-by-one the boxes themselves were shifted to avoid -- the box for a
// party narrowing the column to its left. Unreachable today, since a marked table is always a
// served one and a served table carries no rows in the document for this to run over, and written
// this way anyway: that is a fact about two other functions, it is nowhere near this one, and a
// filter that is only correct because of it is a filter waiting for the day a marked table is drawn
// with its rows in it.
function drawnFilter(table) {{
  const want = boxesOf(table)
    .map(b => [Number(b.dataset.c), b.value.trim().toLowerCase()])
    .filter(([, text]) => text);
  let hits = 0;
  for (const tr of table.tBodies[0].rows) {{
    let ok = true;
    for (const [at, text] of want) {{
      const cell = tr.cells[at];
      if (!cell || !cell.textContent.toLowerCase().includes(text)) {{ ok = false; break; }}
    }}
    tr.classList.toggle('hide', !ok);
    if (ok) hits++;
  }}
  const note = table.closest('.tbl').querySelector('.match');
  note.hidden = !want.length;
  note.textContent = want.length ? hits + ' of ' + table.dataset.drawn + ' rows on this page match' : '';
}}

// What a step said about one row, as a cell the reader can see and a browser can read out.
//
// Both, deliberately. A tick, a warning mark and a cross tell somebody looking at the screen which
// rows to open and tell a screen reader nothing at all, and a reader who cannot get from the mark
// back to "held 'Canadian Business Growth Fund', which did not read as Float64" is back to
// distrusting a count -- which is the whole of what this column exists to stop. So the level lands
// as a class the sheet draws three ways, and the reason -- the column that degraded, and what that
// cell held -- lands as the cell's own announced text.
//
// The drawings are written here, one per level, rather than being derived from the level's name: a
// level with nothing drawn for it renders as an empty cell, which reads as a row that is fine.
function mark(m) {{
  const td = document.createElement('td');
  td.className = 'lvl ' + m.level;
  const drawn = {{ ok: '\u2713', warning: '\u25b2', error: '\u2715' }};
  td.textContent = drawn[m.level] || '?';
  td.title = m.level + (m.reason || []).map(r => ' \u2014 ' + r.column + ' ' + r.detail).join(' ');
  td.setAttribute('aria-label', td.title);
  return td;
}}

function cell(v) {{
  const td = document.createElement('td');
  if (v === null) {{
    const s = document.createElement('span');
    s.className = 'nul';
    s.textContent = 'null';
    td.append(s);
  }} else td.textContent = v;
  return td;
}}

async function live(table) {{
  const port = table.dataset.port;
  const qs = new URLSearchParams({{ port, offset: table.dataset.offset, limit: table.dataset.limit }});
  for (const b of boxesOf(table)) if (b.value.trim()) qs.set('f.' + b.dataset.col, b.value.trim());
  // A parameter of its own, not one more `f.`: the boxes narrow a column by substring, and a level
  // asked for that way would match `warning` from `warn` and would mean something else entirely on
  // a port whose data has a column called level. The service answers to this name and refuses
  // anything that is not a published level, which is why the control offers only those.
  const levelPick = table.querySelector('tr.flts select.lvl');
  if (levelPick && levelPick.value) qs.set('level', levelPick.value);
  const note = table.closest('.tbl').querySelector('.match');
  const pager = table.closest('.tbl').querySelector('.pager');
  let answer;
  try {{ const r = await fetch('api/rows?' + qs); answer = await r.json(); if (!r.ok) throw new Error(answer.error); }}
  catch (err) {{ note.hidden = false; note.textContent = String(err.message || err); return; }}

  const body = table.tBodies[0];
  // The mark is taken at the row's own index in the window's marks, because the two sequences the
  // service hands back line up by position and nothing else joins them: a running counter over a
  // body that was not cleared, or a lookup by some value, draws a warning against a row that never
  // had one. And it goes in before the cells, or the reader's first column is the data's first
  // column and the status is somewhere in the middle of the table.
  body.replaceChildren(...answer.rows.map((row, i) => {{
    const tr = document.createElement('tr');
    if (answer.status) tr.append(mark(answer.status[i]));
    tr.append(...row.map(cell));
    return tr;
  }}));

  const from = answer.rows.length ? answer.offset + 1 : 0;
  const to = answer.offset + answer.rows.length;
  const filtered = answer.matched !== answer.total;
  note.hidden = false;
  const span = from + '-' + to + ' of ';
  note.textContent = filtered
    ? span + answer.matched.toLocaleString() + ' matching (of ' + answer.total.toLocaleString() + ')'
    : span + answer.total.toLocaleString() + ' rows';
  pager.hidden = false;
  pager.querySelector('.at').textContent = '';
  pager.querySelector('[data-page=prev]').disabled = answer.offset <= 0;
  pager.querySelector('[data-page=next]').disabled = to >= answer.matched;
}}

function refresh(table) {{
  if (LIVE && table.dataset.port) live(table); else drawnFilter(table);
}}

let typing;
wrap.addEventListener('input', e => {{
  const box = e.target.closest('input.flt');
  if (!box) return;
  const table = box.closest('table');
  table.dataset.offset = 0;
  clearTimeout(typing);
  typing = setTimeout(() => refresh(table), 180);
}});

// Picked rather than typed, so it asks at once: the delay above is there to let somebody finish a
// word, and there is no half-finished choice. Back to the top of the port, like any other
// narrowing -- page three of the whole port is not page three of the flagged rows.
wrap.addEventListener('change', e => {{
  const control = e.target.closest('tr.flts select.lvl');
  if (!control) return;
  const table = control.closest('table');
  table.dataset.offset = 0;
  refresh(table);
}});

wrap.addEventListener('click', e => {{
  const f = e.target.closest('.flip button');
  if (f) {{
    const card = f.closest('.stepdata');
    for (const b of card.querySelectorAll('.flip button'))
      b.setAttribute('aria-pressed', String(b === f));
    for (const side of card.querySelectorAll('.side'))
      side.hidden = side.dataset.side !== f.dataset.side;
    return;
  }}
  const b = e.target.closest('.pager button');
  if (!b) return;
  const table = b.closest('.tbl').querySelector('table');
  const step = parseInt(table.dataset.limit, 10);
  const at = parseInt(table.dataset.offset, 10);
  table.dataset.offset = Math.max(0, b.dataset.page === 'next' ? at + step : at - step);
  refresh(table);
}});

// Zooming is a multiplication on the drawing's own stated size: the viewBox keeps the coordinate
// system, so every node, edge and label scales together and nothing is laid out twice. Bounded at
// both ends -- a flow zoomed to nothing cannot be zoomed back, and one zoomed to ten times its size
// is a single node filling a pane.
//
// The lower bound is small because a real flow is wide: fifty-three steps draw nine thousand pixels
// across, so fitting one into a pane is about eight per cent. Nothing is readable at that size and
// nothing is meant to be -- fit answers "what shape is this", and reading is what the other end of
// the range and the step list are for. A floor that stopped short of fitting would leave the one
// control named after fitting unable to do it.
//
// It opens at actual size rather than fitted, though: a reader arrives to read a step, not to
// survey the shape, and a flow that opens unreadably small has to be zoomed before it can be used
// at all. Fit is one click away for the moment the shape IS the question.
const flow = wrap.querySelector('.canvas svg');
const level = wrap.querySelector('.zoom .level');
let zoom = 1;

function applyZoom(next) {{
  zoom = Math.min(3, Math.max(0.04, next));
  flow.style.width = (flow.dataset.w * zoom) + 'px';
  flow.style.height = (flow.dataset.h * zoom) + 'px';
  level.textContent = Math.round(zoom * 100) + '%';
}}

function fitZoom() {{
  // The pane's own width, less the room a scrollbar takes, so "fit" does not leave one behind.
  const room = flow.parentElement.clientWidth - 8;
  applyZoom(room / flow.dataset.w);
}}

wrap.addEventListener('click', e => {{
  const button = e.target.closest('.zoom button');
  if (!button) return;
  const how = button.dataset.zoom;
  if (how === 'fit') fitZoom();
  else if (how === 'reset') applyZoom(1);
  else applyZoom(how === 'in' ? zoom * 1.25 : zoom / 1.25);
}});

// Ctrl or command with the wheel, which is what a hand already reaches for, and only over the
// drawing -- taking the wheel anywhere else would break scrolling the page.
wrap.querySelector('.canvas').addEventListener('wheel', e => {{
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  applyZoom(zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
}}, {{ passive: false }});

// A port is fetched the first time it is opened, and not before: a page of closed ports costs
// nothing at all, which is what keeps it the same size however many steps the flow grows to.
wrap.addEventListener('toggle', e => {{
  const details = e.target;
  if (!details.open || !LIVE) return;
  const table = details.querySelector('table.data[data-lazy]');
  if (!table) return;
  delete table.dataset.lazy;
  live(table);
}}, true);

fetch('api/ports')
  .then(r => {{ LIVE = r.ok; }})
  .catch(() => {{ LIVE = false; }})
  .finally(() => {{
    if (LIVE) return;
    // Opened as a plain file with nothing serving it: say so once, where the rows would have been,
    // rather than leaving every port looking empty.
    for (const table of wrap.querySelectorAll('table.data[data-lazy]')) {{
      const note = table.closest('.tbl').querySelector('.match');
      note.hidden = false;
      note.textContent = 'rows are served by the preview service, which is not answering';
    }}
  }});
wrap.addEventListener('click', e => {{
  const b = e.target.closest('.picker button');
  if (b) pick(b.dataset.pick);
}});
pick('all');
// Opened on a step, but NOT traced -- the false is load-time and is argued at choose(). The page
// arrives showing the whole flow lit, with one step chosen in the panes beside it; the trace starts
// when the reader points at something.
choose(wrap.dataset.first, false);
applyZoom(1);
</script>
"""


def _refuse_a_borrowed_checkout() -> None:
    """Stop before rendering a page out of some other copy of this repository.

    This repository is worked on in git worktrees against one editable install that points at the
    main checkout. Run a script from a worktree without ``PYTHONPATH=<worktree>/src`` and
    ``dantron_etl`` resolves there instead: the page is then built by the main checkout's tools, and
    nothing in it says so. That has happened here once already -- a run that read a pipeline file the
    author had never touched and looked entirely ordinary on the way past. The project root is now
    resolved from this file rather than from an imported package, so it can only disagree with this
    checkout when ``ETL_PROJECT_ROOT`` points elsewhere -- which is the same mistake, spelled out.

    A page is a statement about one flow, so it has to be a statement about *this* one. The check is
    cheap, it names the fix, and it is worth more than trusting every future caller to remember.
    """
    here = Path(__file__).resolve().parents[1]
    # dantron_etl/__init__.py -> dantron_etl -> src -> the checkout it was installed from.
    resolved = {"the project root": PROJECT_ROOT, "dantron_etl": Path(dantron_etl.__file__).parents[2]}
    borrowed = {name: root for name, root in resolved.items() if root.resolve() != here}
    if borrowed:
        named = ", and ".join(f"{name} is {root}" for name, root in borrowed.items())
        raise SystemExit(
            f"refusing to run: this script is in {here}, {named}.\n"
            f"The page would be built from another checkout's pipelines and tools. Run it as:\n"
            f"  PYTHONPATH={here / 'src'} .venv/bin/python {Path(__file__).resolve().relative_to(here)}"
        )


def main() -> None:
    _refuse_a_borrowed_checkout()
    spec = read_pipeline_file(PROJECT_ROOT / "pipelines" / "fixture.yaml", rules={"revenue_type": "REVENUE"})
    trace = run_pipeline(spec)
    focus = FocusSpec(label="party", identity=("party",))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build(spec, trace, focus), encoding="utf-8")
    print(f"{len(trace.events)} steps, {len(trace.shapes)} ports -> {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
