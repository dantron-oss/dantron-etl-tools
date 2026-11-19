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
    """Each container's members, split by the lane they are in, as (lane row, members).

    Split rather than taken whole because `_slots` reserves cells of one lane's grid, and a lane is
    the only thing that grid is indexed by. A container declared over two lanes is not a shape this
    page has a drawing for -- its outline would have to reach across the band between them, over
    every step of both lanes in its columns -- so the honest thing is to place each half properly in
    its own lane and let the outline be the bounding box of the two, which is what it is today. The
    guarantees below are per lane and so is the claim they support.

    No flow read from a file can get here with one, though: `_refuse_bad_groups` refuses a group
    whose steps sit in two lanes when the document is loaded, and says to move the step or split the
    container. What reaches this split is a `Pipeline` built in Python, which is how the layout's
    own cases are written. It stays because it is what makes "a claim is a rectangle of one lane's
    grid" true of every input rather than of the ones the loader happens to allow -- take it out and
    a two-lane container silently has half its members placed against the wrong lane's occupancy,
    which is two steps drawn at one coordinate, the defect this whole function exists under.
    """
    split: list[tuple[int, tuple[str, ...]]] = []
    for group in spec.containers:
        mine: dict[int, list[str]] = {}
        for member in spec.container_members(group.id):
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


def _slots(spec: Pipeline, depth: dict[str, int], lanes: dict[str, int]) -> dict[str, int]:
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
    # The lane's name is drawn separately from its band, and last of everything, because the strip
    # of canvas it sits in -- the gap above the band -- is also where a container starting at the
    # first column puts its header. One container in a hundred and fifty lands there. Painted with
    # the bands, the name goes under that header and cannot be read at all; painted last, over a
    # halo, both read, and the name wins the few characters they share. The name is the one label
    # on the canvas a reader cannot recover from anything else on screen, so it is the one to keep.
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
        # Something to click. The outline is `fill:none`, and an SVG shape is a target only where
        # it is actually painted, so until this rect existed a container's inside was not a target
        # at all -- a click there went through it to the lane stripe behind, and the one part a
        # reader could actually hit was 1.2 pixels of dashed line. This covers the header strip: the
        # bar the title is written on, which the layout already keeps clear of steps.
        #
        # Its width is the title's, estimated from the character count rather than measured, because
        # the width of a run of text is only known in a browser and this is written on a machine
        # with none. An estimate is the right tool for a hit area and would be the wrong one for
        # anything drawn -- a few pixels either way moves where a click stops working, and nothing
        # a reader can see. It is deliberately not the whole box: a target that covered a container
        # end to end would sit over every connection that crosses it and take the clicks that light
        # those, which is the one thing on this canvas that already worked well.
        f'<rect class="ghit" x="{left}" y="{top}" width="{12 + len(group.title) * 7.0 + 8}"'
        f' height="{GROUP_HEAD}"/>'
        # The plate the raised title sits on. Drawn here, hidden, rather than built when a reader
        # clicks: a rectangle sized to a run of text can only be measured in a browser, and this
        # page is written on a machine with none -- but the same character-count estimate that
        # places the hit area above places this, and it is the same sentence in the same face. The
        # estimate is safe for both because a few pixels either way moves an edge nobody measures.
        #
        # A rectangle rather than a heavier halo on the glyphs, because a border is what was asked
        # for and a border around a halo follows the outline of the letters -- which reads as a
        # smudge round the words rather than as a label the sentence is printed on.
        f'<rect class="gplate" x="{left + 5}" y="{top + 4}"'
        f' width="{len(group.title) * 7.0 + 14}" height="22" rx="4"/>'
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

