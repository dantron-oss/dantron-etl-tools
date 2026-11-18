"""Narrowing a finished run to one entity, so a wrong figure can be traced to the step that made it.

The runner leaves a trace: every frame a flow produced, keyed by the port that produced it. That
answers "what happened". It does not answer *show me this one thing, at every step* -- and that is
what slicing answers. A slice reads a finished run and hands back a parallel view of it, one entry
per port, filtered to the rows that belong to one value. It never re-runs anything: a reduction sums
over siblings and a pairing matches across a whole frame, so a flow re-run over one entity's rows
would answer a different question, with numbers that look entirely reasonable. Reading rather than
re-running is also what lets the same finished run be sliced again for the next entity and answer
the same way both times.

Identity is a tuple of candidate column names, tried first-present at every port, because a detailed
frame and a reduced one often name the same entity differently. There is deliberately no per-port
column override: a slice applies one value, the entity's own, and naming a key column at a reduced
grain would filter that column for a value it never holds and match nothing, silently. The only
per-port declaration a focus may make is a :class:`RelatedKey`, which reaches a grain that names no
entity at all by learning, at a port where the entity does resolve, the values of some other column
-- so the value a related port is matched on is always learned, never the one a reader typed.

Four different answers exist at a port, and conflating any two of them produces a believable wrong
number. A port that names the entity and holds none of its rows is a fact about the entity
(``matched``, ``rows_focused=0``). A port that holds no column naming any entity at all is a fact
about the port, not the entity -- reporting it as nought rows would send a reader hunting a filter
that never fired (``no_grain``, ``rows_focused=None``). A port reached only through a related key is
its own status because the value it is matched on is not the reader's own (``related``). And a port
a live run already released its frame for is a fact about the run, not about what the entity held
there (``not_retained``, ``rows_focused=None``) -- what is still knowable without the frame, namely
its size and its columns, is still reported. No-grain is checked ahead of not-retained, because
whether a port names an entity at all is knowable from its shape alone, with no frame required.

An entity value nobody's rows hold is refused, because a view reporting nought rows at every single
port is indistinguishable, by eye, from an entity a flow genuinely dropped at its very first step --
which is the wrong answer this module exists to prevent. That refusal only fires when it can be
backed up: it needs at least one column-resolved port with a retained frame to check against, so a
live run, which retains none, can refute nothing and must not try.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import polars as pl

from .spec import PortRef
from .runner import PipelineTrace
from .errors import ToolError

__all__ = [
    "EntityOption",
    "FocusError",
    "FocusSpec",
    "FocusedTrace",
    "MissingEvidenceError",
    "PortSlice",
    "RelatedKey",
    "RelatedKeyError",
    "SliceStatus",
    "UnknownEntityError",
    "available_entities",
    "slice_trace",
]

#: What a port's slice reports. ``matched`` names the entity directly and its row count is a fact
#: about the entity; ``related`` is matched through a learned key rather than the entity's own
#: value; ``no_grain`` is a fact about the port, which carries no column naming any entity at all;
#: ``not_retained`` is a fact about the run, whose frame a live run already released.
SliceStatus = Literal["matched", "related", "no_grain", "not_retained"]


class FocusError(ToolError):
    """The base of every refusal this module raises, so a caller has one family to catch.

    Shares its base with every other tool boundary in this library rather than inventing a second
    hierarchy for the one module that reads a trace instead of running one.
    """


class UnknownEntityError(FocusError):
    """A value was asked for that no column-resolved, retained port holds anywhere on the trace.

    Distinct from a value that some ports hold and others do not -- that is an ordinary slice, not a
    refusal. This fires only where it can be backed up by evidence; see :func:`slice_trace`.
    """


class MissingEvidenceError(FocusError):
    """A slice needs a port, or a frame, that this run never produced or already let go of.

    Raised for a related key whose origin port a partial run never reached at all -- as opposed to a
    port that ran but whose frame a live run later released, which is reported as the ``not_retained``
    status on the slice itself rather than as a refusal, because that is a fact worth showing rather
    than a request that cannot be answered.
    """


class RelatedKeyError(FocusError):
    """A :class:`RelatedKey` names a port and column pairing that cannot actually be resolved.

    Distinct from :class:`MissingEvidenceError`: this is a mis-declaration -- the key is absent as a
    column at one end, or the origin names no entity at all -- and the fix is to correct the focus,
    not to re-run the flow.
    """


@dataclass(frozen=True, slots=True)
class RelatedKey:
    """Reach a port at a grain that names no entity, through a column learned somewhere else.

    ``from_port`` is where the key's values are learned, filtered to the entity's own rows there --
    so the value ultimately applied at the port this key resolves is never the one a reader typed.
    """

    from_port: PortRef
    key: str


@dataclass(frozen=True, slots=True)
class FocusSpec:
    """What identifies one entity, and which ports can only be reached through a learned key.

    ``identity`` is tried first-present, in declaration order, at every port that is not named in
    ``via`` -- the first candidate a port's own columns actually carry, not the first one that
    happens to hold the value being sliced for, and never more than one at once. This rule is
    unenforced across ports: nothing here checks that two candidates never appear together on one
    port, because a slice cannot detect a rule no fixture ever exercises the failure of.

    ``via`` is the only per-port thing a focus may say, and it may only say "reach this one through
    a :class:`RelatedKey`" -- naming a plain column here instead is refused, because a slice applies
    one value and a column named at a reduced grain would filter for a value it never holds.
    """

    label: str
    identity: tuple[str, ...]
    via: Mapping[PortRef, RelatedKey] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.identity:
            raise FocusError("identity must name at least one candidate column; an empty tuple names no entity at all")
        for port, related in self.via.items():
            if not isinstance(port, PortRef):
                raise FocusError(
                    f"via key {port!r} is not a PortRef; a focus names a port the way a trace names "
                    f"one, PortRef(step, port) -- a bare string that is misspelled would silently "
                    f"read as a port with no grain rather than a mistake in the focus"
                )
            if not isinstance(related, RelatedKey):
                raise FocusError(
                    f"via[{port!s}] must be a RelatedKey, not {related!r}; a focus has no per-port "
                    f"column override -- a slice applies one value, the entity's own, and naming a "
                    f"key column at a reduced grain would filter that column for a value it never "
                    f"holds and match nothing, silently"
                )
            if related.from_port == port:
                raise FocusError(
                    f"via[{port!s}] learns from {port!s} itself; a related key's values must come "
                    f"from a port where the entity actually resolves, never the port it is applied to"
                )
        object.__setattr__(self, "via", MappingProxyType(dict(self.via)))

    def identity_column(self, columns: tuple[str, ...]) -> str | None:
        """The first candidate in :attr:`identity` that ``columns`` carries, or ``None``.

        ``None`` means this port names no entity at all, which is a fact about the port and never a
        reason to fall back to matching every row or none of them -- both are the plausible wrong
        answer this module exists to prevent.
        """
        for candidate in self.identity:
            if candidate in columns:
                return candidate
        return None


@dataclass(frozen=True, slots=True)
class PortSlice:
    """One port's frame narrowed to one entity, and everything it took to narrow it.

    ``rows_at_port`` and ``columns`` come from the trace's own shapes, so they are reported even for
    a released port -- how large a port was and what it carried is knowable without holding its
    frame. ``rows_focused`` is ``0`` when the entity's row count there truly is nought, and ``None``
    when there is no such question at this port to answer at all: no column names an entity there,
    or no frame survived to be asked. ``frame`` is ``None`` under exactly those same two conditions,
    so a shared reference row is never drawn under one entity's name, and a released frame is never
    fabricated. ``reason`` is empty for a plain, column-matched slice, which has nothing to explain.
    """

    port: PortRef
    status: SliceStatus
    rows_focused: int | None
    rows_at_port: int
    columns: tuple[str, ...]
    matched_on: str | None
    related: RelatedKey | None
    related_values: tuple[Any, ...]
    reason: str
    frame: pl.DataFrame | None


@dataclass(frozen=True, slots=True)
class FocusedTrace:
    """A whole run seen through one entity: one slice per port the run produced, in its own order.

    Reading, never writing: every slice is a fresh frame, so the run this was built from is left
    exactly as it was and can be sliced again for the next entity with the same answer both times.
    """

    pipeline: str
    label: str
    value: Any
    slices: tuple[PortSlice, ...]

    def slice(self, port: PortRef) -> PortSlice:
        """The slice for ``port``, or a refusal naming the ports this focused trace actually has."""
        for one in self.slices:
            if one.port == port:
                return one
        available = ", ".join(str(one.port) for one in self.slices) or "none"
        raise FocusError(f"no port {port!s} on this focused trace; ports are: {available}")


@dataclass(frozen=True, slots=True)
class EntityOption:
    """One entity a finished run can be sliced to, and where it was seen while counting.

    ``rows_by_port`` and :meth:`rows_at` are all there is: an entity's count is only ever meaningful
    against one named port, because the same entity is one row at a parent grain and several at a
    child grain, and any single number spanning two of those would be wrong at every other grain than
    the one it happened to be taken from. There is deliberately no scalar row count here at all.
    """

    value: Any
    ports: tuple[PortRef, ...] = ()
    rows_by_port: Mapping[PortRef, int] = field(default_factory=lambda: MappingProxyType({}))

    def rows_at(self, port: PortRef) -> int:
        """How many of this entity's rows ``port`` carried, or ``0`` if it never appeared there."""
        return self.rows_by_port.get(port, 0)


# ---------------------------------------------------------------------------
# Slicing a finished run
# ---------------------------------------------------------------------------


def slice_trace(trace: PipelineTrace, focus: FocusSpec, value: Any) -> FocusedTrace:
    """Narrow every port ``trace`` produced to the rows carrying ``value``, one slice per port.

    ``trace`` is read, never written: every slice is built from a fresh, filtered frame, so the same
    finished run can be sliced again for the next entity and the run's own frames are untouched.
    Ports are visited in the order the run produced them -- a partial run from a failure slices to
    exactly the ports it reached.

    Refuses, before building anything, a ``value`` that no column-resolved and retained port holds
    anywhere on the trace: a view of nought rows everywhere is indistinguishable, by eye, from an
    entity a flow dropped at its very first step, which is the wrong answer this function exists to
    prevent. That refusal only fires where it can be backed up by at least one such port; a live run,
    which retains none, can refute nothing and is sliced as-is.

    Raises :class:`RelatedKeyError` for a related key absent as a column at either end, and
    :class:`MissingEvidenceError` for a related key whose origin port this run never reached at all.
    """
    _refuse_unheld_value(trace, focus, value)
    slices = tuple(_build_slice(trace, focus, port, value) for port in trace.shapes)
    return FocusedTrace(pipeline=trace.pipeline, label=focus.label, value=value, slices=slices)


def _refuse_unheld_value(trace: PipelineTrace, focus: FocusSpec, value: Any) -> None:
    checked_any = False
    for port, shape in trace.shapes.items():
        column = focus.identity_column(shape.columns)
        if column is None or port not in trace.frames:
            continue
        checked_any = True
        if trace.frames[port].filter(pl.col(column) == value).height > 0:
            return
    if checked_any:
        raise UnknownEntityError(
            f"{focus.label} {value!r} is not held by any column-resolved, retained port on this "
            f"trace; call available_entities(trace, focus) to see which {focus.label} values are"
        )


def _build_slice(trace: PipelineTrace, focus: FocusSpec, port: PortRef, value: Any) -> PortSlice:
    shape = trace.shapes[port]
    related = focus.via.get(port)
    if related is not None:
        return _related_slice(trace, focus, port, related, value)
    column = focus.identity_column(shape.columns)
    if column is None:
        return PortSlice(
            port=port,
            status="no_grain",
            rows_focused=None,
            rows_at_port=shape.rows,
            columns=shape.columns,
            matched_on=None,
            related=None,
            related_values=(),
            reason=(
                f"port {port!s} carries no column naming a {focus.label} "
                f"({', '.join(shape.columns) if shape.columns else '(no columns)'}); that is a fact "
                f"about this port, not about any one {focus.label}"
            ),
            frame=None,
        )
    if port not in trace.frames:
        return PortSlice(
            port=port,
            status="not_retained",
            rows_focused=None,
            rows_at_port=shape.rows,
            columns=shape.columns,
            matched_on=column,
            related=None,
            related_values=(),
            reason=(
                f"port {port!s} produced a frame ({shape.rows} row(s), columns {shape.columns}), but "
                f"it was released under retain='live' before this trace was read; not retained"
            ),
            frame=None,
        )
    matched = trace.frames[port].filter(pl.col(column) == value)
    return PortSlice(
        port=port,
        status="matched",
        rows_focused=matched.height,
        rows_at_port=shape.rows,
        columns=shape.columns,
        matched_on=column,
        related=None,
        related_values=(),
        reason="",
        frame=matched,
    )


def _related_slice(
    trace: PipelineTrace,
    focus: FocusSpec,
    port: PortRef,
    related: RelatedKey,
    value: Any,
) -> PortSlice:
    shape = trace.shapes[port]
    if related.key not in shape.columns:
        raise RelatedKeyError(
            f"the related key {related.key!r} declared for port {port!s} is not a column there; its "
            f"columns are {shape.columns}. Either the key is misspelled, or it belongs to a different port"
        )
    if related.from_port not in trace.shapes:
        produced = ", ".join(str(p) for p in trace.shapes) or "none"
        raise MissingEvidenceError(
            f"port {port!s} is reached through a key learned at {related.from_port!s}, but this run "
            f"never produced {related.from_port!s}; ports it did produce: {produced}"
        )
    origin_shape = trace.shapes[related.from_port]
    if related.key not in origin_shape.columns:
        raise RelatedKeyError(
            f"the related key {related.key!r} that port {port!s} learns from {related.from_port!s} is "
            f"not a column there; its columns are {origin_shape.columns}. Either end may be the "
            f"misspelling -- the key has to be a column both ports share"
        )
    origin_column = focus.identity_column(origin_shape.columns)
    if origin_column is None:
        raise RelatedKeyError(
            f"port {port!s} learns its key from {related.from_port!s}, but {related.from_port!s} "
            f"carries no column naming a {focus.label} at all, so no rows there can be attributed to "
            f"one; its columns are {origin_shape.columns}"
        )
    if related.from_port not in trace.frames or port not in trace.frames:
        return PortSlice(
            port=port,
            status="not_retained",
            rows_focused=None,
            rows_at_port=shape.rows,
            columns=shape.columns,
            matched_on=related.key,
            related=related,
            related_values=(),
            reason=(
                f"port {port!s} is reached through related key {related.key!r} learned from "
                f"{related.from_port!s}, but the frame(s) needed were released under retain='live' "
                f"before this trace was read; not retained"
            ),
            frame=None,
        )
    origin = trace.frames[related.from_port]
    learned = (
        origin.filter(pl.col(origin_column) == value)
        .drop_nulls(subset=[related.key])
        .get_column(related.key)
        .unique(maintain_order=True)
        .to_list()
    )
    values = tuple(learned)
    matched = trace.frames[port].filter(pl.col(related.key).is_in(values))
    return PortSlice(
        port=port,
        status="related",
        rows_focused=matched.height,
        rows_at_port=shape.rows,
        columns=shape.columns,
        matched_on=related.key,
        related=related,
        related_values=values,
        reason=(
            f"port {port!s} carries no column naming a {focus.label} directly; its rows are matched "
            f"through key {related.key!r}, whose value(s) were learned from port {related.from_port!s} "
            f"for this {focus.label}"
        ),
        frame=matched,
    )


# ---------------------------------------------------------------------------
# What a picker may offer
# ---------------------------------------------------------------------------


def _sort_key(value: Any) -> tuple[str, Any]:
    """Order a picker's list by value, grouped by type name first so mixed grains stay comparable."""
    return type(value).__name__, value


def available_entities(
    trace: PipelineTrace,
    focus: FocusSpec,
    *,
    port: PortRef | None = None,
) -> tuple[EntityOption, ...]:
    """Every entity a finished run can be sliced to, each counted once and against a named port.

    ``port`` narrows the question to one port; left unset, every column-resolved port on the trace
    is scanned. A port reached only through a related key offers nobody to pick, because a row there
    belongs to several entities at once and attributing it to one would double-count the picker.

    Raises :class:`MissingEvidenceError` when no column-resolved port this call would scan still
    holds a retained frame -- a live run holds none, and nothing here can be counted without one.
    """
    if port is not None and port not in trace.shapes:
        available = ", ".join(str(p) for p in trace.shapes) or "none"
        raise FocusError(f"no port {port!s} on this trace; ports are: {available}")

    ports_to_scan = (port,) if port is not None else tuple(trace.shapes)
    resolvable = [(p, focus.identity_column(trace.shapes[p].columns)) for p in ports_to_scan]
    retained = [(p, column) for p, column in resolvable if column is not None and p in trace.frames]
    if not retained:
        raise MissingEvidenceError(
            f"no column-resolved port among {', '.join(str(p) for p in ports_to_scan) or 'none'} has "
            f"a retained frame (retain={trace.retain!r}); call run_pipeline(..., retain='all') to get "
            f"a list of {focus.label} values"
        )

    order: list[Any] = []
    ports_by_value: dict[Any, list[PortRef]] = {}
    rows_by_value: dict[Any, dict[PortRef, int]] = {}
    for p, column in retained:
        frame = trace.frames[p]
        counted = frame.drop_nulls(subset=[column]).group_by(column, maintain_order=True).agg(pl.len().alias("n"))
        for entity_value, rows in zip(
            counted.get_column(column).to_list(), counted.get_column("n").to_list(), strict=True
        ):
            if entity_value not in ports_by_value:
                order.append(entity_value)
                ports_by_value[entity_value] = []
                rows_by_value[entity_value] = {}
            ports_by_value[entity_value].append(p)
            rows_by_value[entity_value][p] = rows

    return tuple(
        EntityOption(
            value=entity_value,
            ports=tuple(ports_by_value[entity_value]),
            rows_by_port=MappingProxyType(rows_by_value[entity_value]),
        )
        for entity_value in sorted(order, key=_sort_key)
    )
