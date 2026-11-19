"""Narrow a finished run to one entity, such as one client, so a wrong figure can be followed step by step.

Describe the entity with a :class:`FocusSpec`, then call ``slice_trace(trace, focus, value)``. You
get a :class:`FocusedTrace`: every port of the run, each filtered to that entity's rows, with a
status that says how the port was matched. ``available_entities(trace, focus)`` lists the values
you can pick. Nothing is re-run; the run's own frames are read and left unchanged.

Why: docs/design/loading-and-running.md#focus
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import polars as pl

from .spec import PortRef
from .trace import PipelineTrace
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

#: How one port was matched to the entity:
#: ``matched``: the port has an identity column; its row count is the entity's (it may be 0).
#: ``related``: matched through a :class:`RelatedKey` learned at another port.
#: ``no_grain``: the port has no column naming any entity, so there is nothing to count.
#: ``not_retained``: the run freed this port's frame under ``retain="live"``.
SliceStatus = Literal["matched", "related", "no_grain", "not_retained"]


class FocusError(ToolError):
    """Raised when a focus cannot be built or applied. Every refusal in this module is one."""


class UnknownEntityError(FocusError):
    """Raised when no kept frame with an identity column holds the value asked for.

    A value that some ports hold and others do not is an ordinary slice. See :func:`slice_trace`.
    """


class MissingEvidenceError(FocusError):
    """Raised when the run never produced a port the focus needs, or kept no frame to count from.

    A port that ran but whose frame was freed is not refused: its slice has status
    ``not_retained``.
    """


class RelatedKeyError(FocusError):
    """Raised when a :class:`RelatedKey` is declared wrongly.

    The key column is missing at one end, or the port it learns from has no identity column. Fix
    the focus; re-running the flow will not help.
    """


@dataclass(frozen=True, slots=True)
class RelatedKey:
    """A way to reach a port that has no identity column, through a key column it shares with another port.

    At ``from_port`` the entity's rows are found and the values of ``key`` read from them. The
    target port is then filtered to rows whose ``key`` holds one of those values.
    """

    from_port: PortRef
    key: str


@dataclass(frozen=True, slots=True)
class FocusSpec:
    """Says which columns identify an entity, and which ports to reach through a related key.

    - ``label``: what the entity is called in messages, such as ``"client"``.
    - ``identity``: candidate column names. At each port the first one the port has is used, and
      only that one. A detailed frame and a summary often name the entity differently.
    - ``via``: ``{PortRef: RelatedKey}`` for ports with no identity column.

    Refused when built: an empty ``identity``; a ``via`` key that is not a :class:`PortRef`; a
    ``via`` value that is not a :class:`RelatedKey`; a related key that learns from its own port.

    Why: docs/design/loading-and-running.md#one-value-and-no-per-port-column
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
        """Return the first candidate in :attr:`identity` that ``columns`` has, or ``None`` if none."""
        for candidate in self.identity:
            if candidate in columns:
                return candidate
        return None


@dataclass(frozen=True, slots=True)
class PortSlice:
    """One port of the run, narrowed to one entity.

    - ``status``: how the port was matched (see :data:`SliceStatus`).
    - ``rows_focused``: the entity's row count here. ``0`` means none; ``None`` means the question
      has no answer here (status ``no_grain`` or ``not_retained``).
    - ``rows_at_port`` and ``columns``: the whole port's size and columns, known even when its
      frame was freed.
    - ``matched_on``: the column filtered on. ``related`` and ``related_values``: the related key
      and the values learned for it.
    - ``reason``: a sentence explaining any status other than ``matched``.
    - ``frame``: the entity's rows, or ``None`` when ``rows_focused`` is ``None``.
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
    """A whole run seen through one entity: one :class:`PortSlice` per port, in run order.

    :func:`slice_trace` returns it. Look up one port with ``focused.slice(PortRef(step, port))``.
    """

    pipeline: str
    label: str
    value: Any
    slices: tuple[PortSlice, ...]

    def slice(self, port: PortRef) -> PortSlice:
        """Return the slice for ``port``. Refuses a port that is not here, listing those that are."""
        for one in self.slices:
            if one.port == port:
                return one
        available = ", ".join(str(one.port) for one in self.slices) or "none"
        raise FocusError(f"no port {port!s} on this focused trace; ports are: {available}")


@dataclass(frozen=True, slots=True)
class EntityOption:
    """One entity value a run can be narrowed to, with the ports it appeared on.

    ``rows_by_port`` (or :meth:`rows_at`) gives its row count at each port. There is no single
    total, because one entity can be one row in a summary and many in the detail.
    """

    value: Any
    ports: tuple[PortRef, ...] = ()
    rows_by_port: Mapping[PortRef, int] = field(default_factory=lambda: MappingProxyType({}))

    def rows_at(self, port: PortRef) -> int:
        """Return this entity's row count at ``port``, or ``0`` if it never appeared there."""
        return self.rows_by_port.get(port, 0)


# ---------------------------------------------------------------------------
# Slicing a finished run
# ---------------------------------------------------------------------------


def slice_trace(trace: PipelineTrace, focus: FocusSpec, value: Any) -> FocusedTrace:
    """Narrow every port of ``trace`` to the rows where the entity's column equals ``value``.

    Call it as ``slice_trace(trace, focus, "C-104")``. Returns a :class:`FocusedTrace` with one
    slice per port, in the order the run produced them; a run that failed part way gives the ports
    it reached. The trace is not changed.

    Refuses:

    - :class:`UnknownEntityError`: no kept frame with an identity column holds ``value``. Checked
      only when at least one such frame exists.
    - :class:`RelatedKeyError`: a related key column is missing at either end.
    - :class:`MissingEvidenceError`: a related key learns from a port the run never produced.

    Why: docs/design/loading-and-running.md#a-value-nobody-holds-is-refused
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
            f"{_failure_first(trace)}{focus.label} {value!r} is not held by any column-resolved, retained "
            f"port on this trace; call available_entities(trace, focus) to see which {focus.label} values are"
        )


def _failure_first(trace: PipelineTrace) -> str:
    """Return the failure that ended ``trace`` as the opening of a refusal, or ``""``.

    On a failed run the failure is usually the cause of the focus refusal, so it is said first.
    """
    last = trace.events[-1] if trace.events else None
    if last is None or last.level != "error":
        return ""
    return f"The run stopped when step {last.step_id!r} failed: {last.message}. Because of that, "


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
    """Sort key for a picker's list: by type name first, so values of mixed types can be compared."""
    return type(value).__name__, value


def available_entities(
    trace: PipelineTrace,
    focus: FocusSpec,
    *,
    port: PortRef | None = None,
) -> tuple[EntityOption, ...]:
    """List the entity values a run can be narrowed to, as sorted :class:`EntityOption` items.

    Scans every kept port that has an identity column, or only ``port`` when given. Null values are
    skipped. Ports reached only through a related key are not scanned, because a row there belongs
    to several entities.

    Refuses a ``port`` the run did not produce (:class:`FocusError`), and raises
    :class:`MissingEvidenceError` when no scanned port has a kept frame with an identity column.
    """
    if port is not None and port not in trace.shapes:
        available = ", ".join(str(p) for p in trace.shapes) or "none"
        raise FocusError(f"no port {port!s} on this trace; ports are: {available}")

    ports_to_scan = (port,) if port is not None else tuple(trace.shapes)
    resolvable = [(p, focus.identity_column(trace.shapes[p].columns)) for p in ports_to_scan]
    retained = [(p, column) for p, column in resolvable if column is not None and p in trace.frames]
    if not retained:
        scanned = ", ".join(str(p) for p in ports_to_scan) or "none"
        advice = (
            f"; call run_pipeline(..., retain='all') to get a list of {focus.label} values"
            if trace.retain != "all"
            else f"; none of those ports has a column that holds the {focus.label}"
        )
        raise MissingEvidenceError(
            f"{_failure_first(trace)}no column-resolved port among {scanned} has a retained frame "
            f"(retain={trace.retain!r}){advice}"
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
