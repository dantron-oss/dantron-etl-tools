"""Following one entity through a saved run, a window of rows at a time.

The page no longer carries a copy of every port per entity. Instead the server asks the saved
ports for the rows that belong to one entity, using the same rules as ``focus.slice_trace``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ..focus import FocusSpec, RelatedKeyError
from ..preview import PortStore, PreviewError, _inside
from ..spec import PortRef


def _columns(store: PortStore, port: str) -> list[str]:
    entry = store.ports().get(port)
    if entry is None:
        raise PreviewError(f"no port {port!r} in this run; it has: {', '.join(sorted(store.ports()))}")
    return list(entry["columns"])


def _texts(store: PortStore, port: str, match: dict[str, list[str]], wanted: str) -> list[str]:
    """Every distinct text of ``wanted`` at ``port`` on the rows ``match`` keeps, in order."""
    entry = store.ports()[port]
    scan = pl.scan_parquet(_inside(store.directory, entry["file"]))
    for name, values in match.items():
        scan = scan.filter(pl.col(name).cast(pl.String, strict=False).is_in(values))
    return (
        scan.select(pl.col(wanted).cast(pl.String, strict=False))
        .drop_nulls()
        .unique(maintain_order=True)
        .collect()
        .get_column(wanted)
        .to_list()
    )


def focus_match(store: PortStore, focus: FocusSpec, port: str, value: str) -> tuple[str, Any]:
    """The rows of ``port`` that belong to ``value``, as a ``match`` for ``PortStore.window``.

    Returns ``("match", {column: [texts]})``, or ``("no_grain", reason)`` for a port with no column
    naming the entity. A port named in ``focus.via`` is matched on its key, using the key values
    found at the port it learns from. A via port that cannot be narrowed that way raises
    ``RelatedKeyError`` naming it, as ``focus.slice_trace`` does.
    """
    step, _, name = port.partition(".")
    related = focus.via.get(PortRef(step, name))
    if related is not None:
        origin = f"{related.from_port.step}.{related.from_port.port}"
        if related.key not in _columns(store, port):
            raise RelatedKeyError(f"port {port} has no column {related.key!r} to match the related key on")
        origin_columns = _columns(store, origin)
        if related.key not in origin_columns:
            raise RelatedKeyError(
                f"port {port} learns its key {related.key!r} from {origin}, which has no such column"
            )
        identity = focus.identity_column(tuple(origin_columns))
        if identity is None:
            raise RelatedKeyError(
                f"port {port} learns its key from {origin}, but {origin} has no column naming a {focus.label}"
            )
        learned = _texts(store, origin, {identity: [str(value)]}, related.key)
        return "match", {related.key: learned}

    column = focus.identity_column(tuple(_columns(store, port)))
    if column is None:
        return "no_grain", (
            f"port {port} has no column naming a {focus.label}, so it cannot be narrowed to one. "
            f"It is shown whole."
        )
    return "match", {column: [str(value)]}
