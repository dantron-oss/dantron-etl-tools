"""The page templates, stylesheets and scripts that ship inside the package, and filling them in.

A template is an HTML file with markers like ``@@canvas@@``. Python fills each marker with markup it
built from the run. The stylesheet and the script are separate files, read here and written into the
page, so a saved page is still one file that opens on its own.

Markers are filled in one pass. Text put in for one marker is never searched for another, so a value
from the data that happens to contain ``@@`` stays as it is.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping, Sequence
from functools import cache
from importlib import resources
from typing import Any

__all__ = ["escape", "page", "page_data", "read_asset"]

_MARKER = re.compile(r"@@([a-z_]+)@@")


def escape(value: object) -> str:
    """``value`` as text that is safe inside HTML, quotes included."""
    return html.escape(str(value), quote=True)


def page_data(data: Mapping[str, Any]) -> str:
    """``data`` as JSON in the page's ``page-data`` script element, for the page's script to read.

    ``</`` is written as ``<\\/`` so a value cannot close the script element early.
    """
    text = json.dumps(data, default=str).replace("</", "<\\/")
    return f'<script type="application/json" id="page-data">{text}</script>'


@cache
def read_asset(name: str) -> str:
    """One file from ``templates/`` or ``static/``, read from the installed package."""
    folder = "templates" if name.endswith(".html") else "static"
    return resources.files(__package__).joinpath(folder, name).read_text(encoding="utf-8")


def fill(template: str, values: dict[str, str]) -> str:
    """``template`` with every ``@@name@@`` replaced by ``values[name]``.

    A marker with no value, or a value with no marker, is a mistake in this package, so both refuse.
    """
    wanted = set(_MARKER.findall(template))
    missing = sorted(wanted - set(values))
    unused = sorted(set(values) - wanted)
    if missing or unused:
        raise ValueError(f"template markers and values differ: missing {missing}, unused {unused}")
    return _MARKER.sub(lambda found: values[found.group(1)], template)


def page(kind: str, /, *, scripts: Sequence[str] | None = None, **values: str) -> str:
    """The page ``kind``: its template filled in, with the shared theme, its stylesheet and its script.

    ``scripts`` names the scripts the page runs, in order. Each one's stylesheet comes too, and each
    runs as a script element of its own. Left out, the page runs only ``<kind>.js``.
    """
    names = [kind] if scripts is None else list(scripts)
    extra_css = "".join(read_asset(f"{name}.css") for name in names if name != kind)
    values = {
        "css": read_asset("theme.css") + read_asset(f"{kind}.css") + extra_css,
        "script": "</script>\n<script>".join(read_asset(f"{name}.js") for name in names),
        **values,
    }
    return fill(read_asset(f"{kind}.html"), values)
