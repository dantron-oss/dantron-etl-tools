"""The tool catalog as a page: one card per tool, grouped by category, with a search box."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..catalog import CATEGORY_ORDER, catalog
from .assets import escape as _e
from .assets import page

__all__ = ["build_page", "write_catalog"]


def _cell(value: object) -> str:
    if value is None:
        return '<span class="nul">null</span>'
    if isinstance(value, bool):
        return f'<span class="bool">{"true" if value else "false"}</span>'
    if isinstance(value, (int, float)):
        return f'<span class="num">{_e(value)}</span>'
    return _e(value)


def _frame_table(frame_data: dict[str, Any]) -> str:
    cols = list(frame_data["schema"])
    head = "".join(f"<th>{_e(c)}<em>{_e(frame_data['schema'][c])}</em></th>" for c in cols)
    if not frame_data["rows"]:
        span = max(len(cols), 1)
        body = f'<tr><td class="empty" colspan="{span}">no rows &mdash; the columns are still here</td></tr>'
    else:
        body = "".join(
            "<tr>" + "".join(f"<td>{_cell(row.get(c))}</td>" for c in cols) + "</tr>" for row in frame_data["rows"]
        )
    return f'<div class="scroll"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _control(name: str, spec: dict[str, Any]) -> str:
    """One configuration field drawn as the control it will be when this panel is editable.

    A typed template is what makes a form generatable without a line of per-tool layout code, so
    the catalogue draws the form rather than describing it: a reader sees the shape of the thing
    they will fill in, and a field whose allowed values are three words is visibly a choice of
    three rather than a sentence about three. The controls are disabled here because nothing on
    this page belongs to a pipeline yet -- a pipeline is a file, and editing arrives with the
    loader that reads one.
    """
    kind = spec["type"]
    shown = "" if spec.get("default") is None else json.dumps(spec["default"], default=str).strip('"')
    if spec.get("allowed"):
        options = "".join(f"<option{' selected' if str(v) == shown else ''}>{_e(v)}</option>" for v in spec["allowed"])
        return f'<select class="ctl" disabled>{options}</select>'
    if kind == "boolean":
        on = " checked" if spec.get("default") is True else ""
        return f'<label class="tog"><input type="checkbox" disabled{on}><span></span></label>'
    if kind in {"integer", "number"}:
        return f'<input class="ctl num" type="text" disabled value="{_e(shown)}" placeholder="a {_e(kind)}">'
    if kind == "expression":
        return (
            '<textarea class="ctl mono" rows="1" disabled '
            f'placeholder="a condition over the columns">{_e(shown)}</textarea>'
        )
    if kind in {"array", "object"}:
        inner = "an object" if kind == "object" else f"a list of {_e(spec.get('items', {}).get('type', 'values'))}"
        return f'<div class="ctl group-note">{inner}</div>'
    return f'<input class="ctl" type="text" disabled value="{_e(shown)}" placeholder="{_e(kind)}">'


def _nest(inner: str) -> str:
    """Wrap a nested template in its own group, or contribute nothing when a field has no shape."""
    return f'<div class="nest">{inner}</div>' if inner else ""


def _config_form(config: dict[str, Any], help_text: dict[str, str], depth: int = 0) -> str:
    """The whole template as a form, nesting where the template nests."""
    out = []
    for name, spec in config.items():
        flags = []
        if spec.get("required"):
            flags.append('<span class="req">required</span>')
        if spec.get("non_empty"):
            flags.append('<span class="req">non-empty</span>')
        if spec.get("default") is not None:
            flags.append('<span class="dflt">default</span>')
        extra = f'<p class="fh">{_e(help_text[name])}</p>' if name in help_text else ""
        nested = ""
        if spec.get("items") and spec["items"].get("fields"):
            # The element itself says what one entry in the list is; without this the reader gets
            # the members and never the sentence that tells them what a member belongs to.
            element = spec["items"].get("description", "")
            preamble = f'<p class="elem">{_e(element)}</p>' if element else ""
            nested = preamble + _config_form(spec["items"]["fields"], {}, depth + 1)
        elif spec.get("items"):
            nested = _config_form({f"{name}[]": spec["items"]}, {}, depth + 1)
        elif spec.get("fields"):
            nested = _config_form(spec["fields"], {}, depth + 1)
        out.append(
            f'<div class="field">'
            f'<div class="lbl"><code>{_e(name)}</code>'
            f'<span class="ty">{_e(spec["type"])}</span>{"".join(flags)}</div>'
            f"{_control(name, spec)}"
            f'<p class="hint">{_e(spec["description"])}</p>{extra}'
            f"{_nest(nested)}"
            f"</div>"
        )
    return "".join(out)


def _tool_card(tool: dict[str, Any]) -> str:
    haystack = f"{tool['name']} {tool['description']} {tool['category']}".lower()
    ins = "".join(f'<span class="anchor in">{_e(p)}</span>' for p in tool["input_ports"])
    if tool["ports_from_config"]:
        # Say where these names came from. They are the first worked example's ports; a step of
        # this tool reads whatever its own configuration names.
        ins += '<span class="pill">ports come from the configuration</span>'
    outs = "".join(f'<span class="anchor out">{_e(p)}</span>' for p in tool["output_ports"])
    notes = (
        '<ul class="notes">' + "".join(f"<li>{_e(n)}</li>" for n in tool["notes"]) + "</ul>" if tool["notes"] else ""
    )
    examples = []
    for ex in tool["examples"]:
        cfg = json.dumps(ex["configuration"], indent=2, default=str)
        order = '<span class="pill">row order checked</span>' if ex["row_order_significant"] else ""
        ports_in = "".join(
            f'<div class="port"><h5><span class="anchor in">{_e(n)}</span></h5>{_frame_table(f)}</div>'
            for n, f in ex["inputs"].items()
        )
        ports_out = "".join(
            f'<div class="port"><h5><span class="anchor out">{_e(n)}</span></h5>{_frame_table(f)}</div>'
            for n, f in ex["outputs"].items()
        )
        examples.append(
            f'<article class="example"><h4>{_e(ex["scenario"])}{order}</h4>'
            f'<pre class="cfg">{_e(cfg)}</pre>'
            f'<p class="why">{_e(ex["explanation"])}</p>'
            f'<div class="flow"><div class="side"><h6>in</h6>{ports_in}</div>'
            f'<div class="arrow" aria-hidden="true">&rarr;</div>'
            f'<div class="side"><h6>out</h6>{ports_out}</div></div></article>'
        )
    return (
        f'<section class="tool" id="tool-{_e(tool["name"])}" data-search="{_e(haystack)}">'
        f'<header><div class="band"><div class="anchors left">{ins}</div>'
        f'<div class="titles"><h2><code>{_e(tool["name"])}</code></h2>'
        f'<span class="cat">{_e(tool["category"])}</span></div>'
        f'<div class="anchors right">{outs}</div></div>'
        f'<p class="lede">{_e(tool["description"])}</p></header>'
        f'<p class="detail">{_e(tool["detail"])}</p>{notes}'
        f'<h3>Configuration</h3><div class="form">{_config_form(tool["config"], tool["field_help"])}</div>'
        f'<h3>Worked examples <span class="pill">executed by the test suite</span></h3>'
        f"{''.join(examples)}</section>"
    )


def build_page(data: dict[str, Any]) -> str:
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for tool in data["tools"]:
        by_cat.setdefault(tool["category"], []).append(tool)
    order = [c for c in CATEGORY_ORDER if c in by_cat] + sorted(set(by_cat) - set(CATEGORY_ORDER))

    rail = "".join(
        f"<h4>{_e(cat)}</h4><ul>"
        + "".join(f'<li><a href="#tool-{_e(t["name"])}"><code>{_e(t["name"])}</code></a></li>' for t in by_cat[cat])
        + "</ul>"
        for cat in order
    )
    cards = "".join(_tool_card(t) for cat in order for t in by_cat[cat])
    return page("catalog", rail=rail, cards=cards, count=str(data["tool_count"]))




def write_catalog(out_dir: Path) -> tuple[Path, Path]:
    """Write ``tool_catalog.html`` and ``tool_catalog.json`` into ``out_dir`` and return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = catalog()
    json_path = out_dir / "tool_catalog.json"
    html_path = out_dir / "tool_catalog.html"
    json_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    html_path.write_text(build_page(data), encoding="utf-8")
    return html_path, json_path
