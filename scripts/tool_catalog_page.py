"""Generate the tool catalog: one page a reader can browse, one JSON a model can read.

Both are generated from the registry rather than maintained beside it, because a catalogue
maintained by hand is a catalogue that disagrees with its tools. Everything rendered here is
already declared by each tool for its own reasons -- the typed configuration template exists so a
configuration can be validated, and the worked examples exist because conformance executes them --
so the page cannot describe a tool that does not behave that way without the suite going red first.
"""

from __future__ import annotations

import html
from dataclasses import fields
from typing import Any

import orjson
import polars as pl

from dantron_etl.paths import project_root
from dantron_etl.authoring import REGISTRY, ConfigField
from dantron_etl.contract import catalog_input_ports, ports_from_config

# Resolved from this script's own location rather than from the library's -- see `dantron_etl.paths`.
PROJECT_ROOT = project_root(__file__)
OUT_DIR = PROJECT_ROOT / "output_files"
HTML_PATH = OUT_DIR / "tool_catalog.html"
JSON_PATH = OUT_DIR / "tool_catalog.json"

CATEGORY_ORDER = ("input", "preparation", "transform", "combine", "text", "assert", "output")


def _field_json(spec: ConfigField) -> dict[str, Any]:
    """One configuration field, flattened for a reader that is not Python.

    Built from the dataclass rather than field by field, so a property added to a configuration
    field reaches the catalogue because it exists rather than because somebody remembered a line
    here. The hand-written version had three branches whose false arm was "say nothing", which is
    a silent omission by construction and exactly the drift this page cannot afford.
    """
    out: dict[str, Any] = {}
    for field in fields(spec):
        name, value = field.name, getattr(spec, field.name)
        # Only absence is skipped. Skipping a falsey value would drop a meaningful
        # default=False -- the very silent omission this rewrite exists to remove.
        if value is None:
            continue
        if name == "items":
            out["items"] = _field_json(value)
        elif name == "fields":
            out["fields"] = {inner: _field_json(f) for inner, f in value.items()}
        elif name == "enum":
            out["allowed"] = list(value)
        else:
            out[name] = value
    out.setdefault("required", spec.required)
    out.setdefault("description", spec.description)
    return out


def _frame_json(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "rows": frame.to_dicts(),
    }


def catalog() -> dict[str, Any]:
    """The whole catalogue as plain data -- the model-facing interface, and diffable."""
    tools = []
    for tool in REGISTRY.tools():
        helped = tool.help()
        tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                # A tool that works its ports out from a step's configuration has no fixed ports to
                # show, so the catalogue shows the ones its first worked example reads and says so
                # with the flag beside them. Both halves are needed: the names, and where they
                # came from.
                "input_ports": list(catalog_input_ports(tool)),
                "ports_from_config": ports_from_config(tool),
                "output_ports": list(tool.output_ports),
                "detail": helped.detailed_description.strip(),
                "notes": list(helped.notes),
                "config": {name: _field_json(spec) for name, spec in tool.config_template().items()},
                "field_help": dict(helped.field_help),
                "examples": [
                    {
                        "scenario": ex.scenario,
                        "configuration": dict(ex.configuration),
                        "explanation": ex.explanation,
                        "row_order_significant": getattr(ex, "check_row_order", False),
                        "inputs": {name: _frame_json(f) for name, f in ex.inputs.items()},
                        "outputs": {name: _frame_json(f) for name, f in ex.outputs.items()},
                    }
                    for ex in helped.examples
                ],
            }
        )
    return {"tool_count": len(tools), "tools": tools}


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


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
    shown = "" if spec.get("default") is None else orjson.dumps(spec["default"]).decode().strip('"')
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
        cfg = orjson.dumps(ex["configuration"], option=orjson.OPT_INDENT_2).decode()
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
    return TEMPLATE.format(rail=rail, cards=cards, count=data["tool_count"])


TEMPLATE = """<title>Transformation Tool Catalog</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root {{
  --ground:#f4f4f1; --surface:#fff; --ink:#16181b; --muted:#5c6470; --line:#dfdfd9;
  --accent:#0d6e70; --accent-soft:#e5f0f0; --warn:#8a4b1f; --warn-soft:#f6ece3;
  --sans:'Archivo','Segoe UI',system-ui,sans-serif; --mono:'IBM Plex Mono',ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#131618; --surface:#191d20; --ink:#e8eae6; --muted:#98a0a8; --line:#2a2f34;
    --accent:#46b3b0; --accent-soft:#12312f; --warn:#c98a57; --warn-soft:#2a1f16;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#131618; --surface:#191d20; --ink:#e8eae6; --muted:#98a0a8; --line:#2a2f34;
  --accent:#46b3b0; --accent-soft:#12312f; --warn:#c98a57; --warn-soft:#2a1f16;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--ground); color:var(--ink); font-family:var(--sans); line-height:1.55; }}
code, pre, .data, .ty {{ font-family:var(--mono); }}
.wrap {{ display:grid; grid-template-columns:minmax(190px,220px) minmax(0,1fr); gap:2.5rem;
  max-width:1180px; margin:0 auto; padding:2.5rem 1.25rem 5rem; }}
@media (max-width:820px) {{ .wrap {{ grid-template-columns:1fr; gap:1.5rem; }} nav {{ position:static !important; }} }}

header.top {{ grid-column:1/-1; border-bottom:2px solid var(--ink); padding-bottom:1.25rem; margin-bottom:.5rem; }}
header.top h1 {{ font-size:clamp(1.7rem,3.4vw,2.5rem); font-weight:700; letter-spacing:-.02em;
  margin:0 0 .35rem; text-wrap:balance; }}
header.top p {{ margin:0; color:var(--muted); max-width:62ch; }}
header.top .n {{ color:var(--accent); font-weight:600; }}

nav {{ position:sticky; top:1.5rem; align-self:start; font-size:.9rem; }}
nav h4 {{ font-size:.7rem; text-transform:uppercase; letter-spacing:.11em; color:var(--muted);
  margin:1.4rem 0 .4rem; font-weight:600; }}
nav h4:first-child {{ margin-top:0; }}
nav ul {{ list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:.15rem; }}
nav a {{ color:var(--ink); text-decoration:none; display:block; padding:.2rem .45rem; border-radius:4px; }}
nav a:hover, nav a:focus-visible {{ background:var(--accent-soft); color:var(--accent); outline:none; }}
nav input {{ width:100%; font:inherit; font-size:.85rem; padding:.45rem .55rem; margin-bottom:.5rem;
  border:1px solid var(--line); border-radius:5px; background:var(--surface); color:var(--ink); }}

.tool {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
  padding:1.5rem 1.6rem 1.75rem; margin-bottom:2rem; }}
.tool[hidden] {{ display:none !important; }}
.band {{ display:flex; align-items:center; gap:.9rem; flex-wrap:wrap; }}
.titles {{ display:flex; align-items:baseline; gap:.6rem; flex:1 1 auto; }}
.tool h2 {{ margin:0; font-size:1.35rem; font-weight:600; letter-spacing:-.01em; }}
.tool h2 code {{ font-size:inherit; }}
.cat {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; color:var(--muted); font-weight:600; }}
.anchors {{ display:flex; gap:.3rem; flex-wrap:wrap; }}
.anchor {{ font-family:var(--mono); font-size:.72rem; padding:.15rem .5rem; border-radius:999px;
  border:1px solid var(--accent); white-space:nowrap; }}
.anchor.in {{ color:var(--accent); background:transparent; }}
.anchor.out {{ color:var(--surface); background:var(--accent); border-color:var(--accent); }}
.lede {{ margin:.75rem 0 0; font-size:1.02rem; }}
.detail {{ color:var(--muted); margin:.9rem 0 0; max-width:72ch; }}
.notes {{ margin:.9rem 0 0; padding-left:1.1rem; color:var(--warn); font-size:.92rem; }}
.notes li {{ margin:.25rem 0; }}
.tool h3 {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.11em; color:var(--muted);
  margin:1.9rem 0 .6rem; font-weight:600; border-top:1px solid var(--line); padding-top:1rem; }}
.pill {{ font-family:var(--sans); font-size:.6rem; letter-spacing:.08em; text-transform:uppercase;
  background:var(--accent-soft); color:var(--accent); padding:.15rem .45rem;
  border-radius:3px; margin-left:.5rem; font-weight:600; }}

.form {{ display:grid; gap:1rem; }}
.field {{ display:grid; gap:.3rem; }}
.lbl {{ display:flex; align-items:baseline; gap:.5rem; flex-wrap:wrap; }}
.lbl code {{ font-size:.88rem; font-weight:600; }}
.ty {{ color:var(--accent); font-size:.7rem; font-weight:600; letter-spacing:.04em; }}
.req {{ color:var(--warn); text-transform:uppercase; letter-spacing:.06em; font-size:.6rem; font-weight:600;
  border:1px solid var(--warn); border-radius:3px; padding:0 .25rem; }}
.dflt {{ color:var(--muted); font-size:.62rem; text-transform:uppercase; letter-spacing:.07em; font-weight:600; }}
.ctl {{ font:inherit; font-size:.85rem; width:100%; max-width:34rem; padding:.35rem .5rem;
  border:1px solid var(--line); border-radius:5px; background:var(--ground); color:var(--ink);
  opacity:1; -webkit-text-fill-color:var(--ink); }}
.ctl:disabled {{ cursor:default; }}
.ctl.mono {{ font-family:var(--mono); resize:none; }}
.ctl.num {{ max-width:12rem; font-variant-numeric:tabular-nums; }}
select.ctl {{ appearance:none; background-image:linear-gradient(45deg,transparent 50%,var(--muted) 50%),
  linear-gradient(135deg,var(--muted) 50%,transparent 50%);
  background-position:calc(100% - 14px) 55%, calc(100% - 9px) 55%;
  background-size:5px 5px,5px 5px; background-repeat:no-repeat; padding-right:2rem; }}
.group-note {{ color:var(--muted); font-style:italic; background:transparent; border-style:dashed; }}
.tog {{ display:inline-flex; align-items:center; }}
.tog input {{ position:absolute; opacity:0; pointer-events:none; }}
.tog span {{ width:2.1rem; height:1.15rem; border-radius:999px; background:var(--line);
  position:relative; transition:background .15s; }}
.tog span::after {{ content:""; position:absolute; top:2px; left:2px; width:.9rem; height:.9rem;
  border-radius:50%; background:var(--surface); transition:transform .15s;
  box-shadow:0 1px 2px rgba(0,0,0,.25); }}
.tog input:checked + span {{ background:var(--accent); }}
.tog input:checked + span::after {{ transform:translateX(.95rem); }}
.hint {{ margin:0; color:var(--muted); font-size:.88rem; max-width:64ch; }}
.fh {{ margin:.15rem 0 0; color:var(--muted); font-size:.84rem; font-style:italic; max-width:64ch; }}
.elem {{ margin:0 0 .15rem; color:var(--ink); font-size:.86rem; }}
.nest {{ margin:.45rem 0 .2rem .2rem; padding-left:.9rem; border-left:2px solid var(--accent-soft);
  display:grid; gap:.8rem; }}

.example {{ border:1px solid var(--line); border-radius:8px; padding:1rem 1.1rem; margin-bottom:1rem; }}
.example h4 {{ margin:0 0 .6rem; font-size:.98rem; font-weight:600; }}
pre.cfg {{ background:var(--ground); border:1px solid var(--line); border-radius:6px;
  padding:.65rem .8rem; font-size:.8rem; overflow-x:auto; margin:0; }}
.why {{ color:var(--muted); font-size:.92rem; margin:.7rem 0 0; max-width:72ch; }}
.flow {{ display:grid; grid-template-columns:1fr auto 1fr; gap:1rem; align-items:start; margin-top:1rem; }}
@media (max-width:700px) {{ .flow {{ grid-template-columns:1fr; }} .arrow {{ transform:rotate(90deg); }} }}
.side h6 {{ font-size:.62rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted);
  margin:0 0 .45rem; font-weight:600; }}
.arrow {{ color:var(--accent); font-size:1.3rem; align-self:center; }}
.port {{ margin-bottom:.8rem; }}
.port h5 {{ margin:0 0 .3rem; font-weight:400; }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; }}
table.data {{ border-collapse:collapse; font-size:.78rem; width:100%; font-variant-numeric:tabular-nums; }}
table.data th {{ text-align:left; padding:.35rem .55rem; border-bottom:1px solid var(--line);
  background:var(--ground); font-weight:600; white-space:nowrap; }}
table.data th em {{ display:block; font-style:normal; font-weight:400; font-size:.85em; color:var(--muted); }}
table.data td {{ padding:.3rem .55rem; border-bottom:1px solid var(--line); white-space:nowrap; }}
table.data tr:last-child td {{ border-bottom:none; }}
td.empty {{ color:var(--muted); font-style:italic; font-family:var(--sans); }}
.nul {{ color:var(--muted); font-style:italic; }}
.num {{ color:var(--accent); }}
.bool {{ color:var(--warn); }}
footer {{ grid-column:1/-1; color:var(--muted); font-size:.85rem; border-top:1px solid var(--line);
  padding-top:1rem; max-width:72ch; }}
</style>
<div class="wrap">
  <header class="top">
    <h1>Transformation Tool Catalog</h1>
    <p><span class="n">{count}</span> generic tools. Every field, default and worked example on this page is
    read from the tools themselves &mdash; and the suite executes each example against its tool, so an
    example here cannot have drifted from what the tool actually does.</p>
  </header>
  <nav>
    <input id="q" type="search" placeholder="Filter tools&hellip;" aria-label="Filter tools">
    {rail}
  </nav>
  <main>{cards}</main>
  <footer>Outlined chips are input ports; filled chips are outputs. A tool with two outputs keeps what it
  set aside rather than discarding it &mdash; that is why nothing a flow drops has to be reconstructed later.</footer>
</div>
<script>
const q = document.getElementById('q');
q.addEventListener('input', () => {{
  const term = q.value.trim().toLowerCase();
  for (const card of document.querySelectorAll('.tool')) {{
    card.hidden = term && !card.dataset.search.includes(term);
  }}
}});
</script>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = catalog()
    JSON_PATH.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))
    HTML_PATH.write_text(build_page(data), encoding="utf-8")
    print(f"{data['tool_count']} tools -> {HTML_PATH} ({HTML_PATH.stat().st_size:,} bytes)")
    print(f"{' ' * len(str(data['tool_count']))} -> {JSON_PATH} ({JSON_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
