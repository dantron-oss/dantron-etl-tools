"""An architecture page for the tool library, generated from the library itself.

A document describing code drifts from it the moment either changes, and a document describing code
*this* library's way -- by hand, in prose, with the reasoning that matters -- drifts fastest, because
the reasoning is the part nobody updates. So nothing here is written twice: the layers are read from
the modules' own docstrings, the tools from the registry, and the line counts from the files. What is
written by hand is the one thing generation cannot supply -- why each layer exists at all -- and it
is one sentence each.

Every item opens its own source, fetched from the preview service, so "here is where the tools are"
and "here is the code" are the same click.

    .venv/bin/python scripts/etl_architecture_page.py
"""

import ast
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dantron_etl  # noqa: E402,F401  -- importing the package is what fills the registry
from dantron_etl.paths import project_root  # noqa: E402
from dantron_etl.authoring import CATEGORIES, REGISTRY, Tool  # noqa: E402
from dantron_etl.contract import catalog_input_ports, ports_from_config  # noqa: E402

# Resolved from this script's own location rather than from the library's -- see `dantron_etl.paths`.
PROJECT_ROOT = project_root(__file__)
OUT_DIR = PROJECT_ROOT / "output_files" / "runs" / "architecture"

#: The one thing reading the code cannot tell you: why the layer is there. One sentence each, and
#: the module's own docstring supplies everything else.
LAYERS: tuple[tuple[str, str, str], ...] = (
    (
        "contract",
        "The contract",
        "What a tool is: its ports, its settings, its documentation, and the questions the library asks it.",
    ),
    ("config", "A tool's settings", "The template a step's configuration is held to, checked without a frame."),
    ("ports", "The envelope", "Named frames in and out, so a side output cannot be forgotten without a word."),
    ("status", "What happened to a row", "Which rows a step degraded and why, carried beside the port."),
    (
        "boundary",
        "The boundary",
        "Every rule enforced on each call to a tool, rather than remembered by each tool's author.",
    ),
    (
        "frame_checks",
        "A setting against a frame",
        "The questions most tools ask once the frames exist, answered once and in one wording.",
    ),
    (
        "conformance",
        "Keeping the catalog true",
        "Every tool held to the catalog's rules, and every worked example run and compared.",
    ),
    (
        "expr",
        "The expression language",
        "The small, safe language a configuration computes in -- and the only thing here that reads text a "
        "non-developer wrote and turns it into something that runs.",
    ),
    ("spec", "What a flow is", "Steps, edges and the order they run in, as plain data a canvas can draw."),
    ("rules", "Values from a rules file", "Each ${rules.*} reference spelled for the field it lands in."),
    (
        "loader",
        "A flow, as a data file",
        "Loading a definition into something checked: every tool named, every field typed, every edge real, "
        "before a single row moves.",
    ),
    (
        "runner",
        "Running one",
        "Walking the steps in dependency order and leaving behind evidence of what each one did.",
    ),
    ("walk", "Planning a run", "Which steps a partial or resumed run walks, decided before any of them runs."),
    ("trace", "The evidence", "What a run leaves behind: one event per step and every port's shape."),
    (
        "focus",
        "Following one party",
        "Narrowing a finished run to a single entity, so a figure can be chased through every step that "
        "touched it.",
    ),
    (
        "preview",
        "Looking at the result",
        "Serving a finished run a window at a time, so the whole of it can be searched without any of it "
        "being carried into a page.",
    ),
    ("db", "Reaching a database", "Connections named by environment variable, and identifiers held to a grammar."),
    ("column_types", "Naming a type", "The words a flow uses to say what a column is, in one place for every tool."),
)


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _summary(source: str) -> str:
    """A module's own first paragraph, which is where it already explains itself."""
    doc = ast.get_docstring(ast.parse(source)) or ""
    body = doc.split("\n\n")
    lead = body[1] if len(body) > 1 and body[0].count(" ") < 14 else body[0]
    return " ".join(lead.split())


def _module_card(name: str, title: str, why: str) -> str:
    path = f"src/dantron_etl/{name}.py"
    source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    return (
        f'<article class="item" data-path="{_e(path)}">'
        f"<h3>{_e(title)}<code>{_e(name)}.py</code>"
        f'<span class="lines">{len(source.splitlines()):,} lines</span></h3>'
        f'<p class="why">{_e(why)}</p>'
        f'<p class="says">{_e(_summary(source))}</p>'
        f'<button data-open="{_e(path)}">read the code</button></article>'
    )


def _in_ports(tool: Tool) -> str:
    """What to print for a tool's input ports on a page with no step in front of it."""
    ports = ", ".join(catalog_input_ports(tool))
    if not ports:
        return "reads a source"
    if ports_from_config(tool):
        return f"{ports} (from the configuration)"
    return ports


def _tool_card(name: str) -> str:
    tool = REGISTRY.get(name)
    path = f"src/dantron_etl/tools/{name}.py"
    source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    fields = ", ".join(tool.config_template()) or "nothing to configure"
    return (
        f'<article class="item" data-path="{_e(path)}">'
        f"<h3><code>{_e(name)}</code>"
        f'<span class="lines">{len(source.splitlines()):,} lines</span></h3>'
        f'<p class="says">{_e(tool.description)}</p>'
        f"<dl><dt>in</dt><dd>{_e(_in_ports(tool))}</dd>"
        f"<dt>out</dt><dd>{_e(', '.join(tool.output_ports) or 'writes a destination')}</dd>"
        f"<dt>config</dt><dd>{_e(fields)}</dd></dl>"
        f'<button data-open="{_e(path)}">read the code</button></article>'
    )


def build() -> str:
    layers = "".join(_module_card(name, title, why) for name, title, why in LAYERS)

    by_category: dict[str, list[str]] = {}
    for name in sorted(REGISTRY.names()):
        by_category.setdefault(REGISTRY.get(name).category, []).append(name)
    tools = "".join(
        f'<h2 id="cat-{_e(category)}">{_e(category)}'
        f'<span class="count">{len(by_category.get(category, ()))}</span></h2>'
        f'<div class="grid">{"".join(_tool_card(name) for name in by_category.get(category, ()))}</div>'
        for category in CATEGORIES
        if by_category.get(category)
    )

    nav = (
        '<a href="#layers">the library</a>'
        + "".join(
            f'<a href="#cat-{_e(category)}">{_e(category)}</a>' for category in CATEGORIES if by_category.get(category)
        )
        + '<a href="#harness">how it is checked</a>'
    )

    total = sum(
        len((PROJECT_ROOT / f"src/dantron_etl/{name}.py").read_text(encoding="utf-8").splitlines())
        for name, _, _ in LAYERS
    )
    summary = (
        f"<b>{len(REGISTRY.names())}</b> tools &middot; <b>{len(LAYERS)}</b> modules &middot; "
        f"<b>{total:,}</b> lines that reach nothing outside themselves"
    )
    return TEMPLATE.format(summary=summary, nav=nav, layers=layers, tools=tools)


TEMPLATE = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>The transformation tool library</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#f4f4f1; --surface:#fff; --ink:#16181b; --muted:#5c6470; --line:#dfdfd9;
  --accent:#0d6e70; --accent-soft:#e5f0f0;
  --sans:'Archivo','Segoe UI',system-ui,sans-serif; --mono:'IBM Plex Mono',ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark) {{ :root:not([data-theme="light"]) {{
  --ground:#131618; --surface:#191d20; --ink:#e8eae6; --muted:#98a0a8; --line:#2a2f34;
  --accent:#46b3b0; --accent-soft:#12312f; }} }}
:root[data-theme="dark"] {{
  --ground:#131618; --surface:#191d20; --ink:#e8eae6; --muted:#98a0a8; --line:#2a2f34;
  --accent:#46b3b0; --accent-soft:#12312f; }}
* {{ box-sizing:border-box; scrollbar-width:thin; scrollbar-color:var(--line) transparent; }}
::-webkit-scrollbar {{ width:7px; height:7px; }}
::-webkit-scrollbar-thumb {{ background:var(--line); border-radius:4px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
html, body {{ height:100%; margin:0; }}
body {{ background:var(--ground); color:var(--ink); font-family:var(--sans); line-height:1.55; }}

.shell {{ height:100vh; display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr);
  grid-template-rows:auto minmax(0,1fr); gap:.6rem; padding:.6rem; }}
header.top {{ grid-column:1 / 3; display:flex; align-items:baseline; gap:1rem; flex-wrap:wrap;
  padding:.2rem .3rem; }}
header.top h1 {{ font-size:1.15rem; margin:0; letter-spacing:-.01em; }}
header.top p {{ margin:0; color:var(--muted); font-size:.85rem; }}
header.top b {{ color:var(--accent); }}
nav {{ margin-left:auto; display:flex; gap:.1rem; flex-wrap:wrap; }}
nav a {{ font:600 .66rem var(--sans); text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); text-decoration:none; padding:.2rem .45rem; border-radius:5px; }}
nav a:hover {{ background:var(--accent-soft); color:var(--accent); }}

.map, .code {{ min-height:0; overflow:auto; background:var(--surface);
  border:1px solid var(--line); border-radius:9px; padding:1rem 1.1rem; }}
h2 {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.12em; color:var(--muted);
  margin:1.6rem 0 .6rem; display:flex; align-items:center; gap:.5rem; }}
h2:first-child {{ margin-top:0; }}
h2 .count {{ font:500 .66rem var(--mono); color:var(--accent); background:var(--accent-soft);
  padding:.05rem .35rem; border-radius:4px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(15rem,1fr)); gap:.6rem; }}
.item {{ border:1px solid var(--line); border-radius:8px; padding:.7rem .8rem; background:var(--surface); }}
.item.reading {{ border-color:var(--accent); background:var(--accent-soft); }}
.item h3 {{ margin:0 0 .3rem; font-size:.9rem; display:flex; align-items:baseline; gap:.45rem;
  flex-wrap:wrap; }}
.item h3 code {{ font:500 .74rem var(--mono); color:var(--accent); }}
.item .lines {{ margin-left:auto; font:400 .66rem var(--mono); color:var(--muted); }}
.item .why {{ margin:0 0 .35rem; font-size:.82rem; }}
.item .says {{ margin:0 0 .5rem; font-size:.78rem; color:var(--muted); }}
.item dl {{ display:grid; grid-template-columns:auto 1fr; gap:.1rem .5rem; margin:0 0 .55rem;
  font-size:.72rem; }}
.item dt {{ color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  font-size:.62rem; padding-top:.1rem; }}
.item dd {{ margin:0; font-family:var(--mono); }}
.item button {{ font:600 .68rem var(--sans); color:var(--muted); background:none;
  border:1px solid var(--line); border-radius:5px; padding:.15rem .5rem; cursor:pointer; }}
.item button:hover {{ border-color:var(--accent); color:var(--accent); }}

.code header {{ display:flex; align-items:baseline; gap:.6rem; margin-bottom:.5rem;
  position:sticky; top:0; background:var(--surface); padding-bottom:.4rem; }}
.code header b {{ font:600 .78rem var(--mono); }}
.code header span {{ color:var(--muted); font-size:.72rem; }}
.code pre {{ margin:0; font:400 .72rem var(--mono); white-space:pre; }}
.code .idle {{ color:var(--muted); font-size:.84rem; }}
</style>

<div class="shell">
  <header class="top"><h1>The transformation tool library</h1><p>{summary}</p><nav>{nav}</nav></header>
  <div class="map">
    <h2 id="layers">the library<span class="count">{{modules}}</span></h2>
    <div class="grid">{layers}</div>
    {tools}
    <h2 id="harness">how it is checked</h2>
    <div class="grid">
      <article class="item" data-path="tests/test_dantron_etl_boundary.py">
        <h3>Independence</h3>
        <p class="says">Every module imported in a fresh subprocess, asserting that nothing outside this
        library reaches it -- and its sources scanned for any caller's vocabulary.</p>
        <button data-open="tests/test_dantron_etl_boundary.py">read the code</button></article>
      <article class="item" data-path="src/dantron_etl/conformance.py">
        <h3>Conformance</h3>
        <p class="says">Every tool's worked examples are executed on every check, so an example that stops
        being true stops the build rather than misleading a reader.</p>
        <button data-open="src/dantron_etl/conformance.py">read the code</button></article>
      <article class="item" data-path="scripts/mrr_period_oracle.py">
        <h3>Identity against an engine</h3>
        <p class="says">A converted flow compared step by step against the code it replaces, over real
        data, cent-exact on money and exact on everything else.</p>
        <button data-open="scripts/mrr_period_oracle.py">read the code</button></article>
    </div>
  </div>
  <div class="code"><header><b class="at">nothing open</b><span class="note"></span></header>
    <p class="idle">Pick anything on the left to read its source here.</p><pre></pre></div>
</div>

<script>
const shell = document.querySelector('.shell');
const at = shell.querySelector('.code .at');
const note = shell.querySelector('.code .note');
const idle = shell.querySelector('.code .idle');
const body = shell.querySelector('.code pre');

async function open(path) {{
  for (const item of shell.querySelectorAll('.item'))
    item.classList.toggle('reading', item.dataset.path === path);
  at.textContent = path;
  note.textContent = 'reading...';
  idle.hidden = true;
  try {{
    const answer = await fetch('/api/source?path=' + encodeURIComponent(path));
    const text = await answer.text();
    if (!answer.ok) throw new Error(JSON.parse(text).error);
    body.textContent = text;
    note.textContent = text.split('\\n').length.toLocaleString() + ' lines';
  }} catch (err) {{
    body.textContent = '';
    note.textContent = String(err.message || err);
  }}
  shell.querySelector('.code').scrollTop = 0;
}}

shell.addEventListener('click', e => {{
  const button = e.target.closest('[data-open]');
  if (button) open(button.dataset.open);
}});
</script>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = build().replace("{modules}", str(len(LAYERS)))
    (OUT_DIR / "page.html").write_text(page, encoding="utf-8")
    # An empty manifest, so the service discovers this as something to serve. It has no ports: it is
    # a document about the code rather than a run of anything.
    (OUT_DIR / "ports.json").write_text("{}", encoding="utf-8")
    print(f"{len(REGISTRY.names())} tools, {len(LAYERS)} modules -> {OUT_DIR / 'page.html'} ({len(page):,} bytes)")


if __name__ == "__main__":
    main()
