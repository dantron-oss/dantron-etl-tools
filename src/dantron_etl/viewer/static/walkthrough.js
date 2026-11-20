const wrap = document.querySelector('.wrap');

// The step being read. A panel that arrives after the reader has moved on is not drawn over the
// new step.
let chosen = null;

// Show one step: its configuration and its data beside the flow. Click a node or a line in the
// step list. Async because the data panel is fetched (see ensurePanel).
async function choose(stepId, readerAsked = true) {
  chosen = stepId;
  // Hide .stepdata too: the panel of the step just left goes with the rest of the selection, or its
  // rows would sit under the new step's name while the fetch runs.
  for (const section of wrap.querySelectorAll('.detail, .stepdata'))
    section.hidden = section.dataset.step !== stepId;
  for (const node of wrap.querySelectorAll('.node'))
    node.classList.toggle('chosen', node.id === 'node-' + stepId);
  for (const line of wrap.querySelectorAll('.index button'))
    line.setAttribute('aria-current', String(line.dataset.goto === stepId));
  // Trace only when the reader asked. The page chooses its first step on load, and tracing then would
  // fade most of the canvas before any question was asked (a late step lights 427 of 672 nodes).
  // Do not make this unconditional.
  if (readerAsked) traceFrom(stepId);

  await ensurePanel(stepId);
  // The panel arrives hidden, after the pass above ran, so show it again now. Unless the reader has
  // chosen another step meanwhile: then this panel must stay hidden.
  if (chosen !== stepId) return;
  for (const section of wrap.querySelectorAll('.stepdata'))
    section.hidden = section.dataset.step !== stepId;
  // Re-apply the entity picker to the new panel through pick(), so one function decides what a chosen
  // entity looks like; without it the panel opens on nothing.
  pick(picked);
  const open = wrap.querySelector(`.stepdata[data-step="${stepId}"] details[open] table.data[data-lazy]`);
  if (open) { delete open.dataset.lazy; live(open); }
}

// One notice for a panel that could not be fetched, made once and moved, so five clicks give one
// message.
let notice = null;

// Fetch a step's panel when it is opened and keep it in the page. Embedding every panel made a real
// page 8.98 MB, 7.76 MB of it panels. See docs/design/viewer.md.
async function ensurePanel(stepId) {
  const pane = wrap.querySelector('.datapane');
  if (notice) notice.hidden = true;
  // A panel already in the page is used as is, so clicking back and forth does not fetch or add twice.
  if (wrap.querySelector(`.stepdata[data-step="${stepId}"]`)) return;
  try {
    const answer = await fetch('api/panel?step=' + encodeURIComponent(stepId));
    if (!answer.ok) throw new Error('the service answered ' + answer.status);
    const markup = await answer.text();
    pane.insertAdjacentHTML('beforeend', markup);
  } catch (err) {
    // Nothing answered (a refusal, a dropped connection, or the page opened as a plain file): say so once,
    // where the panel would be. Not a .stepdata section, so the step is asked for again later.
    if (!notice) {
      notice = document.createElement('p');
      notice.className = 'more';
      pane.append(notice);
    }
    notice.hidden = false;
    notice.textContent = 'this step is drawn by the preview service, which is not answering';
  }
}

// Following one connection. On a flow of 900 connections a line cannot be traced by eye, so clicking
// a step lights its whole chain to the ends in both directions and fades the rest.
let WIRING = null;

// Where a connection goes back to when it stops being lit: before the first step box. Found on first
// use, since `flow` is declared later. A container counts as a step box, because its steps are
// drawn inside it and insertBefore needs a direct child of the drawing.
let FIRST_NODE;
function firstNode() {
  if (FIRST_NODE === undefined) FIRST_NODE = flow ? flow.querySelector('.group, .node') : null;
  return FIRST_NODE;
}

// The front layer, found once: `paint` asks for it per element.
let FRONT;
function front() {
  if (FRONT === undefined) FRONT = flow ? flow.querySelector('.front') : null;
  return FRONT;
}

// Which container's title is raised, and the sibling it goes back in front of.
let LIFTED = null;
let LIFTED_HOME = null;

function putTitleBack() {
  if (!LIFTED) return;
  const title = wrap.querySelector('text.gtitle.lifted');
  const plate = wrap.querySelector('rect.gplate.lifted');
  // Put the plate and the title back in their written order, in front of the remembered sibling.
  if (plate) {
    plate.classList.remove('lifted');
    LIFTED.insertBefore(plate, LIFTED_HOME);
  }
  if (title) {
    title.classList.remove('lifted');
    LIFTED.insertBefore(title, LIFTED_HOME);
  }
  LIFTED.classList.remove('reading');
  LIFTED = null;
  LIFTED_HOME = null;
}

function paint(el, state) {
  el.classList.remove('direct', 'upstream', 'downstream', 'dim');
  el.classList.add(state);
  // A lit connection is raised above the steps, so its caption at the midpoint is not covered by a
  // step that sits between its ends.
  if (el.classList.contains('edge') && flow) {
    if (state === 'dim') {
      const at = firstNode();
      if (at && el.nextElementSibling !== at) flow.insertBefore(el, at);
    } else {
      // Below the front layer, so a raised container title stays on top.
      const layer = front();
      if (layer) flow.insertBefore(el, layer);
      else flow.appendChild(el);
    }
  }
}

// Where every connection runs once a container has been closed. Run over the whole drawing, since
// a connection's two ends can be in two containers; each end asks whether its container is closed.
function repointGroups() {
  // Where each closed container's own bar stands, read off the drawing. Read rather than remembered:
  // the position is already stated in the markup, and a second copy of it drifts.
  const shut = new Map();
  for (const box of wrap.querySelectorAll('[data-group].collapsed')) {
    const bar = box.querySelector('rect.gpill');
    shut.set(box.dataset.group, {
      x: Number(bar.getAttribute('x')), y: Number(bar.getAttribute('y')),
      w: Number(bar.getAttribute('width')), h: Number(bar.getAttribute('height')),
    });
  }
  for (const edge of wrap.querySelectorAll('.edge')) {
    const fromGroup = edge.dataset.fromGroup, toGroup = edge.dataset.toGroup;
    const a = shut.get(fromGroup), b = shut.get(toGroup);
    // Both ends in one closed container: the container's own wiring, put away with it.
    edge.classList.toggle('gone', Boolean(a && b && fromGroup === toGroup));
    // An end in a closed container attaches to its bar; an open end goes back to the coordinates the
    // page wrote down for it (a hidden node measures as nothing).
    const x1 = a ? a.x + a.w : Number(edge.dataset.sx);
    const y1 = a ? a.y + a.h / 2 : Number(edge.dataset.sy);
    const x2 = b ? b.x : Number(edge.dataset.tx);
    const y2 = b ? b.y + b.h / 2 : Number(edge.dataset.ty);
    const mid = (x1 + x2) / 2;
    const d = 'M' + x1 + ',' + y1 + ' C' + mid + ',' + y1 + ' ' + mid + ',' + y2 + ' ' + x2 + ',' + y2;
    // Move the visible line and its wide transparent twin together, or clicks land on a neighbour.
    for (const path of edge.querySelectorAll('path')) path.setAttribute('d', d);
    const label = edge.querySelector('text');
    if (label) { label.setAttribute('x', mid); label.setAttribute('y', (y1 + y2) / 2 - 6); }
  }
}

// The step whose chain is lit, or null: what the reader last asked for.
let TRACED = null;

function traceFrom(stepId) {
  if (!WIRING) {
    // Who feeds whom, read off the drawing once and kept.
    const groups = [...wrap.querySelectorAll('.edge')];
    const feeds = new Map(), fedBy = new Map();
    for (const edge of groups) {
      if (!feeds.has(edge.dataset.from)) feeds.set(edge.dataset.from, []);
      if (!fedBy.has(edge.dataset.to)) fedBy.set(edge.dataset.to, []);
      feeds.get(edge.dataset.from).push({ edge: edge, other: edge.dataset.to });
      fedBy.get(edge.dataset.to).push({ edge: edge, other: edge.dataset.from });
    }
    WIRING = { groups: groups, feeds: feeds, fedBy: fedBy };
  }
  const lit = new Map();
  const marked = new Map([[stepId, 'direct']]);
  for (const side of [['upstream', WIRING.fedBy], ['downstream', WIRING.feeds]]) {
    // A worklist with a visited set, so a flow that splits and rejoins is walked once.
    const seen = new Set([stepId]);
    const work = [stepId];
    while (work.length) {
      const at = work.pop();
      for (const link of side[1].get(at) || []) {
        // Steps wired directly to the chosen one get their own state; the others say which way data runs.
        const state = at === stepId ? 'direct' : side[0];
        lit.set(link.edge, state);
        if (marked.get(link.other) !== 'direct') marked.set(link.other, state);
        if (!seen.has(link.other)) { seen.add(link.other); work.push(link.other); }
      }
    }
  }
  TRACED = stepId;
  for (const edge of WIRING.groups) paint(edge, lit.get(edge) || 'dim');
  for (const node of wrap.querySelectorAll('.node'))
    paint(node, marked.get(node.id.replace(/^node-/, '')) || 'dim');
}

function clearTrace() {
  TRACED = null;
  // Every state named out, rather than whichever ones this trace happened to set: a clear that
  // misses one leaves part of the canvas dark with nothing on the page left to say why.
  for (const el of wrap.querySelectorAll('.edge, .node'))
    el.classList.remove('direct', 'upstream', 'downstream', 'dim');
  // Put every lifted connection back under the steps in one pass.
  const at = firstNode();
  if (flow && at)
    for (const edge of wrap.querySelectorAll('.edge')) flow.insertBefore(edge, at);
}

wrap.addEventListener('click', e => {
  const rail = e.target.closest('.rail');
  if (rail) { wrap.classList.toggle('railed'); return; }

  const fold = e.target.closest('.folds button');
  if (fold) {
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
  }

  // The clear button sits outside `.zoom`, whose handler treats every button there as a zoom control.
  const back = e.target.closest('[data-trace]');
  if (back) { clearTrace(); return; }

  // A container opening and closing, found by attribute so the control can be drawn any way.
  const shutter = e.target.closest('[data-collapse]');
  if (shutter) {
    const box = wrap.querySelector('#group-' + shutter.dataset.collapse);
    // Toggled, never added. A container that closes and cannot be opened again is a section of the
    // flow the reader has permanently lost, with the page's own control the thing that lost it.
    box.classList.toggle('collapsed');
    repointGroups();
    return;
  }

  const edge = e.target.closest('.edge');
  if (edge) {
    // Clicking a line lights it and its two ends; the step being read does not change.
    e.preventDefault();
    for (const other of wrap.querySelectorAll('.edge')) paint(other, other === edge ? 'direct' : 'dim');
    for (const node of wrap.querySelectorAll('.node')) paint(node, 'dim');
    for (const end of [edge.dataset.from, edge.dataset.to]) {
      const at = wrap.querySelector('#node-' + end);
      if (at) paint(at, 'direct');
    }
    return;
  }

  // A step chosen from the list is also brought into view on the flow: a reader who could not find
  // it by scanning still needs to know where it sits, or the list answers only half the question.
  const line = e.target.closest('.index button');
  if (line) {
    choose(line.dataset.goto);
    const found = wrap.querySelector('#node-' + line.dataset.goto);
    if (found) found.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
    return;
  }

  const node = e.target.closest('.node');
  if (node) {
    e.preventDefault();
    const step = node.id.replace(/^node-/, '');
    // Clicking the already-traced step puts the trace away; clicking another step traces that one.
    if (TRACED === step) { clearTrace(); TRACED = null; }
    // The step's own id, not the element's; a test reads this call.
    else { choose(node.id.replace(/^node-/, '')); }
    return;
  }
  // A click on the container itself (outline, header, inside) raises its title, since titles are
  // longer than their boxes and overlap. Checked last: steps and the collapse control are inside the
  // container element. A raised title is outside its container, so it is matched as a second way in.
  const box = e.target.closest('.group') || (e.target.closest('.lifted') && LIFTED);
  if (box) {
    e.preventDefault();
    // Clicking the container already forward puts it back, the same way clicking a lit step puts
    // its chain away. Clicking a different one moves the title on, so only ever one is raised.
    if (LIFTED === box) putTitleBack();
    else {
      putTitleBack();
      const title = box.querySelector('text.gtitle');
      // Read off the drawing here rather than through the memo above: that memo exists because
      // `paint` asks per element, and this asks once per click.
      const layer = flow.querySelector('.front');
      if (title && layer) {
        LIFTED = box;
        LIFTED_HOME = title.nextSibling;
        box.classList.add('reading');
        const plate = box.querySelector('rect.gplate');
        if (plate) {
          plate.classList.add('lifted');
          layer.appendChild(plate);
        }
        title.classList.add('lifted');
        layer.appendChild(title);
      }
    }
    return;
  }

  // A click on empty canvas puts away whatever is showing: the trace or a raised title.
  if ((TRACED !== null || LIFTED !== null) && e.target.closest('.canvas') && !e.target.closest('.group')) {
    clearTrace();
    TRACED = null;
    putTitleBack();
  }
});

// The two grips. Sizes live in CSS variables on the wrapper, so the layout is one grid and the
// drag is two numbers -- nothing is measured, positioned or re-flowed by hand.
for (const grip of wrap.querySelectorAll('.grip')) {
  grip.addEventListener('pointerdown', down => {
    const vertical = grip.classList.contains('grip-v');
    grip.setPointerCapture(down.pointerId);
    grip.classList.add('dragging');
    document.body.classList.add('resizing');

    const move = at => {
      const box = wrap.querySelector('.panes').getBoundingClientRect();
      const share = vertical
        ? (at.clientX - box.left) / box.width
        : (at.clientY - box.top) / box.height;
      // Kept away from both edges: a pane dragged to nothing is a pane nobody can drag back.
      const bounded = Math.min(0.85, Math.max(0.15, share));
      wrap.style.setProperty(vertical ? '--split-x' : '--split-y', (bounded * 100).toFixed(1) + '%');
    };
    const up = () => {
      grip.classList.remove('dragging');
      document.body.classList.remove('resizing');
      grip.removeEventListener('pointermove', move);
      grip.removeEventListener('pointerup', up);
    };
    grip.addEventListener('pointermove', move);
    grip.addEventListener('pointerup', up);
  });
}
// Which entity the reader is following, or '' for everyone. Kept because a panel that arrives after
// it was chosen has to be put on the same one.
let picked = '';

// Follow one entity: every open port asks the service again for that entity's rows. A panel
// arrives with its view hidden, so this also shows it.
function pick(key) {
  const changed = key !== picked;
  picked = key;
  for (const b of wrap.querySelectorAll('.picker button'))
    b.setAttribute('aria-pressed', String(b.dataset.pick === key));
  for (const v of wrap.querySelectorAll('.view')) v.hidden = false;
  if (!changed || !LIVE) return;
  for (const table of wrap.querySelectorAll('table.data[data-port]:not([data-lazy])')) {
    table.dataset.offset = 0;
    live(table);
  }
}

// The picker's search: every entity the run can be followed by is asked of the service, a page of
// matches at a time, and each match is a button.
const pickBox = wrap.querySelector('.picker input.pick');
let picks = null;
let pickNote = null;

async function findEntities() {
  const qs = new URLSearchParams({ q: pickBox.value.trim(), limit: 50 });
  let answer;
  try { const r = await fetch('api/entities?' + qs); answer = await r.json(); if (!r.ok) throw new Error(answer.error); }
  catch (err) { pickNote.textContent = String(err.message || err); return; }
  const everyone = document.createElement('button');
  everyone.type = 'button';
  everyone.dataset.pick = '';
  everyone.textContent = 'everyone';
  picks.replaceChildren(everyone, ...answer.values.map(value => {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.pick = value;
    b.textContent = value;
    b.setAttribute('aria-pressed', String(value === picked));
    return b;
  }));
  pickNote.textContent = answer.reason
    || answer.matched + ' of ' + answer.total + ' match' + (answer.matched > answer.values.length
      ? ', showing the first ' + answer.values.length : '');
}

if (pickBox) {
  picks = document.createElement('div');
  picks.className = 'picks';
  pickNote = document.createElement('span');
  pickNote.className = 'picknote';
  pickBox.closest('.picker').append(pickNote, picks);
  let searching;
  pickBox.addEventListener('input', () => {
    if (!pickBox.value.trim() && picked) pick('');
    clearTimeout(searching);
    searching = setTimeout(findEntities, 180);
  });
  pickBox.addEventListener('focus', () => { if (!picks.children.length) findEntities(); });
}
// Two modes: with the server, a filter covers the whole port and fetches a window; as a plain file,
// the boxes narrow the rows in the page. The note under the table says which happened.
let LIVE = false;

function boxesOf(table) { return [...table.querySelectorAll('tr.flts input.flt')]; }

// Each box says which cell it narrows, because a status column shifts the cells by one.
function drawnFilter(table) {
  const want = boxesOf(table)
    .map(b => [Number(b.dataset.c), b.value.trim().toLowerCase()])
    .filter(([, text]) => text);
  let hits = 0;
  for (const tr of table.tBodies[0].rows) {
    let ok = true;
    for (const [at, text] of want) {
      const cell = tr.cells[at];
      if (!cell || !cell.textContent.toLowerCase().includes(text)) { ok = false; break; }
    }
    tr.classList.toggle('hide', !ok);
    if (ok) hits++;
  }
  const note = table.closest('.tbl').querySelector('.match');
  note.hidden = !want.length;
  note.textContent = want.length ? hits + ' of ' + table.dataset.drawn + ' rows on this page match' : '';
}

// A step's verdict on one row: a mark a reader sees and a reason a screen reader reads (the column,
// and what the cell held). Each level's mark is written out, so a level with none shows as empty.
function mark(m) {
  const td = document.createElement('td');
  td.className = 'lvl ' + m.level;
  const drawn = { ok: '\u2713', warning: '\u25b2', error: '\u2715' };
  td.textContent = drawn[m.level] || '?';
  td.title = m.level + (m.reason || []).map(r => ' \u2014 ' + r.column + ' ' + r.detail).join(' ');
  td.setAttribute('aria-label', td.title);
  return td;
}

function cell(v) {
  const td = document.createElement('td');
  if (v === null) {
    const s = document.createElement('span');
    s.className = 'nul';
    s.textContent = 'null';
    td.append(s);
  } else td.textContent = v;
  return td;
}

async function live(table) {
  const port = table.dataset.port;
  const qs = new URLSearchParams({ port, offset: table.dataset.offset, limit: table.dataset.limit });
  for (const b of boxesOf(table)) if (b.value.trim()) qs.set('f.' + b.dataset.col, b.value.trim());
  // The level is its own parameter, not a column filter: as a substring, 'warn' would match 'warning',
  // and a data column called level would confuse it. The server accepts only published levels.
  const levelPick = table.querySelector('tr.flts select.lvl');
  if (levelPick && levelPick.value) qs.set('level', levelPick.value);
  if (picked) qs.set('focus', picked);
  const note = table.closest('.tbl').querySelector('.match');
  const pager = table.closest('.tbl').querySelector('.pager');
  let answer;
  try { const r = await fetch('api/rows?' + qs); answer = await r.json(); if (!r.ok) throw new Error(answer.error); }
  catch (err) { note.hidden = false; note.textContent = String(err.message || err); return; }

  const body = table.tBodies[0];
  // Each mark is taken at its row's index in the window, since that is all that lines them up, and
  // goes before the cells so the status is the first column.
  body.replaceChildren(...answer.rows.map((row, i) => {
    const tr = document.createElement('tr');
    if (answer.status) tr.append(mark(answer.status[i]));
    tr.append(...row.map(cell));
    return tr;
  }));

  const from = answer.rows.length ? answer.offset + 1 : 0;
  const to = answer.offset + answer.rows.length;
  const filtered = answer.matched !== answer.total;
  note.hidden = false;
  const span = from + '-' + to + ' of ';
  // A port with no column naming the entity is shown whole, and the service says why.
  note.textContent = (answer.no_grain ? answer.no_grain + ' ' : '') + (filtered
    ? span + answer.matched.toLocaleString() + ' matching (of ' + answer.total.toLocaleString() + ')'
    : span + answer.total.toLocaleString() + ' rows');
  pager.hidden = false;
  pager.querySelector('.at').textContent = '';
  pager.querySelector('[data-page=prev]').disabled = answer.offset <= 0;
  pager.querySelector('[data-page=next]').disabled = to >= answer.matched;
}

function refresh(table) {
  if (LIVE && table.dataset.port) live(table); else drawnFilter(table);
}

let typing;
wrap.addEventListener('input', e => {
  const box = e.target.closest('input.flt');
  if (!box) return;
  const table = box.closest('table');
  table.dataset.offset = 0;
  clearTimeout(typing);
  typing = setTimeout(() => refresh(table), 180);
});

// A choice asks at once (no typing delay) and goes back to the top of the port.
wrap.addEventListener('change', e => {
  const control = e.target.closest('tr.flts select.lvl');
  if (!control) return;
  const table = control.closest('table');
  table.dataset.offset = 0;
  refresh(table);
});

wrap.addEventListener('click', e => {
  const f = e.target.closest('.flip button');
  if (f) {
    const card = f.closest('.stepdata');
    for (const b of card.querySelectorAll('.flip button'))
      b.setAttribute('aria-pressed', String(b === f));
    for (const side of card.querySelectorAll('.side'))
      side.hidden = side.dataset.side !== f.dataset.side;
    return;
  }
  const b = e.target.closest('.pager button');
  if (!b) return;
  const table = b.closest('.tbl').querySelector('table');
  const step = parseInt(table.dataset.limit, 10);
  const at = parseInt(table.dataset.offset, 10);
  table.dataset.offset = Math.max(0, b.dataset.page === 'next' ? at + step : at - step);
  refresh(table);
});

// Zoom scales the viewBox, so nothing is laid out twice. The lower bound is small enough to fit a wide
// flow (about 8%); the page opens at actual size, because a reader comes to read a step.
const flow = wrap.querySelector('.canvas svg');
const level = wrap.querySelector('.zoom .level');
let zoom = 1;

function applyZoom(next) {
  zoom = Math.min(3, Math.max(0.04, next));
  flow.style.width = (flow.dataset.w * zoom) + 'px';
  flow.style.height = (flow.dataset.h * zoom) + 'px';
  level.textContent = Math.round(zoom * 100) + '%';
}

function fitZoom() {
  // The pane's own width, less the room a scrollbar takes, so "fit" does not leave one behind.
  const room = flow.parentElement.clientWidth - 8;
  applyZoom(room / flow.dataset.w);
}

wrap.addEventListener('click', e => {
  const button = e.target.closest('.zoom button');
  if (!button) return;
  const how = button.dataset.zoom;
  if (how === 'fit') fitZoom();
  else if (how === 'reset') applyZoom(1);
  else applyZoom(how === 'in' ? zoom * 1.25 : zoom / 1.25);
});

// Ctrl or command with the wheel, which is what a hand already reaches for, and only over the
// drawing -- taking the wheel anywhere else would break scrolling the page.
wrap.querySelector('.canvas').addEventListener('wheel', e => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  applyZoom(zoom * (e.deltaY < 0 ? 1.1 : 1 / 1.1));
}, { passive: false });

// A port is fetched the first time it is opened, and not before: a page of closed ports costs
// nothing at all, which is what keeps it the same size however many steps the flow grows to.
wrap.addEventListener('toggle', e => {
  const details = e.target;
  if (!details.open || !LIVE) return;
  const table = details.querySelector('table.data[data-lazy]');
  if (!table) return;
  delete table.dataset.lazy;
  live(table);
}, true);

fetch('api/ports')
  .then(r => { LIVE = r.ok; })
  .catch(() => { LIVE = false; })
  .finally(() => {
    if (LIVE) return;
    // Opened as a plain file with nothing serving it: say so once, where the rows would have been,
    // rather than leaving every port looking empty.
    for (const table of wrap.querySelectorAll('table.data[data-lazy]')) {
      const note = table.closest('.tbl').querySelector('.match');
      note.hidden = false;
      note.textContent = 'rows are served by the preview service, which is not answering';
    }
  });
wrap.addEventListener('click', e => {
  const b = e.target.closest('.picker button');
  if (!b) return;
  pickBox.value = b.dataset.pick;
  pick(b.dataset.pick);
});
pick('');
// Open on the first step without tracing it (see choose()).
choose(wrap.dataset.first, false);
applyZoom(1);
