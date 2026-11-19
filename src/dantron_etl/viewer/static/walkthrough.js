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
async function choose(stepId, readerAsked = true) {
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
  const open = wrap.querySelector(`.stepdata[data-step="${stepId}"] details[open] table.data[data-lazy]`);
  if (open) { delete open.dataset.lazy; live(open); }
}

// The one notice standing in for a panel that could not be had, made once and then moved: a reader
// clicking through five steps with nothing serving them is told once, where they are looking,
// rather than handed five copies of the same sentence down the pane.
let notice = null;

// The panel of the step being opened, fetched as it is opened and left in the page afterwards.
// Measured on the real flow, carrying all 672 of them was 7.76 MB of an 8.98 MB page and 24,937
// search boxes for the browser to lay out before it could paint -- for the 671 steps the reader was
// not looking at. Named for what it guarantees rather than for what it hands back, which is
// nothing at all: the caller waits for it and then finds the panel in the page for itself.
async function ensurePanel(stepId) {
  const pane = wrap.querySelector('.datapane');
  if (notice) notice.hidden = true;
  // The page in the document is the cache: a section already answering to this step is the one the
  // service would send back. Without the return, a reader clicking between two steps re-fetches
  // both every time and the second copy lands beside the first, so the pane holds two panels for
  // one step and shows whichever the selection above reaches first.
  if (wrap.querySelector(`.stepdata[data-step="${stepId}"]`)) return;
  try {
    const answer = await fetch('api/panel?step=' + encodeURIComponent(stepId));
    if (!answer.ok) throw new Error('the service answered ' + answer.status);
    const markup = await answer.text();
    pane.insertAdjacentHTML('beforeend', markup);
  } catch (err) {
    // A refusal, a dead socket, or the page opened as a plain file with nothing serving it. Said
    // once, where the panel would have been, because that is where the reader is looking -- a line
    // in the console is a message to whoever wrote the page, and the reader is somebody looking at
    // a figure.
    //
    // Deliberately NOT written as a .stepdata section: the cache above would then find it, this
    // step would never be asked for again, and a reader coming back once the service is answering
    // would be locked out of it with nothing on the page to say why.
    if (!notice) {
      notice = document.createElement('p');
      notice.className = 'more';
      pane.append(notice);
    }
    notice.hidden = false;
    notice.textContent = 'this step is drawn by the preview service, which is not answering';
  }
}

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
function firstNode() {
  if (FIRST_NODE === undefined) FIRST_NODE = flow ? flow.querySelector('.group, .node') : null;
  return FIRST_NODE;
}

// The layer that holds whatever must be above everything, found once. Memoised for the same
// reason `firstNode` is: `paint` asks for it per element, and the layer is the last child of a
// drawing with thousands of them, so a fresh lookup each time is the whole tree each time.
let FRONT;
function front() {
  if (FRONT === undefined) FRONT = flow ? flow.querySelector('.front') : null;
  return FRONT;
}

// Which container's title is forward, and the sibling it sits before when it is not. The title is a
// direct child of its container, so those two are the whole of where it came from -- and remembering
// where rather than re-deriving it is what keeps a put-back exact if the container's parts are ever
// drawn in a different order.
let LIFTED = null;
let LIFTED_HOME = null;

function putTitleBack() {
  if (!LIFTED) return;
  const title = wrap.querySelector('text.gtitle.lifted');
  const plate = wrap.querySelector('rect.gplate.lifted');
  // Back in the order they were written in -- the plate, then the sentence printed on it -- which
  // putting both in front of the title's old next sibling restores without remembering a second
  // position: they were adjacent when they left.
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
  // A lit connection is lifted above the steps, because its caption sits at the midpoint and a long
  // connection's midpoint lands on whatever step happens to sit between its two ends -- which draws
  // after it and covers it. Lifting the whole group keeps the line and its caption together, and
  // only ever applies to the handful a reader is actually following.
  if (el.classList.contains('edge') && flow) {
    if (state === 'dim') {
      const at = firstNode();
      if (at && el.nextElementSibling !== at) flow.insertBefore(el, at);
    } else {
      // Above the steps but below the front layer: appending to the end of the drawing would put a
      // lit line over the one title a reader has brought forward, which is the thing on this canvas
      // that is meant to be above everything.
      const layer = front();
      if (layer) flow.insertBefore(el, layer);
      else flow.appendChild(el);
    }
  }
}

// Where every connection runs, once some container has been closed.
//
// Run over the whole drawing rather than over the container that just moved, because the two ends of
// a connection can sit in two different containers: answering for one of them alone would leave a
// line attached to a box a reader shut a moment ago and has since re-opened. Every connection is
// asked the same two questions -- is the container it starts in closed, is the one it lands in
// closed -- and is drawn from the answers, so the state of the canvas is the state of the canvas
// whatever order the containers were clicked in.
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
    if (label) { label.setAttribute('x', mid); label.setAttribute('y', (y1 + y2) / 2 - 6); }
  }
}

// Which step's chain is lit, or null. Held here rather than read back off the drawing, because
// "is this step the one already traced" is a question about what the reader last asked for, and a
// class on a node answers a different one -- `chosen` stays on through a clear.
let TRACED = null;

function traceFrom(stepId) {
  if (!WIRING) {
    // Who feeds whom, read off the drawing once and then kept. A click touches every connection on
    // the canvas; re-deriving the graph from the DOM each time would make a question about one step
    // cost a pass over the whole flow, twice.
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
    // A worklist rather than one hop. The visited set is not tidiness: a flow that splits and
    // rejoins -- which every one of these does -- would otherwise be walked once per path back to
    // the fork, and a diamond a few steps deep costs more than the canvas is worth.
    const seen = new Set([stepId]);
    const work = [stepId];
    while (work.length) {
      const at = work.pop();
      for (const link of side[1].get(at) || []) {
        // Touching the chosen step is a different fact from being three tools away in the same
        // direction, so it gets its own state: `direct` says what this step is wired to, the two
        // side states say which way the data runs through everything beyond it.
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
  // And back under the steps, in one pass rather than per connection: a lifted connection left
  // above them would draw over a step box for the rest of the session with nothing lit to explain
  // why. Order among the connections themselves does not matter -- they never overlap meaningfully.
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

  // The way back. It sits beside the zoom buttons but outside `.zoom`, because that handler takes
  // every button inside the bar and reads a `data-zoom` off it -- a clear control placed there
  // would fall through to the else and zoom the canvas out on its way to clearing.
  const back = e.target.closest('[data-trace]');
  if (back) { clearTrace(); return; }

  // A container opening and closing. Asked for by attribute rather than by class, so the control can
  // be drawn however it needs to be drawn -- what the page answers is the thing that says which
  // container it belongs to.
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
    // A reader picking one line out of a crowded area is asking where that line goes. Answering a
    // different question -- swapping both panes, scrolling a node into view, re-fetching a port --
    // takes their place away over a click they made to keep it. So this lights the line and its two
    // ends, and the step being read does not move.
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
    // Clicking the step whose chain is already lit puts the chain away and leaves the step chosen:
    // a reader who has followed a connection and finished with it wants the canvas back, not a
    // different step open. Clicking a DIFFERENT step traces that one, which is the ordinary case.
    if (TRACED === step) { clearTrace(); TRACED = null; }
    // Written out again rather than passing `step`: a case pins that what reaches choose() is the
    // step's own id and not the element's, and it reads the call to know. The guard above is the
    // same expression for the same reason.
    else { choose(node.id.replace(/^node-/, '')); }
    return;
  }
  // The container itself -- its outline, its header, the space inside it. Last of the three,
  // because a step and the collapse control are both drawn INSIDE the container element, so a click
  // on either is also a click on the container; both branches above answer first and return.
  //
  // What it does is bring that container's title forward. Titles are longer than the boxes they
  // belong to and the canvas is crowded, so the title a reader is actually reading is the one that
  // has to win -- and clicking the thing you are reading is how you say which one that is.
  // The raised title is the second way in, because raising it moves it out of its container: a
  // reader who clicks the sentence they are reading to put it down again is clicking something that
  // is no longer inside the box it belongs to, and `closest` would answer nothing.
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

  // The canvas itself, between the steps. Nothing there is a thing to select, so a click on it can
  // only mean "put this away" -- which is where a reader's hand goes first, ahead of finding a
  // control. Only when something is showing, so an idle click does not fight the pan; and both
  // things that can be showing are named, because a title raised over a canvas with nothing lit
  // would otherwise have no way down.
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
// Two modes. With the service answering, a filter runs over the WHOLE port and a window of rows
// is fetched; without it -- the page opened as a plain file -- the same boxes narrow the rows that
// were written into the page. The note under each table says which of the two just happened, so a
// count is never read as covering more than it did.
let LIVE = false;

function boxesOf(table) { return [...table.querySelectorAll('tr.flts input.flt')]; }

// Each box is asked which cell it narrows rather than being counted off in order, because on a
// table with a status column the two differ by one: the boxes still stand one per data column, and
// the cells they read have a mark in front of them. Counting held while the two were the same and
// would have been the same off-by-one the boxes themselves were shifted to avoid -- the box for a
// party narrowing the column to its left. Unreachable today, since a marked table is always a
// served one and a served table carries no rows in the document for this to run over, and written
// this way anyway: that is a fact about two other functions, it is nowhere near this one, and a
// filter that is only correct because of it is a filter waiting for the day a marked table is drawn
// with its rows in it.
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
  // A parameter of its own, not one more `f.`: the boxes narrow a column by substring, and a level
  // asked for that way would match `warning` from `warn` and would mean something else entirely on
  // a port whose data has a column called level. The service answers to this name and refuses
  // anything that is not a published level, which is why the control offers only those.
  const levelPick = table.querySelector('tr.flts select.lvl');
  if (levelPick && levelPick.value) qs.set('level', levelPick.value);
  if (picked) qs.set('focus', picked);
  const note = table.closest('.tbl').querySelector('.match');
  const pager = table.closest('.tbl').querySelector('.pager');
  let answer;
  try { const r = await fetch('api/rows?' + qs); answer = await r.json(); if (!r.ok) throw new Error(answer.error); }
  catch (err) { note.hidden = false; note.textContent = String(err.message || err); return; }

  const body = table.tBodies[0];
  // The mark is taken at the row's own index in the window's marks, because the two sequences the
  // service hands back line up by position and nothing else joins them: a running counter over a
  // body that was not cleared, or a lookup by some value, draws a warning against a row that never
  // had one. And it goes in before the cells, or the reader's first column is the data's first
  // column and the status is somewhere in the middle of the table.
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

// Picked rather than typed, so it asks at once: the delay above is there to let somebody finish a
// word, and there is no half-finished choice. Back to the top of the port, like any other
// narrowing -- page three of the whole port is not page three of the flagged rows.
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
// Opened on a step, but NOT traced -- the false is load-time and is argued at choose(). The page
// arrives showing the whole flow lit, with one step chosen in the panes beside it; the trace starts
// when the reader points at something.
choose(wrap.dataset.first, false);
applyZoom(1);
