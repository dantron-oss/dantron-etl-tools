// A file a step names: its full path and what the viewer knows about it, its first rows, and a
// button that asks the viewer to show it in this computer's file manager.
//
// The page never sends a path. It names the pipeline, the step and the setting, and the server looks
// the path up. Text from the server only ever goes in through textContent or an input's value.
window.etlFiles = (() => {
  const tokenTag = document.querySelector('meta[name="run-token"]');
  const token = tokenTag ? tokenTag.content : '';
  const api = (pipeline, what, query) =>
    '/' + encodeURIComponent(pipeline) + '/api/' + what + '?' + new URLSearchParams(query);

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(text, enabled) {
    const made = el('button', '', text);
    made.type = 'button';
    made.disabled = !enabled;
    return made;
  }

  // The server's answer as data, or { error } with the server's own sentence.
  async function ask(url, options) {
    let answer;
    try { answer = await fetch(url, options); } catch (err) { return { error: 'The viewer stopped answering.' }; }
    let body = {};
    try { body = await answer.json(); } catch (err) { body = {}; }
    return answer.ok ? body : { error: body.error || 'The viewer answered ' + answer.status + '.' };
  }

  function grid(columns, rows) {
    const table = el('table', 'fgrid');
    const head = table.createTHead().insertRow();
    for (const name of columns) head.append(el('th', '', name));
    const body = table.createTBody();
    for (const row of rows) {
      const line = body.insertRow();
      for (let at = 0; at < columns.length; at++) {
        const value = row[at];
        line.append(value === null || value === undefined ? el('td', 'nul', 'null') : el('td', '', value));
      }
    }
    const scroll = el('div', 'scroll');
    scroll.append(table);
    return scroll;
  }

  function showPreview(where, shown, again) {
    if (shown.error) { where.replaceChildren(el('p', 'bad', shown.error)); return; }
    const parts = [];
    if (shown.kind === 'workbook' && shown.sheets.length) {
      const label = el('label', 'fsheet', 'sheet ');
      const pick = el('select');
      for (const name of shown.sheets) {
        const option = el('option', '', name);
        option.value = name;
        option.selected = name === shown.sheet;
        pick.append(option);
      }
      pick.addEventListener('change', () => again(pick.value));
      label.append(pick);
      parts.push(label);
    }
    if ((shown.kind === 'table' || shown.kind === 'workbook') && shown.columns.length)
      parts.push(grid(shown.columns, shown.rows));
    if (shown.kind === 'text') parts.push(el('pre', 'ftext', shown.lines.join('\n')));
    parts.push(el('p', 'fnote', shown.note));
    where.replaceChildren(...parts);
  }

  // One file, from the facts the server sent: the full path as plain text, what is known, and the
  // buttons. "Open location" appears only where this machine has a file manager to open.
  function box(pipeline, info) {
    const root = el('div', 'filebox');
    if (info.error) { root.append(el('p', 'bad', info.error)); return root; }
    const field = el('p', 'fpath', info.path || info.written);
    field.title = 'the full path of the file';
    const facts = el('ul', 'ffacts');
    for (const line of info.said) facts.append(el('li', '', line));
    const open = button('Open location', info.can_open);
    const look = button('Preview', info.can_preview);
    const buttons = el('p', 'fbuttons');
    if (info.can_open) buttons.append(open);
    buttons.append(look);
    const said = el('p', 'fanswer');
    const shown = el('div', 'fpreview');
    const which = { pipeline: pipeline, step: info.step, setting: info.setting };

    open.addEventListener('click', async () => {
      open.disabled = true;
      said.textContent = 'Asking the viewer to open it…';
      const answer = await ask('/api/open-location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Run-Token': token },
        body: JSON.stringify(which),
      });
      said.textContent = answer.error || answer.message;
      open.disabled = false;
    });
    const load = async sheet => {
      shown.replaceChildren(el('p', 'fnote', 'Reading the file…'));
      const query = { step: info.step, setting: info.setting };
      if (sheet) query.sheet = sheet;
      showPreview(shown, await ask(api(pipeline, 'file-preview', query), { headers: { 'X-Run-Token': token } }), load);
    };
    look.addEventListener('click', () => load());
    root.append(field, facts, buttons, said, shown);
    return root;
  }

  // A run's page: a "Show file" button on each setting that names a file, when a viewer serves it.
  const bar = document.querySelector('.runbar[data-pipeline]');
  if (token && bar) {
    const pipeline = bar.dataset.pipeline;
    for (const setting of document.querySelectorAll('.detail[data-step] .cfg[data-file]')) {
      const step = setting.closest('.detail').dataset.step;
      const show = button('Show file', true);
      show.classList.add('fshow');
      show.addEventListener('click', async () => {
        show.disabled = true;
        const info = await ask(api(pipeline, 'file', { step: step, setting: setting.dataset.file }),
          { headers: { 'X-Run-Token': token } });
        show.remove();
        setting.after(box(pipeline, info));
      });
      setting.lastElementChild.append(show);
    }
  }

  return { box: box };
})();
