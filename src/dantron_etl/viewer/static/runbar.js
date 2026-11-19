// The run buttons, shared by the home page, a pipeline's page and a run's page.
//
// The server draws the buttons and the progress line. This wires them: a preview is posted at once;
// a real run first shows every write and asks the reader to type the pipeline name. While a run
// goes, the progress line says which step it is on. When it ends the page is loaded again.
//
// Text from the data only ever goes in through textContent.
(() => {
  const tokenTag = document.querySelector('meta[name="run-token"]');
  const token = tokenTag ? tokenTag.content : '';
  const base = name => '/' + encodeURIComponent(name) + '/';
  const each = (attr, name) => document.querySelectorAll('[' + attr + '="' + CSS.escape(name) + '"]');

  function say(name, text) {
    for (const line of each('data-progress', name)) line.textContent = text;
  }

  function busy(name, on) {
    for (const button of document.querySelectorAll('[data-run][data-pipeline="' + CSS.escape(name) + '"]'))
      button.disabled = on;
  }

  function showSteps(name, events) {
    for (const list of each('data-steps', name))
      list.replaceChildren(...events.map(event => {
        const item = document.createElement('li');
        item.className = event.level;
        item.textContent = event.step + ' (' + event.tool + '): ' + event.rows_out + ' rows, ' + event.ms + ' ms';
        return item;
      }));
  }

  async function post(payload) {
    const answer = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Run-Token': token },
      body: JSON.stringify(payload),
    });
    let body = {};
    try { body = await answer.json(); } catch (err) { body = { error: 'the viewer answered ' + answer.status }; }
    return { status: answer.status, body: body };
  }

  async function watch(name) {
    busy(name, true);
    for (;;) {
      let now;
      try {
        const answer = await fetch(base(name) + 'api/progress');
        now = await answer.json();
      } catch (err) {
        say(name, 'The viewer stopped answering.');
        busy(name, false);
        return;
      }
      showSteps(name, now.events);
      const last = now.events[now.events.length - 1];
      if (now.state === 'running') {
        say(name, 'Running a ' + now.kind + ': ' + now.events.length + ' of ' + now.total + ' steps done'
          + (last ? ', last ' + last.step : '') + '.');
      } else if (now.state === 'idle') {
        busy(name, false);
        return;
      } else {
        say(name, 'The run ' + now.state + (now.failed_step ? ' at step ' + now.failed_step : '') + '. Loading it.');
        const line = each('data-progress', name)[0];
        const then = line && line.dataset.then;
        if (then) location.assign(then); else location.reload();
        return;
      }
      await new Promise(done => setTimeout(done, 400));
    }
  }

  async function preview(name) {
    busy(name, true);
    say(name, 'Starting a preview.');
    const answer = await post({ pipeline: name, kind: 'preview' });
    if (answer.status === 202) watch(name);
    else { say(name, answer.body.error || 'The run did not start.'); busy(name, false); }
  }

  function cell(row, text) {
    const td = document.createElement('td');
    td.textContent = text;
    row.append(td);
  }

  function writesTable(plan) {
    if (!plan.writes.length) {
      const none = document.createElement('p');
      none.textContent = 'This pipeline writes nothing.';
      return none;
    }
    const table = document.createElement('table');
    const head = table.createTHead().insertRow();
    for (const title of ['step', 'what', 'target', 'detail']) {
      const th = document.createElement('th');
      th.textContent = title;
      head.append(th);
    }
    const body = table.createTBody();
    for (const write of plan.writes) {
      const row = body.insertRow();
      cell(row, write.step);
      cell(row, write.tool + ' (' + write.kind + ')');
      cell(row, write.target);
      cell(row, write.detail);
    }
    return table;
  }

  // The real run: every write shown, then the reader types the pipeline name before anything runs.
  async function real(name) {
    let plan;
    try {
      // The plan loads the pipeline, which may run project code, so it needs the token too.
      const answer = await fetch(base(name) + 'api/plan', { headers: { 'X-Run-Token': token } });
      plan = await answer.json();
      if (!answer.ok) throw new Error(plan.error);
    } catch (err) {
      say(name, String(err.message || err));
      return;
    }
    if (plan.problem) { say(name, plan.problem); return; }

    const dialog = document.createElement('dialog');
    dialog.className = 'confirm';
    const heading = document.createElement('h2');
    heading.textContent = 'A real run of ' + name + ' writes these';
    const writes = document.createElement('div');
    writes.append(writesTable(plan));
    const label = document.createElement('label');
    label.textContent = 'Type the pipeline name, ' + name + ', to confirm: ';
    const typed = document.createElement('input');
    typed.type = 'text';
    typed.autocomplete = 'off';
    label.append(typed);
    const said = document.createElement('p');
    said.className = 'said';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.textContent = 'Cancel';
    const go = document.createElement('button');
    go.type = 'button';
    go.className = 'real';
    go.textContent = 'Write these';
    go.disabled = true;
    const buttons = document.createElement('p');
    buttons.className = 'buttons';
    buttons.append(cancel, go);
    dialog.append(heading, writes, label, said, buttons);

    typed.addEventListener('input', () => { go.disabled = typed.value !== name; });
    cancel.addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => dialog.remove());
    go.addEventListener('click', async () => {
      go.disabled = true;
      const answer = await post({ pipeline: name, kind: 'real', confirm: typed.value, digest: plan.digest });
      if (answer.status === 202) { dialog.close(); watch(name); return; }
      said.textContent = answer.body.error || 'The run did not start.';
      // What it writes changed since it was shown: show the new list and ask again.
      if (answer.body.plan && answer.body.plan.digest) {
        plan = answer.body.plan;
        writes.replaceChildren(writesTable(plan));
        typed.value = '';
      }
    });
    document.body.append(dialog);
    dialog.showModal();
    typed.focus();
  }

  document.addEventListener('click', e => {
    const button = e.target.closest('[data-run]');
    if (!button) return;
    const name = button.dataset.pipeline;
    if (button.dataset.run === 'real') real(name); else preview(name);
  });

  for (const line of document.querySelectorAll('[data-progress][data-running="1"]')) watch(line.dataset.progress);
})();
