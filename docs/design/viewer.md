# The viewer: design notes

Why the viewer (`dantron-etl view`) works as it does. The code in `src/dantron_etl/viewer/` says what
it does; these notes keep the decisions, the alternatives that lost, and what was measured.

## What the owner asked for

Owner's approval, 2026-11-18: opening the viewer must be "one easy command in their terminal, or
something that they can click", then "select the pipeline from the listing and then view". So:

- `dantron-etl view`, no arguments, from a project folder. It finds the pipelines, starts the server on
  127.0.0.1, opens the browser and prints a clickable address. A second `view` reuses a running viewer
  instead of failing on the port. The server stays in the foreground with a "press ctrl-c to stop" line.
- A home page lists every pipeline: name, steps, when it last ran, whether that run succeeded.
  Opening the viewer never runs anything.
- `dantron-etl new` writes a starter project, a double-click launcher for Windows and macOS, and a VS
  Code task, so there is something to click instead of a command to type.

The viewer is not imported by `import dantron_etl`: a flow that only runs never loads it. The
dependency runs one way: the viewer reads pipelines, and a pipeline never imports the viewer. It
renders and runs; it never transforms data.

## Preview runs and real runs

**The Run button's default is a preview.** A preview runs every step except the writers (tools in the
`output` category: tables and files) and every step fed by one, so nothing leaves the project. The page
says it was a preview and lists the steps held back.

**A real run is a separate, labelled action, never one click.** Before it starts, the page shows exactly
what it will write: each writer step's file or table, and what happens to what is there. The reader
confirms by typing the pipeline's name.

**The confirmation is tied to what was shown.** The server works out a digest (SHA-256) of every write:
step, tool, target, and the writer's whole configuration as compact JSON, so any setting change alters
it, not only the ones put into words. The page sends back the digest it showed. If the pipeline file
changed since, the digests differ and the run is refused with "what this pipeline writes has changed
since it was shown. Look again.", and the new plan is returned. A run is refused before anything runs
or is saved.

**A table write names its database, and the digest covers it.** The words for a table write say which
database the connection's environment variable points at when the list is made: host, port when
stated, and database name (`describe_connection` in `db.py`, used by both the page and the command).
They are part of what the digest covers, so pointing the variable at another database between looking
and confirming refuses the run. A change of password alone leaves the digest as it was: the target is
the same. The user and password are never shown. An address is read the way the Postgres driver reads
it, and if that reading would put any part of the password into the host, port or database name (an
`@` or `/` not written as `%40` or `%2F`, or a password that is also the database name), nothing of
the address is shown and the words say why.

**A run never loses the last good one.** A run is written into a hidden folder first and then renamed
into place. A run that fails or is refused goes into `<name>.failed`, leaving `<name>` (the last good
run) as it was. A later good run deletes the stale failed one. A lock is held while slot folders are
swapped, and a swap interrupted half way is recovered on the next read. A pipeline already running is
refused a second start.

`save_run` lets a project's own script save a run it made, so the viewer lists it like its own.

**A saved run is for its owner only.** The hidden folder a run is written into is made `0700` before
anything goes in, so no other user can open a file in it while it fills; every folder and file in it
is then set to `0700` and `0600` before it is renamed into place, and the runs folder and any folder
above it that the run had to make are `0700` too. `chmod` is used, so the umask cannot loosen it.
Folders that already existed, older runs and the files a pipeline's output steps write are not
touched. A system that refuses the change (Windows keeps only a read-only flag; some mounted drives
refuse) does not fail the run.

**`save_rows: false` keeps the shape of a run, not its rows.** Requested by a second project whose
data must not be copied to disk. The run is the same in memory; what is saved changes. `ports.json`
still lists every port's row count, columns and types, with `file` set to null, and no Parquet or
status file is written; `entities.json` is not written; each panel shows a port's columns and the
reason in place of a grid that would fetch rows; the page has no picker; `run.json` says
`rows_saved: false`, and the server answers a request for rows or entities with the same sentence.
A failure's message is kept, because a run that stopped needs to say why, but the example rows a
`DataError` appends are cut out (`examples_text` is the exact text it appended). A refusal's own
sentence can still name a value; that is the line drawn, and the docs say so. Nothing in the viewer
feeds saved rows back into a run (`cached=` is only ever passed by a caller), so the setting needs
no rule about reuse. **Show file** reads the file a step names, live, and keeps nothing, so it is
unchanged.

## A local-only server with a token

The server binds 127.0.0.1 by default (finding G-01) and answers only this machine:

- **Host check.** A request whose `Host` header is not a loopback name (127.0.0.1, localhost, ::1) is
  refused. This stops a web page elsewhere from reaching the viewer through DNS rebinding.
- **Run token.** The server makes a random token (`secrets.token_urlsafe(32)`) when it starts and writes
  it into every page it serves (`<meta name="run-token">`). Starting a run needs that token, compared in
  constant time, so another site cannot start a run by posting to the port.
- **Origin check.** A POST that carries an `Origin` header must come from a loopback name.
- **Names are looked up, never joined onto paths.** A pipeline or run name in a request is looked up in
  what was found on disk, so a crafted name cannot reach a file outside the project. The source route
  serves only files a run references, never dotfiles.
- **Small bodies.** A POST body is capped; a run request is a few short fields.

When the server is deliberately bound to a non-loopback address (for example inside a container),
the Host check is off and the server's own host is accepted as an origin; the token still applies.

The project is read from disk again on every request, so a new pipeline file appears on the next reload.
Reading a project reads YAML only; project code (a rules loader) runs only when a pipeline is run.

## Looking at a file a step names

Owner's request, 2026-11-19: "click a file and it will open, because then I can see exactly what file
is being brought in", and "like click to open the file location", as a browser's downloads list does.

**Which settings are files.** A setting is a file path when its tool's template marks it
`file_path=True` (the marker the runner's `base_dir` already reads). The viewer reads the marker, not a
list of tool names, so a new tool that reads or writes a file gets **Show file** with no change here.
Only top-level settings are looked at, because only those are joined onto `base_dir` by the runner.

**The path is the run's path.** The pipeline is loaded as a run loads it (rules filled in), then the
runner's own `with_base_dir` is applied from `run_folder(project)`, the folder the viewer's runs pass
as `base_dir`. A path still holding a `{...}` placeholder cannot be worked out and is shown as such.

**The browser never sends a path.** It sends the pipeline, the step and the setting, and the server looks
the path up. So the only files the viewer can preview or open are files a pipeline names, and a
crafted request has nothing to name a different one with. A file outside the project is allowed when
a step names it, since that is the pipeline author's choice, and the page says it is outside. A link
is followed, because the run follows it, and the page names the file it leads to. Looking a path up
loads the pipeline, which may run a project's rules code, so the GETs need the run token, as the plan
does.

**The server opens the location, not the page.** A browser page cannot open a folder on the computer.
The local server can, so "Open location" is a POST with the run token, like starting a run. The
command is chosen by one pure function from plain facts (the platform, the environment, which programs
are on `PATH`), so every case is tested without a desktop, in this order:

| machine | command |
|---|---|
| Windows | `explorer /select, <file>` (Explorer's exit code means nothing, so it is ignored) |
| macOS | `open -R <file>` |
| WSL (a WSL kernel, a WSL session, `explorer.exe` and `wslpath`) | `explorer.exe /select, <wslpath -w file>` |
| A Linux desktop (`DISPLAY` or `WAYLAND_DISPLAY`, and `xdg-open`) | asks the file manager over D-Bus to select the file, falling back to `xdg-open <folder>` |
| none of these | nothing; the page does not offer the button, and leaves the path and Preview |

**Never an editor.** An earlier version opened the file in VS Code when the viewer ran from a VS Code
terminal, which covers a dev container. The owner tried it and turned it down: he wants the file's
location in the file manager, as a browser's "show in folder" gives, not the file open in an editor.
A dev container has no file manager to reach, because the viewer runs inside it, walled off from the
computer's desktop, and its files are on the container's own disk. So there the page shows the full
path and Preview, and no "Open location" button. `has_file_manager` decides this once, when the server
starts, from the same facts `how_to_open` uses, and a test keeps the two in agreement.

A file not written yet has its folder opened instead, and the reader is told. Commands are argument lists, never a shell, every path is
absolute so a file named `-n.csv` cannot read as an option, and the server waits at most five seconds.

## Why the page is plain files

A run's page is a template (`templates/*.html`) with markers such as `@@canvas@@`, plus a stylesheet
and a script (`static/`). Python builds only the fragments that carry data and fills the markers in one
pass, so text from the data that happens to contain `@@` is never searched for another marker. The CSS
and JavaScript are real files, so they can be read, linted and edited as what they are, rather than as
strings inside Python. The saved page inlines them, so it is still one file that opens on its own:
opened without the server it still shows every step and its configuration; the rows need the server.

There is no front-end build step and no framework: one less toolchain for a consumer to install, and
nothing between the file on disk and what the browser runs.

## Keeping the page small

**Panels are fetched, not embedded.** Measured on a real flow of 672 steps, carrying every step's data
panel in the page was 7.76 MB of an 8.98 MB page, and 24,937 search boxes for the browser to lay out
before it could paint, for the 671 steps the reader was not looking at. Each panel is now saved beside
the run and fetched when its step is opened; only the first step's panel is in the page, since the page
opens on it before any fetch could answer. A panel that arrives after the reader has clicked elsewhere is
not drawn over the step they moved to.

**A served port carries no rows in the page.** Its header and search boxes are drawn; a window of rows is
fetched when it is opened. A page then stays the same size as a flow grows by steps. A slice for one
entity is not a port the server knows, so it still carries its rows.

**Search is per column.** The question a reader brings is about a column (which rows carry this account
code), and one box over the whole row would answer a different question. On a served port the search
goes to the server and covers every row; the note under the table says whether a count covers the
window drawn or the whole port.

**The status column.** When a step marked rows (cast_columns' `set_null`, for example), a first column
shows a tick, a warning mark or a cross per row, with the reason (the column and what the cell held) as
the cell's announced text, so a screen reader gets it too. It is drawn only for ports some step marked;
a column of ticks on every port is one a reader stops looking at. It is set apart in the markup, carries
no dtype, and its filter is a choice of three levels, not a text box.

## The canvas

**Everything is placed from the pipeline and the run.** Nothing is placed by hand; the shape comes from
the edges. Steps that share a lane and a depth get separate slots, or one would be drawn on top of the
other and its data would appear from nowhere.

**Containers claim a rectangle of the grid.** A container spanning columns c0..c1 with `deep` members in
its busiest column claims that rectangle plus the slot above for its header, and nothing else is put
there. The outline is then exactly the claim, so "no stranger inside an outline" and "no two outlines
crossing" both reduce to "no two claims share a cell". The margins make that hold across boundaries:
`GROUP_PAD` 12 < (COL_W - NODE_W) / 2 = 28 keeps an outline inside its columns, and `GROUP_PAD` +
`GROUP_HEAD` = 58 < 2 x `LANE_PAD` + `LANE_GAP` = 70 keeps it inside its lane. Change a margin past those
bounds and containers collide where `_slots` never looks. Widest containers are placed first, in a fixed
tie-broken order so two renders of one file draw the same page. A flow with no containers is laid out
exactly as before containers existed.

**What containers cost, measured.** On a real flow of 672 steps, 150 containers and 152 columns, the
canvas grows from 3,146px unboxed to 8,346px boxed (2.65 times). Turning one cause off at a time: the
header row costs 2,240px (a header needs 46 of a slot's 80px but takes the whole slot; a finer grid
could win most of the rest back); the claims' column spans cost 1,840px (claiming only the members'
columns gives 6,506px, but the outline is a bounding box, so a step in between would be inside a
container that does not own it). Placement order is not where the height is: widest-first gives 8,346px,
left to right 8,826, declaration order 8,666, and the busiest column of the busiest lane needs 52 rows on
its own. There is nothing left in the packing.

**A header at the top of a lane takes no extra slot,** because the gap above a lane is already larger
than a header. The lane's name is drawn last, over a halo, because a container starting at column 0
writes its header in that same gap (1 container in 150 on the real flow); the name is the one label a
reader cannot recover from anything else on screen.

**A closed container keeps its width.** The bar it shrinks to carries the title across its width; a
narrower bar would cut every title. Closing gives back height and, mostly, clutter: on the real flow,
483 of 765 connections (63%) are internal to a container and go away with it. Its steps are drawn inside
the container element, so closing is one class on one element. Connections stay at the root of the
drawing, because a lit connection is raised with `appendChild` and one inside a container would leave
its box; each connection names its container at both ends (as `""` when there is none, since
`undefined === undefined` in a browser would make every ordinary connection look internal) and its
end coordinates, so re-opening puts it back without re-deriving geometry in JavaScript.

**Following a connection.** A 1.4px line behind six other tools cannot be traced by eye on a flow of 900
connections. Clicking a step lights its whole chain to the ends in both directions and fades everything
else. Each connection has a transparent 14px twin along the same path to make it clickable. A container's
header strip has a hit area sized from the title's character count (the page is written without a
browser to measure text); it does not cover the whole box, which would steal clicks from connections.

**Zoom** scales the SVG viewBox, so nothing is laid out twice. The lower bound is small because fitting a
wide flow into a pane can mean about 8%; the page opens at actual size because a reader comes to read a
step, and fit is one click away.

**Row counts on containers are grouped** (`11,834`), like every other count on the page; the tooltip
carries the plain figures for copying.

## The catalog page

One card per tool, grouped by category, generated from the registry, so it cannot drift from the tools.
Each setting is drawn as the control it would be in an editable panel, disabled for now: a reader sees
the shape of what they will fill in, and three allowed values read visibly as a choice of three.
