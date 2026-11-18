# CLAUDE.md

## What this repo is

A library of generic data transformation **tools**. A pipeline is a YAML file that names tools and
configures them. Nobody writes transformation code. A run leaves evidence: every step's output can
be opened and read.

The whole point is one question: *"why is this number what it is?"* The answer has to name a
step, show the rows that went in and came out, and let a non-developer look. Everything in this
repo serves that.

The library is client-neutral. It ships into other people's repositories, and those people have
never heard of the data it was first built against.

## Read these first

| file | what it is for |
|---|---|
| `src/dantron_etl/MANUAL.md` | **How to use the tools to build a pipeline.** Written for agents. Ships inside the package. A test checks it names every tool, every category and every expression function, and states the right tool count. Treat it as authoritative. |
| `README.md` | The public face: the tool list, install, the ideas worth knowing. |
| this file | How to **build and change the library itself**, and the rules behind it. |

This file does not repeat the manual. If a question is about writing a pipeline, the answer is in
the manual.

## Commands

```bash
uv sync --extra dev                         # set up from uv.lock. Plain `uv sync` leaves out pytest, ruff, orjson
.venv/bin/pytest tests/ -q                  # ~3,800 tests, 2-4 min (pyproject puts src/ and the root on the path)
.venv/bin/ruff check src tests scripts      # lint. This is the gate
.venv/bin/python scripts/tool_catalog_page.py          # writes output_files/tool_catalog.{html,json}
.venv/bin/python scripts/pipeline_walkthrough_page.py <name>  # runs pipelines/<name>.yaml, saves the run to
                                                       # output_files/runs/<name>/, opens it in the viewer
.venv/bin/python scripts/etl_architecture_page.py      # the architecture page, read from the library itself
.venv/bin/python scripts/serve_runs.py                 # serves output_files/runs/ on :8899
```

The `Makefile` holds shortcuts for all of these (`make` lists them): `make test`, `make test-all`
(starts the test Postgres too), `make run p=<name>`, `make catalog`. Each is one command and nothing
more, so on Windows, which has no `make`, the command itself is what to run. Keep it that way: logic
goes in Python, where it is tested and runs on both systems.

A project is `pipelines/<name>.yaml`, its inputs in `pipelines/<name>/`, and
`pipelines/<name>/settings.yaml` holding its `rules` and its `focus` column. `fixture` is the worked
example. The runner script never reshapes data: it runs the flow, saves what the run produced, and
draws it.

The versions are fixed in `uv.lock`, which is committed. Keep it that way: without it, a fresh
checkout resolves the newest of everything, and once that meant polars 2.x (40 failing tests) and a
ruff with new default rules (147 lint findings). `pyproject.toml` caps polars below 2 and pins ruff
exactly for the same reason. Moving either one is a deliberate change: raise the bound, fix what
breaks, run `uv lock`, and commit the lock with the fix.

About 160 tests need Postgres. They read a DSN from `DANTRON_TEST_POSTGRES` and **skip** without
it, so a checkout with no database still runs everything else. If you change `read_table`,
`write_table`, `publish_tables` or `db.py`, set it and run them. `compose.yaml` holds a throwaway
server for exactly this, on port 5434 so it misses any other Postgres on the machine:

```bash
uv sync --extra dev --extra postgres        # the drivers those tests use
docker compose up -d --wait                 # data is in memory, gone on `docker compose down`
export DANTRON_TEST_POSTGRES=postgresql://etl_test:etl_test@localhost:5434/etl_test
```

Never point `DANTRON_TEST_POSTGRES` at a server holding real data. The tests stay inside schemas of
their own, but a test server is the place to find out they do not.

`ruff format` is not applied across the repo. Do not reformat files you are not otherwise changing.

**Check which copy you are importing.** A consumer may have this checkout installed in editable
mode, and a test run from inside another environment can import a different copy of the package
than the one you are editing. If anything looks impossible, run
`.venv/bin/python -c "import dantron_etl; print(dantron_etl.__file__)"`.

## How the library is laid out

| module | what it owns |
|---|---|
| `__init__.py` | The public API for running a flow: `read_pipeline`, `load_pipeline`, `run_pipeline`, `Pipeline`, `Step`, `PortRef`, `PipelineTrace`, `StepEvent`, `CachedFrame`, the focus names, `REGISTRY`, `get_tool`, `COLUMN_TYPES`, `project_root`, and the errors. |
| `authoring.py` | Everything a tool author needs, gathered from the modules below. Every file in `tools/` imports from here and nowhere else in the core. |
| `errors.py` | The error family. Everything descends from `ToolError`. |
| `config.py` | `ConfigField`, the template a tool's settings are checked against, default filling, and finding the expressions a configuration holds. |
| `ports.py` | `Ports`, the named-frame envelope a step reads and returns. |
| `status.py` | Per-row status: its shape and the helpers that build one. |
| `contract.py` | The `Tool` protocol, the optional `DerivesInputPorts` member, `ToolHelp`, `WorkedExample`, the categories, and asking which ports a step reads. |
| `boundary.py` | `run_tool`, the boundary every call to a tool goes through. |
| `frame_checks.py` | The helpers a tool calls to check its configuration against the frames it was given (`require_columns`, `compile_for`, `check_output_column` ...). |
| `registry.py` | `ToolRegistry`, the shipped `REGISTRY`, and `get_tool`. |
| `example_world.py` | The scratch directory, temporary environment and in-memory database a worked example runs in. |
| `conformance.py` | `check_tool` and `check_registry`: every catalog rule, and every worked example run and compared. |
| `expr/` | The bracket expression language. Parsed with `ast`, checked against a whitelist, compiled to a polars expression. Never `eval`. One module per job; the package docstring lists them. Every function is defined once, in `expr/functions.py`. |
| `spec.py` | What a loaded flow is: `Pipeline`, `Step`, `PortRef`, `Lane`, `Group`, the walk order, the fingerprint. |
| `rules.py` | Resolving `${rules.*}` references in a step's configuration. |
| `loader.py` | `read_pipeline` and `load_pipeline`. Every refusal that can happen before a frame exists: duplicate ids, unknown tools, bad edges, cycles, bad config, unresolved `${rules.*}`, dead-end steps, bad containers. |
| `walk.py` | Works out a run before it starts: which steps `until=` and `cached=` leave to walk, the frames handed in, and when each frame can be freed. |
| `trace.py` | What a run leaves behind: `StepEvent`, `PipelineTrace`, `CachedFrame`, and the failures that carry a partial trace. |
| `runner.py` | `run_pipeline`. One frame per port, one event per step, a trace you can read afterwards. `until=` runs part of a flow, `cached=` resumes from earlier frames, `retain=` chooses memory over inspectability. |
| `tool.py`, `pipeline.py` | Old import paths, re-exporting the names above. Removed once consumers move to the new ones. |
| `focus.py` | Narrows a finished run to one entity so a wrong figure can be followed step by step. Reads the run; never re-runs it. |
| `preview.py` | Serves a run's ports a window at a time, so a page can search all the rows, not only the drawn ones. |
| `db.py` | An environment variable name in, an open connection out. Drivers picked by URI scheme and imported lazily. The table helpers the table tools share. |
| `paths.py` | `project_root(__file__)`. Where the caller's project is, or a refusal. |
| `column_types.py` | The type names a flow may write (`String`, `Float64`, `Date` ...). |
| `tools/*.py` | One tool per file. Each registers itself at the bottom: `REGISTRY.register(Filter())`. `tools/__init__.py` imports every module it finds, so `import dantron_etl` fills the catalog. A module whose name starts with `_` is a helper, not a tool, and is skipped: `_spreadsheet.py` holds what a cell and a tab can hold, and `_workbook_*.py` hold the parts of `write_workbook`. |
| `scripts/` | The page builders and the run server. **Not shipped in the wheel.** They render; they never transform data. |
| `pipelines/fixture.yaml` | A small neutral flow used to prove load, run, focus and render end to end. |

The counts in `MANUAL.md` are test-checked. The ones in docstrings and in `README.md` are not.

## The rules

These are not style preferences. Each one was learned by getting it wrong.

### The library stays client-neutral

- No client or product name, and no client vocabulary, in anything under `src/dantron_etl/` —
  code, docstrings, comments, examples, the manual. `tests/package/test_boundary.py` holds the
  banned words (`FORBIDDEN_TERMS`) and fails the build on any of them. In `.md` files the package's
  own name is allowed; in `.py` files it is not, which is why imports are relative.
- No import of a client package, an orchestrator, or pandas (`FORBIDDEN_IMPORTS`). The boundary
  test imports every module in a clean interpreter to prove it.
- Keep this file, commit messages and test names neutral too, even though no test checks them.
  Write "a revenue flow" or "a client's ledger", not the client's name. The repo is meant to be
  handed to the next client as is.

### A new tool is a last resort

The rule that matters most. Manual §2 and §8 have the full argument. In short:

- During the first real pipeline, three capabilities were declared missing. Each one was **two
  existing tools in sequence.** One of them had already been built, with 60 tests, before that was
  noticed. It was thrown away.
- The mistake is always the same: ask "does one tool do this?", correctly answer no, and stop.
  Try two tools. Then three. Ask whether the *frame* is the problem: a "fill the blanks" request was
  really "the rows do not exist yet".
- **A tool reshapes the frame. An expression function computes one column.** If it is one column
  out, it is a function in `expr/functions.py`, not a tool.
- **Extend the language by whitelisting a function, never by adding syntax.** No user-defined
  functions, no attribute access, no subscripts, no lambdas. That refusal is a security boundary:
  a pipeline file is edited by people who are not developers.
- No custom tools, ever. Every tool must be generic and reusable on another client's data. A tool
  that exists for one report is a bug.
- A tool added is maintained forever and ships to every consumer. When unsure, raise it as a
  question with a worked example, not as a plan.

### No script reshapes data

Scripts may run a pipeline, compare its output, and render a page. They may never transform the
data. A figure produced outside a step has no configuration anyone can read, no place on the drawn
page, and none of the boundary's checks. If a script needs to reshape something, the library is
missing a capability, and that is a tool or a function to add.

### Refuse early, by name, on the schema

- If a configuration cannot work on a frame's *shape*, refuse it whether the frame has rows or not.
  A million rows and zero rows must be refused the same way.
- Every refusal names the field, the column, what it held, and what was asked of it.
- Never fail quietly. A value that matches nothing while reporting success is the worst outcome
  this library can produce. Most of the code exists to turn those into named refusals.
- Refusals come before side effects. Nothing may be refused after a step has already written
  somewhere.

### Each tool follows the SQL operation it is named after

That is how null handling is decided, and why there is no shared "null mode" setting. `join`
follows `=`, so a null key pairs with nothing. `group_by` follows `GROUP BY` and `remove_duplicates`
follows `DISTINCT`, so two nulls are one group. `sum` over all-null values is null, not zero. A
new tool states which operation it mirrors, in its `help().notes`.

### A guarantee in prose is not a guarantee

This project's most repeated lesson. Every rule that lived only in a docstring was found broken
somewhere once it was finally measured. If a docstring or a note claims something, a test must
fail when it stops being true.

- Every `WorkedExample` is **executed** by the conformance check and compared with its stated
  output. That is what makes the generated catalog trustworthy. There is no exemption; a tool that
  needs a database writes its example against in-memory SQLite.
- The catalog is generated from the registry. Never write tool documentation by hand anywhere else.

### Other standing rules

- **Do not mutate an input frame.** The boundary checks each input's schema before and after a step.
- **Vectorised only.** No row-by-row Python inside a tool. (`format_number` and `uuid5` are the
  known per-row exceptions, documented as such.)
- **Import polars and the standard library at module level, nothing else.** YAML, openpyxl and
  database drivers are imported inside the function that needs them. Every module-level import is
  one that every consumer is forced to carry. That is also why `preview.py` uses `json`, not
  `orjson`.
- **Never find the project root from the library's own `__file__`.** Installed, that path is
  inside site-packages, and it fails silently by returning a real but wrong directory. Callers pass
  their own `__file__` to `project_root()`.
- **Config names an environment variable, never a credential.** A pipeline file must be safe to
  commit and to hand to another client as a template.
- **Enum values must survive YAML.** Never `yes`, `no`, `on`, `off`, `true`, `null` or anything
  else YAML rewrites. `check_tool` refuses them.

## How to work in this repo

### Test the mirror case, then run the mutation

Across several build cycles, every serious gap a review found had one shape: **a rule tested on one
side and not its mirror.** The type rule tested when written first, not when written second. The
single-output step cached, not the multi-output one. The non-empty frame, not the empty one. The
symmetric case is the one the author did not picture, and it is where wrong numbers live.

So before calling a change done: for each rule, name the mirror case and check it is tested. Then
break the code on purpose and confirm a test goes red. Reasoning about coverage is not the same as
running the mutation.

For the page builders, assert a **property** of the output, not that a string appears in it. Eleven
mutations once survived because `assert "while (" in body` passes for a loop that does nothing.

### Review rounds

The library was built test-first, a cycle at a time. When an adversarial review round is used, run **one** round,
and act only on findings where a reader would see a wrong figure or wrong information. Coverage gaps
in code that is already correct go into the next round's brief, not into a loop. Zero rounds is not
the setting either: one round once found that the entire feature being built could be replaced by a
constant with every test still green.

### Write plainly

Docstrings, `description`, `notes`, `why` and commit messages are read by finance people and by
developers at other clients. The existing prose is accurate but too dense, and it reads as
machine-written. A rewrite pass is owed across `src/dantron_etl/`. **Do not start it unprompted.**
Write anything new in the plainer style: short sentences, one idea each, what it does before why.
Manual §4b has the full rules and a worked before-and-after.

### The owner

- Not a career developer. The reference point is Alteryx, where every transform is a box you open,
  inspect and edit. Explain in plain terms, and frame decisions by their effect on figures.
- **Confirm before building.** Discuss a plan and get agreement before writing code.
- A diagram or a read-only view is not a deliverable. Tooling must let the owner *act*: either edit
  in the tool and have it change the real file, or point at the exact file and line to change.
- Commit only when asked. Push only when asked.

## Changing the library

### Adding an expression function

Each function is defined once, in the function table (`DEFINITIONS` in `expr/functions.py`). To add one:

1. Write the function that builds its column from the arguments.
2. Add one entry to the table: the name, the arguments in order and what each must hold, the
   family and type of the result, the builder, a one-line summary and an example.
   `ALLOWED_FUNCTIONS`, the arity check and the result types are all read from the entry, and a
   test refuses an entry with a part missing.
3. Write its tests. The tests from the commit that added `uuid5` (`f22c8a3`) are a good template.
4. Add it to `MANUAL.md` §4.2. A test checks the manual lists every function.

### Adding a tool

Only after the questions in manual §2 have been answered in writing. Then the contract in manual
§8.2. `tools/filter.py` is the smallest complete example to copy the shape from. A tool imports
everything it needs from `authoring.py`. Checklist:

1. One file in `tools/`, registering itself at the bottom.
2. Every config field typed and described, required or defaulted.
3. `help()` with at least one worked example, the rulings in `notes`, and per-field help.
4. `validate_config` for cross-field rules only. `validate_inputs` for anything answerable from the
   schema.
5. Tests for the tool, including the mirror cases above.
6. Add it to the table in `MANUAL.md` §3.1 **and update the count** there and in `README.md`.
7. Regenerate the catalog and look at the tool's page.

A tool whose input ports depend on its configuration (like `write_workbook`, one port per sheet)
implements `input_ports_for` and leaves `input_ports` empty. Never both. See `DerivesInputPorts`.

### Changing an existing tool

Consumers' pipeline files depend on the exact field names, port names and behaviours. Renaming a
field or a port, or changing what a tool does with nulls or empty frames, breaks someone's flow at
load or, worse, changes their figures quietly. Treat those as breaking changes and say so in the
commit message.

## Consumers

The first real consumer is a client's revenue flow of several hundred steps. Almost every rule in
this file and in the manual was learned building it. A consumer may install a local checkout of
this repo in **editable mode**, and then a change here is live in that consumer's next run, before
anything is committed. That flow compares every stage against an independent implementation to the
cent. It is the best integration test this library has: after a change to a tool's behaviour, it
is worth running there before calling the change done.

Lessons from that consumer that belong to the library:

- **Publishing many tables together** is `publish_tables`: one transaction, all or nothing. It
  compares the frame's columns with the table's and refuses a mismatch in **both** directions. A
  column added to a table by hand and not to the flow stopped every publish for twelve days, and
  nothing reported it because the refusal was correct. A consumer needs its own freshness alert.
- **A cache fingerprint covers the file, never the data.** A cached port taken before the sources
  were reloaded resumes over yesterday's figures with a clean-looking run. Manual §7b.
- **`write_workbook`'s `replace` governs a tab, not the file.** A tab the step stops writing stays in
  the existing workbook until the file itself is deleted.
- **A dead-end step is refused at load**, so removing an output orphans its whole upstream chain one
  refusal at a time. Dropping six report tabs from that flow took 74 steps with them. Work out the
  whole chain from the wiring first rather than deleting step by step.

## Where this is going

The goal is a composable block library with a conformance test battery, reused across clients and
across kinds of work: raw-to-clean loads, transformations, report building. The same tools, the
same layout per tool, the same tests, so the second pipeline costs almost nothing to read.

The viewer follows one layout: a canvas generated from the pipeline file and never authored in, a
configuration panel per step, and a searchable data grid. The dependency runs one way: the viewer
reads pipelines; a pipeline never imports the viewer. A framework proven against one consumer is not
yet proven; a second, unrelated pipeline is the real test of every abstraction here.

## Git

- `develop` is the working branch. Work happens on `feature/<name>` and merges into `develop` with
  `--no-ff`, titled `Merge feature/<name> into develop`. `main` trails `develop`.
- `develop` is also GitHub's default branch, so a pull request targets it unless told otherwise.
- Commit titles are plain sentences that say what changed for the reader ("Publish a set of related
  tables, all of them or none of them"). The body explains why, in plain prose.
