# dantron-etl-tools

Configurable data transformation tools. A pipeline is a **data file**, not code, and a run leaves
evidence of what it did to every row.

```yaml
steps:
  - id: receivable_lines
    tool: filter
    config:
      predicate: "[type] = 'ACCREC' and [amount] > 0"
    inputs: {rows: source_lines.rows}
```

```python
from dantron_etl import PortRef, read_pipeline, run_pipeline

spec = read_pipeline("pipelines/mine.yaml")
trace = run_pipeline(spec)
trace.frame(PortRef("receivable_lines", "true"))
```

## Quick start

Install the library into a virtual environment. This gives you the command `dantron-etl`.

```bash
uv venv
uv pip install git+https://github.com/DanTron-Inc/dantron-etl-tools
```

Make a project with a working example in it, and open the viewer:

```bash
dantron-etl new demo
cd demo
dantron-etl view
```

Your browser opens on a list of the project's pipelines. There is one, `example`. It has not run
yet, so the list says so and offers to run it. Press **Run preview**. Every step runs except the
one that writes a file, so nothing outside the project's run folder changes. Then open the pipeline:
you see the flow as a drawing, each step's settings, and the rows at every port. Type a customer's
name in the search box at the top to follow that customer through every step.

When the preview looks right, **Real run…** lists the file the pipeline will write and asks you to
type the pipeline's name before it writes. The terminal shows the viewer's address. Press ctrl-c
there to stop the viewer.

If you would rather click than type, the new project has a launcher: double-click
`Open ETL viewer.cmd` on Windows or `Open ETL viewer.command` on macOS. In VS Code, run the task
**Open ETL viewer**.

## Why it exists

A transformation written as a script is answerable only by reading the script. When somebody asks
"why is this number what it is", the honest answer has to name a step, show the rows going in and
the rows coming out, and let them look. That is what this is for: every step is a configured
instance of a general tool, every step's output is inspectable, and the definition of a run is one
file a non-developer can read.

## The 30 tools

| category | tools |
|---|---|
| **input** | `read_csv`, `read_excel`, `read_table` |
| **preparation** | `cast_columns`, `filter`, `limit_rows`, `match_one_to_one`, `record_id`, `remove_duplicates`, `rename_columns`, `select_columns`, `sort` |
| **transform** | `assemble_date`, `case_when`, `cross_tab`, `date_diff`, `expand_date_range`, `formula`, `group_by`, `make_groups`, `unpivot` |
| **combine** | `join`, `union` |
| **text** | `normalize_text`, `regex_extract_groups` |
| **assert** | `assert_rows` |
| **output** | `publish_tables`, `write_csv`, `write_excel`, `write_table` |

Every tool carries executable worked examples, and a conformance test runs each example on its
stated input and compares it to its stated output. The catalog is generated from the registry, so a
tool's page cannot drift from the tool.

## Ideas worth knowing before you use it

**Ports, not steps.** A step's inputs and outputs are *named* frames. `filter` has `true` and
`false`; `join` has `joined`, `left_only` and `right_only`. Rows a step sets aside are a first-class
output you can route somewhere, not a number reconstructed later.

**One small expression language.** `[column]` references, a fixed set of operators and a whitelist of
functions, compiled through `ast` to a polars expression — never `eval`.

**A tool mirrors the SQL operation it is named after.** That is why `join` says two nulls are not
equal and `group_by` says they are: each follows `=` and `GROUP BY` respectively. It is one rule with
three instances rather than three exceptions.

**A run is evidence.** `run_pipeline` returns a trace: one event per step, the shape of every port,
and the frames themselves. `until=` runs part of a flow and stops; `cached=` resumes from a previous
run's frames and refuses a cache whose configuration has since changed.

## Install

```bash
uv pip install git+https://github.com/DanTron-Inc/dantron-etl-tools
uv pip install "dantron-etl-tools[postgres] @ git+https://github.com/DanTron-Inc/dantron-etl-tools"
```

A pipeline file written for an older version is refused at load with the new name of each renamed
word; `python -m dantron_etl.migrate --write <file-or-folder>` rewrites it, keeping its comments.

Postgres drivers are an extra, deliberately: `read_table` and `write_table` import theirs lazily by
URI scheme and name the package to install when it is missing, and SQLite needs nothing at all —
which is what lets every worked example in the catalog execute with no server.

## Paths

Nothing here infers your project's root from its own location. An installed package lives in
site-packages, so walking up from its own file finds the virtualenv, not your project — silently,
which is worse than failing. Callers say where they are:

```python
from dantron_etl.paths import project_root
ROOT = project_root(__file__)  # $ETL_PROJECT_ROOT, else the nearest pyproject.toml, .git or etl-project.yaml
```

## The command

`dantron-etl` works from anywhere inside a project. A project is a folder with an `etl-project.yaml`,
a `pyproject.toml` or a `.git` in it. `dantron-etl <command> --help` explains each command.

| command | what it does |
|---|---|
| `dantron-etl view` | Starts the viewer on this computer only, opens your browser, and prints the address. Runs until ctrl-c. A second `view` for the same project opens the one already running. |
| `dantron-etl run <pipeline>` | Runs a preview: every step except the writers and the steps after them. Prints each step as it runs and saves the run for the viewer. |
| `dantron-etl run <pipeline> --write` | A real run. Lists every file and table it will write and asks you to type the pipeline's name first. `--yes` skips the question, for scheduled jobs. |
| `dantron-etl check <file>` | Loads a pipeline without running it and prints every problem it finds, or `ok, N steps`. |
| `dantron-etl tools` | Every tool, grouped by what it does. |
| `dantron-etl help <tool>` | One tool in full: what it does, its ports, a table of its settings, its notes, and a worked example you can paste. |
| `dantron-etl functions` | The functions an expression may use, with an example each. |
| `dantron-etl catalog` | Writes the tool catalog as a page (`tool_catalog.html`) and as JSON. The viewer shows the same page from its **Tool catalog** link. |
| `dantron-etl migrate <files>` | Rewrites pipeline files written before a rename. Add `--write` to change them. |
| `dantron-etl new <folder>` | Makes a new project with a working example. |

`python -m dantron_etl` is the same command.

### The project settings file

`etl-project.yaml` is optional. Without it, pipelines are the `*.yaml` files in `pipelines/` and runs
are saved in `output_files/runs/`. With it, you can change both and say, per pipeline, where its
`${rules.*}` values come from and which column the viewer follows:

```yaml
pipeline_folders: [pipelines]
runs_folder: output_files/runs
pipelines:
  sales:
    rules: {status: PAID}                 # the values, written here
    focus: customer
  month_end:
    rules_file: config/month_end.yaml     # or read from a file: name: value, or name: {value:, why:}
    focus: [account, customer]            # the first of these each step has
  ledger:
    rules_loader: my_project.rules:load   # or returned by a function in your project
```

A `rules_loader` runs your project's own Python code. It runs only when the pipeline is checked or
run, never when the viewer lists the project. A pipeline that uses `${rules.*}` with no rules set is
shown with the lines to add. The older `pipelines/<name>/settings.yaml` (with `rules` and `focus`)
still works.

### What a run leaves

Each run is saved in `output_files/runs/<pipeline>/`: the rows of every port, one panel per step,
and a page you can open on its own. A run is moved into place only when it finished. If a run
fails, it is saved beside the last good one as `<pipeline>.failed/`, and the viewer shows the step
that failed and its error at the top. The last good run is still there.

A project's own script can run a pipeline itself, for example to compare it with another system
first, and still have the viewer show the run. It saves the run with `save_run`:

```python fragment
from dantron_etl import StepFailedError, run_pipeline
from dantron_etl.viewer import save_run
from dantron_etl.viewer.project import find_project, load_entry

project = find_project(__file__)
spec = load_entry(project, project.pipelines["sales"])
try:
    save_run(project, "sales", spec, run_pipeline(spec, base_dir=project.root))
except StepFailedError as failure:
    save_run(project, "sales", spec, failure.trace, failure=failure)
```

### Safety

The viewer listens on `127.0.0.1`, so only your computer can reach it. It refuses a request whose
address is not this computer, and a run can only be started from a page the viewer itself served.
`--host 0.0.0.0` opens it to your network: anyone who can reach it can read every row. Inside a dev
container, leave the host alone; VS Code forwards the port for you.

## For agents

`src/dantron_etl/MANUAL.md` is written for LLM agents building pipelines with these tools. It leads
with the rule that matters most — **a new tool is a last resort** — and documents the expression
language, the YAML traps that bite, and how to know your pipeline is right. A test asserts the
manual describes the library that actually exists: every registered tool named, no invented ones,
the count correct.

## Development

```bash
uv sync --extra dev --extra postgres
.venv/bin/pytest tests/ -q           # about 4,000 tests, laid out by area: core, expr, tools, viewer, package, flows
.venv/bin/ruff check src tests
```

On Windows the same commands work with `.venv\Scripts\` in place of `.venv/bin/`.

About 160 of those tests need a Postgres server and skip without one, so a checkout with no database
still runs the rest rather than reporting a failure it cannot fix. To run them, start the test
server in `compose.yaml` (it needs Docker) and say where it is:

```bash
docker compose up -d --wait
export DANTRON_TEST_POSTGRES=postgresql://etl_test:etl_test@localhost:5434/etl_test
```

In Windows PowerShell, set it with
`$env:DANTRON_TEST_POSTGRES = "postgresql://etl_test:etl_test@localhost:5434/etl_test"`.
`docker compose down` stops the server, and its data goes with it.

The suite includes a boundary test that imports every shipped file in an interpreter where client
packages do not exist, which is the world this library lives in once installed.
