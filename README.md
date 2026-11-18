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
from dantron_etl.pipeline import read_pipeline_file
from dantron_etl.runner import run_pipeline

spec = read_pipeline_file("pipelines/mine.yaml")
trace = run_pipeline(spec)
trace.frame(PortRef("receivable_lines", "true"))
```

## Why it exists

A transformation written as a script is answerable only by reading the script. When somebody asks
"why is this number what it is", the honest answer has to name a step, show the rows going in and
the rows coming out, and let them look. That is what this is for: every step is a configured
instance of a general tool, every step's output is inspectable, and the definition of a run is one
file a non-developer can read.

## The 31 tools

| category | tools |
|---|---|
| **input** | `read_csv`, `read_excel`, `read_table` |
| **preparation** | `cast_columns`, `filter`, `limit_rows`, `match_one_to_one`, `record_id`, `remove_duplicates`, `rename_columns`, `select_columns`, `sort` |
| **transform** | `assemble_date`, `case_when`, `connected_components`, `cross_tab`, `date_diff`, `expand_date_range`, `formula`, `group_by`, `unpivot` |
| **combine** | `join`, `union` |
| **text** | `normalize_text`, `regex_extract_groups` |
| **assert** | `assert_rows` |
| **output** | `publish_tables`, `write_csv`, `write_excel`, `write_table`, `write_workbook` |

Every tool carries executable worked examples, and a conformance test runs each example on its
stated input and compares it to its stated output. The catalog is generated from the registry, so a
tool's page cannot drift from the tool.

## Ideas worth knowing before you use it

**Ports, not steps.** A step's inputs and outputs are *named* frames. `filter` has `true` and
`false`; `join` has `matched`, `left_only` and `right_only`. Rows a step sets aside are a first-class
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

Postgres drivers are an extra, deliberately: `read_table` and `write_table` import theirs lazily by
URI scheme and name the package to install when it is missing, and SQLite needs nothing at all —
which is what lets every worked example in the catalog execute with no server.

## Paths

Nothing here infers your project's root from its own location. An installed package lives in
site-packages, so walking up from its own file finds the virtualenv, not your project — silently,
which is worse than failing. Callers say where they are:

```python
from dantron_etl.paths import project_root
ROOT = project_root(__file__)          # $ETL_PROJECT_ROOT, else the nearest pyproject.toml or .git
```

## Looking at a run

**These live in `scripts/`, which is not part of the installed package** — a `pip install` gives you
the tools, the runner and the manual, not the page builders. Clone the repository to use them:

```bash
make run p=fixture     # run pipelines/fixture.yaml, save the run, and open its page
make view              # the viewer for every saved run, on http://localhost:8899/
make catalog           # every tool, its settings and its worked examples
make                   # list every command
```

The walkthrough draws the flow as wired, follows one entity through it, and shows each step's rows
beside its configuration. Selecting a step lights its whole upstream and downstream chain.

### A project

`make run p=<name>` runs a project. A project is three things with one name:

| | |
|---|---|
| `pipelines/<name>.yaml` | the flow |
| `pipelines/<name>/` | its input files |
| `pipelines/<name>/settings.yaml` | the values the flow reads as `${rules.<name>}`, and the column that identifies one entity, so the page can follow it |

The sample project, `fixture`, is the one to copy. Its settings file:

```yaml
rules:
  revenue_type: REVENUE
focus: party
```

The run is saved in `output_files/runs/<name>/`, replacing the last one. The page reads its rows
from the viewer, so `make run` starts the viewer if it is not already running and opens the page in
your browser. Ctrl-C stops the viewer. If a step fails, the run is still saved up to that step, so
you can open the failing step and look at the rows that went into it.

On Windows, without `make`, run the same thing as
`uv run python scripts/pipeline_walkthrough_page.py <name>`.

## For agents

`src/dantron_etl/MANUAL.md` is written for LLM agents building pipelines with these tools. It leads
with the rule that matters most — **a new tool is a last resort** — and documents the expression
language, the YAML traps that bite, and how to know your pipeline is right. A test asserts the
manual describes the library that actually exists: every registered tool named, no invented ones,
the count correct.

## Development

```bash
uv sync --extra dev --extra postgres
.venv/bin/pytest tests/ -q           # about 3,800 tests
.venv/bin/ruff check src tests scripts
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
