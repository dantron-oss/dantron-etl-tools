# dantron-etl-tools

Build data pipelines by writing a YAML file instead of code. Each step names one of a fixed set of
tools (read a file, filter, join, total, write a workbook) and gives it settings. Every run keeps the
rows at every step, and a viewer in your browser lets anyone see what each step did to each row.

It is for people who need to answer "why is this number what it is?": the answer names a step,
shows the rows that went in and came out, and can be checked by someone who does not program.

## Quick start (5 minutes)

You need Python 3.13 and [uv](https://docs.astral.sh/uv/) (or pip).

**1. Install** into a virtual environment, and activate it:

```bash
uv venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
uv pip install git+https://github.com/DanTron-Inc/dantron-etl-tools
```

With pip instead of uv: `python -m venv .venv`, activate it the same way, then
`pip install git+https://github.com/DanTron-Inc/dantron-etl-tools`.

Check it worked:

```bash
dantron-etl --version
```

**2. Make a project** with a working example in it, and open the viewer:

```bash
dantron-etl new demo
cd demo
dantron-etl view
```

The terminal prints the viewer's address and your browser opens on it. The page lists the
project's pipelines. There is one, **Paid orders by customer**. It has not run yet, so the page
says so and offers to run it.

Press **Run preview**. A preview runs every step except the one that writes a file, so nothing
outside the project's run folder changes. Then open the pipeline. You see:

- the steps drawn as a flow, from reading `orders.csv` to the totals per customer;
- each step's settings and its `why`, a sentence saying what the step is for;
- the rows at every step. The `paid` step shows the paid orders on its `true` port and the rest on
  `false`.

Pick a customer in the box at the top to see only their rows at every step.

**3. Change a step.** Leave the viewer running and open `pipelines/example.yaml` in an editor.
Find the step `by_customer` and add a line under `aggregations`, so it also shows each customer's
largest order:

```yaml fragment
      aggregations:
        - {name: paid_total, function: sum, value: "[value]"}
        - {name: paid_orders, function: count}
        - {name: largest_order, function: max, value: "[value]"}
```

In a second terminal, in the `demo` folder (activate the environment there too), run a preview:

```bash
dantron-etl run example
```

```text
Preview of example: the steps that write are held back.
  orders (read_csv): 12 rows out, 35 ms
  paid (filter): true: 7 rows, false: 5 rows, 3 ms
  order_value (formula): 7 rows out, 1 ms
  by_customer (group_by): 5 rows out, 4 ms
Held back, nothing written: save
Finished: 4 of 5 steps ran.
```

Reload the page in your browser: `by_customer` now has a `largest_order` column. (The viewer's
**Run preview** button does the same as that command.)

**4. Write the file for real.** `dantron-etl run example --write` lists the file it will write,
asks you to type `example` to go ahead, and writes `output/paid_by_customer.csv`. In the viewer,
**Real run…** does the same.

Press ctrl-c in the first terminal to stop the viewer.

If you would rather click than type: the new project has `Open ETL viewer.cmd` (Windows) and
`Open ETL viewer.command` (macOS) to double-click, and a VS Code task, **Open ETL viewer**.

## Ideas worth knowing

**A pipeline is a file.** It lists steps; each step names a tool and its settings. Nobody writes
code that changes the data, so the file is the whole answer to "what does this do?". It can be
reviewed, compared and handed on.

**Steps have named ports.** `filter` sends the rows that pass to `true` and the rest to `false`.
`join` gives `joined`, `left_only` and `right_only`. Rows a step sets aside are still there to look
at, to write out, or to check. Nothing disappears without a place you can find it.

**One small formula language.** `[amount] > 0 and [status] = 'PAID'`. Columns in brackets, text in
single quotes, a short list of functions. It is checked, never run as code, so a pipeline file is
safe to accept from anyone.

**A run is evidence.** Every run keeps the rows at every port and what each step did. The viewer
shows the last run; opening it never runs anything.

**Preview, then the real run.** A preview runs everything except the steps that write files or
tables. A real run writes, and tells you exactly what before it starts.

**Mistakes are refused early, by name.** A misspelt tool or setting, a missing input, a step whose
output nothing uses, a rule with no value: each is refused when the file loads, before any data is
read. `dantron-etl check <file>` does just that check.

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

Each has a page in [docs/tools](docs/tools/README.md), and `dantron-etl help <tool>` prints the same
in the terminal. Every tool's worked examples are run by the test suite.

## Where to go next

| page | what it covers |
|---|---|
| [Your first pipeline](docs/quickstart.md) | Write a pipeline from nothing: read, filter, join, total, write. |
| [Concepts](docs/concepts.md) | Pipelines, steps, ports, containers, rules, runs, preview and real runs. |
| [The pipeline file](docs/pipeline-file.md) | Every key the file may hold, rules, paths and the YAML traps. |
| [Expressions](docs/expressions.md) | The formula language and every function. |
| [Tools](docs/tools/README.md) | Every tool's ports, settings, rules and worked examples. |
| [Cookbook](docs/cookbook.md) | Common shapes, ready to copy. |
| [Checking your pipeline](docs/checking-your-pipeline.md) | How to know the figures are right. |
| [The command](docs/command.md) | Every `dantron-etl` command. |
| [Upgrading](docs/upgrading.md) | What changed in 0.2 and how to migrate your files. |
| [CHANGELOG](CHANGELOG.md) | What changed in each release. |
| [CONTRIBUTING](CONTRIBUTING.md) | Working on the library itself. |

From Python, everything to load and run a pipeline is in the package:

```python
from dantron_etl import read_pipeline, run_pipeline

spec = read_pipeline("pipelines/example.yaml", rules={"status": "paid"})
trace = run_pipeline(spec, until="by_customer")
print(trace.frame("by_customer.rows"))
```

Database tables (`read_table`, `write_table`, `publish_tables`) work with SQLite out of the box. For
Postgres, install the extra: `uv pip install "dantron-etl-tools[postgres] @ git+https://github.com/DanTron-Inc/dantron-etl-tools"`.

For agents writing pipelines, `src/dantron_etl/MANUAL.md` ships inside the package with the rules
that matter most.

## Licence

A licence is to be added before public release.
