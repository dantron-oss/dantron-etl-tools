# Concepts

The ideas behind the library, each in a few paragraphs. Read [Your first pipeline](quickstart.md)
first if you have not: this page is easier with a running pipeline in front of you.

## A pipeline is a file

A pipeline is a YAML file. It lists **steps**. Each step names a **tool** and gives it settings.
Nobody writes transformation code: the tools do the work, and the file says how they are set up.

That makes the file the whole answer to "what does this pipeline do?". It can be read by someone
who does not program, reviewed in a pull request, and handed to another project as a template.
Scripts may run a pipeline, compare its results and draw them, but they never reshape the data
themselves. A figure made outside a step has no settings anyone can read and no place on the page.

## Tools

A **tool** is one kind of data operation: `filter`, `join`, `group_by`, `write_csv` and so on. The
library ships a fixed set of them, listed in [tools](tools/README.md) and by `dantron-etl tools`.
Each tool states its settings, the rules it follows and at least one worked example, and the test
suite runs every worked example.

Each tool follows the SQL operation it is named after. That decides the questions people otherwise
argue about. `join` follows `=`, so a blank key matches nothing. `group_by` follows `GROUP BY`, so
blank keys form one group. `sum` over nothing but blanks is blank, not zero.

A tool changes the shape of the table: which rows there are and which columns. A calculation that
gives one new column from the others is a **function** inside an expression, used by a tool such as
`formula`. See [Expressions](expressions.md).

## Steps

A step is one use of a tool:

```yaml fragment
- id: paid
  tool: filter
  why: Only paid orders count as sales.
  config:
    predicate: "[status] = 'PAID'"
  inputs: {rows: orders.rows}
```

- `id` is the step's name. Other steps use it to read this step's output.
- `tool` is the tool's name.
- `config` holds the tool's settings. A setting the tool does not have is refused.
- `inputs` says which earlier output each of the tool's inputs reads.
- `why` is a plain sentence for the reader. The viewer shows it beside the step.

Every key a step may hold is listed in [The pipeline file](pipeline-file.md).

## Ports

A step's inputs and outputs are named. Each name is a **port**. `filter` reads `rows` and gives
`true` and `false`. `join` reads `left` and `right` and gives `joined`, `left_only` and
`right_only`. You wire them with `step.port`: `inputs: {left: paid.true, right: customers.rows}`.

The point of the extra ports is that rows a step sets aside are still there to look at, and to use.
The orders that matched no customer are on `left_only`. You can see them in the viewer, write them
to a file, or check that there are none. Nothing disappears without a place you can find it.

Every output port always exists, even when no row reaches it; it is then empty with the right
columns. You only wire the ports you need.

## Every output is used

Each step's rows must go somewhere: into another step, into a file or table, or into a check
(`assert_rows`). A step whose output nothing reads is a **dead end**, and the file is refused when
it loads. This catches a step left behind by an edit, which would otherwise run and be ignored.

## Lanes, stages and containers

These change how the pipeline is drawn. They never change what it computes.

- A **lane** is a horizontal band on the drawing, such as "Sources", "Figures" and "Outputs". Lanes
  are declared at the top of the file with an `id`, a `title` and an `order`, and each step says
  which lane it is in.
- A **stage** is a short word on a step, shown in the step's detail. It is free text, so a
  misspelt stage is not caught.
- A **container** is a titled box round a few steps that do one job, such as "Read the three source
  files". A reader can close it to one bar. Containers are declared at the top of the file with an
  `id` and a `title`, and each step says which container it is in. All the steps in one container
  must be in the same lane.

## Rules

A **rule** is a value a pipeline reads by name, written `${rules.name}`. It keeps the values people
change (a status, a start date, a threshold) out of the steps:

```yaml fragment
config:
  predicate: "[status] = '${rules.status}'"
```

The values live in the project's `etl-project.yaml`, under the pipeline's name:

```yaml fragment
pipelines:
  sales:
    rules:
      status: PAID
```

They can also come from a separate YAML file (`rules_file:`) or from a function in your project's
Python code (`rules_loader:`). A pipeline that uses a rule nobody gave a value is refused when it
loads, by name. Inside an expression the value is quoted for you; anywhere else it is put in as it
is. [The pipeline file](pipeline-file.md#rules) has the details.

## The project

A **project** is a folder with an `etl-project.yaml` in it (a `pyproject.toml` or a `.git` folder
also marks one). The command finds it by looking upward from the folder you are in. Without
settings, pipelines are the `*.yaml` files in `pipelines/`, and runs are saved in
`output_files/runs/`. The project file can change both, and gives each pipeline its rules and the
column the viewer follows:

```yaml fragment
pipeline_folders: [pipelines]
runs_folder: output_files/runs
pipelines:
  sales:
    rules: {status: PAID}
    focus: customer_id
```

**Focus** names the column that identifies the thing you want to follow, such as a customer. In
the viewer you can then pick one customer and see only their rows at every step.

## Chains

A **chain** runs several pipelines in order: a pipeline that reads another's output runs after it,
and the ones that do not need each other run side by side. It is a short file in `chains/` that
names the pipelines and what each runs after, with values the whole chain shares, such as the
period. The order is checked against the files and tables the pipelines pass on before anything
runs. See [Chains](chains.md).

## Preview and real run

A **preview** runs every step except the ones that write outside the run: file and table writers,
and anything after them. Nothing is published. This is what `dantron-etl run <pipeline>` and the
viewer's **Run preview** button do.

A **real run** runs every step, writers included. `dantron-etl run <pipeline> --write` first lists
every file and table it will write, and asks you to type the pipeline's name. The viewer's
**Real run…** asks the same. `--yes` skips the question, for scheduled jobs.

For a table, the list says which database the step's connection points at as you look: the host,
the port when the address states one, and the database name, so you can tell the test database from
the live one. It never shows the password or the user. If the environment variable is not set, or
its address cannot be read without risking the password, the list says so instead. If the
connection points somewhere else by the time you confirm, the run is refused and you look again.

## A run is evidence

Every run leaves behind the rows on every port, one event per step (rows in and out, time taken),
and a page that draws it all. The viewer shows the last run of each pipeline; opening the viewer
never runs anything.

A run is saved in `output_files/runs/<pipeline>/`. A run that fails is saved beside the last good
one, as `<pipeline>.failed/`, and the viewer shows the failing step and its error. The last good run
is kept.

A saved run holds real rows, which can be personal or financial data. On Linux and macOS, each
folder a run makes is readable by its owner only (`0700`), and so is each file in it (`0600`),
whatever the umask. A folder that was already there, such as the project folder or a runs folder you
made, keeps its permissions; so do older runs, and the files your pipeline's output steps write. On
Windows, and on a drive that refuses the change, nothing is changed: the run is still saved, and its
files are only as private as the folder they are in, so keep the runs folder somewhere only you can
read.

### A project that must not keep copies of its data

Set `save_rows: false` in `etl-project.yaml`, at the top for every pipeline or under one pipeline
to override the top:

```yaml fragment
save_rows: false
pipelines:
  sales:
    save_rows: true
```

A run still needs its rows in memory while it runs; only what is kept afterwards changes. The saved
run keeps the steps, whether each ran, the time each took, the row count and the column names and
types at every port, and what a real run wrote. It keeps no row from any port, and no list of the
values the viewer could follow. The viewer still draws the flow, the settings, the counts and the
columns; where it would show rows, or let you follow one customer, it says the rows are not saved
and names the setting.

A failed step's message is kept, because it is what says why the run stopped, and its sentence can
name a value, such as the cell that stopped a read or a key that matched nothing. The example rows a
refusal lists after it are left out. The viewer never reuses a saved run's rows for a later run, so
nothing else changes.

From Python, `run_pipeline` returns the same evidence as a **trace**:

```python
from dantron_etl import read_pipeline, run_pipeline

spec = read_pipeline("pipelines/sales.yaml")
trace = run_pipeline(spec)
print(trace.frame("with_customer.left_only"))
```

`run_pipeline` is a real run: it runs the writers too.

## Refusals come early

The library would rather stop than give a plausible wrong answer. Most mistakes are refused when
the file loads, before any data is read: an unknown tool, setting or key, a port that does not
exist, a rule with no value, a dead end, a writer whose folder is missing. Mistakes that depend on
the data, such as a column that is not there or a division by zero, stop the step that meets them,
and the message names the step, the setting and the column.

## Next

- [The pipeline file](pipeline-file.md): every key, and the YAML traps.
- [Expressions](expressions.md): the formula language.
- [Cookbook](cookbook.md): common shapes, ready to copy.
- [Checking your pipeline](checking-your-pipeline.md): how to know it is right.
