# Chains: running several pipelines in order

**Status: agreed with the owner 2026-11-27, and built. The user's page is [docs/chains.md](../chains.md).**

A chain runs several pipelines in order. Some run first, the ones that need their outputs run
after them, and the ones that do not need each other run side by side. A chain is a short YAML file
that names the pipelines and what each one runs after. It can be run from the command line, drawn
in the viewer as a graph, and run under Prefect without being rewritten.

Each heading below is one decision: what was decided, why, and what was rejected.

## A worked example

Four pipelines in one project. Each one is an ordinary pipeline file that runs and opens on its own.

| pipeline | reads | writes |
|---|---|---|
| `load_sources` | `inbox/ledger.xlsx`, `inbox/customers.csv` | `stage/ledger.csv`, `stage/customers.csv` |
| `clean_ledger` | `stage/ledger.csv` | `clean/ledger_${rules.period}.csv` |
| `clean_customers` | `stage/customers.csv` | `clean/customers.csv` |
| `revenue` | `clean/ledger_${rules.period}.csv`, `clean/customers.csv` | `reports/revenue_${rules.period}.xlsx` |

The chain, `chains/month_end.yaml`:

```yaml
title: Month-end close

parameters:
  period: "2026-11"            # the default. Change it for one run with --set period=2026-12

pipelines:
  - name: load_sources
  - name: clean_ledger
    after: [load_sources]
  - name: clean_customers
    after: [load_sources]
  - name: revenue
    after: [clean_ledger, clean_customers]
```

How it is drawn. Each line is labelled with what passes along it:

```
                          stage/ledger.csv        clean/ledger_2026-11.csv
                       ┌──────────────────▶ clean_ledger ──────────────────┐
  inbox/ledger.xlsx    │                                                   ▼
  inbox/customers.csv ─▶ load_sources                                   revenue ─▶ reports/revenue_2026-11.xlsx
                       │                                                   ▲
                       └──────────────────▶ clean_customers ───────────────┘
                          stage/customers.csv     clean/customers.csv
```

`clean_ledger` and `clean_customers` run side by side. `revenue` starts when both have finished.

## A chain is not a tool

**Decided.** Running pipelines in order is a layer above pipelines, the way a pipeline is a layer
above tools. It is not a step inside a pipeline.

**Why.** A step hands frames in and out and leaves one event on the trace. A step that ran a whole
pipeline would hide every step inside it behind one box. It would also write files partway through
the parent's run, so the runner could no longer refuse a run before anything is written. And it
would break the cache, whose fingerprint covers one file. Each pipeline in a chain keeps its own
page, its own saved run and its own checks, exactly as when it runs alone.

**Rejected.** A `run_pipeline` tool. A pipeline file listing other pipelines as steps.

## Pipelines pass data through files and tables, not in memory

**Decided.** One pipeline hands data to the next the way it does today: it writes a file or a table,
and the next one reads it.

**Why.** Every hand-off then stays on disk, where it can be opened and checked. Each pipeline can
still be run and opened on its own, against what is already there. And it is how Prefect runs
things: each task may run in a different process or on a different machine, so a frame held in
memory could not be passed between them.

**Rejected.** Passing frames in memory with `sources=`. Besides the reasons above, a frame read back
from a CSV file can have different column types from the frame that was written. A chain that passed
frames in memory could then give different figures from the same pipelines run one at a time.

## `after` is written down, and checked against the hand-offs

**Decided.** The chain file says what each pipeline runs after. Before anything runs, the library
works out every file and table each pipeline reads and writes, and refuses the chain by name if the
two disagree:

- *"revenue reads clean/customers.csv, which clean_customers writes, but revenue does not run after
  clean_customers. Add clean_customers to revenue's after."* Without this, the two run side by side
  and revenue reads last month's file. Nothing would report it.
- Two pipelines in the chain write the same file or table.
- A pipeline runs after one that reads what it writes. The order is backwards.
- Loops, a name that is not a pipeline in the project, a name listed twice, an empty chain.

Running after a pipeline that writes nothing you read is allowed. The graph draws that line dashed:
the order matters for some reason the files do not show. A file or table that no pipeline in the
chain writes comes from outside the chain. The graph lists those, so you can see what the chain
depends on.

**How a read is matched to a write.**

- A file: by its full path, after rules are filled in and relative to the project folder.
- A table: by the connection's environment variable name, the schema and the table.
  `publish_tables` counts as one write per table.

Two different environment variable names that point at the same database are not seen as the same.
The check would miss that hand-off. The chain page says this in one line.

**Why written down, and not only worked out.** A written `after` is something a person can read and
change. Worked out alone, the order would change without anyone touching the chain whenever a path
was edited in one pipeline. Checked against the hand-offs, a written `after` cannot be wrong in the
direction that changes figures.

**Getting started.** `dantron-etl chain suggest <name> a b c ...` writes a first chain file from the
hand-offs it finds between those pipelines. That is how the client's thirty pipelines become a chain:
you review the suggested file instead of typing it.

**Rejected.** Order worked out from the files alone, with nothing written down. Order written down
and never checked.

## Parameters shared by the whole chain

**Decided.** A chain can declare parameters, each with a default value. A run can change any of
them: `--set period=2026-12` on the command line, or a flow parameter in Prefect. Each parameter is
handed to every pipeline in the chain that uses `${rules.<name>}`. For that run, it replaces the
value the pipeline's own settings give.

- The chain's page and its saved run show each parameter and where it went. For example *"period:
  2026-12 (from the chain; clean_ledger's own setting says 2026-11)"*.
- Refused: a parameter that no pipeline in the chain uses (a misspelling would otherwise do nothing),
  a value given for a parameter the chain does not declare, and a value of a different kind from its
  default (a date where the default is a date, text where it is text).
- Parameters are filled in before the hand-off check. So `clean/ledger_${rules.period}.csv` is matched
  under the period the run will actually use.

Parameters are optional. A chain without any runs each pipeline with its own settings.

**Why.** A month-end chain is the same work every month with one value changed. Typing that value
once, for the whole chain, is the only way to be sure every pipeline used the same one.

**Rejected.** Silently letting the chain's value win with nothing shown. Refusing any rule a
pipeline also sets itself: that would rule out the main thing parameters are for.

## The chain's page is a graph and nothing more

**Decided.** The viewer lists chains beside pipelines. A chain's page draws one box per pipeline and
one line per `after`, labelled with the files and tables passed along it. The page shows no rows.
Clicking a box opens that pipeline's own page, where its steps and rows are. Opening the page runs
nothing.

After a chain has run, each box also shows how that pipeline did in that run: finished, failed, or
not started because something before it failed. It also shows when the run finished and links to
the pipeline's saved run.

**Why.** The owner asked for it: "a preview just shows the workflow in a graph ... all the detail can
be opened and previewed individually".

## Running a chain

**Decided.**

- `dantron-etl run month_end` checks the chain, prints the order and the hand-offs, and runs nothing.
- `--write` runs it for real. As with one pipeline, it first lists everything every pipeline will
  write, and you type the chain's name to confirm. If anything changes between being shown and being
  run, the run is refused.
- `--set name=value` changes a parameter for this run.
- `--from clean_ledger` runs that pipeline and everything after it, for a rerun after a fix.
  `--until revenue` runs that pipeline and everything before it. This is the same idea as `until` on
  one pipeline.
- Pipelines that do not depend on each other run side by side, each in its own process, up to
  `max_parallel` at a time. The default is 2. `max_parallel` is set in the chain file.

**What `max_parallel` does.** The chain decides *which* pipelines run and in what order.
`max_parallel` decides only *how many run at the same moment*. Say ten pipelines are all waiting on
`load_sources`. When it finishes, all ten may start. With `max_parallel: 2`, two start, and each time
one finishes the next one starts. All ten still run, and the figures are the same whatever the
setting. Only the time the run takes and the load on the machine change.

**Why a default of 2.** Each running pipeline holds its data in memory, so ten at once can need ten
times the memory of one. A machine that runs out of memory does not refuse the run cleanly: the
process is killed partway, after some pipelines may already have written their files. Running many
at once also gains less than it seems, because polars already uses every core of the machine for a
single pipeline. Raise the setting when your pipelines are known to be small. Lower it to 1 when one
of them is large. Under Prefect the setting is not used: Prefect has its own limits.

**Rejected.** No limit by default. A slow run is a nuisance; a run killed halfway through its writes
leaves files from two different runs side by side.

**When a pipeline fails.** Everything after it is not started. Pipelines that do not depend on it
carry on. The run is reported as failed and lists what was not started. The files and tables those
pipelines write still hold their previous contents, and the chain's page says so for each box.

**Why carry on with the others.** Their figures do not depend on the failure, so they are right.
And it is what Prefect does, so a chain does the same thing wherever it runs.

**What is saved.** Each pipeline's run is saved exactly as today, with the chain's name and the
values it was given. The chain's run is saved as a small file in `<runs folder>/<name>.chain/`: its
values, which pipelines it covered, and each pipeline's outcome and timings. A pipeline name has no
dot, so that folder can never be a pipeline's.
Only the latest run of each pipeline is kept, so a later run of one pipeline on its own replaces the
run the chain linked to. The chain's record keeps its own summary, and its page says when a
pipeline's saved run has been replaced since.

## Prefect

**Decided.** The library never imports Prefect. A test already enforces this
(`tests/package/test_boundary.py`), because the library ships to clients who use other schedulers.
Instead, the library makes two things public:

- `load_chain(root, name)`: the checked chain, with its order, what each pipeline runs after, and its
  parameters.
- `run_member(root, chain, name, values, logger=...)`: run one pipeline of the chain for real with
  the chain's parameter values, save the run, and return its record.

A recipe in the docs ([Running a chain under Prefect](../chains.md#running-a-chain-under-prefect))
turns a chain into a Prefect flow in about thirty lines, and the manual points agents to it. One task
per pipeline, `after` becomes `wait_for`, and the chain's parameters become one flow parameter. The
flow calls `load_chain` first, so the whole check runs before any task starts. It reads the chain file
each time it runs, so editing the YAML changes the flow, with nothing to regenerate. The recipe was
run under Prefect 3.9 before it was written into the docs.

| in a chain | in Prefect |
|---|---|
| the chain | a flow |
| a pipeline in it | a task run |
| `after` | `wait_for` |
| parameters | the flow's `parameters` |
| a failed pipeline | a failed task; the tasks waiting on it do not run |
| the run's messages | Prefect's run log (the runner already accepts its logger) |
| `max_parallel` | the flow's task runner, `ThreadPoolTaskRunner(max_workers=2)` in the recipe |

Scheduling, retries and alerts are Prefect's job, not the library's.

**Why build the flow from the YAML each run.** Python written out once from the chain file would go
stale the moment someone edited the file. Building the flow each run cannot go stale.

**Rejected.** Generating a Python file per chain. A Prefect adapter inside the library. A separate
`dantron-etl-prefect` package for now: worth making only once a second client needs it.

## What a chain does not do

- It does not pass data between pipelines in memory (see above).
- It does not schedule. Running at 6am on the first working day is Prefect's job.
- No chain inside a chain, no conditions ("run B only if A wrote rows"), no loops over values. Each
  waits until a real chain needs it.
- It never reshapes data. It runs pipelines, saves what they did, and draws them.

## How we will know it works

Each rule gets a test, and so does its mirror case:

- Revenue reads what clean_customers writes without being after it: refused. The same with a table,
  with `publish_tables`, and with a path that contains `${rules.period}`.
- The mirror: running after a pipeline that writes nothing you read is allowed and drawn dashed.
- Two writers of one file: refused, whichever of them comes first in the file.
- A parameter that no pipeline uses, and a parameter that two pipelines use: refused, and handed to
  both, respectively.
- A failure in one branch: the pipelines after it are not started, the other branch finishes. The
  same failure in the other branch gives the mirror result.
- `--from` and `--until` on the first, the last and a middle pipeline.
- A real run refused when a pipeline file changes between the plan being shown and the run.

Before calling it done, each rule is broken on purpose to confirm a test goes red. Then the client's
own pipelines are converted to a chain and run against their usual output.

## Settled with the owner

1. **The name.** Agreed 2026-11-27: "chain".
2. **Where chain files live.** Agreed 2026-11-27: a `chains/` folder beside `pipelines/`, set by
   `chain_folders` in `etl-project.yaml`.
3. **`max_parallel`'s default.** Agreed 2026-11-27: 2, explained in the user docs as well as here
   (see "Running a chain").
