# Checking your pipeline

A pipeline that runs and gives plausible figures has not yet shown that the figures are right.
This page is about showing it.

## Compare with an independent answer

Find something that already knows the answer: the system the pipeline replaces, a spreadsheet
someone signed off, a total worked out by hand for one month. Then compare.

- **Compare the whole table, not just the total.** A total can agree while the rows behind it are
  wrong: two errors that cancel, or an amount on the wrong customer. Compare row by row, on the key.
- **Compare the columns' types first, then the values.** Two tables can hold equal values with
  different types. A comparison that looks only at values calls them the same.
- **Money to the cent, everything else exactly.** Decimals worked out in a different order can
  differ in the last digit. A comparison that fails on that is not telling you anything.
- **Write the comparison as a script that runs the pipeline and compares.** The script may run,
  compare and report. It must never change the data: a fix belongs in a step, where it can be seen.

From Python:

```python
import polars as pl
from polars.testing import assert_frame_equal

from dantron_etl import read_pipeline, run_pipeline

spec = read_pipeline("pipelines/sales.yaml")
trace = run_pipeline(spec)
ours = trace.frame("by_region.rows").sort("region")
theirs = pl.read_csv("checks/sales_by_region_signed_off.csv").sort("region")
assert_frame_equal(ours, theirs, check_dtypes=True, abs_tol=0.005)
```

`run_pipeline` runs every step, the writers included: it is a real run. To stop before the writers,
give `until` the last step you need, as below.

## Follow one customer through every step

When a figure is wrong, find the step where it went wrong. In the viewer, set `focus` for the
pipeline in `etl-project.yaml` to the column that names a customer (or an account, or an order).
Then pick one value at the top of the page: every step shows only that customer's rows, and you can
see where they change.

## Put checks in the pipeline

`assert_rows` stops the run when a row breaks a rule, at the step where it happens, and names the
rows. That is better than finding a wrong total three steps later. Good things to assert:

- **A key is unique** before you join to it. A repeated key repeats every amount joined to it.
  Remove duplicates by the key and assert that the `duplicate` port is empty.
- **Nothing went unmatched** when everything should match: assert that a join's `left_only` port
  is empty.
- **A required value is there:** `[region] is not null`.
- **Totals agree** between two routes to the same figure: join the two totals and assert
  `abs([a] - [b]) < 0.005`.
- **A value is in range:** `[rate] > 0 and [rate] < 10`.

A port that must be empty gets an assertion no row can pass, and permission to be empty:

```yaml fragment
- id: no_unknown_customers
  tool: assert_rows
  config:
    label: every order names a known customer
    predicate: "false"
    allow_empty: true
  inputs: {rows: with_customer.left_only}
```

An assertion on a port with no rows fails unless the step says `allow_empty: true`. "No rows" is
more often a mistake upstream than a quiet day, so you have to say when it is fine.

The [cookbook](cookbook.md#a-check-that-stops-the-run) has a whole example.

## Run part of a pipeline

A large pipeline is slow to run whole while you work on one corner of it. From Python,
`run_pipeline` can stop early and start again from where it stopped.

`until` runs one step and every step it depends on, and nothing else:

```python
from dantron_etl import read_pipeline, run_pipeline

spec = read_pipeline("pipelines/sales.yaml")
trace = run_pipeline(spec, until="with_customer")
```

`cached` hands the rows of a port to a later run, so the steps before it do not run again:

```python
from dantron_etl import CachedFrame, read_pipeline, run_pipeline

spec = read_pipeline("pipelines/sales.yaml")
first = run_pipeline(spec, until="by_region")
port = "by_region.rows"
later = run_pipeline(spec, cached={port: CachedFrame(first.frame(port), spec.fingerprint(port))})
```

A step with several output ports, such as `join`, is cached on all of them or none: give one
`CachedFrame` per port.

The fingerprint is taken over the steps that made the port: their tools, settings and wiring. If you
change any of them, the cache is refused by name instead of used. Changing a `why`, a lane, a stage
or a container does not change the fingerprint.

### The one trap: a cache does not know your data changed

The fingerprint covers the pipeline file, never the data. If the source files or tables are
reloaded, a cached port still holds the old rows, the steps that would read the new ones do not
run, and the run looks clean. Nothing can detect this, because the reads did not happen.

So: **throw a cache away whenever the sources change, and take a fresh one.** Also take a fresh one
after upgrading the library, because a tool may now compute something differently. And run the
whole pipeline once at the end: a cached run shows what your edit did, not that the whole pipeline
still agrees with its independent answer.

## Keep the memory down on a large run

By default a run keeps every port's rows, so you can look at all of them. A scheduled run of a large
pipeline does not need that. `retain="live"` frees each port's rows as soon as nothing left in the
run reads them, and `keep` names the ports to hold on to anyway:

```python
from dantron_etl import read_pipeline, run_pipeline

spec = read_pipeline("pipelines/sales.yaml")
trace = run_pipeline(spec, retain="live", keep=["by_region.rows"])
```
