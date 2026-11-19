# Your first pipeline

This page walks you through writing a pipeline from nothing. It reads two CSV files, keeps the paid
orders, adds each order's customer, totals the sales by region and writes the totals to a CSV file.
Then you look at it in the viewer and change it.

It takes about fifteen minutes. You need the library installed (see the [README](../README.md)), so
that `dantron-etl --version` prints a version.

## 1. Make a project folder

A project is a folder with a file called `etl-project.yaml` in it. The file can be empty to start
with. Pipelines go in a `pipelines/` folder inside it.

```bash
mkdir shop
cd shop
mkdir pipelines
mkdir pipelines/sales
mkdir output
```

Create an empty `etl-project.yaml` in `shop` with your editor. (`dantron-etl new` makes a project
with an example already in it. Here you build one by hand, so you see every part.)

## 2. Put the data in

Save this as `pipelines/sales/orders.csv`:

```text
order_id,customer_id,status,amount
1001,C1,PAID,120.00
1002,C2,PAID,80.50
1003,C1,OPEN,45.00
1004,C3,PAID,200.00
1005,C9,PAID,15.00
1006,C2,CANCELLED,60.00
1007,C3,PAID,99.50
```

And this as `pipelines/sales/customers.csv`:

```text
customer_id,name,region
C1,Harbour Books,North
C2,Pine & Co,South
C3,Larkspur Cafe,North
C4,Quayside Hardware,South
```

Customer `C9` has an order but is not on the customer list. Customer `C4` is on the list but has
no orders. A good pipeline shows you both.

## 3. Write the pipeline

Save this as `pipelines/sales.yaml`. Each part is explained below it.

```yaml
name: Paid sales by region

steps:
  - id: orders
    tool: read_csv
    why: Every order, as the shop's system exported it.
    config:
      path: pipelines/sales/orders.csv
      column_types:
        - {column: customer_id, type: String}

  - id: customers
    tool: read_csv
    why: Who each customer is and which region they are in.
    config:
      path: pipelines/sales/customers.csv

  - id: paid
    tool: filter
    why: Only paid orders count as sales. Open and cancelled orders leave on the "false" port.
    config:
      predicate: "[status] = 'PAID'"
    inputs: {rows: orders.rows}

  - id: with_customer
    tool: join
    why: >
      Adds each order's customer name and region. An order whose customer is not on the list
      leaves on the "left_only" port instead of disappearing.
    config:
      keys: [customer_id]
    inputs: {left: paid.true, right: customers.rows}

  - id: by_region
    tool: group_by
    why: One row per region, with the paid total and the number of orders behind it.
    config:
      by: [region]
      aggregations:
        - {name: paid_total, function: sum, value: "[amount]"}
        - {name: orders, function: count}
    inputs: {rows: with_customer.joined}

  - id: save
    tool: write_csv
    why: The totals as a file anyone can open.
    config:
      path: output/sales_by_region.csv
      overwrite: true
    inputs: {rows: by_region.rows}
```

What each part means:

- **`name`** is the title the viewer shows. Every pipeline file needs one.
- **`steps`** is the list of things to do. Each step has an `id` you choose, a `tool` from the
  [tool list](tools/README.md), the tool's settings under `config`, and a `why`.
- **`why`** is a sentence for the next person who reads the pipeline. The viewer shows it beside the
  step. Say what the step is for, not what its settings already say.
- **`inputs`** says where a step's rows come from, written `step.port`. A **port** is a named
  output of a step. Most tools have one output port, `rows`. Some have more: `filter` sends the
  rows that pass to `true` and the rest to `false`, and `join` gives `joined`, `left_only` and
  `right_only`.

And each step:

1. **`orders`** reads the orders file. `column_types` says `customer_id` is text. Without it, a
   column of codes that happen to look like numbers would be read as numbers.
2. **`customers`** reads the customer list.
3. **`paid`** keeps the rows where `status` is `PAID`. `[status]` names a column. Text is in single
   quotes. The whole expression is in double quotes so YAML reads it as one piece of text.
4. **`with_customer`** joins the paid orders to the customers on `customer_id`. An order whose
   customer is missing goes to `left_only`. A customer with no paid order goes to `right_only`.
5. **`by_region`** makes one row per region: the sum of `amount` and a count of the orders.
6. **`save`** writes the totals to `output/sales_by_region.csv`. `overwrite: true` lets the next
   run replace the file. Without it, a second run stops, because a file is already there.

To see every setting a tool has, run `dantron-etl help join` (or any tool name), or open its page
under [tools](tools/README.md).

## 4. Check it

```bash
dantron-etl check pipelines/sales.yaml
```

This reads the file the way a run would, but reads no data and writes nothing. It prints
`ok, 6 steps`, or each problem with the step and setting it is in. A misspelt tool or setting, a
missing `inputs`, a step whose output nothing uses, and an output folder that does not exist are
all caught here.

## 5. Run a preview

```bash
dantron-etl run sales
```

```text
Preview of sales: the steps that write are held back.
  orders (read_csv): 7 rows out, 11 ms
  customers (read_csv): 4 rows out, 2 ms
  paid (filter): true: 5 rows, false: 2 rows, 3 ms
  with_customer (join): joined: 4 rows, left_only: 1 rows, right_only: 1 rows, 14 ms
  by_region (group_by): 2 rows out, 5 ms
Held back, nothing written: save
Finished: 5 of 6 steps ran.
```

A **preview** runs every step except the ones that write files or tables. Nothing outside the
project's run folder changes. The run is saved, so the viewer can show you every port.

You can already read the answer from the counts: 5 paid orders, 4 of them matched a customer, 1 did
not (`left_only`), and 1 customer had no paid order (`right_only`).

## 6. Look at it in the viewer

```bash
dantron-etl view
```

Your browser opens on the list of the project's pipelines. Click **Paid sales by region**. You see
the steps drawn as a flow. Click a step to see its settings, its `why`, and the rows on each of its
ports. Open `with_customer` and its `left_only` port: order 1005 for customer `C9` is there.

Press ctrl-c in the terminal to stop the viewer.

## 7. Change it

Order 1005 was paid, but it is missing from the totals because its customer is unknown. Say you
would rather count it under a blank region than drop it. Change the join to keep every paid order:

```yaml fragment
    config:
      keys: [customer_id]
      how: left
```

Run the preview again:

```bash
dantron-etl run sales
```

`with_customer` now gives 5 rows on `joined`, and `by_region` gives 3 rows: North, South, and a
row with no region holding the 15.00 from order 1005. `left_only` still lists order 1005, so the
unknown customer stays easy to find.

## 8. Write the file for real

When the preview looks right:

```bash
dantron-etl run sales --write
```

It lists every file the run will write and asks you to type the pipeline's name before it goes
ahead. Then `output/sales_by_region.csv` holds:

```text
region,paid_total,orders
North,419.5,3
South,80.5,1
,15.0,1
```

In the viewer, **Real run…** does the same, with the same question.

## When something goes wrong

- **`config has a key that is not a name: True (bool)`.** YAML reads some bare words, such as `on`,
  `yes` and `no`, as true or false. You probably wrote `on:` as a setting. No tool has a setting
  called `on`; `join` uses `keys`. See [the YAML traps](pipeline-file.md#yaml-traps).
- **`unknown config field(s): key`.** The setting name is misspelt or belongs to another tool.
  `dantron-etl help <tool>` lists the tool's settings.
- **`is a dead end: nothing consumes any of its output`.** Every step's rows must go somewhere:
  into another step, a file, or a check. Remove the step, or use its output.
- **`the folder 'output' does not exist`.** A file writer needs its folder to exist before the run.
  Create it, or change the step's `path`.
- **`refuses to overwrite the file already at`.** Add `overwrite: true` to the writer.
- **A step stops on an empty frame.** File and table writers stop rather than write nothing. Add
  `allow_empty: true` to the step if an empty result is fine.

## Next

- [Concepts](concepts.md): ports, containers, rules, runs and previews, in a few pages.
- [Cookbook](cookbook.md): the shapes most pipelines need.
- [Expressions](expressions.md): the formula language and every function.
- [Checking your pipeline](checking-your-pipeline.md): how to know the figures are right.
