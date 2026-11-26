# Cookbook

The shapes most pipelines need, each as a small pipeline you can copy. Every example here loads, and
a test checks that it still does. File names are placeholders: point `path` at your own files.

Most of these are two or three tools in a row. That is normal. Before deciding a tool is missing,
try two existing ones in sequence.

## Add a column from a lookup list

Each order gets its customer's region. `how: left` keeps every order, with a blank region where the
customer is not on the list. Those orders are also on `left_only`, so you can see which they are.

```yaml
name: Orders with their region

steps:
  - id: orders
    tool: read_csv
    config: {path: data/orders.csv}

  - id: customers
    tool: read_csv
    config: {path: data/customers.csv}

  - id: with_region
    tool: join
    why: Every order, with its customer's region where the customer is known.
    config:
      keys: [customer_id]
      how: left
    inputs: {left: orders.rows, right: customers.rows}

  - id: save
    tool: write_csv
    config: {path: output/orders_with_region.csv, overwrite: true}
    inputs: {rows: with_region.joined}
```

When the two files spell the key differently, use `left_keys: [customer_id]` and
`right_keys: [id]` instead of `keys`.

A lookup list with a key repeated on it gives the order once per match, and every amount is
repeated. Check the list's key is unique first: see [a check that stops the run](#a-check-that-stops-the-run).

## Report the rows that matched nothing

The same join gives the unmatched rows on its own ports. Write them to a file someone can work
through.

```yaml
name: Orders for unknown customers

steps:
  - id: orders
    tool: read_csv
    config: {path: data/orders.csv}

  - id: customers
    tool: read_csv
    config: {path: data/customers.csv}

  - id: matched
    tool: join
    config: {keys: [customer_id]}
    inputs: {left: orders.rows, right: customers.rows}

  - id: unknown_customers
    tool: write_csv
    why: Orders whose customer is not on the customer list. Each one needs a customer added.
    config: {path: output/unknown_customers.csv, overwrite: true, allow_empty: true}
    inputs: {rows: matched.left_only}

  - id: customers_without_orders
    tool: write_csv
    config: {path: output/customers_without_orders.csv, overwrite: true, allow_empty: true}
    inputs: {rows: matched.right_only}

  - id: save
    tool: write_csv
    config: {path: output/matched.csv, overwrite: true}
    inputs: {rows: matched.joined}
```

`allow_empty: true` lets a report be empty: no unknown customers is the good outcome.

## Total by group, and keep the detail

`group_by` keeps only the key and the columns it computes. `join_text` keeps a list of what went
into each total, and `count` says how many rows. Sort first, so the list is in a stated order.

```yaml
name: Paid total per customer

steps:
  - id: orders
    tool: read_csv
    config: {path: data/orders.csv}

  - id: in_order
    tool: sort
    why: The references below are listed in order number order.
    config:
      by:
        - {column: customer_id}
        - {column: order_id}
    inputs: {rows: orders.rows}

  - id: per_customer
    tool: group_by
    config:
      by: [customer_id]
      aggregations:
        - {name: total, function: sum, value: "[amount]"}
        - {name: orders, function: count}
        - {name: order_ids, function: join_text, value: "[order_id]", separator: ", "}
    inputs: {rows: in_order.rows}

  - id: save
    tool: write_csv
    config: {path: output/per_customer.csv, overwrite: true}
    inputs: {rows: per_customer.rows}
```

`join_text` needs text: if `order_id` is read as a number, add `column_types` to `read_csv` or a
`cast_columns` step to make it `String`.

To put the group's total on every detail row instead, join the totals back to the detail on the
key.

## A cross tab with zeros in the empty cells

`cross_tab` turns rows into columns: one row per customer, one column per month. A customer with no
sales in a month gets an empty cell, because no row said anything about it. To show `0` there, make
the missing rows first: every customer with every month, then the real totals joined on.

```yaml
name: Sales by customer and month

steps:
  - id: sales
    tool: read_csv
    config: {path: data/sales.csv}

  - id: customers
    tool: remove_duplicates
    why: Each customer once.
    config: {by: [customer_id]}
    inputs: {rows: sales_keys.rows}

  - id: sales_keys
    tool: select_columns
    why: Only the customer column, so that removing duplicates leaves one row per customer.
    config: {columns: [customer_id]}
    inputs: {rows: sales.rows}

  - id: months
    tool: read_csv
    why: Every month the table should show, whether or not anything was sold in it.
    config: {path: data/months.csv}

  - id: customer_key
    tool: formula
    config:
      columns:
        - {name: all, value: "1"}
    inputs: {rows: customers.unique}

  - id: month_key
    tool: formula
    config:
      columns:
        - {name: all, value: "1"}
    inputs: {rows: months.rows}

  - id: every_cell
    tool: join
    why: Every customer paired with every month, by joining on a column that is 1 on both sides.
    config: {keys: [all]}
    inputs: {left: customer_key.rows, right: month_key.rows}

  - id: totals
    tool: group_by
    config:
      by: [customer_id, month]
      aggregations:
        - {name: amount, function: sum, value: "[amount]"}
    inputs: {rows: sales.rows}

  - id: filled
    tool: join
    config:
      keys: [customer_id, month]
      how: left
    inputs: {left: every_cell.joined, right: totals.rows}

  - id: zeros
    tool: formula
    why: A month with no sales is a sale of zero.
    config:
      columns:
        - {name: amount, value: "coalesce([amount], 0.0)", replace: true}
    inputs: {rows: filled.joined}

  - id: grid
    tool: cross_tab
    config:
      by: [customer_id]
      names_from: month
      values_from: amount
      function: sum
    inputs: {rows: zeros.rows}

  - id: save
    tool: write_csv
    config: {path: output/sales_grid.csv, overwrite: true}
    inputs: {rows: grid.rows}
```

## A date as text

To write a date as `2025-01` (a month heading, say), make it text with `cast_columns`, then take the
part you want with `regex_extract`:

```yaml
name: Month headings

steps:
  - id: entries
    tool: read_csv
    config:
      path: data/entries.csv
      column_types:
        - {column: posted_on, type: Date}

  - id: as_text
    tool: cast_columns
    config:
      columns:
        - {column: posted_on, type: String}
    inputs: {rows: entries.rows}

  - id: month
    tool: formula
    config:
      columns:
        - {name: month, value: "regex_extract([posted_on], '^([0-9]{4}-[0-9]{2})', 1)"}
    inputs: {rows: as_text.rows}

  - id: save
    tool: write_csv
    config: {path: output/entries_by_month.csv, overwrite: true}
    inputs: {rows: month.rows}
```

## A number inside a sentence

`format_number` writes a number as text with the separators you choose. `concat` puts it in the
sentence.

```yaml
name: Customer statements

steps:
  - id: balances
    tool: read_csv
    config: {path: data/balances.csv}

  - id: sentence
    tool: formula
    config:
      columns:
        - name: line
          value: "concat([name], ' owes ', format_number([balance], 2, ',', '.'), ' as at month end.')"
    inputs: {rows: balances.rows}

  - id: save
    tool: write_csv
    config: {path: output/statements.csv, overwrite: true}
    inputs: {rows: sentence.rows}
```

A balance of `1234.5` gives `Harbour Books owes 1,234.50 as at month end.`

## Stack every sheet of a workbook

One step reads every sheet whose name matches a pattern, stacked into one table, with a column
saying which sheet each row came from. A new sheet next month is read without editing the pipeline.

```yaml
name: All regions

steps:
  - id: regions
    tool: read_excel
    why: One sheet per region, all with the same headings.
    config:
      path: data/regions.xlsx
      sheets_matching: "^Region "
      sheet_column: sheet

  - id: save
    tool: write_csv
    config: {path: output/all_regions.csv, overwrite: true}
    inputs: {rows: regions.rows}
```

To stack two different files with the same columns, read each and use `union`, which matches
columns by name and keeps every row, duplicates included.

## Remove duplicates, keeping the latest

Sort so the row you want comes first in each group, then keep the first of each.

```yaml
name: Latest address per customer

steps:
  - id: addresses
    tool: read_csv
    config:
      path: data/addresses.csv
      column_types:
        - {column: updated_on, type: Date}

  - id: newest_first
    tool: sort
    config:
      by:
        - {column: customer_id}
        - {column: updated_on, direction: desc}
    inputs: {rows: addresses.rows}

  - id: latest
    tool: remove_duplicates
    why: The newest address for each customer. Older ones leave on the duplicate port.
    config:
      by: [customer_id]
      keep: first
    inputs: {rows: newest_first.rows}

  - id: save
    tool: write_csv
    config: {path: output/latest_addresses.csv, overwrite: true}
    inputs: {rows: latest.unique}
```

Without the `sort`, "first" means whichever row happened to arrive first, which can change from one
run to the next.

## A check that stops the run

`assert_rows` stops the run when any row does not satisfy its predicate, and names the first rows
that fail. When every row passes, they go on unchanged on its `rows` port, so a check can sit in the
middle of a pipeline or at the end of a branch. Put one wherever a wrong row would make a wrong
figure.

```yaml
name: Customer list checks

steps:
  - id: customers
    tool: read_csv
    config: {path: data/customers.csv}

  - id: every_customer_has_a_region
    tool: assert_rows
    config:
      label: every customer has a region
      predicate: "[region] is not null and trim([region]) != ''"
    inputs: {rows: customers.rows}

  - id: by_key
    tool: remove_duplicates
    config: {by: [customer_id]}
    inputs: {rows: customers.rows}

  - id: no_customer_twice
    tool: assert_rows
    why: A customer listed twice would double every order joined to it.
    config:
      label: each customer is listed once
      predicate: "false"
      allow_empty: true
    inputs: {rows: by_key.duplicate}

  - id: save
    tool: write_csv
    config: {path: output/customers.csv, overwrite: true}
    inputs: {rows: by_key.unique}
```

The second check uses a pattern worth knowing: a port that must be empty gets an assertion no row
can pass (`false`) and `allow_empty: true`. Any row that arrives stops the run.

An assertion on a frame with no rows fails unless the step says `allow_empty: true`, because "no
rows" is usually a mistake upstream.

## Match a payment to the orders it pays within a date window

A customer pays for several orders with one payment. Pair every payment with each of the customer's
orders placed in the 60 days before it, then let `match_one_to_many` find the one set of orders
that adds up to the payment. A payment that no set adds up to, or two sets do, is set aside with a
reason for a person to look at.

```yaml
name: Payments and the orders they pay

steps:
  - id: payments
    tool: read_csv
    config:
      path: data/payments.csv
      column_types:
        - {column: paid_on, type: Date}
        - {column: payment_amount, type: Decimal, places: 2}

  - id: orders
    tool: read_csv
    config:
      path: data/orders.csv
      column_types:
        - {column: ordered_on, type: Date}
        - {column: order_amount, type: Decimal, places: 2}

  - id: same_customer
    tool: join
    why: Every payment beside every order of the same customer.
    config: {keys: [customer_id]}
    inputs: {left: payments.rows, right: orders.rows}

  - id: in_window
    tool: filter
    why: An order can only be paid on or after the day it was placed, and within 60 days.
    config:
      predicate: "days_between([ordered_on], [paid_on], 'elapsed') >= 0 and days_between([ordered_on], [paid_on], 'elapsed') <= 60"
    inputs: {rows: same_customer.joined}

  - id: paid_orders
    tool: match_one_to_many
    why: Each payment with the one set of up to five orders that adds up to it.
    config:
      whole_key: payment_id
      whole_amount: payment_amount
      part_key: order_id
      part_amount: order_amount
      max_parts: 5
      reason_column: reason
    inputs: {candidates: in_window.true}

  - id: save
    tool: write_csv
    config: {path: output/paid_orders.csv, overwrite: true}
    inputs: {rows: paid_orders.matched}

  - id: to_review
    tool: write_csv
    why: Payments that could not be matched to exactly one set of orders.
    config: {path: output/payments_to_review.csv, overwrite: true, allow_empty: true}
    inputs: {rows: paid_orders.rejected}
```

Read the amounts as `Decimal`. A `Float64` amount is refused, because totals of decimal-point
numbers do not add up exactly. To allow a cent either way, add `tolerance: 0.01`. A wider window or
a larger `max_parts` finds more sets, and so more payments that add up two ways.
