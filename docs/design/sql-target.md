# The SQL compile target, and why it was removed

## What it was

The expression language once had two outputs. Every expression compiled to a polars expression,
which is how every tool runs it. The same text could also compile to a SQL `WHERE` clause plus the
values to bind into it, through `compile_sql(text, columns, dialect=...)`.

The idea was that a read from a database could cut rows at the database instead of loading the whole
table and filtering it in memory. This is often called a pushdown.

The SQL output supported two dialects, SQLite and Postgres. Each dialect published a subset of the
language it could be given, through `sql_subset(dialect)`. Anything outside the subset was refused
by name, with a message saying to filter after the read instead. The subsets were measured, not
guessed: each function in them was run against the real engine and its rows compared with the
polars result. Where an engine answered differently, the function was left out. For example, `upper`
folds `ß` differently in each engine, and `round` breaks ties a different way.

Column names reached the SQL only as quoted identifiers that passed a plain-name grammar. Every
other value the author wrote was a bound parameter, never part of the SQL text.

## Why it was removed

Nothing used it. No tool called `compile_sql`. `read_table` reads `SELECT * FROM <table>` and has no
field for a condition, and no client pipeline or script reached the SQL output either.

It still cost a lot to keep. It was about 520 lines of the expression module and about 1,000 lines
of tests, including a real-database comparison for every function. Every new expression function
had to be decided for SQL as well: either spelled for both dialects and measured, or refused with a
written reason. It also made the compiler harder to read, because every operation went through a
"target" object that could be either output.

A feature with no caller is also a feature nobody notices breaking. Removing it made the compiler
build polars expressions directly.

## How to bring it back

The last version that has it is commit `0a79238` on `develop`. Look at:

- `src/dantron_etl/expr.py`: `compile_sql`, `sql_subset`, `SQL_DIALECTS`, `SQL_PARAMSTYLES`,
  `SQL_IDENTIFIER`, the class `_SqlTarget`, the per-dialect spellings `_SQLITE_SPELLINGS` and
  `_POSTGRES_SPELLINGS`, and the refusal reasons in `_OUTSIDE_THE_SUBSET`.
- `tests/test_dantron_etl_expr.py`, the section headed "The second compile target". Its strongest
  test filters the same rows in polars and in a real database and requires the same keys back.

Since then, each function is defined once, in the function table in `src/dantron_etl/expr/functions.py`.
That table is where a SQL version would go: one more field on each function entry, holding either
its spelling per dialect or the reason it cannot be pushed down. The compiler's tree walk would need
a second way to build each operator (comparison, `and`/`or`, `in`, `is null`), which is what
`_SqlTarget` did.

Bring it back only when a tool needs it, such as a `read_table` with a condition field. Bring the
agreement tests back with it: a pushdown that answers differently from the in-memory filter changes
figures without raising anything.
